"""AnalysisHost: queue order, drop-oldest, cancel, one-shot priority,
VRAM admission, stderr capture, shutdown. Fake engines only."""

from __future__ import annotations

import sys
import threading

import numpy as np
import pytest

from acestep.analysis import host as host_mod
from acestep.analysis.engine import (
    AnalysisRequest,
    AnalysisResult,
    EngineBusy,
    EngineUnavailable,
)
from acestep.analysis.host import AnalysisHost
from acestep.analysis.registry import AnalyzerSpec

WAIT = 5.0


def _req(tag: int) -> AnalysisRequest:
    return AnalysisRequest(
        audio=np.zeros(16, np.float32), sr=16000, t0_s=float(tag), span_s=1.0,
        context_lead_s=0.0, context_tail_s=0.0, source_epoch=0, revision=tag,
    )


class FakeEngine:
    def __init__(self, name, size, gate=None, order=None, stderr_text=None, fail=None):
        self.name = name
        self.size = size
        self.gate = gate
        self.order = order if order is not None else []
        self.stderr_text = stderr_text
        self.fail = fail
        self.unloads = 0
        self.started = threading.Event()

    def analyze(self, request):
        self.order.append(request.revision)
        self.started.set()
        if self.gate is not None:
            assert self.gate.wait(WAIT)
        if self.stderr_text:
            print(self.stderr_text, file=sys.stderr)
        if self.fail is not None:
            raise self.fail
        return AnalysisResult("note", request.t0_s, request.span_s, [], 0, request.revision)

    def unload(self):
        self.unloads += 1


class Harness:
    """A host over one fake ``fake`` analyzer, recording callbacks."""

    def __init__(self, vram=None, resident=None, probe=None, **engine_kw):
        self.order: list[int] = []
        self.made: list[tuple[str, str]] = []
        self.engines: dict[str, FakeEngine] = {}
        self.shutdowns = 0
        self.engine_kw = engine_kw

        def make(size, device):
            self.made.append((size, device))
            eng = FakeEngine("fake", size, order=self.order, **self.engine_kw)
            self.engines[size] = eng
            return eng

        def spec_shutdown():
            self.shutdowns += 1

        self.spec = AnalyzerSpec(
            name="fake", display_name="Fake", item_kind="note", engine_name="fakemodel",
            make_engine=make, probe=probe or (lambda: None), licence="L",
            sizes=("s", "m"), default_size="s", placement="gpu",
            resident_gb=resident or {}, shutdown=spec_shutdown,
        )
        self.host = AnalysisHost(
            spec_lookup=self._lookup, vram_probe=lambda: vram, live_capacity=3,
        )
        self.results: list[int] = []
        self.errors: list[tuple[int, BaseException]] = []
        self.threads: set[str] = set()
        self.cv = threading.Condition()

    def _lookup(self, name):
        if name != "fake":
            raise KeyError(f"unknown analyzer {name!r}")
        return self.spec

    def submit(self, tag, token=None, size=None):
        def ok(res):
            with self.cv:
                self.threads.add(threading.current_thread().name)
                self.results.append(res.revision)
                self.cv.notify_all()

        def err(exc):
            with self.cv:
                self.threads.add(threading.current_thread().name)
                self.errors.append((tag, exc))
                self.cv.notify_all()

        self.host.submit("fake", size, _req(tag), ok, err, token)

    def wait_for(self, n_done):
        with self.cv:
            assert self.cv.wait_for(lambda: len(self.results) + len(self.errors) >= n_done, WAIT)


def _block_first(h: Harness):
    """Submit tag 0 and wait until the engine is inside it, holding the gate."""
    gate = threading.Event()
    h.engine_kw["gate"] = gate
    h.submit(0)
    deadline = threading.Event()
    for _ in range(500):
        if h.engines.get("s") and h.engines["s"].started.is_set():
            break
        deadline.wait(0.01)
    assert h.engines["s"].started.is_set()
    return gate


def test_live_fifo_and_callbacks_on_worker_thread():
    h = Harness()
    for tag in range(3):
        h.submit(tag)
    h.wait_for(3)
    assert h.results == [0, 1, 2]
    assert h.threads == {"analysis-host"}
    assert h.made == [("s", "cuda")]  # one lazy load, reused
    h.host.shutdown()


def test_drop_oldest_when_full():
    h = Harness()
    gate = _block_first(h)
    for tag in (1, 2, 3, 4, 5):  # capacity 3: 1 and 2 are dropped
        h.submit(tag)
    gate.set()
    h.wait_for(6)
    assert h.results == [0, 3, 4, 5]
    dropped = [(t, str(e)) for t, e in h.errors]
    assert dropped == [(1, "dropped"), (2, "dropped")]
    assert all(isinstance(e, EngineBusy) for _, e in h.errors)
    assert h.threads == {"analysis-host"}
    h.host.shutdown()


def test_cancel_withdraws_queued_only():
    h = Harness()
    gate = _block_first(h)
    tok_a, tok_b = object(), object()
    h.submit(1, tok_a)
    h.submit(2, tok_b)
    h.submit(3, tok_a)
    assert h.host.cancel(tok_a) == 2
    assert h.host.cancel(tok_a) == 0
    gate.set()
    h.wait_for(2)
    assert h.results == [0, 2]
    assert h.errors == []
    h.host.shutdown()


def test_one_shot_jumps_the_queue():
    h = Harness()
    gate = _block_first(h)
    h.submit(1)
    h.submit(2)
    out = {}

    def one_shot():
        out["res"] = h.host.run_one_shot("fake", None, _req(99))

    t = threading.Thread(target=one_shot)
    t.start()
    # Let the one-shot reach the queue before releasing the in-flight job.
    for _ in range(500):
        if h.host._one_shot:
            break
        threading.Event().wait(0.01)
    gate.set()
    t.join(WAIT)
    h.wait_for(3)
    assert out["res"].revision == 99
    assert h.order == [0, 99, 1, 2]
    h.host.shutdown()


def test_one_shot_raises_engine_error():
    h = Harness(fail=ValueError("bad audio"))
    with pytest.raises(ValueError, match="bad audio"):
        h.host.run_one_shot("fake", "m", _req(1))
    h.host.shutdown()


def test_vram_admission_refuses():
    h = Harness(vram={"available_gb": 0.6}, resident={"s": 0.3, "m": 2.0})
    # 0.3 + 0.5 > 0.6
    with pytest.raises(EngineUnavailable, match="not enough free GPU memory for Fake s"):
        h.host.run_one_shot("fake", "s", _req(1))
    assert h.made == []
    h.host.shutdown()


def test_vram_admission_admits_and_skips_without_telemetry():
    h = Harness(vram={"available_gb": 0.81}, resident={"s": 0.3})
    assert h.host.run_one_shot("fake", "s", _req(1)).revision == 1
    h.host.shutdown()
    h2 = Harness(vram=None, resident={"s": 100.0})
    assert h2.host.run_one_shot("fake", "s", _req(2)).revision == 2
    h2.host.shutdown()


def test_unknown_analyzer_size_and_probe_are_unavailable():
    h = Harness(probe=lambda: "not here")
    with pytest.raises(EngineUnavailable, match="unknown analyzer"):
        h.host.run_one_shot("nope", None, _req(1))
    with pytest.raises(EngineUnavailable, match="not available"):
        h.host.run_one_shot("fake", "xl", _req(1))
    with pytest.raises(EngineUnavailable, match="not here"):
        h.host.engine_info("fake", None)
    with pytest.raises(EngineUnavailable, match="not here"):
        h.host.run_one_shot("fake", None, _req(1))
    h.host.shutdown()


def test_load_failure_is_user_facing():
    h = Harness()

    def broken(size, device):
        raise OSError("gated repo")

    h.spec = AnalyzerSpec(
        name="fake", display_name="Fake", item_kind="note", make_engine=broken,
        probe=lambda: None, sizes=("s",), default_size="s",
    )
    with pytest.raises(EngineUnavailable, match="Fake s failed to load: OSError: gated repo"):
        h.host.run_one_shot("fake", None, _req(1))
    h.host.shutdown()


def test_engine_info_does_not_load():
    h = Harness()
    info = h.host.engine_info("fake", None)
    assert info == {
        "name": "fake", "size": "s", "loaded": False, "licence": "L",
        "engine": "fakemodel", "display_name": "Fake",
    }
    assert h.made == []
    h.host.run_one_shot("fake", None, _req(1))
    assert h.host.engine_info("fake", "s")["loaded"] is True
    assert h.host.engine_info("fake", "m")["loaded"] is False
    h.host.shutdown()


def test_stderr_captured_and_logged(capsys):
    from loguru import logger

    lines = []
    sink = logger.add(lambda m: lines.append(m), level="DEBUG")
    try:
        h = Harness(stderr_text="[muscriptor] generate total: 1.00s")
        h.host.run_one_shot("fake", None, _req(1))
        h.host.shutdown()
    finally:
        logger.remove(sink)
    assert "[muscriptor]" not in capsys.readouterr().err
    assert any("analysis_engine_stderr" in m and "generate total" in m for m in lines)


def test_shutdown_unloads_and_fails_queued():
    h = Harness()
    gate = _block_first(h)
    h.submit(1)
    h.submit(2, size="m")
    threading.Timer(0.2, gate.set).start()
    assert h.host.shutdown() == 1
    h.wait_for(3)
    assert h.results == [0]
    assert sorted(t for t, _ in h.errors) == [1, 2]
    assert all(isinstance(e, EngineUnavailable) for _, e in h.errors)
    assert h.engines["s"].unloads == 1
    assert h.shutdowns == 1
    assert h.host.loaded() == []
    with pytest.raises(EngineUnavailable, match="shut down"):
        h.host.run_one_shot("fake", None, _req(3))
    h.submit(4)
    h.wait_for(4)
    assert isinstance(h.errors[-1][1], EngineUnavailable)
    assert h.host.shutdown() == 0


def test_callback_exception_does_not_kill_worker():
    h = Harness()
    h.host.submit("fake", None, _req(1), lambda r: 1 / 0, lambda e: None, None)
    assert h.host.run_one_shot("fake", None, _req(2)).revision == 2
    h.host.shutdown()


def test_get_host_singleton_and_shutdown_host(monkeypatch):
    monkeypatch.setattr(host_mod, "_HOST", None)
    assert host_mod.shutdown_host() == 0
    a = host_mod.get_host()
    assert host_mod.get_host() is a
    assert host_mod.shutdown_host() == 0
    b = host_mod.get_host()
    assert b is not a
    host_mod.shutdown_host()
