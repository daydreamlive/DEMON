"""GPU-free tests for the SA3 TRT DiT CUDA graph key / invalidation logic."""

from pathlib import Path

import pytest
import torch

from acestep.engine import sa3_trt
from acestep.engine.sa3_trt import _LaunchGraphCache


@pytest.mark.parametrize(
    "value, expected",
    [(None, True), ("", True), ("1", True), ("0", False), ("false", False),
     ("OFF", False), (" no ", False)],
)
def test_env_default_on_with_escape_hatch(monkeypatch, value, expected):
    if value is None:
        monkeypatch.delenv(sa3_trt.TRT_CUDA_GRAPH_ENV, raising=False)
    else:
        monkeypatch.setenv(sa3_trt.TRT_CUDA_GRAPH_ENV, value)
    assert sa3_trt.cuda_graph_enabled() is expected


def test_cache_hit_on_same_key():
    c = _LaunchGraphCache(True)
    g = object()
    c.store(("k",), g)
    assert c.lookup(("k",)) is g
    assert c.lookup(("k",)) is g
    assert c.captures == 1 and c.invalidations == 0


def test_cache_drops_graph_on_key_change():
    c = _LaunchGraphCache(True)
    c.store(("a",), object())
    assert c.lookup(("b",)) is None
    assert c.graph is None and c.key is None and c.invalidations == 1
    # The old key does not resurrect the dropped graph.
    assert c.lookup(("a",)) is None
    assert c.invalidations == 1


def test_cache_disable_drops_graph():
    c = _LaunchGraphCache(True)
    c.store(("a",), object())
    c.disable("capture_failed: boom")
    assert not c.enabled and c.graph is None
    assert c.disabled_reason == "capture_failed: boom"


def test_refit_epoch_counts_per_engine():
    e1, e2 = object(), object()
    assert sa3_trt.engine_refit_epoch(e1) == 0
    assert sa3_trt.note_engine_refit(e1) == 1
    assert sa3_trt.note_engine_refit(e1) == 2
    assert sa3_trt.engine_refit_epoch(e1) == 2
    assert sa3_trt.engine_refit_epoch(e2) == 0


def _fake_dit(L=8):
    """An SA3TRTDit with CPU buffers and stand-in engine/context, built
    without __init__ (no TensorRT, no GPU)."""
    dit = object.__new__(sa3_trt.SA3TRTDit)
    dit.engine = object()
    dit.engine_path = Path("sa3_m_dit_l1_8_8") / "sa3_m_dit_l1_8_8.trt"
    dit._ctx = object()
    dit._L = L
    dit._x = torch.zeros(1, sa3_trt.IO_CHANNELS, L)
    dit._t = torch.zeros(1)
    dit._t5_hidden = torch.zeros(1, sa3_trt.T5_TOKENS, sa3_trt.COND_DIM)
    dit._t5_mask = torch.zeros(1, sa3_trt.T5_TOKENS)
    dit._seconds = torch.ones(1)
    dit._local_add = torch.zeros(1, 257, L)
    dit._velocity = torch.zeros(1, sa3_trt.IO_CHANNELS, L)
    dit._graphs = _LaunchGraphCache(True)
    return dit


def test_key_stable_across_steps_and_input_writes():
    dit = _fake_dit()
    k1 = dit._graph_key()
    # In-place input writes (what every step does) keep shape and address.
    dit._x.copy_(torch.randn_like(dit._x))
    dit._t[0] = 0.5
    assert dit._graph_key() == k1


def test_key_changes_on_refit():
    dit = _fake_dit()
    k1 = dit._graph_key()
    dit._graphs.store(k1, "graph")
    sa3_trt.note_engine_refit(dit.engine)
    k2 = dit._graph_key()
    assert k2 != k1
    assert dit._graphs.lookup(k2) is None  # stale capture dropped


def test_key_changes_on_buffer_address_or_shape():
    dit = _fake_dit()
    k1 = dit._graph_key()
    dit._velocity = torch.zeros_like(dit._velocity)  # new address
    k2 = dit._graph_key()
    assert k2 != k1
    dit._x = torch.zeros(1, sa3_trt.IO_CHANNELS, 16)  # new shape
    assert dit._graph_key() not in (k1, k2)


def test_key_changes_on_engine_or_context():
    dit = _fake_dit()
    k1 = dit._graph_key()
    dit._ctx = object()
    k2 = dit._graph_key()
    assert k2 != k1
    dit.engine = object()
    assert dit._graph_key() not in (k1, k2)


def test_invalidate_graph_forces_recapture():
    dit = _fake_dit()
    k = dit._graph_key()
    dit._graphs.store(k, "graph")
    dit.invalidate_graph()
    assert dit._graphs.lookup(k) is None


def test_refit_mirror_bumps_epoch():
    from acestep.engine.sa3_trt_lora import SA3TRTRefitMirror

    class _Refitter:
        def get_all_weights(self):
            return ["w0"]

        def refit_cuda_engine(self):
            return True

    class _TRT:
        float32 = "f32"
        float16 = "f16"

    lin = torch.nn.Linear(2, 2, bias=False)
    root = torch.nn.Module()
    root.lin = lin
    import json
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        m = Path(d) / "m.json"
        m.write_text(json.dumps({"version": 1, "weights": {
            "lin": {"initializer": "w0"}}}), encoding="utf-8")
        engine = object()
        mirror = SA3TRTRefitMirror(engine, root, m, _refitter=_Refitter(), _trt=_TRT)
    # Mark the module dirty so a sync pushes and commits without a real
    # parametrization chain.
    mirror._dirty.add("lin")
    import acestep.engine.sa3_trt_lora as lora_mod

    pushed = []
    orig = lora_mod.set_typed_weights
    lora_mod.set_typed_weights = lambda *a, **k: pushed.append(a[2])
    try:
        before = sa3_trt.engine_refit_epoch(engine)
        assert mirror.sync(reason="test") == 1
    finally:
        lora_mod.set_typed_weights = orig
    assert pushed == ["w0"]
    assert sa3_trt.engine_refit_epoch(engine) == before + 1
