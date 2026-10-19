"""GPU-free tests for the shared TensorRT launcher (acestep.engine.trt.launch)
and for SA3TRTDit composing it."""

import contextlib
from pathlib import Path

import pytest
import torch

from acestep.engine import sa3_trt
from acestep.engine.trt import launch as trt_launch
from acestep.engine.trt.launch import TrtLaunch, launch_key, note_engine_refit


class _FakeEngine:
    """Identity only (plus a context factory for the SA3TRTDit test)."""

    def __init__(self, ctx=None):
        self._ctx = ctx

    def create_execution_context(self):
        return self._ctx


class _FakeCtx:
    """Stand-in execution context recording every call in order."""

    def __init__(self, events=None):
        self.events = events if events is not None else []
        self.shapes = {}
        self.addresses = {}
        self.calls = []
        self.executes = 0
        self.execute_ok = True

    def set_input_shape(self, name, shape):
        self.shapes[name] = tuple(shape)
        self.calls.append(("shape", name, tuple(shape)))
        return True

    def get_tensor_shape(self, name):
        # SA3 DiT: velocity has x's shape.
        return self.shapes["x"]

    def set_tensor_address(self, name, ptr):
        self.addresses[name] = ptr
        self.calls.append(("address", name, ptr))
        return True

    def execute_async_v3(self, stream):
        self.executes += 1
        self.events.append("execute")
        return self.execute_ok


class _FakeStream:
    cuda_stream = 0

    def __init__(self, events=None, name="stream"):
        self.events = events if events is not None else []
        self.name = name
        self.waited = []

    def wait_stream(self, other):
        self.waited.append(other)
        self.events.append(("wait", other.name))

    def synchronize(self):
        self.events.append("sync")


class _FakeGraph:
    instances = []

    def __init__(self):
        self.replays = 0
        self.captured = False
        _FakeGraph.instances.append(self)

    def capture_begin(self, capture_error_mode=None):
        assert capture_error_mode == "thread_local"
        self.captured = True

    def capture_end(self):
        pass

    def replay(self):
        self.replays += 1


@pytest.fixture
def fake_cuda(monkeypatch):
    events = []
    caller = _FakeStream(events, name="caller")
    _FakeGraph.instances = []
    monkeypatch.setattr(torch.cuda, "CUDAGraph", _FakeGraph)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda *a, **k: caller)
    monkeypatch.setattr(torch.cuda, "stream", lambda s: contextlib.nullcontext())
    monkeypatch.setattr(
        torch.cuda, "Stream", lambda *a, **k: _FakeStream(events, name="dedicated")
    )
    return {"events": events, "caller": caller, "graphs": _FakeGraph}


def _launch(cuda_graph=True, **kw):
    ctx = _FakeCtx()
    lt = TrtLaunch(_FakeEngine(), ctx, label="test", cuda_graph=cuda_graph, **kw)
    x = torch.zeros(1, 4, 8)
    y = torch.zeros(1, 4, 8)
    lt.bind("x", x)
    lt.bind("y", y, is_input=False)
    return lt, ctx, x, y


def test_bind_records_shapes_and_addresses_in_order(fake_cuda):
    lt, ctx, x, y = _launch()
    assert ctx.calls == [
        ("shape", "x", (1, 4, 8)),
        ("address", "x", x.data_ptr()),
        ("address", "y", y.data_ptr()),
    ]
    assert [n for n, _ in lt.bindings] == ["x", "y"]
    assert lt.bindings[0][1] is x and lt.bindings[1][1] is y
    assert lt.key() == launch_key(lt.engine, ctx, lt.bindings)


def test_bind_raises_when_address_rejected(fake_cuda):
    ctx = _FakeCtx()
    ctx.set_tensor_address = lambda name, ptr: False
    lt = TrtLaunch(_FakeEngine(), ctx, cuda_graph=True)
    with pytest.raises(RuntimeError):
        lt.bind("x", torch.zeros(2))


def test_rebind_replaces_and_invalidates(fake_cuda):
    lt, ctx, x, y = _launch()
    lt.launch()
    assert len(lt.graphs) == 1
    x2 = torch.zeros(1, 4, 8)
    lt.bind("x", x2)
    assert [n for n, _ in lt.bindings] == ["x", "y"]
    assert lt.bindings[0][1] is x2
    assert len(lt.graphs) == 0 and lt.graphs.invalidations == 1
    assert ctx.addresses["x"] == x2.data_ptr()


def test_launch_captures_once_and_replays(fake_cuda):
    lt, ctx, x, y = _launch()
    for _ in range(3):
        lt.launch()
    # One warm-up execute + one captured enqueue, then replays only.
    assert ctx.executes == 2
    graphs = fake_cuda["graphs"].instances
    assert len(graphs) == 1 and graphs[0].replays == 3
    assert lt.graphs.captures == 1


def test_buffer_address_change_recaptures(fake_cuda):
    lt, ctx, x, y = _launch()
    lt.launch()
    k1 = lt.key()
    lt._bindings[1] = ("y", torch.zeros_like(y))  # new address, same name
    assert lt.key() != k1
    lt.launch()
    assert lt.graphs.captures == 2 and ctx.executes == 4


def test_capture_failure_disables_and_warmup_stands(monkeypatch, fake_cuda):
    lt, ctx, x, y = _launch()

    def boom(self, capture_error_mode=None):
        raise RuntimeError("capture not supported")

    monkeypatch.setattr(_FakeGraph, "capture_begin", boom)
    lt.launch()  # no exception: the warm-up execute produced the result
    assert not lt.graphs.enabled
    assert lt.graphs.disabled_reason.startswith("capture_failed")
    assert ctx.executes == 1
    lt.launch()
    assert ctx.executes == 2  # plain execute from now on


def test_cuda_graph_false_never_captures(fake_cuda):
    lt, ctx, x, y = _launch(cuda_graph=False)
    for _ in range(3):
        lt.launch()
    assert fake_cuda["graphs"].instances == [] and ctx.executes == 3


def test_cuda_graph_none_reads_env(monkeypatch, fake_cuda):
    monkeypatch.setenv(trt_launch.TRT_CUDA_GRAPH_ENV, "0")
    assert not TrtLaunch(_FakeEngine(), _FakeCtx()).graphs.enabled
    monkeypatch.delenv(trt_launch.TRT_CUDA_GRAPH_ENV)
    assert TrtLaunch(_FakeEngine(), _FakeCtx()).graphs.enabled


def test_failed_execute_raises_with_label(fake_cuda):
    lt, ctx, x, y = _launch(cuda_graph=False)
    ctx.execute_ok = False
    with pytest.raises(RuntimeError, match="test launch failed"):
        lt.launch()


def test_refit_epoch_bump_recaptures(fake_cuda):
    lt, ctx, x, y = _launch()
    lt.launch()
    lt.launch()
    assert lt.graphs.captures == 1
    note_engine_refit(lt.engine)
    lt.launch()
    assert lt.graphs.captures == 2
    assert [g.replays for g in fake_cuda["graphs"].instances] == [2, 1]


def test_wait_stream_on_caller_before_execute(fake_cuda):
    lt, ctx, x, y = _launch(cuda_graph=False)
    ctx.events = fake_cuda["events"]
    lt.launch()
    assert lt.stream.waited == [fake_cuda["caller"]]
    assert fake_cuda["events"] == [("wait", "caller"), "execute", "sync"]


def test_explicit_stream_is_used(fake_cuda):
    s = _FakeStream(name="mine")
    lt = TrtLaunch(_FakeEngine(), _FakeCtx(), cuda_graph=False, stream=s)
    assert lt.stream is s


def test_sa3_reexports_are_the_shared_objects():
    assert sa3_trt.TrtLaunch is TrtLaunch
    assert sa3_trt._LaunchGraphCache is trt_launch.LaunchGraphCache
    assert sa3_trt._launch_key is trt_launch.launch_key
    assert sa3_trt.note_engine_refit is trt_launch.note_engine_refit
    assert sa3_trt._REFIT_EPOCH is trt_launch._REFIT_EPOCH


class _CpuTorch:
    """sa3_trt's ``torch`` with ``device("cuda")`` mapped to the CPU, so
    SA3TRTDit.__init__ runs without a GPU."""

    def __getattr__(self, name):
        return getattr(torch, name)

    @staticmethod
    def device(*a, **k):
        return torch.device("cpu")


def test_sa3_dit_composes_launch_and_captures_once(monkeypatch, fake_cuda):
    ctx = _FakeCtx()
    engine = _FakeEngine(ctx)
    monkeypatch.setattr(sa3_trt, "_deserialize_engine", lambda p: engine)
    monkeypatch.setattr(sa3_trt, "torch", _CpuTorch())
    L = 8
    dit = sa3_trt.SA3TRTDit(
        Path("sa3_m_dit_l1_8_8") / "e.trt", latent_frames=L, seconds_total=10.0
    )
    assert isinstance(dit._launch, TrtLaunch)
    assert dit._stream is dit._launch.stream and dit._graphs is dit._launch.graphs
    assert [n for n, _ in dit._launch.bindings] == [n for n, _ in dit._bindings()]
    assert dit._graph_key() == dit._launch.key()
    # Every buffer bound at its own address; velocity is not an input.
    for name, buf in dit._bindings():
        assert ctx.addresses[name] == buf.data_ptr()
    assert not any(c[0] == "shape" and c[1] == "velocity" for c in ctx.calls)

    bundle = {
        "cross_attn_cond": torch.randn(1, sa3_trt.T5_TOKENS + 1, sa3_trt.COND_DIM),
        "cross_attn_mask": torch.ones(1, sa3_trt.T5_TOKENS + 1),
    }
    for i in range(3):
        out = dit.step_bundle(torch.randn(1, sa3_trt.IO_CHANNELS, L), 0.1 * i, bundle)
        assert out is dit._velocity
    assert ctx.executes == 2
    graphs = fake_cuda["graphs"].instances
    assert len(graphs) == 1 and graphs[0].replays == 3
    assert dit._launch.stream.waited == [fake_cuda["caller"]] * 3

    dit.invalidate_graph()
    dit.step_bundle(torch.randn(1, sa3_trt.IO_CHANNELS, L), 0.5, bundle)
    assert dit._graphs.captures == 2
