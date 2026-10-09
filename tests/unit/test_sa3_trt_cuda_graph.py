"""GPU-free tests for the SA3 TRT DiT and SAME decoder CUDA graph key /
invalidation logic."""

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


# ---------------------------------------------------------------------------
# Bounded (multi-key) cache: the SAME decoders keep one graph per shape.
# ---------------------------------------------------------------------------


def test_bounded_cache_keeps_keys_and_evicts_lru():
    c = _LaunchGraphCache(True, capacity=2)
    ga, gb, gc = object(), object(), object()
    c.store(("a",), ga)
    c.store(("b",), gb)
    # A miss on a multi-key cache drops nothing.
    assert c.lookup(("x",)) is None
    assert c.lookup(("a",)) is ga and c.lookup(("b",)) is gb
    c.lookup(("a",))  # a is now most recently used
    c.store(("c",), gc)
    assert len(c) == 2 and c.evictions == 1
    assert c.lookup(("b",)) is None  # LRU evicted
    assert c.lookup(("a",)) is ga and c.lookup(("c",)) is gc


def test_bounded_cache_discard_and_disable():
    c = _LaunchGraphCache(True, capacity=4)
    c.store(("a",), object())
    c.store(("b",), object())
    c.discard(("a",))
    c.discard(("zzz",))  # absent: no-op
    assert c.lookup(("a",)) is None and len(c) == 1 and c.invalidations == 1
    c.disable("capture_failed: boom")
    assert not c.enabled and len(c) == 0


class _FakeCtx:
    """Stand-in TensorRT execution context: output = 4 samples per frame."""

    def __init__(self):
        self.shape = None
        self.addresses = {}
        self.executes = 0

    def set_input_shape(self, name, shape):
        self.shape = tuple(shape)
        return True

    def get_tensor_shape(self, name):
        return (1, 2, self.shape[-1] * 4)

    def set_tensor_address(self, name, ptr):
        self.addresses[name] = ptr

    def execute_async_v3(self, stream):
        self.executes += 1
        return True


class _FakeStream:
    cuda_stream = 0

    def wait_stream(self, other):
        pass

    def synchronize(self):
        pass


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
    import contextlib

    _FakeGraph.instances = []
    monkeypatch.setattr(torch.cuda, "CUDAGraph", _FakeGraph)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda *a, **k: _FakeStream())
    monkeypatch.setattr(torch.cuda, "stream", lambda s: contextlib.nullcontext())
    return _FakeGraph


def _fake_decoder(enabled=True, capacity=8):
    """A SAME decoder with CPU buffers and stand-in engine/context, built
    without __init__ (no TensorRT, no GPU)."""
    dec = object.__new__(sa3_trt.SameLWindowTRTDecoder)
    dec.engine = object()
    dec.engine_path = Path("same_l_decode_window_x") / "e.trt"
    dec._ctx = _FakeCtx()
    dec._stream = _FakeStream()
    dec._in_dtype = torch.float32
    dec._out_name = "audio"
    dec._out_dtype = torch.float32
    dec._out_buf = None
    dec._device = torch.device("cpu")
    dec._slots = sa3_trt.OrderedDict()
    dec._graphs = _LaunchGraphCache(enabled, capacity)
    return dec


def test_decoder_captures_once_per_shape_and_replays(fake_cuda):
    dec = _fake_decoder()
    for _ in range(3):
        out = dec.decode(torch.randn(1, 256, 10))
        assert out.shape == (2, 40)
    # One warm-up execute + one captured enqueue, then replays only.
    assert dec._ctx.executes == 2
    assert len(fake_cuda.instances) == 1 and fake_cuda.instances[0].replays == 3
    # Bound addresses are the shape's persistent buffers, not the caller's.
    in_buf, out_buf = dec._slots[(1, 256, 10)]
    assert dec._ctx.addresses == {"latent": in_buf.data_ptr(), "audio": out_buf.data_ptr()}


def test_decoder_copies_input_in_place(fake_cuda):
    dec = _fake_decoder()
    dec.decode(torch.randn(1, 256, 6))
    ptr = dec._slots[(1, 256, 6)][0].data_ptr()
    lat2 = torch.randn(1, 256, 6)
    dec.decode(lat2)
    in_buf = dec._slots[(1, 256, 6)][0]
    assert in_buf.data_ptr() == ptr
    assert torch.equal(in_buf, lat2)


def test_decoder_shape_change_gets_its_own_graph(fake_cuda):
    dec = _fake_decoder()
    for T in (10, 12, 10, 12, 10):
        assert dec.decode(torch.randn(1, 256, T)).shape == (2, 4 * T)
    assert dec._graphs.captures == 2 and len(dec._graphs) == 2
    assert dec._ctx.executes == 4  # warm-up + capture enqueue per shape
    assert [g.replays for g in fake_cuda.instances] == [3, 2]
    # Each shape's key binds its own buffers.
    k10 = dec._slot_key(dec._slots[(1, 256, 10)])
    k12 = dec._slot_key(dec._slots[(1, 256, 12)])
    assert k10 != k12


def test_decoder_cache_is_bounded(fake_cuda):
    dec = _fake_decoder(capacity=3)
    for T in (4, 6, 8, 10):
        dec.decode(torch.randn(1, 256, T))
    assert len(dec._graphs) == 3 and len(dec._slots) == 3
    assert (1, 256, 4) not in dec._slots
    # The evicted shape re-captures when it comes back.
    dec.decode(torch.randn(1, 256, 4))
    assert dec._graphs.captures == 5
    assert (1, 256, 6) not in dec._slots
    assert len(dec._graphs) == 3


def test_decoder_disabled_env_never_captures(monkeypatch, fake_cuda):
    monkeypatch.setenv(sa3_trt.TRT_CUDA_GRAPH_ENV, "0")
    dec = _fake_decoder(enabled=sa3_trt.cuda_graph_enabled())
    lat = torch.randn(1, 256, 10)
    for _ in range(3):
        dec.decode(lat)
    assert fake_cuda.instances == [] and dec._ctx.executes == 3
    assert not dec._slots
    # Plain path binds the caller's latent directly (the pre-#378 path).
    assert dec._ctx.addresses["latent"] == lat.data_ptr()


def test_decoder_capture_failure_falls_back_to_plain(monkeypatch, fake_cuda):
    dec = _fake_decoder()

    def boom(self, capture_error_mode=None):
        raise RuntimeError("capture not supported")

    monkeypatch.setattr(_FakeGraph, "capture_begin", boom)
    out = dec.decode(torch.randn(1, 256, 10))  # never fails the decode
    assert out.shape == (2, 40)
    assert not dec._graphs.enabled
    assert dec._graphs.disabled_reason.startswith("capture_failed")
    before = dec._ctx.executes
    dec.decode(torch.randn(1, 256, 10))
    assert dec._ctx.executes == before + 1  # plain execute from now on
