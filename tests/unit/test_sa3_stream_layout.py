"""Stacked-stream layouts (``acestep.plugins.StreamLayout``), Tier 1, no GPU.

Two properties:

* **Stock is untouched.** With no extension, or an extension whose
  ``stream_layout()`` returns None, every layout-aware site runs the stock
  single-stream code: same conditioning, same engine lookup, same adapter
  surface, bit-identical rendered audio.
* **A layout flows end to end.** A fake 3-stream layout with a tiny fake
  DiT rides the ring at the stacked length, each stream is decoded on its
  own slice (full and windowed codecs), and the server-side mix applies
  per-stream gains delivered through the extension's namespaced knobs.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest
import torch

from acestep.engine import sa3_context as ctx_mod
from acestep.engine import sa3_stream_helpers as helpers
from acestep.engine import sa3_trt
from acestep.engine.sa3_adapter import SA3Adapter
from acestep.engine.sa3_stream_helpers import SA3Conditioning
from acestep.plugins import (
    ModelExtension,
    ModelExtensionRuntime,
    ModelExtensionSpec,
    StreamLayout,
)
from acestep.plugins.api import LoadedPlugin, PluginManifest
from acestep.plugins.model_extensions import stream_layout_of
from acestep.plugins.selection import select_model_extension
from acestep.streaming.generator_backend import TickContext
from acestep.streaming.knobs import KnobSpec, KnobState
from acestep.streaming.sa3_backend import (
    DELIVERY_SAMPLE_RATE,
    SA3_SAMPLE_RATE,
    SA3Backend,
    sa3_knob_specs,
)

from tests.unit.test_sa3_backend import _FakeCodec, _schedule_builder_factory
from tests.unit.test_sa3_extension_integration import _Context as _StockContext
from tests.unit.test_sa3_song_length import DS, _Sam, stub_vendor  # noqa: F401

C = 256
N_STREAMS = 3
L = 8                       # latent frames per stream
N44 = L * 4096              # one stream's window, model-rate samples
CTX = TickContext(playhead_s=0.0, buffer_duration_s=10.0)
PLUGIN_ID = "layout_probe"
GAINS = tuple(f"plugin_{PLUGIN_ID}_gain_{i}" for i in range(N_STREAMS))


# ---- fakes ------------------------------------------------------------------


class _StackDit(torch.nn.Module):
    """Tiny DiT: a constant velocity per stream block, recording shapes."""

    def __init__(self):
        super().__init__()
        self.shapes: list = []

    def forward(self, x, t, **kwargs):
        self.shapes.append(tuple(x.shape))
        v = torch.zeros_like(x)
        n = x.shape[-1] // L
        for i in range(n):
            v[..., i * L:(i + 1) * L] = 0.25 * (i + 1)
        return v


class _SliceCodec(_FakeCodec):
    """Full-decode codec that records every latent it is handed and
    returns a constant equal to that latent's mean (so streams differ)."""

    def __init__(self):
        super().__init__()
        self.seen: list = []

    def decode_full(self, latent_bct, *, decode_seed=None):
        self.decodes += 1
        self.seen.append(latent_bct.detach().clone())
        assert latent_bct.shape == (1, C, L), latent_bct.shape
        return torch.full((2, N44), float(latent_bct.mean()) * 0.1)


class _SliceWindowCodec:
    """Windowed codec (the medium path): records per-call latents."""

    def __init__(self):
        self.seen: list = []

    def decode_window(self, latent_bct, start, n):
        self.seen.append(latent_bct.detach().clone())
        assert latent_bct.shape == (1, C, L), latent_bct.shape
        return torch.full((2, int(n)), float(latent_bct.mean()) * 0.1)


class _LayoutRuntime(ModelExtensionRuntime):
    def __init__(self, layout_kw=None):
        self.gains = [1.0] * N_STREAMS
        self.noise_calls = 0
        self.mixed: list = []
        kw = {"n_streams": N_STREAMS, "mix": self._mix, "make_noise": self._noise}
        kw.update(layout_kw or {})
        self.layout = StreamLayout(**kw)

    def stream_layout(self):
        return self.layout

    def apply_controls(self, values):
        for i, name in enumerate(GAINS):
            if name in values:
                self.gains[i] = float(values[name])

    def _mix(self, streams):
        self.mixed.append(streams.copy())
        g = np.asarray(self.gains, dtype=np.float32)[:, None, None]
        return (streams * g).sum(axis=0)

    def _noise(self, request):
        self.noise_calls += 1
        gen = torch.Generator().manual_seed(int(request.seed))
        return torch.randn(1, N_STREAMS * L, C, generator=gen)


class _NoneRuntime(ModelExtensionRuntime):
    """An installed extension that does not change the stream geometry."""


class _Ext(ModelExtension):
    def __init__(self, context):
        self.context = context


def _selected():
    spec = ModelExtensionSpec(
        name="probe", family="sa3", create=_Ext,
        knob_specs=tuple(
            KnobSpec(name, default=1.0, min_val=0.0, max_val=1.5) for name in GAINS
        ),
    )
    plugins = {
        PLUGIN_ID: LoadedPlugin(
            manifest=PluginManifest(id=PLUGIN_ID, version="0.1.0", plugin_api=1),
            model_extensions={"probe": spec},
        )
    }
    return select_model_extension(f"{PLUGIN_ID}.probe", family="sa3", plugins=plugins)


def _stacked_cond():
    return SA3Conditioning(
        cond_bundle={
            "cross_attn_cond": torch.ones(1, 3, 4),
            "cross_attn_mask": torch.ones(1, 3),
            "cfg_scale": 1.0,
        },
        sched_args={},
        latent_frames=N_STREAMS * L,
        audio_sample_size=N44,
        n_streams=N_STREAMS,
    )


class _LayoutContext(_StockContext):
    """from_context stand-in whose installed runtime carries a layout."""

    def __init__(self, runtime, codec=None):
        super().__init__(runtime)
        self.dit_made = _StackDit()
        self.codec = codec if codec is not None else _SliceCodec()
        self.encodes = 0

    def stream_layout(self):
        return stream_layout_of(self.extension_runtime)

    def make_dit(self, **_kw):
        return self.dit_made

    def make_codec(self, **_kw):
        return self.codec

    def encode_source(self, audio_input, audio_sample_size):
        self.encodes += 1
        assert int(audio_sample_size) == N44  # ONE stream's window
        return torch.randn(1, C, N_STREAMS * L)


def _layout_backend(runtime=None, codec=None, **kw):
    runtime = runtime if runtime is not None else _LayoutRuntime()
    context = _LayoutContext(runtime, codec=codec)
    ext = _selected()
    b = SA3Backend.from_context(
        context,
        prompt="p",
        duration_s=N44 / SA3_SAMPLE_RATE,
        cond=_stacked_cond(),
        source_latent_bct=torch.randn(1, C, N_STREAMS * L),
        knob_state=KnobState(sa3_knob_specs(extension_specs=ext.knob_specs)),
        steps=2,
        depth=1,
        vae_window_s=0.1,
        model_extension=ext,
        **kw,
    )
    return b, context, runtime


def _knobs(b, **over):
    return {**b.read_knobs(), "steps_override": b._steps, "seed": 7, **over}


def _run_until_fresh(b, knobs, ticks=10):
    fresh = 0
    for _ in range(ticks):
        if b.produce(knobs, CTX, "generate"):
            fresh += 1
    assert fresh >= 1
    return b._current_result


# ---- the hook itself ----------------------------------------------------------


def test_hook_defaults_to_none_and_tolerates_foreign_runtimes():
    assert ModelExtensionRuntime().stream_layout() is None
    assert stream_layout_of(None) is None
    assert stream_layout_of(object()) is None          # not a subclass
    assert stream_layout_of(_NoneRuntime()) is None
    # A one-stream layout with nothing to do IS the stock path.
    trivial = _LayoutRuntime({"n_streams": 1, "mix": None, "make_noise": None})
    assert stream_layout_of(trivial) is None


def test_hook_validates_what_it_returns():
    class _Bad(ModelExtensionRuntime):
        def stream_layout(self):
            return {"n_streams": 3}

    with pytest.raises(TypeError):
        stream_layout_of(_Bad())
    with pytest.raises(ValueError):
        StreamLayout(n_streams=0)
    with pytest.raises(ValueError):
        StreamLayout(n_streams=2, tokens_per_stream=0)


# ---- conditioning prep ---------------------------------------------------------


class _RecordingModel:
    """Wrap the stub model so the inpaint tensors the helper builds are seen."""

    def __init__(self, inner):
        self._inner = inner

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def get_conditioning_inputs(self, tensors):
        self.inpaint_mask = tensors["inpaint_mask"][0]
        self.inpaint_input = tensors["inpaint_masked_input"][0]
        return self._inner.get_conditioning_inputs(tensors)


def test_stock_conditioning_is_unchanged_by_the_new_arguments(stub_vendor):
    sam = _Sam()
    a = helpers.prepare_sa3_conditioning(
        sam, prompt="p", duration=10.0, steps=8, duration_padding_sec=6.0,
    )
    b = helpers.prepare_sa3_conditioning(
        sam, prompt="p", duration=10.0, steps=8, duration_padding_sec=6.0,
        n_streams=1, tokens_per_stream=None,
    )
    assert a.n_streams == b.n_streams == 1
    assert a.latent_frames == b.latent_frames == a.audio_sample_size // DS
    assert a.stream_frames == a.latent_frames
    assert torch.equal(a.cond_bundle["padding_mask"], b.cond_bundle["padding_mask"])
    assert a.sched_args["fallback_seq_len"] == b.sched_args["fallback_seq_len"]
    assert torch.equal(
        a.sched_args["effective_seq_len"], b.sched_args["effective_seq_len"],
    )
    # The stock schedule follows the label, not the window.
    assert a.sched_args["effective_seq_len"].item() == math.ceil(10.0 * 44100 / DS)


def test_layout_conditioning_is_sized_for_the_stacked_streams(stub_vendor):
    sam = _Sam()
    sam.model = _RecordingModel(sam.model)
    cond = helpers.prepare_sa3_conditioning(
        sam, prompt="p", duration=30.0, steps=8, duration_padding_sec=6.0,
        song_seconds_total=180.0, n_streams=N_STREAMS, tokens_per_stream=L,
    )
    total = N_STREAMS * L
    # The pinned stream length fixes the per-stream window ...
    assert cond.audio_sample_size == L * DS
    assert cond.stream_frames == L
    # ... and the ring carries all streams stacked.
    assert cond.latent_frames == total
    assert cond.n_streams == N_STREAMS
    assert tuple(sam.model.inpaint_mask.shape) == (1, 1, total)
    assert tuple(sam.model.inpaint_input.shape) == (1, C, total)
    mask = cond.cond_bundle["padding_mask"]
    assert tuple(mask.shape) == (1, total) and bool(mask.all())
    # A 180 s label must not move the schedule off the stacked length.
    assert cond.sched_args["effective_seq_len"].item() == total
    assert cond.sched_args["fallback_seq_len"] == total


def test_layout_without_a_pinned_length_derives_it_from_duration(stub_vendor):
    sam = _Sam()
    stock = helpers.prepare_sa3_conditioning(
        sam, prompt="p", duration=5.0, steps=8, duration_padding_sec=0.0,
    )
    cond = helpers.prepare_sa3_conditioning(
        sam, prompt="p", duration=5.0, steps=8, duration_padding_sec=0.0,
        n_streams=2,
    )
    assert cond.audio_sample_size == stock.audio_sample_size
    assert cond.latent_frames == 2 * stock.latent_frames


# ---- SA3Context seam (no model load) -------------------------------------------


def _bare_context(runtime=None):
    c = ctx_mod.SA3Context.__new__(ctx_mod.SA3Context)
    c.model_id = "medium"
    c.sam = _Sam()
    c.downsampling_ratio = DS
    c.sample_rate = 44100
    c.song_seconds = None
    c.schedule_from_window = False
    c.outro_pad_s = 0.0
    c.extension_runtime = runtime
    c._helpers = helpers
    return c


def test_context_stock_dit_still_consults_the_engine_lookup(monkeypatch):
    calls = []

    def _find(model_id, latent_frames, *, want_refittable=False):
        calls.append(latent_frames)
        return None

    monkeypatch.setattr(sa3_trt, "find_dit_engine", _find)
    for runtime in (None, _NoneRuntime()):
        c = _bare_context(runtime)
        assert c.stream_layout() is None
        assert c.make_dit(latent_frames=40, seconds_total=30.0, backend="tensorrt") is c.dit
    assert calls == [40, 40]


def test_context_layout_dit_bypasses_the_stock_lookup(monkeypatch):
    def _find(*_a, **_k):
        raise AssertionError("stock engine lookup consulted under a layout")

    monkeypatch.setattr(sa3_trt, "find_dit_engine", _find)
    made = []
    sentinel = object()

    def _make_dit(context, latent_frames, seconds_total, backend):
        made.append((latent_frames, seconds_total, backend))
        return sentinel if backend == "tensorrt" else None

    c = _bare_context(_LayoutRuntime({"make_dit": _make_dit}))
    assert c.make_dit(latent_frames=24, seconds_total=30.0, backend="tensorrt") is sentinel
    # The factory may decline (None): the eager DiT runs.
    assert c.make_dit(latent_frames=24, seconds_total=30.0, backend="eager") is c.dit
    assert made == [(24, 30.0, "tensorrt"), (24, 30.0, "eager")]
    # No factory at all: eager, still never the stock lookup.
    c2 = _bare_context(_LayoutRuntime())
    assert c2.make_dit(latent_frames=24, seconds_total=30.0, backend="tensorrt") is c2.dit


def test_context_prepare_cond_passes_the_layout(stub_vendor):
    c = _bare_context(_LayoutRuntime({"tokens_per_stream": L}))
    cond = c.prepare_cond(prompt="p", duration=30.0, steps=8)
    assert cond.latent_frames == N_STREAMS * L
    assert cond.audio_sample_size == L * DS
    stock = _bare_context(None).prepare_cond(prompt="p", duration=30.0, steps=8)
    assert stock.n_streams == 1


def test_context_clamps_duration_onto_a_pinned_stream_length(monkeypatch):
    monkeypatch.setattr(sa3_trt, "max_dit_engine_latents", lambda _m: 1)
    pinned = _bare_context(_LayoutRuntime({"tokens_per_stream": L}))
    cap = L * DS / 44100
    assert pinned.clamp_duration_for_trt(60.0, backend="tensorrt") == pytest.approx(cap)
    assert pinned.clamp_duration_for_trt(60.0, backend="eager") == pytest.approx(cap)
    assert pinned.clamp_duration_for_trt(0.1, backend="tensorrt") == 0.1
    # Unpinned layout: the stock (single-stream) engine cap never applies.
    free = _bare_context(_LayoutRuntime())
    assert free.clamp_duration_for_trt(60.0, backend="tensorrt") == 60.0


def test_context_encode_source_stacks_streams(monkeypatch):
    one = torch.randn(1, C, L)
    monkeypatch.setattr(
        helpers, "encode_sa3_source", lambda sam, audio, size: one.clone(),
    )
    audio = (44100, torch.zeros(2, 100))
    stock = _bare_context(None)
    assert torch.equal(stock.encode_source(audio, N44), one)
    # Default under a layout: the one encode repeated into every stream.
    c = _bare_context(_LayoutRuntime())
    out = c.encode_source(audio, N44)
    assert tuple(out.shape) == (1, C, N_STREAMS * L)
    for i in range(N_STREAMS):
        assert torch.equal(out[..., i * L:(i + 1) * L], one)
    # A layout encoder wins, and is handed ONE stream's window.
    seen = []

    def _enc(context, audio_input, size):
        seen.append(size)
        return torch.zeros(1, C, N_STREAMS * L)

    c2 = _bare_context(_LayoutRuntime({"encode_source": _enc}))
    assert torch.equal(c2.encode_source(audio, N44), torch.zeros(1, C, N_STREAMS * L))
    assert seen == [N44]


# ---- adapter -------------------------------------------------------------------


def test_stock_adapter_has_no_noise_hook():
    a = SA3Adapter(
        _StackDit(), schedule_builder=_schedule_builder_factory(2),
        device="cpu", dtype=torch.float32,
    )
    assert getattr(a, "make_noise", None) is None


def test_adapter_noise_hook_checks_the_slot_geometry():
    class _Req:
        latent_frames = N_STREAMS * L
        seed = 3

    a = SA3Adapter(
        _StackDit(), schedule_builder=_schedule_builder_factory(2),
        device="cpu", dtype=torch.float64,
        make_noise=lambda r: torch.ones(1, N_STREAMS * L, C),
    )
    out = a.make_noise(_Req())
    assert out.dtype == torch.float64 and tuple(out.shape) == (1, N_STREAMS * L, C)
    bad = SA3Adapter(
        _StackDit(), schedule_builder=_schedule_builder_factory(2),
        device="cpu", dtype=torch.float32,
        make_noise=lambda r: torch.ones(1, L, C),
    )
    with pytest.raises(ValueError):
        bad.make_noise(_Req())


# ---- backend: stock parity -------------------------------------------------------


def _stock_backend(runtime):
    from tests.unit.test_sa3_backend import _FakeCond, N44 as S_N44, T as S_T

    cond = _FakeCond()
    cond = SA3Conditioning(
        cond_bundle=cond.cond_bundle, sched_args={},
        latent_frames=S_T, audio_sample_size=S_N44,
    )
    torch.manual_seed(0)
    source = torch.randn(1, C, S_T)
    return SA3Backend.from_context(
        _StockContext(runtime),
        prompt="p",
        duration_s=S_N44 / SA3_SAMPLE_RATE,
        cond=cond,
        source_latent_bct=source,
        knob_state=KnobState(sa3_knob_specs()),
        steps=2, depth=1, vae_window_s=0.1,
    )


def test_stock_backend_is_bit_identical_with_or_without_a_none_layout():
    renders = []
    for runtime in (None, _NoneRuntime()):
        b = _stock_backend(runtime)
        assert b._layout is None and b._n_streams == 1
        assert getattr(b.adapter, "make_noise", None) is None
        assert b.capabilities().swap_resize is True
        knobs = _knobs(b, sa3_denoise=0.6)
        _run_until_fresh(b, knobs)
        renders.append((b.render_full().pcm, b.render_window(0.05).pcm))
    assert np.array_equal(renders[0][0], renders[1][0])
    assert np.array_equal(renders[0][1], renders[1][1])


# ---- backend: a 3-stream layout end to end ----------------------------------------


def test_layout_rides_the_ring_at_the_stacked_length():
    b, context, runtime = _layout_backend()
    assert b._n_streams == N_STREAMS
    # Geometry is ONE stream's playable window, not n times it.
    assert b.playable_duration_s() == pytest.approx(N44 / SA3_SAMPLE_RATE)
    assert b.geometry().duration_s == pytest.approx(N44 / SA3_SAMPLE_RATE)
    assert b.geometry().channels == 2
    # The layout fixes geometry: swap stays, swap-resize is off.
    caps = b.capabilities()
    assert caps.swap is True and caps.swap_resize is False

    latent = _run_until_fresh(b, _knobs(b, sa3_denoise=0.7))
    assert tuple(latent.shape) == (1, N_STREAMS * L, C)
    assert context.dit_made.shapes and all(
        s[-1] == N_STREAMS * L for s in context.dit_made.shapes
    )
    assert runtime.noise_calls >= 1          # layout noise drew every slot


def test_layout_decodes_each_stream_and_mixes_with_gains():
    b, context, runtime = _layout_backend()
    latent = _run_until_fresh(b, _knobs(b, sa3_denoise=0.7))
    codec = context.codec

    full = b.render_full().pcm
    # One decode per stream, each on its own slice, in order.
    assert codec.decodes == N_STREAMS
    native = latent.movedim(1, 2)
    assert torch.equal(torch.cat(codec.seen, dim=-1), native)
    means = [float(native[..., i * L:(i + 1) * L].mean()) * 0.1 for i in range(N_STREAMS)]
    assert len(set(round(m, 6) for m in means)) == N_STREAMS  # streams differ
    mid = full.shape[0] // 2
    assert full.shape[1] == 2
    assert full[mid, 0] == pytest.approx(float(np.clip(sum(means), -1, 1)), abs=1e-4)
    assert runtime.mixed[-1].shape[0] == N_STREAMS

    # A gain change arrives through the extension's own knobs and lands at
    # the next render with no regeneration and no re-decode.
    gains = {GAINS[0]: 1.0, GAINS[1]: 0.0, GAINS[2]: 0.5}
    # (A gap-fill "skip" tick: the prepare half runs, nothing generates.)
    assert b.produce(_knobs(b, sa3_denoise=0.7, **gains), CTX, "skip") is False
    chunk = b.render_window(0.05)
    assert codec.decodes == N_STREAMS
    assert chunk.pcm.shape == (round(0.1 * DELIVERY_SAMPLE_RATE), 2)
    expect = means[0] * 1.0 + means[1] * 0.0 + means[2] * 0.5
    assert chunk.pcm[100, 0] == pytest.approx(float(np.clip(expect, -1, 1)), abs=1e-4)
    assert chunk.pcm[100, 0] != pytest.approx(full[chunk.start_sample + 100, 0], abs=1e-4)


def test_layout_windowed_codec_decodes_one_window_per_stream():
    codec = _SliceWindowCodec()
    b, _context, runtime = _layout_backend(codec=codec)
    assert b._windowed_codec
    latent = _run_until_fresh(b, _knobs(b, sa3_denoise=0.7))
    chunk = b.render_window(0.05)
    assert len(codec.seen) == N_STREAMS
    assert torch.equal(torch.cat(codec.seen, dim=-1), latent.movedim(1, 2))
    assert chunk.pcm.shape == (round(0.1 * DELIVERY_SAMPLE_RATE), 2)
    means = [float(s.mean()) * 0.1 for s in codec.seen]
    assert chunk.pcm[100, 0] == pytest.approx(float(np.clip(sum(means), -1, 1)), abs=1e-4)
    assert runtime.mixed and runtime.mixed[-1].shape == (N_STREAMS,) + chunk.pcm.shape


def test_layout_default_mix_is_a_clamped_sum():
    runtime = _LayoutRuntime({"mix": None})
    b, context, _ = _layout_backend(runtime=runtime)
    latent = _run_until_fresh(b, _knobs(b, sa3_denoise=0.7))
    native = latent.movedim(1, 2)
    expect = sum(
        float(native[..., i * L:(i + 1) * L].mean()) * 0.1 for i in range(N_STREAMS)
    )
    full = b.render_full().pcm
    assert full[full.shape[0] // 2, 1] == pytest.approx(float(np.clip(expect, -1, 1)), abs=1e-4)
    assert float(np.abs(full).max()) <= 1.0


def test_layout_swap_reencodes_at_the_fixed_geometry():
    b, context, _ = _layout_backend()
    _run_until_fresh(b, _knobs(b, sa3_denoise=0.7))
    assert b.handle_swap_source(torch.zeros(2, 1000), 44100) is None
    assert context.encodes == 1
    assert tuple(b._source_latent_btc.shape) == (1, N_STREAMS * L, C)
    with pytest.raises(RuntimeError):
        b.handle_swap_source(torch.zeros(2, 1000), 44100, duration_s=5.0)


# ---- path-loaded engines ------------------------------------------------------------


def _engine(tmp_path, dirname, sha=None):
    d = tmp_path / dirname
    d.mkdir(parents=True)
    f = d / f"{dirname}.trt"
    f.write_bytes(b"plan")
    if sha is not None:
        (d / f"{dirname}.trt.metadata.json").write_text(
            json.dumps({sa3_trt.ENGINE_WEIGHTS_KEY: sha}), encoding="utf-8",
        )
    return f


def test_open_dit_engine_at_checks_weights_and_placement(tmp_path, monkeypatch):
    engines = tmp_path / "trt_engines"
    engines.mkdir()
    monkeypatch.setattr(sa3_trt, "trt_engines_dir", lambda: engines)
    loaded = []
    monkeypatch.setattr(
        sa3_trt, "_deserialize_engine",
        lambda path, exclusive=False: loaded.append((path, exclusive)) or "engine",
    )
    good = _engine(tmp_path / "elsewhere", "stacked_dit", sha="ab" * 32)
    assert sa3_trt.open_dit_engine_at(good, expected_weights_sha256="AB" * 32) == "engine"
    assert loaded == [(good, False)]

    with pytest.raises(sa3_trt.EngineWeightsMismatch):
        sa3_trt.open_dit_engine_at(good, expected_weights_sha256="cd" * 32)
    no_sidecar = _engine(tmp_path / "elsewhere", "bare_dit")
    with pytest.raises(sa3_trt.EngineWeightsMismatch):
        sa3_trt.open_dit_engine_at(no_sidecar, expected_weights_sha256="ab" * 32)
    with pytest.raises(FileNotFoundError):
        sa3_trt.open_dit_engine_at(tmp_path / "missing.trt", expected_weights_sha256="ab")

    # Under a stock DiT engine name the stock lookup would pick it up.
    stock_named = _engine(engines, "sa3_m_dit_l32_323_646", sha="ab" * 32)
    assert sa3_trt.is_stock_dit_engine_path(stock_named)
    with pytest.raises(ValueError):
        sa3_trt.open_dit_engine_at(stock_named, expected_weights_sha256="ab" * 32)
    assert len(loaded) == 1


# ---- render_stream: a layout's own whole-window render of one stream --------------


class _OwnRender:
    """``render_stream`` stand-in: a fixed whole-window render for one
    stream index, None for the others; records every call."""

    def __init__(self, index, value, n):
        self.index = index
        self.audio = np.full((n, 2), value, dtype=np.float32)
        self.calls: list = []

    def __call__(self, index, latent):
        self.calls.append((index, tuple(latent.shape)))
        return self.audio if index == self.index else None


def _full_len():
    return int(math.ceil(N44 * DELIVERY_SAMPLE_RATE / SA3_SAMPLE_RATE))


def test_render_stream_defaults_to_none():
    assert StreamLayout(n_streams=2).render_stream is None


def test_render_stream_replaces_one_full_decode():
    own = _OwnRender(1, 0.3, _full_len())
    b, context, runtime = _layout_backend(runtime=_LayoutRuntime({"render_stream": own}))
    latent = _run_until_fresh(b, _knobs(b, sa3_denoise=0.7))
    codec = context.codec

    full = b.render_full().pcm
    # Streams 0 and 2 decoded on their own slices; stream 1 never decoded.
    assert codec.decodes == N_STREAMS - 1
    native = latent.movedim(1, 2)
    assert torch.equal(codec.seen[0], native[..., 0:L])
    assert torch.equal(codec.seen[1], native[..., 2 * L:3 * L])
    assert [c[0] for c in own.calls] == list(range(N_STREAMS))
    assert all(shape == (1, C, L) for _, shape in own.calls)
    mixed = runtime.mixed[-1]
    assert np.array_equal(mixed[1], own.audio)
    m0 = float(native[..., 0:L].mean()) * 0.1
    m2 = float(native[..., 2 * L:3 * L].mean()) * 0.1
    mid = full.shape[0] // 2
    assert full[mid, 0] == pytest.approx(float(np.clip(m0 + 0.3 + m2, -1, 1)), abs=1e-4)


def test_render_stream_windowed_path_slices_the_whole_window_render():
    codec = _SliceWindowCodec()
    n = _full_len()
    own = _OwnRender(0, 0.0, n)
    own.audio[:, :] = np.linspace(-0.5, 0.5, n, dtype=np.float32)[:, None]
    b, _context, runtime = _layout_backend(
        runtime=_LayoutRuntime({"render_stream": own}), codec=codec,
    )
    latent = _run_until_fresh(b, _knobs(b, sa3_denoise=0.7))
    chunk = b.render_window(0.05)
    # Only streams 1 and 2 hit the windowed decoder.
    assert len(codec.seen) == N_STREAMS - 1
    native = latent.movedim(1, 2)
    assert torch.equal(torch.cat(codec.seen, dim=-1), native[..., L:])
    s, w = chunk.start_sample, chunk.pcm.shape[0]
    assert np.array_equal(runtime.mixed[-1][0], own.audio[s:s + w])


def test_render_stream_short_audio_is_padded_in_a_window():
    codec = _SliceWindowCodec()
    own = _OwnRender(2, 0.2, 10)          # far shorter than any window
    b, _context, runtime = _layout_backend(
        runtime=_LayoutRuntime({"render_stream": own}), codec=codec,
    )
    _run_until_fresh(b, _knobs(b, sa3_denoise=0.7))
    chunk = b.render_window(0.0)
    stream = runtime.mixed[-1][2]
    assert stream.shape == chunk.pcm.shape
    assert np.all(stream[:10] == np.float32(0.2)) and np.all(stream[10:] == 0.0)


def test_render_stream_wrong_shapes_raise():
    own = _OwnRender(0, 0.1, 17)          # not the full-decode length
    b, _context, _runtime = _layout_backend(runtime=_LayoutRuntime({"render_stream": own}))
    _run_until_fresh(b, _knobs(b, sa3_denoise=0.7))
    with pytest.raises(ValueError, match="differ in shape"):
        b.render_full()

    flat = _LayoutRuntime({"render_stream": lambda i, lat: np.zeros(5, np.float32)})
    b, _context, _runtime = _layout_backend(runtime=flat)
    _run_until_fresh(b, _knobs(b, sa3_denoise=0.7))
    with pytest.raises(ValueError, match=r"expected \[N, channels\]"):
        b.render_full()


def test_render_stream_returning_none_is_bit_identical_to_no_hook():
    out = []
    for kw in ({}, {"render_stream": lambda i, lat: None}):
        torch.manual_seed(0)
        b, _context, _runtime = _layout_backend(runtime=_LayoutRuntime(kw))
        _run_until_fresh(b, _knobs(b, sa3_denoise=0.7))
        out.append((b.render_full().pcm.copy(), b.render_window(0.05).pcm.copy()))
    assert np.array_equal(out[0][0], out[1][0])
    assert np.array_equal(out[0][1], out[1][1])
