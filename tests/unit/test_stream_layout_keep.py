"""A stream layout's per-time hard keep (``StreamLayout.keep_mask``) and the
swap's ``keep`` field. Tier 1, no GPU.

* ``keep_mask`` returning ``(mask, target)`` puts ``target`` on the kept
  frames of every step's x0 (the finished latent holds it exactly); None,
  or a layout without the hook, is bit-identical to no hook.
* ``swap_source(keep=...)`` reaches the layout's ``encode_source`` as
  ``keep=``; a swap without it calls every hook exactly as before.
"""

from __future__ import annotations

import threading
import types

import pytest
import torch

from acestep.engine.sa3_adapter import SA3Adapter
from acestep.nodes.types import Audio
from acestep.streaming.session import StreamingSession
from tests.unit.test_sa3_backend import _schedule_builder_factory
from tests.unit.test_sa3_stream_layout import (
    C, L, N_STREAMS, _LayoutRuntime, _StackDit, _knobs, _layout_backend,
    _run_until_fresh,
)
from tests.unit.test_swap_resize_session import SAMPLE_RATE, _StubBackend, _SwapSession

T = N_STREAMS * L


class _KeepRuntime(_LayoutRuntime):
    def __init__(self, keep=None):
        self.keep_calls = 0
        self.keep = keep
        super().__init__({"keep_mask": self._keep})

    def _keep(self, request):
        self.keep_calls += 1
        return self.keep


def _final(runtime):
    torch.manual_seed(0)
    b, _context, _ = _layout_backend(runtime=runtime)
    return _run_until_fresh(b, _knobs(b, sa3_denoise=0.7, seed=7))


def _keep(a, b):
    mask = torch.zeros(1, 1, T, dtype=torch.bool)
    mask[..., a:b] = True
    target = torch.zeros(1, C, T)
    target[..., a:b] = torch.arange(C * (b - a), dtype=torch.float32).reshape(1, C, b - a) / 100
    return mask, target


def test_stock_adapter_has_no_keep_hook_and_shapes_are_checked():
    a = SA3Adapter(_StackDit(), schedule_builder=_schedule_builder_factory(2),
                   device="cpu", dtype=torch.float32)
    assert getattr(a, "keep_mask", None) is None
    bad = SA3Adapter(_StackDit(), schedule_builder=_schedule_builder_factory(2),
                     device="cpu", dtype=torch.float32,
                     keep_mask=lambda r: (torch.ones(1, 1, L, dtype=torch.bool), torch.zeros(1, C, L)))
    req = types.SimpleNamespace(latent_frames=T)
    with pytest.raises(ValueError):
        bad.keep_mask(req)
    m, t = _keep(3, 9)
    ok = SA3Adapter(_StackDit(), schedule_builder=_schedule_builder_factory(2),
                    device="cpu", dtype=torch.float32, keep_mask=lambda r: (m, t))
    mask, target = ok.keep_mask(req)
    assert mask.shape == (1, T, 1) and mask.dtype == torch.bool
    assert torch.equal(target, t.movedim(1, 2))


def test_no_keep_is_bit_identical():
    stock = _final(_LayoutRuntime())
    rt = _KeepRuntime(None)
    none = _final(rt)
    assert rt.keep_calls                       # consulted ...
    assert torch.equal(none, stock)            # ... and None = no hook, bit-exact


def test_kept_frames_hold_the_target_and_the_rest_moves():
    stock = _final(_LayoutRuntime())
    a, b = L + 2, L + 7                        # inside stream 1
    mask, target = _keep(a, b)
    out = _final(_KeepRuntime((mask, target)))
    want = target.movedim(1, 2)
    assert torch.equal(out[:, a:b], want[:, a:b])
    assert not torch.equal(out, stock)


def test_backend_hands_keep_to_the_layout_encode_only_when_given():
    seen = []

    def enc(ctx, audio, n, **kw):
        seen.append(kw)
        return torch.zeros(1, C, T)

    b, _context, _ = _layout_backend(runtime=_LayoutRuntime({"encode_source": enc}))
    calls = []
    shape = b._source_latent_btc.movedim(1, 2).shape

    def rec(waveform, sample_rate, sample_size, **kw):
        calls.append(kw)
        return torch.zeros(shape)

    b._source_encoder = rec
    b.handle_swap_source(torch.randn(10, 1000), SAMPLE_RATE)
    b.handle_swap_source(torch.randn(10, 1000), SAMPLE_RATE, keep={"drums": [2.0, 14.0]})
    assert calls == [{}, {"keep": {"drums": [2.0, 14.0]}}]


def test_context_encode_passes_keep_to_the_layout():
    from acestep.engine.sa3_context import SA3Context
    from acestep.plugins.model_extensions import StreamLayout

    seen = []
    layout = StreamLayout(n_streams=N_STREAMS,
                          encode_source=lambda ctx, a, n, **kw: seen.append(kw))
    ctx = types.SimpleNamespace(stream_layout=lambda: layout)
    SA3Context.encode_source(ctx, (SAMPLE_RATE, torch.zeros(2, 8)), 4)
    SA3Context.encode_source(ctx, (SAMPLE_RATE, torch.zeros(2, 8)), 4, keep={"bass": [0, 1]})
    assert seen == [{}, {"keep": {"bass": [0, 1]}}]


class _KeepStub(_StubBackend):
    def handle_swap_source(self, waveform, sample_rate, duration_s=None, keep=None):
        out = super().handle_swap_source(waveform, sample_rate, duration_s)
        self.calls[-1]["keep"] = keep
        return out


def test_session_swap_carries_keep_to_the_hook():
    backend = _KeepStub(swap_resize=False)
    sess = _SwapSession(backend)
    wf = torch.ones(10, SAMPLE_RATE)
    sess._apply_swap_backend_owned(backend.handle_swap_source, wf, None, None, {"drums": [1, 2]})
    assert backend.calls[-1]["keep"] == {"drums": [1, 2]}
    # absent: the hook is called exactly as before (a stub without the kwarg)
    plain = _StubBackend(swap_resize=False)
    sess = _SwapSession(plain)
    sess.swap(1.0, channels=10)
    assert plain.calls[-1]["channels"] == 10


def test_swap_source_stages_the_keep_field():
    state = types.SimpleNamespace(
        _lock=threading.Lock(), swap_pending={}, prompt_text="p", last_activity_ts=0.0)
    eng = types.SimpleNamespace(render_anchor_s=1.0, set_render_anchor_queue=lambda q: None)
    fake = types.SimpleNamespace(state=state, audio_eng=eng)
    swap = StreamingSession.swap_source.__wrapped__
    audio = Audio(waveform=torch.zeros(10, 8), sample_rate=SAMPLE_RATE)
    swap(fake, audio, keep={"drums": [2.0, 14.0]})
    assert state.swap_pending["keep"] == {"drums": [2.0, 14.0]}
    swap(fake, audio)
    assert state.swap_pending["keep"] is None
    swap(fake, audio, keep="not a mapping")
    assert state.swap_pending["keep"] is None
