"""Extend in place under a stream layout. Tier 1, no GPU.

* The swap-resize path is on under a layout (``swap_resize`` declared).
* A swap's ``grow`` field reaches the layout's ``encode_source`` as
  ``grow=`` (session -> backend -> context), exactly like ``keep``.
* After the encode the layout's ``carried_latent`` is asked once: a tensor
  at the swap's stacked length becomes the cached latent the runner
  renders until the first new slot lands; without one (no grow) the
  backend has nothing to render, as before.
"""

from __future__ import annotations

import threading
import types

import torch

from acestep.engine.sa3_stream_helpers import SA3Conditioning
from acestep.nodes.types import Audio
from acestep.streaming.sa3_backend import SA3_SAMPLE_RATE, delivered_samples
from acestep.streaming.session import StreamingSession
from tests.unit.test_sa3_backend import _FakeCodec, _schedule_builder_factory
from tests.unit.test_sa3_stream_layout import (
    C, L, N44, N_STREAMS, _LayoutRuntime, _StackDit, _layout_backend,
)
from tests.unit.test_swap_resize_session import SAMPLE_RATE, _StubBackend, _SwapSession

L_NEW = 2 * L
N44_NEW = L_NEW * 4096


class _AnyLenCodec(_FakeCodec):
    """Full-decode codec for one stream at any length: a constant equal to
    the slice's mean, so the rendered audio says which latent was decoded."""

    def decode_full(self, latent_bct, *, decode_seed=None):
        self.decodes += 1
        assert latent_bct.shape[:2] == (1, C), latent_bct.shape
        n = int(latent_bct.shape[-1]) * 4096
        return torch.full((2, n), float(latent_bct.mean()) * 0.1)


class _GrowRuntime(_LayoutRuntime):
    """A layout whose encode_source records its kwargs and whose
    carried_latent hands back ``carry`` once."""

    def __init__(self, carry=None):
        self.carry = carry
        self.enc_kw: list = []
        self.carried_calls = 0
        super().__init__({
            "encode_source": self._enc, "carried_latent": self._carried,
        })

    def _enc(self, ctx, audio, n, **kw):
        self.enc_kw.append(kw)
        return torch.zeros(1, C, N_STREAMS * L)

    def _carried(self):
        self.carried_calls += 1
        out, self.carry = self.carry, None
        return out


def _stacked_cond(frames_per_stream):
    return SA3Conditioning(
        cond_bundle={
            "cross_attn_cond": torch.ones(1, 3, 4),
            "cross_attn_mask": torch.ones(1, 3),
            "cfg_scale": 1.0,
        },
        sched_args={},
        latent_frames=N_STREAMS * frames_per_stream,
        audio_sample_size=frames_per_stream * 4096,
        n_streams=N_STREAMS,
    )


def _grow_backend(carry):
    """The layout backend with a fake resizer landing on L_NEW per stream
    and an encoder that records its kwargs."""
    runtime = _GrowRuntime(carry)
    b, _context, _ = _layout_backend(runtime=runtime, codec=_AnyLenCodec())
    encodes: list = []

    def encoder(waveform, sample_rate, sample_size, **kw):
        encodes.append((int(sample_size), kw))
        return torch.zeros(1, C, N_STREAMS * (int(sample_size) // 4096))

    def resizer(new_duration_s, tags_a, tags_b, steps):
        cond = _stacked_cond(L_NEW)
        return (
            cond.audio_sample_size / SA3_SAMPLE_RATE, cond, None,
            _StackDit(), _schedule_builder_factory,
        )

    b._source_encoder = encoder
    b._resizer = resizer
    return b, runtime, encodes


def _carry():
    torch.manual_seed(3)
    return torch.randn(1, C, N_STREAMS * L_NEW)


def test_layout_declares_swap_resize():
    b, _context, _ = _layout_backend()
    assert b._resizer is not None
    assert b.capabilities().swap_resize is True


def test_grow_reaches_the_encoder_and_the_carried_latent_is_rendered():
    carry = _carry()
    b, runtime, encodes = _grow_backend(carry)
    keep = {"drums": [[0.0, 1.0]]}
    grow = {"from_s": 1.0}

    got = b.handle_swap_source(
        torch.zeros(10, 1000), SAMPLE_RATE,
        duration_s=N44_NEW / SA3_SAMPLE_RATE, keep=keep, grow=grow,
    )

    # The new playable length at the grown geometry.
    assert got == delivered_samples(N44_NEW)
    assert int(b._cond.latent_frames) == N_STREAMS * L_NEW
    # keep and grow both reached the encoder, at the NEW window.
    assert encodes == [(N44_NEW, {"keep": keep, "grow": grow})]
    # Asked once; the carried latent is the cached render source.
    assert runtime.carried_calls == 1
    assert torch.equal(b._last_result_latent, carry.movedim(1, 2))
    assert b._last_result_latent.is_contiguous()
    assert b._current_result is None
    assert b.has_renderable_state()
    # Any window of the new geometry renders from it at once.
    chunk = b.render_window(N44_NEW / SA3_SAMPLE_RATE * 0.75)
    assert chunk is not None and chunk.pcm.shape[0] > 0


def test_no_grow_leaves_nothing_to_render_as_before():
    b, runtime, encodes = _grow_backend(None)

    got = b.handle_swap_source(
        torch.zeros(10, 1000), SAMPLE_RATE,
        duration_s=N44_NEW / SA3_SAMPLE_RATE,
    )

    assert got == delivered_samples(N44_NEW)
    assert encodes == [(N44_NEW, {})]
    assert runtime.carried_calls == 1
    assert b._last_result_latent is None
    assert not b.has_renderable_state()


def test_a_carried_latent_at_another_length_is_dropped():
    wrong = torch.randn(1, C, N_STREAMS * L)          # the OLD length
    b, _runtime, _encodes = _grow_backend(wrong)

    b.handle_swap_source(
        torch.zeros(10, 1000), SAMPLE_RATE,
        duration_s=N44_NEW / SA3_SAMPLE_RATE, grow={"from_s": 1.0},
    )

    assert b._last_result_latent is None


def test_session_swap_with_grow_sets_the_new_length_and_the_carried_latent():
    """Session -> real layout backend: a swap with duration_s + grow adopts
    the grown playback length and leaves the carried latent cached."""
    carry = _carry()
    b, runtime, encodes = _grow_backend(carry)
    sess = _SwapSession(b, max_seconds=N44 / SA3_SAMPLE_RATE)
    wf = torch.zeros(10, int(N44_NEW / SA3_SAMPLE_RATE * SAMPLE_RATE))

    sess._apply_swap_backend_owned(
        b.handle_swap_source, wf, None, N44_NEW / SA3_SAMPLE_RATE,
        {"drums": [[0.0, 1.0]]}, {"from_s": 1.0},
    )

    assert sess.bus.events and type(sess.bus.events[-1]).__name__ == "SwapReady"
    assert sess.state.playback_samples == delivered_samples(N44_NEW)
    assert encodes[-1][1] == {"keep": {"drums": [[0.0, 1.0]]}, "grow": {"from_s": 1.0}}
    assert torch.equal(b._last_result_latent, carry.movedim(1, 2))


def test_session_swap_without_grow_calls_the_hook_as_before():
    plain = _StubBackend(swap_resize=False)
    sess = _SwapSession(plain)
    sess.swap(1.0, channels=10)
    assert "grow" not in plain.calls[-1]


def test_swap_source_stages_the_grow_field():
    state = types.SimpleNamespace(
        _lock=threading.Lock(), swap_pending={}, prompt_text="p", last_activity_ts=0.0)
    eng = types.SimpleNamespace(render_anchor_s=1.0, set_render_anchor_queue=lambda q: None)
    fake = types.SimpleNamespace(state=state, audio_eng=eng)
    swap = StreamingSession.swap_source.__wrapped__
    audio = Audio(waveform=torch.zeros(10, 8), sample_rate=SAMPLE_RATE)
    swap(fake, audio, grow={"from_s": 16.0})
    assert state.swap_pending["grow"] == {"from_s": 16.0}
    swap(fake, audio)
    assert state.swap_pending["grow"] is None
    swap(fake, audio, grow="not a mapping")
    assert state.swap_pending["grow"] is None


def test_context_encode_passes_grow_to_the_layout():
    from acestep.engine.sa3_context import SA3Context
    from acestep.plugins.model_extensions import StreamLayout

    seen = []
    layout = StreamLayout(n_streams=N_STREAMS,
                          encode_source=lambda ctx, a, n, **kw: seen.append(kw))
    ctx = types.SimpleNamespace(stream_layout=lambda: layout)
    SA3Context.encode_source(ctx, (SAMPLE_RATE, torch.zeros(2, 8)), 4)
    SA3Context.encode_source(ctx, (SAMPLE_RATE, torch.zeros(2, 8)), 4, grow={"from_s": 2.0})
    SA3Context.encode_source(
        ctx, (SAMPLE_RATE, torch.zeros(2, 8)), 4,
        keep={"bass": [0, 1]}, grow={"from_s": 2.0},
    )
    assert seen == [{}, {"grow": {"from_s": 2.0}},
                    {"keep": {"bass": [0, 1]}, "grow": {"from_s": 2.0}}]


def test_layout_carried_latent_defaults_to_none():
    from acestep.plugins.model_extensions import StreamLayout

    assert StreamLayout(n_streams=2).carried_latent is None
