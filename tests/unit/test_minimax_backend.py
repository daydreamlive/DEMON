"""MiniMaxBackend against the Tier-1 GeneratorBackend contract.

Fakes for the DiT and the decoder, so this runs on CPU with no weights.
What is under test is the contract the runner actually relies on:
delivery geometry at 48 kHz, the knob manifest, the produce modes, the
anchor-adoption that gives this family its source to cover, and the
render-window slicing — including the one that bites hardest, that
``render_window`` must not hand back a view of the decode cache,
because the runner crossfades into that array in place.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from acestep.engine.minimax_adapter import MINIMAX_COND_DIM, MiniMaxAdapter
from acestep.streaming.generator_backend import TickContext
from acestep.streaming.knobs import KnobState
from acestep.streaming.minimax_backend import (
    DELIVERY_SAMPLE_RATE,
    MINIMAX_LATENT_RATE_HZ,
    MiniMaxBackend,
    minimax_knob_specs,
    minimax_latent_frames,
)

DURATION_S = 1.0
T = minimax_latent_frames(DURATION_S)
C = 128


class _FakeDit(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, x, t, cond):
        self.calls += 1
        # A mild contraction: converges instead of exploding over steps,
        # so a drained pipeline yields a finite, checkable latent.
        return -0.5 * x


class _RampCodec:
    """Deterministic decoder: a linear ramp over the whole song.

    A ramp makes the resample-and-slice arithmetic checkable by eye —
    the sample at song position p is p / total, so a window's first
    value tells you exactly where it was cut from.
    """

    def __init__(self):
        self.decodes = 0

    def decode_full(self, latent_bct):
        self.decodes += 1
        n = int(round(DURATION_S * 44100))
        ramp = torch.linspace(0.0, 1.0, n)
        return torch.stack([ramp, ramp], dim=0)


def _cond(fill: float = 1.0) -> dict:
    return {"encoder_hidden_states": torch.full((1, T, MINIMAX_COND_DIM), fill)}


def _backend(**kw) -> MiniMaxBackend:
    dit = kw.pop("dit", None) or _FakeDit()
    codec = kw.pop("codec", None) or _RampCodec()
    steps = kw.pop("steps", 4)
    adapter = MiniMaxAdapter(
        dit,
        schedule_builder=lambda d: torch.linspace(float(d), 0.0, steps + 1),
        device="cpu",
        dtype=torch.float32,
    )
    params = dict(
        adapter=adapter,
        codec=codec,
        cond=_cond(),
        schedule_builder_factory=lambda c, n: (
            lambda d: torch.linspace(float(d), 0.0, n + 1)
        ),
        knob_state=KnobState(minimax_knob_specs()),
        duration_s=DURATION_S,
        steps=steps,
        depth=kw.pop("depth", 2),
        vae_window_s=kw.pop("vae_window_s", 0.1),
    )
    params.update(kw)
    return MiniMaxBackend(**params)


def _ctx(playhead=0.0) -> TickContext:
    return TickContext(playhead_s=playhead, buffer_duration_s=DURATION_S)


def _run(backend, ticks: int) -> int:
    fresh = 0
    for _ in range(ticks):
        knobs = backend.read_knobs()
        if backend.produce(knobs, _ctx(), "generate"):
            fresh += 1
    return fresh


# ---- contract surface -----------------------------------------------------


def test_geometry_delivers_at_48k():
    """The runner hardcodes 48 kHz and never calls geometry(); a family
    that reports its native 44.1 kHz here would still be resampled, but
    a family that DELIVERS 44.1 kHz would play at the wrong speed."""
    geo = _backend().geometry()
    assert geo.sample_rate == DELIVERY_SAMPLE_RATE
    assert geo.channels == 2
    assert geo.chunk_rate_hz == pytest.approx(MINIMAX_LATENT_RATE_HZ)
    assert geo.duration_s == pytest.approx(DURATION_S)


def test_capabilities_start_minimal():
    caps = _backend().capabilities()
    assert caps.refines_audio is True
    # No converted audio encoder ships with this checkpoint, so anything
    # that needs one must stay off until it does.
    assert caps.swap is False
    assert caps.write_audio is False
    assert caps.timbre is False
    assert caps.lora is False


def test_vae_window_attribute_is_present():
    # session.py reads this unguarded when it builds the runner.
    assert _backend(vae_window_s=0.25).vae_window == pytest.approx(0.25)


def test_knob_defaults_round_trip_through_read_knobs():
    knobs = _backend().read_knobs()
    assert knobs["minimax_denoise"] == pytest.approx(1.0)
    assert knobs["minimax_shift"] == pytest.approx(1.0)
    assert knobs["minimax_cond_strength"] == pytest.approx(1.0)
    assert "seed" in knobs and "x0_target" in knobs


# ---- produce --------------------------------------------------------------


def test_produce_eventually_yields_a_fresh_generation():
    b = _backend(steps=4, depth=2)
    assert _run(b, 12) > 0
    assert b.has_renderable_state()


def test_skip_mode_produces_nothing_and_reuse_readopts():
    b = _backend(steps=4, depth=2)
    _run(b, 12)
    assert b.produce(b.read_knobs(), _ctx(), "skip") is False
    assert b.produce(b.read_knobs(), _ctx(), "reuse") is True


def test_prepare_runs_even_when_the_step_is_skipped():
    """Live control changes must keep landing on in-flight work; the
    prepare half runs in every mode."""
    b = _backend()
    b.knob_state.update({"minimax_shift": 2.0})
    b.produce(b.read_knobs(), _ctx(), "skip")
    assert b.adapter.shift_alpha == pytest.approx(2.0)


def test_anchor_is_adopted_from_the_first_generation():
    """The family's whole continuity story: with no audio encoder, the
    song it covers is its own first render."""
    b = _backend(steps=4, depth=2)
    assert b._source_latent is None
    _run(b, 12)
    assert b._source_latent is not None
    assert b._source_latent.shape == (1, T, C)


def test_step_change_signals_a_rebuild():
    b = _backend(steps=4)
    knobs = b.read_knobs()
    knobs["steps_override"] = 9
    assert b.rebuild_imminent(knobs) is True
    b.produce(knobs, _ctx(), "generate")
    assert b._steps == 9
    # Settled: no further rebuild for the same value.
    assert b.rebuild_imminent(knobs) is False


# ---- render ---------------------------------------------------------------


def test_render_window_lands_at_the_requested_position():
    b = _backend(steps=4, depth=2, vae_window_s=0.1)
    _run(b, 12)
    chunk = b.render_window(0.5)
    assert chunk is not None
    assert chunk.start_sample == int(round(0.5 * DELIVERY_SAMPLE_RATE))
    assert chunk.pcm.shape[1] == 2
    assert chunk.pcm.dtype == np.float32
    # A ramp decoded and resampled: halfway through the song is ~0.5.
    assert float(chunk.pcm[0, 0]) == pytest.approx(0.5, abs=2e-2)


def test_render_window_returns_a_copy_not_a_view():
    """The runner crossfades INTO this array in place. If it were a view
    of the decode cache, every later render would be contaminated."""
    b = _backend(steps=4, depth=2)
    _run(b, 12)
    first = b.render_window(0.2)
    first.pcm[:] = -7.0
    second = b.render_window(0.2)
    assert not np.allclose(second.pcm, -7.0)


def test_decode_is_cached_per_latent():
    codec = _RampCodec()
    b = _backend(steps=4, depth=2, codec=codec)
    _run(b, 12)
    b.render_window(0.05)          # first render pays for the decode
    before = codec.decodes
    assert before > 0
    b.render_window(0.1)
    b.render_window(0.2)
    b.render_window(0.3)
    # render_window is called up to twice per tick; a full decode per
    # call would dominate the tick budget.
    assert codec.decodes == before


def test_render_window_clamps_at_the_song_end():
    b = _backend(steps=4, depth=2, vae_window_s=0.1)
    _run(b, 12)
    chunk = b.render_window(DURATION_S - 0.01)
    assert chunk is not None
    assert chunk.start_sample + chunk.pcm.shape[0] <= int(
        round(DURATION_S * DELIVERY_SAMPLE_RATE)
    ) + 1


def test_no_render_before_first_generation():
    b = _backend()
    assert b.render_window(0.0) is None
    assert b.render_full() is None
    assert b.has_renderable_state() is False


# ---- conditioning ---------------------------------------------------------


def test_cond_strength_scales_toward_the_unconditional_branch():
    b = _backend()
    full = b._cond_for_tick(1.0)["encoder_hidden_states"]
    half = b._cond_for_tick(0.5)["encoder_hidden_states"]
    zero = b._cond_for_tick(0.0)["encoder_hidden_states"]
    torch.testing.assert_close(half, full * 0.5)
    # 0.0 is the model's own uncond input, not an extrapolation.
    assert torch.count_nonzero(zero) == 0


def test_prompt_blend_endpoints_return_the_verbatim_bundles():
    a, bb = _cond(1.0), _cond(2.0)
    b = _backend(cond=a, cond_b=bb)
    b.handle_set_prompt_blend(0.0)
    assert b._active_cond is a
    b.handle_set_prompt_blend(1.0)
    assert b._active_cond is bb


def test_prompt_blend_midpoint_preserves_norm():
    """Slerp, not lerp: a linear midpoint collapses the conditioning
    norm and sounds washed out."""
    torch.manual_seed(0)
    a = {"encoder_hidden_states": torch.randn(1, T, MINIMAX_COND_DIM)}
    bb = {"encoder_hidden_states": torch.randn(1, T, MINIMAX_COND_DIM)}
    b = _backend(cond=a, cond_b=bb)
    b.handle_set_prompt_blend(0.5)
    mid = b._active_cond["encoder_hidden_states"]

    n_a = a["encoder_hidden_states"].norm(dim=-1)
    n_mid = mid.norm(dim=-1)
    n_lerp = (
        0.5 * a["encoder_hidden_states"] + 0.5 * bb["encoder_hidden_states"]
    ).norm(dim=-1)
    assert float(n_mid.mean()) > float(n_lerp.mean())
    assert float(n_mid.mean()) == pytest.approx(float(n_a.mean()), rel=0.15)


# ---- feedback -------------------------------------------------------------


def test_feedback_tap_blends_a_past_latent_into_the_anchor():
    b = _backend(steps=4, depth=2)
    _run(b, 16)
    anchor = b._source_latent.clone()
    prep = {"feedback": 0.0, "feedback_depth": 1}
    torch.testing.assert_close(b._tapped_source(prep), anchor)

    prep = {"feedback": 1.0, "feedback_depth": 1}
    tapped = b._tapped_source(prep)
    assert tapped is not None
    # Fully tapped is the past latent, not the anchor.
    torch.testing.assert_close(tapped, b._latent_history[-1])


def test_feedback_history_is_bounded():
    b = _backend(steps=2, depth=2)
    _run(b, 40)
    assert len(b._latent_history) <= b._max_feedback_depth
