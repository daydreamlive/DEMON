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
    MINIMAX_VAE_DECODE_FRAMES,
    plan_decode_window,
    MINIMAX_DEFAULT_GUIDANCE,
    MINIMAX_DEFAULT_SHIFT,
    MINIMAX_DEFAULT_STEPS,
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
    """Deterministic decoder over exactly the frames it is handed.

    Length-aware on purpose: a decoder that ignored its input length
    would hide the whole point of a windowed render, which is that the
    backend asks for a slice and gets a slice.
    """

    def __init__(self):
        self.decodes = 0
        self.frames_seen: list = []

    def decode_full(self, latent_bct):
        self.decodes += 1
        frames = int(latent_bct.shape[-1])
        self.frames_seen.append(frames)
        ramp = torch.linspace(0.0, 1.0, frames * 512)
        return torch.stack([ramp, ramp], dim=0)


class _PositionCodec:
    """Stamps every output sample with the latent frame it came from.

    Audio value at a sample IS the value of latent channel 0 at the
    frame that produced it, so a caller can set the latent to a ramp of
    frame indices and read straight off the returned audio which frames
    the backend actually decoded. That is the property a windowed decode
    has to get right and the one a whole-song decode gets right by
    accident.
    """

    def __init__(self):
        self.decodes = 0

    def decode_full(self, latent_bct):
        self.decodes += 1
        v = latent_bct[0, 0].float()            # [frames]
        aud = v.repeat_interleave(512)
        return torch.stack([aud, aud], dim=0)


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
    assert knobs["minimax_shift"] == pytest.approx(MINIMAX_DEFAULT_SHIFT)
    assert knobs["minimax_cond_strength"] == pytest.approx(1.0)
    assert knobs["minimax_guidance"] == pytest.approx(MINIMAX_DEFAULT_GUIDANCE)
    assert "seed" in knobs and "x0_target" in knobs


def test_sampler_defaults_stay_in_the_measured_regime():
    """The three sampler defaults are a measured operating point, not
    taste, and they move together.

    ``scripts/minimax/minimax_quality_ablation.py`` grids them against a
    reference trajectory. Guidance is the single largest lever -- an
    unguided run plateaus around 0.11 log-mel from the reference and
    stays there through 40 steps, while 8 guided steps beat it -- and
    step count trades against the schedule warp roughly one for one, so
    lowering steps without raising shift silently gives up most of what
    the steps were buying. This test is here so that anyone retuning
    them re-runs the grid rather than nudging a constant.
    """
    assert MINIMAX_DEFAULT_GUIDANCE > 1.0, (
        "guidance 1.0 disables CFG; measured at ~0.19 log-mel against "
        "~0.03 guided, which is not a speed/quality trade worth taking"
    )
    assert MINIMAX_DEFAULT_STEPS >= 12
    # The measured pairing: 30 steps want shift 1.0, 16 want 2.0, 12
    # want 3.0. Anything far off that line is untested territory.
    assert 1.0 <= MINIMAX_DEFAULT_SHIFT <= 3.0
    if MINIMAX_DEFAULT_STEPS <= 16:
        assert MINIMAX_DEFAULT_SHIFT >= 1.5, (
            "few steps need the schedule warped toward the noise end; "
            "re-run the ablation grid before decoupling these"
        )


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
    """Reads the frames the request names, not merely some frames.

    With ``_PositionCodec`` and a latent stamped with frame indices, the
    returned audio says out loud which latent frames were decoded. This
    is the assertion that catches a sample-phase error in the windowed
    plan, which is otherwise inaudible in isolation and only shows up as
    a seam after the runner has crossfaded over it.
    """
    b = _backend(steps=4, depth=2, vae_window_s=0.1, codec=_PositionCodec())
    _run(b, 12)
    stamp = torch.arange(T, dtype=torch.float32)
    b._last_result_latent = stamp.view(1, T, 1).expand(1, T, C).contiguous()

    for t in (0.0, 0.25, 0.5, 0.8):
        chunk = b.render_window(t)
        assert chunk is not None
        assert chunk.start_sample == int(round(t * DELIVERY_SAMPLE_RATE))
        assert chunk.pcm.shape[1] == 2
        assert chunk.pcm.dtype == np.float32
        want = t * MINIMAX_LATENT_RATE_HZ
        assert float(chunk.pcm[0, 0]) == pytest.approx(want, abs=1.0), (
            f"window at {t}s decoded frame {chunk.pcm[0, 0]:.1f}, "
            f"expected ~{want:.1f}"
        )


def test_render_window_decodes_only_a_window():
    """O(window), not O(song). The whole point of the change."""
    codec = _RampCodec()
    b = _backend(steps=4, depth=2, vae_window_s=0.1, codec=codec)
    _run(b, 12)
    codec.frames_seen.clear()
    for t in (0.1, 0.4, 0.7):
        b.render_window(t)
    assert codec.frames_seen, "no decode happened"
    assert set(codec.frames_seen) == {MINIMAX_VAE_DECODE_FRAMES}, (
        f"decoded {codec.frames_seen} frames per render; a windowed "
        f"backend decodes exactly {MINIMAX_VAE_DECODE_FRAMES} every time, "
        f"and the song here is {T}"
    )
    assert MINIMAX_VAE_DECODE_FRAMES < T


def test_render_window_publishes_its_decode_cost():
    """Without this the latency trace reads a flat 0.0 ms and a decode
    problem is invisible to every instrument in the project."""
    b = _backend(steps=4, depth=2, vae_window_s=0.1)
    _run(b, 12)
    b.last_dec_ms = 0.0
    b.render_window(0.3)
    assert b.last_dec_ms > 0.0


def test_render_window_returns_a_copy_not_a_view():
    """The runner crossfades INTO this array in place. If it were a view
    of the decode cache, every later render would be contaminated."""
    b = _backend(steps=4, depth=2)
    _run(b, 12)
    first = b.render_window(0.2)
    first.pcm[:] = -7.0
    second = b.render_window(0.2)
    assert not np.allclose(second.pcm, -7.0)


def test_guard_wraps_at_the_song_head():
    """The leading guard at t=0 has to come from the song's tail.

    The ring loops, so that is not an approximation — the tail is
    literally what plays into the head. Zero-padding there instead would
    put a decoder edge transient at the loop point, once per lap.
    """
    b = _backend(steps=4, depth=2, vae_window_s=0.1, codec=_PositionCodec())
    _run(b, 12)
    stamp = torch.arange(T, dtype=torch.float32)
    b._last_result_latent = stamp.view(1, T, 1).expand(1, T, C).contiguous()
    plan = plan_decode_window(0, int(round(0.1 * DELIVERY_SAMPLE_RATE)), T)
    assert plan.frame_start < 0, "head window did not reach back into the tail"
    # And it still renders, rather than tripping an index error.
    assert b.render_window(0.0) is not None


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


# ---- construction ---------------------------------------------------------


class _FakeContext:
    """Minimal stand-in for MiniMaxContext's construction surface."""

    device = "cpu"
    dtype = torch.float32

    def __init__(self):
        self.prepared: list = []

    def make_dit(self, *, latent_frames, backend="eager"):
        return _FakeDit()

    def make_codec(self, *, backend="eager"):
        return _RampCodec()

    def make_schedule_builder(self, cond, steps):
        return lambda d: torch.linspace(float(d), 0.0, int(steps) + 1)

    def prepare_cond(self, *, prompt, duration_s, lyrics="", capture=None):
        self.prepared.append(prompt)
        return _cond()


def test_from_context_threads_the_context_through():
    """handle_set_prompt has to re-run the AR stage, so it needs the
    context. Losing it here fails only on the first prompt change,
    which is a long way from where the mistake would be."""
    ctx = _FakeContext()
    b = MiniMaxBackend.from_context(
        ctx,
        cond=_cond(),
        knob_state=KnobState(minimax_knob_specs()),
        duration_s=DURATION_S,
        steps=4,
        depth=2,
        vae_window_s=0.1,
    )
    assert b._context is ctx

    b.handle_set_prompt("a different idea")
    assert ctx.prepared == ["a different idea"]


def test_prompt_swap_rejects_a_geometry_change():
    """Duration is fixed for the session lifetime; a capture made at a
    different length would silently break the ring buffer's T-coherence."""
    ctx = _FakeContext()
    b = MiniMaxBackend.from_context(
        ctx,
        cond=_cond(),
        knob_state=KnobState(minimax_knob_specs()),
        duration_s=DURATION_S,
        steps=4,
        depth=2,
        vae_window_s=0.1,
    )
    ctx.prepare_cond = lambda **kw: {
        "encoder_hidden_states": torch.zeros(1, T + 5, MINIMAX_COND_DIM)
    }
    with pytest.raises(ValueError, match="latent geometry"):
        b.handle_set_prompt("wrong length")


# ---- guidance -------------------------------------------------------------
#
# The bug these guard against shipped once and was silent: the backend
# ran with no CFG at all while the reference pipeline runs guidance 1.7.
# Nothing raised, no gate moved -- latent-domain parity is measured on
# single forwards, and the streamed output was merely worse. So the
# tests here assert the *observable* consequences: that a second forward
# happens, that it carries zeros rather than the capture, and that the
# combine reduces to the operator the reference uses.


class _CondSensitiveDit(torch.nn.Module):
    """Records the conditioning of every forward and depends on it.

    ``_FakeDit`` ignores its cond, which makes it blind to exactly the
    failure being tested: a negative pass that re-sends the positive
    bundle produces v_neg == v_pos, so guidance silently becomes a
    no-op that still costs a forward.
    """

    def __init__(self):
        super().__init__()
        self.seen: list = []

    def forward(self, x, t, cond):
        self.seen.append(float(cond.abs().mean()))
        return -0.5 * x + 0.01 * float(cond.abs().mean())


def _guided_backend(**kw):
    dit = _CondSensitiveDit()
    return dit, _backend(dit=dit, **kw)


def test_guidance_runs_a_negative_pass_carrying_zeros():
    dit, b = _guided_backend(steps=4, depth=1)
    b.knob_state.update({"minimax_guidance": 1.7})
    _run(b, 6)

    assert dit.seen, "no forward ran"
    positive = [v for v in dit.seen if v > 0.0]
    negative = [v for v in dit.seen if v == 0.0]
    assert positive, "the capture never reached the DiT"
    assert negative, (
        "no forward saw the all-zeros bundle, so guidance ran against "
        "the positive conditioning and was a no-op"
    )
    # Full CFG: one negative forward per positive forward.
    assert len(negative) == len(positive)


def test_guidance_of_one_skips_the_negative_pass_entirely():
    dit, b = _guided_backend(steps=4, depth=1)
    b.knob_state.update({"minimax_guidance": 1.0})
    _run(b, 6)
    assert dit.seen
    assert all(v > 0.0 for v in dit.seen), (
        "guidance 1.0 must cost one forward per step, not two"
    )


def test_request_asks_for_textbook_cfg_not_stock_apg():
    """The combine operator is part of the fix, not a detail.

    Stock APG clamps the guidance delta with a norm threshold tuned for
    ACE's latent scale; on a 689-frame MiniMax latent that throttles it
    almost to nothing and measures ~4x worse than plain CFG. These three
    values are what reduce APG to the reference's own operator.
    """
    _, b = _guided_backend(steps=4, depth=1)
    _run(b, 6)
    req = b._last_request
    assert req.apg_eta == pytest.approx(1.0)
    assert req.apg_norm_threshold <= 0.0
    assert req.apg_momentum == pytest.approx(0.0)
    assert req.has_cfg, "guidance_curve + neg_aux_cond must enable CFG"


def test_uncond_bundle_is_zeros_and_stable_across_ticks():
    """Identity matters: an accelerated wrapper keys its staging cache
    on the tensor object, so a fresh zeros_like per tick would miss it
    every time."""
    _, b = _guided_backend(steps=4, depth=1)
    first = b._uncond_bundle()["encoder_hidden_states"]
    assert torch.count_nonzero(first) == 0
    assert first.shape == b._active_cond["encoder_hidden_states"].shape
    assert b._uncond_bundle()["encoder_hidden_states"] is first
