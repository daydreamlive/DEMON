"""MiniMaxBackend against the Tier-1 GeneratorBackend contract.

Fakes for the AR stage, the DiT and the decoder, so this runs on CPU
with no weights. What is under test is what the runner actually relies
on from an APPEND-ONLY family, which is a different contract from the
diffusion backends':

* the declarations (capabilities, geometry, knob manifest) a client
  gates its panels on;
* ``render_window`` ignoring its position hint, wrapping at the rolling
  window, and never handing back a view the runner would crossfade in
  place;
* the 44.1 -> 48 kHz conversion staying phase-exact across block
  boundaries, which is the one piece of arithmetic here that fails
  silently and sounds like aliasing;
* the generation worker's own loop advancing the frontier.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from acestep.engine.minimax_ar import ARControls, ReplayARStream
from acestep.engine.minimax_render import (
    CARRY_LATENT_FRAMES,
    CHUNK_AR_FRAMES,
    DEFAULT_GUIDANCE,
    DEFAULT_SHIFT,
    DEFAULT_STEPS,
    HOP_AR_FRAMES,
    LATENT_PER_AR_DEN,
    LATENT_PER_AR_NUM,
    MINIMAX_SAMPLE_RATE,
    MINIMAX_UPSAMPLE,
    MiniMaxChunkRenderer,
)
from acestep.streaming.generator_backend import (
    TickContext,
    UnsupportedOperation,
)
from acestep.streaming.knobs import KnobState
from acestep.streaming.minimax_backend import (
    DELIVERY_SAMPLE_RATE,
    XFADE,
    MiniMaxBackend,
    _DeliveryResampler,
    minimax_knob_specs,
)

FUSED = 32
COND = 8
CH = 4
WINDOW_S = 4.0


class _FakeCondEncoder:
    def __call__(self, frame_hiddens):
        frames = int(frame_hiddens.shape[1])
        out = frames * LATENT_PER_AR_NUM // LATENT_PER_AR_DEN
        x = frame_hiddens.transpose(1, 2)[:, :COND]
        return torch.nn.functional.interpolate(
            x, size=out, mode="nearest",
        ).transpose(1, 2)


class _ZeroDit:
    in_channels = CH

    def __call__(self, x, t, cond):
        return torch.zeros_like(x)


class _ToneCodec:
    """A decoder whose output depends only on absolute position, so a
    block-wise resample can be checked against a whole-buffer one."""

    def decode_full(self, latent_bct):
        frames = int(latent_bct.shape[-1])
        base = float(latent_bct[0, 0, 0])
        n = frames * MINIMAX_UPSAMPLE
        idx = torch.arange(n, dtype=torch.float32) + base * MINIMAX_UPSAMPLE
        wave = torch.sin(idx * 0.017) * 0.5
        return torch.stack([wave, wave * 0.5], dim=0)


def _frames(n, base=0):
    idx = torch.arange(base, base + n, dtype=torch.float32)
    return idx.view(1, n, 1).repeat(1, 1, FUSED)


def _backend(*, ar_frames=800, window_s=WINDOW_S, start_worker=False):
    renderer = MiniMaxChunkRenderer(
        _ZeroDit(), _FakeCondEncoder(), device="cpu", dtype=torch.float32,
        chunk_ar_frames=CHUNK_AR_FRAMES,
        carry_latent_frames=CARRY_LATENT_FRAMES,
        latent_channels=CH,
    )
    return MiniMaxBackend(
        ar_stream=ReplayARStream(_frames(ar_frames)),
        renderer=renderer,
        codec=_ToneCodec(),
        knob_state=KnobState(minimax_knob_specs()),
        window_s=window_s,
        steps=2,
        start_worker=start_worker,
    )


# ---------------------------------------------------------------------------
# Declarations
# ---------------------------------------------------------------------------


def test_capabilities_are_all_off():
    caps = _backend().capabilities()
    on = [k for k, v in vars(caps).items() if v]
    assert on == [], f"append-only family must claim nothing: {on}"


def test_geometry_declares_delivery_rate_and_the_AR_frame_rate():
    geo = _backend().geometry()
    assert geo.sample_rate == DELIVERY_SAMPLE_RATE
    assert geo.channels == 2
    assert geo.duration_s == pytest.approx(WINDOW_S)
    # The AR ACOUSTIC frame rate, 25 Hz -- not the 86.133 Hz DiT latent
    # rate. Conflating the two is what produced this integration's first
    # round of throughput claims, so pin it.
    assert geo.chunk_rate_hz == pytest.approx(25.0)
    assert geo.chunk_rate_hz != pytest.approx(
        MINIMAX_SAMPLE_RATE / MINIMAX_UPSAMPLE
    )


def test_knob_manifest_splits_ar_from_renderer_and_reuses_shared_specs():
    from acestep.streaming.knobs import knob_specs as registry_knob_specs

    specs = {s.name: s for s in minimax_knob_specs()}
    ar = {
        "minimax_temperature", "minimax_top_k", "minimax_ar_guidance",
        "minimax_endless",
        "minimax_reprompt_history_s",
    }
    render = {
        "minimax_guidance", "minimax_shift", "minimax_cond_strength",
        "minimax_hop", "minimax_lead", "minimax_steps",
    }
    assert ar | render | {"seed"} == set(specs)
    assert all(specs[n].group == "minimax" for n in ar | render)

    # Shared knobs come out of the registry rather than being
    # re-declared here, so a semantic fork is impossible rather than
    # merely detected by the homonym guard.
    shared = {s.name: s for s in registry_knob_specs(False)}
    assert specs["seed"] == shared["seed"]
    # steps_override is deliberately NOT reused: it means ACE's turbo
    # step count, defaults to 8 and caps at 16, and inheriting it reset
    # every minimax session to a measurably broken render.
    assert "steps_override" not in specs
    assert specs["minimax_steps"].default == DEFAULT_STEPS
    assert specs["minimax_steps"].max_val > shared["steps_override"].max_val

    # The ring-only knobs are gone: there is no cover, no source anchor
    # and no batch axis to feed back into.
    for gone in ("minimax_denoise", "x0_target", "feedback", "feedback_depth"):
        assert gone not in specs


def test_defaults_match_the_measured_operating_point():
    specs = {s.name: s for s in minimax_knob_specs()}
    assert specs["minimax_shift"].default == DEFAULT_SHIFT
    assert specs["minimax_guidance"].default == DEFAULT_GUIDANCE
    assert specs["minimax_hop"].default == HOP_AR_FRAMES
    assert DEFAULT_STEPS >= 12, "8 unwarped steps is a broken render here"


def test_prompt_blend_is_refused_rather_than_ignored():
    with pytest.raises(UnsupportedOperation) as exc:
        _backend().handle_set_prompt_blend(0.5)
    assert exc.value.capability == "prompt_blend"


def test_set_prompt_is_queued_for_the_worker():
    b = _backend()
    b.handle_set_prompt("darkwave, 120 bpm")
    assert b._reprompt_request == ("darkwave, 120 bpm", None)


# ---------------------------------------------------------------------------
# 44.1 kHz -> 48 kHz, append-only
# ---------------------------------------------------------------------------


def _sig(n):
    idx = torch.arange(n, dtype=torch.float32)
    return torch.stack([
        torch.sin(idx * 0.013) + 0.3 * torch.sin(idx * 0.31),
        torch.cos(idx * 0.021),
    ], dim=0)


@pytest.mark.parametrize("block", [512, 1024, 4096, 8192])
def test_blockwise_resample_matches_a_whole_buffer_resample(block):
    """The failure this guards is silent: 44100/48000 reduces to
    147/160 and a latent frame is 512 native samples, so a block that
    starts on a frame boundary resamples onto a DIFFERENT sample phase
    than the whole buffer does. On broadband material that is a ~17%
    relative error, not a rounding detail."""
    import torchaudio

    total = 200_000
    src = _sig(total)
    reference = torchaudio.functional.resample(
        src, MINIMAX_SAMPLE_RATE, DELIVERY_SAMPLE_RATE,
    ).transpose(0, 1).numpy()

    r = _DeliveryResampler()
    out = []
    for lo in range(0, total, block):
        r.push(lo, src[:, lo:lo + block].numpy())
        while True:
            got = r.pop()
            if got is None:
                break
            out.append(got)
    got = np.concatenate(out)

    assert got.shape[0] > 0
    # Everything emitted must match the reference sample for sample.
    ref = reference[:got.shape[0]]
    assert np.max(np.abs(got - ref)) < 2e-5, "phase or filter-edge error"
    # And it must not lag the input by more than the filter pad plus one
    # ratio unit, or the conversion is quietly buffering seconds.
    lag_native = total - r.emitted_native
    assert lag_native <= _DeliveryResampler.PAD + 147


def test_resampler_refuses_a_gap():
    r = _DeliveryResampler()
    r.push(0, _sig(1000).numpy())
    with pytest.raises(ValueError, match="frontier is at"):
        r.push(2000, _sig(1000).numpy())


def test_resampler_emits_exact_ratio_lengths():
    r = _DeliveryResampler()
    r.push(0, _sig(147 * 100 + _DeliveryResampler.PAD + 13).numpy())
    block = r.pop()
    assert block is not None
    assert block.shape[0] % 160 == 0
    assert r.emitted_native % 147 == 0


# ---------------------------------------------------------------------------
# Append-only emission
# ---------------------------------------------------------------------------


def _feed(backend, samples):
    """Hand the backend delivery-rate audio as if the worker made it."""
    idx = np.arange(samples, dtype=np.float32).reshape(-1, 1)
    backend._out.append(np.repeat(idx, 2, axis=1))
    backend._out_samples += samples


def test_render_window_ignores_the_position_hint():
    b = _backend()
    _feed(b, 4800)
    b.produce(b.read_knobs(), TickContext(0.0, 0.0), "generate")
    first = b.render_window(t_start_s=3.7)
    assert first is not None
    assert first.start_sample == 0, "append-only: audio goes at the frontier"


def test_emission_is_contiguous_and_wraps_at_the_window():
    b = _backend(window_s=1.0)      # 48000 delivery samples
    total = 0
    seen = []
    for _ in range(8):
        _feed(b, 20000)
        b.produce(b.read_knobs(), TickContext(0.0, 0.0), "generate")
        while True:
            chunk = b.render_window(0.0)
            if chunk is None:
                break
            seen.append((chunk.start_sample, int(chunk.pcm.shape[0])))
            total += 1
    assert total >= 3
    for start, length in seen:
        assert 0 <= start < b.window_samples
        assert start + length <= b.window_samples, "a chunk must not span the seam"


def test_each_emission_reissues_the_previous_tail_for_the_crossfade():
    b = _backend()
    _feed(b, 30000)
    b.produce(b.read_knobs(), TickContext(0.0, 0.0), "generate")
    first = b.render_window(0.0)
    _feed(b, 30000)
    b.produce(b.read_knobs(), TickContext(0.0, 0.0), "generate")
    second = b.render_window(0.0)

    assert first is not None and second is not None
    assert second.start_sample == first.start_sample + first.pcm.shape[0] - XFADE
    # The runner crossfades the head of `second` against what is already
    # in the ring; those samples must be identical, or every emission
    # start smears.
    np.testing.assert_allclose(
        second.pcm[:XFADE], first.pcm[-XFADE:], rtol=0, atol=0,
    )


def test_emitted_pcm_is_owned_not_a_view():
    """The runner crossfades into the array it is handed, in place."""
    b = _backend()
    _feed(b, 5000)
    b.produce(b.read_knobs(), TickContext(0.0, 0.0), "generate")
    chunk = b.render_window(0.0)
    before = chunk.pcm.copy()
    chunk.pcm[:100] = -12345.0
    _feed(b, 5000)
    b.produce(b.read_knobs(), TickContext(0.0, 0.0), "generate")
    nxt = b.render_window(0.0)
    assert not np.array_equal(nxt.pcm[:XFADE], np.full((XFADE, 2), -12345.0))
    assert np.array_equal(b._tail, before[-XFADE:])


def test_render_window_is_none_when_nothing_is_pending():
    b = _backend()
    assert b.render_window(0.0) is None
    assert b.has_renderable_state() is False


# ---------------------------------------------------------------------------
# The generation worker's own loop
# ---------------------------------------------------------------------------


def test_worker_steps_advance_the_frontier():
    b = _backend(ar_frames=600)
    controls = b._render_controls
    # Drive the worker body by hand: no thread, so the assertions are
    # about the loop's arithmetic rather than about timing.
    for _ in range(200):
        if not b._advance_ar():
            break
    assert b.ar_frames >= CHUNK_AR_FRAMES
    assert b._render_chunk(controls) is True
    assert b.chunks == 1
    assert b.frontier_s() > 0.0

    frontier = b.frontier_s()
    for _ in range(200):
        if not b._advance_ar():
            break
    assert b._render_chunk(controls) is True
    assert b.frontier_s() > frontier, "the second chunk must extend the song"


def test_produce_publishes_controls_to_the_worker():
    b = _backend()
    knobs = dict(b.read_knobs())
    knobs.update({
        "minimax_shift": 1.25,
        "minimax_guidance": 2.5,
        "minimax_cond_strength": 0.4,
        "minimax_hop": 25,
        "minimax_lead": 2.0,
        "minimax_steps": 24,
        "seed": 99,
    })
    b.produce(knobs, TickContext(0.0, 0.0), "generate")
    got, hop, lead = b._snapshot()
    assert (got.shift, got.guidance, got.cond_strength) == (1.25, 2.5, 0.4)
    assert (got.steps, got.seed) == (24, 99)
    assert (hop, lead) == (25, 2.0)


def test_ar_controls_come_off_the_knob_bank():
    b = _backend()
    b.knob_state.update({
        "minimax_temperature": 1.4,
        "minimax_top_k": 12,
        "minimax_ar_guidance": 2.0,
    })
    got = b._ar_controls()
    assert got == ARControls(temperature=1.4, top_k=12, guidance=2.0)


def test_telemetry_reports_the_rates_a_listener_can_hear():
    class _State:
        params: dict = {}

    b = _backend()
    b.state = _State()
    # Cumulative, the way the worker accumulates it: 300 frames in
    # 16.08 s is 53.6 ms each. The per-sample attributes are the last
    # batch only and must NOT be what the echo reports.
    b.ar_frames, b.ar_wall_s = 300, 16.08
    b.ar_ms_per_frame = 999.0
    b.chunks, b.render_wall_s = 4, 2.0
    b.chunk_render_ms = 999.0
    b.produce(b.read_knobs(), TickContext(0.0, 0.0), "generate")
    b.on_fresh_generation(b.read_knobs())
    p = b.state.params
    assert p["num_gens"] == 1
    # 40 ms of audio per frame against 53.6 ms of wall clock.
    assert p["ar_ms_per_frame"] == pytest.approx(53.6, abs=0.05)
    assert p["ar_realtime"] == pytest.approx(0.746, abs=1e-3)
    assert p["chunk_render_ms"] == pytest.approx(500.0, abs=0.1)
    for key in ("frontier_lead_s", "ar_frames", "chunks"):
        assert key in p
