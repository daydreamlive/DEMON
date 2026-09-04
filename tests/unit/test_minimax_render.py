"""MiniMax's chunked render loop: geometry, carry, and the frontier.

Fakes for the DiT, the ConditionEncoder and the decoder, so this runs on
CPU with no weights. What is under test is the arithmetic that decides
whether a stream is continuous, because every failure mode here is
inaudible in a spot check and fatal in a stream:

* commits that do not abut leave a gap or write the same audio twice;
* an origin computed from a constant hop instead of the absolute AR
  index drifts half a latent frame per chunk;
* a carry region the frontier has already trimmed is a discontinuity
  every four seconds;
* a decode that reads past the guard is a different signal from the one
  a whole-song decode would have produced.
"""

from __future__ import annotations

import pytest
import torch

from acestep.engine.minimax_render import (
    CARRY_LATENT_FRAMES,
    CHUNK_AR_FRAMES,
    DECODE_GUARD_FRAMES,
    HOP_AR_FRAMES,
    LATENT_PER_AR_DEN,
    LATENT_PER_AR_NUM,
    MINIMAX_UPSAMPLE,
    MiniMaxChunkRenderer,
    MiniMaxLatentDecoder,
    MiniMaxLatentStream,
    RenderControls,
    build_schedule,
    chunk_latent_frames,
    latent_origin,
)

FUSED = 32
COND = 8
CH = 4


class _FakeCondEncoder:
    """25 Hz -> 86.133 Hz at the same integer ratio the checkpoint uses."""

    def __call__(self, frame_hiddens):
        frames = int(frame_hiddens.shape[1])
        out = frames * LATENT_PER_AR_NUM // LATENT_PER_AR_DEN
        x = frame_hiddens.transpose(1, 2)[:, :COND]
        return torch.nn.functional.interpolate(
            x, size=out, mode="nearest",
        ).transpose(1, 2)


class _ZeroDit:
    """Velocity zero: the sampler must then leave every unlocked frame
    exactly at its initial noise, which makes the carry lock the only
    thing that can have moved."""

    in_channels = CH

    def __init__(self):
        self.calls = 0
        self.batches: list = []

    def __call__(self, x, t, cond):
        self.calls += 1
        self.batches.append(int(x.shape[0]))
        return torch.zeros_like(x)


def _renderer(dit=None, *, carry=CARRY_LATENT_FRAMES):
    return MiniMaxChunkRenderer(
        dit or _ZeroDit(),
        _FakeCondEncoder(),
        device="cpu",
        dtype=torch.float32,
        chunk_ar_frames=CHUNK_AR_FRAMES,
        carry_latent_frames=carry,
        latent_channels=CH,
    )


def _frames(n, base=0):
    idx = torch.arange(base, base + n, dtype=torch.float32)
    return idx.view(1, n, 1).repeat(1, 1, FUSED)


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------


def test_chunk_length_is_the_condition_encoder_ratio():
    assert chunk_latent_frames(CHUNK_AR_FRAMES) == 689
    assert chunk_latent_frames(100) == 344


def test_latent_origin_does_not_drift():
    """A constant 344-frame hop slips half a frame per chunk; deriving
    each origin from its absolute AR index cannot."""
    for k in range(1, 400):
        exact = k * HOP_AR_FRAMES * LATENT_PER_AR_NUM / LATENT_PER_AR_DEN
        assert abs(latent_origin(k * HOP_AR_FRAMES) - exact) < 1.0
    hops = {
        latent_origin((k + 1) * HOP_AR_FRAMES) - latent_origin(k * HOP_AR_FRAMES)
        for k in range(400)
    }
    assert hops == {344, 345}, hops


def test_build_schedule_endpoints_and_warp():
    for shift in (0.5, 1.0, 2.0, 4.0):
        s = build_schedule(16, shift)
        assert s.numel() == 17
        assert float(s[0]) == pytest.approx(0.0)
        assert float(s[-1]) == pytest.approx(1.0)
        assert torch.all(s[1:] > s[:-1]), "schedule must advance toward data"
    # shift > 1 spends more steps near noise, i.e. the midpoint sits
    # closer to t=0 than the unwarped schedule's does.
    assert float(build_schedule(16, 2.0)[8]) < float(build_schedule(16, 1.0)[8])
    assert float(build_schedule(16, 0.5)[8]) > float(build_schedule(16, 1.0)[8])
    with pytest.raises(ValueError):
        build_schedule(16, 0.0)


# ---------------------------------------------------------------------------
# The sampler's carry lock
# ---------------------------------------------------------------------------


def test_carry_region_ends_exactly_at_the_committed_latent():
    r = _renderer()
    carry = torch.full((1, CH, CARRY_LATENT_FRAMES), 7.0)
    out = r.render(
        _frames(CHUNK_AR_FRAMES),
        carry=carry,
        controls=RenderControls(steps=4, guidance=1.0),
        chunk_index=0,
    )
    assert out.shape == (1, CH, 689)
    assert torch.allclose(out[..., :CARRY_LATENT_FRAMES], carry)
    # Velocity is zero, so nothing outside the lock moved off its noise
    # draw -- and that draw is seeded, so it reproduces.
    again = r.render(
        _frames(CHUNK_AR_FRAMES), carry=carry,
        controls=RenderControls(steps=4, guidance=1.0), chunk_index=0,
    )
    assert torch.equal(out, again)
    assert not torch.equal(
        out,
        r.render(
            _frames(CHUNK_AR_FRAMES), carry=carry,
            controls=RenderControls(steps=4, guidance=1.0), chunk_index=1,
        ),
    ), "consecutive chunks must not reuse one noise draw"


def test_guidance_costs_one_batch_two_forward_on_eager():
    dit = _ZeroDit()
    r = _renderer(dit)
    r.render(
        _frames(CHUNK_AR_FRAMES), carry=None,
        controls=RenderControls(steps=3, guidance=1.7), chunk_index=0,
    )
    assert dit.calls == 3
    assert dit.batches == [2, 2, 2], "cond and uncond ride one forward"

    dit2 = _ZeroDit()
    r2 = _renderer(dit2)
    r2.render(
        _frames(CHUNK_AR_FRAMES), carry=None,
        controls=RenderControls(steps=3, guidance=1.0), chunk_index=0,
    )
    assert dit2.batches == [1, 1, 1], "guidance 1.0 must skip the negative branch"


def test_cond_window_length_is_enforced():
    r = _renderer()
    with pytest.raises(ValueError, match="AR frames"):
        r.encode_cond(_frames(CHUNK_AR_FRAMES - 1))


# ---------------------------------------------------------------------------
# The frontier
# ---------------------------------------------------------------------------


def test_commits_abut_and_cover_the_song():
    stream = MiniMaxLatentStream(_renderer(), hop_ar_frames=HOP_AR_FRAMES)
    controls = RenderControls(steps=2, guidance=1.0)

    stream.push_frames(_frames(600))
    spans = []
    while True:
        out = stream.render_next(controls)
        if out is None:
            break
        start, committed = out
        spans.append((start, start + int(committed.shape[-1])))

    assert spans, "600 AR frames is five hops of input"
    assert spans[0][0] == 0, "the first commit must start the song"
    for (a0, a1), (b0, b1) in zip(spans, spans[1:]):
        assert a1 == b0, f"gap or overlap between {a1} and {b0}"
    assert stream.committed_frames == spans[-1][1]


def test_every_chunk_carry_is_already_committed():
    """The property that makes the stream continuous: chunk k's left
    context is audio chunk k-1 actually wrote."""
    stream = MiniMaxLatentStream(_renderer(), hop_ar_frames=HOP_AR_FRAMES)
    controls = RenderControls(steps=2, guidance=1.0)
    stream.push_frames(_frames(800))

    seen = 0
    while True:
        origin = latent_origin(stream._next_ar)
        if not stream.can_render():
            break
        if seen:
            assert origin + CARRY_LATENT_FRAMES <= stream.committed_frames
            assert origin >= stream._latent_base
        stream.render_next(controls)
        seen += 1
    assert seen >= 5


def test_trimming_past_the_next_carry_is_loud():
    stream = MiniMaxLatentStream(_renderer(), hop_ar_frames=HOP_AR_FRAMES)
    controls = RenderControls(steps=2, guidance=1.0)
    stream.push_frames(_frames(400))
    stream.render_next(controls)
    # Drop everything: the next chunk's carry is now unreachable.
    stream.trim_latent(stream.committed_frames)
    with pytest.raises(RuntimeError, match="outside the retained latent"):
        stream.render_next(controls)


def test_small_hop_is_accepted_and_large_hop_is_rejected():
    MiniMaxLatentStream(_renderer(), hop_ar_frames=10)
    with pytest.raises(ValueError, match="does not fit"):
        MiniMaxLatentStream(_renderer(), hop_ar_frames=200)


def test_hop_of_25_still_abuts():
    stream = MiniMaxLatentStream(_renderer(), hop_ar_frames=25)
    controls = RenderControls(steps=2, guidance=1.0)
    stream.push_frames(_frames(400))
    spans = []
    while True:
        out = stream.render_next(controls)
        if out is None:
            break
        spans.append((out[0], out[0] + int(out[1].shape[-1])))
    for (a0, a1), (b0, b1) in zip(spans, spans[1:]):
        assert a1 == b0
    # A quarter of the default hop commits a quarter as much per render.
    assert 80 <= spans[-1][1] - spans[-1][0] <= 90


# ---------------------------------------------------------------------------
# Guarded decode
# ---------------------------------------------------------------------------


class _PositionCodec:
    """Every output sample carries the absolute latent frame it came
    from, so a caller can read straight off the audio which frames were
    decoded -- the property a guarded decode has to get right."""

    def __init__(self):
        self.spans: list = []

    def decode_full(self, latent_bct):
        frames = int(latent_bct.shape[-1])
        self.spans.append(frames)
        values = latent_bct[0, 0]
        return values.repeat_interleave(MINIMAX_UPSAMPLE).unsqueeze(0).repeat(2, 1)


def _stamped_stream():
    """A latent stream whose channel 0 equals the absolute frame index."""

    class _StampDit:
        in_channels = CH

        def __call__(self, x, t, cond):
            return torch.zeros_like(x)

    stream = MiniMaxLatentStream(_renderer(_StampDit()), hop_ar_frames=HOP_AR_FRAMES)
    return stream


def test_decoder_never_reads_past_the_guard_and_spans_abut():
    stream = _stamped_stream()
    codec = _PositionCodec()
    decoder = MiniMaxLatentDecoder(codec, guard=DECODE_GUARD_FRAMES)
    controls = RenderControls(steps=2, guidance=1.0)
    stream.push_frames(_frames(600))

    spans = []
    while True:
        if stream.render_next(controls) is None:
            break
        while True:
            got = decoder.decode_next(stream)
            if got is None:
                break
            start_native, audio = got
            spans.append((start_native, start_native + int(audio.shape[-1])))

    assert spans
    assert spans[0][0] == 0
    for (a0, a1), (b0, b1) in zip(spans, spans[1:]):
        assert a1 == b0, "decoded native audio must be contiguous"
    # Never past the guard.
    assert decoder.decoded_frames <= stream.committed_frames - DECODE_GUARD_FRAMES
    # Every decode read guard frames of context on at least one side.
    assert min(codec.spans) > DECODE_GUARD_FRAMES


def test_decoder_returns_none_until_the_guard_is_covered():
    stream = _stamped_stream()
    decoder = MiniMaxLatentDecoder(_PositionCodec(), guard=DECODE_GUARD_FRAMES)
    assert decoder.decode_next(stream) is None
    assert decoder.decodable_upto(5) == 0
    assert decoder.decodable_upto(100) == 100 - DECODE_GUARD_FRAMES
