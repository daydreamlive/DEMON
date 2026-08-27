"""The windowed-decode plan for MiniMax-Music3.

``plan_decode_window`` is pure arithmetic over three quantities that do
not divide each other: 512 native samples per latent frame, a 147:160
resample ratio, and a fixed decode span. The failure mode is a
sample-phase error -- the window lands half a delivery sample off the
grid a whole-song resample would produce, disagrees with it by ~17%
relative RMS on broadband material, and is completely inaudible in
isolation because each window is individually fine. It only shows up as
a seam, after a crossfade has already smeared it.

So the plan is separated from the decode and tested exhaustively here,
on CPU with no weights. The companion GPU check that the plan actually
reproduces a full decode lives in
``scripts/minimax/minimax_decode_profile.py``.
"""

from __future__ import annotations

import pytest

from acestep.streaming.minimax_backend import (
    DELIVERY_SAMPLE_RATE,
    MINIMAX_LATENT_RATE_HZ,
    MINIMAX_UPSAMPLE,
    MINIMAX_VAE_DECODE_FRAMES,
    MINIMAX_VAE_GUARD_FRAMES,
    minimax_max_vae_window_s,
    plan_decode_window,
)

TOTAL_FRAMES = 689
WINDOW_S = 0.36
LENGTH = int(round(WINDOW_S * DELIVERY_SAMPLE_RATE))
_NUM, _DEN = 160, 147


def _starts():
    """Window starts across the song, including both edges."""
    total = TOTAL_FRAMES * MINIMAX_UPSAMPLE * _NUM // _DEN
    return [0, 1, 147, 1000, 5000, total // 3, total // 2,
            total - LENGTH - 1, total - LENGTH]


def _every_start(step=37):
    """A dense sweep, because the interesting failures are positional.

    The kept span is converted with floor on one end and ceil on the
    other, so whether a window touches 32 or 33 latent frames depends on
    where it starts. A handful of spot checks passes happily while a
    third of real start positions would have blown the guard. The step
    is coprime-ish with 512 and 147 so the sweep does not accidentally
    sample one phase.
    """
    total = TOTAL_FRAMES * MINIMAX_UPSAMPLE * _NUM // _DEN
    return range(0, total - LENGTH, step)


# ---- the phase invariant ---------------------------------------------------


@pytest.mark.parametrize("start", _starts())
def test_block_starts_on_an_exact_delivery_sample(start):
    """The whole point. After the trim the block's first native sample is
    a multiple of 147, so its delivery position is an integer and the
    offset back into it needs no rounding."""
    plan = plan_decode_window(start, LENGTH, TOTAL_FRAMES)
    n0 = plan.frame_start * MINIMAX_UPSAMPLE + plan.trim_native
    assert n0 % _DEN == 0, (
        "block start is not on a 147-sample boundary; the resample will "
        "land on a different phase grid than a whole-song resample"
    )


@pytest.mark.parametrize("start", _starts())
def test_offset_lands_the_keep_exactly_on_the_request(start):
    plan = plan_decode_window(start, LENGTH, TOTAL_FRAMES)
    n0 = plan.frame_start * MINIMAX_UPSAMPLE + plan.trim_native
    assert n0 // _DEN * _NUM + plan.offset == start


@pytest.mark.parametrize("start", _starts())
def test_trim_is_smaller_than_one_latent_frame(start):
    """The trim eats into the leading guard, so it has to be small
    compared with it or the guard is not what it claims to be."""
    plan = plan_decode_window(start, LENGTH, TOTAL_FRAMES)
    assert 0 <= plan.trim_native < _DEN
    assert plan.trim_native < MINIMAX_UPSAMPLE


# ---- the guard -------------------------------------------------------------


@pytest.mark.parametrize("start", _starts())
def test_guard_survives_on_both_sides(start):
    """The requested span must sit at least ``guard`` frames inside the
    decoded span, after the alignment trim has taken its bite."""
    plan = plan_decode_window(start, LENGTH, TOTAL_FRAMES)
    p0 = start * _DEN / _NUM
    p1 = (start + LENGTH) * _DEN / _NUM
    first_native = plan.frame_start * MINIMAX_UPSAMPLE + plan.trim_native
    last_native = (plan.frame_start + plan.frames) * MINIMAX_UPSAMPLE
    lead = (p0 - first_native) / MINIMAX_UPSAMPLE
    trail = (last_native - p1) / MINIMAX_UPSAMPLE
    assert lead >= MINIMAX_VAE_GUARD_FRAMES - 1, f"lead guard {lead:.2f}"
    assert trail >= MINIMAX_VAE_GUARD_FRAMES - 1, f"trail guard {trail:.2f}"


@pytest.mark.parametrize("start", _starts())
def test_offset_is_inside_the_block(start):
    plan = plan_decode_window(start, LENGTH, TOTAL_FRAMES)
    block = (plan.frames * MINIMAX_UPSAMPLE - plan.trim_native) * _NUM // _DEN
    assert 0 <= plan.offset
    assert plan.offset + plan.length <= block


def test_decode_span_is_fixed_regardless_of_position():
    """A constant shape is what lets a TensorRT decoder engine exist and
    what stops a live vae_window change from silently eating the guard."""
    spans = {plan_decode_window(s, LENGTH, TOTAL_FRAMES).frames
             for s in _starts()}
    assert spans == {MINIMAX_VAE_DECODE_FRAMES}


# ---- cyclic guard ----------------------------------------------------------


def test_guard_wraps_off_the_front_of_the_song():
    """At the head the leading guard has to come from the tail. The ring
    buffer loops, so that is not an approximation -- the tail is what
    actually plays into the head."""
    plan = plan_decode_window(0, LENGTH, TOTAL_FRAMES)
    assert plan.frame_start < 0


def test_guard_wraps_off_the_end_of_the_song():
    total = TOTAL_FRAMES * MINIMAX_UPSAMPLE * _NUM // _DEN
    plan = plan_decode_window(total - LENGTH, LENGTH, TOTAL_FRAMES)
    assert plan.frame_start + plan.frames > TOTAL_FRAMES


# ---- degenerate and error cases --------------------------------------------


def test_short_song_decodes_whole_and_anchors_at_zero():
    plan = plan_decode_window(0, LENGTH, MINIMAX_VAE_DECODE_FRAMES - 8)
    assert plan.frame_start == 0
    assert plan.frames == MINIMAX_VAE_DECODE_FRAMES - 8
    assert plan.trim_native == 0
    assert plan.offset == 0


def test_window_wider_than_the_guard_allows_is_refused():
    """Better a loud error at the seam than a quiet one: a window this
    wide would silently run with less guard than the decoder needs."""
    too_wide = int(minimax_max_vae_window_s() * DELIVERY_SAMPLE_RATE) + 4800
    with pytest.raises(ValueError, match="guard"):
        plan_decode_window(1000, too_wide, TOTAL_FRAMES)


def test_the_shipped_window_fits_the_shipped_decode_span():
    """Guards the constants against each other rather than against a
    number typed in a test."""
    assert WINDOW_S <= minimax_max_vae_window_s()
    keep = int(MINIMAX_VAE_DECODE_FRAMES - 2 * MINIMAX_VAE_GUARD_FRAMES)
    assert keep / MINIMAX_LATENT_RATE_HZ >= WINDOW_S


# ---- the dense sweep -------------------------------------------------------


def test_every_start_position_keeps_its_guard():
    """The off-by-one this caught: at 56 decode frames the plan raised on
    roughly a third of start positions, because 0.36 s of audio is 31.008
    latent frames and a floor/ceil straddle makes that 33."""
    worst = None
    for start in _every_start():
        plan = plan_decode_window(start, LENGTH, TOTAL_FRAMES)
        p0 = start * _DEN / _NUM
        p1 = (start + LENGTH) * _DEN / _NUM
        first = plan.frame_start * MINIMAX_UPSAMPLE + plan.trim_native
        last = (plan.frame_start + plan.frames) * MINIMAX_UPSAMPLE
        lead = (p0 - first) / MINIMAX_UPSAMPLE
        trail = (last - p1) / MINIMAX_UPSAMPLE
        m = min(lead, trail)
        if worst is None or m < worst[0]:
            worst = (m, start)
    assert worst[0] >= MINIMAX_VAE_GUARD_FRAMES - 1, (
        f"start {worst[1]} left only {worst[0]:.2f} frames of guard"
    )


def test_every_start_position_lands_exactly():
    for start in _every_start():
        plan = plan_decode_window(start, LENGTH, TOTAL_FRAMES)
        n0 = plan.frame_start * MINIMAX_UPSAMPLE + plan.trim_native
        assert n0 % _DEN == 0
        assert n0 // _DEN * _NUM + plan.offset == start


def test_every_start_position_fits_inside_its_block():
    for start in _every_start():
        plan = plan_decode_window(start, LENGTH, TOTAL_FRAMES)
        block = (plan.frames * MINIMAX_UPSAMPLE - plan.trim_native) * _NUM // _DEN
        assert 0 <= plan.offset
        assert plan.offset + plan.length <= block, f"overrun at {start}"
