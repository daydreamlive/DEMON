"""Unit tests for the headless PRIMARY client (MCP headless sessions).

Covers the transport-free pieces: the simulated playhead, the
lead/staleness lag tracker, and the binary slice codec round-trip
against the server-side :class:`SliceCodec`. No GPU, no WebSocket, no
torch — time is injected via a fake monotonic clock.
"""

import numpy as np
import pytest

from demos.realtime_motion_graph_web.audio_codec import SliceCodec
from demos.realtime_motion_graph_web.headless_client import (
    LagTracker,
    PlayheadSim,
    apply_slice,
    decode_slice_frame,
    fold_lead_samples,
)
from demos.realtime_motion_graph_web.protocol import (
    SAMPLE_RATE,
    SLICE_FLAG_DELTA,
)


class FakeClock:
    def __init__(self, t: float = 1000.0):
        self.t = t

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


# ---------------------------------------------------------------------------
# PlayheadSim
# ---------------------------------------------------------------------------


def test_playhead_advances_at_wall_rate():
    clock = FakeClock()
    p = PlayheadSim(10 * SAMPLE_RATE, now_fn=clock)
    assert p.sample() == 0
    clock.advance(1.0)
    assert p.sample() == SAMPLE_RATE
    assert p.seconds() == pytest.approx(1.0)


def test_playhead_wraps_modulo_buffer():
    clock = FakeClock()
    p = PlayheadSim(4 * SAMPLE_RATE, now_fn=clock)
    clock.advance(9.0)
    assert p.seconds() == pytest.approx(1.0)


def test_playhead_reset_on_swap():
    clock = FakeClock()
    p = PlayheadSim(100 * SAMPLE_RATE, now_fn=clock)
    clock.advance(10.0)
    p.reset(50 * SAMPLE_RATE)
    assert p.sample() == 0
    assert p.duration_samples == 50 * SAMPLE_RATE


# ---------------------------------------------------------------------------
# Lead fold
# ---------------------------------------------------------------------------


def test_fold_lead_ahead_and_behind():
    n = 10 * SAMPLE_RATE
    # Slice 0.5 s ahead of the playhead.
    assert fold_lead_samples(SAMPLE_RATE, SAMPLE_RATE // 2, n) == (
        SAMPLE_RATE // 2
    )
    # Slice 0.5 s behind.
    assert fold_lead_samples(SAMPLE_RATE // 2, SAMPLE_RATE, n) == (
        -(SAMPLE_RATE // 2)
    )
    # Ahead across the wrap: playhead near the end, slice near the start.
    assert fold_lead_samples(100, n - 100, n) == 200


# ---------------------------------------------------------------------------
# LagTracker
# ---------------------------------------------------------------------------


def test_tracker_lead_and_staleness_healthy():
    clock = FakeClock()
    n = 60 * SAMPLE_RATE
    tr = LagTracker(n, now_fn=clock)
    # Slices land 0.5 s ahead of the playhead, every 0.1 s.
    playhead = 0
    for _ in range(50):
        clock.advance(0.1)
        playhead += int(0.1 * SAMPLE_RATE)
        tr.on_slice(
            start_sample=playhead + SAMPLE_RATE // 2,
            num_samples=int(0.36 * SAMPLE_RATE),
            playhead_sample=playhead,
        )
        tr.on_tick(playhead)
    rep = tr.report(window_s=None)
    assert rep["slices"]["count"] == 50
    assert rep["slices"]["behind_playhead"] == 0
    assert rep["slices"]["lead_s"]["p50"] == pytest.approx(0.5, abs=0.01)
    assert rep["ticks"]["stale_ticks"] == 0
    # Audio under the playhead was written ~0.5 s before it arrived,
    # so staleness stays well under a second.
    assert rep["ticks"]["staleness_s"]["max"] < 1.0


def test_tracker_detects_generation_lag():
    clock = FakeClock()
    n = 60 * SAMPLE_RATE
    tr = LagTracker(n, now_fn=clock)
    playhead = 0
    # Phase 1: healthy for 5 s.
    for _ in range(50):
        clock.advance(0.1)
        playhead += int(0.1 * SAMPLE_RATE)
        tr.on_slice(
            start_sample=playhead + SAMPLE_RATE // 2,
            num_samples=int(0.36 * SAMPLE_RATE),
            playhead_sample=playhead,
        )
        tr.on_tick(playhead)
    # Phase 2: generator stalls for 10 s — playhead keeps moving, no
    # slices land, staleness at the playhead grows.
    for _ in range(100):
        clock.advance(0.1)
        playhead += int(0.1 * SAMPLE_RATE)
        tr.on_tick(playhead)
    # Phase 3: generator resumes but writes BEHIND the playhead.
    clock.advance(0.1)
    rec = tr.on_slice(
        start_sample=playhead - SAMPLE_RATE,
        num_samples=int(0.36 * SAMPLE_RATE),
        playhead_sample=playhead,
    )
    assert rec.lead_s == pytest.approx(-1.0, abs=0.01)
    rep = tr.report(window_s=None, stale_threshold_s=3.0)
    assert rep["slices"]["behind_playhead"] == 1
    assert rep["ticks"]["stale_ticks"] > 0
    assert rep["ticks"]["staleness_s"]["max"] > 9.0
    assert "worst_stale" in rep["ticks"]


def test_tracker_window_filters_old_records():
    clock = FakeClock()
    tr = LagTracker(60 * SAMPLE_RATE, now_fn=clock)
    tr.on_slice(start_sample=0, num_samples=100, playhead_sample=0)
    clock.advance(100.0)
    tr.on_slice(
        start_sample=SAMPLE_RATE, num_samples=100,
        playhead_sample=SAMPLE_RATE // 2,
    )
    rep = tr.report(window_s=10.0)
    assert rep["slices"]["count"] == 1


def test_tracker_timeline_rollup():
    clock = FakeClock()
    tr = LagTracker(60 * SAMPLE_RATE, now_fn=clock)
    for i in range(30):
        clock.advance(0.1)
        ph = i * int(0.1 * SAMPLE_RATE)
        tr.on_slice(
            start_sample=ph + SAMPLE_RATE // 4, num_samples=1000,
            playhead_sample=ph,
        )
        tr.on_tick(ph)
    rep = tr.report(window_s=None, include_timeline=True)
    tl = rep["timeline"]
    assert len(tl) == 3
    assert all("min_lead_s" in row and "max_staleness_s" in row for row in tl)


# ---------------------------------------------------------------------------
# Slice codec round-trip vs the server-side SliceCodec
# ---------------------------------------------------------------------------


def _server_and_client_buffers(n_samples=48000, channels=2, seed=0):
    rng = np.random.default_rng(seed)
    initial = rng.standard_normal((n_samples, channels)).astype(np.float32)
    codec = SliceCodec(initial)
    client_mirror = initial.copy()
    return rng, codec, client_mirror


def test_slice_roundtrip_matches_server_mirror():
    rng, codec, client_mirror = _server_and_client_buffers()
    for i in range(5):
        ss = i * 4000
        fresh = rng.standard_normal((4000, 2)).astype(np.float32)
        frame = codec.encode(
            fresh, start_sample=ss, channels=2,
            tick_ms=12.5, dec_ms=3.5, num_gens=i,
        )
        sl = decode_slice_frame(frame)
        assert sl.flags == SLICE_FLAG_DELTA
        assert sl.start_sample == ss
        assert sl.num_samples == 4000
        assert sl.tick_ms == pytest.approx(12.5)
        assert sl.dec_ms == pytest.approx(3.5)
        assert sl.num_gens == i
        apply_slice(client_mirror, sl)
    # The client reconstruction must stay byte-identical to the server's
    # mirror (the anti-ghosting invariant in SliceCodec.encode).
    np.testing.assert_array_equal(client_mirror, codec.mirror)


def test_slice_roundtrip_overlapping_windows_converge():
    rng, codec, client_mirror = _server_and_client_buffers()
    # Heavily overlapping re-patches of the same region, like the
    # windowed runner produces.
    target = rng.standard_normal((6000, 2)).astype(np.float32)
    for i in range(10):
        frame = codec.encode(
            target, start_sample=1000, channels=2,
            tick_ms=0.0, dec_ms=0.0, num_gens=i,
        )
        apply_slice(client_mirror, decode_slice_frame(frame))
    np.testing.assert_array_equal(client_mirror, codec.mirror)
    # And the reconstruction is within float16 quantization of truth.
    np.testing.assert_allclose(
        client_mirror[1000:7000], target, atol=2e-2,
    )


def test_decode_rejects_short_frame():
    with pytest.raises(ValueError):
        decode_slice_frame(b"\x01\x02")
