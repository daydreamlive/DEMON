"""Unit coverage for client playhead-report staleness compensation.

The failure mode under test: when params messages spend time queued
(congested uplink, tunnel buffering, server recv backlog), each arriving
``playback_pos`` describes where the client's playhead WAS at send time.
Re-anchoring the runner's playhead clock on those values verbatim walks
its estimate steadily into the past, so freshly rendered slices land
behind the listener and the client audibly falls back to the raw source.

``ReportStalenessEstimator`` measures each report's queueing delay from
the client send stamp; ``_RemotePlayheadClock`` advances its anchor by
that amount (via ``audio_eng.position_staleness_s``) so the estimate
stays pinned to the client's true playhead.
"""

import time

import numpy as np
import pytest

from acestep.streaming.audio_engine import AudioEngine
from acestep.streaming.generator_backend import LeadProfile
from acestep.streaming.pipeline_runner import (
    SAMPLE_RATE,
    PipelineRunner,
    ReportStalenessEstimator,
    _RemotePlayheadClock,
)


def _make_engine(duration_s: float = 60.0) -> AudioEngine:
    data = np.zeros((int(SAMPLE_RATE * duration_s), 2), dtype=np.float32)
    return AudioEngine(data, SAMPLE_RATE)


# ---------------------------------------------------------------------------
# ReportStalenessEstimator
# ---------------------------------------------------------------------------

def test_estimator_fresh_reports_read_zero():
    est = ReportStalenessEstimator()
    # Constant offset = clock origin delta + constant transit. None of it
    # is queueing, so staleness must be ~0 throughout.
    for i in range(100):
        now = 1000.0 + i * 0.008
        client = now - 5.0  # arbitrary origin delta + fixed transit
        assert est.staleness_s(client, now) == pytest.approx(0.0, abs=1e-9)


def test_estimator_measures_queueing_excess():
    est = ReportStalenessEstimator()
    # Establish the baseline with fresh reports...
    for i in range(50):
        now = 1000.0 + i * 0.008
        est.staleness_s(now - 5.0, now)
    # ...then a congestion episode: reports arrive with growing extra
    # delay on top of the baseline offset. Staleness must track the
    # excess, not the absolute offset.
    for extra in (0.1, 0.5, 1.0, 4.8):
        now = 1100.0
        assert est.staleness_s(now - 5.0 - extra, now) == pytest.approx(
            extra, abs=1e-6,
        )


def test_estimator_window_evicts_stale_minima():
    est = ReportStalenessEstimator()
    est.staleness_s(995.0, 1000.0)  # offset 5.0 baseline
    # Far past the window, only newer (larger-offset) buckets remain; the
    # old minimum must no longer anchor the baseline, so a report at the
    # new steady offset reads as fresh again (graceful degradation).
    later = 1000.0 + est._WINDOW_S * 3
    est.staleness_s(later - 7.0, later)
    assert est.staleness_s(later + 1.0 - 7.0, later + 1.0) == pytest.approx(
        0.0, abs=1e-9,
    )


# ---------------------------------------------------------------------------
# _RemotePlayheadClock + staleness
# ---------------------------------------------------------------------------

def test_clock_advances_anchor_by_staleness(monkeypatch):
    eng = _make_engine()
    fake_now = [1000.0]
    monkeypatch.setattr(
        "acestep.streaming.pipeline_runner.time.monotonic",
        lambda: fake_now[0],
    )
    clock = _RemotePlayheadClock(eng)

    # A report stamped 2 s stale: the client's true playhead is at
    # observed + 2 s. The clock must project forward, not anchor on the
    # raw observed value.
    eng.position = 10 * SAMPLE_RATE
    eng.position_staleness_s = 2.0
    assert clock.seconds() == pytest.approx(12.0, abs=1e-3)

    # Free-run continues from the projected anchor.
    fake_now[0] += 1.0
    assert clock.seconds() == pytest.approx(13.0, abs=1e-3)


def test_clock_without_stamp_matches_legacy(monkeypatch):
    eng = _make_engine()
    fake_now = [1000.0]
    monkeypatch.setattr(
        "acestep.streaming.pipeline_runner.time.monotonic",
        lambda: fake_now[0],
    )
    clock = _RemotePlayheadClock(eng)
    # No client stamp -> staleness 0.0 -> behavior identical to the
    # pre-compensation clock: anchor on observed, advance by wall time.
    eng.position = 30 * SAMPLE_RATE
    assert clock.seconds() == pytest.approx(30.0, abs=1e-3)
    fake_now[0] += 0.5
    assert clock.seconds() == pytest.approx(30.5, abs=1e-3)


# ---------------------------------------------------------------------------
# Transport-lead controller (client landing-lead feedback)
# ---------------------------------------------------------------------------

class _FakeBackend:
    name = "fake"

    def lead_profile(self):
        return LeadProfile()


class _FakeState:
    running = False
    params: dict = {}
    last_activity_ts = 0.0


def _make_runner():
    eng = _make_engine()
    runner = PipelineRunner(
        _FakeBackend(), eng, state=_FakeState(), vae_window=0.36,
    )
    return runner, eng


def test_transport_extra_raises_on_negative_lead():
    runner, eng = _make_runner()
    base = runner._decode_advance_s()
    eng.observed_slice_lead_s = -1.0
    eng.observed_slice_lead_wall_s = time.monotonic()
    runner._fold_slice_lead_report()
    # Deficit = margin - (-1.0); the advance grows by exactly that.
    deficit = runner._slice_lead_margin_s + 1.0
    assert runner._transport_extra_s == pytest.approx(deficit)
    assert runner._decode_advance_s() == pytest.approx(base + deficit)


def test_transport_extra_rate_limited_per_report():
    runner, eng = _make_runner()
    eng.observed_slice_lead_s = -0.5
    eng.observed_slice_lead_wall_s = time.monotonic()
    runner._fold_slice_lead_report()
    once = runner._transport_extra_s
    # Same report folded again: no double-count.
    runner._fold_slice_lead_report()
    assert runner._transport_extra_s == pytest.approx(once)
    # A NEW report (fresh wall stamp) raises additively.
    eng.observed_slice_lead_wall_s = time.monotonic() + 0.01
    runner._fold_slice_lead_report()
    assert runner._transport_extra_s == pytest.approx(once * 2)


def test_transport_extra_capped_and_healthy_reports_dont_raise():
    runner, eng = _make_runner()
    eng.observed_slice_lead_s = -100.0
    eng.observed_slice_lead_wall_s = time.monotonic()
    runner._fold_slice_lead_report()
    assert runner._transport_extra_s == runner._transport_extra_cap_s
    # Healthy lead (above margin): no further raise.
    runner._transport_extra_s = 0.4
    eng.observed_slice_lead_s = 0.3
    eng.observed_slice_lead_wall_s = time.monotonic() + 0.01
    runner._fold_slice_lead_report()
    assert runner._transport_extra_s == pytest.approx(0.4)


def test_transport_extra_ignored_while_loop_band_armed():
    runner, eng = _make_runner()
    eng.loop_band = (1.0, 5.0)
    eng.observed_slice_lead_s = -3.0
    eng.observed_slice_lead_wall_s = time.monotonic()
    runner._fold_slice_lead_report()
    assert runner._transport_extra_s == 0.0


def test_transport_decay_holds_in_band_and_releases_on_headroom():
    runner, eng = _make_runner()
    runner._transport_extra_s = 1.0
    # Recent report sitting INSIDE the hold band (margin..margin+hyst):
    # no decay — this is the saturated-link steady state.
    eng.observed_slice_lead_s = runner._slice_lead_margin_s + 0.05
    eng.observed_slice_lead_wall_s = time.monotonic()
    runner._fold_slice_lead_report()
    runner._note_decode_gap()
    assert runner._transport_extra_s == pytest.approx(1.0)
    # Report with comfortable headroom: decay engages.
    eng.observed_slice_lead_s = (
        runner._slice_lead_margin_s
        + runner._transport_decay_hysteresis_s
        + 0.5
    )
    eng.observed_slice_lead_wall_s = time.monotonic() + 0.01
    runner._fold_slice_lead_report()
    time.sleep(0.02)
    runner._note_decode_gap()
    assert runner._transport_extra_s < 1.0
