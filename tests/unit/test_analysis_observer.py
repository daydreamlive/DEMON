"""Canvas observer, scheduling policy and governor (no threads, no GPU)."""

import math

import numpy as np
import pytest

from acestep.analysis import scheduler as sch
from acestep.analysis.observer import (
    CONTENT_CHANGE,
    PLAYHEAD_STALE_S,
    CanvasObserver,
    content_change,
    content_fingerprint,
)
from acestep.analysis.scheduler import (
    BASIS_PROVISIONAL,
    BASIS_SETTLED,
    Governor,
    SchedulerState,
    basis_for,
    crossed,
    is_late,
    next_cell,
)
from acestep.streaming.audio_engine import AudioEngine
from acestep.streaming.events import AudioReady, EventBus, SwapReady

SR = 48000


class Clock:
    def __init__(self, t=100.0):
        self.t = t

    def __call__(self):
        return self.t


def _state(n=10, **kw):
    base = dict(
        now=100.0, cell_s=4.0, context_s=0.5, duration_s=4.0 * n,
        refines_audio=True, dirty=[True] * n, dirty_since=[0.0] * n,
        last_analysed=[-math.inf] * n, horizon_s=8.0,
    )
    base.update(kw)
    return SchedulerState(**base)


# ---- scheduler ------------------------------------------------------------


def test_ring_picks_nearest_ahead_of_playhead():
    assert next_cell(_state(playhead_s=9.0)) == 3  # cell 3 starts at 12.0
    assert next_cell(_state(playhead_s=12.0)) == 3
    # Wraps at the canvas end.
    assert next_cell(_state(playhead_s=39.0)) == 0


def test_ring_horizon_then_rest_in_play_order():
    dirty = [False] * 10
    dirty[1] = True   # 4.0: 28 s ahead of a playhead at 16 (wrapped)
    dirty[7] = True   # 28.0: 12 s ahead, beyond an 8 s horizon
    st = _state(playhead_s=16.0, dirty=dirty)
    assert next_cell(st) == 7  # both beyond the horizon: play order
    dirty[5] = True   # 20.0: 4 s ahead, inside the horizon
    assert next_cell(_state(playhead_s=16.0, dirty=dirty)) == 5


def test_ring_loop_band_wraps_inside_the_band():
    # Band [8, 20): cells 2, 3, 4. Playhead at 19 wraps to 8 next.
    st = _state(playhead_s=19.0, loop_band=(8.0, 20.0))
    assert next_cell(st) == 2
    # Cells outside the band come after every in-band cell.
    dirty = [True] * 10
    dirty[2] = dirty[3] = dirty[4] = False
    assert next_cell(_state(playhead_s=19.0, loop_band=(8.0, 20.0), dirty=dirty)) == 5


def test_ring_without_playhead_is_write_order():
    since = [5.0, 1.0, 3.0] + [9.0] * 7
    assert next_cell(_state(dirty_since=since)) == 1


def test_append_only_waits_for_the_frontier():
    st = _state(refines_audio=False, frontier_s=None)
    assert next_cell(st) is None
    # Cell 0 window ends at 4.5 (cell + context); a frontier at 4.4 is short.
    assert next_cell(_state(refines_audio=False, frontier_s=4.4)) is None
    assert next_cell(_state(refines_audio=False, frontier_s=4.5)) == 0
    since = [3.0, 1.0] + [0.0] * 8
    assert next_cell(_state(refines_audio=False, frontier_s=9.0, dirty_since=since)) == 1


def test_reanalysis_cap_and_in_flight():
    last = [100.0 - sch.REANALYSIS_CAP_S + 0.1] + [-math.inf] * 9
    assert next_cell(_state(playhead_s=0.0, last_analysed=last)) == 1
    last[0] = 100.0 - sch.REANALYSIS_CAP_S
    assert next_cell(_state(playhead_s=0.0, last_analysed=last)) == 0
    assert next_cell(_state(playhead_s=0.0, in_flight=frozenset({0}))) == 1
    assert next_cell(_state(dirty=[False] * 10)) is None


def test_basis_late_and_crossing():
    assert basis_for(refines_audio=False, playhead_passed=False, revision_unchanged=False) == BASIS_SETTLED
    assert basis_for(refines_audio=True, playhead_passed=False, revision_unchanged=True) == BASIS_PROVISIONAL
    assert basis_for(refines_audio=True, playhead_passed=True, revision_unchanged=False) == BASIS_PROVISIONAL
    assert basis_for(refines_audio=True, playhead_passed=True, revision_unchanged=True) == BASIS_SETTLED

    assert is_late(12.0, 4.0, 13.0, refines_audio=True, period_len=40.0)
    assert not is_late(12.0, 4.0, 10.0, refines_audio=True, period_len=40.0)
    assert not is_late(12.0, 4.0, 17.0, refines_audio=True, period_len=40.0)
    assert is_late(12.0, 4.0, 30.0, refines_audio=False)
    assert not is_late(12.0, 4.0, None, refines_audio=True, period_len=40.0)

    assert crossed(15.0, 16.5, 16.0, 40.0)
    assert not crossed(15.0, 15.9, 16.0, 40.0)
    assert crossed(39.5, 0.5, 40.0, 40.0)        # wrap at the period end
    assert not crossed(5.0, 30.0, 16.0, 40.0)    # a seek, not play


# ---- governor -------------------------------------------------------------


def test_governor_tick_baseline_backoff_and_recovery():
    g = Governor(refines_audio=True)
    assert g.mode(0.0) == "unknown"
    t = 0.0
    for _ in range(sch.TICK_BASELINE_AFTER):
        t += 0.1
        g.observe_tick(100.0, t)
    assert g.mode(t) == "ok" and g.may_submit(t)
    for _ in range(30):
        t += 0.1
        g.observe_tick(200.0, t)
    assert g.mode(t) == "backoff" and not g.may_submit(t)
    for _ in range(60):
        t += 0.1
        g.observe_tick(100.0, t)
    # Healthy again, but recovery needs RECOVERY_S of continuous health.
    assert g.healthy(t)


def test_governor_unknown_is_paced():
    g = Governor(refines_audio=True)
    assert g.may_submit(10.0)
    g.note_submit(10.0)
    assert not g.may_submit(10.0 + sch.UNKNOWN_HEALTH_INTERVAL_S - 0.1)
    assert g.may_submit(10.0 + sch.UNKNOWN_HEALTH_INTERVAL_S)


def test_governor_frontier_lead_and_slice_lead():
    g = Governor(refines_audio=False)
    g.observe_frontier_lead(5.0, 1.0)
    assert g.mode(1.0) == "ok"
    g.observe_frontier_lead(0.5, 2.0)
    assert g.mode(2.0) == "backoff"
    g.observe_frontier_lead(5.0, 2.5)
    assert g.mode(2.0 + sch.RECOVERY_S - 0.1) == "backoff"
    assert g.mode(2.0 + sch.RECOVERY_S) == "ok"

    r = Governor(refines_audio=True)
    r.observe_slice_lead(-0.2, reported_at=5.0, now=5.0)
    assert r.mode(5.0) == "backoff"
    # A stale report no longer counts.
    s = Governor(refines_audio=True)
    s.observe_slice_lead(-0.2, reported_at=0.0, now=sch.HEALTH_REPORT_FRESH_S + 1.0)
    assert s.mode(sch.HEALTH_REPORT_FRESH_S + 1.0) == "unknown"


# ---- observer -------------------------------------------------------------


def _obs(seconds=16.0, clock=None, bus=None, **kw):
    eng = AudioEngine(np.zeros((int(seconds * SR), 2), np.float32), SR)
    clock = clock or Clock()
    obs = CanvasObserver(eng, bus, 4.0, 0.5, clock=clock, **kw)
    return eng, obs, clock


def test_observer_starts_dirty_and_tracks_writes():
    eng, obs, clock = _obs()
    assert obs.n_cells == 4 and all(obs.dirty)
    for i in range(4):
        obs.snapshot_cell(i)
    assert not any(obs.dirty)
    # A write inside cell 1 near its start also dirties cell 0 through the
    # context (cell 0's window reaches 4.5 s).
    clock.t = 101.0
    eng.patch_window(np.ones((SR // 10, 2), np.float32), int(4.2 * SR))
    # Analysed cells are only pending until the worker's content check.
    assert obs.pending == [True, True, False, False] and not any(obs.dirty)
    assert obs.check_content() == 2
    assert obs.dirty == [True, True, False, False]
    assert not any(obs.pending)
    assert obs.dirty_since[0] == 101.0
    assert obs.cell_revision[1] == eng.revision
    assert obs.frontier_s() == pytest.approx(4.3)


def _noise(seconds, seed=0):
    rng = np.random.default_rng(seed)
    return (0.1 * rng.standard_normal((int(seconds * SR), 2))).astype(np.float32)


def test_fingerprint_change_scores():
    a = _noise(5.0, seed=1)[:, 0]
    fp = content_fingerprint(a, SR)
    assert fp.shape == (50, 24)
    assert content_change(content_fingerprint(a, SR), fp) == 0.0
    assert content_change(content_fingerprint(a * 0.999, SR), fp) < CONTENT_CHANGE
    assert content_change(content_fingerprint(_noise(5.0, seed=2)[:, 0] * 2.0, SR), fp) > CONTENT_CHANGE
    silent = content_fingerprint(np.zeros(5 * SR, np.float32), SR)
    assert content_change(silent, silent) == 0.0
    assert content_change(fp, silent) == math.inf
    assert content_change(fp[:10], fp) == math.inf


def test_identical_rewrite_keeps_the_cell_clean_and_current():
    eng, obs, _ = _obs()
    src = _noise(16.0)
    eng.patch_window(src, 0)
    req = obs.snapshot_cell(1)
    eng.patch_window(src[4 * SR:8 * SR].copy(), 4 * SR)  # same music again
    assert obs.pending[1] and obs.result_verdict(1, req.source_epoch, req.revision) == "unchecked"
    assert obs.check_content() == 0
    assert not obs.dirty[1] and not obs.pending[1]
    assert obs.is_current(1, req.source_epoch, req.revision)
    eng.patch_window(_noise(4.0, seed=3) * 3.0, 4 * SR)  # new music
    assert obs.check_content() >= 1
    assert obs.dirty[1] and not obs.is_current(1, req.source_epoch, req.revision)


def test_observer_snapshot_request_shape_and_staleness():
    eng, obs, _ = _obs()
    eng.patch_window(np.full((16 * SR, 2), 0.25, np.float32), 0)
    req = obs.snapshot_cell(1, {"instruments": ["piano"]})
    assert req.audio.ndim == 1 and req.audio.dtype == np.float32
    assert req.t0_s == 4.0 and req.span_s == 4.0
    assert req.context_lead_s == pytest.approx(0.5) and req.context_tail_s == pytest.approx(0.5)
    assert len(req.audio) == 5 * SR
    assert req.options == {"instruments": ["piano"]}
    assert obs.is_current(1, req.source_epoch, req.revision)
    # Edge cell: context clipped at the canvas start.
    first = obs.snapshot_cell(0)
    assert first.context_lead_s == 0.0 and len(first.audio) == int(4.5 * SR)
    # A later write to the window makes the earlier snapshot stale.
    eng.patch_window(np.ones((100, 2), np.float32), int(8.2 * SR))
    assert not obs.is_current(1, req.source_epoch, req.revision)
    assert obs.is_current(0, first.source_epoch, first.revision)
    obs.close()
    eng.patch_window(np.ones((100, 2), np.float32), 0)  # detached: no error


def test_observer_epoch_from_swap_ready_and_ticks():
    bus = EventBus()
    epochs, ticks = [], []
    eng = AudioEngine(np.zeros((16 * SR, 2), np.float32), SR)
    clock = Clock()
    obs = CanvasObserver(
        eng, bus, 4.0, 0.5, clock=clock,
        on_tick=lambda ms, now: ticks.append(ms), on_epoch=epochs.append,
    )
    req = obs.snapshot_cell(0)
    eng.swap(np.zeros((8 * SR, 2), np.float32))
    assert obs.n_cells == 2 and obs.frontier_s() is None
    bus.publish(SwapReady(**_swap_ready_kwargs(epoch=1)))
    bus.publish(AudioReady(
        audio=np.zeros((1, 2), np.float32), start_sample=0, num_samples=1,
        channels=2, tick_ms=42.0, dec_ms=1.0, num_gens=1, params={},
    ))
    _drain(bus)
    assert epochs == [1] and ticks == [42.0]
    assert obs.epoch == 1
    assert not obs.is_current(0, req.source_epoch, req.revision)
    obs.close()
    bus.close()


def test_observer_playhead_reports_and_no_playhead():
    eng, obs, clock = _obs(seconds=40.0)
    assert obs.playhead_s() is None and not obs.has_playhead()
    eng.position = 10 * SR
    eng.position_report_wall_s = clock.t
    clock.t += 0.5
    assert obs.playhead_s() == pytest.approx(10.5)
    eng.loop_band = (8.0, 10.2)
    assert obs.playhead_s() == pytest.approx(8.3)
    clock.t += PLAYHEAD_STALE_S
    assert obs.playhead_s() is None


def _swap_ready_kwargs(epoch):
    return dict(
        duration=8.0, sample_rate=SR, channels=2, bpm=120, key="C major",
        time_signature="4", fixture_name=None,
        initial_buffer=np.zeros((1, 2), np.float32), source_epoch=epoch,
    )


def _drain(bus, timeout=2.0):
    import time

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if all(len(s._buf) == 0 for s in bus.subscribers()):
            time.sleep(0.02)
            return
        time.sleep(0.005)
