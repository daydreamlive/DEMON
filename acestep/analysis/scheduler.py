"""Pure scheduling policy for an analysis tap: which cell next, what basis.

No threads, no clocks of its own, no numpy: every input arrives in a
:class:`SchedulerState` snapshot (or as arguments) so the policy is unit
testable on its own. The tap (:mod:`acestep.analysis.tap`) owns the loop
and calls in here once per wake-up.

Two canvas kinds, from the backend's ``Capabilities.refines_audio``:

- **append-only** (``refines_audio`` False, MRT2 / MiniMax): audio is
  written once at a moving frontier. A dirty cell is analysed only when
  its whole input window (cell plus context, clipped to the canvas) lies
  behind the frontier, oldest write first. Results are ``settled``.
- **ring** (``refines_audio`` True, ACE / SA3 / YuE2): the whole window
  is re-rendered lap after lap. Dirty cells nearest ahead of the playhead
  in play order go first (wrapping inside the loop band when one is
  armed), so notes usually land before they sound; the rest follow in
  play order. Without playhead reports, write order. Results are
  ``provisional`` until the playhead has passed the cell with its
  revision unchanged since the analysis, then ``settled``.

A cell is re-analysed at most once per :data:`REANALYSIS_CAP_S`, which
bounds the cost of continuous denoise churn. Rewrites that leave a cell's
music unchanged never reach the scheduler as dirty: the observer's
content check (:func:`acestep.analysis.observer.content_fingerprint`)
filters them out first.

The :class:`Governor` holds work back when the session's own health
degrades. It has no absolute ceiling: it compares the session against
its own baseline.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Sequence

#: Minimum seconds between two analyses of the same cell. Set from the
#: live run.
REANALYSIS_CAP_S = 2.0

#: Tick-time EMA smoothing factor for ring canvases. Set from the live run.
TICK_EMA_ALPHA = 0.1
#: Ticks observed before the EMA is trusted as a baseline. Set from the
#: live run.
TICK_BASELINE_AFTER = 20
#: Tick EMA above this multiple of the baseline is unhealthy. Set from the
#: live run.
TICK_UNHEALTHY_RATIO = 1.25
#: Append-only canvases: frontier lead over the playhead below this many
#: seconds is unhealthy. Set from the live run.
FRONTIER_LEAD_MIN_S = 1.0
#: Seconds of continuous health before backoff ends. Set from the live run.
RECOVERY_S = 3.0
#: When health is unknown (no ticks, no frontier), at most one request per
#: this many seconds. Set from the live run.
UNKNOWN_HEALTH_INTERVAL_S = 2.0
#: A buffer-health report (client slice lead) older than this is ignored.
#: Set from the live run.
HEALTH_REPORT_FRESH_S = 2.0

BASIS_PROVISIONAL = "provisional"
BASIS_SETTLED = "settled"


@dataclass
class SchedulerState:
    """Everything :func:`next_cell` needs, captured at one instant.

    Cell ``i`` covers ``[i * cell_s, min((i + 1) * cell_s, duration_s))``
    canvas seconds. ``dirty_since[i]`` is the monotonic time of the first
    write that dirtied the cell since its last analysis (write order);
    ``last_analysed[i]`` is the monotonic time the cell was last submitted
    (``-inf`` when never). ``frontier_s`` is the append-only write
    frontier (``None`` before the first write of the epoch). ``playhead_s``
    is ``None`` when the session has no live playhead.
    """

    now: float
    cell_s: float
    context_s: float
    duration_s: float
    refines_audio: bool
    dirty: Sequence[bool]
    dirty_since: Sequence[float]
    last_analysed: Sequence[float]
    horizon_s: float = 8.0
    frontier_s: Optional[float] = None
    playhead_s: Optional[float] = None
    loop_band: Optional[tuple] = None
    in_flight: frozenset = field(default_factory=frozenset)

    @property
    def n_cells(self) -> int:
        return len(self.dirty)

    def cell_start(self, i: int) -> float:
        return i * self.cell_s

    def cell_end(self, i: int) -> float:
        return min((i + 1) * self.cell_s, self.duration_s)

    def window_end(self, i: int) -> float:
        """End of cell ``i``'s input window (cell plus context), clipped."""
        return min(self.cell_end(i) + self.context_s, self.duration_s)


def n_cells_for(duration_s: float, cell_s: float) -> int:
    """Number of cells covering a canvas of ``duration_s`` seconds."""
    if duration_s <= 0 or cell_s <= 0:
        return 0
    return int(math.ceil(duration_s / cell_s - 1e-9))


def _band(state: SchedulerState) -> Optional[tuple]:
    band = state.loop_band
    if band is None:
        return None
    a, b = float(band[0]), float(band[1])
    a = max(0.0, a)
    b = min(state.duration_s, b)
    if b - a <= 0.0:
        return None
    return a, b


def _overlaps(state: SchedulerState, i: int, band: tuple) -> bool:
    return state.cell_start(i) < band[1] and state.cell_end(i) > band[0]


def play_distance(t: float, playhead: float, period_len: float) -> float:
    """Seconds of forward play from ``playhead`` until ``t`` when play
    wraps every ``period_len`` seconds (the canvas, or the loop band)."""
    if period_len <= 0:
        return 0.0
    return (t - playhead) % period_len


def _eligible(state: SchedulerState, i: int) -> bool:
    if not state.dirty[i] or i in state.in_flight:
        return False
    return state.now - state.last_analysed[i] >= REANALYSIS_CAP_S


def next_cell(state: SchedulerState) -> Optional[int]:
    """The cell to analyse next, or ``None`` when nothing is due."""
    n = state.n_cells
    if n == 0:
        return None
    candidates = [i for i in range(n) if _eligible(state, i)]
    if not candidates:
        return None

    if not state.refines_audio:
        if state.frontier_s is None:
            return None
        ready = [i for i in candidates if state.window_end(i) <= state.frontier_s + 1e-9]
        if not ready:
            return None
        return min(ready, key=lambda i: (state.dirty_since[i], i))

    if state.playhead_s is None:
        return min(candidates, key=lambda i: (state.dirty_since[i], i))

    band = _band(state)
    ph = float(state.playhead_s)

    def key(i: int):
        start = state.cell_start(i)
        if band is not None and _overlaps(state, i, band):
            # Inside the band the play order wraps at the band.
            s = min(max(start, band[0]), band[1])
            d = play_distance(s, ph, band[1] - band[0])
            group = 0
        else:
            d = play_distance(start, ph, state.duration_s)
            group = 1 if band is not None else 0
        within = 0 if d <= state.horizon_s else 1
        return (group, within, d, i)

    return min(candidates, key=key)


def is_late(
    range_start_s: float,
    span_s: float,
    playhead_s: Optional[float],
    *,
    refines_audio: bool,
    period_len: float = 0.0,
) -> bool:
    """True when the playhead has already passed ``range_start_s`` in play
    order. Append-only: the playhead is past the range start. Ring: the
    range is sounding now (the playhead sits inside it within the wrap
    period), so part of it was heard before the items arrived. Without a
    playhead, never late."""
    if playhead_s is None:
        return False
    if not refines_audio or period_len <= 0:
        return playhead_s > range_start_s
    behind = play_distance(playhead_s, range_start_s, period_len)
    return behind < span_s


def crossed(prev_s: Optional[float], cur_s: Optional[float], t: float,
            period_len: float) -> bool:
    """True when forward play from ``prev_s`` to ``cur_s`` passed ``t``.

    A jump of half the period or more is a seek, not play, and crosses
    nothing."""
    if prev_s is None or cur_s is None or period_len <= 0:
        return False
    step = play_distance(cur_s, prev_s, period_len)
    if step <= 0.0 or step >= period_len / 2.0:
        return False
    to_t = play_distance(t, prev_s, period_len)
    return 0.0 < to_t <= step


def basis_for(*, refines_audio: bool, playhead_passed: bool, revision_unchanged: bool) -> str:
    """``settled`` for append-only results (they are only scheduled behind
    the frontier) and for ring cells the playhead passed with their
    revision unchanged since the analysis; otherwise ``provisional``."""
    if not refines_audio:
        return BASIS_SETTLED
    if playhead_passed and revision_unchanged:
        return BASIS_SETTLED
    return BASIS_PROVISIONAL


class Governor:
    """Backs analysis off when the session's own health degrades.

    Signals, buffer health first:

    - client-reported slice landing lead (fresh reports only): negative
      means slices land after the listener heard the region;
    - append-only canvases: frontier lead over the playhead below
      :data:`FRONTIER_LEAD_MIN_S`;
    - ring canvases: tick-time EMA above :data:`TICK_UNHEALTHY_RATIO`
      times its baseline. The baseline is the lowest EMA seen after
      :data:`TICK_BASELINE_AFTER` ticks, so load added once analysis runs
      never raises it.

    With no tick and no frontier observed, health is unknown and work is
    paced at one request per :data:`UNKNOWN_HEALTH_INTERVAL_S`. Backoff
    ends after :data:`RECOVERY_S` of continuous health. All methods take
    ``now`` (monotonic seconds) so tests drive time explicitly.
    """

    def __init__(self, refines_audio: bool):
        self.refines_audio = bool(refines_audio)
        self._ema: Optional[float] = None
        self._ticks = 0
        self._baseline: Optional[float] = None
        self._frontier_lead: Optional[float] = None
        self._slice_lead: Optional[float] = None
        self._slice_lead_at = -math.inf
        self._last_unhealthy = -math.inf
        self._last_submit = -math.inf

    # ---- observations -------------------------------------------------

    def observe_tick(self, tick_ms: float, now: float) -> None:
        if tick_ms is None or not math.isfinite(tick_ms) or tick_ms <= 0:
            return
        self._ticks += 1
        if self._ema is None:
            self._ema = float(tick_ms)
        else:
            self._ema += TICK_EMA_ALPHA * (float(tick_ms) - self._ema)
        if self._ticks >= TICK_BASELINE_AFTER:
            if self._baseline is None or self._ema < self._baseline:
                self._baseline = self._ema
        self._update(now)

    def observe_frontier_lead(self, seconds: float, now: float) -> None:
        self._frontier_lead = float(seconds)
        self._update(now)

    def observe_slice_lead(self, seconds: Optional[float], reported_at: float, now: float) -> None:
        """Fold a client slice-lead report (``AudioEngine.
        observed_slice_lead_s`` and its wall stamp)."""
        if seconds is None:
            return
        self._slice_lead = float(seconds)
        self._slice_lead_at = float(reported_at)
        self._update(now)

    # ---- verdicts -----------------------------------------------------

    def _unhealthy_now(self, now: float) -> bool:
        if (self._slice_lead is not None
                and now - self._slice_lead_at <= HEALTH_REPORT_FRESH_S
                and self._slice_lead < 0.0):
            return True
        if not self.refines_audio:
            return self._frontier_lead is not None and self._frontier_lead < FRONTIER_LEAD_MIN_S
        if self._baseline is not None and self._ema is not None:
            return self._ema > TICK_UNHEALTHY_RATIO * self._baseline
        return False

    def _update(self, now: float) -> None:
        if self._unhealthy_now(now):
            self._last_unhealthy = now

    def known(self) -> bool:
        if self.refines_audio:
            return self._ema is not None
        return self._frontier_lead is not None

    def healthy(self, now: float) -> bool:
        self._update(now)
        return now - self._last_unhealthy >= RECOVERY_S

    def mode(self, now: float) -> str:
        """``"unknown"``, ``"backoff"`` or ``"ok"``."""
        if not self.healthy(now):
            return "backoff"
        if not self.known():
            return "unknown"
        return "ok"

    def may_submit(self, now: float) -> bool:
        m = self.mode(now)
        if m == "backoff":
            return False
        if m == "unknown":
            return now - self._last_submit >= UNKNOWN_HEALTH_INTERVAL_S
        return True

    def note_submit(self, now: float) -> None:
        self._last_submit = now
