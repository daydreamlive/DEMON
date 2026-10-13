"""Session-side canvas observer: which analysis cells are dirty, and why.

The observer is the analysis seam's read-only view of one session's
canvas. It never writes audio. It learns about writes from the
:class:`~acestep.streaming.audio_engine.AudioEngine` write hook (the record
of every ``swap`` / ``patch_window`` / ``patch``, with the canvas revision
it produced), not from the bus: an ``AudioReady`` can describe a trimmed
subset of a patched window and can be dropped under backpressure. The bus
is only the wake-up: ``AudioReady`` carries ``tick_ms`` for the governor,
``SwapReady`` carries the new ``source_epoch``.

Per cell it keeps the revision of the newest write that touched the cell's
input window (cell plus context), a dirty flag with the time it was first
dirtied (write order), and the time it was last handed to an engine. It
also tracks the write frontier (end of the newest write since the epoch;
on an append-only canvas writes are contiguous, so this is the furthest
point written, and it follows a wrap), the loop band (read from the audio
engine) and the playhead.

Cell audio is read with :meth:`AudioEngine.snapshot` under the engine's
lock, tagged with ``(source_epoch, revision)``; a result is published only
if both still match (:meth:`CanvasObserver.result_verdict`). The
frontier-follow path reads arbitrary ranges instead
(:meth:`CanvasObserver.snapshot_range`); on a canvas that only grows at
the frontier every later write lands past such a range, so its result is
current unless the epoch changed or a write landed behind the frontier
inside the range (:meth:`CanvasObserver.range_verdict`).

Ring canvases rewrite every cell on every pass, mostly with the same
music. A write to a cell that has been analysed therefore only marks it
*pending*; the tap's worker later fingerprints the cell's audio
(:func:`content_fingerprint`) and compares it with the fingerprint of the
audio last analysed (:meth:`CanvasObserver.check_content`). Below
:data:`CONTENT_CHANGE` the write did not change the content: the cell
stays clean and its result keeps its revision (and so its settled
status). At or above it the cell is dirty, as for any write. The write
hook itself never touches audio.
"""

from __future__ import annotations

import math
import threading
import time
from collections import deque
from typing import Any, Callable, Mapping, Optional

import numpy as np
from loguru import logger

from acestep.analysis.engine import AnalysisRequest
from acestep.analysis.scheduler import FollowWindow, SchedulerState, n_cells_for
from acestep.streaming.audio_engine import WRITE_SWAP
from acestep.streaming.events import AudioReady, SwapReady

#: Without a client playhead report for this long (and no local output
#: stream), the session counts as having no playhead.
PLAYHEAD_STALE_S = 2.0

#: Relative L1 change between two cell fingerprints at or above which a
#: rewrite counts as new content (the cell is re-analysed). Set from the
#: live run.
CONTENT_CHANGE = 0.15
#: Fingerprint geometry: band count, frame length and band edges.
FINGERPRINT_BANDS = 24
FINGERPRINT_FRAME_S = 0.1
FINGERPRINT_FMIN_HZ = 40.0
FINGERPRINT_FMAX_HZ = 16000.0

#: Writes behind the frontier remembered for :meth:`CanvasObserver.
#: range_verdict`; older ones only raise a floor revision.
BACKWRITE_MEMORY = 64

#: Verdicts of :meth:`CanvasObserver.result_verdict`.
VERDICT_CURRENT = "current"
VERDICT_STALE = "stale"
VERDICT_UNCHECKED = "unchecked"


def content_fingerprint(audio: np.ndarray, sr: int) -> np.ndarray:
    """Coarse spectral fingerprint of mono ``audio``: energy in
    :data:`FINGERPRINT_BANDS` log-spaced bands per
    :data:`FINGERPRINT_FRAME_S` frame (Hann-windowed, non-overlapping),
    shape ``(frames, bands)``. Cheap enough to run per cell per tap pass
    (a 5 s window at 48 kHz is fifty 4800-point FFTs)."""
    frame = max(1, int(round(FINGERPRINT_FRAME_S * sr)))
    x = np.asarray(audio, dtype=np.float32).reshape(-1)
    n_frames = len(x) // frame
    if n_frames == 0:
        return np.zeros((0, FINGERPRINT_BANDS))
    window = np.hanning(frame).astype(np.float32)
    frames = x[: n_frames * frame].reshape(n_frames, frame) * window
    power = np.abs(np.fft.rfft(frames, axis=1)) ** 2
    freqs = np.fft.rfftfreq(frame, 1.0 / sr)
    fmax = min(FINGERPRINT_FMAX_HZ, sr / 2.0)
    edges = np.geomspace(FINGERPRINT_FMIN_HZ, fmax, FINGERPRINT_BANDS + 1)
    band = np.searchsorted(edges, freqs, side="right") - 1
    out = np.zeros((n_frames, FINGERPRINT_BANDS))
    for b in range(FINGERPRINT_BANDS):
        sel = band == b
        if sel.any():
            out[:, b] = power[:, sel].sum(axis=1)
    return out


def content_change(a: np.ndarray, b: np.ndarray) -> float:
    """Relative L1 distance of fingerprint ``a`` from reference ``b``:
    ``inf`` when the shapes differ, 0 when both are silent."""
    if a.shape != b.shape:
        return math.inf
    ref = float(np.abs(b).sum())
    diff = float(np.abs(a - b).sum())
    if ref <= 1e-12:
        return 0.0 if diff <= 1e-12 else math.inf
    return diff / ref


class CanvasObserver:
    """Dirty-cell, frontier, epoch and playhead bookkeeping for one session.

    ``cell_s`` / ``context_s`` come from the analyzer's spec. ``on_tick``
    (``tick_ms, now``) and ``on_epoch`` (``source_epoch``) are called on the
    bus subscriber's thread; keep them cheap. :meth:`close` detaches the
    write listener and the bus subscription.

    ``frontier_fn`` (the backend's optional ``analysis_frontier_s``) reports
    how much of the canvas is already final, in seconds, for a canvas that
    is final without being written (a static clip on an append-only
    family, whose frontier no write would ever reveal). It is read
    whenever the grid resets (attach, swap, epoch) and seeds the frontier;
    writes move it as usual. It is called outside this observer's lock and
    must not block (it can run inside a writer's notification).
    """

    def __init__(
        self,
        audio_eng,
        bus,
        cell_s: float,
        context_s: float,
        *,
        source_epoch: int = 0,
        on_tick: Optional[Callable[[float, float], None]] = None,
        on_epoch: Optional[Callable[[int], None]] = None,
        frontier_fn: Optional[Callable[[], Optional[float]]] = None,
        clock: Callable[[], float] = time.monotonic,
    ):
        if cell_s <= 0:
            raise ValueError("cell_s must be > 0")
        self.audio_eng = audio_eng
        self.sr = int(audio_eng.sr)
        self.cell_s = float(cell_s)
        self.context_s = max(0.0, float(context_s))
        self.epoch = int(source_epoch)
        self._on_tick = on_tick
        self._on_epoch = on_epoch
        self._frontier_fn = frontier_fn
        self._clock = clock
        self._lock = threading.Lock()
        self._closed = False

        now = clock()
        self._resize(len(audio_eng.current), int(audio_eng.revision), now)
        self.frontier_samples: Optional[int] = self._seed_frontier()

        audio_eng.add_write_listener(self._on_write)
        self._sub = bus.subscribe(self._on_event, name="analysis-observer") if bus is not None else None
        self._bus = bus

    # ---- grid ---------------------------------------------------------

    def _resize(self, n_samples: int, revision: int, now: float) -> None:
        """(Re)build the cell grid for a canvas of ``n_samples``; every
        cell starts dirty. Caller holds ``_lock`` (or is ``__init__``)."""
        self.n_samples = int(n_samples)
        self.duration_s = self.n_samples / self.sr if self.sr else 0.0
        n = n_cells_for(self.duration_s, self.cell_s)
        self.cell_revision = [revision] * n
        # Revision of the newest write that changed the cell's content; a
        # rewrite the content check finds unchanged does not bump it.
        self.content_revision = [revision] * n
        self.dirty = [True] * n
        self.dirty_since = [now] * n
        self.last_analysed = [-math.inf] * n
        # A write landed on an analysed cell; its content is unchecked.
        self.pending = [False] * n
        # Fingerprint of the audio last snapshotted for analysis.
        self.fingerprint: list = [None] * n
        # (revision, start sample) of recent writes that began behind the
        # frontier, and the newest revision forgotten from that record.
        self._backwrites: deque = deque()
        self._backwrite_floor = -1

    def _seed_frontier(self) -> Optional[int]:
        """The frontier after a grid reset: ``frontier_fn``'s answer in
        samples, clipped to the canvas, or ``None`` (learn it from writes).
        Called without ``_lock`` held."""
        fn = self._frontier_fn
        if fn is None:
            return None
        try:
            seconds = fn()
        except Exception as exc:  # noqa: BLE001 - a hook must never break the observer
            logger.warning("analysis_observer_frontier_fn_failed error={}", exc)
            return None
        if seconds is None:
            return None
        return max(0, min(int(round(float(seconds) * self.sr)), len(self.audio_eng.current)))

    @property
    def n_cells(self) -> int:
        return len(self.dirty)

    def cell_bounds_s(self, i: int) -> tuple:
        """``(start_s, end_s)`` of cell ``i`` in canvas seconds."""
        start = i * self.cell_s
        return start, min(start + self.cell_s, self.duration_s)

    def _window_samples(self, i: int) -> tuple:
        """``(s0, s1, t0, t1)``: cell ``i``'s input window in samples (cell
        plus context, clipped to the canvas) and the cell in seconds."""
        t0, t1 = self.cell_bounds_s(i)
        w0 = max(0.0, t0 - self.context_s)
        w1 = min(self.duration_s, t1 + self.context_s)
        return int(round(w0 * self.sr)), int(round(w1 * self.sr)), t0, t1

    def _cells_touching(self, start: int, end: int) -> range:
        """Cells whose input window overlaps samples ``[start, end)``."""
        n = self.n_cells
        if n == 0 or end <= start:
            return range(0)
        t0 = start / self.sr - self.context_s
        t1 = end / self.sr + self.context_s
        lo = max(0, int(math.floor(t0 / self.cell_s)))
        hi = min(n - 1, int(math.ceil(t1 / self.cell_s)) - 1)
        return range(lo, hi + 1) if hi >= lo else range(0)

    # ---- inputs -------------------------------------------------------

    def _on_write(self, start: int, end: int, revision: int, kind: str) -> None:
        """AudioEngine write listener (writer's thread, outside its lock)."""
        now = self._clock()
        seeded = self._seed_frontier() if kind == WRITE_SWAP else None
        with self._lock:
            if self._closed:
                return
            if kind == WRITE_SWAP:
                self._resize(end, revision, now)
                self.frontier_samples = seeded
                return
            for i in self._cells_touching(start, end):
                self.cell_revision[i] = max(self.cell_revision[i], revision)
                if self.dirty[i] or self.fingerprint[i] is None:
                    self._dirty_locked(i, now)
                else:
                    self.pending[i] = True
            if self.frontier_samples is not None and start < self.frontier_samples:
                if len(self._backwrites) >= BACKWRITE_MEMORY:
                    forgotten, _ = self._backwrites.popleft()
                    self._backwrite_floor = max(self._backwrite_floor, forgotten)
                self._backwrites.append((int(revision), int(start)))
            self.frontier_samples = int(end)

    def _dirty_locked(self, i: int, now: float) -> None:
        """Cell ``i``'s content changed: adopt its newest revision and put
        it on the to-do list. Caller holds ``_lock``."""
        self.content_revision[i] = self.cell_revision[i]
        self.pending[i] = False
        if not self.dirty[i]:
            self.dirty[i] = True
            self.dirty_since[i] = now

    def _on_event(self, event: Any) -> None:
        """Bus listener (subscriber thread): tick telemetry and epoch."""
        if self._closed:
            return
        if isinstance(event, AudioReady):
            if self._on_tick is not None:
                self._on_tick(float(event.tick_ms), self._clock())
        elif isinstance(event, SwapReady):
            epoch = int(getattr(event, "source_epoch", self.epoch + 1))
            now = self._clock()
            seeded = self._seed_frontier()
            with self._lock:
                if epoch == self.epoch:
                    return
                self.epoch = epoch
                self._resize(len(self.audio_eng.current), int(self.audio_eng.revision), now)
                self.frontier_samples = seeded
            if self._on_epoch is not None:
                self._on_epoch(epoch)

    # ---- playhead -----------------------------------------------------

    def has_playhead(self, now: Optional[float] = None) -> bool:
        """A local output stream drives ``position``, or a client playhead
        report arrived within :data:`PLAYHEAD_STALE_S`."""
        if getattr(self.audio_eng, "_stream", None) is not None:
            return True
        wall = float(getattr(self.audio_eng, "position_report_wall_s", 0.0) or 0.0)
        if wall <= 0.0:
            return False
        now = self._clock() if now is None else now
        return now - wall < PLAYHEAD_STALE_S

    def loop_band(self) -> Optional[tuple]:
        band = getattr(self.audio_eng, "loop_band", None)
        if band is None:
            return None
        a, b = float(band[0]), float(band[1])
        return (a, b) if b > a else None

    def playhead_s(self, now: Optional[float] = None) -> Optional[float]:
        """The playhead in canvas seconds, or ``None`` without one.

        A reported position is extrapolated by the time since the report
        (playback runs at 1x), wrapping inside the loop band when the
        report lies in it, else at the canvas end."""
        now = self._clock() if now is None else now
        if not self.has_playhead(now):
            return None
        duration = self.duration_s
        if duration <= 0:
            return None
        pos = float(self.audio_eng.position) / self.sr
        if getattr(self.audio_eng, "_stream", None) is None:
            wall = float(getattr(self.audio_eng, "position_report_wall_s", 0.0) or 0.0)
            elapsed = max(0.0, now - wall)
            band = self.loop_band()
            if band is not None and band[0] <= pos < band[1]:
                return band[0] + (pos - band[0] + elapsed) % (band[1] - band[0])
            pos += elapsed
        return pos % duration

    def frontier_s(self) -> Optional[float]:
        f = self.frontier_samples
        return None if f is None else f / self.sr

    # ---- tap-facing ---------------------------------------------------

    def scheduler_state(
        self,
        *,
        now: float,
        refines_audio: bool,
        horizon_s: float,
        in_flight: frozenset = frozenset(),
    ) -> SchedulerState:
        """A consistent snapshot of the bookkeeping for the scheduler."""
        playhead = self.playhead_s(now)
        with self._lock:
            return SchedulerState(
                now=now,
                cell_s=self.cell_s,
                context_s=self.context_s,
                duration_s=self.duration_s,
                refines_audio=refines_audio,
                dirty=list(self.dirty),
                dirty_since=list(self.dirty_since),
                last_analysed=list(self.last_analysed),
                horizon_s=horizon_s,
                frontier_s=self.frontier_s(),
                playhead_s=playhead,
                loop_band=self.loop_band(),
                in_flight=in_flight,
            )

    def snapshot_cell(self, i: int, options: Optional[Mapping[str, Any]] = None,
                      *, now: Optional[float] = None) -> AnalysisRequest:
        """Read cell ``i``'s input window and mark the cell analysed.

        The window is the cell plus ``context_s`` each side, clipped at the
        canvas edges, downmixed to mono float32. The cell's dirty flag is
        cleared here; :meth:`mark_dirty` puts it back when the result is
        dropped."""
        now = self._clock() if now is None else now
        with self._lock:
            if i < 0 or i >= self.n_cells:
                raise IndexError(f"cell {i} out of range ({self.n_cells} cells)")
            epoch = self.epoch
            s0, s1, t0, t1 = self._window_samples(i)
            self.dirty[i] = False
            self.last_analysed[i] = now
        audio, revision = self._mono_snapshot(s0, s1)
        fingerprint = content_fingerprint(audio, self.sr)
        with self._lock:
            if epoch == self.epoch and i < self.n_cells:
                self.fingerprint[i] = fingerprint
                self.pending[i] = False
                if self.cell_revision[i] > revision:
                    # A write landed between clearing the flag and the read.
                    self._dirty_locked(i, now)
                else:
                    self.content_revision[i] = self.cell_revision[i]
        w0, w1 = s0 / self.sr, s1 / self.sr
        return AnalysisRequest(
            audio=audio,
            sr=self.sr,
            t0_s=t0,
            span_s=t1 - t0,
            context_lead_s=t0 - w0,
            context_tail_s=w1 - t1,
            source_epoch=epoch,
            revision=int(revision),
            options=dict(options or {}),
        )

    def snapshot_range(self, window: FollowWindow,
                       options: Optional[Mapping[str, Any]] = None) -> AnalysisRequest:
        """Read a frontier-follow window: the audio ``[window_start_s,
        window_end_s)`` as mono float32, with the reported stretch
        ``[emit_lo_s, emit_hi_s)`` as the request's cell and the rest as
        context. Touches no cell bookkeeping."""
        with self._lock:
            epoch = self.epoch
            n = self.n_samples
        s0 = max(0, min(int(round(window.window_start_s * self.sr)), n))
        s1 = max(s0, min(int(round(window.window_end_s * self.sr)), n))
        audio, revision = self._mono_snapshot(s0, s1)
        w0, w1 = s0 / self.sr, s1 / self.sr
        t0 = min(max(window.emit_lo_s, w0), w1)
        t1 = min(max(window.emit_hi_s, t0), w1)
        return AnalysisRequest(
            audio=audio,
            sr=self.sr,
            t0_s=t0,
            span_s=t1 - t0,
            context_lead_s=t0 - w0,
            context_tail_s=w1 - t1,
            source_epoch=epoch,
            revision=int(revision),
            options=dict(options or {}),
        )

    def range_verdict(self, source_epoch: int, revision: int, end_s: float) -> str:
        """Whether a :meth:`snapshot_range` result taken at ``revision`` and
        reaching ``end_s`` may be published: :data:`VERDICT_STALE` after a
        swap, or when a write that began behind the frontier and before
        ``end_s`` landed after the snapshot (or may have: the record of
        such writes is bounded, and forgetting one newer than the snapshot
        counts against it); else :data:`VERDICT_CURRENT`. Appends at the
        frontier never make a range stale."""
        end = int(round(float(end_s) * self.sr))
        with self._lock:
            if source_epoch != self.epoch:
                return VERDICT_STALE
            if self._backwrite_floor > revision:
                return VERDICT_STALE
            for rev, start in self._backwrites:
                if rev > revision and start < end:
                    return VERDICT_STALE
            return VERDICT_CURRENT

    def _mono_snapshot(self, s0: int, s1: int) -> tuple:
        """``(mono float32 copy of samples [s0, s1), revision)``."""
        audio, revision = self.audio_eng.snapshot(s0, s1)
        if audio.ndim == 2:
            audio = audio.mean(axis=1) if audio.shape[1] > 1 else audio[:, 0]
        return np.ascontiguousarray(audio, dtype=np.float32), int(revision)

    def check_content(self, now: Optional[float] = None) -> int:
        """Resolve pending cells: fingerprint each one's current audio and
        compare it with the audio last analysed. A change of at least
        :data:`CONTENT_CHANGE` dirties the cell; a smaller one leaves it
        clean with its content revision, so its result stays current and
        can settle. Runs on the tap's worker, never on the write hook.
        Returns the number of cells found changed."""
        now = self._clock() if now is None else now
        with self._lock:
            epoch = self.epoch
            todo = [i for i, p in enumerate(self.pending) if p]
        changed = 0
        for i in todo:
            with self._lock:
                if self._closed or epoch != self.epoch or i >= self.n_cells or not self.pending[i]:
                    continue
                reference = self.fingerprint[i]
                s0, s1, _t0, _t1 = self._window_samples(i)
            audio, revision = self._mono_snapshot(s0, s1)
            delta = (
                math.inf if reference is None
                else content_change(content_fingerprint(audio, self.sr), reference)
            )
            with self._lock:
                if epoch != self.epoch or i >= self.n_cells or not self.pending[i]:
                    continue
                if delta >= CONTENT_CHANGE:
                    self._dirty_locked(i, now)
                    changed += 1
                elif self.cell_revision[i] <= revision:
                    self.pending[i] = False
                # Otherwise it was written again after this read; the next
                # pass checks the newer audio.
        return changed

    def result_verdict(self, i: int, source_epoch: int, revision: int) -> str:
        """Whether a result computed from the snapshot at ``revision`` may
        be published: :data:`VERDICT_CURRENT` when no swap and no
        content-changing write to cell ``i``'s input window happened since,
        :data:`VERDICT_STALE` when one did, :data:`VERDICT_UNCHECKED` when
        later writes landed that :meth:`check_content` has not compared
        yet."""
        with self._lock:
            if source_epoch != self.epoch or i < 0 or i >= self.n_cells:
                return VERDICT_STALE
            if self.content_revision[i] > revision:
                return VERDICT_STALE
            if self.pending[i] and self.cell_revision[i] > revision:
                return VERDICT_UNCHECKED
            return VERDICT_CURRENT

    def is_current(self, i: int, source_epoch: int, revision: int) -> bool:
        """True when :meth:`result_verdict` is :data:`VERDICT_CURRENT`."""
        return self.result_verdict(i, source_epoch, revision) == VERDICT_CURRENT

    def cell_state(self, i: int) -> tuple:
        """``(content_revision, dirty, pending)`` of cell ``i``."""
        with self._lock:
            return self.content_revision[i], self.dirty[i], self.pending[i]

    def mark_dirty(self, i: int) -> None:
        """Put cell ``i`` back on the to-do list (dropped or failed result).
        The re-analysis cap still applies from the last submission."""
        now = self._clock()
        with self._lock:
            if 0 <= i < self.n_cells:
                self._dirty_locked(i, now)

    def close(self) -> None:
        """Detach the write listener and the bus subscription. Idempotent."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
        try:
            self.audio_eng.remove_write_listener(self._on_write)
        except Exception as exc:  # noqa: BLE001 - teardown must not raise
            logger.warning("analysis_observer_detach_failed error={}", exc)
        if self._sub is not None and self._bus is not None:
            try:
                self._bus.unsubscribe(self._sub)
            except Exception as exc:  # noqa: BLE001
                logger.warning("analysis_observer_unsubscribe_failed error={}", exc)
