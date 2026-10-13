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
if both still match (:meth:`CanvasObserver.is_current`).
"""

from __future__ import annotations

import math
import threading
import time
from typing import Any, Callable, Mapping, Optional

import numpy as np
from loguru import logger

from acestep.analysis.engine import AnalysisRequest
from acestep.analysis.scheduler import SchedulerState, n_cells_for
from acestep.streaming.audio_engine import WRITE_SWAP
from acestep.streaming.events import AudioReady, SwapReady

#: Without a client playhead report for this long (and no local output
#: stream), the session counts as having no playhead.
PLAYHEAD_STALE_S = 2.0


class CanvasObserver:
    """Dirty-cell, frontier, epoch and playhead bookkeeping for one session.

    ``cell_s`` / ``context_s`` come from the analyzer's spec. ``on_tick``
    (``tick_ms, now``) and ``on_epoch`` (``source_epoch``) are called on the
    bus subscriber's thread; keep them cheap. :meth:`close` detaches the
    write listener and the bus subscription.
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
        self._clock = clock
        self._lock = threading.Lock()
        self._closed = False

        now = clock()
        self._resize(len(audio_eng.current), int(audio_eng.revision), now)
        self.frontier_samples: Optional[int] = None

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
        self.dirty = [True] * n
        self.dirty_since = [now] * n
        self.last_analysed = [-math.inf] * n

    @property
    def n_cells(self) -> int:
        return len(self.dirty)

    def cell_bounds_s(self, i: int) -> tuple:
        """``(start_s, end_s)`` of cell ``i`` in canvas seconds."""
        start = i * self.cell_s
        return start, min(start + self.cell_s, self.duration_s)

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
        with self._lock:
            if self._closed:
                return
            if kind == WRITE_SWAP:
                self._resize(end, revision, now)
                self.frontier_samples = None
                return
            for i in self._cells_touching(start, end):
                self.cell_revision[i] = max(self.cell_revision[i], revision)
                if not self.dirty[i]:
                    self.dirty[i] = True
                    self.dirty_since[i] = now
            self.frontier_samples = int(end)

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
            with self._lock:
                if epoch == self.epoch:
                    return
                self.epoch = epoch
                self._resize(len(self.audio_eng.current), int(self.audio_eng.revision), now)
                self.frontier_samples = None
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
            t0, t1 = self.cell_bounds_s(i)
            self.dirty[i] = False
            self.last_analysed[i] = now
        w0 = max(0.0, t0 - self.context_s)
        w1 = min(self.duration_s, t1 + self.context_s)
        s0 = int(round(w0 * self.sr))
        s1 = int(round(w1 * self.sr))
        audio, revision = self.audio_eng.snapshot(s0, s1)
        if audio.ndim == 2:
            audio = audio.mean(axis=1) if audio.shape[1] > 1 else audio[:, 0]
        audio = np.ascontiguousarray(audio, dtype=np.float32)
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

    def is_current(self, i: int, source_epoch: int, revision: int) -> bool:
        """True when no swap and no write to cell ``i``'s input window
        happened after the snapshot that produced ``revision``."""
        with self._lock:
            if source_epoch != self.epoch or i < 0 or i >= self.n_cells:
                return False
            return self.cell_revision[i] <= revision

    def cell_state(self, i: int) -> tuple:
        """``(revision, dirty)`` of cell ``i``."""
        with self._lock:
            return self.cell_revision[i], self.dirty[i]

    def mark_dirty(self, i: int) -> None:
        """Put cell ``i`` back on the to-do list (dropped or failed result).
        The re-analysis cap still applies from the last submission."""
        now = self._clock()
        with self._lock:
            if 0 <= i < self.n_cells and not self.dirty[i]:
                self.dirty[i] = True
                self.dirty_since[i] = now

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
