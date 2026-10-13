"""One live analysis channel: (session, analyzer) -> ``analysis_*`` events.

An :class:`AnalysisTap` owns a :class:`~acestep.analysis.observer.
CanvasObserver` (one per tap in v1; two analyzers on one session each
keep their own, which costs one write listener and one bus subscription
apiece), a :class:`~acestep.analysis.scheduler.Governor`, and a 100 ms
worker loop. Each wake-up it asks the scheduler for the next cell,
snapshots it under the audio engine's lock and submits it to the
process-wide host (:class:`~acestep.analysis.engine.AnalysisSubmitter`)
with this tap's cancel token. One request per tap is in flight at a time.

Results arrive on the host's worker thread. A result is published as an
:class:`~acestep.streaming.events.AnalysisUpdate` only when its
``(source_epoch, revision)`` still matches the cell; otherwise it counts
as ``dropped`` and the cell goes back on the to-do list. When writes
landed on the cell after the snapshot and their content is not compared
yet, the result is held until the worker's next pass has run the
observer's content check (a ring rewrite with the same music keeps the
result). On a ring canvas a provisional cell is re-published as
``settled`` (same items, no new analysis) once the playhead passes it
with its content unchanged. A truncated result (the engine abandoned a
runaway window) counts as ``dropped``, still replaces its range with no
items so nothing stale survives, and never settles.

State goes out as :class:`~acestep.streaming.events.AnalysisState` on
enable (followed by a ``clear`` update), on mode change, on error and on
disable. A swap (new ``source_epoch``) withdraws queued work and sends a
``clear`` with the new epoch.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Optional

from loguru import logger

from acestep.analysis.engine import (
    AnalysisResult,
    AnalysisSubmitter,
    EngineBusy,
    EngineUnavailable,
)
from acestep.analysis.observer import (
    VERDICT_CURRENT,
    VERDICT_UNCHECKED,
    CanvasObserver,
)
from acestep.analysis.registry import AnalyzerSpec
from acestep.analysis.scheduler import (
    BASIS_PROVISIONAL,
    BASIS_SETTLED,
    Governor,
    basis_for,
    crossed,
    is_late,
    next_cell,
)
from acestep.streaming.events import AnalysisState, AnalysisUpdate

#: Worker loop period (seconds).
TAP_INTERVAL_S = 0.1
#: Default lookahead horizon when the analyzer declares no ``horizon_s``.
DEFAULT_HORIZON_S = 8.0

MODE_LOOKAHEAD = "lookahead"
MODE_CONFIRM = "confirm"
MODE_NO_PLAYHEAD = "no_playhead"
MODE_BACKOFF = "backoff"
MODE_OFF = "off"


@dataclass
class _Published:
    """What the tap last published for one cell."""

    revision: int
    range_start_s: float
    span_s: float
    kind: str
    items: tuple
    basis: str
    truncated: bool = False
    crossed: bool = False


class AnalysisTap:
    """A live analysis channel for one analyzer on one session.

    ``options`` must already be validated by ``spec.validate_options``.
    ``next_seq`` returns the session-wide monotonic update sequence.
    ``refines_audio`` picks the canvas kind (ring vs append-only).
    """

    def __init__(
        self,
        *,
        spec: AnalyzerSpec,
        options: Mapping[str, Any],
        submitter: AnalysisSubmitter,
        audio_eng,
        bus,
        refines_audio: bool,
        source_epoch: int,
        next_seq: Callable[[], int],
        clock: Callable[[], float] = time.monotonic,
        interval_s: float = TAP_INTERVAL_S,
    ):
        self.spec = spec
        self.name = spec.name
        self.options = dict(options)
        self.size = spec.resolve_size(self.options.get("size"))
        horizon = self.options.get("horizon_s")
        self.horizon_s = float(horizon) if horizon is not None else DEFAULT_HORIZON_S
        self.submitter = submitter
        self.audio_eng = audio_eng
        self.bus = bus
        self.refines_audio = bool(refines_audio)
        self._initial_epoch = int(source_epoch)
        self._next_seq = next_seq
        self._clock = clock
        self._interval_s = float(interval_s)

        self.engine_name = ""
        self.licence = spec.licence
        self.dropped = 0
        self.error: Optional[str] = None
        self.enabled = False
        self.mode = MODE_OFF

        self._lock = threading.RLock()
        self._token: Optional[object] = None
        self._in_flight: Optional[int] = None
        self._published: dict[int, _Published] = {}
        # Results waiting for the content check of later writes to the cell.
        self._held: dict[int, AnalysisResult] = {}
        self._prev_playhead: Optional[float] = None
        self._governor = Governor(self.refines_audio)
        self._observer: Optional[CanvasObserver] = None
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    # ---- lifecycle ----------------------------------------------------

    def start(self) -> bool:
        """Probe the engine, attach the observer, publish the enable state
        and the opening ``clear``, start the worker. Returns False (after
        publishing ``analysis_state`` with the error) when the engine
        cannot run here."""
        try:
            info = self.submitter.engine_info(self.name, self.size)
        except EngineUnavailable as exc:
            self._fail(str(exc))
            return False
        except Exception as exc:  # noqa: BLE001 - surfaced to the client
            self._fail(f"{type(exc).__name__}: {exc}")
            return False
        self.engine_name = str(info.get("name") or self.name)
        self.licence = str(info.get("licence") or self.spec.licence or "")
        self.size = str(info.get("size") or self.size)

        with self._lock:
            self._token = object()
            self.enabled = True
            self._observer = CanvasObserver(
                self.audio_eng,
                self.bus,
                self.spec.window_s,
                self.spec.context_s,
                source_epoch=self._initial_epoch,
                on_tick=self._on_tick,
                on_epoch=self._on_epoch,
                clock=self._clock,
            )
            self.mode = self._compute_mode(self._clock())
            self._publish_state()
            self._publish_clear(self._observer.epoch)
        self._thread = threading.Thread(
            target=self._run, name=f"analysis-tap-{self.name}", daemon=True,
        )
        self._thread.start()
        logger.info(
            "analysis_tap_enabled tap={} engine={} size={} refines_audio={} horizon_s={}",
            self.name, self.engine_name, self.size, self.refines_audio, self.horizon_s,
        )
        return True

    def stop(self, *, error: Optional[str] = None, publish: bool = True) -> None:
        """Disable: withdraw queued work, detach, publish the ``off`` state.
        Never joins the worker (safe from any thread, including the host
        worker's result callback). Idempotent."""
        with self._lock:
            if not self.enabled:
                return
            self.enabled = False
            self._stop.set()
            token, self._token = self._token, None
            self._in_flight = None
            self._held.clear()
            if error is not None:
                self.error = error
            self.mode = MODE_OFF
        if token is not None:
            self.dropped += self._cancel(token)
        obs = self._observer
        if obs is not None:
            obs.close()
        if publish:
            self._publish_state()
        logger.info("analysis_tap_disabled tap={} dropped={} error={}", self.name, self.dropped, error)

    def close(self, timeout: float = 1.0) -> None:
        """:meth:`stop`, then join the worker for at most ``timeout`` s."""
        self.stop()
        t = self._thread
        if t is not None and t is not threading.current_thread():
            t.join(timeout)

    def _fail(self, message: str) -> None:
        self.error = message
        self.enabled = False
        self.mode = MODE_OFF
        self._publish_state()
        logger.warning("analysis_tap_failed tap={} error={}", self.name, message)

    def _cancel(self, token: object) -> int:
        try:
            return int(self.submitter.cancel(token) or 0)
        except Exception as exc:  # noqa: BLE001 - teardown must not raise
            logger.warning("analysis_tap_cancel_failed tap={} error={}", self.name, exc)
            return 0

    # ---- publishing ---------------------------------------------------

    def _publish_state(self) -> None:
        self.bus.publish(AnalysisState(
            tap=self.name,
            enabled=self.enabled,
            engine=self.engine_name,
            size=self.size,
            licence=self.licence,
            mode=self.mode,
            error=self.error,
            dropped=self.dropped,
        ))

    def _publish_clear(self, epoch: int) -> None:
        self.bus.publish(AnalysisUpdate(
            tap=self.name, seq=self._next_seq(), source_epoch=epoch,
            range_start_s=0.0, span_s=0.0, basis=BASIS_PROVISIONAL,
            late=False, clear=True, kind=self.spec.item_kind, items=(),
            playhead_s=None, wall_s=0.0,
        ))

    # ---- bus callbacks (observer thread) ------------------------------

    def _on_tick(self, tick_ms: float, now: float) -> None:
        with self._lock:
            self._governor.observe_tick(tick_ms, now)

    def _on_epoch(self, epoch: int) -> None:
        with self._lock:
            if not self.enabled:
                return
            old, self._token = self._token, object()
            self._in_flight = None
            self._published.clear()
            self._held.clear()
            self._prev_playhead = None
        if old is not None:
            self.dropped += self._cancel(old)
        self._publish_clear(epoch)

    # ---- worker -------------------------------------------------------

    def _run(self) -> None:
        while not self._stop.wait(self._interval_s):
            try:
                self.step()
            except Exception as exc:  # noqa: BLE001 - keep the loop alive
                logger.opt(exception=True).error("analysis_tap_step_failed tap={} error={}", self.name, exc)

    def _compute_mode(self, now: float, playhead: Optional[float] = None) -> str:
        if not self.enabled:
            return MODE_OFF
        if self._governor.mode(now) == "backoff":
            return MODE_BACKOFF
        if not self.refines_audio:
            return MODE_CONFIRM
        obs = self._observer
        if obs is None or not obs.has_playhead(now):
            return MODE_NO_PLAYHEAD
        return MODE_LOOKAHEAD

    def _period_for(self, start_s: float, end_s: float) -> tuple:
        """``(period_len, end_in_period)`` for play-order maths: the loop
        band when the range overlaps it, else the canvas."""
        obs = self._observer
        band = obs.loop_band() if obs is not None else None
        if band is not None and start_s < band[1] and end_s > band[0]:
            return band[1] - band[0], min(end_s, band[1])
        return (obs.duration_s if obs is not None else 0.0), end_s

    def step(self) -> None:
        """One scheduling pass. Public so tests and the bench drive it
        without the thread."""
        with self._lock:
            if not self.enabled:
                return
            obs = self._observer
        # Content check of rewritten cells: audio reads and FFTs, outside
        # the tap lock so host callbacks never wait on it.
        obs.check_content()
        with self._lock:
            if not self.enabled or obs is not self._observer:
                return
            now = self._clock()
            playhead = obs.playhead_s(now)
            eng = self.audio_eng
            self._governor.observe_slice_lead(
                getattr(eng, "observed_slice_lead_s", None),
                float(getattr(eng, "observed_slice_lead_wall_s", 0.0) or 0.0),
                now,
            )
            frontier = obs.frontier_s()
            if not self.refines_audio and frontier is not None and playhead is not None:
                self._governor.observe_frontier_lead(frontier - playhead, now)

            self._resolve_held(playhead)
            self._settle_passed(playhead)
            self._prev_playhead = playhead

            mode = self._compute_mode(now, playhead)
            if mode != self.mode:
                self.mode = mode
                self._publish_state()

            if self._in_flight is not None or not self._governor.may_submit(now):
                return
            state = obs.scheduler_state(
                now=now, refines_audio=self.refines_audio, horizon_s=self.horizon_s,
            )
            i = next_cell(state)
            if i is None:
                return
            request = obs.snapshot_cell(i, self.options, now=now)
            token = self._token
            self._in_flight = i
            self._governor.note_submit(now)
        try:
            self.submitter.submit(
                self.name, self.size, request,
                lambda result, i=i, token=token: self._on_result(i, token, result),
                lambda exc, i=i, token=token: self._on_error(i, token, exc),
                token,
            )
        except Exception as exc:  # noqa: BLE001 - treat like an engine error
            self._on_error(i, token, exc)

    def _settle_passed(self, playhead: Optional[float]) -> None:
        """Re-publish provisional ring cells as settled once the playhead
        has passed them with their revision unchanged. Caller holds the
        lock."""
        if not self.refines_audio or playhead is None:
            return
        obs = self._observer
        for i, rec in list(self._published.items()):
            if rec.basis != BASIS_PROVISIONAL or rec.truncated:
                continue
            if not rec.crossed:
                period, end = self._period_for(rec.range_start_s, rec.range_start_s + rec.span_s)
                rec.crossed = crossed(self._prev_playhead, playhead, end, period)
                if not rec.crossed:
                    continue
            revision, dirty, pending = obs.cell_state(i)
            if pending:
                continue  # decided after the next content check
            rec.crossed = False
            basis = basis_for(
                refines_audio=True, playhead_passed=True,
                revision_unchanged=(revision <= rec.revision and not dirty),
            )
            if basis != BASIS_SETTLED:
                continue
            rec.basis = BASIS_SETTLED
            self.bus.publish(AnalysisUpdate(
                tap=self.name, seq=self._next_seq(), source_epoch=obs.epoch,
                range_start_s=rec.range_start_s, span_s=rec.span_s,
                basis=BASIS_SETTLED, late=False, clear=False, kind=rec.kind,
                items=rec.items, playhead_s=playhead, wall_s=0.0,
            ))

    # ---- host callbacks (host worker thread) --------------------------

    def _on_result(self, i: int, token: object, result: AnalysisResult) -> None:
        with self._lock:
            if token is not self._token or not self.enabled:
                return
            self._in_flight = None
            verdict = self._observer.result_verdict(i, result.source_epoch, result.revision)
            if verdict == VERDICT_UNCHECKED:
                self._held[i] = result
                return
            self._finish(i, result, verdict, self._observer.playhead_s(self._clock()))

    def _resolve_held(self, playhead: Optional[float]) -> None:
        """Publish or drop held results now that the content check ran.
        Caller holds the lock."""
        obs = self._observer
        for i, result in list(self._held.items()):
            verdict = obs.result_verdict(i, result.source_epoch, result.revision)
            if verdict == VERDICT_UNCHECKED:
                continue
            del self._held[i]
            self._finish(i, result, verdict, playhead)

    def _finish(self, i: int, result: AnalysisResult, verdict: str,
                playhead: Optional[float]) -> None:
        """Publish a current result, or count a stale one as dropped and
        re-dirty its cell. Caller holds the lock."""
        obs = self._observer
        if verdict != VERDICT_CURRENT:
            self.dropped += 1
            obs.mark_dirty(i)
            return
        truncated = bool(result.truncated)
        if truncated:
            # The range is still replaced (empty), so a runaway window
            # never leaves the previous analysis on screen.
            self.dropped += 1
            logger.warning(
                "analysis_window_truncated tap={} cell={} wall_s={:.2f}",
                self.name, i, result.wall_s,
            )
        period, _ = self._period_for(result.range_start_s, result.range_start_s + result.span_s)
        basis = BASIS_PROVISIONAL if truncated else basis_for(
            refines_audio=self.refines_audio, playhead_passed=False,
            revision_unchanged=True,
        )
        items = () if truncated else tuple(dict(it) for it in result.items)
        self._published[i] = _Published(
            revision=result.revision, range_start_s=float(result.range_start_s),
            span_s=float(result.span_s), kind=result.kind, items=items, basis=basis,
            truncated=truncated,
        )
        self.bus.publish(AnalysisUpdate(
            tap=self.name, seq=self._next_seq(), source_epoch=result.source_epoch,
            range_start_s=float(result.range_start_s), span_s=float(result.span_s),
            basis=basis,
            late=is_late(
                result.range_start_s, result.span_s, playhead,
                refines_audio=self.refines_audio, period_len=period,
            ),
            clear=False, kind=result.kind, items=items,
            playhead_s=playhead, wall_s=float(result.wall_s),
        ))
        if result.wall_s > result.span_s > 0:
            logger.warning(
                "analysis_slower_than_realtime tap={} cell={} wall_s={:.2f} span_s={:.2f}",
                self.name, i, result.wall_s, result.span_s,
            )
        if self.error is not None:
            self.error = None
            self._publish_state()

    def _on_error(self, i: int, token: object, exc: BaseException) -> None:
        with self._lock:
            if token is not self._token or not self.enabled:
                return
            self._in_flight = None
            obs = self._observer
            if isinstance(exc, EngineBusy):
                self.dropped += 1
                obs.mark_dirty(i)
                return
            if isinstance(exc, EngineUnavailable):
                message = str(exc)
            else:
                message = f"{type(exc).__name__}: {exc}"
                logger.warning("analysis_tap_engine_error tap={} cell={} error={}", self.name, i, message)
                obs.mark_dirty(i)
                if message != self.error:
                    self.error = message
                    self._publish_state()
                return
        self.stop(error=message)
