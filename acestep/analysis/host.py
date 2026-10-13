"""Process-wide analysis host: owns every loaded listening model.

One :class:`AnalysisHost` per process (:func:`get_host`). It implements
:class:`acestep.analysis.engine.AnalysisSubmitter` for session-side taps
and a blocking :meth:`AnalysisHost.run_one_shot` for interactive callers
(the plugin's ``midi_transcribe``). Shape, deliberately small:

- one worker thread, one inference in flight per process;
- one bounded live queue (:data:`LIVE_QUEUE_CAPACITY`); a full queue drops
  its OLDEST live request, whose ``on_error`` receives
  ``EngineBusy("dropped")``;
- one-shot requests jump the queue: the worker always takes a waiting
  one-shot before the next live request (the one in flight finishes);
- engines are built lazily per ``(analyzer, size)`` through
  ``AnalyzerSpec.make_engine`` under a load lock, after VRAM admission for
  GPU analyzers. A live tap and the one-shot asking for the same size
  share one instance.

Every callback (``on_result``, ``on_error``, including drop notices) runs
on the worker thread, never on the submitter's, so a tap may submit while
holding its own lock.

Third-party engines print progress to stderr on every call (MuScriptor
prints three lines); the host captures ``sys.stderr`` around each engine
call and logs it at DEBUG. The redirect is process-wide for the call's
duration, so other Python-level stderr writes in that window land in the
same DEBUG line rather than on the console.
"""

from __future__ import annotations

import contextlib
import io
import threading
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Optional

from loguru import logger

from acestep.analysis.engine import (
    AnalysisEngine,
    AnalysisRequest,
    AnalysisResult,
    EngineBusy,
    EngineUnavailable,
)

#: Live requests held at most; the oldest is dropped beyond this.
LIVE_QUEUE_CAPACITY = 8
#: Headroom (GiB) demanded on top of an engine's declared resident size.
VRAM_HEADROOM_GB = 0.5
#: Bound on joining the worker at shutdown.
_SHUTDOWN_JOIN_S = 5.0

SpecLookup = Callable[[str], Any]
VramProbe = Callable[[], Optional[Mapping[str, float]]]


def _default_lookup(name: str):
    from acestep.analysis.registry import get_analyzer

    return get_analyzer(name)


def _default_vram() -> Optional[Mapping[str, float]]:
    from acestep.gpu_config import get_vram_telemetry

    return get_vram_telemetry()


@dataclass
class _Job:
    analyzer: str
    size: Optional[str]
    request: AnalysisRequest
    on_result: Callable[[AnalysisResult], None]
    on_error: Callable[[BaseException], None]
    token: object = None
    one_shot: bool = False


@dataclass
class _OneShotSlot:
    done: threading.Event = field(default_factory=threading.Event)
    result: Optional[AnalysisResult] = None
    error: Optional[BaseException] = None


class AnalysisHost:
    """The process's listening models and the single queue in front of them.

    ``spec_lookup`` and ``vram_probe`` default to the registry and
    :func:`acestep.gpu_config.get_vram_telemetry`; tests inject fakes.
    """

    def __init__(
        self,
        spec_lookup: Optional[SpecLookup] = None,
        vram_probe: Optional[VramProbe] = None,
        live_capacity: int = LIVE_QUEUE_CAPACITY,
    ):
        self._lookup = spec_lookup or _default_lookup
        self._vram = vram_probe or _default_vram
        self._capacity = int(live_capacity)
        self._cond = threading.Condition()
        self._live: deque[_Job] = deque()
        self._one_shot: deque[_Job] = deque()
        # (job, error) pairs whose on_error is still owed; the worker
        # delivers them so no callback runs on a submitter's thread.
        self._owed: list[tuple[_Job, BaseException]] = []
        self._closed = False
        self._worker: Optional[threading.Thread] = None
        self._engines: dict[tuple[str, str], AnalysisEngine] = {}
        self._load_lock = threading.Lock()

    # ------------------------------------------------------------------
    # AnalysisSubmitter

    def submit(
        self,
        analyzer: str,
        size: Optional[str],
        request: AnalysisRequest,
        on_result: Callable[[AnalysisResult], None],
        on_error: Callable[[BaseException], None],
        cancel_token: object,
    ) -> None:
        """Queue ``request`` at live priority. Returns at once; the result
        or error arrives through the callbacks on the worker thread."""
        job = _Job(analyzer, size, request, on_result, on_error, cancel_token)
        with self._cond:
            if self._closed:
                self._owed.append((job, EngineUnavailable("analysis host is shut down")))
            else:
                while len(self._live) >= self._capacity:
                    self._owed.append((self._live.popleft(), EngineBusy("dropped")))
                self._live.append(job)
            self._ensure_worker_locked()
            self._cond.notify_all()

    def cancel(self, cancel_token: object) -> int:
        """Withdraw every queued live request carrying ``cancel_token``.
        No callback fires for withdrawn work. The request in flight (if
        any) is not interrupted and still delivers."""
        with self._cond:
            keep = [j for j in self._live if j.token is not cancel_token]
            withdrawn = len(self._live) - len(keep)
            if withdrawn:
                self._live = deque(keep)
            return withdrawn

    def engine_info(self, analyzer: str, size: Optional[str]) -> Mapping[str, Any]:
        """What ``analysis_state`` reports, without loading anything.
        Raises :class:`EngineUnavailable` (user-facing message) for an
        unknown analyzer, an unknown size or a failing probe."""
        spec, resolved = self._resolve(analyzer, size)
        reason = spec.probe()
        if reason:
            raise EngineUnavailable(reason)
        with self._load_lock:
            loaded = (spec.name, resolved) in self._engines
        return {
            "name": spec.name,
            "size": resolved,
            "loaded": loaded,
            "licence": spec.licence,
            "engine": spec.engine_id,
            "display_name": spec.display_name,
        }

    # ------------------------------------------------------------------
    # Interactive

    def run_one_shot(
        self,
        analyzer: str,
        size: Optional[str],
        request: AnalysisRequest,
        timeout: Optional[float] = None,
    ) -> AnalysisResult:
        """Run ``request`` ahead of all queued live work and block until it
        finishes. Raises whatever the engine raised, or
        :class:`EngineUnavailable` if the host shuts down first, or
        ``TimeoutError`` after ``timeout`` seconds (the request still runs)."""
        slot = _OneShotSlot()

        def _ok(result: AnalysisResult) -> None:
            slot.result = result
            slot.done.set()

        def _err(exc: BaseException) -> None:
            slot.error = exc
            slot.done.set()

        job = _Job(analyzer, size, request, _ok, _err, one_shot=True)
        with self._cond:
            if self._closed:
                raise EngineUnavailable("analysis host is shut down")
            self._one_shot.append(job)
            self._ensure_worker_locked()
            self._cond.notify_all()
        if not slot.done.wait(timeout):
            raise TimeoutError(f"{analyzer} one-shot did not finish within {timeout} s")
        if slot.error is not None:
            raise slot.error
        assert slot.result is not None
        return slot.result

    # ------------------------------------------------------------------
    # Lifetime

    def loaded(self) -> list[tuple[str, str]]:
        """``(analyzer, size)`` of every loaded engine."""
        with self._load_lock:
            return sorted(self._engines)

    def shutdown(self) -> int:
        """Stop the worker, fail queued work with :class:`EngineUnavailable`,
        unload every engine and run each spec's ``shutdown`` hook. Returns
        the number of engines unloaded. Safe to call twice."""
        with self._cond:
            self._closed = True
            pending = list(self._one_shot) + list(self._live)
            self._one_shot.clear()
            self._live.clear()
            owed, self._owed = self._owed, []
            worker = self._worker
            self._cond.notify_all()
        if worker is not None and worker is not threading.current_thread():
            worker.join(_SHUTDOWN_JOIN_S)
            if worker.is_alive():
                logger.warning("analysis_host_shutdown worker still busy after {} s", _SHUTDOWN_JOIN_S)
        gone = EngineUnavailable("analysis host is shut down")
        for job in pending:
            self._call(job.on_error, gone)
        for job, exc in owed:
            self._call(job.on_error, exc)

        with self._load_lock:
            engines, self._engines = self._engines, {}
        names = set()
        for (name, size), engine in engines.items():
            names.add(name)
            try:
                engine.unload()
            except Exception as exc:  # noqa: BLE001 - shutdown is best effort
                logger.warning("analysis_engine_unload_failed analyzer={} size={} error={}", name, size, exc)
        for name in sorted(names):
            try:
                spec = self._lookup(name)
            except Exception:  # noqa: BLE001
                continue
            if spec.shutdown is not None:
                try:
                    spec.shutdown()
                except Exception as exc:  # noqa: BLE001
                    logger.warning("analysis_spec_shutdown_failed analyzer={} error={}", name, exc)
        if engines:
            logger.info("analysis_host_shutdown unloaded={}", len(engines))
        return len(engines)

    # ------------------------------------------------------------------
    # Worker

    def _ensure_worker_locked(self) -> None:
        if self._worker is None or not self._worker.is_alive():
            self._worker = threading.Thread(
                target=self._run, name="analysis-host", daemon=True,
            )
            self._worker.start()

    def _run(self) -> None:
        while True:
            with self._cond:
                while not (self._closed or self._owed or self._one_shot or self._live):
                    self._cond.wait()
                owed, self._owed = self._owed, []
                if self._closed:
                    job = None
                elif self._one_shot:
                    job = self._one_shot.popleft()
                elif self._live:
                    job = self._live.popleft()
                else:
                    job = None
            for owed_job, exc in owed:
                self._call(owed_job.on_error, exc)
            if job is None:
                if self._closed:
                    return
                continue
            self._execute(job)

    def _execute(self, job: _Job) -> None:
        try:
            spec, size = self._resolve(job.analyzer, job.size)
            engine = self._engine_for(spec, size)
            buf = io.StringIO()
            try:
                with contextlib.redirect_stderr(buf):
                    result = engine.analyze(job.request)
            finally:
                captured = buf.getvalue().strip()
                if captured:
                    logger.debug(
                        "analysis_engine_stderr analyzer={} size={} text={!r}",
                        spec.name, size, captured,
                    )
        except BaseException as exc:  # noqa: BLE001 - every failure goes to the caller
            if not isinstance(exc, (EngineBusy, EngineUnavailable)):
                logger.opt(exception=exc).warning(
                    "analysis_failed analyzer={} one_shot={} error={}",
                    job.analyzer, job.one_shot, exc,
                )
            self._call(job.on_error, exc)
            return
        self._call(job.on_result, result)

    @staticmethod
    def _call(fn: Callable[[Any], None], arg: Any) -> None:
        try:
            fn(arg)
        except Exception as exc:  # noqa: BLE001 - a callback must not kill the worker
            logger.opt(exception=exc).warning("analysis_callback_failed error={}", exc)

    # ------------------------------------------------------------------
    # Engines

    def _resolve(self, analyzer: str, size: Optional[str]):
        try:
            spec = self._lookup(analyzer)
        except KeyError as exc:
            raise EngineUnavailable(str(exc.args[0]) if exc.args else str(exc)) from None
        try:
            resolved = spec.resolve_size(size)
        except ValueError as exc:
            raise EngineUnavailable(str(exc)) from None
        return spec, resolved

    def _engine_for(self, spec, size: str) -> AnalysisEngine:
        key = (spec.name, size)
        with self._load_lock:
            engine = self._engines.get(key)
            if engine is not None:
                return engine
            reason = spec.probe()
            if reason:
                raise EngineUnavailable(reason)
            device = "cuda" if spec.placement == "gpu" else "cpu"
            if spec.placement == "gpu" and size in spec.resident_gb:
                telemetry = self._vram()
                need = float(spec.resident_gb[size]) + VRAM_HEADROOM_GB
                if telemetry is not None and float(telemetry.get("available_gb", 0.0)) < need:
                    raise EngineUnavailable(
                        f"not enough free GPU memory for {spec.display_name} {size}"
                    )
            buf = io.StringIO()
            try:
                with contextlib.redirect_stderr(buf):
                    engine = spec.make_engine(size, device)
            except EngineUnavailable:
                raise
            except Exception as exc:  # noqa: BLE001 - surfaced user-facing
                logger.opt(exception=exc).warning(
                    "analysis_engine_load_failed analyzer={} size={} error={}",
                    spec.name, size, exc,
                )
                raise EngineUnavailable(
                    f"{spec.display_name} {size} failed to load: {type(exc).__name__}: {exc}"
                ) from exc
            finally:
                captured = buf.getvalue().strip()
                if captured:
                    logger.debug("analysis_engine_load_stderr analyzer={} text={!r}", spec.name, captured)
            self._engines[key] = engine
            logger.info("analysis_engine_loaded analyzer={} size={} device={}", spec.name, size, device)
            return engine


_HOST: Optional[AnalysisHost] = None
_HOST_LOCK = threading.Lock()


def get_host() -> AnalysisHost:
    """The process-wide host, created on first use (and again after
    :func:`shutdown_host`)."""
    global _HOST
    with _HOST_LOCK:
        if _HOST is None:
            _HOST = AnalysisHost()
        return _HOST


def shutdown_host() -> int:
    """Shut the process-wide host down if one was ever created; returns
    the number of engines unloaded (0 when none). For the server's exit
    path, next to the families' ``shutdown`` hooks."""
    global _HOST
    with _HOST_LOCK:
        host, _HOST = _HOST, None
    return host.shutdown() if host is not None else 0
