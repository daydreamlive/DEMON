"""Local session log: a JSONL event stream mirroring the ledger schema.

One file per streaming session under
``<provenance dir>/sessions/<session_id>.jsonl``. Every line is one
timestamped event::

    {"schema": 1, "ts": "...Z", "session_id": "...", "event": "...", ...}

The event vocabulary follows the Session Record design (spec 02 §5):
model/LoRA identifiers at session start, prompt and parameter changes,
seed-audio fingerprints, a rolling hash chain over generated output
slices (checkpointed, not per-slice — per-slice lines at 20-50/s would
bloat the log for no evidentiary gain), and user-action records
(action type, payload summary, wall-clock ts). A future "upload my
local history" feature can replay these files into the cloud ledger
unchanged, which is the whole point of sharing the schema (spec 02 §7).

Wiring: :func:`attach_session` subscribes a
:class:`SessionLogTap` to the session's typed event bus
(:mod:`acestep.streaming.events`). The session registry
(:mod:`acestep.streaming.registry`) calls attach/detach when a handle
carrying a ``bus`` is registered/unregistered, so transport adapters
only have to hand the bus over. Torch-free, like the registry.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np
from loguru import logger

from acestep.provenance import session_logs_dir
from acestep.provenance.ledger_client import LedgerClient
from acestep.streaming.events import (
    AudioReady,
    AudioWriteFailed,
    AudioWritten,
    CommandFailed,
    DepthApplied,
    LoraCatalogUpdate,
    ParamsEcho,
    ParamsUpdate,
    PromptApplied,
    PromptBlendEcho,
    SessionError,
    SessionReady,
    StemAssets,
    StemFailed,
    StructureCleared,
    StructureFailed,
    StructureSet,
    SubscriberDropped,
    SwapFailed,
    SwapReady,
    TimbreCleared,
    TimbreFailed,
    TimbreSet,
)

__all__ = [
    "SESSION_LOG_SCHEMA_VERSION",
    "SessionLogWriter",
    "SessionLogTap",
    "attach_session",
    "detach_session",
    "get_tap",
    "record_user_action",
    "latest_session_summary",
]

SESSION_LOG_SCHEMA_VERSION = 1

# Chain checkpoints land every N hashed slices (plus one at session
# end), mirroring the ledger's periodic checkpoint seals.
_CHAIN_CHECKPOINT_EVERY = 256

# Params messages arrive at ~125 Hz; user-action records for them are
# diffed against the last logged values and rate-limited.
_PARAMS_ACTION_MIN_INTERVAL_S = 1.0

# Telemetry fields of the wire "params" message that are not user
# actions (playhead/flow-control reporting).
_PARAMS_TELEMETRY_KEYS = frozenset(
    {"type", "playback_pos", "client_time", "slice_lead_s", "slice_bytes_rx"},
)

_MAX_SUMMARY_STR = 300
_MAX_SUMMARY_LIST = 24


def _utc_ts() -> str:
    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )


def buffer_fingerprint(arr: np.ndarray) -> str:
    """Numpy analogue of :func:`acestep.track_assets.waveform_fingerprint`
    for ``[N, C]`` / ``[N]`` float buffers carried on bus events: mono
    mix, fixed 4096-point decimation grid, quantize, sha256. Same
    robustness intent (stable across benign decode round-trips); kept
    torch-free because the tap runs on pods and local CPUs alike.
    """
    a = np.asarray(arr, dtype=np.float32)
    mono = a.mean(axis=1) if a.ndim == 2 else a.reshape(-1)
    n = int(mono.shape[-1])
    if n == 0:
        return "empty"
    grid = 4096
    if n > grid:
        idx = np.round(np.linspace(0, n - 1, grid)).astype(np.int64)
        mono = mono[idx]
    quantized = np.round(np.ascontiguousarray(mono) * 10_000.0).astype(np.int32)
    return hashlib.sha256(quantized.tobytes()).hexdigest()


def _summarize(value: Any) -> Any:
    """Shrink a payload for a user-action record: bounded strings and
    lists, no binary. Prompt text stays intact up to the cap — the log
    is local-only, so prompt content is allowed here (spec 02 §5) even
    though only counts ever enter a manifest."""
    if isinstance(value, (bytes, bytearray, memoryview)):
        return {"bytes": len(value)}
    if isinstance(value, str):
        return value if len(value) <= _MAX_SUMMARY_STR else value[:_MAX_SUMMARY_STR] + "…"
    if isinstance(value, dict):
        return {str(k): _summarize(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        if len(value) > _MAX_SUMMARY_LIST:
            return {"items": len(value)}
        return [_summarize(v) for v in value]
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    return repr(value)[:_MAX_SUMMARY_STR]


class SessionLogWriter:
    """Append-only JSONL writer. Thread-safe; one flush per event so a
    crash loses at most the in-flight line. Never raises out of
    :meth:`record` — a broken log must not take down the session."""

    def __init__(self, path: Path, session_id: str) -> None:
        self.path = path
        self.session_id = session_id
        self._lock = threading.Lock()
        self._failed = False
        path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(path, "a", encoding="utf-8")

    def record(self, event: str, **fields: Any) -> None:
        line = {
            "schema": SESSION_LOG_SCHEMA_VERSION,
            "ts": _utc_ts(),
            "session_id": self.session_id,
            "event": event,
            **fields,
        }
        try:
            payload = json.dumps(line, default=str, separators=(",", ":"))
            with self._lock:
                if self._fh.closed:
                    return
                self._fh.write(payload + "\n")
                self._fh.flush()
        except Exception as exc:  # noqa: BLE001
            if not self._failed:
                self._failed = True
                logger.warning(
                    "session log write failed path={} error={}",
                    self.path, exc,
                )

    def file_sha256(self) -> str | None:
        try:
            with self._lock:
                if not self._fh.closed:
                    self._fh.flush()
            return hashlib.sha256(self.path.read_bytes()).hexdigest()
        except Exception:  # noqa: BLE001
            return None

    def close(self) -> None:
        with self._lock:
            try:
                self._fh.close()
            except Exception:  # noqa: BLE001
                pass


class SessionLogTap:
    """Event-bus subscriber that serializes session events to the local
    log (and forwards them to the ledger client when one is configured).

    Owns the output-slice rolling hash chain: ``head = sha256(head_hex
    || slice_bytes)``, checkpointed every ``_CHAIN_CHECKPOINT_EVERY``
    slices and sealed into the ``session_end`` record. Runs entirely on
    the subscription's drainer thread plus whichever thread calls
    :meth:`record_user_action`; shared counters sit behind one lock.
    """

    def __init__(
        self,
        bus: Any,
        session_id: str,
        *,
        meta: dict | None = None,
        snapshot: Any = None,
        log_dir: Path | None = None,
    ) -> None:
        self.session_id = session_id
        d = Path(log_dir) if log_dir is not None else session_logs_dir()
        self.writer = SessionLogWriter(d / f"{session_id}.jsonl", session_id)
        self.ledger = LedgerClient(session_id=session_id)

        self._lock = threading.Lock()
        self._started_wall = time.time()
        self._chain_head = hashlib.sha256(b"").hexdigest()
        self._slices = 0
        self._counts = {
            "events": 0,
            "prompt_changes": 0,
            "param_changes": 0,
            "user_actions": 0,
        }
        self._prompts_seen: set[str] = set()
        self._last_params: dict = {}
        self._last_params_wall = 0.0
        self._closed = False

        snap: dict = {}
        if callable(snapshot):
            try:
                snap = snapshot() or {}
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "session snapshot failed for provenance log: {}", exc,
                )
        meta = dict(meta or {})
        loras = [
            e.get("id")
            for e in snap.get("lora_catalog") or []
            if isinstance(e, dict) and e.get("state") == "enabled"
        ]
        self._model = meta.get("checkpoint") or snap.get("checkpoint")
        self._loras = loras
        self._record(
            "session_start",
            model=self._model,
            loras=loras,
            fixture_name=meta.get("fixture_name") or snap.get("fixture_name"),
            prompt=snap.get("prompt"),
            bpm=snap.get("bpm"),
            key=snap.get("key"),
            time_signature=snap.get("time_signature"),
            extra={
                k: v for k, v in meta.items()
                if k not in ("checkpoint", "fixture_name")
            },
        )
        if isinstance(snap.get("prompt"), str) and snap["prompt"]:
            self._prompts_seen.add(snap["prompt"])

        self._sub = bus.subscribe(self._on_event, name="provenance")
        self._bus = bus

    # ---- recording -------------------------------------------------------

    def _record(self, event: str, **fields: Any) -> None:
        with self._lock:
            self._counts["events"] += 1
        self.writer.record(event, **fields)
        self.ledger.post_event({"event": event, **fields})

    def record_user_action(
        self, action: str, payload: dict, *, source: str = "ws",
    ) -> None:
        """One authorship-action record: action type, payload summary,
        wall-clock ts (spec 02 §5 timeline). ``params`` actions are
        diffed against the last logged knob values and rate-limited so
        the ~125 Hz knob channel doesn't flood the log."""
        if self._closed:
            return
        if action == "params":
            values = {
                k: v for k, v in payload.items()
                if k not in _PARAMS_TELEMETRY_KEYS
            }
            with self._lock:
                changed = {
                    k: v for k, v in values.items()
                    if self._last_params.get(k) != v
                }
                now = time.monotonic()
                if not changed or (
                    now - self._last_params_wall < _PARAMS_ACTION_MIN_INTERVAL_S
                ):
                    return
                self._last_params.update(values)
                self._last_params_wall = now
                self._counts["user_actions"] += 1
                self._counts["param_changes"] += 1
            self._record(
                "user_action", action="params", source=source,
                summary=_summarize(changed),
            )
            return
        with self._lock:
            self._counts["user_actions"] += 1
            if action == "prompt":
                self._counts["prompt_changes"] += 1
                tags = payload.get("tags")
                if isinstance(tags, str) and tags:
                    self._prompts_seen.add(tags)
        summary = _summarize({
            k: v for k, v in payload.items() if k != "type"
        })
        self._record(
            "user_action", action=action, source=source, summary=summary,
        )

    # ---- bus tap ---------------------------------------------------------

    def _on_event(self, event: Any) -> None:
        try:
            self._dispatch_event(event)
        except Exception as exc:  # noqa: BLE001 — tap must never wedge the drainer
            logger.warning("provenance tap event failed: {}", exc)

    def _dispatch_event(self, event: Any) -> None:
        if isinstance(event, AudioReady):
            self._on_slice(event)
        elif isinstance(event, ParamsUpdate):
            # Per-slice telemetry snapshot; authorship-irrelevant.
            return
        elif isinstance(event, PromptApplied):
            with self._lock:
                self._counts["prompt_changes"] += 1
                if event.tags:
                    self._prompts_seen.add(event.tags)
            self._record("prompt_change", tags=_summarize(event.tags))
        elif isinstance(event, PromptBlendEcho):
            self._record("prompt_blend", value=event.value)
        elif isinstance(event, ParamsEcho):
            with self._lock:
                self._counts["param_changes"] += 1
            self._record("params_external", raw=_summarize(event.raw))
        elif isinstance(event, DepthApplied):
            self._record("depth_change", value=event.value)
        elif isinstance(event, LoraCatalogUpdate):
            loras = [
                e.get("id")
                for e in event.catalog or []
                if isinstance(e, dict) and e.get("state") == "enabled"
            ]
            with self._lock:
                self._loras = loras
            self._record("lora_catalog", loras=loras)
        elif isinstance(event, SessionReady):
            self._record(
                "session_ready",
                duration=event.duration,
                sample_rate=event.sample_rate,
                bpm=event.bpm,
                key=event.key,
                time_signature=event.time_signature,
                pipeline_depth=event.pipeline_depth,
                source_sha256=buffer_fingerprint(event.initial_buffer),
            )
        elif isinstance(event, SwapReady):
            self._record(
                "source_swap",
                fixture_name=event.fixture_name,
                duration=event.duration,
                bpm=event.bpm,
                key=event.key,
                source_epoch=event.source_epoch,
                seed_sha256=buffer_fingerprint(event.initial_buffer),
            )
        elif isinstance(event, TimbreSet):
            self._record("timbre_set", name=event.name, duration=event.duration)
        elif isinstance(event, TimbreCleared):
            self._record("timbre_cleared")
        elif isinstance(event, StructureSet):
            self._record("structure_set", name=event.name, duration=event.duration)
        elif isinstance(event, StructureCleared):
            self._record("structure_cleared")
        elif isinstance(event, AudioWritten):
            self._record(
                "audio_written",
                start_s=event.start_s,
                end_s=event.end_s,
                source_epoch=event.source_epoch,
            )
        elif isinstance(event, StemAssets):
            self._record(
                "stem_assets",
                fixture_name=event.fixture_name,
                source_mode=event.source_mode,
                frames=event.frames,
            )
        elif isinstance(event, (
            CommandFailed, SwapFailed, StemFailed, TimbreFailed,
            StructureFailed, AudioWriteFailed, SessionError,
        )):
            self._record(
                "error",
                kind=type(event).__name__,
                error=_summarize(getattr(event, "error", None)
                                 or getattr(event, "message", "")),
            )
        elif isinstance(event, SubscriberDropped):
            # Our own queue overflowed: the chain has a gap from here on.
            self._record("log_tap_dropped", reason=event.reason)

    def _on_slice(self, event: AudioReady) -> None:
        data = np.ascontiguousarray(event.audio).tobytes()
        with self._lock:
            self._chain_head = hashlib.sha256(
                self._chain_head.encode("ascii") + data,
            ).hexdigest()
            self._slices += 1
            head, n = self._chain_head, self._slices
            checkpoint = n % _CHAIN_CHECKPOINT_EVERY == 0
        self.ledger.post_slice_hash(head, n)
        if checkpoint:
            self._record(
                "output_chain_checkpoint",
                chain_head=head,
                slices=n,
                start_sample=event.start_sample,
            )

    # ---- summary / lifecycle ---------------------------------------------

    def timeline_summary(self) -> dict:
        """Counts-only view of the timeline for the manifest's
        ``com.daydream.session`` assertion — never prompt content."""
        with self._lock:
            elapsed = time.time() - self._started_wall
            return {
                **self._counts,
                "distinct_prompts": len(self._prompts_seen),
                "slices_hashed": self._slices,
                "duration_s": round(elapsed, 3),
            }

    def manifest_summary(self) -> dict:
        with self._lock:
            model, loras = self._model, list(self._loras)
        return {
            "session_id": self.session_id,
            "model": model,
            "loras": loras,
            "timeline_summary": self.timeline_summary(),
            "session_log_sha256": self.writer.file_sha256(),
        }

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._bus.unsubscribe(self._sub)
            self._sub.join(timeout=5)
        except Exception:  # noqa: BLE001
            pass
        with self._lock:
            head, n = self._chain_head, self._slices
        self._record(
            "session_end",
            output_chain_head=head,
            slices_hashed=n,
            timeline_summary=self.timeline_summary(),
        )
        self.ledger.close()
        self.writer.close()


# ---------------------------------------------------------------------------
# Process-global tap registry, driven by acestep.streaming.registry
# ---------------------------------------------------------------------------

_taps: dict[str, SessionLogTap] = {}
_taps_order: list[str] = []
_taps_lock = threading.Lock()


def attach_session(handle: Any, *, log_dir: Path | None = None) -> Optional[SessionLogTap]:
    """Attach a log tap for a registered session handle (must carry a
    ``bus``; ``snapshot`` and ``provenance_meta`` are optional). Returns
    ``None`` — after one warning — if the tap could not be created."""
    bus = getattr(handle, "bus", None)
    if bus is None:
        return None
    try:
        tap = SessionLogTap(
            bus,
            handle.id,
            meta=getattr(handle, "provenance_meta", None),
            snapshot=getattr(handle, "snapshot", None),
            log_dir=log_dir,
        )
    except Exception as exc:  # noqa: BLE001 — never block session registration
        logger.warning(
            "session log attach failed session_id={} error={}",
            getattr(handle, "id", "?"), exc,
        )
        return None
    with _taps_lock:
        _taps[handle.id] = tap
        _taps_order.append(handle.id)
    logger.info("session_log_attached path={}", tap.writer.path)
    return tap


def detach_session(session_id: str) -> None:
    with _taps_lock:
        tap = _taps.pop(session_id, None)
        if session_id in _taps_order:
            _taps_order.remove(session_id)
    if tap is not None:
        tap.close()


def get_tap(session_id: str) -> Optional[SessionLogTap]:
    with _taps_lock:
        return _taps.get(session_id)


def record_user_action(
    session_id: str, action: str, payload: dict, *, source: str = "ws",
) -> None:
    """Module-level convenience for transport adapters: no-op when the
    session has no attached tap (e.g. the brief window before the
    registry registers the handle)."""
    tap = get_tap(session_id)
    if tap is not None:
        tap.record_user_action(action, payload, source=source)


def latest_session_summary() -> Optional[dict]:
    """Manifest-ready summary of the most recently attached live
    session, or ``None`` when no session is active (e.g. offline
    precompute scripts)."""
    with _taps_lock:
        if not _taps_order:
            return None
        tap = _taps.get(_taps_order[-1])
    return tap.manifest_summary() if tap is not None else None
