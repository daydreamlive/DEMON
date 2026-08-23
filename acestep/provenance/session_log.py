"""Local session log: a JSONL event stream mirroring the ledger schema.

One file per streaming session under
``<provenance dir>/sessions/<session_id>.jsonl``. Every line is one
event envelope in the **shared** schema of the cloud action log, so
"upload my local history" can replay these files into the cloud ledger
unchanged::

    {"stream": "local", "seq": 0, "type": "session.config",
     "ts": 1751871234567, "payload": {...}}

| Field | Meaning |
|---|---|
| ``stream`` | Always ``"local"`` for this file. |
| ``seq`` | Per-file, contiguous from 0 (single local stream). |
| ``type`` | Namespaced event type (``session.config``, ``action.prompt``, ``action.param``, ``action.lora``, ``action.seed_audio``, ``action.transport``, ``input.source``, ``session.note`` …), matching what :mod:`ledger_client` sends. |
| ``ts`` | Wall-clock **milliseconds since epoch** (ISO-8601 is never on the wire). |
| ``ppq`` | Optional DAW playhead in PPQ; omitted when unknown. |
| ``payload`` | Type-specific object. Prompt text is allowed here (local-only); only counts ever leave the log as summaries. |

This is the Level-1 "recipe log" tier: decoded slices are **counted**
for the summary but never hashed — audio-slice hashing and its
cross-checking belong to the follow-up cryptographic tier.

Wiring: :func:`attach_session` subscribes a :class:`SessionLogTap` to
the session's typed event bus (:mod:`acestep.streaming.events`). The
session registry (:mod:`acestep.streaming.registry`) calls attach/detach
when a handle carrying a ``bus`` is registered/unregistered, so transport
adapters only have to hand the bus over. Torch-free, like the registry.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
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
    "LOCAL_STREAM",
    "SessionLogWriter",
    "SessionLogTap",
    "attach_session",
    "detach_session",
    "get_tap",
    "record_user_action",
]

# File-envelope stream label for local logs.
LOCAL_STREAM = "local"

# Params messages arrive at ~125 Hz; action.param records for them are
# diffed against the last logged values and rate-limited.
_PARAMS_ACTION_MIN_INTERVAL_S = 1.0

# Telemetry fields of the wire "params" message that are not user
# actions (playhead/flow-control reporting).
_PARAMS_TELEMETRY_KEYS = frozenset(
    {"type", "playback_pos", "client_time", "slice_lead_s", "slice_bytes_rx"},
)

_MAX_SUMMARY_STR = 300
_MAX_SUMMARY_LIST = 24


def _now_ms() -> int:
    """Wall-clock ms since epoch."""
    return int(time.time() * 1000)


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
    """Shrink a payload value: bounded strings and lists, no binary.
    Prompt text stays intact up to the cap — the log is local-only, so
    prompt content is allowed here even though only counts ever leave
    the log as summaries."""
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
    """Append-only JSONL writer for the shared event envelope. Owns the
    local stream's contiguous ``seq`` counter. Thread-safe; one flush per
    event so a crash loses at most the in-flight line. Never raises out
    of :meth:`record` — a broken log must not take down the session."""

    def __init__(self, path: Path, session_id: str) -> None:
        self.path = path
        self.session_id = session_id
        self._lock = threading.Lock()
        self._seq = 0
        self._failed = False
        path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(path, "a", encoding="utf-8")

    def record(
        self,
        type: str,
        payload: dict,
        *,
        ts: int | None = None,
        ppq: float | None = None,
    ) -> None:
        try:
            with self._lock:
                if self._fh.closed:
                    return
                line = {
                    "stream": LOCAL_STREAM,
                    "seq": self._seq,
                    "type": type,
                    "ts": _now_ms() if ts is None else int(ts),
                    "payload": payload,
                }
                if ppq is not None:
                    line["ppq"] = float(ppq)
                payload_str = json.dumps(
                    line, default=str, separators=(",", ":"),
                )
                self._fh.write(payload_str + "\n")
                self._fh.flush()
                self._seq += 1
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
    log in the shared schema (and forwards them to the ledger client
    when one is configured).

    Counts decoded output slices for the summary but does not hash them
    (see the module docstring). Runs on the subscription's drainer
    thread plus whichever thread calls :meth:`record_user_action`;
    shared counters sit behind one lock.
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
        self._slices = 0            # decoded slices seen on the bus
        self._counts = {
            "events": 0,
            "prompt_changes": 0,
            "param_changes": 0,
            "user_actions": 0,
        }
        self._prompts_seen: set[str] = set()
        self._input_sources_seen: set[str] = set()
        self._last_params: dict = {}
        # None until the first logged change: time.monotonic() has an
        # arbitrary epoch (near zero on some platforms), so "0.0" would
        # silently rate-limit the session's first knob change away.
        self._last_params_wall: float | None = None
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
        # session.config: model + LoRA identifiers + initial config,
        # mirroring the WS config handshake.
        self._record(
            "session.config",
            {
                "model": self._model,
                "loras": loras,
                "fixture_name": meta.get("fixture_name")
                or snap.get("fixture_name"),
                "prompt": snap.get("prompt"),
                "bpm": snap.get("bpm"),
                "key": snap.get("key"),
                "time_signature": snap.get("time_signature"),
                "extra": {
                    k: v for k, v in meta.items()
                    # fixture_path is a server filesystem path — consumed
                    # below for the input hash, never recorded verbatim;
                    # initial_source_sha256 lands as an input.source event.
                    if k not in ("checkpoint", "fixture_name",
                                 "fixture_path", "initial_source_sha256")
                },
            },
        )
        if isinstance(snap.get("prompt"), str) and snap["prompt"]:
            self._prompts_seen.add(snap["prompt"])

        # Initial input source, two commitments: the adapter-computed
        # fingerprint of the waveform actually consumed (covers client
        # uploads that never exist as pod files — recorded synchronously),
        # and the source FILE's sha256 when a cached file exists (hashed
        # off-thread; a ~10 MB read must not sit in the
        # session-registration path). Later user-provided sources are
        # recorded via their buffer fingerprints as SessionReady/SwapReady
        # arrive on the bus.
        initial_fp = meta.get("initial_source_sha256")
        if initial_fp:
            self._record_input_source(
                initial_fp, "initial_source",
                {"algo": "sha256:buffer_fingerprint",
                 "fixture_name": meta.get("fixture_name")
                 or snap.get("fixture_name")},
            )
        fixture_path = meta.get("fixture_path")
        if fixture_path:
            threading.Thread(
                target=self._hash_initial_source,
                args=(Path(fixture_path),
                      meta.get("fixture_name") or snap.get("fixture_name")),
                name="provenance-input-hash",
                daemon=True,
            ).start()

        self._sub = bus.subscribe(self._on_event, name="provenance")
        self._bus = bus

    # ---- input-source commitments -----------------------------------------

    def _hash_initial_source(self, path: Path, fixture_name: str | None) -> None:
        """SHA-256 the initial source file and record it as an input
        source. Runs on its own daemon thread; fail-open like everything
        else here — an unreadable file logs one warning and that's it."""
        try:
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
        except OSError as exc:
            logger.warning(
                "input source hash failed for session={} path={}: {}",
                self.session_id, path, exc,
            )
            return
        if self._closed:
            return
        self._record_input_source(
            digest, "initial_source",
            {"algo": "sha256:file", "fixture_name": fixture_name},
        )

    def _record_input_source(
        self, input_sha256: str, label: str, extra: dict | None = None,
    ) -> None:
        """Record one consumed input source (``input.source``): the
        input's hash plus how it was computed. A file hash stays directly
        comparable to ``shasum -a 256 <file>``."""
        with self._lock:
            self._input_sources_seen.add(input_sha256)
        payload: dict = {"sha256": input_sha256, "label": label}
        if extra:
            payload.update({k: v for k, v in extra.items() if v is not None})
        self._record("input.source", payload)

    # ---- recording -------------------------------------------------------

    def _record(self, type: str, payload: dict, *, ppq: float | None = None) -> None:
        ts = _now_ms()
        with self._lock:
            self._counts["events"] += 1
        self.writer.record(type, payload, ts=ts, ppq=ppq)
        self.ledger.post_event(type, payload, ts=ts, ppq=ppq)

    def record_user_action(
        self, action: str, payload: dict, *, source: str = "ws",
    ) -> None:
        """One authorship-action record in the shared schema. ``params``
        actions map to ``action.param`` and are diffed against the last
        logged knob values and rate-limited so the ~125 Hz knob channel
        doesn't flood the log."""
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
                    self._last_params_wall is not None
                    and now - self._last_params_wall
                    < _PARAMS_ACTION_MIN_INTERVAL_S
                ):
                    return
                self._last_params.update(values)
                self._last_params_wall = now
                self._counts["user_actions"] += 1
                self._counts["param_changes"] += 1
            self._record(
                "action.param",
                {"source": source, "changed": _summarize(changed)},
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
        body = {"source": source}
        if isinstance(summary, dict):
            body.update(summary)
        else:
            body["value"] = summary
        if action == "prompt":
            self._record("action.prompt", body)
        else:
            body["action"] = action
            self._record("session.note", {"note": "user_action", **body})

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
            self._record("action.prompt", {"prompt": _summarize(event.tags)})
        elif isinstance(event, PromptBlendEcho):
            self._record(
                "action.param", {"name": "prompt_blend", "value": event.value},
            )
        elif isinstance(event, ParamsEcho):
            if getattr(event, "origin", "external") == "external":
                # Deliberate MCP/control-bus action: record verbatim.
                with self._lock:
                    self._counts["param_changes"] += 1
                self._record("action.param", {"raw": _summarize(event.raw)})
            else:
                # The performer's own knob stream (up to 125 Hz during a
                # drag): reuse the diff + rate-limit path so the log gets
                # "what changed", not a firehose.
                self.record_user_action(
                    "params", dict(event.raw), source="primary",
                )
        elif isinstance(event, DepthApplied):
            self._record(
                "action.param", {"name": "depth", "value": event.value},
            )
        elif isinstance(event, LoraCatalogUpdate):
            loras = [
                e.get("id")
                for e in event.catalog or []
                if isinstance(e, dict) and e.get("state") == "enabled"
            ]
            with self._lock:
                self._loras = loras
            self._record("action.lora", {"loras": loras})
        elif isinstance(event, SessionReady):
            fp = buffer_fingerprint(event.initial_buffer)
            self._record(
                "action.seed_audio",
                {
                    "sha256": fp,
                    "label": "initial_source",
                    "duration_sec": event.duration,
                    "sample_rate": event.sample_rate,
                    "bpm": event.bpm,
                    "key": event.key,
                    "time_signature": event.time_signature,
                    "pipeline_depth": event.pipeline_depth,
                },
            )
            self._record_input_source(
                fp, "initial_source", {"algo": "sha256:buffer_fingerprint"},
            )
        elif isinstance(event, SwapReady):
            fp = buffer_fingerprint(event.initial_buffer)
            self._record(
                "action.seed_audio",
                {
                    "sha256": fp,
                    "label": "source_swap",
                    "fixture_name": event.fixture_name,
                    "duration_sec": event.duration,
                    "bpm": event.bpm,
                    "key": event.key,
                    "source_epoch": event.source_epoch,
                },
            )
            self._record_input_source(
                fp, "source_swap",
                {"algo": "sha256:buffer_fingerprint",
                 "fixture_name": event.fixture_name},
            )
        elif isinstance(event, TimbreSet):
            self._record(
                "action.param",
                {"name": "timbre", "op": "set",
                 "timbre": event.name, "duration": event.duration},
            )
        elif isinstance(event, TimbreCleared):
            self._record("action.param", {"name": "timbre", "op": "clear"})
        elif isinstance(event, StructureSet):
            self._record(
                "action.param",
                {"name": "structure", "op": "set",
                 "structure": event.name, "duration": event.duration},
            )
        elif isinstance(event, StructureCleared):
            self._record("action.param", {"name": "structure", "op": "clear"})
        elif isinstance(event, AudioWritten):
            self._record(
                "action.transport",
                {"op": "write_audio", "start_s": event.start_s,
                 "end_s": event.end_s, "source_epoch": event.source_epoch},
            )
        elif isinstance(event, StemAssets):
            self._record(
                "session.note",
                {"note": "stem_assets", "fixture_name": event.fixture_name,
                 "source_mode": event.source_mode, "frames": event.frames},
            )
        elif isinstance(event, (
            CommandFailed, SwapFailed, StemFailed, TimbreFailed,
            StructureFailed, AudioWriteFailed, SessionError,
        )):
            self._record(
                "session.note",
                {"note": "error", "kind": type(event).__name__,
                 "error": _summarize(getattr(event, "error", None)
                                     or getattr(event, "message", ""))},
            )
        elif isinstance(event, SubscriberDropped):
            # Our own queue overflowed: the log has a gap from here on.
            self._record(
                "session.note",
                {"note": "log_tap_dropped", "reason": event.reason},
            )

    def _on_slice(self, event: AudioReady) -> None:
        # Count decoded slices for the summary — Level 1 never hashes
        # audio; slice hashing belongs to the follow-up cryptographic
        # tier.
        with self._lock:
            self._slices += 1

    # ---- summary / lifecycle ---------------------------------------------

    def timeline_summary(self) -> dict:
        """Counts-only view of the timeline — never prompt content."""
        with self._lock:
            elapsed = time.time() - self._started_wall
            return {
                **self._counts,
                "distinct_prompts": len(self._prompts_seen),
                "distinct_input_sources": len(self._input_sources_seen),
                "slices": self._slices,
                "duration_s": round(elapsed, 3),
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
            n = self._slices
        self._record(
            "session.note",
            {
                "note": "session_end",
                "slices": n,
                "timeline_summary": self.timeline_summary(),
            },
        )
        self.ledger.close()
        self.writer.close()


# ---------------------------------------------------------------------------
# Process-global tap registry, driven by acestep.streaming.registry
# ---------------------------------------------------------------------------

_taps: dict[str, SessionLogTap] = {}
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
    logger.info("session_log_attached path={}", tap.writer.path)
    return tap


def detach_session(session_id: str) -> None:
    with _taps_lock:
        tap = _taps.pop(session_id, None)
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
