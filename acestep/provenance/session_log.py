"""Local session log: a JSONL event stream mirroring the ledger schema.

One file per streaming session under
``<provenance dir>/sessions/<session_id>.jsonl``. Every line is one
event envelope in the **shared** schema of spec 06 §2.2, so "upload my
local history" can replay these files into the cloud ledger unchanged
(spec 02 §7)::

    {"stream": "local", "seq": 0, "type": "session.config",
     "ts": 1751871234567, "payload": {...}}

| Field | Meaning |
|---|---|
| ``stream`` | Always ``"local"`` for this file (spec 06 §2.2 file envelope). |
| ``seq`` | Per-file, contiguous from 0 (single local stream). |
| ``type`` | Namespaced event type (``session.config``, ``action.prompt``, ``action.param``, ``action.lora``, ``action.seed_audio``, ``action.transport``, ``session.note`` …), matching what :mod:`ledger_client` sends. |
| ``ts`` | Wall-clock **milliseconds since epoch** (spec 06 §0: ISO-8601 is never on the wire). |
| ``ppq`` | Optional DAW playhead in PPQ; omitted when unknown. |
| ``payload`` | Type-specific object. Prompt text is allowed here (local-only, spec 02 §5); only counts ever enter a manifest. |

Slice hashing: this tap counts decoded slices seen on the bus, but does
**not** hash them. The §2.3 slice hash is SHA-256 over the *uncompressed
interleaved float16 downlink payload bytes* — which are produced in the
per-subscriber transport codec, not on this bus (the bus carries fully
reconstructed float32 audio). Pod-side slice hashing therefore lives in
the codec and is reported here via :meth:`SessionLogTap.record_pod_slice_hash`
→ ``slice.pod_hash`` ledger events, so the pod and client hash the same
bytes and can cross-check (spec 06 §2.3).

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
    "record_pod_slice_hash",
    "session_summary_for",
    "latest_session_summary",
]

# File-envelope stream label for local logs (spec 06 §2.2).
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
    """Wall-clock ms since epoch (spec 06 §0)."""
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
    prompt content is allowed here (spec 02 §5) even though only counts
    ever enter a manifest."""
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
    """Append-only JSONL writer for the spec 06 §2.2 envelope. Owns the
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
    log in the shared §2.2 schema (and forwards them to the ledger client
    when one is configured).

    Counts decoded output slices for the summary but does not hash them
    (see the module docstring): the §2.3 slice hash is over the transport
    codec's float16 downlink bytes, reported via
    :meth:`record_pod_slice_hash`. Runs on the subscription's drainer
    thread plus whichever thread calls :meth:`record_user_action` /
    :meth:`record_pod_slice_hash`; shared counters sit behind one lock.
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
        self._slice_hashes = 0      # slice.pod_hash reports forwarded
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
        # Input commitment chain (06 §2.2 ``input.chain_head``): one head
        # over every source this session consumed. First input → head is
        # the input's own hash (so a single-input session's record head is
        # directly comparable to ``sha256 <file>``); each further input →
        # head = sha256(prev_head || input_hash). The seal lifts the last
        # head into the public record's ``input_chain_head``.
        self._input_chain_head: str | None = None

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
        # mirroring the WS config handshake (spec 06 §2.2).
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
                    # initial_source_sha256 lands as input.chain_head.
                    if k not in ("checkpoint", "fixture_name",
                                 "fixture_path", "initial_source_sha256")
                },
            },
        )
        if isinstance(snap.get("prompt"), str) and snap["prompt"]:
            self._prompts_seen.add(snap["prompt"])

        # Initial input source, two commitments: the adapter-computed
        # fingerprint of the waveform actually consumed (covers client
        # uploads that never exist as pod files — committed synchronously,
        # so it is always the chain's first entry), and the source FILE's
        # sha256 when a cached file exists (hashed off-thread; a ~10 MB
        # read must not sit in the session-registration path). Later
        # user-provided sources are committed via their buffer
        # fingerprints as SessionReady/SwapReady arrive on the bus.
        initial_fp = meta.get("initial_source_sha256")
        if initial_fp:
            self._extend_input_chain(
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

    # ---- input commitment chain (06 §2.2) --------------------------------

    def _hash_initial_source(self, path: Path, fixture_name: str | None) -> None:
        """SHA-256 the initial source file and commit it to the input
        chain. Runs on its own daemon thread; fail-open like everything
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
        self._extend_input_chain(
            digest, "initial_source",
            {"algo": "sha256:file", "fixture_name": fixture_name},
        )

    def _extend_input_chain(
        self, input_sha256: str, label: str, extra: dict | None = None,
    ) -> None:
        """Fold one input hash into the chain and record the new head.
        The event's ``head`` is what the ledger seal lifts into the public
        record; ``input_sha256`` keeps the individual input verifiable."""
        with self._lock:
            prev = self._input_chain_head
            head = (
                input_sha256 if prev is None
                else hashlib.sha256(
                    (prev + input_sha256).encode("ascii")
                ).hexdigest()
            )
            self._input_chain_head = head
        payload: dict = {
            "head": head, "input_sha256": input_sha256, "label": label,
        }
        if extra:
            payload.update({k: v for k, v in extra.items() if v is not None})
        self._record("input.chain_head", payload)

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
        """One authorship-action record in the §2.2 schema. ``params``
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
                    now - self._last_params_wall < _PARAMS_ACTION_MIN_INTERVAL_S
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

    def record_pod_slice_hash(
        self,
        *,
        sha256: str,
        start_sample: int,
        num_samples: int,
        channels: int,
        slice_seq: int | None = None,
        mac_verified: bool | None = None,
    ) -> None:
        """Report a pod-side output-slice hash (spec 06 §2.3): SHA-256
        over the uncompressed interleaved float16 downlink payload bytes,
        computed in the transport codec where those exact bytes exist.
        Emits a ``slice.pod_hash`` ledger event through this session's
        ledger client so it shares the pod stream's contiguous seq."""
        if self._closed:
            return
        with self._lock:
            self._slice_hashes += 1
        self.ledger.post_slice_hash(
            sha256=sha256,
            start_sample=start_sample,
            num_samples=num_samples,
            channels=channels,
            slice_seq=slice_seq,
            mac_verified=mac_verified,
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
            self._record("action.prompt", {"prompt": _summarize(event.tags)})
        elif isinstance(event, PromptBlendEcho):
            self._record(
                "action.param", {"name": "prompt_blend", "value": event.value},
            )
        elif isinstance(event, ParamsEcho):
            with self._lock:
                self._counts["param_changes"] += 1
            self._record("action.param", {"raw": _summarize(event.raw)})
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
            self._extend_input_chain(
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
            self._extend_input_chain(
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
        # Count decoded slices for the summary. The bus carries fully
        # reconstructed float32 audio, NOT the float16 downlink bytes the
        # client hashes, so hashing here could never cross-check — that
        # hash is produced in the transport codec and reported via
        # record_pod_slice_hash (spec 06 §2.3).
        with self._lock:
            self._slices += 1

    # ---- summary / lifecycle ---------------------------------------------

    def timeline_summary(self) -> dict:
        """Counts-only view of the timeline for the manifest's
        ``com.daydream.session`` assertion — never prompt content."""
        with self._lock:
            elapsed = time.time() - self._started_wall
            return {
                **self._counts,
                "distinct_prompts": len(self._prompts_seen),
                "slices": self._slices,
                "slice_hashes": self._slice_hashes,
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
            n, h = self._slices, self._slice_hashes
        self._record(
            "session.note",
            {
                "note": "session_end",
                "slices": n,
                "slice_hashes": h,
                "timeline_summary": self.timeline_summary(),
            },
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


def record_pod_slice_hash(
    session_id: str,
    *,
    sha256: str,
    start_sample: int,
    num_samples: int,
    channels: int,
    slice_seq: int | None = None,
    mac_verified: bool | None = None,
) -> None:
    """Module-level convenience for the transport codec: forward a
    pod-side slice hash (spec 06 §2.3) to the session's tap, no-op when
    there is none."""
    tap = get_tap(session_id)
    if tap is not None:
        tap.record_pod_slice_hash(
            sha256=sha256,
            start_sample=start_sample,
            num_samples=num_samples,
            channels=channels,
            slice_seq=slice_seq,
            mac_verified=mac_verified,
        )


def session_summary_for(session_id: str) -> Optional[dict]:
    """Manifest-ready summary of a **specific** session (spec 06 §2.5
    binds a record to the session that produced the asset). ``None`` when
    that session has no active tap — callers must then emit null session
    fields rather than borrow another session's identity."""
    tap = get_tap(session_id)
    return tap.manifest_summary() if tap is not None else None


def latest_session_summary() -> Optional[dict]:
    """Summary of the most recently attached live session, or ``None``.

    Do NOT use this to bind a manifest to an asset: under concurrent
    sessions the newest tap is not necessarily the one that produced a
    given asset (audit F7/G5). Use :func:`session_summary_for` with the
    id of the session that produced the asset instead. Retained only for
    diagnostics / single-session callers that explicitly want "whatever
    is live now".
    """
    with _taps_lock:
        if not _taps_order:
            return None
        tap = _taps.get(_taps_order[-1])
    return tap.manifest_summary() if tap is not None else None
