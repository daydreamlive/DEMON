"""Thin client for the Provenance Action Log service.

Best-effort background reporter for one session's event stream. It
batches events and POSTs them to::

    POST {DEMON_LEDGER_URL}/sessions/{sessionId}/events
    Authorization: Bearer {DEMON_LEDGER_TOKEN}
    { "events": [ {seq, type, ts, ppq?, payload}, ... ] }

This is the Level-1 "recipe log" tier: the service acknowledges with a
plain ``{event_count}`` — no hash chains, no signed receipts.

``DEMON_LEDGER_URL`` already includes the ``/v1`` prefix (it is the
``ledgerBaseUrl`` the queue broker hands the pod at session bootstrap).
The bearer token is the per-session **pod ledger token** (``dlt_pod_…``),
delivered to the pod in the broker→pod session assignment; it is
supplied here via ``DEMON_LEDGER_TOKEN`` and/or a constructor param.
With **either** the URL or the token unset the whole client is a
**complete no-op**: construction is free, every method returns
immediately, and no thread is started.

**Dev bootstrap (broker emulation).** Until the real queue broker calls
``POST /internal/v1/sessions``, nothing delivers a per-session pod token
to a pod. For staging/dev E2E testing the client can emulate the broker
itself: when the token is unset but ``DEMON_LEDGER_INTERNAL_SECRET`` is,
the worker thread mints this session via the internal endpoint before
its first flush (user id from ``DEMON_LEDGER_USER_ID``, default
``"dev"``). Fail-open like the rest of the client: a failed mint logs
one WARNING and reporting stays off. Production pods must never hold the
internal secret — the broker delivers the token instead.

Async-safe by construction: callers (the bus drainer thread, the WS
dispatch thread, async handlers) only ever enqueue onto a bounded
in-memory queue; one daemon worker owns all network I/O and is the sole
assigner of the per-stream ``seq``. Overflow drops the *oldest*
un-flushed event — the local session log, not this client, is the
durable record, and provenance reporting is fail-open for playback.
Because ``seq`` is assigned by the worker at flush time (never at
enqueue time), a dropped event never punches a hole in the sequence:
the surviving events stay contiguous from 0 and the ledger's seq-gap
check is not tripped by best-effort drops.
"""

from __future__ import annotations

import json
import os
import queue
import threading
import time
import urllib.error
import urllib.request
from typing import Optional

from loguru import logger

__all__ = [
    "LEDGER_URL_ENV",
    "LEDGER_TOKEN_ENV",
    "LEDGER_INTERNAL_SECRET_ENV",
    "LEDGER_USER_ENV",
    "LedgerClient",
]

LEDGER_URL_ENV = "DEMON_LEDGER_URL"
LEDGER_TOKEN_ENV = "DEMON_LEDGER_TOKEN"
# Dev-only broker emulation (see module docstring): with the token unset,
# these let the worker mint the session itself.
LEDGER_INTERNAL_SECRET_ENV = "DEMON_LEDGER_INTERNAL_SECRET"
LEDGER_USER_ENV = "DEMON_LEDGER_USER_ID"

_QUEUE_MAX = 1024
_REQUEST_TIMEOUT_S = 5.0
# Batch caps (max 500 events / 1 MiB per request on the service side).
_BATCH_MAX = 500
# Flush cadence: whichever comes first, a full batch or this interval.
_FLUSH_INTERVAL_S = 2.0


def _now_ms() -> int:
    """Wall-clock ms since epoch — the on-the-wire timestamp form
    (ISO-8601 is never used on the wire)."""
    return int(time.time() * 1000)


class LedgerClient:
    """Fire-and-forget batched event reporter for one session's action
    log. A no-op unless both a base URL and a bearer token are
    configured."""

    def __init__(
        self,
        base_url: str | None = None,
        *,
        session_id: str = "",
        token: str | None = None,
    ) -> None:
        self.base_url = (
            base_url if base_url is not None
            else os.environ.get(LEDGER_URL_ENV, "")
        ).rstrip("/")
        self.token = (
            token if token is not None
            else os.environ.get(LEDGER_TOKEN_ENV, "")
        )
        # Dev bootstrap (module docstring): armed only when no token was
        # delivered. The mint happens on the worker thread, never here.
        self._bootstrap_secret = (
            "" if self.token
            else os.environ.get(LEDGER_INTERNAL_SECRET_ENV, "")
        )
        self._bootstrap_user = os.environ.get(LEDGER_USER_ENV, "dev")
        self.session_id = session_id
        self._warned = False
        self._queue: Optional[queue.Queue] = None
        self._worker: Optional[threading.Thread] = None
        # Assigned by the worker thread only (single writer): the next
        # per-stream seq. Contiguous from 0 by construction.
        self._next_seq = 0
        # Last event count the service acknowledged.
        self._acked_count: Optional[int] = None

    @property
    def enabled(self) -> bool:
        """Reporting is live only when the pod holds both the ledger URL
        and its write-scoped pod token — or, in dev, the internal secret
        with which the worker can mint one."""
        return bool(self.base_url) and bool(
            self.token or self._bootstrap_secret
        )

    @property
    def acked_count(self) -> Optional[int]:
        """Latest server-acknowledged event count for this session."""
        return self._acked_count

    # ---- reporting -------------------------------------------------------

    def post_event(
        self,
        type: str,
        payload: dict | None = None,
        *,
        ts: int | None = None,
        ppq: float | None = None,
    ) -> None:
        """Enqueue one authorship / config event. ``seq`` is assigned
        later, by the worker, so best-effort drops never create gaps."""
        if not self.enabled:
            return
        self._enqueue(type, payload or {}, ts=ts, ppq=ppq)

    def close(self, timeout: float = 2.0) -> None:
        if self._queue is None:
            return
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            # Make room for the sentinel so the worker can drain + exit.
            try:
                self._queue.get_nowait()
                self._queue.put_nowait(None)
            except (queue.Empty, queue.Full):
                pass
        if self._worker is not None:
            self._worker.join(timeout=timeout)

    # ---- worker ----------------------------------------------------------

    def _enqueue(
        self,
        type: str,
        payload: dict,
        *,
        ts: int | None,
        ppq: float | None,
    ) -> None:
        if self._queue is None:
            self._queue = queue.Queue(maxsize=_QUEUE_MAX)
            self._worker = threading.Thread(
                target=self._drain, name="ledger-client", daemon=True,
            )
            self._worker.start()
        item = {
            "type": type,
            "ts": _now_ms() if ts is None else int(ts),
            "payload": payload,
        }
        if ppq is not None:
            item["ppq"] = float(ppq)
        try:
            self._queue.put_nowait(item)
        except queue.Full:
            # Drop the oldest un-flushed item to make room (fail-open):
            # the local log is the durable record. seq is assigned at
            # flush, so the drop leaves the surviving stream contiguous.
            try:
                self._queue.get_nowait()
                self._queue.put_nowait(item)
            except (queue.Empty, queue.Full):
                pass

    def _drain(self) -> None:
        assert self._queue is not None
        if not self.token and self._bootstrap_secret:
            self._bootstrap()
        batch: list[dict] = []
        while True:
            timeout = _FLUSH_INTERVAL_S if batch else None
            try:
                item = self._queue.get(timeout=timeout)
            except queue.Empty:
                # Idle flush interval elapsed with events pending.
                self._flush(batch)
                batch = []
                continue
            if item is None:
                self._flush(batch)
                return
            batch.append(item)
            if len(batch) >= _BATCH_MAX:
                self._flush(batch)
                batch = []

    def _flush(self, batch: list[dict]) -> None:
        if not batch:
            return
        if not self.token:
            # Bootstrap failed (or never armed): fail-open — the local
            # session log remains the durable record.
            return
        # Assign contiguous seq now (worker is the single seq writer).
        events = []
        for item in batch:
            event = {"seq": self._next_seq, "type": item["type"],
                     "ts": item["ts"], "payload": item["payload"]}
            if "ppq" in item:
                event["ppq"] = item["ppq"]
            events.append(event)
            self._next_seq += 1
        body = json.dumps({"events": events}, default=str).encode("utf-8")
        url = f"{self.base_url}/sessions/{self.session_id}/events"
        try:
            req = urllib.request.Request(
                url,
                data=body,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.token}",
                },
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=_REQUEST_TIMEOUT_S) as resp:
                raw = resp.read()
            try:
                data = json.loads(raw)
            except (json.JSONDecodeError, UnicodeDecodeError):
                return
            if isinstance(data, dict):
                count = data.get("event_count")
                if isinstance(count, int):
                    self._acked_count = count
        except (urllib.error.URLError, OSError, ValueError) as exc:
            if not self._warned:
                self._warned = True
                logger.warning(
                    "ledger post failed url={} error={} "
                    "(further failures silenced)",
                    url, exc,
                )

    def _bootstrap(self) -> None:
        """Dev-only broker emulation (module docstring): mint this
        session via ``POST /internal/v1/sessions`` and adopt the returned
        pod token. Runs once, on the worker thread, before the first
        flush. Fail-open: on any failure reporting stays disabled and one
        WARNING is logged."""
        # DEMON_LEDGER_URL includes the public /v1 prefix; the internal
        # surface lives beside it at /internal/v1.
        origin = (
            self.base_url[: -len("/v1")]
            if self.base_url.endswith("/v1") else self.base_url
        )
        url = f"{origin}/internal/v1/sessions"
        body = json.dumps({
            "session_id": self.session_id,
            "user_id": self._bootstrap_user,
        }).encode("utf-8")
        try:
            req = urllib.request.Request(
                url,
                data=body,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self._bootstrap_secret}",
                },
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=_REQUEST_TIMEOUT_S) as resp:
                data = json.loads(resp.read())
            token = data.get("pod_ledger_token") if isinstance(data, dict) else None
            if not isinstance(token, str) or not token:
                raise ValueError("response carried no pod_ledger_token")
            self.token = token
            logger.info(
                "ledger dev bootstrap minted session={} user={} "
                "(broker emulation)",
                self.session_id, self._bootstrap_user,
            )
        except (urllib.error.URLError, OSError, ValueError,
                json.JSONDecodeError) as exc:
            logger.warning(
                "ledger dev bootstrap failed url={} session={} error={} "
                "(reporting disabled; local session log unaffected)",
                url, self.session_id, exc,
            )
