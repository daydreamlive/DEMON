"""Thin client for the Provenance Ledger service (spec 06 §2.3).

Best-effort background reporter for one session's **pod** event stream.
It batches events and POSTs them to::

    POST {DEMON_LEDGER_URL}/sessions/{sessionId}/events
    Authorization: Bearer {DEMON_LEDGER_TOKEN}
    { "events": [ {seq, type, ts, ppq?, payload}, ... ] }

and parses the signed per-slice receipts the ledger returns
(``{stream, chain_head, receipts:[{seq, chain_head, ts, receipt_kid,
sig}]}``, spec 06 §2.3).

``DEMON_LEDGER_URL`` already includes the ``/v1`` prefix (it is the
``ledgerBaseUrl`` the queue broker hands the pod at session bootstrap,
spec 06 §1). The bearer token is the per-session **pod ledger token**
(``dlt_pod_…``), delivered to the pod in the broker→pod session
assignment; it is supplied here via ``DEMON_LEDGER_TOKEN`` and/or a
constructor param. With **either** the URL or the token unset the whole
client is a **complete no-op**: construction is free, every method
returns immediately, and no thread is started.

Async-safe by construction: callers (the bus drainer thread, the WS
dispatch thread, async handlers) only ever enqueue onto a bounded
in-memory queue; one daemon worker owns all network I/O and is the sole
assigner of the per-stream ``seq``. Overflow drops the *oldest*
un-flushed event — the local session log, not this client, is the
durable record, and provenance reporting is fail-open for playback
(spec 06 §7). Because ``seq`` is assigned by the worker at flush time
(never at enqueue time), a dropped event never punches a hole in the
sequence: the surviving events stay contiguous from 0 and the ledger's
seq-gap check (spec 06 §2.3) is not tripped by best-effort drops.
"""

from __future__ import annotations

import json
import os
import queue
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Optional

from loguru import logger

__all__ = [
    "LEDGER_URL_ENV",
    "LEDGER_TOKEN_ENV",
    "SLICE_POD_HASH_TYPE",
    "LedgerReceipt",
    "LedgerClient",
]

LEDGER_URL_ENV = "DEMON_LEDGER_URL"
LEDGER_TOKEN_ENV = "DEMON_LEDGER_TOKEN"

# Event type whose ingestion returns a signed receipt (spec 06 §2.3).
SLICE_POD_HASH_TYPE = "slice.pod_hash"

_QUEUE_MAX = 1024
_REQUEST_TIMEOUT_S = 5.0
# Batch caps (spec 06 §2.3: max 500 events / 1 MiB per request).
_BATCH_MAX = 500
# Flush cadence for non-receipted events: whichever comes first, a
# slice-hash report (flushed immediately for its receipt) or this
# interval (spec 06 §2.3 recommends 2 s).
_FLUSH_INTERVAL_S = 2.0


def _now_ms() -> int:
    """Wall-clock ms since epoch — the on-the-wire timestamp form
    (spec 06 §0: ISO-8601 is never used on the wire)."""
    return int(time.time() * 1000)


@dataclass(frozen=True)
class LedgerReceipt:
    """Signed per-slice acknowledgement (spec 06 §2.3): an Ed25519
    signature over the ``ddp-receipt:v1`` framing, committing to the
    chain head at a given ``seq``. Field names track the wire response
    exactly (``sig`` not ``signature``, ``receipt_kid`` not ``key_id``)."""

    seq: Optional[int] = None
    chain_head: Optional[str] = None
    ts: Optional[int] = None
    receipt_kid: Optional[str] = None
    sig: Optional[str] = None
    raw: dict = field(default_factory=dict)

    @classmethod
    def from_response(cls, data: dict) -> Optional["LedgerReceipt"]:
        """Parse the latest receipt out of an ingestion response
        ``{stream, chain_head, receipts:[...]}``. Returns ``None`` when
        the batch produced no receipts (e.g. a batch of only non-slice
        events, which are chained but not individually receipted)."""
        if not isinstance(data, dict):
            return None
        receipts = data.get("receipts")
        if not isinstance(receipts, list) or not receipts:
            return None
        last = receipts[-1]
        if not isinstance(last, dict):
            return None
        return cls(
            seq=last.get("seq"),
            chain_head=last.get("chain_head") or data.get("chain_head"),
            ts=last.get("ts"),
            receipt_kid=last.get("receipt_kid"),
            sig=last.get("sig"),
            raw=last,
        )


class LedgerClient:
    """Fire-and-forget batched event reporter for one session's pod
    stream. A no-op unless both a base URL and a bearer token are
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
        self.session_id = session_id
        self._last_receipt: Optional[LedgerReceipt] = None
        self._chain_head: Optional[str] = None
        self._warned = False
        self._queue: Optional[queue.Queue] = None
        self._worker: Optional[threading.Thread] = None
        # Assigned by the worker thread only (single writer): the next
        # per-stream seq. Contiguous from 0 by construction.
        self._next_seq = 0

    @property
    def enabled(self) -> bool:
        """Reporting is live only when the pod holds both the ledger URL
        and its write-scoped pod token (spec 06 §1)."""
        return bool(self.base_url) and bool(self.token)

    @property
    def last_receipt(self) -> Optional[LedgerReceipt]:
        """Most recent parsed receipt (spec 06 §2.3: the reporter keeps
        the latest signed commitment)."""
        return self._last_receipt

    @property
    def chain_head(self) -> Optional[str]:
        """Latest server-reported chain head for this stream."""
        return self._chain_head

    # ---- reporting -------------------------------------------------------

    def post_event(
        self,
        type: str,
        payload: dict | None = None,
        *,
        ts: int | None = None,
        ppq: float | None = None,
    ) -> None:
        """Enqueue one authorship / config event (spec 06 §2.2). ``seq``
        is assigned later, by the worker, so best-effort drops never
        create gaps."""
        if not self.enabled:
            return
        self._enqueue(type, payload or {}, ts=ts, ppq=ppq, flush_now=False)

    def post_slice_hash(
        self,
        *,
        sha256: str,
        start_sample: int,
        num_samples: int,
        channels: int,
        slice_seq: int | None = None,
        mac_verified: bool | None = None,
        ts: int | None = None,
    ) -> None:
        """Enqueue a ``slice.pod_hash`` event (spec 06 §2.3): SHA-256 over
        the uncompressed interleaved float16 slice payload bytes, plus the
        geometry copied from the WS slice header. Flushed promptly because
        every slice-hash event gets a signed receipt."""
        if not self.enabled:
            return
        payload: dict = {
            "start_sample": int(start_sample),
            "num_samples": int(num_samples),
            "channels": int(channels),
            "sha256": sha256,
        }
        # slice_seq is the pod's monotonic per-session slice counter and
        # the join key for pod/client cross-checking (spec 06 §2.3 / §3).
        if slice_seq is not None:
            payload["slice_seq"] = int(slice_seq)
        if mac_verified is not None:
            payload["mac_verified"] = bool(mac_verified)
        self._enqueue(
            SLICE_POD_HASH_TYPE, payload, ts=ts, ppq=None, flush_now=True,
        )

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
        flush_now: bool,
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
            "_flush": flush_now,
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
            flush_now = item.pop("_flush", False)
            batch.append(item)
            if flush_now or len(batch) >= _BATCH_MAX:
                self._flush(batch)
                batch = []

    def _flush(self, batch: list[dict]) -> None:
        if not batch:
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
                head = data.get("chain_head")
                if isinstance(head, str):
                    self._chain_head = head
                receipt = LedgerReceipt.from_response(data)
                if receipt is not None:
                    self._last_receipt = receipt
        except (urllib.error.URLError, OSError, ValueError) as exc:
            if not self._warned:
                self._warned = True
                logger.warning(
                    "ledger post failed url={} error={} "
                    "(further failures silenced)",
                    url, exc,
                )
