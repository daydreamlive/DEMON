"""Thin client for the Provenance Ledger service (spec 06 §2.3).

Best-effort background reporter for one session's **pod** event stream.
It batches events and POSTs them to::

    POST {DEMON_LEDGER_URL}/sessions/{sessionId}/events
    Authorization: Bearer {DEMON_LEDGER_TOKEN}
    { "events": [ {seq, type, ts, ppq?, payload}, ... ] }

and parses the signed per-slice receipts the ledger returns
(``{stream, chain_head, receipts:[{seq, chain_head, ts, receipt_kid,
sig}]}``, spec 06 §2.3).

Receipts are **verified as they arrive** per the full 06 §2.4
procedure (key-cert chain, validity window, Ed25519 over the
``ddp-receipt:v1`` framing, and — step 5 — equality with the chain
head this client recomputes from its *own* submitted events via
:class:`~acestep.provenance.receipts.ChainMirror`). Verification is
fail-open: a failure logs one WARNING, bumps
:attr:`~LedgerClient.receipts_unverified` and records
:attr:`~LedgerClient.last_verification_failure`; it never interrupts
reporting or streaming.

``DEMON_LEDGER_URL`` already includes the ``/v1`` prefix (it is the
``ledgerBaseUrl`` the queue broker hands the pod at session bootstrap,
spec 06 §1). The bearer token is the per-session **pod ledger token**
(``dlt_pod_…``), delivered to the pod in the broker→pod session
assignment; it is supplied here via ``DEMON_LEDGER_TOKEN`` and/or a
constructor param. With **either** the URL or the token unset the whole
client is a **complete no-op**: construction is free, every method
returns immediately, and no thread is started.

**Dev bootstrap (broker emulation).** Until the real queue broker calls
``POST /internal/v1/sessions`` (spec 06 §2.8), nothing delivers a
per-session pod token to a pod. For staging/dev E2E testing the client
can emulate the broker itself: when the token is unset but
``DEMON_LEDGER_INTERNAL_SECRET`` is, the worker thread mints this
session via the internal endpoint before its first flush (user id from
``DEMON_LEDGER_USER_ID``, default ``"dev"``). Fail-open like the rest
of the client: a failed mint logs one WARNING and reporting stays off.
Production pods must never hold the internal secret — the broker
delivers the token instead.

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

from acestep.provenance.receipts import ChainMirror, ReceiptVerifier

__all__ = [
    "LEDGER_URL_ENV",
    "LEDGER_TOKEN_ENV",
    "LEDGER_INTERNAL_SECRET_ENV",
    "LEDGER_USER_ENV",
    "SLICE_POD_HASH_TYPE",
    "LedgerReceipt",
    "LedgerClient",
]

LEDGER_URL_ENV = "DEMON_LEDGER_URL"
LEDGER_TOKEN_ENV = "DEMON_LEDGER_TOKEN"
# Dev-only broker emulation (see module docstring): with the token unset,
# these let the worker mint the session itself via 06 §2.8.
LEDGER_INTERNAL_SECRET_ENV = "DEMON_LEDGER_INTERNAL_SECRET"
LEDGER_USER_ENV = "DEMON_LEDGER_USER_ID"

# This client reports the pod stream (its token is dlt_pod_…, spec 06 §2.1).
_STREAM = "pod"

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
        # Dev bootstrap (module docstring): armed only when no token was
        # delivered. The mint happens on the worker thread, never here.
        self._bootstrap_secret = (
            "" if self.token
            else os.environ.get(LEDGER_INTERNAL_SECRET_ENV, "")
        )
        self._bootstrap_user = os.environ.get(LEDGER_USER_ENV, "dev")
        self.session_id = session_id
        self._last_receipt: Optional[LedgerReceipt] = None
        self._chain_head: Optional[str] = None
        self._warned = False
        self._queue: Optional[queue.Queue] = None
        self._worker: Optional[threading.Thread] = None
        # Assigned by the worker thread only (single writer): the next
        # per-stream seq. Contiguous from 0 by construction.
        self._next_seq = 0
        # Receipt verification (spec 06 §2.4), owned by the worker
        # thread. The mirror replays our own submitted events through
        # the §2.1 chain rules so step 5 checks receipts against *our*
        # history, not just their signatures.
        self._mirror: Optional[ChainMirror] = None
        self._verifier: Optional[ReceiptVerifier] = None
        self._mirror_broken: Optional[str] = None
        self._verify_warned = False
        self.receipts_verified = 0
        self.receipts_unverified = 0
        self.last_verification_failure: Optional[str] = None

    @property
    def enabled(self) -> bool:
        """Reporting is live only when the pod holds both the ledger URL
        and its write-scoped pod token (spec 06 §1) — or, in dev, the
        internal secret with which the worker can mint one."""
        return bool(self.base_url) and bool(
            self.token or self._bootstrap_secret
        )

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
        abs_sha256: str | None = None,
        canvas_root: str | None = None,
        canvas_chunk: int | None = None,
        ts: int | None = None,
    ) -> None:
        """Enqueue a ``slice.pod_hash`` event (spec 06 §2.3): SHA-256 over
        the uncompressed interleaved float16 slice payload bytes, plus the
        geometry copied from the WS slice header, plus (when supplied) the
        ``abs_sha256`` export-forensics hash over the region's post-update
        float32 client reconstruction."""
        if not self.enabled:
            return
        payload: dict = {
            "start_sample": int(start_sample),
            "num_samples": int(num_samples),
            "channels": int(channels),
            "sha256": sha256,
        }
        if abs_sha256 is not None:
            payload["abs_sha256"] = abs_sha256
        if canvas_root is not None:
            payload["canvas_root"] = canvas_root
        if canvas_chunk is not None:
            payload["canvas_chunk"] = int(canvas_chunk)
        # Batched, not flush-per-slice: one HTTP round-trip per slice can't
        # keep up with live slice rates over WAN (each RTT ~300 ms vs tens
        # of slices/s), so the bounded queue dropped most slice reports —
        # and every dropped slice is a hole export forensics can't see
        # through. The 2 s interval flush batches dozens of slice events
        # per POST (well under the 06 §2.3 caps of 500 events / 1 MiB);
        # receipts stay contemporaneous to within ~2 s + RTT.
        # slice_seq is the pod's monotonic per-session slice counter and
        # the join key for pod/client cross-checking (spec 06 §2.3 / §3).
        if slice_seq is not None:
            payload["slice_seq"] = int(slice_seq)
        if mac_verified is not None:
            payload["mac_verified"] = bool(mac_verified)
        self._enqueue(
            SLICE_POD_HASH_TYPE, payload, ts=ts, ppq=None, flush_now=False,
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
            flush_now = item.pop("_flush", False)
            batch.append(item)
            if flush_now or len(batch) >= _BATCH_MAX:
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
        # Extend the local chain mirror before the POST: it models the
        # history we *submit* (spec 06 §2.1: "recompute the chain head
        # ... from your own submitted events").
        self._mirror_extend(events)
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
                receipts = data.get("receipts")
                if isinstance(receipts, list):
                    for r in receipts:
                        self._verify_receipt(r)
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
        session via ``POST /internal/v1/sessions`` (spec 06 §2.8) and
        adopt the returned pod token. Runs once, on the worker thread,
        before the first flush. Fail-open: on any failure reporting
        stays disabled and one WARNING is logged."""
        # DEMON_LEDGER_URL includes the public /v1 prefix; the internal
        # surface lives beside it at /internal/v1 (spec 06 §2.8).
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
                "(broker emulation, 06 §2.8)",
                self.session_id, self._bootstrap_user,
            )
        except (urllib.error.URLError, OSError, ValueError,
                json.JSONDecodeError) as exc:
            logger.warning(
                "ledger dev bootstrap failed url={} session={} error={} "
                "(reporting disabled; local session log unaffected)",
                url, self.session_id, exc,
            )

    # ---- receipt verification (spec 06 §2.4) -----------------------------

    def _mirror_extend(self, events: list[dict]) -> None:
        """Replay the outgoing batch through the local chain mirror.
        Fail-open: a payload the mirror cannot canonicalize (non-JSON
        type, JS-unsafe integer) permanently marks the mirror broken —
        subsequent receipts count as unverified with that reason, and
        reporting itself is untouched."""
        if self._mirror_broken is not None:
            return
        if self._mirror is None:
            self._mirror = ChainMirror(self.session_id, _STREAM)
        try:
            for event in events:
                self._mirror.append(event)
        except Exception as exc:  # noqa: BLE001 — never break reporting
            self._mirror_broken = f"local chain mirror failed: {exc}"
            logger.warning(
                "ledger receipt verification degraded for session={}: {}",
                self.session_id, self._mirror_broken,
            )

    def _verify_receipt(self, receipt: object) -> None:
        """Run the full §2.4 procedure on one arriving receipt and keep
        the verified/unverified tallies. Never raises."""
        try:
            if self._mirror_broken is not None:
                ok, reason = False, self._mirror_broken
            elif not isinstance(receipt, dict):
                ok, reason = False, "malformed receipt (not an object)"
            else:
                if self._verifier is None:
                    self._verifier = ReceiptVerifier(f"{self.base_url}/keys")
                seq = receipt.get("seq")
                expected = (
                    self._mirror.head_at(seq)
                    if self._mirror is not None and isinstance(seq, int)
                    else None
                )
                ok, reason = self._verifier.verify(
                    session_id=self.session_id,
                    stream=_STREAM,
                    seq=seq,
                    chain_head=receipt.get("chain_head"),
                    ts=receipt.get("ts"),
                    receipt_kid=receipt.get("receipt_kid"),
                    sig=receipt.get("sig"),
                    expected_head=expected,
                )
        except Exception as exc:  # noqa: BLE001 — fail-open by contract
            ok, reason = False, f"receipt verification error: {exc}"
        if ok:
            self.receipts_verified += 1
            return
        self.receipts_unverified += 1
        self.last_verification_failure = reason
        if not self._verify_warned:
            self._verify_warned = True
            logger.warning(
                "ledger receipt FAILED verification session={}: {} "
                "(further failures silenced; see receipts_unverified / "
                "last_verification_failure)",
                self.session_id, reason,
            )
