"""Thin client for the (future) Provenance Ledger service.

POSTs session events and output-slice chain heads to
``$DEMON_LEDGER_URL`` and parses the signed receipt the ledger returns
per slice-hash report (spec 02 §5). The service does not exist yet, so
the default state is a **complete no-op**: with the env var unset,
construction is free, every method returns immediately, and no thread
is started.

Async-safe by construction: callers (bus drainer threads, the WS
dispatch thread, async handlers) only ever enqueue onto a bounded
in-memory queue; one daemon worker owns all network I/O. Overflow
drops the oldest report — the local session log, not this client, is
the durable record.
"""

from __future__ import annotations

import json
import os
import queue
import threading
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Optional

from loguru import logger

__all__ = ["LEDGER_URL_ENV", "LedgerReceipt", "LedgerClient"]

LEDGER_URL_ENV = "DEMON_LEDGER_URL"

_QUEUE_MAX = 1024
_REQUEST_TIMEOUT_S = 5.0


@dataclass(frozen=True)
class LedgerReceipt:
    """Signed per-report acknowledgement: an Ed25519 countersignature
    over the current chain head + timestamp (spec 02 §5)."""

    chain_head: Optional[str] = None
    signature: Optional[str] = None
    key_id: Optional[str] = None
    ts: Optional[str] = None
    raw: dict = field(default_factory=dict)

    @classmethod
    def from_response(cls, data: dict) -> "LedgerReceipt":
        receipt = data.get("receipt", data)
        if not isinstance(receipt, dict):
            receipt = {}
        return cls(
            chain_head=receipt.get("chain_head"),
            signature=receipt.get("signature"),
            key_id=receipt.get("key_id"),
            ts=receipt.get("ts"),
            raw=receipt,
        )


class LedgerClient:
    """Fire-and-forget event/slice-hash reporter for one session."""

    def __init__(
        self,
        base_url: str | None = None,
        *,
        session_id: str = "",
    ) -> None:
        self.base_url = (
            base_url if base_url is not None
            else os.environ.get(LEDGER_URL_ENV, "")
        ).rstrip("/")
        self.session_id = session_id
        self._last_receipt: Optional[LedgerReceipt] = None
        self._warned = False
        self._queue: Optional[queue.Queue] = None
        self._worker: Optional[threading.Thread] = None

    @property
    def enabled(self) -> bool:
        return bool(self.base_url)

    @property
    def last_receipt(self) -> Optional[LedgerReceipt]:
        """Most recent parsed receipt (spec: the client keeps the latest
        signed commitment; rtmg-vst stores it in plugin state, here it
        just rides the session)."""
        return self._last_receipt

    # ---- reporting -------------------------------------------------------

    def post_event(self, event: dict) -> None:
        if not self.enabled:
            return
        self._enqueue(f"/sessions/{self.session_id}/events", event)

    def post_slice_hash(self, chain_head: str, index: int) -> None:
        if not self.enabled:
            return
        self._enqueue(
            f"/sessions/{self.session_id}/slices",
            {"chain_head": chain_head, "index": index},
        )

    def close(self, timeout: float = 2.0) -> None:
        if self._queue is None:
            return
        self._queue.put(None)
        if self._worker is not None:
            self._worker.join(timeout=timeout)

    # ---- worker ----------------------------------------------------------

    def _enqueue(self, path: str, payload: dict) -> None:
        if self._queue is None:
            self._queue = queue.Queue(maxsize=_QUEUE_MAX)
            self._worker = threading.Thread(
                target=self._drain, name="ledger-client", daemon=True,
            )
            self._worker.start()
        try:
            self._queue.put_nowait((path, payload))
        except queue.Full:
            try:
                self._queue.get_nowait()
                self._queue.put_nowait((path, payload))
            except (queue.Empty, queue.Full):
                pass

    def _drain(self) -> None:
        assert self._queue is not None
        while True:
            item = self._queue.get()
            if item is None:
                return
            path, payload = item
            try:
                req = urllib.request.Request(
                    self.base_url + path,
                    data=json.dumps(payload, default=str).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(
                    req, timeout=_REQUEST_TIMEOUT_S,
                ) as resp:
                    body = resp.read()
                try:
                    self._last_receipt = LedgerReceipt.from_response(
                        json.loads(body),
                    )
                except (json.JSONDecodeError, UnicodeDecodeError):
                    pass
            except (urllib.error.URLError, OSError, ValueError) as exc:
                if not self._warned:
                    self._warned = True
                    logger.warning(
                        "ledger post failed url={} error={} "
                        "(further failures silenced)",
                        self.base_url + path, exc,
                    )
