"""Local JSONL session log: bus tap wiring, §2.2 envelope schema, slice
counting, and the batched ledger client (spec 06 §2.3).

Pure Python — no GPU, no c2pa. Drives a real
:class:`acestep.streaming.events.EventBus` through the registry hook
(:func:`acestep.streaming.registry.register` attaches the tap when the
handle carries a ``bus``) and asserts the on-disk JSONL record uses the
shared ledger schema: ``{stream:"local", seq, type, ts(ms), payload}``
envelopes with namespaced type names, a ``session.config`` opener, and a
``session.note`` seal.

Determinism: :meth:`SessionLogTap.close` (via ``unregister``)
unsubscribes and joins the drainer, and the drainer delivers all
already-queued events before exiting — so once ``unregister`` returns,
every published event is on disk and the seal is the last line.
"""

from __future__ import annotations

import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import acestep.provenance.session_log as session_log_mod
from acestep.provenance.ledger_client import LedgerClient
from acestep.provenance.session_log import (
    LOCAL_STREAM,
    get_tap,
    latest_session_summary,
    record_pod_slice_hash,
    record_user_action,
    session_summary_for,
)
from acestep.streaming import registry
from acestep.streaming.events import (
    AudioReady,
    EventBus,
    PromptApplied,
    SwapFailed,
)


def _slice(tag: int) -> AudioReady:
    audio = np.full((16, 2), float(tag), dtype=np.float32)
    return AudioReady(
        audio=audio, start_sample=tag * 16, num_samples=16, channels=2,
        tick_ms=0.0, dec_ms=0.0, num_gens=tag, params={},
    )


def _register(bus: EventBus, session_id: str, **kwargs) -> registry.SessionHandle:
    handle = registry.SessionHandle(
        id=session_id,
        started_at=time.time(),
        inject=lambda data, audio: None,
        snapshot=kwargs.pop("snapshot", lambda: {}),
        bus=bus,
        provenance_meta=kwargs.pop("provenance_meta", None),
    )
    assert not kwargs
    registry.register(handle)
    return handle


def _read_log(tmp_path: Path, session_id: str) -> list[dict]:
    p = tmp_path / "provenance" / "sessions" / f"{session_id}.jsonl"
    assert p.is_file(), f"missing session log at {p}"
    return [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines()]


def _by_type(lines: list[dict], type_: str) -> list[dict]:
    return [l for l in lines if l["type"] == type_]


def _notes(lines: list[dict], note: str) -> list[dict]:
    return [l for l in lines
            if l["type"] == "session.note" and l["payload"].get("note") == note]


def test_session_log_schema_and_lifecycle(tmp_path, monkeypatch):
    monkeypatch.setenv("ACESTEP_PROVENANCE_DIR", str(tmp_path / "provenance"))
    bus = EventBus()
    snapshot = lambda: {  # noqa: E731 — mirrors adapter snapshot shape
        "lora_catalog": [
            {"id": "lead-guitar", "state": "enabled"},
            {"id": "muted", "state": "disabled"},
        ],
        "prompt": "warm analog dub",
        "bpm": 120,
        "key": "C minor",
        "time_signature": "4",
    }
    _register(
        bus, "sess-schema",
        snapshot=snapshot,
        provenance_meta={"checkpoint": "ace_step_v1", "fixture_name": "loop60"},
    )

    bus.publish(PromptApplied(tags="dark techno"))
    bus.publish(SwapFailed(error="no such fixture"))
    record_user_action("sess-schema", "prompt", {"type": "prompt", "tags": "dub 2"})
    registry.unregister("sess-schema")

    lines = _read_log(tmp_path, "sess-schema")

    # Every line carries the shared §2.2 envelope: a "local" stream label,
    # a contiguous-from-0 seq, a namespaced string type, an integer
    # epoch-ms ts (never an ISO string), and a payload object.
    for i, line in enumerate(lines):
        assert line["stream"] == LOCAL_STREAM
        assert line["seq"] == i
        assert isinstance(line["type"], str) and "." in line["type"]
        assert isinstance(line["ts"], int)
        assert line["ts"] > 1_600_000_000_000  # plausibly ms, not seconds
        assert isinstance(line["payload"], dict)

    # session.config is first and carries the identity fields in payload.
    start = lines[0]
    assert start["type"] == "session.config"
    cfg = start["payload"]
    assert cfg["model"] == "ace_step_v1"
    assert cfg["loras"] == ["lead-guitar"]
    assert cfg["fixture_name"] == "loop60"
    assert cfg["prompt"] == "warm analog dub"
    assert cfg["bpm"] == 120

    # Prompts (bus PromptApplied + user action) both map to action.prompt.
    prompts = _by_type(lines, "action.prompt")
    assert {p["payload"].get("prompt") for p in prompts} >= {"dark techno"}
    user_prompt = [p for p in prompts if p["payload"].get("tags") == "dub 2"]
    assert len(user_prompt) == 1
    assert user_prompt[0]["payload"]["source"] == "ws"

    # A failure lands as a session.note error with the kind + message.
    errors = _notes(lines, "error")
    assert len(errors) == 1
    assert errors[0]["payload"]["kind"] == "SwapFailed"
    assert errors[0]["payload"]["error"] == "no such fixture"

    # session_end seals the log with counts (never prompt content).
    end = lines[-1]
    assert end["type"] == "session.note"
    assert end["payload"]["note"] == "session_end"
    assert end["payload"]["slices"] == 0
    assert end["payload"]["slice_hashes"] == 0
    summary = end["payload"]["timeline_summary"]
    assert summary["prompt_changes"] == 2  # bus PromptApplied + user action
    assert summary["user_actions"] == 1
    assert summary["distinct_prompts"] == 3  # snapshot + applied + user action
    # The seal's summary is computed before the session_end line itself
    # is counted.
    assert summary["events"] == len(lines) - 1
    assert summary["duration_s"] >= 0.0

    # No masquerading float32 slice-hash chain remains (audit F3).
    assert not _by_type(lines, "output_chain_checkpoint")
    assert all("chain_head" not in l["payload"] for l in lines)

    # Detach removed the tap from the process-global registry.
    assert get_tap("sess-schema") is None
    assert latest_session_summary() is None
    assert session_summary_for("sess-schema") is None


def test_slice_counting_and_pod_slice_hash(tmp_path, monkeypatch):
    monkeypatch.setenv("ACESTEP_PROVENANCE_DIR", str(tmp_path / "provenance"))
    bus = EventBus()
    _register(bus, "sess-slice")

    # Decoded slices on the bus are counted, not hashed (the §2.3 hash is
    # over the transport codec's float16 downlink bytes, reported
    # separately via record_pod_slice_hash).
    for i in range(6):
        bus.publish(_slice(i))
    for seq in range(3):
        record_pod_slice_hash(
            "sess-slice",
            sha256="ab" * 32,
            start_sample=seq * 16,
            num_samples=16,
            channels=2,
            slice_seq=seq,
        )
    registry.unregister("sess-slice")

    lines = _read_log(tmp_path, "sess-slice")
    # Per-slice hashes go to the ledger, not the local JSONL (anti-bloat).
    assert not _by_type(lines, "slice.pod_hash")

    end = lines[-1]
    assert end["payload"]["note"] == "session_end"
    assert end["payload"]["slices"] == 6
    assert end["payload"]["slice_hashes"] == 3
    assert end["payload"]["timeline_summary"]["slices"] == 6
    assert end["payload"]["timeline_summary"]["slice_hashes"] == 3


def test_params_actions_are_diffed_and_rate_limited(tmp_path, monkeypatch):
    monkeypatch.setenv("ACESTEP_PROVENANCE_DIR", str(tmp_path / "provenance"))
    bus = EventBus()
    _register(bus, "sess-params")
    tap = get_tap("sess-params")
    assert tap is not None

    # First params action logs only the changed knobs; telemetry keys
    # are stripped.
    record_user_action(
        "sess-params", "params",
        {"type": "params", "steer": 0.5, "flow": 0.1, "playback_pos": 12.5},
    )
    # Inside the rate-limit window: suppressed even though a knob moved.
    record_user_action("sess-params", "params", {"steer": 0.9})
    # Identical values are suppressed regardless of the window.
    monkeypatch.setattr(session_log_mod.time, "monotonic", lambda: 1e9)
    record_user_action(
        "sess-params", "params", {"steer": 0.5, "flow": 0.1},
    )
    # Outside the window with a real change: logs the diff only.
    record_user_action("sess-params", "params", {"steer": 0.7, "flow": 0.1})

    registry.unregister("sess-params")
    lines = _read_log(tmp_path, "sess-params")
    actions = _by_type(lines, "action.param")
    assert [a["payload"]["changed"] for a in actions] == [
        {"steer": 0.5, "flow": 0.1},
        {"steer": 0.7},
    ]
    assert all(a["payload"]["source"] == "ws" for a in actions)
    assert lines[-1]["payload"]["timeline_summary"]["param_changes"] == 2


def test_registry_hook_is_inert_without_bus(tmp_path, monkeypatch):
    monkeypatch.setenv("ACESTEP_PROVENANCE_DIR", str(tmp_path / "provenance"))
    handle = registry.SessionHandle(
        id="sess-no-bus",
        started_at=time.time(),
        inject=lambda data, audio: None,
        snapshot=lambda: {},
    )
    registry.register(handle)
    registry.unregister("sess-no-bus")

    assert get_tap("sess-no-bus") is None
    assert not (tmp_path / "provenance" / "sessions").exists()


# ---------------------------------------------------------------------------
# Batched ledger client (spec 06 §2.3): wire shape, auth, seq, receipts
# ---------------------------------------------------------------------------


class _CapturingLedger:
    """A throwaway HTTP server that captures ingestion requests and
    answers with a §2.3-shaped receipt response."""

    def __init__(self):
        self.requests: list[dict] = []
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):  # silence
                pass

            def do_POST(self):
                length = int(self.headers.get("Content-Length", 0))
                body = json.loads(self.rfile.read(length) or b"{}")
                outer.requests.append({
                    "path": self.path,
                    "auth": self.headers.get("Authorization"),
                    "content_type": self.headers.get("Content-Type"),
                    "body": body,
                })
                if self.path.endswith("/internal/v1/sessions"):
                    # Broker bootstrap surface (spec 06 §2.8), used by the
                    # client's dev broker emulation.
                    resp = json.dumps({
                        "pod_ledger_token": "dlt_pod_minted",
                        "client_ledger_token": "dlt_cli_minted",
                        "slice_mac_key": "bWFj",
                        "ledger_base_url": outer.base_url,
                    }).encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(resp)))
                    self.end_headers()
                    self.wfile.write(resp)
                    return
                last_seq = body["events"][-1]["seq"]
                resp = json.dumps({
                    "stream": "pod",
                    "chain_head": "9c" * 32,
                    "receipts": [{
                        "seq": last_seq,
                        "chain_head": "9c" * 32,
                        "ts": 1751871234890,
                        "receipt_kid": "rk-2026-07-07",
                        "sig": "base64url-signature",
                    }],
                }).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(resp)))
                self.end_headers()
                self.wfile.write(resp)

        self._server = HTTPServer(("127.0.0.1", 0), Handler)
        self.port = self._server.server_address[1]
        self._thread = threading.Thread(
            target=self._server.serve_forever, daemon=True,
        )
        self._thread.start()

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}/v1"

    def close(self):
        self._server.shutdown()
        self._server.server_close()


def test_ledger_client_noop_without_url_or_token():
    # No config at all → disabled, no thread, no crash.
    c = LedgerClient(base_url="", session_id="s", token="")
    assert not c.enabled
    c.post_event("action.prompt", {"prompt": "x"})
    c.post_slice_hash(sha256="ab" * 32, start_sample=0, num_samples=16, channels=2)
    c.close()
    assert c._worker is None
    # URL but no token is still a no-op (the pod token is mandatory).
    c2 = LedgerClient(base_url="http://x/v1", session_id="s", token=None)
    assert not c2.enabled


def test_ledger_client_batches_events_with_auth_and_contiguous_seq():
    server = _CapturingLedger()
    try:
        client = LedgerClient(
            base_url=server.base_url, session_id="sess_9f2c", token="dlt_pod_abc",
        )
        assert client.enabled
        client.post_event("session.config", {"model": "acestep-1.5"}, ts=1000)
        client.post_event("action.prompt", {"prompt": "warm keys"}, ts=1001)
        # A slice-hash report forces a prompt flush and yields a receipt.
        client.post_slice_hash(
            sha256="cd" * 32, start_sample=4177920, num_samples=96000,
            channels=2, slice_seq=812, ts=1002,
        )
        client.close(timeout=5.0)
    finally:
        server.close()

    assert len(server.requests) >= 1
    req = server.requests[0]
    # Path + auth per §2.3.
    assert req["path"] == "/v1/sessions/sess_9f2c/events"
    assert req["auth"] == "Bearer dlt_pod_abc"
    assert req["content_type"] == "application/json"
    # Batched envelope with contiguous seq from 0.
    events = req["body"]["events"]
    assert [e["seq"] for e in events] == list(range(len(events)))
    types = [e["type"] for e in events]
    assert types[:2] == ["session.config", "action.prompt"]
    # The slice.pod_hash event carries the §2.3 payload.
    slice_ev = [e for e in events if e["type"] == "slice.pod_hash"][-1]
    assert slice_ev["payload"] == {
        "start_sample": 4177920, "num_samples": 96000, "channels": 2,
        "sha256": "cd" * 32, "slice_seq": 812,
    }
    # Every event uses integer epoch-ms ts.
    assert all(isinstance(e["ts"], int) for e in events)


def test_ledger_client_dev_bootstrap_mints_pod_token(monkeypatch):
    # No token delivered, but the internal secret is set: the worker mints
    # the session itself (broker emulation, 06 §2.8) before its first flush.
    server = _CapturingLedger()
    try:
        monkeypatch.delenv("DEMON_LEDGER_TOKEN", raising=False)
        monkeypatch.setenv("DEMON_LEDGER_INTERNAL_SECRET", "int_sec_1")
        monkeypatch.setenv("DEMON_LEDGER_USER_ID", "user_dev_1")
        client = LedgerClient(base_url=server.base_url, session_id="sess_bs")
        assert client.enabled, "the internal secret arms the client"
        client.post_slice_hash(
            sha256="ab" * 32, start_sample=0, num_samples=16, channels=2,
            slice_seq=0,
        )
        client.close(timeout=5.0)
    finally:
        server.close()

    assert len(server.requests) == 2
    mint = server.requests[0]
    assert mint["path"] == "/internal/v1/sessions"
    assert mint["auth"] == "Bearer int_sec_1"
    assert mint["body"] == {"session_id": "sess_bs", "user_id": "user_dev_1"}
    ingest = server.requests[1]
    assert ingest["path"] == "/v1/sessions/sess_bs/events"
    assert ingest["auth"] == "Bearer dlt_pod_minted", (
        "events must be reported with the minted pod token"
    )


def test_ledger_client_dev_bootstrap_failure_is_fail_open(monkeypatch):
    # Secret set but no server listening: the mint fails, reporting stays
    # off, and nothing raises into the caller.
    monkeypatch.delenv("DEMON_LEDGER_TOKEN", raising=False)
    monkeypatch.setenv("DEMON_LEDGER_INTERNAL_SECRET", "int_sec_1")
    client = LedgerClient(
        base_url="http://127.0.0.1:1/v1", session_id="sess_bs_fail",
    )
    assert client.enabled
    client.post_event("action.prompt", {"prompt": "x"})
    client.close(timeout=5.0)
    assert client.token == "", "a failed mint must not fabricate a token"
    assert client.last_receipt is None


def test_ledger_client_parses_receipt_field_names():
    server = _CapturingLedger()
    try:
        client = LedgerClient(
            base_url=server.base_url, session_id="s", token="dlt_pod_x",
        )
        client.post_slice_hash(
            sha256="ef" * 32, start_sample=0, num_samples=16, channels=2,
            slice_seq=0,
        )
        client.close(timeout=5.0)
    finally:
        server.close()

    r = client.last_receipt
    assert r is not None
    # Correct wire field names (sig / receipt_kid), not the old ones.
    assert r.sig == "base64url-signature"
    assert r.receipt_kid == "rk-2026-07-07"
    assert r.chain_head == "9c" * 32
    assert r.ts == 1751871234890
    assert client.chain_head == "9c" * 32
