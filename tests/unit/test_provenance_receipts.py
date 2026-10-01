"""Client-side ledger receipt verification (spec 06 §2.4).

Pure Python — no GPU, no c2pa. Covers the three layers of
:mod:`acestep.provenance.receipts` and their wiring into
:class:`~acestep.provenance.ledger_client.LedgerClient`:

- **JCS + chain golden vector**: the fixed events, canonical strings
  and chain heads below are copied verbatim from the server's own
  regression fixtures (pipelines-provenance ``test/provenance/
  chain.test.ts`` / ``jcs.test.ts``). Both implementations hashing the
  same inputs to the same heads is the whole point of §2.4 step 5 —
  these values must NEVER be regenerated from this codebase.
- **§2.4 procedure**: server-shaped key documents and receipts built
  with ephemeral Ed25519 keys and the exact ``ddp-keycert:v1`` /
  ``ddp-receipt:v1`` framings; every element (sig, chain_head, ts,
  key-cert) is tampered in turn and must fail with a telling reason.
- **LedgerClient integration**: a stub ledger that really chains and
  really signs; receipts verify as they arrive, tampering bumps the
  unverified counter, and nothing ever raises into the reporter
  (fail-open).
"""

from __future__ import annotations

import base64
import json
import math
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

import pytest

from acestep.provenance.ledger_client import LedgerClient
from acestep.provenance.receipts import (
    ChainMirror,
    ReceiptVerifier,
    advance_head,
    event_hash,
    genesis_head,
    jcs,
    key_cert_message,
    receipt_message,
)

# ---------------------------------------------------------------------------
# Golden fixture — copied VERBATIM from the server's chain.test.ts.
# ---------------------------------------------------------------------------

GOLDEN_SESSION = "sess_test01"
GOLDEN_STREAM = "client"
GOLDEN_EVENTS = [
    {
        "seq": 0,
        "type": "action.prompt",
        "ts": 1751871234567,
        "payload": {"prompt": "warm analog keys", "slot": "A"},
    },
    {
        "seq": 1,
        "type": "action.param",
        "ts": 1751871234890,
        "ppq": 3840.25,
        "payload": {"name": "cutoff", "value": 0.42},
    },
    {
        "seq": 2,
        "type": "slice.client_hash",
        "ts": 1751871235000,
        "payload": {
            "slice_seq": 812,
            "start_sample": 4177920,
            "num_samples": 96000,
            "channels": 2,
            "sha256": "ab34cd56ab34cd56ab34cd56ab34cd56"
                      "ab34cd56ab34cd56ab34cd56ab34cd56",
            "mac_verified": True,
        },
    },
]
GOLDEN = {
    "genesis": "5dbdea7c3069a7a5af26257c7a8a9899d5a6ace861e7173a0a45d8ed5f9ea4d6",
    "event0Hash": "311d3b371d059b44fa137d8922b822c24f05d8b881ec78144ac390d09a5e7ed9",
    "head1": "84c9291fd02dee9a3bca470469c0edce22cee49b837d53cfdb2ec049ac542905",
    "head2": "12b769bb9d8eda564f5a81c279d2f563d60220042a3e6f9b9447b34acebb11f9",
    "head3": "03cbf0fde0a7b6238600859273b5c7f6088f8701d20206715c9eabe3ce63de9b",
}


# ---------------------------------------------------------------------------
# JCS canonicalization (RFC 8785, ECMAScript number/string formatting)
# ---------------------------------------------------------------------------


def test_jcs_matches_server_golden_canonical_form():
    # Same assertion as the server's jcs.test.ts: the exact bytes a
    # reporter must produce locally for receipt verification to work.
    assert jcs(GOLDEN_EVENTS[0]) == (
        '{"payload":{"prompt":"warm analog keys","slot":"A"},'
        '"seq":0,"ts":1751871234567,"type":"action.prompt"}'
    )


def test_jcs_sorts_keys_recursively_and_deterministically():
    assert jcs({"b": 1, "a": {"d": 2, "c": 3}}) == '{"a":{"c":3,"d":2},"b":1}'
    x = {}
    x["z"] = 1
    x["a"] = 2
    y = {}
    y["a"] = 2
    y["z"] = 1
    assert jcs(x) == jcs(y)


def test_jcs_numbers_serialize_like_ecmascript_json_stringify():
    # The jcs.test.ts vector plus the notation boundaries JS uses.
    assert jcs(3840.25) == "3840.25"
    assert jcs(1e21) == "1e+21"          # not Python's "1e+21" by luck: e+ form
    assert jcs(0.000001) == "0.000001"   # Python repr says "1e-06"
    assert jcs(1e-7) == "1e-7"           # exponential below 1e-6, JS "e-" form
    assert jcs(1751871234567) == "1751871234567"
    assert jcs(-0.0) == "0"              # JSON.stringify(-0) === "0"
    assert jcs(1.0) == "1"               # integral floats drop the ".0"
    assert jcs(0.42) == "0.42"


def test_jcs_rejects_values_a_js_peer_cannot_hash_identically():
    with pytest.raises(TypeError):
        jcs(float("nan"))
    with pytest.raises(TypeError):
        jcs(math.inf)
    with pytest.raises(TypeError):
        jcs(2 ** 53)  # beyond Number.MAX_SAFE_INTEGER: JS would round
    with pytest.raises(TypeError):
        jcs({"payload": object()})


def test_jcs_escapes_strings_like_json_stringify():
    assert jcs('a"b\\c\n') == json.dumps('a"b\\c\n', ensure_ascii=False)
    assert jcs("\x07") == '"\\u0007"'
    assert jcs("é中") == '"é中"'  # non-ASCII stays literal


# ---------------------------------------------------------------------------
# Chain rules — cross-implementation golden vector (the §2.4 step-5 core)
# ---------------------------------------------------------------------------


def test_chain_golden_vector_byte_matches_the_server():
    # genesis = SHA-256 of the ddp:v1 domain-separation string.
    assert genesis_head(GOLDEN_SESSION, GOLDEN_STREAM).hex() == GOLDEN["genesis"]
    # First event hash, then every intermediate head, must equal the
    # values the server's independent implementation pins.
    assert event_hash(GOLDEN_EVENTS[0]).hex() == GOLDEN["event0Hash"]
    head = genesis_head(GOLDEN_SESSION, GOLDEN_STREAM)
    heads = []
    for event in GOLDEN_EVENTS:
        head = advance_head(head, event_hash(event))
        heads.append(head.hex())
    assert heads == [GOLDEN["head1"], GOLDEN["head2"], GOLDEN["head3"]]


def test_chain_mirror_tracks_per_seq_heads():
    mirror = ChainMirror(GOLDEN_SESSION, GOLDEN_STREAM)
    for event in GOLDEN_EVENTS:
        mirror.append(event)
    assert mirror.head_hex == GOLDEN["head3"]
    assert mirror.head_at(0) == GOLDEN["head1"]
    assert mirror.head_at(1) == GOLDEN["head2"]
    assert mirror.head_at(2) == GOLDEN["head3"]
    assert mirror.head_at(3) is None


def test_ppq_changes_the_event_hash_only_when_present():
    with_ppq = event_hash(GOLDEN_EVENTS[1])
    without = event_hash({**GOLDEN_EVENTS[1], "ppq": None})
    assert with_ppq != without


# ---------------------------------------------------------------------------
# Framings — exact bytes per 06 §2.3/§2.4
# ---------------------------------------------------------------------------


def test_framing_bytes_are_newline_delimited_utf8():
    assert receipt_message("sess_9f2c", "pod", 42, "9c" * 32, 1751871234890) == (
        b"ddp-receipt:v1\nsess_9f2c\npod\n42\n" + b"9c" * 32 + b"\n1751871234890"
    )
    assert key_cert_message("rk-1", "PUBB64", 100, 200) == (
        b"ddp-keycert:v1\nrk-1\nPUBB64\n100\n200"
    )


# ---------------------------------------------------------------------------
# §2.4 verification procedure with server-shaped key material
# ---------------------------------------------------------------------------


def _b64url(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def _keypair() -> tuple[Ed25519PrivateKey, str]:
    priv = Ed25519PrivateKey.generate()
    pub = priv.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)
    return priv, _b64url(pub)


class _Ledger:
    """Ephemeral key material + the server's exact signing behaviour
    (provenance.keys.ts keysDocument / signReceipt)."""

    def __init__(self, *, now_ms: int | None = None):
        self.now_ms = now_ms if now_ms is not None else int(time.time() * 1000)
        self.sealing_priv, self.sealing_pub = _keypair()
        self.receipt_priv, self.receipt_pub = _keypair()
        self.kid = "rk-testcafe"
        self.not_before = self.now_ms - 60_000
        self.not_after = self.now_ms + 48 * 3600_000

    def keys_document(self, *, tamper_cert: bool = False) -> dict:
        cert_sig = self.sealing_priv.sign(
            key_cert_message(
                self.kid, self.receipt_pub, self.not_before, self.not_after,
            )
        )
        if tamper_cert:
            cert_sig = bytes([cert_sig[0] ^ 0xFF]) + cert_sig[1:]
        return {
            "sealing_key": {
                "kid": "sk-testcafe", "alg": "Ed25519",
                "public_key": self.sealing_pub,
                "not_before": self.not_before,
                "not_after": self.not_after + 300 * 24 * 3600_000,
            },
            "receipt_keys": [{
                "kid": self.kid, "alg": "Ed25519",
                "public_key": self.receipt_pub,
                "not_before": self.not_before,
                "not_after": self.not_after,
                "cert_sig": _b64url(cert_sig),
            }],
        }

    def sign_receipt(
        self, session_id: str, stream: str, seq: int,
        chain_head: str, ts: int,
    ) -> dict:
        sig = self.receipt_priv.sign(
            receipt_message(session_id, stream, seq, chain_head, ts)
        )
        return {
            "seq": seq, "chain_head": chain_head, "ts": ts,
            "receipt_kid": self.kid, "sig": _b64url(sig),
        }


def _mirror_and_head() -> tuple[ChainMirror, str]:
    mirror = ChainMirror(GOLDEN_SESSION, GOLDEN_STREAM)
    for event in GOLDEN_EVENTS:
        mirror.append(event)
    return mirror, mirror.head_at(2)


def test_verify_accepts_a_server_shaped_receipt():
    ledger = _Ledger()
    verifier = ReceiptVerifier("http://unused.invalid/v1/keys")
    assert verifier.install_keys_document(ledger.keys_document()) == 1

    mirror, head2 = _mirror_and_head()
    receipt = ledger.sign_receipt(
        GOLDEN_SESSION, GOLDEN_STREAM, 2, head2, ledger.now_ms,
    )
    ok, reason = verifier.verify(
        session_id=GOLDEN_SESSION, stream=GOLDEN_STREAM,
        expected_head=mirror.head_at(receipt["seq"]), **receipt,
    )
    assert (ok, reason) == (True, None)


def test_verify_rejects_every_tampered_element():
    ledger = _Ledger()
    verifier = ReceiptVerifier("http://unused.invalid/v1/keys")
    verifier.install_keys_document(ledger.keys_document())
    mirror, head2 = _mirror_and_head()
    good = ledger.sign_receipt(
        GOLDEN_SESSION, GOLDEN_STREAM, 2, head2, ledger.now_ms,
    )

    _MIRROR = object()  # sentinel: "look the head up in the mirror"

    def check(receipt: dict, expected_head: object = _MIRROR) -> str:
        ok, reason = verifier.verify(
            session_id=GOLDEN_SESSION, stream=GOLDEN_STREAM,
            expected_head=(
                mirror.head_at(receipt["seq"])
                if expected_head is _MIRROR else expected_head
            ),
            **receipt,
        )
        assert not ok, f"tampered receipt must not verify: {receipt}"
        return reason

    # Tampered signature.
    bad_sig = _b64url(bytes(64))
    assert "signature" in check({**good, "sig": bad_sig})
    # Tampered chain_head: fails step 5 BEFORE any signature check —
    # a ledger claiming a different history is caught even if it signs it.
    forged_head = "0" * 64
    forged = ledger.sign_receipt(
        GOLDEN_SESSION, GOLDEN_STREAM, 2, forged_head, ledger.now_ms,
    )
    assert "chain_head mismatch" in check(forged)
    # Tampered ts: the framing no longer matches what was signed.
    assert "signature" in check({**good, "ts": good["ts"] + 1})
    # ts outside the receipt key's validity window (step 3).
    late_ts = ledger.not_after + 1000
    late = ledger.sign_receipt(
        GOLDEN_SESSION, GOLDEN_STREAM, 2, head2, late_ts,
    )
    assert "validity" in check(late)
    # Unknown kid.
    assert "kid" in check({**good, "receipt_kid": "rk-nope"})
    # A receipt for a seq we never submitted (no local head).
    assert "no local chain head" in check(good, expected_head=None)
    # Malformed fields never raise.
    assert "malformed" in check({**good, "seq": "2"})


def test_tampered_key_cert_disables_the_receipt_key():
    ledger = _Ledger()
    verifier = ReceiptVerifier("http://unused.invalid/v1/keys")
    # A cert_sig that does not verify with the sealing key must drop the
    # receipt key entirely (steps 1–2): receipts then fail on kid lookup.
    assert verifier.install_keys_document(
        ledger.keys_document(tamper_cert=True)
    ) == 0
    mirror, head2 = _mirror_and_head()
    receipt = ledger.sign_receipt(
        GOLDEN_SESSION, GOLDEN_STREAM, 2, head2, ledger.now_ms,
    )
    ok, reason = verifier.verify(
        session_id=GOLDEN_SESSION, stream=GOLDEN_STREAM,
        expected_head=mirror.head_at(2), **receipt,
    )
    assert not ok
    assert "kid" in reason


# ---------------------------------------------------------------------------
# LedgerClient wiring: verify receipts as they arrive, fail-open
# ---------------------------------------------------------------------------


class _SigningLedgerServer:
    """A stub ledger that really chains (§2.1 rules) and really signs
    (§2.3 receipts, §2.4 key document). ``tamper`` switches it to
    returning receipts for a forged history."""

    def __init__(self, *, tamper: str | None = None):
        self.ledger = _Ledger()
        self.tamper = tamper
        outer = self
        heads: dict[str, bytes] = {}

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):  # silence
                pass

            def _send(self, payload: dict):
                raw = json.dumps(payload).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

            def do_GET(self):
                if not self.path.endswith("/keys"):
                    self.send_error(404)
                    return
                self._send(outer.ledger.keys_document())

            def do_POST(self):
                session_id = self.path.split("/sessions/")[1].split("/")[0]
                length = int(self.headers.get("Content-Length", 0))
                body = json.loads(self.rfile.read(length))
                head = heads.get(session_id) or genesis_head(session_id, "pod")
                receipts = []
                for event in body["events"]:
                    head = advance_head(head, event_hash(event))
                    if event["type"].startswith("slice."):
                        chain_head = head.hex()
                        if outer.tamper == "chain_head":
                            chain_head = "f" * 64
                        receipt = outer.ledger.sign_receipt(
                            session_id, "pod", event["seq"], chain_head,
                            outer.ledger.now_ms,
                        )
                        if outer.tamper == "sig":
                            receipt["sig"] = _b64url(bytes(64))
                        receipts.append(receipt)
                heads[session_id] = head
                self._send({
                    "stream": "pod", "chain_head": head.hex(),
                    "receipts": receipts,
                })

        self._server = HTTPServer(("127.0.0.1", 0), Handler)
        self._thread = threading.Thread(
            target=self._server.serve_forever, daemon=True,
        )
        self._thread.start()

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self._server.server_address[1]}/v1"

    def close(self):
        self._server.shutdown()
        self._server.server_close()


def _run_session(server: _SigningLedgerServer, session_id: str) -> LedgerClient:
    client = LedgerClient(
        base_url=server.base_url, session_id=session_id, token="dlt_pod_x",
    )
    try:
        client.post_event("session.config", {"model": "acestep-1.5"}, ts=1000)
        client.post_event(
            "action.param", {"name": "cutoff", "value": 0.42},
            ts=1001, ppq=3840.25,
        )
        client.post_slice_hash(
            sha256="cd" * 32, start_sample=0, num_samples=96000,
            channels=2, slice_seq=0, ts=1002,
        )
        client.post_slice_hash(
            sha256="ef" * 32, start_sample=96000, num_samples=96000,
            channels=2, slice_seq=1, ts=1003,
        )
        client.close(timeout=5.0)
    finally:
        server.close()
    return client


def test_ledger_client_verifies_receipts_end_to_end():
    server = _SigningLedgerServer()
    client = _run_session(server, "sess_e2e_ok")
    # Both slice receipts arrived and passed the full §2.4 procedure —
    # including step 5 against the client's own chain mirror.
    assert client.receipts_verified == 2
    assert client.receipts_unverified == 0
    assert client.last_verification_failure is None


def test_ledger_client_counts_forged_chain_head_receipts():
    server = _SigningLedgerServer(tamper="chain_head")
    client = _run_session(server, "sess_e2e_forged")
    # The ledger signed a history that is not ours: step 5 must catch it,
    # fail-open (reporting kept running; nothing raised).
    assert client.receipts_verified == 0
    assert client.receipts_unverified == 2
    assert "chain_head mismatch" in client.last_verification_failure


def test_ledger_client_counts_bad_signature_receipts():
    server = _SigningLedgerServer(tamper="sig")
    client = _run_session(server, "sess_e2e_badsig")
    assert client.receipts_verified == 0
    assert client.receipts_unverified == 2
    assert "signature" in client.last_verification_failure


def test_ledger_client_fails_open_when_keys_endpoint_is_down():
    # Server chains + signs correctly but the key document is
    # unreachable (404) → the verifier cannot certify any receipt key;
    # receipts count as unverified and the reporter keeps working.
    server = _SigningLedgerServer()
    client = LedgerClient(
        base_url=server.base_url, session_id="sess_e2e_nokeys",
        token="dlt_pod_x",
    )
    # Pre-seed the lazily built verifier with a keys URL that 404s.
    client._verifier = ReceiptVerifier(f"{server.base_url}/nope-keys")
    try:
        client.post_slice_hash(
            sha256="ab" * 32, start_sample=0, num_samples=96000,
            channels=2, slice_seq=0, ts=1002,
        )
        client.close(timeout=5.0)
    finally:
        server.close()
    assert client.receipts_verified == 0
    assert client.receipts_unverified == 1
    assert "kid" in client.last_verification_failure
