"""Client-side ledger receipt verification (spec 06 §2.4).

The Provenance Ledger acknowledges every slice-hash event with a signed
receipt (spec 06 §2.3). A receipt is only worth something if the client
runs the full §2.4 verification procedure — in particular **step 5**:
recompute the chain head from *your own* submitted events and require
the receipt to commit to exactly that head. Signature checks alone only
prove the ledger signed *something*; step 5 proves it signed *your
history*.

Three pieces, mirroring the server implementation byte-for-byte
(``provenance.keys.ts`` / ``chain.ts`` / ``jcs.ts``):

- :func:`jcs` — RFC 8785 canonical JSON with ECMAScript
  ``JSON.stringify`` number/string formatting, the byte layer under
  every ledger hash (spec 06 §0).
- :class:`ChainMirror` — the client's local replica of the §2.1 chain
  rules (``event_hash = SHA-256(JCS(event))``, genesis domain-separation
  head, ``head_n = SHA-256(head_{n-1} || event_hash_n)``), tracking the
  head after every submitted seq.
- :class:`ReceiptVerifier` — fetches + caches ``GET {base}/keys``
  (spec 06 §2.4), validates the key-cert chain (sealing pubkey →
  ``ddp-keycert:v1`` cert over the receipt key), and verifies each
  receipt's Ed25519 signature over the ``ddp-receipt:v1`` framing plus
  the step-5 chain-head equality.

Number-formatting caveats (JCS = ECMAScript formatting, not Python's):

- Integers are serialized as plain digits; magnitudes ``>= 2**53``
  raise ``TypeError`` because a JS peer cannot represent them exactly
  (the server would hash a rounded value and heads would diverge).
- Floats follow the ECMAScript Number-to-string algorithm: integral
  floats drop the ``.0`` (``1.0`` → ``"1"``), ``-0.0`` → ``"0"``,
  fixed notation for decimal exponents in ``(-7, 21]`` and exponential
  (``1e-7``, ``1e+21``) outside it. Python and JS both emit
  shortest-round-trip digits for IEEE-754 doubles, so digit sequences
  match; only the notation had to be ported.
- Non-finite numbers and non-JSON types raise ``TypeError``. Callers
  are fail-open: a payload the mirror cannot canonicalize marks the
  receipt unverifiable, it never breaks streaming.

Requires the optional ``cryptography`` dependency (``provenance``
extra) for signature checks; without it verification degrades to a
logged "unverified" outcome, never an exception.
"""

from __future__ import annotations

import base64
import hashlib
import json
import math
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Optional

from loguru import logger

__all__ = [
    "jcs",
    "event_hash",
    "genesis_head",
    "advance_head",
    "receipt_message",
    "key_cert_message",
    "ChainMirror",
    "ReceiptKey",
    "ReceiptVerifier",
]

_REQUEST_TIMEOUT_S = 5.0
# Floor between key-document fetch attempts (receipt keys rotate daily,
# spec 06 §2.4; a fetch happens only when a receipt names an unknown
# kid) so a broken endpoint is not hammered from the flush path.
_KEYS_RETRY_MIN_S = 60.0

_MAX_SAFE_INT = 2 ** 53


# ---------------------------------------------------------------------------
# RFC 8785 (JCS) canonicalization — port of the server's jcs.ts
# ---------------------------------------------------------------------------


def _format_float(x: float) -> str:
    """ECMAScript Number-to-string (ECMA-262 §6.1.6.1.20) for a nonzero,
    finite, non-integral-shortcut float. Python's ``repr`` supplies the
    same shortest-round-trip digit sequence ES computes; this reshapes
    it into ES notation (fixed for decimal exponent in (-7, 21],
    ``d[.ddd]e±N`` outside)."""
    sign = "-" if x < 0 else ""
    r = repr(abs(x))
    if "e" in r:
        mantissa, _, exp_s = r.partition("e")
        whole, _, frac = mantissa.partition(".")
        digits = (whole + frac).rstrip("0")
        n = int(exp_s) + len(whole)
    else:
        whole, _, frac = r.partition(".")
        combined = whole + frac
        stripped = combined.lstrip("0")
        n = len(whole) - (len(combined) - len(stripped))
        digits = stripped.rstrip("0")
    k = len(digits)
    if k <= n <= 21:
        body = digits + "0" * (n - k)
    elif 0 < n <= 21:
        body = digits[:n] + "." + digits[n:]
    elif -6 < n <= 0:
        body = "0." + "0" * (-n) + digits
    else:
        e = n - 1
        exp_part = f"e+{e}" if e >= 0 else f"e-{-e}"
        body = digits[0] + ("." + digits[1:] if k > 1 else "") + exp_part
    return sign + body


def _format_number(value: int | float) -> str:
    if isinstance(value, int):
        if abs(value) >= _MAX_SAFE_INT:
            raise TypeError(
                f"JCS: integer {value} exceeds the JS safe-integer range "
                "(a JS peer would hash a rounded value)"
            )
        return str(value)
    if math.isnan(value) or math.isinf(value):
        raise TypeError("JCS: non-finite numbers are not serializable")
    if value == 0.0:
        return "0"  # covers -0.0: JSON.stringify(-0) === "0"
    if value.is_integer() and abs(value) < _MAX_SAFE_INT:
        return str(int(value))  # exact double → same digits, no ".0"
    return _format_float(value)


def jcs(value: object) -> str:
    """RFC 8785 canonical JSON: keys sorted by UTF-16 code units,
    numbers/strings serialized exactly as ECMAScript ``JSON.stringify``
    (spec 06 §0). Byte-matches the server's ``jcs.ts``."""
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return _format_number(value)
    if isinstance(value, str):
        # json.dumps escapes exactly the JSON.stringify set: `"`, `\`,
        # and control chars (short escapes for \b \t \n \f \r).
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, (list, tuple)):
        return "[" + ",".join(jcs(v) for v in value) + "]"
    if isinstance(value, dict):
        parts = []
        # UTF-16BE byte order == UTF-16 code-unit order (JCS sort rule;
        # differs from code-point order only above the BMP).
        for key in sorted(value, key=lambda s: s.encode("utf-16-be")):
            if not isinstance(key, str):
                raise TypeError("JCS: object keys must be strings")
            parts.append(
                json.dumps(key, ensure_ascii=False) + ":" + jcs(value[key])
            )
        return "{" + ",".join(parts) + "}"
    raise TypeError(f"JCS: cannot serialize a {type(value).__name__}")


# ---------------------------------------------------------------------------
# Chain rules (spec 06 §2.1) — port of the server's chain.ts
# ---------------------------------------------------------------------------


def event_hash(event: dict) -> bytes:
    """``event_hash = SHA-256(JCS(event))`` over the client-supplied
    fields only — ``ppq`` enters the canonical form only when the
    reporter sent it, matching the server's ``eventHash``."""
    canonical: dict = {
        "seq": event["seq"],
        "type": event["type"],
        "ts": event["ts"],
        "payload": event["payload"],
    }
    if event.get("ppq") is not None:
        canonical["ppq"] = event["ppq"]
    return hashlib.sha256(jcs(canonical).encode("utf-8")).digest()


def genesis_head(session_id: str, stream: str) -> bytes:
    """``head_0 = SHA-256(UTF8("ddp:v1:" + sessionId + ":" + stream))``."""
    return hashlib.sha256(
        f"ddp:v1:{session_id}:{stream}".encode("utf-8")
    ).digest()


def advance_head(prev_head: bytes, ev_hash: bytes) -> bytes:
    """``head_n = SHA-256(head_{n-1} || event_hash_n)`` — raw digests."""
    return hashlib.sha256(prev_head + ev_hash).digest()


class ChainMirror:
    """Local replica of one stream's hash chain, fed with the exact
    event objects the reporter submits. ``head_at(seq)`` is the step-5
    reference value a receipt for that seq must commit to."""

    # Receipts arrive in the same response as the flush that produced
    # them, so only a recent window of per-seq heads is ever consulted.
    _HEAD_WINDOW = 8192

    def __init__(self, session_id: str, stream: str) -> None:
        self.session_id = session_id
        self.stream = stream
        self._head = genesis_head(session_id, stream)
        self._heads: dict[int, str] = {}

    @property
    def head_hex(self) -> str:
        """Current chain head (lowercase hex64)."""
        return self._head.hex()

    def append(self, event: dict) -> str:
        """Advance the chain with one submitted event; returns the new
        head hex. Raises ``TypeError`` on payloads a JS peer could not
        hash identically (caller treats that as fail-open)."""
        self._head = advance_head(self._head, event_hash(event))
        head_hex = self._head.hex()
        seq = event["seq"]
        self._heads[seq] = head_hex
        stale = seq - self._HEAD_WINDOW
        if stale in self._heads:
            del self._heads[stale]
        return head_hex

    def head_at(self, seq: int) -> Optional[str]:
        """Chain head after the event at ``seq``, if still windowed."""
        return self._heads.get(seq)


# ---------------------------------------------------------------------------
# Signed-message framings (exact bytes per 06 §2.3/§2.4)
# ---------------------------------------------------------------------------


def receipt_message(
    session_id: str, stream: str, seq: int, chain_head_hex: str, ts: int,
) -> bytes:
    """06 §2.3 — per-slice receipt framing (receipt key signs this)."""
    return (
        f"ddp-receipt:v1\n{session_id}\n{stream}\n{seq}"
        f"\n{chain_head_hex}\n{ts}"
    ).encode("utf-8")


def key_cert_message(
    kid: str, public_key_b64url: str, not_before: int, not_after: int,
) -> bytes:
    """06 §2.4 — receipt-key certification framing (sealing key signs
    this)."""
    return (
        f"ddp-keycert:v1\n{kid}\n{public_key_b64url}"
        f"\n{not_before}\n{not_after}"
    ).encode("utf-8")


# ---------------------------------------------------------------------------
# Key document + receipt verification (spec 06 §2.4)
# ---------------------------------------------------------------------------


def _b64url_decode(s: str) -> bytes:
    return base64.urlsafe_b64decode(s + "=" * (-len(s) % 4))


def _verify_ed25519(public_key_raw: bytes, sig: bytes, message: bytes) -> bool:
    """Raw Ed25519 verify; ``ImportError`` propagates so the caller can
    report "cryptography unavailable" distinctly."""
    from cryptography.exceptions import InvalidSignature
    from cryptography.hazmat.primitives.asymmetric.ed25519 import (
        Ed25519PublicKey,
    )

    try:
        Ed25519PublicKey.from_public_bytes(public_key_raw).verify(sig, message)
        return True
    except InvalidSignature:
        return False


@dataclass(frozen=True)
class ReceiptKey:
    """One certified entry of the ``receipt_keys`` list, post key-cert
    validation (steps 1–2 of the §2.4 procedure)."""

    kid: str
    public_key_raw: bytes
    not_before: int
    not_after: int


class ReceiptVerifier:
    """Runs the 06 §2.4 verification procedure for one ledger base URL.

    Fail-open by contract: :meth:`verify` never raises — every failure
    mode (network, missing ``cryptography``, malformed documents, bad
    signatures) comes back as ``(False, reason)``.
    """

    def __init__(self, keys_url: str) -> None:
        self.keys_url = keys_url
        self._keys: dict[str, ReceiptKey] = {}
        self._last_attempt: float = 0.0
        self._fetch_warned = False

    # ---- key document ----------------------------------------------------

    def install_keys_document(self, doc: dict) -> int:
        """Validate a ``GET /v1/keys`` document (steps 1–2): every
        receipt key must carry a ``cert_sig`` by the sealing key over
        the ``ddp-keycert:v1`` framing. Keys failing certification are
        dropped with a warning. Returns the number installed."""
        keys: dict[str, ReceiptKey] = {}
        try:
            sealing = doc["sealing_key"]
            sealing_raw = _b64url_decode(sealing["public_key"])
            entries = doc.get("receipt_keys") or []
        except (KeyError, TypeError, ValueError) as exc:
            logger.warning("ledger keys document malformed: {}", exc)
            self._keys = {}
            return 0
        for entry in entries:
            try:
                kid = entry["kid"]
                public_key = entry["public_key"]
                not_before = int(entry["not_before"])
                not_after = int(entry["not_after"])
                cert_sig = _b64url_decode(entry["cert_sig"])
                certified = _verify_ed25519(
                    sealing_raw,
                    cert_sig,
                    key_cert_message(kid, public_key, not_before, not_after),
                )
            except ImportError:
                logger.warning(
                    "ledger receipt verification disabled: `cryptography` "
                    "is not installed (install the `provenance` extra)"
                )
                self._keys = {}
                return 0
            except (KeyError, TypeError, ValueError) as exc:
                logger.warning("ledger receipt-key entry malformed: {}", exc)
                continue
            if not certified:
                logger.warning(
                    "ledger receipt key kid={} failed key-cert validation "
                    "(cert_sig does not verify with the sealing key) — "
                    "dropping it",
                    entry.get("kid"),
                )
                continue
            keys[kid] = ReceiptKey(
                kid=kid,
                public_key_raw=_b64url_decode(public_key),
                not_before=not_before,
                not_after=not_after,
            )
        self._keys = keys
        return len(keys)

    def _fetch_keys(self) -> None:
        now = time.monotonic()
        if now - self._last_attempt < _KEYS_RETRY_MIN_S:
            return
        self._last_attempt = now
        try:
            req = urllib.request.Request(self.keys_url, method="GET")
            with urllib.request.urlopen(
                req, timeout=_REQUEST_TIMEOUT_S,
            ) as resp:
                doc = json.loads(resp.read())
        except (urllib.error.URLError, OSError, ValueError) as exc:
            if not self._fetch_warned:
                self._fetch_warned = True
                logger.warning(
                    "ledger keys fetch failed url={} error={} "
                    "(receipts will count as unverified until it succeeds)",
                    self.keys_url, exc,
                )
            return
        if isinstance(doc, dict):
            self.install_keys_document(doc)

    def _receipt_key(self, kid: str) -> Optional[ReceiptKey]:
        if kid not in self._keys:
            # Unknown kid → first receipt or a daily rotation; refetch
            # (rate-floored inside _fetch_keys).
            self._fetch_keys()
        return self._keys.get(kid)

    # ---- the §2.4 procedure ----------------------------------------------

    def verify(
        self,
        *,
        session_id: str,
        stream: str,
        seq: object,
        chain_head: object,
        ts: object,
        receipt_kid: object,
        sig: object,
        expected_head: Optional[str],
    ) -> tuple[bool, Optional[str]]:
        """Verify one §2.3 receipt against the local chain mirror.

        ``expected_head`` is the caller's own recomputed chain head at
        ``seq`` (step 5). Returns ``(True, None)`` or
        ``(False, reason)``; never raises."""
        try:
            if (
                not isinstance(seq, int)
                or not isinstance(chain_head, str)
                or not isinstance(ts, int)
                or not isinstance(receipt_kid, str)
                or not isinstance(sig, str)
            ):
                return False, "malformed receipt (missing/mistyped fields)"
            # Step 5 first: it needs no keys and is the check that makes
            # the receipt a commitment to *our* history.
            if expected_head is None:
                return False, (
                    f"no local chain head for seq={seq} "
                    "(mirror gap or out-of-window receipt)"
                )
            if chain_head != expected_head:
                return False, (
                    f"chain_head mismatch at seq={seq}: receipt commits to "
                    f"{chain_head[:16]}… but local history gives "
                    f"{expected_head[:16]}…"
                )
            # Steps 1–2: certified receipt key from the key document.
            key = self._receipt_key(receipt_kid)
            if key is None:
                return False, (
                    f"no certified receipt key for kid={receipt_kid}"
                )
            # Step 3: receipt ts inside the key's validity window.
            if not (key.not_before <= ts <= key.not_after):
                return False, (
                    f"receipt ts={ts} outside key validity "
                    f"[{key.not_before}, {key.not_after}] for "
                    f"kid={receipt_kid}"
                )
            # Step 4: Ed25519 over the exact ddp-receipt:v1 framing.
            try:
                sig_raw = _b64url_decode(sig)
            except (ValueError, TypeError):
                return False, "receipt sig is not valid base64url"
            if not _verify_ed25519(
                key.public_key_raw,
                sig_raw,
                receipt_message(session_id, stream, seq, chain_head, ts),
            ):
                return False, (
                    f"receipt signature invalid for seq={seq} "
                    f"kid={receipt_kid}"
                )
            return True, None
        except ImportError:
            return False, (
                "cryptography is not installed (provenance extra) — "
                "cannot verify receipts"
            )
        except Exception as exc:  # noqa: BLE001 — fail-open by contract
            return False, f"receipt verification error: {exc}"
