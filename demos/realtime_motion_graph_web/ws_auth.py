"""Session tokens on the WebSocket handshake.

A deployment that puts a gate in front of the server (a queue, a seat
service) can hand each client a short-lived token and have the server
refuse sockets without one. The token rides on the socket URL as
``?t=<token>``:

    token   = <payload>.<sig>
    payload = <id>|<expires_at_ms>|<nonce>
    sig     = base64url(HMAC-SHA256(secret, payload)), no padding

Configured from the environment:

- ``DEMON_SESSION_TOKEN_SECRET``: the shared secret. Unset: no checking.
- ``DEMON_SESSION_TOKEN_MODE``: ``off``, ``log`` (the default when a secret
  is set: every socket is let in, and one without a valid token is logged
  with the reason) or ``enforce`` (such a socket is refused with 401).

Tool routes (``/tools/<name>/...``) need ``X-Demon-Session`` naming a live
session, and a session only exists after a socket got in, so enforcing here
gates them too. Torch-free.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import os
import time
import urllib.parse

MODES = ("off", "log", "enforce")


def _b64url(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode()


def sign(secret: str, token_id: str, expires_at_ms: int, nonce: str = "n") -> str:
    """A token in the format :func:`check` accepts (tests, local tools)."""
    payload = f"{token_id}|{expires_at_ms}|{nonce}"
    sig = _b64url(hmac.new(secret.encode(), payload.encode(), hashlib.sha256).digest())
    return f"{payload}.{sig}"


def check(token: str | None, secret: str, now_ms: int | None = None) -> str | None:
    """``None`` when ``token`` is valid for ``secret``, else the reason:
    ``missing``, ``malformed``, ``bad_signature`` or ``expired``."""
    if not token:
        return "missing"
    payload, dot, sig = token.rpartition(".")
    parts = payload.split("|")
    if not dot or len(parts) != 3 or not parts[1].isdigit():
        return "malformed"
    want = _b64url(hmac.new(secret.encode(), payload.encode(), hashlib.sha256).digest())
    if not hmac.compare_digest(want, sig):
        return "bad_signature"
    if int(parts[1]) <= (time.time() * 1000 if now_ms is None else now_ms):
        return "expired"
    return None


def config(env=os.environ) -> tuple[str, str]:
    """``(mode, secret)``. No secret means ``off`` whatever the mode says;
    an unknown mode is refused at startup rather than silently ignored."""
    secret = env.get("DEMON_SESSION_TOKEN_SECRET", "").strip()
    mode = env.get("DEMON_SESSION_TOKEN_MODE", "").strip().lower() or "log"
    if mode not in MODES:
        raise ValueError(f"DEMON_SESSION_TOKEN_MODE must be one of {MODES}, got {mode!r}")
    return ("off" if not secret else mode), secret


def handshake_refusal(path: str, mode: str, secret: str, remote: str, log) -> str | None:
    """For a socket upgrade on ``path``: the reason to refuse it (enforce
    mode), else ``None``. In log mode a bad token is logged and let in."""
    if mode == "off":
        return None
    query = urllib.parse.parse_qs(urllib.parse.urlsplit(path).query)
    reason = check((query.get("t") or [None])[0], secret)
    if reason is None:
        return None
    log("ws_token_rejected mode={} reason={} remote={}", mode, reason, remote)
    return reason if mode == "enforce" else None
