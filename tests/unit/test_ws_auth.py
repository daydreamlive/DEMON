"""Session token on the WebSocket handshake (demos/realtime_motion_graph_web/ws_auth.py)."""

import pytest

from demos.realtime_motion_graph_web import server, ws_auth

SECRET = "s3cret"
# Made with Node: createHmac("sha256", "s3cret").update(payload).digest("base64url").
NODE_TOKEN = "sess-1|4102444800000|abcd.g8fVPfFm5uYYh5FBjLJHPKpB_e4FQTIV5CL89l5UpRk"


def test_check_reasons():
    assert ws_auth.check(NODE_TOKEN, SECRET) is None
    assert ws_auth.check(ws_auth.sign(SECRET, "x", 2_000), SECRET, now_ms=1_000) is None
    assert ws_auth.check(None, SECRET) == "missing"
    assert ws_auth.check("nope", SECRET) == "malformed"
    assert ws_auth.check("a|b|c.sig", SECRET) == "malformed"
    assert ws_auth.check(NODE_TOKEN, "other") == "bad_signature"
    assert ws_auth.check(NODE_TOKEN.replace("sess-1", "sess-2"), SECRET) == "bad_signature"
    assert ws_auth.check(ws_auth.sign(SECRET, "x", 1_000), SECRET, now_ms=1_000) == "expired"


def test_config():
    assert ws_auth.config({}) == ("off", "")
    assert ws_auth.config({"DEMON_SESSION_TOKEN_MODE": "enforce"}) == ("off", "")
    assert ws_auth.config({"DEMON_SESSION_TOKEN_SECRET": "k"}) == ("log", "k")
    assert ws_auth.config({"DEMON_SESSION_TOKEN_SECRET": "k",
                           "DEMON_SESSION_TOKEN_MODE": "Enforce"}) == ("enforce", "k")
    with pytest.raises(ValueError):
        ws_auth.config({"DEMON_SESSION_TOKEN_SECRET": "k", "DEMON_SESSION_TOKEN_MODE": "strict"})


def _upgrade(path, headers=None):
    conn = type("Conn", (), {"remote_address": ("127.0.0.1", 1)})()
    req = type("Req", (), {"path": path,
                           "headers": {"Upgrade": "websocket", **(headers or {})}})()
    return server._process_request(conn, req)


@pytest.mark.parametrize("mode", ["off", "log"])
def test_handshake_lets_in_without_enforce(monkeypatch, mode):
    monkeypatch.setattr(server, "_WS_AUTH", (mode, SECRET if mode != "off" else ""))
    assert _upgrade("/ws") is None
    assert _upgrade("/ws?t=garbage") is None


def test_handshake_enforce(monkeypatch):
    monkeypatch.setattr(server, "_WS_AUTH", ("enforce", SECRET))
    seen = []
    monkeypatch.setattr(server.logger, "warning", lambda *a: seen.append(a))
    assert _upgrade("/ws?t=" + NODE_TOKEN.replace("|", "%7C")) is None
    assert _upgrade(f"/ws?model=x&t={NODE_TOKEN}") is None
    res = _upgrade("/ws", {"CF-Connecting-IP": "203.0.113.9"})
    assert res.status_code == 401 and b"missing" in bytes(res.body)
    assert _upgrade("/ws?t=" + ws_auth.sign(SECRET, "x", 1)).status_code == 401
    assert seen[0][1:] == ("enforce", "missing", "203.0.113.9")
