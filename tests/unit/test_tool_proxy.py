"""/tools/<name>/ proxy: config parsing, session auth, forwarding, health,
exercised through a real websockets server wrapped by tool_proxy."""

from __future__ import annotations

import http.client
import json
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from websockets.datastructures import Headers
from websockets.http11 import Response
from websockets.sync.server import serve as ws_serve

from acestep.streaming import registry
from demos.realtime_motion_graph_web import tool_proxy


class _Echo(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _reply(self):
        body = self.rfile.read(int(self.headers.get("Content-Length") or 0))
        if self.path == "/health":
            out, status = b"ok", 200
        else:
            out, status = json.dumps({
                "method": self.command, "path": self.path, "len": len(body),
                "ctype": self.headers.get("Content-Type"),
                "session": self.headers.get(tool_proxy.SESSION_HEADER),
                "head": body[:8].decode("latin1"),
            }).encode(), 201
        self.send_response(status)
        self.send_header("Content-Type", "application/x-echo")
        self.send_header("Content-Length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)

    do_GET = do_POST = _reply


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def env():
    target = ThreadingHTTPServer(("127.0.0.1", 0), _Echo)
    threading.Thread(target=target.serve_forever, daemon=True).start()
    dead_port = _free_port()

    def process_request(conn, req):
        return Response(200, "OK", Headers([("Content-Length", "2")]), b"ws")

    srv = ws_serve(lambda ws: None, "127.0.0.1", 0, process_request=process_request)
    tool_proxy.configure(
        {"echo": ("127.0.0.1", target.server_address[1]),
         "dead": ("127.0.0.1", dead_port)},
        max_body=1024 * 1024, timeout_s=5, poll_health=False,
    )
    tool_proxy.wrap_server(srv)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    registry.register(registry.SessionHandle("live-sid", time.time(),
                                             lambda d, a: None, lambda: {}))
    yield srv.socket.getsockname()[1]
    registry.unregister("live-sid")
    tool_proxy.configure({}, poll_health=False)
    srv.shutdown()
    target.shutdown()


def _req(port, method, path, body=None, headers=None):
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    c.request(method, path, body=body, headers=headers or {})
    r = c.getresponse()
    out = (r.status, r.getheader("Content-Type"), r.read())
    c.close()
    return out


SID = {tool_proxy.SESSION_HEADER: "live-sid"}


def test_forwards_body_query_status_and_type(env):
    body = b"RIFF" + b"\0" * 300_000
    status, ctype, out = _req(env, "POST", "/tools/echo/v1/run?x=1", body,
                              {**SID, "Content-Type": "audio/wav"})
    assert status == 201 and ctype == "application/x-echo"
    got = json.loads(out)
    assert got == {"method": "POST", "path": "/v1/run?x=1", "len": len(body),
                   "ctype": "audio/wav", "session": None, "head": "RIFF\0\0\0\0"}


def test_get_with_valid_session(env):
    status, _, out = _req(env, "GET", "/tools/echo/info", headers=SID)
    assert status == 201 and json.loads(out)["method"] == "GET"


def test_unknown_tool_404(env):
    assert _req(env, "GET", "/tools/nope/x", headers=SID)[0] == 404


def test_target_down_503(env):
    status, ctype, out = _req(env, "POST", "/tools/dead/x", b"{}", SID)
    assert status == 503 and ctype.startswith("application/json")
    assert json.loads(out)["error"]


@pytest.mark.parametrize("headers", [{}, {tool_proxy.SESSION_HEADER: "bogus"}])
def test_missing_or_invalid_session_401(env, headers):
    assert _req(env, "POST", "/tools/echo/x", b"{}", headers)[0] == 401
    assert _req(env, "GET", "/tools/echo/x", headers=headers)[0] == 401


def test_localhost_health_bypass(env):
    status, _, out = _req(env, "GET", "/tools/echo/health")
    assert (status, out) == (200, b"ok")


def test_body_cap_413_before_reading_body(env):
    # Headers only: the cap is enforced from Content-Length, before the
    # (never sent) body is read.
    with socket.create_connection(("127.0.0.1", env), timeout=10) as s:
        s.sendall(b"POST /tools/echo/x HTTP/1.1\r\nHost: x\r\n"
                  b"X-Demon-Session: live-sid\r\n"
                  b"Content-Length: 1048577\r\n\r\n")
        assert s.recv(64).startswith(b"HTTP/1.0 413")


def test_other_routes_still_reach_websockets(env):
    assert _req(env, "GET", "/api/server-info")[2] == b"ws"


@pytest.mark.parametrize("spec", [
    "a=http://10.0.0.1:1330", "a=https://127.0.0.1:1330",
    "a=http://example.com:80", "a=http://127.0.0.1.evil.com:1",
    "a=http://u:p@127.0.0.1:1", "a=http://127.0.0.1:1/path",
    "Bad=http://127.0.0.1:1", "a=http://127.0.0.1:1,a=http://127.0.0.1:2",
    "noequals",
])
def test_config_rejects(spec):
    with pytest.raises(ValueError):
        tool_proxy.parse_tool_proxies(spec)


def test_config_parses():
    assert tool_proxy.parse_tool_proxies(
        " gain=http://127.0.0.1:1330 , x-2=http://localhost:1331/ ") == {
        "gain": ("127.0.0.1", 1330), "x-2": ("localhost", 1331)}
    assert tool_proxy.parse_tool_proxies("") == {}


def test_healthy_tools_lists_only_healthy(env):
    assert tool_proxy.healthy_tools() == []
    tool_proxy.probe_health()
    assert tool_proxy.healthy_tools() == ["echo"]
