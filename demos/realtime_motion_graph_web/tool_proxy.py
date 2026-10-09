"""Forward ``/tools/<name>/<path>`` on the main port to localhost services.

Deployments can run auxiliary model services on the same machine, bound
to loopback. Only the DEMON port is reachable from outside, so this
module forwards ``GET``/``POST /tools/<name>/<path...>`` to the
configured target -- but only for callers holding a live session on this
server (``X-Demon-Session: <ready.session_id>``). A loopback caller may
hit ``GET /tools/<name>/health`` without the header (local health checks).

Configured with ``DEMON_TOOL_PROXIES`` / ``--tool-proxies``
(``name=http://127.0.0.1:PORT,...``). Targets must be http on
127.0.0.1/localhost.

The websockets HTTP parser refuses request bodies, so tool requests are
picked off the socket before websockets sees it (:func:`wrap_server`) and
served by a stdlib ``BaseHTTPRequestHandler``. Torch-free.
"""

from __future__ import annotations

import http.client
import json
import os
import re
import shutil
import socket
import threading
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler

from acestep.streaming import registry as session_registry

SESSION_HEADER = "X-Demon-Session"
_NAME_RE = re.compile(r"[a-z0-9-]{1,32}")
_LOCAL_HOSTS = {"127.0.0.1", "localhost"}
_LOOPBACK_CLIENTS = {"127.0.0.1", "::1", "::ffff:127.0.0.1"}
_TOOL_PREFIXES = (b"GET /tools/", b"POST /tools/")
CONNECT_TIMEOUT_S = 5.0
HEALTH_POLL_S = 10.0
HEALTH_FRESH_S = 30.0

# Set by configure(); module-global like the session registry.
_proxies: dict[str, tuple[str, int]] = {}
_max_body = 256 * 1024 * 1024
_timeout_s = 120.0
_last_healthy: dict[str, float] = {}


def parse_tool_proxies(spec: str) -> dict[str, tuple[str, int]]:
    """``"a=http://127.0.0.1:1330,b=http://localhost:1331"`` ->
    ``{"a": ("127.0.0.1", 1330), ...}``. Raises ValueError on anything
    that isn't a named http loopback target."""
    out: dict[str, tuple[str, int]] = {}
    for item in filter(None, (s.strip() for s in spec.split(","))):
        name, sep, url = item.partition("=")
        name, url = name.strip(), url.strip()
        if not sep or not _NAME_RE.fullmatch(name):
            raise ValueError(f"tool name must match [a-z0-9-]{{1,32}}: {item!r}")
        if name in out:
            raise ValueError(f"duplicate tool name: {name!r}")
        u = urllib.parse.urlsplit(url)
        if (
            u.scheme != "http" or u.hostname not in _LOCAL_HOSTS
            or u.username or u.password or u.path not in ("", "/")
            or u.query or u.fragment
        ):
            raise ValueError(
                f"tool {name!r} target must be http://127.0.0.1:<port> "
                f"or http://localhost:<port>, got {url!r}"
            )
        out[name] = (u.hostname, u.port or 80)
    return out


def configure(proxies: dict[str, tuple[str, int]], *,
              max_body: int | None = None,
              timeout_s: float | None = None,
              poll_health: bool = True) -> None:
    global _proxies, _max_body, _timeout_s
    _proxies = dict(proxies)
    _last_healthy.clear()
    if max_body is not None:
        _max_body = max_body
    if timeout_s is not None:
        _timeout_s = timeout_s
    if _proxies and poll_health:
        threading.Thread(target=_health_loop, daemon=True,
                         name="tool-proxy-health").start()


def configure_from_env(cli_spec: str | None = None,
                       env=os.environ) -> dict[str, tuple[str, int]]:
    """``--tool-proxies`` (``cli_spec``, wins) or ``DEMON_TOOL_PROXIES``,
    plus ``DEMON_TOOL_PROXY_MAX_BODY_MB`` / ``DEMON_TOOL_PROXY_TIMEOUT_S``."""
    spec = cli_spec if cli_spec is not None else env.get("DEMON_TOOL_PROXIES", "")
    proxies = parse_tool_proxies(spec)
    configure(
        proxies,
        max_body=int(float(env.get("DEMON_TOOL_PROXY_MAX_BODY_MB", "256")) * 1024 * 1024),
        timeout_s=float(env.get("DEMON_TOOL_PROXY_TIMEOUT_S", "120")),
    )
    return proxies


def probe_health() -> None:
    """One ``GET /health`` round over every target; records 200s."""
    for name, (host, port) in list(_proxies.items()):
        conn = http.client.HTTPConnection(host, port, timeout=2)
        try:
            conn.request("GET", "/health")
            if conn.getresponse().status == 200:
                _last_healthy[name] = time.monotonic()
        except Exception:
            pass
        finally:
            conn.close()


def _health_loop() -> None:
    while _proxies:
        probe_health()
        time.sleep(HEALTH_POLL_S)


def healthy_tools() -> list[str]:
    """Tools whose /health answered 200 within HEALTH_FRESH_S. Cached by
    the background poll; never does I/O, so safe on the ready path."""
    now = time.monotonic()
    return sorted(n for n, t in _last_healthy.items()
                  if n in _proxies and now - t <= HEALTH_FRESH_S)


class ToolHandler(BaseHTTPRequestHandler):
    server_version = "DemonTools/1.0"
    # One request per connection: the socket was taken from websockets'
    # accept loop and is closed when we return.
    protocol_version = "HTTP/1.0"

    def log_message(self, fmt, *args) -> None:
        from acestep.engine.obs import logger
        logger.bind(component="http").info(
            "tool_proxy remote={} {}", self.client_address[0], fmt % args,
        )

    def _json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        self._proxy()

    def do_POST(self) -> None:  # noqa: N802
        self._proxy()

    def _proxy(self) -> None:
        path, _, query = self.path.partition("?")
        m = re.fullmatch(r"/tools/([^/]+)(/.*)?", path)
        target = _proxies.get(m.group(1)) if m else None
        if target is None:
            self._json(404, {"error": "unknown tool"})
            return
        rest = m.group(2) or "/"

        sid = self.headers.get(SESSION_HEADER)
        local_health = (
            sid is None and self.command == "GET" and rest == "/health"
            and self.client_address[0] in _LOOPBACK_CLIENTS
        )
        if not local_health and not (sid and session_registry.get(sid)):
            self._json(401, {"error": f"{SESSION_HEADER} must name a live session"})
            return

        if "Transfer-Encoding" in self.headers:
            self._json(411, {"error": "Content-Length required"})
            return
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            self._json(400, {"error": "bad Content-Length"})
            return
        if length < 0 or length > _max_body:
            self._json(413, {"error": f"body exceeds {_max_body} bytes"})
            return

        host, port = target
        conn = http.client.HTTPConnection(host, port, timeout=CONNECT_TIMEOUT_S)
        try:
            try:
                conn.connect()
            except OSError as exc:
                self._json(503, {"error": "tool unavailable", "detail": str(exc)})
                return
            # ponytail: per-socket-op timeout, not a hard wall-clock total.
            conn.sock.settimeout(_timeout_s)
            conn.putrequest(self.command, rest + (f"?{query}" if query else ""),
                            skip_accept_encoding=True)
            if self.headers.get("Content-Type"):
                conn.putheader("Content-Type", self.headers["Content-Type"])
            if self.headers.get("Accept"):
                conn.putheader("Accept", self.headers["Accept"])
            conn.putheader("Content-Length", str(length))
            conn.endheaders()
            remaining = length
            while remaining:
                chunk = self.rfile.read(min(remaining, 1 << 20))
                if not chunk:
                    break
                conn.sock.sendall(chunk)
                remaining -= len(chunk)
            resp = conn.getresponse()
            self.send_response(resp.status, resp.reason)
            self.send_header("Content-Type",
                             resp.getheader("Content-Type") or "application/octet-stream")
            if resp.getheader("Content-Length"):
                self.send_header("Content-Length", resp.getheader("Content-Length"))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            shutil.copyfileobj(resp, self.wfile, 1 << 20)
        except socket.timeout:
            self._safe_json(504, {"error": "tool timed out"})
        except (ConnectionError, http.client.HTTPException) as exc:
            self._safe_json(503, {"error": "tool unavailable", "detail": str(exc)})
        finally:
            conn.close()

    def _safe_json(self, status: int, payload: dict) -> None:
        # Headers may already be out if the upstream died mid-response;
        # then the truncated body is all the client gets.
        try:
            self._json(status, payload)
        except OSError:
            pass


def is_tool_request(head: bytes) -> bool:
    return head.startswith(_TOOL_PREFIXES)


def _peek_head(sock: socket.socket) -> bytes:
    """Peek (without consuming) enough of the request line to classify it."""
    need = max(len(p) for p in _TOOL_PREFIXES)
    saved = sock.gettimeout()
    sock.settimeout(CONNECT_TIMEOUT_S)
    head = b""
    try:
        deadline = time.monotonic() + CONNECT_TIMEOUT_S
        while time.monotonic() < deadline:
            head = sock.recv(need, socket.MSG_PEEK)
            if not head or len(head) >= need or not any(
                p.startswith(head) for p in _TOOL_PREFIXES
            ):
                break
            time.sleep(0.005)  # partial request line: let the rest arrive
    except OSError:
        pass
    finally:
        sock.settimeout(saved)
    return head


def wrap_server(srv) -> None:
    """Route tool requests on a websockets ``sync.server.Server`` to
    :class:`ToolHandler`; everything else goes to websockets unchanged."""
    ws_handler = srv.handler

    def handler(sock, addr):
        if not is_tool_request(_peek_head(sock)):
            ws_handler(sock, addr)
            return
        try:
            ToolHandler(sock, addr, srv)
        except Exception:
            pass
        finally:
            try:
                sock.close()
            finally:
                # websockets' own conn handler does this bookkeeping on exit.
                threads = getattr(srv, "handler_threads", None)
                if threads is not None:
                    with srv.lock:
                        threads.discard(threading.current_thread())

    srv.handler = handler
