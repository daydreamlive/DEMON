"""Tiny HTTP control bus the onboard MCP server uses to drive an
already-running session without opening a separate WebSocket.

Routes (default port 1319):

  GET  /sessions               list active sessions (snapshot per id)
  GET  /sessions/<id>          full snapshot for one session
  POST /sessions/<id>/cmd      inject a command. Body framing:
                                 <4-byte LE uint32 json_len>
                                 <json_len bytes UTF-8 JSON>
                                 <remaining bytes: optional binary audio>
                               JSON is the same shape the front-end's
                               WebSocket sends (e.g. ``{"type":"prompt",
                               "tags":"..."}``). Audio (if present) is
                               the same wire format the WS uses for
                               set_timbre_source / set_structure_source
                               / swap_source: ``<II`` channels+samples
                               header followed by interleaved float32 PCM.

Bound to ``127.0.0.1`` by default so accidentally-exposed services don't
let arbitrary clients write into a running browser session. Override
with ``--control-host`` if you really need cross-host access (typical
case: a remote MCP).
"""

from __future__ import annotations

import json
import re
import struct
import sys
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Optional

from . import session_registry


_CMD_PATH = re.compile(r"^/sessions/([A-Za-z0-9_-]{1,64})/cmd$")
_GET_ONE = re.compile(r"^/sessions/([A-Za-z0-9_-]{1,64})$")


def _serialise_handle(h: session_registry.SessionHandle) -> dict:
    """Embed the live snapshot under a stable envelope so callers can
    distinguish registry metadata (id, started_at) from per-session
    state that may have changed since the session was registered."""
    try:
        snap = h.snapshot()
    except Exception as e:
        snap = {"snapshot_error": str(e)}
    return {"id": h.id, "started_at": h.started_at, **snap}


class _ControlHandler(BaseHTTPRequestHandler):
    server_version = "DemonControl/1.0"

    def log_message(self, fmt: str, *args) -> None:
        sys.stdout.write(
            f"[Control] {self.client_address[0]} - {fmt % args}\n",
        )
        sys.stdout.flush()

    def _send_json(self, status: int, payload: dict | list) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def do_GET(self) -> None:  # noqa: N802 (BaseHTTPRequestHandler convention)
        path = self.path.split("?", 1)[0]
        if path == "/sessions":
            self._send_json(
                HTTPStatus.OK,
                [_serialise_handle(h) for h in session_registry.list_handles()],
            )
            return
        m = _GET_ONE.match(path)
        if m:
            sid = m.group(1)
            h = session_registry.get(sid)
            if h is None:
                self._send_json(HTTPStatus.NOT_FOUND, {"error": "unknown session"})
                return
            self._send_json(HTTPStatus.OK, _serialise_handle(h))
            return
        self._send_json(HTTPStatus.NOT_FOUND, {"error": "not found"})

    def do_POST(self) -> None:  # noqa: N802
        path = self.path.split("?", 1)[0]
        m = _CMD_PATH.match(path)
        if not m:
            self._send_json(HTTPStatus.NOT_FOUND, {"error": "not found"})
            return
        sid = m.group(1)
        handle = session_registry.get(sid)
        if handle is None:
            self._send_json(HTTPStatus.NOT_FOUND, {"error": "unknown session"})
            return

        try:
            clen = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            clen = 0
        if clen <= 0:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": "empty body"})
            return

        try:
            body = self.rfile.read(clen)
        except Exception as e:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": f"read failed: {e}"})
            return
        if len(body) < 4:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": "missing json length prefix"})
            return
        json_len = struct.unpack("<I", body[:4])[0]
        if json_len <= 0 or 4 + json_len > len(body):
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": "json length out of range"})
            return

        try:
            data = json.loads(body[4:4 + json_len].decode("utf-8"))
        except Exception as e:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": f"json decode failed: {e}"})
            return
        audio: Optional[bytes] = body[4 + json_len:] or None

        try:
            handle.inject(data, audio)
        except Exception as e:
            self._send_json(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                {"error": f"inject failed: {e}"},
            )
            return

        self._send_json(HTTPStatus.OK, {"sent": True, "session_id": sid,
                                        "audio_bytes": len(audio) if audio else 0})


def start_control_server(host: str, port: int) -> ThreadingHTTPServer:
    """Bind the control HTTP server and start it on a background thread.

    Returns the server object so callers can call ``shutdown()`` if they
    want a clean exit; in practice we run it as a daemon thread and let
    process exit kill it.
    """
    srv = ThreadingHTTPServer((host, port), _ControlHandler)
    t = threading.Thread(target=srv.serve_forever, daemon=True,
                         name="demon-control-http")
    t.start()
    return srv
