"""Single-port HTTP + WebSocket server for the web-app version of the demo.

Serves the browser client from ``static/`` and runs the same
:func:`demos.realtime_motion_graph.server.handle_client` pipeline on
the **same** TCP port, using the websockets library's
``process_request`` hook to short-circuit non-upgrade requests into
static-file HTTP responses.

Single-port matters for Vast.ai / Docker deploys where only one port
is usually mapped into the container.

Usage:
    python -u -m demos.realtime_motion_graph_web.server
    python -u -m demos.realtime_motion_graph_web.server --host 0.0.0.0 --port 8765
    python -u -m demos.realtime_motion_graph_web.server --no-backend
"""

import json
import mimetypes
import os
import sys
import threading
import time
from pathlib import Path

from websockets.http11 import Response
from websockets.datastructures import Headers
from websockets.sync.server import serve as ws_serve

# The generative backend is imported lazily inside main(): in --no-backend
# mode we skip the import entirely so torch and acestep don't load and the
# GPU stays free for other work while iterating on the front-end.


STATIC_DIR = Path(__file__).parent / "static"
VIDEOS_DIR = STATIC_DIR / "videos"
# tests/fixtures lives at the repo root, two levels above this module
# (demos/realtime_motion_graph_web/ -> repo root -> tests/fixtures).
FIXTURES_DIR = Path(__file__).resolve().parents[2] / "tests" / "fixtures"
_AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}

# Set in main() based on --no-backend; read by _process_request when the
# client polls /api/server-info on startup.
_NO_BACKEND = False
# Set in main() based on --accel; read by the WS handler wrapper.
_ACCEL = "tensorrt"
# Set in main() based on --kiosk / --mode; surfaced to the client via
# /api/server-info so installation-only behaviors (cursor auto-hide,
# idle settings reset) and the initial display mode can be CLI-driven.
_KIOSK = False
_DEFAULT_MODE = "graph"
_VALID_MODES = ("graph", "video")

# Keep the wire compact and don't cache anything so the product team
# always sees the latest JS after a redeploy.
_NO_CACHE_HEADERS = [
    ("Cache-Control", "no-store, must-revalidate"),
    ("Pragma", "no-cache"),
    ("Expires", "0"),
    # Chrome requires this for Web MIDI API device enumeration.
    ("Permissions-Policy", "midi=*"),
]


def _content_type_for(path: Path) -> str:
    # Extra module types that the stdlib mimetypes table misses.
    ext = path.suffix.lower()
    if ext == ".js" or ext == ".mjs":
        return "application/javascript; charset=utf-8"
    if ext == ".css":
        return "text/css; charset=utf-8"
    if ext == ".html":
        return "text/html; charset=utf-8"
    if ext == ".json":
        return "application/json; charset=utf-8"
    if ext == ".wasm":
        return "application/wasm"
    guessed, _ = mimetypes.guess_type(str(path))
    return guessed or "application/octet-stream"


def _resolve_static(url_path: str) -> Path | None:
    """Map a URL path to a file inside ``STATIC_DIR``, or None if missing.

    Refuses any path that tries to escape ``STATIC_DIR`` via ``..`` or
    absolute segments.
    """
    # Strip query string and fragment.
    path = url_path.split("?", 1)[0].split("#", 1)[0]
    if path in ("", "/"):
        path = "/index.html"
    # Leading slash only; reject empty segments from repeated slashes.
    if not path.startswith("/"):
        return None
    rel = path.lstrip("/")
    if not rel:
        rel = "index.html"
    candidate = (STATIC_DIR / rel).resolve()
    try:
        candidate.relative_to(STATIC_DIR.resolve())
    except ValueError:
        return None  # path escape attempt
    if not candidate.is_file():
        return None
    return candidate


def _log_http(remote: str, status: int, method: str, url: str):
    sys.stdout.write(f"[HTTP] {remote} {method} {url} -> {status}\n")
    sys.stdout.flush()


def _process_request(connection, request):
    """Return a :class:`Response` for plain HTTP; return ``None`` to let
    the websockets library finish the WebSocket upgrade.

    This runs BEFORE the WS handshake, so it lets us multiplex HTTP and
    WebSocket on a single TCP port.
    """
    # If this looks like a websocket upgrade, defer to the WS handshake.
    upgrade = request.headers.get("Upgrade", "") or ""
    if upgrade.lower() == "websocket":
        return None

    url = request.path
    try:
        remote = str(connection.remote_address[0]) if connection.remote_address else "?"
    except Exception:
        remote = "?"

    # API: server-info — lets the client know whether the backend is up.
    # In --no-backend mode the client takes the video-only path: it plays
    # the source audio directly and skips the WebSocket connection entirely.
    if url.split("?", 1)[0] == "/api/server-info":
        body = json.dumps({
            "no_backend": _NO_BACKEND,
            "kiosk": _KIOSK,
            "default_mode": _DEFAULT_MODE,
        }).encode()
        _log_http(remote, 200, "GET", url)
        return Response(
            200, "OK",
            Headers([
                ("Content-Type", "application/json; charset=utf-8"),
                ("Content-Length", str(len(body))),
                *_NO_CACHE_HEADERS,
            ]),
            body,
        )

    # API: list LoRAs in MODELS_DIR/loras/.  Cheap (filesystem glob, no
    # torch / no engine load), so the browser can render the Library
    # panel before the user even clicks Play.  Uses the same path
    # resolution the WebSocket pipeline uses, so everyone agrees on
    # what's in the catalog.
    if url.split("?", 1)[0] == "/api/loras":
        from acestep.paths import discover_loras, loras_dir
        try:
            d = loras_dir()
            entries = [
                {
                    "id": p.stem, "name": p.stem, "path": str(p),
                    "state": "registered", "strength": 0.0,
                    "materialized_bytes": 0,
                }
                for p in discover_loras(d)
            ]
        except Exception as e:
            entries = []
            sys.stdout.write(f"[HTTP] /api/loras error: {e}\n")
            sys.stdout.flush()
        body = json.dumps({"dir": str(loras_dir()), "loras": entries}).encode()
        _log_http(remote, 200, "GET", url)
        return Response(
            200, "OK",
            Headers([
                ("Content-Type", "application/json; charset=utf-8"),
                ("Content-Length", str(len(body))),
                *_NO_CACHE_HEADERS,
            ]),
            body,
        )

    # API: list video files in static/videos/
    if url.split("?", 1)[0] == "/api/videos":
        _VIDEO_EXTS = {".mp4", ".webm", ".mov"}
        videos = []
        if VIDEOS_DIR.is_dir():
            videos = sorted(
                f.name for f in VIDEOS_DIR.iterdir()
                if f.is_file() and f.suffix.lower() in _VIDEO_EXTS
            )
        body = json.dumps(videos).encode()
        _log_http(remote, 200, "GET", url)
        return Response(
            200, "OK",
            Headers([
                ("Content-Type", "application/json; charset=utf-8"),
                ("Content-Length", str(len(body))),
                *_NO_CACHE_HEADERS,
            ]),
            body,
        )

    # API: list audio fixtures from tests/fixtures/
    if url.split("?", 1)[0] == "/api/fixtures":
        fixtures = []
        if FIXTURES_DIR.is_dir():
            fixtures = sorted(
                f.name for f in FIXTURES_DIR.iterdir()
                if f.is_file() and f.suffix.lower() in _AUDIO_EXTS
            )
        body = json.dumps(fixtures).encode()
        _log_http(remote, 200, "GET", url)
        return Response(
            200, "OK",
            Headers([
                ("Content-Type", "application/json; charset=utf-8"),
                ("Content-Length", str(len(body))),
                *_NO_CACHE_HEADERS,
            ]),
            body,
        )

    # Serve files from tests/fixtures/ under /fixtures/<name>.
    fixture_match = url.split("?", 1)[0].split("#", 1)[0]
    if fixture_match.startswith("/fixtures/"):
        rel = fixture_match[len("/fixtures/"):]
        if rel and "/" not in rel and "\\" not in rel and not rel.startswith("."):
            candidate = (FIXTURES_DIR / rel).resolve()
            try:
                candidate.relative_to(FIXTURES_DIR.resolve())
            except ValueError:
                candidate = None  # path escape attempt
            if candidate and candidate.is_file() and candidate.suffix.lower() in _AUDIO_EXTS:
                try:
                    body = candidate.read_bytes()
                except OSError as e:
                    msg = f"500 {e}\n".encode()
                    _log_http(remote, 500, "GET", url)
                    return Response(
                        500, "Internal Server Error",
                        Headers([
                            ("Content-Type", "text/plain; charset=utf-8"),
                            ("Content-Length", str(len(msg))),
                            *_NO_CACHE_HEADERS,
                        ]),
                        msg,
                    )
                _log_http(remote, 200, "GET", url)
                return Response(
                    200, "OK",
                    Headers([
                        ("Content-Type", _content_type_for(candidate)),
                        ("Content-Length", str(len(body))),
                        *_NO_CACHE_HEADERS,
                    ]),
                    body,
                )

    target = _resolve_static(url)
    if target is None:
        body = b"404 not found\n"
        _log_http(remote, 404, "GET", url)
        return Response(
            404,
            "Not Found",
            Headers([
                ("Content-Type", "text/plain; charset=utf-8"),
                ("Content-Length", str(len(body))),
                *_NO_CACHE_HEADERS,
            ]),
            body,
        )

    try:
        body = target.read_bytes()
    except OSError as e:
        msg = f"500 {e}\n".encode()
        _log_http(remote, 500, "GET", url)
        return Response(
            500,
            "Internal Server Error",
            Headers([
                ("Content-Type", "text/plain; charset=utf-8"),
                ("Content-Length", str(len(msg))),
                *_NO_CACHE_HEADERS,
            ]),
            msg,
        )

    headers = Headers([
        ("Content-Type", _content_type_for(target)),
        ("Content-Length", str(len(body))),
        *_NO_CACHE_HEADERS,
    ])
    _log_http(remote, 200, "GET", url)
    return Response(200, "OK", headers, body)


def _stub_handle_client(ws):
    """Stub handler used when --no-backend is set. Closes the WS connection
    immediately so the browser sees a clean disconnect instead of hanging."""
    try:
        ws.close(code=1011, reason="ui-only mode (no generative backend)")
    except Exception:
        pass


def main():
    host = "0.0.0.0"
    port = 8765  # single port: serves both HTTP and WebSocket
    accel = "tensorrt"  # decoder + vae backend; overridden by --accel

    args = sys.argv[1:]
    no_backend = "--no-backend" in args or "--ui-only" in args
    if "--host" in args:
        idx = args.index("--host")
        host = args[idx + 1]
    if "--port" in args:
        idx = args.index("--port")
        port = int(args[idx + 1])
    # Back-compat with the old two-port flags: --http-port wins if both set.
    if "--http-port" in args:
        idx = args.index("--http-port")
        port = int(args[idx + 1])
    if "--ws-port" in args and "--http-port" not in args:
        idx = args.index("--ws-port")
        port = int(args[idx + 1])
    if "--accel" in args:
        idx = args.index("--accel")
        accel = args[idx + 1]
    _VALID_ACCEL = ("tensorrt", "compile", "eager")
    if accel not in _VALID_ACCEL:
        raise SystemExit(
            f"[Server] --accel must be one of {_VALID_ACCEL}, got {accel!r}"
        )

    kiosk = "--kiosk" in args
    default_mode = "graph"
    if "--mode" in args:
        idx = args.index("--mode")
        default_mode = args[idx + 1]
    if default_mode not in _VALID_MODES:
        raise SystemExit(
            f"[Server] --mode must be one of {_VALID_MODES}, got {default_mode!r}"
        )

    if not STATIC_DIR.exists():
        raise SystemExit(f"[Server] static dir missing: {STATIC_DIR}")

    global _NO_BACKEND, _ACCEL, _KIOSK, _DEFAULT_MODE
    _NO_BACKEND = no_backend
    _ACCEL = accel
    _KIOSK = kiosk
    _DEFAULT_MODE = default_mode

    if no_backend:
        ws_handler = _stub_handle_client
        print("[Server] --no-backend: GPU/model imports skipped, WS upgrades will close immediately")
    else:
        # Defer the heavy import until we know we need it. Pulling this in
        # loads torch + acestep + TRT machinery; in --no-backend we never
        # touch any of it.
        from demos.realtime_motion_graph.server import handle_client

        def ws_handler(ws):
            handle_client(ws, decoder_backend=accel, vae_backend=accel)

    print(f"[Server] Starting single-port HTTP+WS on :{port}")
    srv = ws_serve(
        ws_handler,
        host,
        port,
        max_size=50 * 1024 * 1024,
        process_request=_process_request,
    )
    ws_thread = threading.Thread(target=srv.serve_forever, daemon=True)
    ws_thread.start()

    browsable_host = "localhost" if host in ("0.0.0.0", "::", "") else host
    extras = [f"mode={default_mode}"]
    if kiosk:
        extras.append("kiosk")
    extra_str = " " + " ".join(f"[{e}]" for e in extras)
    mode = "UI-ONLY (no backend)" if no_backend else f"WEB APP, single port, accel={accel}{extra_str}"
    print()
    print("=" * 60)
    print(f"  Real-Time Motion-to-Music  ({mode})")
    print("=" * 60)
    print(f"  Open:      http://{browsable_host}:{port}/")
    print(f"  WebSocket: ws://{browsable_host}:{port}/")
    print(f"  Fixtures:  {FIXTURES_DIR}")
    print("  Ctrl+C to stop")
    print("=" * 60)
    print()

    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\n[Server] Shutting down...")
        os._exit(0)


if __name__ == "__main__":
    main()
