"""HTTP + WebSocket server for the realtime motion-graph demo backend.

Multiplexes the JSON HTTP API (``/api/server-info``, ``/api/loras``,
``/api/videos``, ``/api/fixtures``), static fixture/video file serving
(``/fixtures/<name>``, ``/videos/<name>``), and the
:func:`.ws_adapter.handle_client` WebSocket pipeline onto a single TCP port,
using the websockets library's ``process_request`` hook to short-circuit
non-upgrade requests into HTTP responses.

The Next.js dev server (``run.py``) proxies these endpoints through to
this backend; in production the same routes are served directly.

Usage:
    python -u -m demos.realtime_motion_graph_web.server
    python -u -m demos.realtime_motion_graph_web.server --host 0.0.0.0 --port 1318
    python -u -m demos.realtime_motion_graph_web.server --no-backend
"""

import json
import mimetypes
import os
import sys
import threading
import time
import urllib.parse
from pathlib import Path

from websockets.http11 import Response
from websockets.datastructures import Headers
from websockets.sync.server import serve as ws_serve

from acestep.engine.obs import configure as configure_logging, logger
from acestep.fixtures import KNOWN_FIXTURES, audio_fixture
from acestep.user_uploads import enumerate_user_uploads, user_upload_audio

# The generative backend is imported lazily inside main(): in --no-backend
# mode we skip the import entirely so torch and acestep don't load and the
# GPU stays free for other work while iterating on the front-end.


VIDEOS_DIR = Path(__file__).parent / "videos"
_VIDEO_EXTS = {".mp4", ".webm", ".mov"}

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
# Set by main() once CLI args are parsed; the HTTP /api/loras and
# /api/* meta endpoints read it so the UI can label the active
# checkpoint scale (2B / 5B) without waiting for the WS ready frame.
# Initialized to the default acestep-v15-turbo (2B) to keep the
# endpoint sane in --no-backend mode where main() may exit early.
_CHECKPOINT: str = "acestep-v15-turbo"
_VALID_MODES = ("graph", "video")

# Short aliases for --checkpoint. Map directly to the canonical
# checkpoint directory name under <MODELS_DIR>/checkpoints/.
_CHECKPOINT_ALIASES = {
    "xl": "acestep-v15-xl-turbo",
}

_NO_CACHE_HEADERS = [
    ("Cache-Control", "no-store, must-revalidate"),
    ("Pragma", "no-cache"),
    ("Expires", "0"),
    # Chrome requires this for Web MIDI API device enumeration.
    ("Permissions-Policy", "midi=*"),
]
_PUBLIC_HTTP_HEADERS = [
    ("Access-Control-Allow-Origin", "*"),
    *_NO_CACHE_HEADERS,
]


def _resolve_video(name: str) -> Path | None:
    """Map a ``/videos/<name>`` request to a file inside ``VIDEOS_DIR``.

    ``name`` is the raw URL segment after ``/videos/``. Refuses anything
    that contains a path separator so a request can't escape ``VIDEOS_DIR``.
    """
    decoded = urllib.parse.unquote(name)
    if not decoded or "/" in decoded or "\\" in decoded or decoded in (".", ".."):
        return None
    candidate = VIDEOS_DIR / decoded
    if candidate.suffix.lower() not in _VIDEO_EXTS:
        return None
    if not candidate.is_file():
        return None
    return candidate


def _log_http(remote: str, status: int, method: str, url: str):
    logger.bind(component="http").info(
        "http_request remote={} method={} url={} status={}",
        remote, method, url, status,
    )


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

    path_only = url.split("?", 1)[0].split("#", 1)[0]

    # API: server-info — lets the client know whether the backend is up.
    if path_only == "/api/server-info":
        # Surface startup-warmup state so the pool can gate "free" on a
        # warmed engine. Read only if the warmup module is already
        # imported — don't force the heavy chain in --no-backend mode
        # (importing acestep.streaming.warmup is cheap on its own, but
        # the warmup state is only populated once the backend imports
        # and runs).
        _warm = None
        _wm = sys.modules.get("acestep.streaming.warmup")
        if _wm is not None:
            _warm = getattr(_wm, "WARMUP_STATE", None)
        body = json.dumps({
            "no_backend": _NO_BACKEND,
            "kiosk": _KIOSK,
            "default_mode": _DEFAULT_MODE,
            "warmup": _warm,
            # Fixtures the pod can load server-side: the client may send
            # {use_server_fixture:true, fixture_name:<one of these>} and
            # skip the ~20 MB PCM upload entirely. Advertised here so the
            # UI only opts in against a backend that supports it (the UI
            # ships via Vercel instantly; the backend via bake, lagged).
            "server_side_fixtures": sorted(KNOWN_FIXTURES),
        }).encode()
        _log_http(remote, 200, "GET", url)
        return Response(
            200, "OK",
            Headers([
                ("Content-Type", "application/json; charset=utf-8"),
                ("Content-Length", str(len(body))),
                # Public, read-only capability probe. The webapp UI
                # (served from a different origin than the pod tunnel)
                # fetches this before the WS handshake to decide whether
                # the pod supports server-side fixture load. Simple GET,
                # no credentials/custom headers → no preflight; a single
                # ACAO:* on the response is sufficient and safe (nothing
                # sensitive here).
                *_PUBLIC_HTTP_HEADERS,
            ]),
            body,
        )

    # API: list LoRAs in MODELS_DIR/loras/.  Cheap (filesystem glob, no
    # torch / no engine load), so the browser can render the Library
    # panel before the user even clicks Play.  Uses the same path
    # resolution the WebSocket pipeline uses, so everyone agrees on
    # what's in the catalog.
    if path_only == "/api/loras":
        from acestep.lora_metadata import load_lora_metadata
        from acestep.paths import (
            checkpoint_scale,
            discover_all_loras,
            extra_lora_dirs,
            loras_dir,
        )
        try:
            # Recursive across the primary library AND every directory
            # in ACESTEP_EXTRA_LORA_DIRS, matching the engine's own
            # register_library() scan so the HTTP catalog and the
            # engine-side catalog stay in lockstep.
            entries = []
            seen_ids: set[str] = set()
            for p in discover_all_loras():
                # Same-stem dedup mirrors LoRAManager.register_lora's
                # first-wins behavior so the UI can't see a phantom id
                # the engine refused to register.
                if p.stem in seen_ids:
                    continue
                seen_ids.add(p.stem)
                md = load_lora_metadata(p).to_wire()
                entries.append({
                    "id": p.stem,
                    "name": md.get("name") or p.stem,
                    "path": str(p),
                    "state": "registered",
                    "strength": 0.0,
                    "materialized_bytes": 0,
                    "metadata": md,
                })
        except Exception as e:
            entries = []
            logger.bind(component="http").exception(
                "loras_listing_failed error={}", e,
            )
        # ``dir`` stays as the primary library root for back-compat
        # (existing clients display it as "LoRA directory").
        # ``extra_dirs`` surfaces extra LoRA dirs from acestep.local.json
        # so the operator can see which training dirs the scan picked up.
        # ``checkpoint`` + ``checkpoint_scale`` let the UI hide LoRAs
        # whose ``metadata.base_model_scale`` doesn't match the active
        # checkpoint without waiting for the WS ready frame.
        body = json.dumps({
            "dir": str(loras_dir()),
            "extra_dirs": [str(p) for p in extra_lora_dirs()],
            "checkpoint": _CHECKPOINT,
            "checkpoint_scale": checkpoint_scale(_CHECKPOINT),
            "loras": entries,
        }).encode()
        _log_http(remote, 200, "GET", url)
        return Response(
            200, "OK",
            Headers([
                ("Content-Type", "application/json; charset=utf-8"),
                ("Content-Length", str(len(body))),
                *_PUBLIC_HTTP_HEADERS,
            ]),
            body,
        )

    # API: list video files in VIDEOS_DIR.
    if path_only == "/api/videos":
        videos: list[str] = []
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
                *_PUBLIC_HTTP_HEADERS,
            ]),
            body,
        )

    # API: list audio fixtures (from the daydreamlive/demon-fixtures-v2 HF dataset).
    # Files are downloaded on-demand by /fixtures/<name>; this endpoint just
    # returns the canonical manifest from acestep.fixtures so the UI can render
    # the picker before any download happens.
    if path_only == "/api/fixtures":
        body = json.dumps(sorted(KNOWN_FIXTURES)).encode()
        _log_http(remote, 200, "GET", url)
        return Response(
            200, "OK",
            Headers([
                ("Content-Type", "application/json; charset=utf-8"),
                ("Content-Length", str(len(body))),
                *_PUBLIC_HTTP_HEADERS,
            ]),
            body,
        )

    # API: knob manifest — the complete, self-describing knob registry
    # (acestep.streaming.knobs.knob_catalog) so any frontend can render and
    # validate controls without re-declaring them. Static schema: the
    # per-session ``lora_str_<id>`` knobs and live values arrive over the WS
    # ready frame / MCP list_knobs. ``?sde=1`` returns the SDE-mode core set.
    # ACAO:* like /api/server-info — the UI fetches this cross-origin (Vercel)
    # before the WS handshake, and the schema carries nothing sensitive.
    if path_only == "/api/knobs":
        from acestep.streaming.knobs import KNOB_SCHEMA_VERSION, knob_catalog
        query = url.split("?", 1)[1] if "?" in url else ""
        sde = "sde=1" in query or "sde=true" in query
        body = json.dumps({
            "version": KNOB_SCHEMA_VERSION,
            "knobs": knob_catalog(sde=sde, loras=[]),
        }).encode()
        _log_http(remote, 200, "GET", url)
        return Response(
            200, "OK",
            Headers([
                ("Content-Type", "application/json; charset=utf-8"),
                ("Content-Length", str(len(body))),
                *_PUBLIC_HTTP_HEADERS,
            ]),
            body,
        )

    # API: WebSocket wire contract — the self-describing command/event
    # vocabulary (demos.realtime_motion_graph_web.protocol.wire_contract) so
    # a re-skinned frontend or MCP agent can build every control message and
    # decode every server event without reverse-engineering protocol.ts. The
    # per-command ``params`` knob payload has its own manifest at /api/knobs.
    # ACAO:* like /api/knobs — fetched cross-origin before the WS handshake,
    # carries nothing sensitive.
    if path_only == "/api/protocol":
        from .protocol import wire_contract
        body = json.dumps(wire_contract()).encode()
        _log_http(remote, 200, "GET", url)
        return Response(
            200, "OK",
            Headers([
                ("Content-Type", "application/json; charset=utf-8"),
                ("Content-Length", str(len(body))),
                *_PUBLIC_HTTP_HEADERS,
            ]),
            body,
        )

    # API: list user uploads sitting in MODELS_DIR/user_uploads/.  Same
    # shape as /api/fixtures so the client can merge both lists into a
    # single picker.
    if path_only == "/api/user_uploads":
        try:
            names = enumerate_user_uploads()
        except Exception as e:
            sys.stdout.write(f"[HTTP] /api/user_uploads error: {e}\n")
            sys.stdout.flush()
            names = []
        body = json.dumps(names).encode()
        _log_http(remote, 200, "GET", url)
        return Response(
            200, "OK",
            Headers([
                ("Content-Type", "application/json; charset=utf-8"),
                ("Content-Length", str(len(body))),
                *_PUBLIC_HTTP_HEADERS,
            ]),
            body,
        )

    # Serve files from the HF fixture dataset under /fixtures/<name>.
    # audio_fixture() validates `name` against KNOWN_FIXTURES (so this is
    # also our path-escape guard) and downloads on first access.
    if path_only.startswith("/fixtures/"):
        rel = path_only[len("/fixtures/"):]
        try:
            candidate = audio_fixture(rel)
        except KeyError:
            candidate = None
        except Exception as e:
            msg = f"500 {e}\n".encode()
            _log_http(remote, 500, "GET", url)
            return Response(
                500, "Internal Server Error",
                Headers([
                    ("Content-Type", "text/plain; charset=utf-8"),
                    ("Content-Length", str(len(msg))),
                    *_PUBLIC_HTTP_HEADERS,
                ]),
                msg,
            )
        if candidate and candidate.is_file():
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
                        *_PUBLIC_HTTP_HEADERS,
                    ]),
                    msg,
                )
            ctype, _ = mimetypes.guess_type(candidate.name)
            _log_http(remote, 200, "GET", url)
            return Response(
                200, "OK",
                Headers([
                    ("Content-Type", ctype or "application/octet-stream"),
                    ("Content-Length", str(len(body))),
                    *_PUBLIC_HTTP_HEADERS,
                ]),
                body,
            )

    # Serve user-uploaded audio under /user_uploads/<name>. Whitelisted
    # by enumerate_user_uploads (which gates by extension and excludes
    # sidecar files) so a path-escape attempt or a request for a
    # sidecar JSON can't slip through. Same shape as /fixtures/<name>.
    if path_only.startswith("/user_uploads/"):
        rel = urllib.parse.unquote(path_only[len("/user_uploads/"):])
        allowed = set(enumerate_user_uploads())
        if rel in allowed:
            try:
                candidate = user_upload_audio(rel)
            except FileNotFoundError:
                candidate = None
            if candidate is not None:
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
                            *_PUBLIC_HTTP_HEADERS,
                        ]),
                        msg,
                    )
                ctype, _ = mimetypes.guess_type(candidate.name)
                _log_http(remote, 200, "GET", url)
                return Response(
                    200, "OK",
                    Headers([
                        ("Content-Type", ctype or "application/octet-stream"),
                        ("Content-Length", str(len(body))),
                        *_PUBLIC_HTTP_HEADERS,
                    ]),
                    body,
                )

    # Serve user-supplied videos under /videos/<name> from VIDEOS_DIR.
    if path_only.startswith("/videos/"):
        target = _resolve_video(path_only[len("/videos/"):])
        if target is not None:
            try:
                body = target.read_bytes()
            except OSError as e:
                msg = f"500 {e}\n".encode()
                _log_http(remote, 500, "GET", url)
                return Response(
                    500, "Internal Server Error",
                    Headers([
                        ("Content-Type", "text/plain; charset=utf-8"),
                        ("Content-Length", str(len(msg))),
                        *_PUBLIC_HTTP_HEADERS,
                    ]),
                    msg,
                )
            ctype, _ = mimetypes.guess_type(target.name)
            _log_http(remote, 200, "GET", url)
            return Response(
                200, "OK",
                Headers([
                    ("Content-Type", ctype or "application/octet-stream"),
                    ("Content-Length", str(len(body))),
                    *_PUBLIC_HTTP_HEADERS,
                ]),
                body,
            )

    body = b"404 not found\n"
    _log_http(remote, 404, "GET", url)
    return Response(
        404,
        "Not Found",
        Headers([
            ("Content-Type", "text/plain; charset=utf-8"),
            ("Content-Length", str(len(body))),
            *_PUBLIC_HTTP_HEADERS,
        ]),
        body,
    )


def _stub_handle_client(ws):
    """Stub handler used when --no-backend is set. Closes the WS connection
    immediately so the browser sees a clean disconnect instead of hanging."""
    try:
        ws.close(code=1011, reason="ui-only mode (no generative backend)")
    except Exception:
        pass


def _run_preflight(decoder_accel: str, vae_accel: str, checkpoint: str) -> None:
    """Fail fast (and loudly) at boot instead of on the first browser
    connection.

    Two checks, mirroring what the first WebSocket session would hit:

    1. Checkpoints — downloads the ACE-Step weights NOW, in the
       terminal where the operator can see progress, rather than
       stalling the first silent WS connect for 5-15 minutes.
    2. TRT engines — when either component runs TensorRT, verify a 60 s
       profile is built (decoder and/or VAE engines, matching the
       backends in use). On failure, print the fix and exit non-zero.

    ``--skip-preflight`` bypasses both checks (e.g. exotic mixed setups,
    or testing the error paths themselves).
    """
    from acestep.model_downloader import ensure_main_model, ensure_dit_model
    from acestep.paths import (
        EngineNotBuiltError,
        available_trt_engines,
        checkpoints_dir,
    )
    from acestep.setup import DEMO_COMMAND, SETUP_COMMAND

    ok, msg = ensure_main_model()
    if ok and checkpoint != "acestep-v15-turbo":
        ok, msg = ensure_dit_model(checkpoint)
    if not ok:
        print()
        print("=" * 64)
        print("  Model checkpoints unavailable")
        print("=" * 64)
        print(f"  {msg}")
        print(f"  expected location: {checkpoints_dir()}")
        print(f"  fix: run `{SETUP_COMMAND}` (or `uv run acestep-download`)")
        print("=" * 64)
        raise SystemExit(1)
    logger.info("preflight_checkpoints_ok checkpoint={}", checkpoint)

    needs: tuple[str, ...] = ()
    if decoder_accel == "tensorrt":
        needs += ("decoder",)
    if vae_accel == "tensorrt":
        needs += ("vae_encode", "vae_decode")
    if not needs:
        return
    trt_profile_checkpoint = (
        checkpoint if decoder_accel == "tensorrt" else "acestep-v15-turbo"
    )
    try:
        available_trt_engines(
            duration_s=60.0,
            needs=needs,
            checkpoint=trt_profile_checkpoint,
        )
    except EngineNotBuiltError as exc:
        print()
        print("=" * 64)
        print("  TensorRT engines not built")
        print("=" * 64)
        print(f"  {exc}")
        print()
        print(f"  fix (recommended): {SETUP_COMMAND}")
        if exc.build_command:
            print(f"  fix (manual):      {exc.build_command}")
        print()
        print("  Or run without TensorRT (slower, long first-tick warmup):")
        print(f"    {DEMO_COMMAND} -- --accel compile")
        print("=" * 64)
        raise SystemExit(1)
    logger.info(
        "preflight_engines_ok needs={} checkpoint={}",
        ",".join(needs), trt_profile_checkpoint,
    )


def main():
    # Wire logging FIRST so even the CLI-arg validation prints flow through
    # the configured sinks. configure() is idempotent so a duplicate call
    # in any nested entry point is a no-op.
    configure_logging()

    host = "0.0.0.0"
    port = 1318  # single port: serves both HTTP and WebSocket
    # Control bus: a tiny HTTP server the demo's onboard MCP server hits
    # to drive an already-running session. Bound to localhost-only by
    # default; override with --control-host / --control-port.
    control_host = "127.0.0.1"
    control_port = 1319
    accel = "tensorrt"  # decoder + vae backend; overridden by --accel
    checkpoint = "acestep-v15-turbo"  # DiT variant; overridden by --checkpoint

    args = sys.argv[1:]
    no_backend = "--no-backend" in args or "--ui-only" in args
    offload_text_encoder = "--offload-text-encoder" in args
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
    # Per-component overrides. Default each to the bulk --accel value so
    # `--accel eager` still sets both. Use case for splitting: a checkpoint
    # whose TRT engines exist for one component but not the other, or
    # debugging one path in eager while the other stays on TRT.
    decoder_accel = accel
    vae_accel = accel
    if "--decoder-accel" in args:
        idx = args.index("--decoder-accel")
        decoder_accel = args[idx + 1]
    if "--vae-accel" in args:
        idx = args.index("--vae-accel")
        vae_accel = args[idx + 1]
    if decoder_accel not in _VALID_ACCEL:
        raise SystemExit(
            f"[Server] --decoder-accel must be one of {_VALID_ACCEL}, got {decoder_accel!r}"
        )
    if vae_accel not in _VALID_ACCEL:
        raise SystemExit(
            f"[Server] --vae-accel must be one of {_VALID_ACCEL}, got {vae_accel!r}"
        )
    if "--checkpoint" in args:
        idx = args.index("--checkpoint")
        checkpoint = args[idx + 1]
        checkpoint = _CHECKPOINT_ALIASES.get(checkpoint, checkpoint)
    if "--control-host" in args:
        idx = args.index("--control-host")
        control_host = args[idx + 1]
    if "--control-port" in args:
        idx = args.index("--control-port")
        control_port = int(args[idx + 1])
    control_disabled = "--no-control" in args

    kiosk = "--kiosk" in args
    default_mode = "graph"
    if "--mode" in args:
        idx = args.index("--mode")
        default_mode = args[idx + 1]
    if default_mode not in _VALID_MODES:
        raise SystemExit(
            f"[Server] --mode must be one of {_VALID_MODES}, got {default_mode!r}"
        )

    global _NO_BACKEND, _ACCEL, _KIOSK, _DEFAULT_MODE, _CHECKPOINT
    _NO_BACKEND = no_backend
    _ACCEL = accel
    _KIOSK = kiosk
    _DEFAULT_MODE = default_mode
    _CHECKPOINT = checkpoint

    if no_backend:
        ws_handler = _stub_handle_client
        logger.info(
            "ui_only_mode skipped=gpu_and_model_imports",
        )
    else:
        # Fail fast at boot on missing checkpoints / engines instead of
        # on the first browser connection (where the failure used to
        # surface as a silent stall or a WS error frame). Also performs
        # the checkpoint download here, visibly, when needed.
        if "--skip-preflight" not in args:
            _run_preflight(decoder_accel, vae_accel, checkpoint)

        # Defer the heavy import until we know we need it. Pulling this in
        # loads torch + acestep + TRT machinery; in --no-backend we never
        # touch any of it.
        from .ws_adapter import handle_client

        def ws_handler(ws):
            handle_client(
                ws,
                decoder_backend=decoder_accel,
                vae_backend=vae_accel,
                checkpoint=checkpoint,
                offload_text_encoder=offload_text_encoder,
            )

        # Pay the one-time cold-start cost (TRT decoder-engine load,
        # LoRA-refit manager, ModelContext / conditioning, first-tick
        # pipeline build) once at boot, BEFORE accepting real traffic,
        # so every real "begin" gets the ~5s warm path instead of ~40s.
        # Synchronous on purpose: the heartbeat sidecar only advertises
        # this pod to the pool after main() proceeds, so the pod isn't
        # routed real users until it's warm. Enable with
        # DEMON_STARTUP_WARMUP=1.
        if os.environ.get("DEMON_STARTUP_WARMUP", "0") != "0":
            from acestep.streaming.warmup import run_startup_warmup

            run_startup_warmup(
                decoder_backend=decoder_accel,
                vae_backend=vae_accel,
                checkpoint=checkpoint,
                offload_text_encoder=offload_text_encoder,
            )

    # Start the MCP control bus FIRST so registry registrations from the
    # WS handler land in an already-listening HTTP server. Skipped in
    # --no-backend mode (no sessions to register) and on --no-control.
    if not no_backend and not control_disabled:
        from . import control_http
        try:
            control_http.start_control_server(control_host, control_port)
            logger.info(
                "control_bus_listening host={} port={}",
                control_host, control_port,
            )
        except OSError as exc:
            logger.warning(
                "control_bus_bind_failed host={} port={} error={}",
                control_host, control_port, exc,
            )

    # Pre-flight: clear any user uploads left behind in
    # MODELS_DIR/user_uploads/ before we start accepting traffic. The
    # rtmg backend is one-session-per-pod, but pods are rented from a
    # pool and reused across users — without this, a fresh process
    # boot on a recycled pod would expose the previous renter's
    # uploads to the next renter via /api/user_uploads (and let them
    # play the audio via /user_uploads/<name>). Also matters for the
    # baked :warm image, which captures /workspace/demon_models — any
    # uploads sitting in the bake host's user_uploads_dir at bake
    # time would otherwise ship into every fresh pod. The per-session
    # wipe in ws_adapter.handle_client covers consecutive sessions on
    # a single process; this one covers cold boot. Skipped in
    # --no-backend (no sessions ever land).
    if not no_backend:
        from acestep.user_uploads import wipe_user_uploads
        try:
            wiped = wipe_user_uploads()
            if wiped:
                logger.info("user_uploads_wiped_at_startup entries={}", wiped)
        except Exception as exc:
            logger.warning("user_uploads_wipe_at_startup_failed error={}", exc)

        # Warm librosa's numba-JIT'd beat tracker (and the key detector)
        # on a tiny clip in the background. Cold, the first upload's BPM
        # analysis pays ~4 s of one-time JIT compilation; warmed here it
        # runs in well under a second. Daemon thread: never blocks boot,
        # and a failure only means the first upload pays the JIT cost as
        # before.
        def _warm_audio_analysis_jit() -> None:
            try:
                import numpy as np

                import librosa
                from acestep.audio.key_detection import detect_key

                t0 = time.monotonic()
                y = np.sin(
                    np.linspace(0, 2400 * np.pi, 48000 * 3),
                ).astype(np.float32)
                librosa.beat.beat_track(y=y, sr=48000)
                detect_key(y, 48000)
                logger.info(
                    "audio_analysis_jit_warmed duration_s={:.1f}",
                    time.monotonic() - t0,
                )
            except Exception as exc:
                logger.warning("audio_analysis_jit_warm_failed error={}", exc)

        threading.Thread(
            target=_warm_audio_analysis_jit,
            name="analysis-jit-warm",
            daemon=True,
        ).start()

    logger.info("server_starting port={}", port)
    srv = ws_serve(
        ws_handler,
        host,
        port,
        # Sized to fit the React UI's MAX_FIXTURE_DURATION_S (240 s)
        # at 48 kHz stereo Float32 (~88 MiB) with comfortable headroom.
        # See web/engine/audio/loadFixture.ts.
        max_size=100 * 1024 * 1024,
        process_request=_process_request,
        # A generation tick or TRT/VAE setup can hold the GIL long enough to
        # starve the keepalive's recv_events thread past the 20s default, so a
        # live session gets closed with `1011 keepalive ping timeout`. Widen
        # the pong deadline to 90s. Trade-off: slower detection of genuinely
        # dead clients — pair with an app-level idle-session reaper.
        ping_timeout=90,
    )
    ws_thread = threading.Thread(target=srv.serve_forever, daemon=True)
    ws_thread.start()

    browsable_host = "localhost" if host in ("0.0.0.0", "::", "") else host
    extras = [f"mode={default_mode}"]
    if kiosk:
        extras.append("kiosk")
    if offload_text_encoder:
        extras.append("text_encoder=offload")
    extras.append(f"ckpt={checkpoint}")
    extra_str = " " + " ".join(f"[{e}]" for e in extras)
    if decoder_accel == vae_accel:
        accel_str = f"accel={decoder_accel}"
    else:
        accel_str = f"accel=decoder:{decoder_accel}+vae:{vae_accel}"
    mode = "UI-ONLY (no backend)" if no_backend else f"WEB APP, {accel_str}{extra_str}"
    # Banner stays as print so the local-dev terminal keeps its
    # human-readable startup splash — structured event below is what
    # pod log collectors and analytics will key on.
    print()
    print("=" * 60)
    print(f"  Real-Time Motion-to-Music  ({mode})")
    print("=" * 60)
    print(f"  WebSocket: ws://{browsable_host}:{port}/")
    print(f"  HTTP API:  http://{browsable_host}:{port}/api/...")
    print(f"  Fixtures:  daydreamlive/demon-fixtures-v2 (HF, {len(KNOWN_FIXTURES)} files, on-demand)")
    print("  Ctrl+C to stop")
    print("=" * 60)
    print()
    logger.info(
        "server_ready host={} port={} mode={} no_backend={} kiosk={} "
        "default_mode={} checkpoint={} decoder_accel={} vae_accel={}",
        browsable_host, port, mode, no_backend, kiosk,
        default_mode, checkpoint, decoder_accel, vae_accel,
    )

    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        logger.info("server_shutdown reason=keyboard_interrupt")
        os._exit(0)


if __name__ == "__main__":
    main()
