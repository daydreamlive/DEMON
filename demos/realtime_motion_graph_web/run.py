#!/usr/bin/env python3
"""Launch the realtime motion-graph demo: Python backend + Next.js frontend.

Spawns two child processes and tees their output with a `[backend]` /
`[web]` prefix so a single terminal shows both. Ctrl-C cleanly tears
both down.

Backend defaults to ``--host 127.0.0.1 --port 1318``. The Next.js dev
server uses ``next dev`` on port 6660; the rewrites in
``web/next.config.ts`` proxy ``/api/*``, ``/fixtures/*``, ``/loras/*``,
and ``/videos/*`` to the backend.

Run from the repo root::

    python -u -m demos.realtime_motion_graph_web.run

or directly::

    python -u demos/realtime_motion_graph_web/run.py

Pass any backend args after ``--``::

    python -u -m demos.realtime_motion_graph_web.run -- --accel eager
"""

from __future__ import annotations

import argparse
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import IO


WEB_DIR = Path(__file__).parent / "web"
ROOT_DIR = Path(__file__).resolve().parents[2]


def _harden_stdout() -> None:
    """On Windows a piped stdout defaults to the legacy code page
    (cp1252), which can't encode characters the children routinely emit
    (Next.js's "▲" logo, tqdm's "▎" blocks). An encode error would kill
    a _tee thread, and with no reader the child blocks on a full pipe
    and the demo freezes minutes later. Replace unencodable characters
    instead. stderr already defaults to errors="backslashreplace" and
    can't raise, so it's left alone.
    """
    try:
        sys.stdout.reconfigure(errors="replace")
    except (AttributeError, OSError, ValueError):
        pass


# ANSI dim/colour helpers for prefixing combined output. Falls back to no
# colour if stdout isn't a TTY (CI logs, redirected output).
def _color(code: str) -> str:
    return code if sys.stdout.isatty() else ""


_RESET = _color("\x1b[0m")
_PREFIXES = {
    "backend": _color("\x1b[36m") + "[backend]" + _RESET,
    "web": _color("\x1b[35m") + "[web]    " + _RESET,
}


def _tee(stream: IO[bytes], label: str) -> None:
    prefix = _PREFIXES[label]
    for raw in iter(stream.readline, b""):
        line = raw.decode("utf-8", errors="replace").rstrip("\n")
        # This thread is the child's only stdout reader; if it dies the
        # child eventually blocks on a full pipe and hangs the demo. No
        # single line is worth that.
        try:
            print(f"{prefix} {line}", flush=True)
        except Exception:
            pass


def _resolve_npm() -> str:
    # On Windows `npm` is `npm.cmd`; subprocess without shell=True only
    # resolves `.exe` via CreateProcess, so we look it up explicitly.
    npm = shutil.which("npm")
    if npm is None:
        sys.exit(
            "npm not found on PATH. Install Node.js 20+ "
            "(https://nodejs.org) and re-run."
        )
    return npm


def _ensure_node_modules(npm: str) -> None:
    if (WEB_DIR / "node_modules").is_dir():
        return
    print(
        f"{_PREFIXES['web']} node_modules missing — running `npm install`...",
        flush=True,
    )
    rc = subprocess.call([npm, "install"], cwd=WEB_DIR)
    if rc != 0:
        sys.exit(f"npm install exited with {rc}")


def _local_backend_host(host: str) -> str:
    return "127.0.0.1" if host in ("0.0.0.0", "::", "") else host


def _wait_for_backend(
    host: str,
    port: int,
    proc: subprocess.Popen[bytes],
) -> int | None:
    """Poll the backend until it serves HTTP, or until it exits.

    The probe is a real GET against the multiplexed HTTP side of the
    port (``/api/server-info``), not a bare TCP connect — a connection
    that opens and closes without sending a request line makes the
    websockets server log a full EOFError traceback right after
    ``server_ready``, which reads like a crash to a fresh user.
    """
    probe_host = _local_backend_host(host)
    if ":" in probe_host:  # bare IPv6 literal needs brackets in a URL
        probe_host = f"[{probe_host}]"
    url = f"http://{probe_host}:{port}/api/server-info"
    while proc.poll() is None:
        try:
            with urllib.request.urlopen(url, timeout=2.0):
                return None
        except urllib.error.HTTPError:
            # Any HTTP status means the server is up and parsing
            # requests; don't spin on a non-200.
            return None
        except OSError:
            time.sleep(0.25)
    return proc.returncode


def main() -> int:
    _harden_stdout()
    parser = argparse.ArgumentParser(
        description="Run the demo backend + Next.js frontend together.",
        epilog=(
            "Anything after `--` is forwarded to the backend "
            "(e.g. `-- --accel eager`)."
        ),
    )
    parser.add_argument("--port", type=int, default=1318, help="Backend port.")
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Backend bind host (default 127.0.0.1).",
    )
    parser.add_argument(
        "--web-port",
        type=int,
        default=6660,
        help="Next.js dev port (default 6660).",
    )
    parser.add_argument(
        "--client-host",
        default=None,
        help=(
            "Address the browser should use to reach the backend, for a "
            "remote client (UI and GPU server on different machines). The "
            "WebSocket connects to this directly, so it must be reachable "
            "from the client — e.g. the server's LAN IP. Defaults to the "
            "local backend address (fine when UI and backend share a host)."
        ),
    )
    parser.add_argument(
        "--no-install",
        action="store_true",
        help="Skip the `npm install` check.",
    )
    parser.add_argument(
        "--demo",
        action="append",
        default=[],
        metavar="PATH",
        help=(
            "Mount an external static demo repo. May be repeated. "
            "Equivalent to forwarding `--demo PATH` to the backend."
        ),
    )
    args, backend_extras = parser.parse_known_args()
    # `--` separator is preserved by parse_known_args; strip it if present.
    if backend_extras and backend_extras[0] == "--":
        backend_extras = backend_extras[1:]

    # Common footgun: launcher-only flags placed *after* `--` get forwarded to
    # the backend instead of parsed here, so e.g. --client-host silently has no
    # effect and the browser falls back to 127.0.0.1. Warn loudly.
    _launcher_only = {"--client-host", "--web-port", "--no-install"}
    _misplaced = [a for a in backend_extras if a.split("=", 1)[0] in _launcher_only]
    if _misplaced:
        print(
            f"WARNING: {', '.join(_misplaced)} came after `--`, so it was "
            f"forwarded to the backend, not parsed by the launcher (it has no "
            f"effect there). Put launcher flags BEFORE `--`, e.g.:\n"
            f"    run.py --host 0.0.0.0 --client-host <server-ip> -- --accel tensorrt",
            flush=True,
        )

    npm = _resolve_npm()
    if not args.no_install:
        _ensure_node_modules(npm)

    backend_cmd = [
        sys.executable,
        "-u",
        "-m",
        "demos.realtime_motion_graph_web",
        "--host",
        args.host,
        "--port",
        str(args.port),
        *[item for demo in args.demo for item in ("--demo", demo)],
        *backend_extras,
    ]
    # The browser uses NEXT_PUBLIC_POD_BASE_URL directly for the WebSocket, so
    # for a remote client it must point at an address the *client* can reach
    # (not 127.0.0.1). Precedence:
    #   1. --client-host <host>                    -> http://<host>:<port>
    #   2. an explicitly pre-set NEXT_PUBLIC_POD_BASE_URL in the environment
    #   3. the local backend address (default; fine when UI+backend share a host)
    remote = bool(args.client_host) or args.host in ("0.0.0.0", "::")

    web_env = os.environ.copy()
    if args.client_host:
        backend_url = f"http://{args.client_host}:{args.port}"
    elif os.environ.get("NEXT_PUBLIC_POD_BASE_URL"):
        backend_url = os.environ["NEXT_PUBLIC_POD_BASE_URL"].rstrip("/")
    else:
        backend_url = f"http://{_local_backend_host(args.host)}:{args.port}"
    web_env["NEXT_PUBLIC_POD_BASE_URL"] = backend_url

    # Build the web dev command. When the client is remote, bind the dev
    # server to all interfaces so it's reachable, and surface a hint if the
    # base URL still points at localhost (the browser can't reach that).
    web_cmd = [npm, "run", "dev", "--", "-p", str(args.web_port)]
    if remote:
        web_cmd += ["-H", "0.0.0.0"]
        if "localhost" in backend_url or "127.0.0.1" in backend_url:
            print(
                f"{_PREFIXES['web']} WARNING: NEXT_PUBLIC_POD_BASE_URL is "
                f"{backend_url}, which a remote browser can't reach. Pass "
                f"--client-host <server-ip> so the WebSocket connects to the "
                f"server, not the client.",
                flush=True,
            )
    print(f"{_PREFIXES['web']} engine base URL (for browser): {backend_url}", flush=True)

    print(f"{_PREFIXES['backend']} {' '.join(backend_cmd)}", flush=True)
    print(
        f"{_PREFIXES['web']} (cwd={WEB_DIR}) {' '.join(web_cmd)}",
        flush=True,
    )
    backend = subprocess.Popen(
        backend_cmd,
        cwd=ROOT_DIR,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    backend_thread = threading.Thread(
        target=_tee, args=(backend.stdout, "backend"), daemon=True
    )
    backend_thread.start()

    print(
        f"{_PREFIXES['backend']} waiting for {backend_url} before starting web",
        flush=True,
    )
    try:
        backend_rc = _wait_for_backend(args.host, args.port, backend)
    except KeyboardInterrupt:
        backend.terminate()
        try:
            backend.wait(timeout=5)
        except subprocess.TimeoutExpired:
            backend.kill()
            backend.wait()
        return 130
    if backend_rc is not None:
        return backend_rc or 1

    web = subprocess.Popen(
        web_cmd,
        cwd=WEB_DIR,
        env=web_env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    banner = _color("\x1b[1;32m")
    ui_host = args.client_host or "localhost"
    print(
        f"\n{banner}>>> Open http://{ui_host}:{args.web_port}/{_RESET}\n",
        flush=True,
    )

    threads = [
        backend_thread,
        threading.Thread(target=_tee, args=(web.stdout, "web"), daemon=True),
    ]
    threads[1].start()

    # Forward SIGINT/SIGTERM to children, then wait. The first child to
    # die brings the other down too — keeps the terminal honest about
    # whether either side crashed.
    def _shutdown(_signum=None, _frame=None) -> None:
        for proc in (web, backend):
            if proc.poll() is None:
                try:
                    proc.terminate()
                except ProcessLookupError:
                    pass

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    rc = 0
    try:
        while True:
            be = backend.poll()
            we = web.poll()
            if be is not None or we is not None:
                rc = (be if be is not None else we) or 0
                break
            try:
                backend.wait(timeout=0.5)
            except subprocess.TimeoutExpired:
                continue
    finally:
        _shutdown()
        for proc in (backend, web):
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()

    return rc


if __name__ == "__main__":
    raise SystemExit(main())
