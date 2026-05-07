#!/usr/bin/env python3
"""Launch the realtime motion-graph demo: Python backend + Next.js frontend.

Spawns two child processes and tees their output with a `[backend]` /
`[web]` prefix so a single terminal shows both. Ctrl-C cleanly tears
both down.

Backend defaults to ``--host 127.0.0.1 --port 8765``. The Next.js dev
server uses ``next dev`` on the default port (3000); the rewrites in
``web/next.config.ts`` proxy ``/api/*``, ``/fixtures/*``, ``/loras/*``,
and ``/videos/*`` to the backend on 8765.

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
from pathlib import Path
from typing import IO


WEB_DIR = Path(__file__).parent / "web"
ROOT_DIR = Path(__file__).resolve().parents[2]


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
        print(f"{prefix} {line}", flush=True)


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


def main() -> int:
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
        "--no-install",
        action="store_true",
        help="Skip the `npm install` check.",
    )
    args, backend_extras = parser.parse_known_args()
    # `--` separator is preserved by parse_known_args; strip it if present.
    if backend_extras and backend_extras[0] == "--":
        backend_extras = backend_extras[1:]

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
        *backend_extras,
    ]
    web_cmd = [npm, "run", "dev", "--", "-p", str(args.web_port)]

    web_env = os.environ.copy()
    web_env["NEXT_PUBLIC_POD_BASE_URL"] = f"http://{args.host}:{args.port}"

    print(f"{_PREFIXES['backend']} {' '.join(backend_cmd)}", flush=True)
    print(
        f"{_PREFIXES['web']} (cwd={WEB_DIR}) {' '.join(web_cmd)}",
        flush=True,
    )
    banner = _color("\x1b[1;32m")
    print(
        f"\n{banner}>>> Open http://localhost:{args.web_port}/ "
        f"(NOT :{args.port} — that's the legacy static UI){_RESET}\n",
        flush=True,
    )

    backend = subprocess.Popen(
        backend_cmd,
        cwd=ROOT_DIR,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    web = subprocess.Popen(
        web_cmd,
        cwd=WEB_DIR,
        env=web_env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    threads = [
        threading.Thread(
            target=_tee, args=(backend.stdout, "backend"), daemon=True
        ),
        threading.Thread(target=_tee, args=(web.stdout, "web"), daemon=True),
    ]
    for t in threads:
        t.start()

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
