"""Local-GPU server lifecycle for the golden harness.

A local GPU box is a first-class target: when no ``--pod-url`` /
``DEMON_POD_URL`` is given and CUDA is available, the suite (and
``runner --local``) spawns ``demos.realtime_motion_graph_web.server``
itself on a free port, waits for the HTTP layer, runs, and tears it
down.

Backend selection follows the server's own default (tensorrt) unless
``DEMON_TEST_ACCEL`` says otherwise (eager/compile/tensorrt) - useful
on a box without prebuilt TRT engines.
"""

import json
import os
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def cuda_available() -> bool:
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:
        return False


class LocalServer:
    """Spawn + own one local streaming server process."""

    def __init__(self, log_path: Path, accel: str | None = None,
                 startup_timeout_s: float = 600.0):
        self.port = _free_port()
        self.url = f"ws://127.0.0.1:{self.port}"
        self.log_path = Path(log_path)
        cmd = [sys.executable, "-m",
               "demos.realtime_motion_graph_web.server",
               "--port", str(self.port), "--no-control"]
        accel = accel or os.environ.get("DEMON_TEST_ACCEL")
        if accel:
            cmd += ["--accel", accel]
        env = dict(os.environ)
        env["PYTHONUTF8"] = "1"  # server logs vs Windows cp1252 console
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log = open(self.log_path, "w", encoding="utf-8")
        self.proc = subprocess.Popen(
            cmd, cwd=_REPO, env=env,
            stdout=self._log, stderr=subprocess.STDOUT)
        self._wait_http(startup_timeout_s)

    def _wait_http(self, timeout_s: float) -> None:
        deadline = time.monotonic() + timeout_s
        info_url = f"http://127.0.0.1:{self.port}/api/server-info"
        while time.monotonic() < deadline:
            if self.proc.poll() is not None:
                raise RuntimeError(
                    f"local server exited with {self.proc.returncode} "
                    f"during startup - see {self.log_path}")
            try:
                with urllib.request.urlopen(info_url, timeout=2) as r:
                    info = json.loads(r.read().decode("utf-8"))
                if not info.get("no_backend"):
                    return
                raise RuntimeError(
                    "local server came up in --no-backend mode; a GPU "
                    "build is required for the golden suite")
            except (urllib.error.URLError, ConnectionError, OSError,
                    TimeoutError):
                time.sleep(1.0)
        self.stop()
        raise TimeoutError(
            f"local server HTTP layer not up within {timeout_s}s - "
            f"see {self.log_path}")

    def stop(self) -> None:
        try:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait(timeout=15)
        finally:
            try:
                self._log.close()
            except Exception:
                pass
