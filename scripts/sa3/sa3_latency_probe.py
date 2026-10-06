"""SA3 real-server latency probe (black box, over the WebSocket protocol).

Spawns ``demos.realtime_motion_graph_web.server`` with the requested
checkpoint/accel on a free port (or targets ``--pod-url``) and drives
ONE 54 s SA3 session through the golden harness's recording client
(:func:`tests.golden.runner.run_scenario`), so every number uses the
harness's definitions:

* ``t_config_to_ready_s``: session config sent -> ``ready`` frame.
* ``t_ready_to_first_slice_s``: ``ready`` -> first audio slice.
* per action: ``ack_gap_ms`` (prompt -> ``prompt_applied``),
  ``effect``-side ``audible_first_ms`` (playhead reaches the first
  post-action slice; partial effect) and ``audible_full_ms`` (playhead
  reaches the generation frontier as of the send: every step of that
  window ran with the new value).
* stream: ``tick_ms`` / ``dec_ms`` / ``slice_gap_ms`` from slice meta.

Actions fire on GENERATION-FRONTIER positions: ``sa3_denoise`` knob
flips at 8 / 16 / 24 s (0.55, 0.85, 0.35) and prompt changes at 32 /
42 s. VRAM per session comes from the server log's ``session_vram``
lines (torch allocated/reserved + device free) plus ``nvidia-smi``
sampled while the session runs.

    .venv/Scripts/python.exe scripts/sa3/sa3_latency_probe.py \
        --checkpoint sa3-small --accel tensorrt --depth 4 \
        --out E:/Projects/sa3-variants/small/latency/trt-d4
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from tests.golden.scenarios import Action, Scenario  # noqa: E402

PROMPT_A = "lofi hip hop, mellow, instrumental"
ACTIONS = [
    Action(8.0, "params", {"raw": {"sa3_denoise": 0.55}}),
    Action(16.0, "params", {"raw": {"sa3_denoise": 0.85}}),
    Action(24.0, "params", {"raw": {"sa3_denoise": 0.35}}),
    Action(32.0, "prompt", {"tags": "aggressive industrial techno, distorted"}),
    Action(42.0, "prompt", {"tags": "warm jazz trio, brushed drums, upright bass"}),
]
_VRAM_RE = re.compile(
    r"session_vram stage=(\w+) free_gb=([\d.]+) available_gb=([\d.]+) "
    r"allocated_gb=([\d.]+) reserved_gb=([\d.]+)")


def _nvidia_smi_used_mib() -> int | None:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10).stdout
        return int(out.strip().splitlines()[0])
    except Exception:
        return None


class _SmiSampler(threading.Thread):
    def __init__(self, period_s: float = 1.0):
        super().__init__(daemon=True)
        self.period_s = period_s
        self.samples: list[int] = []
        self._halt = threading.Event()

    def run(self) -> None:
        while not self._halt.is_set():
            v = _nvidia_smi_used_mib()
            if v is not None:
                self.samples.append(v)
            self._halt.wait(self.period_s)

    def stop(self) -> None:
        self._halt.set()
        self.join(timeout=15)


def _spawn_server(checkpoint: str, accel: str, log_path: Path):
    """LocalServer with a checkpoint arg (the harness class has none)."""
    from tests.golden import local_server as ls

    class _Server(ls.LocalServer):
        def __init__(self):
            import os
            self.port = ls._free_port()
            self.url = f"ws://127.0.0.1:{self.port}"
            self.log_path = Path(log_path)
            cmd = [sys.executable, "-m", "demos.realtime_motion_graph_web.server",
                   "--port", str(self.port), "--no-control",
                   "--checkpoint", checkpoint, "--accel", accel]
            env = dict(os.environ)
            env["PYTHONUTF8"] = "1"
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            self._log = open(self.log_path, "w", encoding="utf-8")
            self.proc = subprocess.Popen(cmd, cwd=REPO, env=env,
                                         stdout=self._log, stderr=subprocess.STDOUT)
            self._wait_http(900.0)

    return _Server()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--checkpoint", default="sa3-small")
    ap.add_argument("--accel", default="tensorrt", choices=("tensorrt", "eager"))
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--duration", type=float, default=54.0)
    ap.add_argument("--pod-url", default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from tests.golden.runner import run_scenario

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    idle_mib = _nvidia_smi_used_mib()
    server = None
    log_path = out / "server.log"
    pod_url = args.pod_url
    t_boot = time.monotonic()
    if not pod_url:
        server = _spawn_server(args.checkpoint, args.accel, log_path)
        pod_url = server.url
    boot_s = round(time.monotonic() - t_boot, 1)
    sampler = _SmiSampler()
    sampler.start()
    try:
        sc = Scenario(
            name=f"sa3_latency_d{args.depth}",
            warmup_skip_s=6.0,
            # Region spans to the end of the loop so the run keeps going
            # until the last prompt change has landed everywhere.
            canonical_s=args.duration - 6.0,
            settle_s=5.0,
            timeout_s=300.0,
            config={"depth": args.depth, "steps": 8, "prompt": PROMPT_A,
                    "sa3_duration_s": args.duration},
            # Frontier positions wrap with the loop: a shorter canvas keeps
            # only the actions that land inside it (20 s: the two first
            # knob flips, no prompt change).
            actions=[a for a in ACTIONS if a.at_s < args.duration - 2.0],
        )
        result = run_scenario(pod_url, sc, out / "bundle", save_blobs=False)
    finally:
        sampler.stop()
        if server is not None:
            server.stop()

    vram = []
    if log_path.is_file():
        for m in _VRAM_RE.finditer(log_path.read_text(encoding="utf-8", errors="replace")):
            vram.append({"stage": m.group(1), "free_gb": float(m.group(2)),
                         "available_gb": float(m.group(3)),
                         "allocated_gb": float(m.group(4)),
                         "reserved_gb": float(m.group(5))})
    report = {
        "checkpoint": args.checkpoint, "accel": args.accel, "depth": args.depth,
        "duration_s": args.duration, "server_boot_s": boot_s,
        "nvidia_smi_idle_mib": idle_mib,
        "nvidia_smi_peak_mib": max(sampler.samples) if sampler.samples else None,
        "session_vram": vram,
        **result,
    }
    (out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    keep = ("status", "t_config_to_ready_s", "t_ready_to_first_slice_s",
            "tick_ms", "dec_ms", "nvidia_smi_peak_mib")
    print(json.dumps({k: report.get(k) for k in keep}))
    for a in report.get("actions", []):
        print(json.dumps({k: a.get(k) for k in (
            "kind", "at_s", "ack_gap_ms", "next_slice_gap_ms",
            "audible_first_ms", "audible_full_ms")}))
    return 0 if report.get("status") == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
