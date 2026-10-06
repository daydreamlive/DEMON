"""SA3 real-server latency probe (session start, knob-to-ear, prompt change).

Spawns ``demos.realtime_motion_graph_web.server --checkpoint <alias>`` on a
free port, then drives one session per depth through the golden harness
(``tests.golden.runner.run_scenario``, black-box over the wire) with two
frontier-triggered actions: an ``sa3_denoise`` knob step and a prompt
change. Per run it records config->ready, ready->first slice, live
tick/decode ms, and per action the ack gap plus audible first/full
(the playhead reaching the first re-generated slice / the generation
frontier as of the send). Server log ``session_vram`` lines are copied
into the report when present.

Run (repo root):
    python scripts/sa3/sa3_latency_probe.py --checkpoint sa3-sfx \
        --accel tensorrt --duration 20 --depths 1 4 --out E:/.../latency
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from tests.golden.local_server import LocalServer  # noqa: E402
from tests.golden.runner import run_scenario  # noqa: E402
from tests.golden.scenarios import FIXTURE_A, Action, Scenario  # noqa: E402

PROMPT_A = "Steady heavy rain on a tin roof"
PROMPT_B = "Busy city street traffic ambience, distant horns"


class _Server(LocalServer):
    """LocalServer plus ``--checkpoint``: the harness class has no
    checkpoint argument, so append it to the spawned command line."""

    def __init__(self, log_path: Path, accel: str, checkpoint: str):
        import subprocess

        orig = subprocess.Popen

        def popen(cmd, *a, **kw):
            return orig([*cmd, "--checkpoint", checkpoint], *a, **kw)

        subprocess.Popen = popen
        try:
            super().__init__(log_path, accel=accel)
        finally:
            subprocess.Popen = orig


def scenario(duration: float, depth: int) -> Scenario:
    # Actions at 30 % / 60 % of the canvas so both land on the first lap
    # of a short loop; the compared region is irrelevant here (latency
    # only) but must fit inside the canvas for the run to terminate.
    return Scenario(
        name=f"sa3_latency_d{depth}",
        fixture=FIXTURE_A,
        warmup_skip_s=min(6.0, 0.1 * duration),
        canonical_s=max(1.0, 0.5 * duration),
        settle_s=min(5.0, 0.1 * duration),
        timeout_s=240.0,
        config={
            "telemetry_version": 1,
            "backend": "sa3",
            "prompt": PROMPT_A,
            "sa3_duration_s": duration,
            "depth": depth,
        },
        actions=[
            Action(round(0.3 * duration, 2), "params", {"raw": {"sa3_denoise": 0.55}}),
            Action(round(0.6 * duration, 2), "prompt", {"tags": PROMPT_B}),
        ],
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--checkpoint", default="sa3-sfx")
    ap.add_argument("--accel", choices=("eager", "tensorrt"), default="tensorrt")
    ap.add_argument("--duration", type=float, default=20.0)
    ap.add_argument("--depths", type=int, nargs="+", default=[1, 4])
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    log = out / "server.log"
    server = _Server(log, args.accel, args.checkpoint)
    reports = []
    try:
        for depth in args.depths:
            run_dir = out / f"{args.accel}-d{depth}"
            res = run_scenario(server.url, scenario(args.duration, depth), run_dir,
                               save_blobs=False)
            keep = {k: res.get(k) for k in (
                "status", "error", "ready", "t_config_to_ready_s",
                "t_ready_to_first_slice_s", "tick_ms", "dec_ms", "slice_gap_ms",
                "lead_s", "n_slices", "realtime_factor", "wall_s")}
            keep["actions"] = [
                {k: a.get(k) for k in ("kind", "at_s", "ack_event", "ack_gap_ms",
                                       "next_slice_gap_ms", "audible_first_ms",
                                       "audible_full_ms")}
                for a in res.get("actions", [])
            ]
            keep["depth"] = depth
            reports.append(keep)
            print(json.dumps(keep), flush=True)
    finally:
        server.stop()
    vram = [ln.strip() for ln in log.read_text(encoding="utf-8", errors="replace").splitlines()
            if re.search(r"session_vram|sa3_dit_(eager|refit|fp8)|sa3_trt_dit_ready|sa3_session_create", ln)]
    summary = {"checkpoint": args.checkpoint, "accel": args.accel,
               "duration_s": args.duration, "runs": reports, "server_log_lines": vram[-40:]}
    (out / f"report-{args.accel}.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
