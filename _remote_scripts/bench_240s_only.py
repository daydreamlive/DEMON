#!/usr/bin/env python3
"""Bench just the 2B turbo 240s rows after a rebuild and append to a CSV."""
import csv
import json
import os
import statistics
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYTHON = str(ROOT / ".venv" / "Scripts" / "python.exe")
PASS = str(ROOT / "_remote_scripts" / "bench_one_pass.py")

ROWS = [
    ("2B-turbo 240s",       "acestep-v15-turbo", "decoder_mixed_refit_b8_240s", 238.0, 8),
]

out_csv = Path(sys.argv[1])

results = []
for label, ckpt, dec, dur, steps in ROWS:
    env = os.environ.copy()
    env["BENCH_CHECKPOINT"] = ckpt
    env["BENCH_DECODER"] = dec
    env["BENCH_DURATION"] = str(dur)
    env["BENCH_INFER_STEPS"] = str(steps)
    env["BENCH_VAE"] = "1"
    env["BENCH_TAG"] = label.replace(" ", "_")
    print(f"\n=== {label} ===", flush=True)
    proc = subprocess.run([PYTHON, PASS, "tensorrt"], env=env, capture_output=True, text=True, cwd=str(ROOT))
    if proc.returncode != 0:
        print("FAILED")
        print("\n".join(proc.stderr.splitlines()[-20:]))
        continue
    res = None
    for line in proc.stdout.splitlines():
        if line.startswith("RESULT:"):
            res = json.loads(line[len("RESULT:"):])
            break
    if not res:
        continue
    timings = res["timings"]
    warm = timings[1:] if len(timings) > 1 else timings
    gens = [t["gen_s"]*1000 for t in warm]
    decs = [t["dec_s"]*1000 for t in warm]
    out = {
        "label": label,
        "kind": "tensorrt",
        "checkpoint": ckpt,
        "decoder": dec,
        "duration": dur,
        "steps": steps,
        "n_seeds": len(timings),
        "n_warm": len(warm),
        "gen_min_ms": min(gens),
        "gen_med_ms": statistics.median(gens),
        "gen_max_ms": max(gens),
        "dec_min_ms": min(decs),
        "dec_med_ms": statistics.median(decs),
        "dec_max_ms": max(decs),
        "total_med_ms": statistics.median(gens) + statistics.median(decs),
        "wall_s": 0.0,
    }
    print(f"  gen={out['gen_med_ms']:.0f}  dec={out['dec_med_ms']:.0f}  total={out['total_med_ms']:.0f}")
    results.append(out)

with open(out_csv, "w", newline="") as f:
    fields = list(results[0].keys()) if results else []
    if fields:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in results:
            w.writerow(r)
print(f"DONE -> {out_csv}")
