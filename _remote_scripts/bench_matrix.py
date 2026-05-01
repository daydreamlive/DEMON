#!/usr/bin/env python3
"""Drive bench_one_pass.py over the full TRT engine matrix.

For each row, spawn bench_one_pass.py in an isolated subprocess so each
measurement starts with a clean CUDA state. Capture the RESULT JSON,
drop the first seed as warmup, compute median of the rest, and write a
CSV.

Usage:
    python bench_matrix.py <out_csv> [pt_only|trt_only|all]

Default mode is "all": one PT row per (checkpoint, duration) plus all TRT
rows. "trt_only" skips the (slow) PT baselines.
"""
import csv
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BENCH_PASS = str(ROOT / "_remote_scripts" / "bench_canonical.py")
PYTHON = str(ROOT / ".venv" / "Scripts" / "python.exe")

# (label, checkpoint, decoder_dir, duration_s, infer_steps, kind)
# kind: "trt" requires the decoder; "pt" ignores it.
TRT_ROWS = [
    # 2B turbo (PRIMARY)
    ("2B-turbo 240s",              "acestep-v15-turbo",     "decoder_mixed_refit_b8_240s",                  238.0, 8),
    # 2B base
    ("2B-base 240s",               "acestep-v15-base",      "decoder_base_mixed_refit_b8_240s",             238.0, 8),
    # XL turbo
    ("XL-turbo 240s",              "acestep-v15-xl-turbo",  "decoder_xl-turbo_bf16mix_dynbatch_b8_240s",    238.0, 8),
]

# PT baselines: one per unique (checkpoint, duration) — version-independent.
PT_ROWS = [
    ("PT 2B-turbo 60s",  "acestep-v15-turbo",     None,  58.0, 8),
    ("PT 2B-turbo 240s", "acestep-v15-turbo",     None, 238.0, 8),
    ("PT 2B-base 60s",   "acestep-v15-base",      None,  58.0, 8),
    ("PT XL-turbo 60s",  "acestep-v15-xl-turbo",  None,  58.0, 8),
]


def run_one(label, checkpoint, decoder, duration, steps, kind):
    """Run a single bench_one_pass.py invocation. Returns dict with timings."""
    env = os.environ.copy()
    env["BENCH_CHECKPOINT"] = checkpoint
    env["BENCH_DURATION"] = str(duration)
    env["BENCH_INFER_STEPS"] = str(steps)
    env["BENCH_VAE"] = "1"
    env["BENCH_LABEL"] = label
    if decoder:
        env["BENCH_DECODER"] = decoder
    arg = "tensorrt" if kind == "trt" else "pytorch"

    print(f"\n=== {label} ({arg}) ===", flush=True)
    print(f"   checkpoint={checkpoint} decoder={decoder} dur={duration} steps={steps}", flush=True)

    t0 = time.time()
    proc = subprocess.run(
        [PYTHON, BENCH_PASS, arg],
        env=env,
        capture_output=True,
        text=True,
        cwd=str(ROOT),
    )
    elapsed = time.time() - t0

    if proc.returncode != 0:
        print(f"   FAILED rc={proc.returncode} in {elapsed:.1f}s", flush=True)
        print("   --- stderr tail ---", flush=True)
        print("\n".join(proc.stderr.splitlines()[-20:]), flush=True)
        print("   --- stdout tail ---", flush=True)
        print("\n".join(proc.stdout.splitlines()[-20:]), flush=True)
        return None

    result = None
    for line in proc.stdout.splitlines():
        if line.startswith("RESULT:"):
            result = json.loads(line[len("RESULT:"):])
            break
    if result is None:
        print(f"   no RESULT line in subprocess output ({elapsed:.1f}s)", flush=True)
        print("   --- stdout tail ---", flush=True)
        print("\n".join(proc.stdout.splitlines()[-20:]), flush=True)
        return None

    out = {
        "label": label,
        "kind": arg,
        "checkpoint": checkpoint,
        "decoder": decoder or "",
        "duration": duration,
        "steps": steps,
        "n_seeds": result["runs"],
        "n_warm": result["warmup"],
        "gen_min_ms": result["gen_min_ms"],
        "gen_med_ms": result["gen_med_ms"],
        "gen_max_ms": result["gen_max_ms"],
        "dec_min_ms": result["dec_min_ms"],
        "dec_med_ms": result["dec_med_ms"],
        "dec_max_ms": result["dec_max_ms"],
        "total_med_ms": result["total_med_ms"],
        "wall_s": elapsed,
    }
    print(f"   gen={out['gen_med_ms']:.0f}ms  dec={out['dec_med_ms']:.0f}ms  total={out['total_med_ms']:.0f}ms  wall={elapsed:.1f}s", flush=True)
    return out


def main():
    out_csv = Path(sys.argv[1])
    mode = sys.argv[2] if len(sys.argv) > 2 else "all"

    rows = []
    if mode in ("all", "trt_only"):
        for label, ckpt, dec, dur, steps in TRT_ROWS:
            rows.append((label, ckpt, dec, dur, steps, "trt"))
    if mode in ("all", "pt_only"):
        for label, ckpt, dec, dur, steps in PT_ROWS:
            rows.append((label, ckpt, dec, dur, steps, "pt"))

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "label", "kind", "checkpoint", "decoder", "duration", "steps",
        "n_seeds", "n_warm",
        "gen_min_ms", "gen_med_ms", "gen_max_ms",
        "dec_min_ms", "dec_med_ms", "dec_max_ms",
        "total_med_ms", "wall_s",
    ]
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()

        for row in rows:
            result = run_one(*row)
            if result:
                writer.writerow(result)
                f.flush()

    print(f"\nDONE -> {out_csv}", flush=True)


if __name__ == "__main__":
    main()
