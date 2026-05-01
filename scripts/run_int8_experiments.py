#!/usr/bin/env python3
"""Build + bench a matrix of INT8 VAE engines and write a summary report.

Variants (all share the same 60s profile: min=125, opt=1500, max=1500
and the calibration latents in <models>/calibration/vae_latents_60s):

  A: entropy calib, no pinning           -> vae_decode_int8_60s
  B: minmax calib, no pinning            -> vae_decode_int8_60s_minmax
  C: entropy calib, pin first+last conv  -> vae_decode_int8_60s_pin
  D: <best calibrator>, pin first+last   -> vae_decode_int8_60s_combined

Usage:
    uv run python scripts/run_int8_experiments.py
    uv run python scripts/run_int8_experiments.py --skip-build A C
    uv run python scripts/run_int8_experiments.py --skip A B   # don't run A or B
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from acestep.paths import models_dir, trt_engines_dir


CALIB_DIR_60S = models_dir() / "calibration" / "vae_latents_60s"
RESULTS_DIR = ROOT / "bench_outputs" / "vae_int8"
JSON_DIR = RESULTS_DIR / "json"


VARIANTS = [
    {
        "id": "A",
        "engine": "vae_decode_int8_60s",
        "build_args": ["--calibrator", "entropy"],
        "desc": "entropy calib, no pinning",
    },
    {
        "id": "B",
        "engine": "vae_decode_int8_60s_minmax",
        "build_args": ["--calibrator", "minmax"],
        "desc": "minmax calib, no pinning",
    },
    {
        "id": "C",
        "engine": "vae_decode_int8_60s_pin",
        "build_args": ["--calibrator", "entropy", "--pin-first", "1", "--pin-last", "1"],
        "desc": "entropy calib, pin first+last conv to fp16",
    },
    # D is filled in dynamically based on B-vs-A quality
]


def _engine_path(name: str) -> Path:
    return trt_engines_dir() / name / f"{name}.engine"


def _run(cmd: list[str], desc: str, log_path: Path):
    """Run a subprocess unbuffered, tee log to file, raise on failure."""
    print(f"\n{'=' * 78}\n[{time.strftime('%H:%M:%S')}] {desc}\n  {' '.join(cmd)}\n{'=' * 78}", flush=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    t0 = time.time()
    with open(log_path, "w") as logf:
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            cwd=str(ROOT), env=env, text=True, bufsize=1,
        )
        for line in proc.stdout:
            sys.stdout.write(line); sys.stdout.flush()
            logf.write(line); logf.flush()
        rc = proc.wait()
    elapsed = time.time() - t0
    print(f"[{time.strftime('%H:%M:%S')}] -> exit={rc}, {elapsed:.0f}s, log={log_path}", flush=True)
    if rc != 0:
        raise RuntimeError(f"Failed: {desc}")
    return elapsed


def build_variant(v: dict, *, force: bool = False) -> float:
    eng = _engine_path(v["engine"])
    if eng.exists() and not force:
        print(f"[skip-build] {v['engine']} exists ({eng.stat().st_size / 1e6:.1f} MB)", flush=True)
        return 0.0
    cmd = [
        "uv", "run", "python", "scripts/build_vae_int8_engine.py",
        "--engine-name", v["engine"],
        "--calib-dir", str(CALIB_DIR_60S),
        "--min-frames", "125",
        "--opt-frames", "1500",
        "--max-frames", "1500",
    ] + v["build_args"]
    log_path = JSON_DIR / f"build_{v['id']}_{v['engine']}.log"
    return _run(cmd, f"BUILD {v['id']}: {v['desc']}", log_path)


def bench_variant(v: dict) -> Path:
    json_out = JSON_DIR / f"{v['id']}_{v['engine']}.json"
    out_dir = RESULTS_DIR / v["engine"]
    cmd = [
        "uv", "run", "python", "tests/benchmarks/bench_vae_int8_regular.py",
        "--int8-engine", v["engine"],
        "--fp16-engine", "vae_decode_fp16_60s",
        "--chunk-frames", "1500",
        "--duration", "60",
        "--runs", "5",
        "--out-dir", str(out_dir),
        "--json-out", str(json_out),
        "--label", f"{v['id']} ({v['desc']})",
    ]
    log_path = JSON_DIR / f"bench_{v['id']}_{v['engine']}.log"
    _run(cmd, f"BENCH {v['id']}: {v['engine']}", log_path)
    return json_out


def _report_row(label, j):
    syn = j.get("synthetic", {})
    wav = j.get("wav_roundtrip", {})
    return {
        "label": label,
        "engine_mb": j.get("engine_size_mb"),
        "syn_int8_ms": syn.get("int8", {}).get("median_ms"),
        "syn_int8_mse_vs_pt": syn.get("int8", {}).get("mse_vs_pt"),
        "syn_fp16_ms": syn.get("fp16", {}).get("median_ms"),
        "syn_fp16_mse_vs_pt": syn.get("fp16", {}).get("mse_vs_pt"),
        "syn_fp16_int8_psnr": syn.get("fp16_vs_int8", {}).get("psnr_db"),
        "wav_int8_ms": wav.get("decode", {}).get("int8", {}).get("median_ms"),
        "wav_int8_mse_vs_pt": wav.get("decode", {}).get("int8", {}).get("mse_vs_pt"),
        "wav_int8_psnr_vs_orig": wav.get("vs_original_audio", {}).get("int8", {}).get("psnr_db"),
        "wav_fp16_psnr_vs_orig": wav.get("vs_original_audio", {}).get("fp16", {}).get("psnr_db"),
        "wav_pt_psnr_vs_orig": wav.get("vs_original_audio", {}).get("pt", {}).get("psnr_db"),
        "wav_fp16_int8_psnr": wav.get("fp16_vs_int8", {}).get("psnr_db"),
    }


def write_report(rows: list[dict], path: Path):
    """Write a markdown report comparing variants."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("# INT8 VAE decode -- variant comparison\n\n")
        f.write(f"_{time.strftime('%Y-%m-%d %H:%M:%S')}_\n\n")
        f.write("All variants share the same 60s decode profile "
                "(min=125, opt=1500, max=1500) and the same 32-latent calibration "
                "set at 1500 frames each. fp16 baseline is `vae_decode_fp16_60s`.\n\n")

        f.write("## Synthetic latent (60s chunk)\n\n")
        f.write("| Variant | Engine | Engine MB | int8 ms | int8 MSE vs PT | fp16-vs-int8 PSNR (dB) |\n")
        f.write("|---|---|---:|---:|---:|---:|\n")
        for r in rows:
            f.write(f"| {r['label']} | "
                    f"`{Path(r.get('engine_path', '')).name if 'engine_path' in r else ''}` | "
                    f"{r['engine_mb']:.0f} | "
                    f"{(r['syn_int8_ms'] or 0):.1f} | "
                    f"{r['syn_int8_mse_vs_pt']:.2e} | "
                    f"{r['syn_fp16_int8_psnr']:.2f} |\n")
        f.write("\n")

        f.write("## Wav round-trip (encode->decode of "
                "`tests/fixtures/inside_confusion.wav`, first 60s)\n\n")
        f.write("| Variant | int8 ms | int8 PSNR vs original | fp16 PSNR vs original | PT PSNR vs original | fp16-vs-int8 PSNR |\n")
        f.write("|---|---:|---:|---:|---:|---:|\n")
        for r in rows:
            f.write(f"| {r['label']} | "
                    f"{(r['wav_int8_ms'] or 0):.1f} | "
                    f"{r['wav_int8_psnr_vs_orig']:.2f} | "
                    f"{r['wav_fp16_psnr_vs_orig']:.2f} | "
                    f"{r['wav_pt_psnr_vs_orig']:.2f} | "
                    f"{r['wav_fp16_int8_psnr']:.2f} |\n")
        f.write("\n")
        f.write("PSNR vs original audio is the most production-relevant metric: "
                "higher is better, and PT is the ceiling.\n")

    print(f"\nReport: {path}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-build", nargs="*", default=[],
                    help="Variant IDs whose build to skip (use existing engine)")
    ap.add_argument("--skip", nargs="*", default=[],
                    help="Variant IDs to skip entirely")
    args = ap.parse_args()

    JSON_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if not CALIB_DIR_60S.exists() or not list(CALIB_DIR_60S.glob("latent_*.pt")):
        print(f"FATAL: missing 60s calibration latents at {CALIB_DIR_60S}")
        print("Run scripts/collect_vae_calibration.py --frames 1500 --output-dir <path> first.")
        sys.exit(1)

    # Run A, B, C
    benches: dict[str, Path] = {}
    for v in VARIANTS:
        if v["id"] in args.skip:
            print(f"[skip] variant {v['id']}", flush=True)
            continue
        if v["id"] not in args.skip_build:
            build_variant(v)
        else:
            print(f"[skip-build flag] {v['engine']}", flush=True)
        benches[v["id"]] = bench_variant(v)

    # Decide D's calibrator from A vs B quality (synthetic int8 MSE vs PT)
    if "A" in benches and "B" in benches:
        a_data = json.loads(benches["A"].read_text())
        b_data = json.loads(benches["B"].read_text())
        a_mse = a_data.get("synthetic", {}).get("int8", {}).get("mse_vs_pt", float("inf"))
        b_mse = b_data.get("synthetic", {}).get("int8", {}).get("mse_vs_pt", float("inf"))
        winner = "entropy" if a_mse <= b_mse else "minmax"
        print(f"\n>>> D calibrator chosen: {winner}  (A={a_mse:.2e}  B={b_mse:.2e})")
    else:
        winner = "entropy"

    if "D" not in args.skip:
        v_d = {
            "id": "D",
            "engine": "vae_decode_int8_60s_combined",
            "build_args": ["--calibrator", winner, "--pin-first", "1", "--pin-last", "1"],
            "desc": f"{winner} calib + pin first+last conv to fp16",
        }
        if "D" not in args.skip_build:
            build_variant(v_d)
        benches["D"] = bench_variant(v_d)

    # Aggregate report
    rows = []
    for v in VARIANTS:
        if v["id"] in benches:
            j = json.loads(benches[v["id"]].read_text())
            r = _report_row(j["label"], j)
            r["engine_path"] = j["int8_engine"]
            rows.append(r)
    if "D" in benches:
        j = json.loads(benches["D"].read_text())
        r = _report_row(j["label"], j)
        r["engine_path"] = j["int8_engine"]
        rows.append(r)

    write_report(rows, RESULTS_DIR / "REPORT.md")


if __name__ == "__main__":
    main()
