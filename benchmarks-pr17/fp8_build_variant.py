"""Build + listen + validate + archive one FP8 engine variant.

Single command that:
  1. Re-patches the bf16 ONNX with the given FP8 args.
  2. Builds the TRT engine (force-rebuild).
  3. Runs the validation script (cos sim + tick latency).
  4. Runs the listen test with --fp8-tag <tag>.
  5. Copies engine, manifest, validation results, and audio into
     benchmarks-pr17/variants/<tag>/.
  6. Appends a row to benchmarks-pr17/variants/INDEX.md.

Engines are never deleted. Every variant gets its own folder. The
INDEX.md is the metrics table — always kept up to date.

Example::

    python benchmarks-pr17/fp8_build_variant.py \\
        --tag w8a8_absmax \\
        --description "Pure W8A8 absmax: no SmoothQuant, no skip" \\
        --smoothquant-alpha 0.0 \\
        --outlier-skip-ratio 0.0

    python benchmarks-pr17/fp8_build_variant.py \\
        --tag sq_a05_full \\
        --description "SmoothQuant alpha=0.5 full coverage" \\
        --smoothquant-alpha 0.5 \\
        --outlier-skip-ratio 0.0
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

ENGINE_PATH = Path(os.path.expanduser(
    "~/.daydream-scope/models/demon/trt_engines/"
    "decoder_xl-turbo_fp8_refit_b4_60s/"
    "decoder_xl-turbo_fp8_refit_b4_60s.engine"
))
ENGINE_META = ENGINE_PATH.with_suffix(ENGINE_PATH.suffix + ".metadata.json")
PATCH_MANIFEST = Path(os.path.expanduser(
    "~/.daydream-scope/models/demon/trt_engines/_onnx_acestep-v15-xl-turbo/"
    "decoder_refit/decoder_refit_dynbatch_fp8_manifest.json"
))
CALIBRATION_NPZ = Path(os.path.expanduser(
    "~/.daydream-scope/models/demon/calibration/decoder_xl_fp8/calibration.npz"
))
ABSMAX_JSON = Path(os.path.expanduser(
    "~/.daydream-scope/models/demon/calibration/decoder_xl_fp8/activation_absmax.json"
))
LISTEN_ROOT = REPO / "benchmarks-pr17" / "listen"
VARIANTS_ROOT = REPO / "benchmarks-pr17" / "variants"
INDEX_MD = VARIANTS_ROOT / "INDEX.md"
VALIDATE_JSON = REPO / "benchmarks-pr17" / "fp8_vs_bf16_results.json"


def _run(cmd: list[str], *, cwd: Path = REPO) -> None:
    """Run a subprocess, streaming output. Exits on failure."""
    print(f"\n$ {' '.join(cmd)}", flush=True)
    t0 = time.perf_counter()
    res = subprocess.run(cmd, cwd=cwd)
    dt = time.perf_counter() - t0
    if res.returncode != 0:
        print(f"FAILED ({dt:.1f}s): {' '.join(cmd)}", file=sys.stderr)
        sys.exit(res.returncode)
    print(f"  ({dt:.1f}s)", flush=True)


def _patch(*, smoothquant_alpha: float, outlier_skip_ratio: float,
           percentile: str, w8a8: bool,
           quantize_attention: bool = False,
           attention_softmax_max: float = 1.05,
           attention_generic_max: float = 10.0) -> None:
    cmd = [
        sys.executable, "benchmarks-pr17/run_fp8_patch.py",
    ]
    if w8a8:
        cmd.append("--w8a8")
    cmd.extend([
        "--percentile", percentile,
        "--outlier-skip-ratio", str(outlier_skip_ratio),
        "--smoothquant-alpha", str(smoothquant_alpha),
    ])
    if quantize_attention:
        cmd.extend([
            "--quantize-attention",
            "--attention-softmax-max", str(attention_softmax_max),
            "--attention-generic-max", str(attention_generic_max),
        ])
    _run(cmd)


def _build(*, smoothquant_alpha: float, w8a8: bool, percentile: str) -> None:
    cmd = [
        sys.executable, "-m", "acestep.engine.trt.build",
        "--checkpoint", "acestep-v15-xl-turbo",
        "--max-duration", "60",
        "--batch-max", "4",
        "--decoder",
        "--decoder-precision", "fp8_mixed",
        "--skip-vae",
        "--skip-onnx",
        "--force-rebuild",
        "--smoothquant-alpha", str(smoothquant_alpha),
        "--activation-percentile", percentile,
    ]
    if w8a8:
        cmd.extend(["--activation-absmax-json", str(ABSMAX_JSON)])
    _run(cmd)


def _validate() -> None:
    _run([sys.executable, "benchmarks-pr17/fp8_vs_bf16_validate.py"])


def _listen(tag: str) -> None:
    _run([
        sys.executable, "benchmarks-pr17/fp8_listen_test.py",
        "--duration", "30",
        "--only", "fp8",
        "--fp8-tag", tag,
    ])


def _archive(*, tag: str, description: str, cli_args: dict) -> dict:
    """Copy engine + metadata + manifest + audio + results into the
    variants folder. Returns the metrics dict used for INDEX.md.
    """
    variant_dir = VARIANTS_ROOT / tag
    variant_dir.mkdir(parents=True, exist_ok=True)

    # Source files we always want preserved.
    sources = [
        (ENGINE_PATH, variant_dir / "engine.engine"),
        (ENGINE_META, variant_dir / "engine.metadata.json"),
        (PATCH_MANIFEST, variant_dir / "fp8_patch_manifest.json"),
        (VALIDATE_JSON, variant_dir / "validation_results.json"),
    ]
    for src, dst in sources:
        if not src.exists():
            print(f"WARN: source missing, skipping: {src}", file=sys.stderr)
            continue
        print(f"  copy {src.name} -> {dst.relative_to(REPO)} ({src.stat().st_size/1e6:.1f} MB)")
        shutil.copy2(src, dst)

    # Audio: the listen test wrote to listen/<tag>/. Copy into archive.
    audio_src = LISTEN_ROOT / tag
    audio_dst = variant_dir / "audio"
    if audio_src.exists():
        if audio_dst.exists():
            print(f"  audio dst already exists, removing: {audio_dst}")
            shutil.rmtree(audio_dst)
        shutil.copytree(audio_src, audio_dst)
        print(f"  copy audio {audio_src.relative_to(REPO)} -> {audio_dst.relative_to(REPO)}")

    # config.json: how to reproduce this exact variant.
    config = {
        "tag": tag,
        "description": description,
        "cli_args": cli_args,
        "reproduce": {
            "patch": (
                f"python benchmarks-pr17/run_fp8_patch.py --w8a8 "
                f"--percentile {cli_args['percentile']} "
                f"--outlier-skip-ratio {cli_args['outlier_skip_ratio']} "
                f"--smoothquant-alpha {cli_args['smoothquant_alpha']}"
            ),
            "build": (
                f"python -m acestep.engine.trt.build --checkpoint acestep-v15-xl-turbo "
                f"--max-duration 60 --batch-max 4 --decoder --decoder-precision fp8_mixed "
                f"--activation-absmax-json {ABSMAX_JSON} "
                f"--smoothquant-alpha {cli_args['smoothquant_alpha']} "
                f"--activation-percentile {cli_args['percentile']} "
                f"--skip-vae --skip-onnx --force-rebuild"
            ),
            "listen": (
                f"python benchmarks-pr17/fp8_listen_test.py --duration 30 "
                f"--only fp8 --fp8-tag {tag}"
            ),
        },
        "calibration_npz": str(CALIBRATION_NPZ),
        "activation_absmax_json": str(ABSMAX_JSON),
    }
    (variant_dir / "config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8",
    )

    # metrics.json: parse from validation_results + manifest + engine size.
    metrics = _build_metrics(variant_dir, tag)
    (variant_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8",
    )
    return metrics


def _build_metrics(variant_dir: Path, tag: str) -> dict:
    val = json.loads((variant_dir / "validation_results.json").read_text(encoding="utf-8"))
    man = json.loads((variant_dir / "fp8_patch_manifest.json").read_text(encoding="utf-8"))
    eng = variant_dir / "engine.engine"

    bf16_mean = val["benchmark"]["bf16"]["mean_ms"]
    fp8_mean = val["benchmark"]["fp8"]["mean_ms"]
    return {
        "tag": tag,
        "engine_size_mb": eng.stat().st_size / 1e6 if eng.exists() else None,
        "cosine_sim_avg": val["numeric_aggregate"]["cosine_sim_avg"],
        "max_abs_diff_peak": val["numeric_aggregate"]["max_abs_diff_peak"],
        "mean_abs_diff_avg": val["numeric_aggregate"]["mean_abs_diff_avg"],
        "tick_latency_ms": {
            "bf16_mean": bf16_mean,
            "fp8_mean": fp8_mean,
            "fp8_p10": val["benchmark"]["fp8"].get("p10_ms"),
            "fp8_p90": val["benchmark"]["fp8"].get("p90_ms"),
        },
        "speedup_vs_bf16": bf16_mean / fp8_mean if fp8_mean else None,
        "matmul_counts": {
            "total_matmul_in_graph": 482,
            "weight_quantized": man.get("quantized_count"),
            "activation_qdq_pairs": man.get("activation_log_count"),
            "smoothquant_applied": man.get("smoothquant_applied_count"),
            "unmatched_weight_inits": len(man.get("unmatched_weight_inits", [])),
        },
        "smoothquant_alpha": man.get("smoothquant_alpha"),
        "activation_percentile": man.get("activation_percentile"),
    }


def _append_index_row(*, tag: str, description: str, metrics: dict) -> None:
    """Append/update a row in INDEX.md for this variant.

    Rows are matched by tag in the leftmost cell; an existing row gets
    replaced. New rows are inserted at the end of the metrics table
    (after the last existing ``| ... |`` row) rather than at the end of
    file — this keeps the surrounding prose intact.
    """
    lines = INDEX_MD.read_text(encoding="utf-8").splitlines()
    cos = metrics["cosine_sim_avg"]
    speed = metrics["speedup_vs_bf16"]
    eng_mb = metrics["engine_size_mb"]
    mc = metrics["matmul_counts"]
    w8a8_count = mc.get("activation_qdq_pairs") or 0
    weight_count = mc.get("weight_quantized") or 0
    fallback = max(0, weight_count - w8a8_count)
    new_row = (
        f"| {tag} | "
        f"**{cos:.4f}** | **{speed:.2f}x** | "
        f"{eng_mb:.0f} | {w8a8_count} | {fallback} | "
        f"{metrics.get('smoothquant_alpha')} | "
        f"{description} |"
    )

    # First pass: see if a row with this tag already exists.
    for i, ln in enumerate(lines):
        if ln.lstrip().startswith(f"| {tag} "):
            lines[i] = new_row
            INDEX_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
            return

    # No existing row: find the last contiguous table row block and
    # insert after it. The table block ends at the first non-table
    # line following the last `| ... |` line.
    last_table_idx = -1
    for i, ln in enumerate(lines):
        stripped = ln.strip()
        if stripped.startswith("|") and stripped.endswith("|"):
            last_table_idx = i

    if last_table_idx == -1:
        # No table found; append to EOF.
        lines.append(new_row)
    else:
        lines.insert(last_table_idx + 1, new_row)

    INDEX_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True,
                    help="Short identifier for the variant folder + listen subdir.")
    ap.add_argument("--description", required=True,
                    help="Human-readable description for INDEX.md.")
    ap.add_argument("--smoothquant-alpha", type=float, default=0.0,
                    help="SmoothQuant alpha (0 = off).")
    ap.add_argument("--outlier-skip-ratio", type=float, default=0.0,
                    help="W8A16 fallback threshold (0 = off).")
    ap.add_argument("--percentile",
                    choices=("absmax", "p99", "p99_9", "p99_99"),
                    default="absmax",
                    help="Activation amax field for scale.")
    ap.add_argument("--no-w8a8", action="store_true",
                    help="Build W8A16 (weight-only FP8) — omit activation Q-DQ entirely.")
    ap.add_argument("--skip-patch", action="store_true",
                    help="Reuse the FP8 ONNX currently on disk (only do build+listen+archive).")
    ap.add_argument("--skip-build", action="store_true",
                    help="Reuse the engine currently on disk.")
    ap.add_argument("--quantize-attention", action="store_true",
                    help="Also Q-DQ the 128 dynamic-input attention MatMuls "
                         "(Q*K^T and attn*V) so they run on FP8 GEMM tactics.")
    ap.add_argument("--attention-softmax-max", type=float, default=1.05,
                    help="Per-tensor activation amax for softmax-output "
                         "operands (default 1.05).")
    ap.add_argument("--attention-generic-max", type=float, default=10.0,
                    help="Per-tensor activation amax for non-softmax "
                         "attention operands (default 10.0).")
    args = ap.parse_args()

    cli_args = {
        "smoothquant_alpha": args.smoothquant_alpha,
        "outlier_skip_ratio": args.outlier_skip_ratio,
        "percentile": args.percentile,
        "w8a8": not args.no_w8a8,
        "quantize_attention": args.quantize_attention,
        "attention_softmax_max": args.attention_softmax_max,
        "attention_generic_max": args.attention_generic_max,
    }

    if not args.skip_patch:
        _patch(
            smoothquant_alpha=args.smoothquant_alpha,
            outlier_skip_ratio=args.outlier_skip_ratio,
            percentile=args.percentile,
            w8a8=cli_args["w8a8"],
            quantize_attention=args.quantize_attention,
            attention_softmax_max=args.attention_softmax_max,
            attention_generic_max=args.attention_generic_max,
        )

    if not args.skip_build:
        _build(
            smoothquant_alpha=args.smoothquant_alpha,
            w8a8=cli_args["w8a8"],
            percentile=args.percentile,
        )

    _validate()
    _listen(args.tag)
    metrics = _archive(
        tag=args.tag,
        description=args.description,
        cli_args=cli_args,
    )
    _append_index_row(
        tag=args.tag,
        description=args.description,
        metrics=metrics,
    )

    print()
    print("=" * 70)
    print(f"VARIANT '{args.tag}' ARCHIVED")
    print("=" * 70)
    print(f"  engine: variants/{args.tag}/engine.engine")
    print(f"  audio:  variants/{args.tag}/audio/*.wav")
    print(f"  cos:    {metrics['cosine_sim_avg']:.4f}")
    print(f"  speed:  {metrics['speedup_vs_bf16']:.2f}x  "
          f"({metrics['tick_latency_ms']['bf16_mean']:.1f} -> "
          f"{metrics['tick_latency_ms']['fp8_mean']:.1f} ms)")
    print(f"  index:  variants/INDEX.md")


if __name__ == "__main__":
    main()
