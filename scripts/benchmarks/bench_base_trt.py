"""Benchmark base model: PyTorch vs TRT with CFG.

Compares latency for the base model DiT with classifier-free guidance,
which is required for quality output from the base model.

Usage:
    uv run python scripts/benchmarks/bench_base_trt.py
    uv run python scripts/benchmarks/bench_base_trt.py --trt-engine trt_engines/decoder_base_mixed_b8_60s/decoder_base_mixed_b8_60s.engine
    uv run python scripts/benchmarks/bench_base_trt.py --steps 25 50 --duration 30
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import torch
torch.set_grad_enabled(False)

from acestep.constants import TASK_INSTRUCTIONS
from acestep.engine.session import Session
from acestep.nodes.types import Curve


# ── helpers ──────────────────────────────────────────────────────────

def find_base_engine(project_root: str) -> str | None:
    """Auto-detect a base decoder TRT engine in trt_engines/."""
    trt_dir = os.path.join(project_root, "trt_engines")
    if not os.path.isdir(trt_dir):
        return None
    for name in sorted(os.listdir(trt_dir)):
        if "base" in name and "decoder" in name and not name.startswith("_"):
            engine = os.path.join(trt_dir, name, f"{name}.engine")
            if os.path.isfile(engine):
                return engine
    return None


def bench_generate(session, label, *, steps, shift, duration, seed,
                   cfg_scale, warmup=2, runs=5):
    """Time session.generate() with CFG and return per-run ms."""
    cond = session.encode_text(
        tags="jazz piano trio, brushed drums, walking bass, 140 bpm",
        lyrics="[instrumental]",
        duration=duration,
        instruction=TASK_INSTRUCTIONS["text2music"],
    )

    # Null conditioning using the model's learned null_condition_emb
    neg_cond = session.null_conditioning(cond)
    T = int(duration * 25)
    gc = Curve(tensor=torch.full((T,), cfg_scale, dtype=torch.bfloat16))

    kwargs = {"negative": neg_cond, "guidance_curve": gc}

    # Warmup
    for _ in range(warmup):
        session.generate(
            conditioning=cond, seed=seed,
            steps=steps, shift=shift, denoise=1.0,
            **kwargs,
        )

    torch.cuda.synchronize()
    times = []
    for i in range(runs):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        session.generate(
            conditioning=cond, seed=seed + i,
            steps=steps, shift=shift, denoise=1.0,
            **kwargs,
        )
        torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - t0) * 1000
        times.append(elapsed_ms)

    avg = sum(times) / len(times)
    per_step = avg / steps
    mn = min(times)
    mx = max(times)
    print(f"  [{label}] avg={avg:.0f}ms  min={mn:.0f}ms  max={mx:.0f}ms  "
          f"per_step={per_step:.1f}ms  ({steps} steps)")
    return {"label": label, "avg_ms": avg, "min_ms": mn, "max_ms": mx,
            "per_step_ms": per_step, "steps": steps, "times": times}


# ── main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Benchmark base model TRT")
    parser.add_argument("--trt-engine", default=None,
                        help="Path to base decoder TRT engine (auto-detected if omitted)")
    parser.add_argument("--steps", type=int, nargs="+", default=[25, 50],
                        help="Step counts to benchmark (default: 25 50)")
    parser.add_argument("--duration", type=float, default=60.0,
                        help="Audio duration in seconds (default: 60)")
    parser.add_argument("--shift", type=float, default=1.0,
                        help="Timestep shift (default: 1.0 for base)")
    parser.add_argument("--guidance-scale", type=float, default=7.5,
                        help="CFG guidance scale (default: 7.5)")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-pytorch", action="store_true",
                        help="Skip PyTorch baseline (TRT-only)")
    parser.add_argument("--skip-trt", action="store_true",
                        help="Skip TRT benchmark (PyTorch-only)")
    args = parser.parse_args()

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    ckpt_root = os.path.join(project_root, "checkpoints")

    # Auto-detect TRT engine
    trt_engine = args.trt_engine
    if trt_engine is None and not args.skip_trt:
        trt_engine = find_base_engine(project_root)
        if trt_engine:
            print(f"Auto-detected base engine: {trt_engine}")
        else:
            print("No base TRT engine found. Run with --skip-trt or build engines first.")
            if not args.skip_pytorch:
                print("Running PyTorch-only benchmark.\n")
                args.skip_trt = True
            else:
                sys.exit(1)

    # VAE engines (shared across variants)
    vae_enc = os.path.join(project_root, "trt_engines", "vae_encode_fp16_60s", "vae_encode_fp16_60s.engine")
    vae_dec = os.path.join(project_root, "trt_engines", "vae_decode_fp16_60s", "vae_decode_fp16_60s.engine")

    results_all = []

    # ── PyTorch baseline ──
    if not args.skip_pytorch:
        print("=" * 60)
        print(f"PYTORCH BASELINE (base model, CFG={args.guidance_scale})")
        print("=" * 60)
        session_pt = Session(
            project_root=ckpt_root,
            config_path="acestep-v15-base",
            use_flash_attention=True,
        )

        for steps in args.steps:
            print(f"\n--- {steps} steps ---")
            r = bench_generate(
                session_pt, f"PT {steps}s",
                steps=steps, shift=args.shift, duration=args.duration,
                seed=args.seed, cfg_scale=args.guidance_scale,
                warmup=args.warmup, runs=args.runs,
            )
            results_all.append(r)

        del session_pt
        torch.cuda.empty_cache()

    # ── TRT ──
    if not args.skip_trt:
        print("\n" + "=" * 60)
        print(f"TRT DECODER (base model, CFG={args.guidance_scale})")
        print(f"  engine: {trt_engine}")
        print("=" * 60)

        trt_engines = {"decoder": trt_engine}
        has_vae_trt = os.path.isfile(vae_enc) and os.path.isfile(vae_dec)
        if has_vae_trt:
            trt_engines["vae_encode"] = vae_enc
            trt_engines["vae_decode"] = vae_dec

        session_trt = Session(
            project_root=ckpt_root,
            config_path="acestep-v15-base",
            decoder_backend="tensorrt",
            vae_backend="tensorrt" if has_vae_trt else "eager",
            use_flash_attention=True,
            trt_engines=trt_engines,
        )

        for steps in args.steps:
            print(f"\n--- {steps} steps (batched CFG) ---")
            r = bench_generate(
                session_trt, f"TRT {steps}s",
                steps=steps, shift=args.shift, duration=args.duration,
                seed=args.seed, cfg_scale=args.guidance_scale,
                warmup=args.warmup, runs=args.runs,
            )
            results_all.append(r)

        del session_trt
        torch.cuda.empty_cache()

    # ── Summary ──
    print("\n" + "=" * 60)
    print(f"SUMMARY (CFG={args.guidance_scale})")
    print("=" * 60)
    print(f"{'Label':<35s} {'Avg(ms)':>8s} {'Min(ms)':>8s} {'Per-step':>10s}")
    print("-" * 65)
    for r in results_all:
        print(f"{r['label']:<35s} {r['avg_ms']:>8.0f} {r['min_ms']:>8.0f} "
              f"{r['per_step_ms']:>8.1f}ms")

    # Speedup comparisons
    print()
    by_label = {r["label"]: r for r in results_all}
    for steps in args.steps:
        pt_key = f"PT {steps}s"
        trt_key = f"TRT {steps}s"
        if pt_key in by_label and trt_key in by_label:
            speedup = by_label[pt_key]["avg_ms"] / by_label[trt_key]["avg_ms"]
            print(f"  {steps}-step: {speedup:.2f}x speedup (TRT vs PyTorch)")


if __name__ == "__main__":
    main()
