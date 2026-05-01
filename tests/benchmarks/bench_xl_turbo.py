"""Benchmark XL turbo (4B DiT, v0.1.6): unaccelerated PyTorch baseline.

Pure PyTorch SDPA path: no torch.compile, no flash-attn, no TRT.
Run this first to establish the floor before TRT acceleration.

Usage:
    uv run python tests/benchmarks/bench_xl_turbo.py
    uv run python tests/benchmarks/bench_xl_turbo.py --steps 8 --duration 60
    uv run python tests/benchmarks/bench_xl_turbo.py --runs 5 --warmup 2
"""

import argparse
import os
import sys
import time

# Belt-and-suspenders: disable torch.compile/dynamo globally before importing torch.
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
os.environ.setdefault("TORCHINDUCTOR_DISABLE", "1")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import torch
torch.set_grad_enabled(False)
# Force-disable dynamo at runtime as well (in case env was set after import elsewhere).
try:
    import torch._dynamo as _dynamo
    _dynamo.config.disable = True
except Exception:
    pass

from acestep.constants import TASK_INSTRUCTIONS
from acestep.engine.session import Session
from acestep.paths import checkpoints_dir


def bench_generate(session, label, *, steps, shift, duration, seed, warmup, runs):
    """Time session.generate() (no CFG) and return per-run ms + peak VRAM."""
    cond = session.encode_text(
        tags="jazz piano trio, brushed drums, walking bass, 140 bpm",
        lyrics="[instrumental]",
        duration=duration,
        instruction=TASK_INSTRUCTIONS["text2music"],
    )

    # Warmup (XL turbo uses no CFG)
    for _ in range(warmup):
        session.generate(
            conditioning=cond, seed=seed,
            steps=steps, shift=shift, denoise=1.0,
        )

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    times = []
    peaks = []
    for i in range(runs):
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        session.generate(
            conditioning=cond, seed=seed + i,
            steps=steps, shift=shift, denoise=1.0,
        )
        torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - t0) * 1000
        peak_gb = torch.cuda.max_memory_allocated() / 1024**3
        times.append(elapsed_ms)
        peaks.append(peak_gb)
        print(f"  run {i+1}/{runs}: {elapsed_ms:7.0f} ms   peak={peak_gb:.2f} GB", flush=True)

    avg = sum(times) / len(times)
    mn = min(times)
    mx = max(times)
    per_step = avg / steps
    rtf = duration / (avg / 1000.0)
    pk_avg = sum(peaks) / len(peaks)
    pk_max = max(peaks)
    print(
        f"\n  [{label}] avg={avg:.0f}ms  min={mn:.0f}ms  max={mx:.0f}ms  "
        f"per_step={per_step:.1f}ms  RTF={rtf:.2f}x  peak={pk_max:.2f}GB"
    )
    return {
        "label": label,
        "avg_ms": avg, "min_ms": mn, "max_ms": mx,
        "per_step_ms": per_step, "steps": steps, "duration_s": duration,
        "rtf": rtf, "peak_gb_max": pk_max, "peak_gb_avg": pk_avg,
        "times_ms": times, "peaks_gb": peaks,
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark XL turbo (no acceleration)")
    parser.add_argument("--config-path", default="acestep-v15-xl-turbo",
                        help="Checkpoint dir name (default: acestep-v15-xl-turbo)")
    parser.add_argument("--steps", type=int, nargs="+", default=[8],
                        help="Step counts to benchmark (default: 8)")
    parser.add_argument("--duration", type=float, default=60.0,
                        help="Audio duration in seconds (default: 60)")
    parser.add_argument("--shift", type=float, default=3.0,
                        help="Timestep shift (default: 3.0 for turbo)")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use-flash", action="store_true",
                        help="Enable flash-attn-2 (default: SDPA)")
    parser.add_argument("--trt-engine", default=None,
                        help="Path to TRT decoder engine. When set, decoder PT "
                             "weights are skipped and TRT runs the diffusion loop.")
    args = parser.parse_args()

    mode = "TRT" if args.trt_engine else "PYTORCH"
    print("=" * 70)
    print(f"XL TURBO {mode}  config={args.config_path}")
    print(f"  compile=False  trt={args.trt_engine}  "
          f"attn={'flash2' if args.use_flash else 'sdpa'}")
    print(f"  duration={args.duration}s  shift={args.shift}  steps={args.steps}")
    print(f"  warmup={args.warmup}  runs={args.runs}  seed={args.seed}")
    print("=" * 70)

    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        print(f"GPU: {props.name}  total_vram={props.total_memory/1024**3:.1f} GB")
    print(f"checkpoints_dir: {checkpoints_dir()}")

    t_load_start = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    session_kwargs = dict(
        project_root=str(checkpoints_dir()),
        config_path=args.config_path,
        use_flash_attention=args.use_flash,
    )
    if args.trt_engine:
        session_kwargs["trt_engines"] = {"decoder": args.trt_engine}
        session_kwargs["decoder_backend"] = "tensorrt"
    session = Session(**session_kwargs)
    torch.cuda.synchronize()
    load_s = time.perf_counter() - t_load_start
    load_peak_gb = torch.cuda.max_memory_allocated() / 1024**3

    # Verify torch.compile actually disabled: decoder must be a plain Module.
    inner_model = session.model.handler.model
    decoder_cls = type(inner_model.decoder).__name__
    vae_cls = type(session.handler.vae).__name__
    is_compiled = "OptimizedModule" in decoder_cls or "OptimizedModule" in vae_cls
    print(f"\nload time: {load_s:.1f}s   load peak VRAM: {load_peak_gb:.2f} GB")
    print(f"decoder class: {decoder_cls}   vae class: {vae_cls}   compiled={is_compiled}")
    print(f"TORCHDYNAMO_DISABLE={os.environ.get('TORCHDYNAMO_DISABLE')}  "
          f"TORCH_COMPILE_DISABLE={os.environ.get('TORCH_COMPILE_DISABLE')}")
    assert not is_compiled, f"torch.compile leaked in: decoder={decoder_cls} vae={vae_cls}"
    print()

    results = []
    for steps in args.steps:
        print(f"--- {steps} steps ---")
        r = bench_generate(
            session, f"PT XL-turbo {steps}step",
            steps=steps, shift=args.shift, duration=args.duration,
            seed=args.seed, warmup=args.warmup, runs=args.runs,
        )
        results.append(r)
        print()

    print("=" * 70)
    print("SUMMARY (XL TURBO, no acceleration)")
    print("=" * 70)
    print(f"{'Label':<30s} {'Avg(ms)':>9s} {'Min(ms)':>9s} {'Per-step':>10s} {'RTF':>7s} {'Peak(GB)':>10s}")
    for r in results:
        print(f"{r['label']:<30s} {r['avg_ms']:>9.0f} {r['min_ms']:>9.0f} "
              f"{r['per_step_ms']:>10.1f} {r['rtf']:>6.2f}x {r['peak_gb_max']:>10.2f}")
    print()
    print(f"load_time_s={load_s:.1f}  load_peak_gb={load_peak_gb:.2f}")


if __name__ == "__main__":
    main()
