"""Decoder TRT engine latency benchmark.

Default shape: 60s, T=750, L_enc=200, B=1, 50 measured iters + 10 warmup.
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import torch

from acestep.engine.trt.runtime import TRTDecoder


def make_inputs(seed: int, B: int, T: int, L: int, device: str = "cuda"):
    g = torch.Generator(device=device).manual_seed(seed)
    return {
        "hidden_states": torch.randn(B, T, 64, generator=g, device=device, dtype=torch.float32),
        "timestep": torch.full((B,), 0.5, device=device, dtype=torch.float32),
        "encoder_hidden_states": torch.randn(B, L, 2048, generator=g, device=device, dtype=torch.float32),
        "context_latents": torch.randn(B, T, 128, generator=g, device=device, dtype=torch.float32),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("engine", type=Path)
    ap.add_argument("--B", type=int, default=1)
    ap.add_argument("--T", type=int, default=750)
    ap.add_argument("--L", type=int, default=200)
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    if not args.engine.is_file():
        print(f"[fatal] engine not found: {args.engine}", file=sys.stderr)
        return 2

    import tensorrt
    print(f"TRT runtime: {tensorrt.__version__}")
    print(f"engine     : {args.engine}")
    print(f"shape      : B={args.B} T={args.T} L_enc={args.L}")
    print(f"iters      : warmup={args.warmup} measured={args.iters}")

    dec = TRTDecoder(args.engine)
    inputs = make_inputs(args.seed, args.B, args.T, args.L)

    for _ in range(args.warmup):
        _ = dec(**inputs)
    torch.cuda.synchronize()

    times_ms: list[float] = []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for _ in range(args.iters):
        start.record()
        _ = dec(**inputs)
        end.record()
        torch.cuda.synchronize()
        times_ms.append(start.elapsed_time(end))

    mean = statistics.mean(times_ms)
    mn = min(times_ms)
    mx = max(times_ms)
    sd = statistics.pstdev(times_ms)
    p95 = sorted(times_ms)[int(0.95 * len(times_ms)) - 1]
    steps_per_s = 1000.0 / mean

    print()
    print(f"  Mean       : {mean:7.3f} ms (+/- {sd:.3f})")
    print(f"  Min/Max    : {mn:7.3f} / {mx:7.3f} ms")
    print(f"  p95        : {p95:7.3f} ms")
    print(f"  Steps/sec  : {steps_per_s:6.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
