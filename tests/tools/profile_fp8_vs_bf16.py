"""Quality check: FP8 engine vs bf16 reference engine.

Runs both engines on the same inputs at the given shape, reports cosine
similarity and abs-diff stats. Uses calibration inputs by default (real
decoder activations), with an optional random fallback for sanity.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import numpy as np
import torch

from acestep.engine.trt.runtime import TRTDecoder


def random_inputs(seed: int, B: int, T: int, L: int, device: str = "cuda"):
    g = torch.Generator(device=device).manual_seed(seed)
    return {
        "hidden_states": torch.randn(B, T, 64, generator=g, device=device, dtype=torch.float32),
        "timestep": torch.full((B,), 0.5, device=device, dtype=torch.float32),
        "encoder_hidden_states": torch.randn(B, L, 2048, generator=g, device=device, dtype=torch.float32),
        "context_latents": torch.randn(B, T, 128, generator=g, device=device, dtype=torch.float32),
    }


def calibration_inputs(cal_path: Path, B: int, batch_idx: int, device: str = "cuda"):
    cal = np.load(str(cal_path))
    s = slice(batch_idx * B, (batch_idx + 1) * B)
    return {
        "hidden_states": torch.from_numpy(cal["hidden_states"][s]).to(device),
        "timestep": torch.from_numpy(cal["timestep"][s]).to(device),
        "encoder_hidden_states": torch.from_numpy(cal["encoder_hidden_states"][s]).to(device),
        "context_latents": torch.from_numpy(cal["context_latents"][s]).to(device),
    }


def run(engine_path: Path, inputs: dict) -> torch.Tensor:
    return TRTDecoder(engine_path)(**inputs).clone()


def compare(a: torch.Tensor, b: torch.Tensor, label_a: str, label_b: str) -> None:
    a32 = a.float()
    b32 = b.float()
    diff = (a32 - b32).abs()
    cos = torch.nn.functional.cosine_similarity(
        a32.flatten().unsqueeze(0), b32.flatten().unsqueeze(0)
    ).item()
    print(f"  {label_a:>4} vs {label_b:>11}  cos={cos:.6f}  "
          f"max|diff|={diff.max().item():.3e}  "
          f"mean|diff|={diff.mean().item():.3e}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bf16", type=Path, required=True, help="Path to bf16 reference engine")
    ap.add_argument("--fp8", type=Path, required=True, help="Path to fp8 engine")
    ap.add_argument("--calibration", type=Path, default=None,
                    help="Path to calibration .npz; if omitted, uses random inputs.")
    ap.add_argument("--B", type=int, default=4)
    ap.add_argument("--T", type=int, default=1500)
    ap.add_argument("--L", type=int, default=200)
    ap.add_argument("--batches", type=int, default=4,
                    help="Number of cal batches to compare (default: 4).")
    args = ap.parse_args()

    for p in (args.bf16, args.fp8):
        if not p.is_file():
            print(f"[fatal] missing: {p}", file=sys.stderr)
            return 2

    if args.calibration is None:
        print(f"# Random inputs, B={args.B} T={args.T} L_enc={args.L}, 3 seeds")
        for seed in (1337, 2024, 4242):
            inputs = random_inputs(seed, args.B, args.T, args.L)
            out_bf16 = run(args.bf16, inputs)
            out_fp8 = run(args.fp8, inputs)
            print(f"seed={seed}")
            compare(out_fp8, out_bf16, "fp8", "bf16")
        return 0

    print(f"# Calibration inputs from {args.calibration}, "
          f"B={args.B} T={args.T} L_enc={args.L}, {args.batches} batches")
    for batch_idx in range(args.batches):
        inputs = calibration_inputs(args.calibration, args.B, batch_idx)
        out_bf16 = run(args.bf16, inputs)
        out_fp8 = run(args.fp8, inputs)
        print(f"batch {batch_idx}")
        compare(out_fp8, out_bf16, "fp8", "bf16")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
