"""Quality + parity check: 2B FP8 engine vs bf16-hybrid engine.

Runs both engines on the same deterministic inputs at B=4 T=1500 L=200,
reports cosine similarity, mean/max absolute diff. This is the same
shape used for the latency comparison.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import torch

from acestep.engine.trt.runtime import TRTDecoder


ENGINES_ROOT = Path.home() / ".daydream-scope/models/demon/trt_engines"
# The .engine slot currently holds the pure bf16_mixed build (the FP8
# baseline). The bf16-hybrid recipe is parked next to it as a sibling.
BF16 = ENGINES_ROOT / "decoder_mixed_refit_b8_60s/decoder_mixed_refit_b8_60s.engine"
FP8 = ENGINES_ROOT / "decoder_fp8_refit_b8_60s/decoder_fp8_refit_b8_60s.engine"


def make_inputs(seed: int, B: int, T: int, L: int, device: str = "cuda"):
    g = torch.Generator(device=device).manual_seed(seed)
    return {
        "hidden_states": torch.randn(B, T, 64, generator=g, device=device, dtype=torch.float32),
        "timestep": torch.full((B,), 0.5, device=device, dtype=torch.float32),
        "encoder_hidden_states": torch.randn(B, L, 2048, generator=g, device=device, dtype=torch.float32),
        "context_latents": torch.randn(B, T, 128, generator=g, device=device, dtype=torch.float32),
    }


def calibration_inputs(B: int, batch_idx: int, device: str = "cuda"):
    import numpy as np
    cal_path = Path.home() / ".daydream-scope/models/demon/calibration/decoder_2b_fp8/calibration.npz"
    cal = np.load(str(cal_path))
    s = slice(batch_idx * B, (batch_idx + 1) * B)
    return {
        "hidden_states": torch.from_numpy(cal["hidden_states"][s]).to(device),
        "timestep": torch.from_numpy(cal["timestep"][s]).to(device),
        "encoder_hidden_states": torch.from_numpy(cal["encoder_hidden_states"][s]).to(device),
        "context_latents": torch.from_numpy(cal["context_latents"][s]).to(device),
    }


def run(engine_path: Path, inputs: dict) -> torch.Tensor:
    dec = TRTDecoder(engine_path)
    return dec(**inputs).clone()


def compare(a: torch.Tensor, b: torch.Tensor, label_a: str, label_b: str) -> None:
    a32 = a.float()
    b32 = b.float()
    diff = (a32 - b32).abs()
    cos = torch.nn.functional.cosine_similarity(
        a32.flatten().unsqueeze(0), b32.flatten().unsqueeze(0)
    ).item()
    print(f"\n  --- {label_a} vs {label_b} ---")
    print(f"  cosine_sim    : {cos:.6f}")
    print(f"  max_abs_diff  : {diff.max().item():.6e}")
    print(f"  mean_abs_diff : {diff.mean().item():.6e}")
    print(f"  l2_norm_a     : {a32.norm().item():.4f}")
    print(f"  l2_norm_b     : {b32.norm().item():.4f}")


def main():
    for p in (BF16, FP8):
        if not p.is_file():
            print(f"[fatal] missing: {p}", file=sys.stderr)
            return 2

    B, T, L = 4, 1500, 200
    print("\n# Random inputs (out of distribution)")
    for seed in (1337, 2024, 4242):
        print(f"\n== seed={seed} B={B} T={T} L_enc={L} ==")
        inputs = make_inputs(seed, B, T, L)
        out_bf16 = run(BF16, inputs)
        out_fp8 = run(FP8, inputs)
        compare(out_fp8, out_bf16, "fp8", "bf16-hybrid")

    print("\n# Calibration inputs (in-distribution real decoder activations)")
    for batch_idx in (0, 50, 100, 150):
        print(f"\n== calibration batch {batch_idx} B={B} T={T} L_enc={L} ==")
        inputs = calibration_inputs(B, batch_idx)
        out_bf16 = run(BF16, inputs)
        out_fp8 = run(FP8, inputs)
        compare(out_fp8, out_bf16, "fp8", "bf16-hybrid")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
