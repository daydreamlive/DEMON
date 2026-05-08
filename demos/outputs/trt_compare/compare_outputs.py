"""Compare isolation-bench outputs across TRT 10.13 vs 10.16."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import torch

DIR = Path(__file__).resolve().parent

# Load decoder + VAE outputs from both TRT versions.
out_a = torch.load(DIR / "outputs_trt10_13.pt", weights_only=False)
print("--- 10.13 outputs:", list(out_a.keys()))

# Decoder-only 10.16 (full bench broke on VAE).
out_b = torch.load(DIR / "outputs_trt10_16_dec.pt", weights_only=False)
print("--- 10.16 outputs:", list(out_b.keys()))

print()
print(f"{'config':30s}  {'shape':30s}  {'mean_abs_diff':>14s}  {'max_abs_diff':>14s}  {'rel_l2':>10s}")
print("-" * 110)
for key in sorted(set(out_a) & set(out_b)):
    a = out_a[key].to(torch.float32)
    b = out_b[key].to(torch.float32)
    if a.shape != b.shape:
        print(f"{key:30s}  shape mismatch  {tuple(a.shape)} vs {tuple(b.shape)}")
        continue
    diff = (a - b).abs()
    mad = diff.mean().item()
    mxd = diff.max().item()
    norm_a = a.norm().item()
    rel = ((a - b).norm() / max(norm_a, 1e-9)).item()
    shape_str = "x".join(str(d) for d in a.shape)
    print(f"{key:30s}  {shape_str:30s}  {mad:14.6e}  {mxd:14.6e}  {rel:10.6e}")

# Note: VAE outputs are not in 10.16 because the engine segfaulted. Print which configs are missing.
missing = set(out_a) - set(out_b)
if missing:
    print()
    print("Configs only in 10.13 (10.16 engines unusable):")
    for k in sorted(missing):
        print(f"  {k}")
