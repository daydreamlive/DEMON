"""Test whether torch._scaled_mm supports NVFP4 GEMM on RTX 5090.

torch._scaled_mm is the official PyTorch FP8-scaled-matmul API. In
recent versions it also accepts FP4 dtypes if cuBLAS-Lt supports them.
This is the most direct path to NVFP4 GEMM throughput on Blackwell.

The test:
  1. Build random tensors in FP8 (known-working) and FP4 (testing).
  2. Call torch._scaled_mm with scale tensors.
  3. Measure throughput; compare against bf16 reference.
  4. Report whether FP4 actually delivers >2x BF16.
"""
from __future__ import annotations

import os
import sys
import time

os.environ.setdefault("PYTHONUTF8", "1")
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import torch

# DiT-like shape (M aligned to 128 for FP4 tactics).
M = 6144
K = 3072
N = 3072
ITERS = 200
WARMUP = 20


def bench(fn, label: str) -> float:
    dev = torch.device("cuda")
    for _ in range(WARMUP):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(ITERS):
        fn()
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / ITERS * 1000
    flops = 2 * M * K * N
    tflops = flops / (dt / 1000) / 1e12
    print(f"  {label:<40s}  {dt:>7.3f} ms   {tflops:>7.1f} TFLOPS")
    return dt


def main():
    print("=" * 70)
    print(f"torch._scaled_mm NVFP4 feasibility ({M}x{K}x{N})")
    print(f"torch: {torch.__version__}")
    print(f"GPU:   {torch.cuda.get_device_name(0)}")
    print(f"CC:    {torch.cuda.get_device_capability(0)}")
    print("=" * 70)

    dev = torch.device("cuda")
    torch.manual_seed(0)

    # BF16 reference.
    a_bf16 = torch.randn(M, K, device=dev, dtype=torch.bfloat16)
    b_bf16 = torch.randn(K, N, device=dev, dtype=torch.bfloat16)

    # FP8 inputs.
    a_fp8 = a_bf16.to(torch.float8_e4m3fn)
    b_fp8 = b_bf16.t().contiguous().to(torch.float8_e4m3fn).t()  # B needs to be col-major
    scale_a_fp8 = torch.tensor(1.0, device=dev, dtype=torch.float32)
    scale_b_fp8 = torch.tensor(1.0, device=dev, dtype=torch.float32)

    print("\nBaselines:")
    bench(lambda: a_bf16 @ b_bf16, "bf16 @ bf16 (torch.mm)")

    # Verify FP8 works.
    print("\nFP8 (sanity check the API):")
    try:
        out_fp8 = torch._scaled_mm(
            a_fp8, b_fp8,
            scale_a=scale_a_fp8, scale_b=scale_b_fp8,
            out_dtype=torch.bfloat16,
        )
        bench(
            lambda: torch._scaled_mm(
                a_fp8, b_fp8,
                scale_a=scale_a_fp8, scale_b=scale_b_fp8,
                out_dtype=torch.bfloat16,
            ),
            "FP8 (e4m3fn) per-tensor scale",
        )
    except Exception as e:
        print(f"  FP8 FAILED: {type(e).__name__}: {e}")

    # FP4 attempt.
    print("\nFP4 (the question):")
    print(f"  torch.float4_e2m1fn_x2 dtype available: {hasattr(torch, 'float4_e2m1fn_x2')}")
    # torch.float4_e2m1fn_x2.to() is not implemented in 2.9.1.
    # But we can construct FP4 tensors via the random byte path then view().
    try:
        # Two FP4 values per byte. M*K elements -> M*K/2 bytes.
        # PyTorch FP4 tensors expose logical shape; storage shape has last-dim halved.
        # We construct via uint8 then view as fp4 if possible.
        rng_bytes_a = torch.randint(0, 256, (M, K // 2), dtype=torch.uint8, device=dev)
        rng_bytes_b = torch.randint(0, 256, (K, N // 2), dtype=torch.uint8, device=dev)
        # Attempt view as float4_e2m1fn_x2.
        try:
            a_fp4 = rng_bytes_a.view(torch.float4_e2m1fn_x2)
            b_fp4 = rng_bytes_b.view(torch.float4_e2m1fn_x2)
            print(f"  Constructed FP4 tensors: a={a_fp4.shape} dtype={a_fp4.dtype}")
        except Exception as e:
            print(f"  Could not view bytes as FP4: {type(e).__name__}: {e}")
            return

        # Try torch._scaled_mm with FP4.
        scale_a_fp4 = torch.tensor(1.0, device=dev, dtype=torch.float32)
        scale_b_fp4 = torch.tensor(1.0, device=dev, dtype=torch.float32)
        try:
            out_fp4 = torch._scaled_mm(
                a_fp4, b_fp4,
                scale_a=scale_a_fp4, scale_b=scale_b_fp4,
                out_dtype=torch.bfloat16,
            )
            print(f"  FP4 scaled_mm output shape: {out_fp4.shape}, dtype: {out_fp4.dtype}")
            bench(
                lambda: torch._scaled_mm(
                    a_fp4, b_fp4,
                    scale_a=scale_a_fp4, scale_b=scale_b_fp4,
                    out_dtype=torch.bfloat16,
                ),
                "FP4 (e2m1) per-tensor scale",
            )
        except Exception as e:
            print(f"  FP4 per-tensor FAILED: {type(e).__name__}: {e}")

        # Try FP4 with block scales (the NVFP4 format).
        # NVFP4 uses FP8 per-block scales of size 16 along contraction axis.
        # torch._scaled_mm with FP4 + block scales is the actual NVFP4 path.
        block = 16
        scale_a_block = torch.full(
            (M, K // block), 1.0, device=dev, dtype=torch.float8_e4m3fn,
        )
        scale_b_block = torch.full(
            (K // block, N), 1.0, device=dev, dtype=torch.float8_e4m3fn,
        )
        try:
            out = torch._scaled_mm(
                a_fp4, b_fp4,
                scale_a=scale_a_block, scale_b=scale_b_block,
                out_dtype=torch.bfloat16,
            )
            print(f"  FP4 + per-block FP8 scale: output {out.shape}")
            bench(
                lambda: torch._scaled_mm(
                    a_fp4, b_fp4,
                    scale_a=scale_a_block, scale_b=scale_b_block,
                    out_dtype=torch.bfloat16,
                ),
                "NVFP4 (FP4 + per-block FP8 scale)",
            )
        except Exception as e:
            print(f"  NVFP4 block scale FAILED: {type(e).__name__}: {e}")

    except Exception as e:
        print(f"  Setup FAILED: {type(e).__name__}: {e}")
        import traceback; traceback.print_exc()


if __name__ == "__main__":
    main()
