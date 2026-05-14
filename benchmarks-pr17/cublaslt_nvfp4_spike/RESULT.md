# cuBLASLt NVFP4 GEMM verification spike

## Goal

Verify that NVFP4 GEMM on RTX 5090 (Blackwell SM_120) actually delivers a
substantial fraction of the marketed peak before committing to a custom TRT
plugin path.

## Result: GATE PASSED — 1409 TFLOPS on production shape

Direct ctypes binding to the cuBLASLt bundled with PyTorch
(`cublasLt64_12.dll`, lib version 120804, CUDA 12.8) executes NVFP4 GEMM
correctly on RTX 5090 and delivers:

| M | N | K | best algo | per-iter | TFLOPS |
|---|---|---|---|---|---|
| 6144 | 3072 | 3072 | 0 | 0.082 ms | **1409.4** |
| 2048 | 2048 | 2048 | 3 | 0.021 ms | 832.3 |
| 4096 | 4096 | 4096 | 2 | 0.101 ms | 1366.0 |
| 8192 | 8192 | 8192 | 1 | 0.725 ms | **1516.9** |

For comparison (prior spikes, same 6144 x 3072 x 3072 shape):
- FP8 via `torch._scaled_mm`: **422 TFLOPS**
- FP8 via TRT W8A8 production pattern (`fp8_stub.py`): **414 TFLOPS**

**NVFP4 / FP8 ratio on production shape: 3.4x. Gate (>500 TFLOPS) cleared by
~2.8x.**

The cublasLt heuristic returns 6 valid algos and all 6 cluster within ~10% of
each other (~1290 - 1409 TFLOPS), so the result is not algo-pick-dependent.
Inputs were uniform FP4 bits (`0x44`) with unit FP8 E4M3 block scales (`0x38`).
Numerical correctness is not validated here — only throughput, which depends
only on the kernel doing the FP4 mma instructions, not on the input
distribution.

## What it took

| Path | Status |
|---|---|
| `torch._scaled_mm` with FP4 inputs | Still FAIL on torch 2.9.1+cu128 (no `copy_` / `to()` for `torch.float4_e2m1fn_x2`, no FP4 path in `_scaled_mm`) |
| ModelOpt 0.43 `quantize_mode="nvfp4"` | Still FAIL (dispatcher only handles int8/fp8/int4) |
| Transformer Engine pip install | Still FAIL (placeholder package; not pursued — ctypes path won outright) |
| **cuBLASLt ctypes wrapper (this spike)** | **SUCCESS** |

Implementation: `benchmarks-pr17/cublaslt_nvfp4_spike/nvfp4_gemm.py`.
~280 lines including comments. Binds 11 cublasLt entry points via ctypes;
allocates packed FP4 tensors with `torch.empty(..., dtype=torch.uint8)` (2
FP4 vals per byte); uses `data_ptr()` for the cuBLASLt API.

### Key configuration that made it work

The first attempt (row-major NN layout for both A and B) returned
`CUBLAS_STATUS_NOT_SUPPORTED` from `cublasLtMatmulAlgoGetHeuristic`. NVFP4
GEMM on Blackwell needs **K-contiguous access for both operands**, i.e. the
"TN" layout:

- A: row-major `(M, K)`, `TransA=N`, `ld=K`, scale_A `(M, K/16)` FP8 E4M3
- B: row-major `(N, K)` (logically B^T), `TransB=T`, `ld=K`, scale_B `(N, K/16)` FP8 E4M3
- D: row-major `(M, N)` BF16

Plus per-operand scale mode `CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3` set on
the matmul descriptor.

## Updated projections for TRT plugin path

Production tick budget (bf16 ref = 140 ms / batch=4 / seq=1500):

| variant | MatMul | non-MatMul | total | speedup |
|---|---|---|---|---|
| bf16 ref | 117 ms (217 TFLOPS) | 23 ms | 140 ms | 1.00x |
| current FP8 (`w8a8_absmax_cal2`) | 75 ms (414 TFLOPS) | 17 ms | 92 ms | 1.52x |
| **NVFP4 plugin (projected)** | **22 ms (1409 TFLOPS)** | 17 ms | **39 ms** | **3.59x** |

Speedup over current FP8 frontier: **2.36x**. Vastly better than the
conservative 1.67x in the original handoff (which assumed ~660 TFLOPS dense
peak, half of what we actually measured).

Caveats baked into the projection:
- Attention bmm/baddbmm (~25 ms in bf16, ~25 ms in FP8) was not quantized in
  the current FP8 variant and stays in bf16. To realize the full 22 ms
  MatMul budget under NVFP4 we keep the same scoping: NVFP4 on Linear MatMuls
  only, attention stays bf16. That is already what the projection assumes.
- 17 ms non-MatMul (norms / elementwise / Q-DQ overhead) is a floor we
  can't push without separate plugin work.

## End-to-end correctness check (post-swizzle)

After landing the SF swizzle, `nvfp4_e2e.py` runs the full pipeline on
production-shaped IID Gaussian data:

| M | N | K | cos vs bf16 | rel L2 |
|---|---|---|---|---|
| 6144 | 3072 | 3072 | 0.990969 | 0.134397 |
| 6000 | 2560 | 2560 | 0.990970 | 0.134382 |
| 6000 | 4096 | 2560 | 0.990971 | 0.134383 |
| 6000 | 2560 | 4096 | 0.990969 | 0.134397 |

`diag.py` further verifies that `cos(ref_decode, cublasLt_output) == 1.0000` —
the cublasLt math is exact relative to my Python decode of the same quantized
representation. The remaining cos < 1.0 vs bf16 is pure FP4 quant noise (about
15% rel L2 for IID Gaussian, as expected from 4-bit precision).

This validates the **whole pipeline** end-to-end in pure Python.

## The scale-layout gotcha (key finding)

cuBLASLt NVFP4 scale tensors must be in the CUTLASS "128 x 4 -> 32 x 16"
swizzled SF layout, not row-major. Without the swizzle, the GEMM runs but
produces garbage (cos ~ 0.89 with NaN at certain shapes). The swizzle is
documented at https://docs.nvidia.com/cuda/cublas/index.html#d-block-scaling-factors-layout
and implemented as `torchao.prototype.mx_formats.utils.to_blocked`:

```python
def to_blocked(scale_2d):  # input: (M, K/16) FP8 E4M3 bytes
    n_row_blocks = ceil_div(M, 128)
    n_col_blocks = ceil_div(K_blocks, 4)
    pad to (n_row_blocks*128, n_col_blocks*4)
    view (n_row_blocks, 128, n_col_blocks, 4).permute(0,2,1,3)
    reshape (-1, 4, 32, 4).transpose(1,2).reshape(-1, 32, 16).flatten()
```

For the eventual TRT plugin: weights' swizzled scales bake at engine build
time; activations' swizzled scales compute per-inference in a fused CUDA
kernel before the cuBLASLt matmul call.

## What unblocks next

The plugin work plan in `HANDOFF_NVFP4.md` is now justified to pursue. The
ctypes wrapper here is reusable as a Python-side reference; the TRT plugin
itself needs C++ (or `tensorrt.IPluginV3` from Python).

Sequencing:
1. Per-block FP8 E4M3 scale derivation from cal2 absmax data (no plugin yet).
2. C++ TRT plugin skeleton: `IPluginV3` exposing `IPluginV3OneRunner` with a
   `cublasLtMatmul` call inside `enqueue`. Plugin attributes: M/N/K, pre-quantized
   FP4 weights (already swizzled), FP8 weight scales (already swizzled), FP32
   weight global scale.
3. Activation-side: per-inference kernel that quantizes bf16 -> FP4 E2M1 +
   computes per-block-16 FP8 E4M3 scales + swizzles them.
4. Replace Linear MatMul + DequantizeLinear pattern in `fp8_onnx.py` (forked
   as `nvfp4_onnx.py`) with the plugin op.
5. Apply existing outlier-skip mechanism per cal2 data.
6. Build, validate, score.

Risk now is purely engineering, not feasibility. ~5-7 days bounded.
