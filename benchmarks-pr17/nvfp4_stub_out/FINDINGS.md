# NVFP4 feasibility findings (TRT 10.16 / RTX 5090 / PyTorch 2.9.1)

## Bottom line

**TRT 10.16 does not pick NVFP4 GEMM tactics for any ONNX Q-DQ pattern we
tried on RTX 5090.** All paths produce identical BF16 SM_80 (Ampere)
GEMM tactics at ~190-217 TFLOPS, with no improvement from FP4 dtype
involvement.

The NVFP4 step-function bet through standard ONNX is blocked on TRT
tooling, not on patcher design.

## Throughput summary (M=6000, K=N=3072 single MatMul)

| Pattern | tick (ms) | TFLOPS | TacticName |
|---|---|---|---|
| BF16 baseline | 0.520 | 217.8 | sm80_xmma_gemm_bf16bf16 |
| W4A16 (FP32 scale, single DQ + Cast) | 0.523 | 216.5 | Myelin fused → sm80_xmma_gemm_bf16bf16 |
| ModelOpt 2-DQ W4A16 (FP8+FP32 scales) | 0.531 | 213.3 | Myelin fused → sm80_xmma_gemm_bf16bf16 |
| W4A4 (FP32 scale, Cast roundtrips) | 0.598 | 189.4 | Myelin fused → sm80_xmma_gemm_bf16bf16 |
| W4A4 + BuilderFlag.FP4 (non-strongly-typed) | 0.600 | 188.9 | Same |

All patterns build and run correctly. The FP4 dtype parses, opset 23 is
accepted, ONNX checker passes, engine deserializes and runs. **The only
problem is TRT's tactic selector never picks an FP4 GEMM tactic.**

## What's actually happening

Inspecting the build log (`nvfp4_stub_build.log:325-329`):

- The activation Q-DQ chain gets fused into a Myelin custom kernel
  `__myl_CastReplDivCastCastMulCast` that materializes BF16 tensors
- The weight DQ becomes `__myl_ReplCastMulCast` producing BF16
- The MatMul receives two BF16 inputs and picks an Ampere BF16 tactic

TRT loads `nvinfer_builder_resource_sm120_10.dll` (Blackwell-specific
builder resource), so the hardware *is* detected. But Myelin's tactic
generator emits SM_80 BF16 kernels regardless. The Blackwell SM_120
FP4-specific tactics are either not enumerated for our shape pattern,
or not recognized as applicable to Q-DQ-fronted MatMul.

## What ModelOpt does differently

NVIDIA's ModelOpt uses a custom op `TRT_FP4QDQ` that they later
*expand* into the same 2-DQ pattern we replicated. So the canonical
ONNX representation is identical, but ModelOpt's pipeline includes
calibration and possibly different builder configurations we haven't
matched. There's no evidence the 2-DQ pattern itself unlocks NVFP4
tactics — the pattern we used is the exact one ModelOpt produces.

Hypothesis: ModelOpt's working NVFP4 deployments rely on TRT version
features (e.g., specific opset combinations, shape alignment, or
builder API calls) that aren't documented and aren't accessible from
the standard ONNX path.

## Shape alignment angle (not tested)

NVFP4 tensor core kernels on Blackwell typically require:
- M (rows) aligned to 8 or 32
- N (cols) aligned to 128 or 256
- K (contraction) aligned to 64 or 128

Our M=6000 is NOT aligned to 128 (=128*46.875). This could be why
the tactic picker rejects FP4 variants. Worth testing with M=6144
before declaring NVFP4 permanently dead, but the larger pattern (no
NVFP4 tactic at all) suggests this isn't the only issue.

## Levers left

1. **Try ModelOpt's full quantize pipeline** on the actual DiT
   ONNX. Risk: same silent failure modes as their FP8 ONNX path (the
   one that gave us 2/353 MatMuls quantized).
2. **Custom TRT plugin** wrapping NVFP4 GEMM. Substantial engineering.
3. **TRT 10.17+** when it lands. Future lever.
4. **Pivot to a different step-function lever** — Hadamard rotation,
   outlier residual decomposition, larger effective batch (≥128
   alignment).
5. **Upgrade PyTorch** (currently 2.9.1, `float4_e2m1fn_x2.to()` not
   implemented; would need it for real weight quantization, not for
   the feasibility test).

## What this means for the project

The fastest existing FP8 variant (`w8a8_absmax_cal2`) at 1.52x over
bf16 remains the production ceiling for the moment. NVFP4 through ONNX
is not an immediate win.

If we want to keep pursuing step-function gains in the quant regime,
the next experiments worth running are:

- ModelOpt's full quantize pipeline on the DiT (1 day, validates whether
  their pipeline produces something different than our 2-DQ pattern)
- Hadamard rotation as a preprocessing pass enabling per-tensor
  activation quant to actually preserve quality at FP4 widths
  (multi-day, real engineering)
