# Handoff: NVFP4 / step-function quantization push

This document captures the state of the FP8 → NVFP4 optimization work as
of session end. Pick up a fresh conversation and use this as the
context.

## Working directory and environment

- Repo: `C:\_dev\projects\DEMON`, branch `ryanontheinside/arch/xl-fp8-wip`
- Hardware: Windows 11 / RTX 5090 / Blackwell SM_120 / TRT 10.16.1.11
- Python: 3.11, torch 2.9.1+cu128, onnx 1.21.0, onnxruntime 1.22.0,
  modelopt 0.43.0
- **Engine rebuilds take ~60 seconds** end-to-end via the harness. Iterate freely.

## Hard rules (durable)

1. **Never delete engines or code.** Use `mv` to `~/.claude-trash/` or
   `/d/daydream-scope-overflow/` when freeing space. Never `rm`.
2. **Never push to GitHub** or take any public-facing action without
   explicit per-action approval immediately prior. Ask in one message,
   wait for reply, then act.
3. **No AI/Claude attribution** anywhere — commit messages, comments,
   PRs, docs. Zero exceptions.
4. **No em dashes in prose** (commas / parens / colons instead).
5. **Stay strictly within scope.** Do exactly what's asked.
6. **Quality is non-negotiable.** No step-count reduction or other
   quality-trading speed wins.
7. The user is on a CTO track at DaydreamLive. Concise, technically
   dense responses. State the tradeoff. Don't pad.

## What the project is optimizing

ACE-Step XL turbo decoder DiT, deployed via TRT engines. Goal: maximize
inference speed at preserved audio quality. Reference is bf16. Speedup
measured as bf16 tick / variant tick at batch=4, seq=1500 (60 s of audio).

## Current Pareto frontier (end of session)

From `benchmarks-pr17/variants/INDEX.md`:

| tag | cos | mel-L1 | speedup | engine MB | cal |
|---|---|---|---|---|---|
| **w8a8_absmax_cal2** | 0.9718 | 0.499 | **1.52x** | 4349 | cal2 |
| **w8a8_skip20_cal2** | 0.9703 | 0.478 | 1.40x | 5086 | cal2 |
| w8a8_skip8 | 0.9713 | 0.560 | 1.36x | 5335 | cal1 |
| w8a8_skip8_cal2 | 0.9702 | 0.476 | 1.27x | 5615 | cal2 |
| **w8a8_skip5_cal2** | 0.9735 | 0.483 | 1.15x | 6704 | cal2 |
| **w8a8_skip3_cal2** | 0.9786 | 0.441 | 1.02x | 8028 | cal2 |
| bf16 (ref) | 1.0000 | 0.000 | 1.00x | 7989 | — |

Production-default: `w8a8_absmax_cal2` (1.52x, cos 0.9718). Highest fp8
quality: `w8a8_skip3_cal2` (cos 0.9786, mel-L1 0.441 — bf16-grade).

## What this session established (verified, definitive)

### TRT 10.16 + FP8 is already at peak

`benchmarks-pr17/fp8_stub.py` shows our production fp8 W8A8 pattern
hits 414.8 TFLOPS on a 6144×3072×3072 MatMul. `torch._scaled_mm` (which
calls cuBLASLt FP8 directly) gets 422.3 TFLOPS at the same shape. So
TRT is at 98% of what cuBLASLt-direct delivers. **There is no
FP8-plugin step function. The TRT FP8 GEMM path is essentially solved.**

### Where the bf16 tick time goes

From `benchmarks-pr17/profile_engine.py`:
- Total tick: ~92 ms (1.52x)
- MatMul: ~75 ms (81% of tick)
  - Quantized Linears (~350 W8A8 MatMuls): ~50 ms at ~414 TFLOPS
  - **Un-quantized attention bmm/baddbmm (128 dynamic-operand MatMuls)**: ~25 ms at bf16 throughput
- Non-MatMul (norms, elementwise, Q-DQ overhead): ~17 ms
- Bandwidth utilization: 2-3% of HBM3X peak — clearly compute-bound on tensor cores.

### NVFP4 via standard ONNX patterns is BLOCKED in TRT 10.16

Tried these patterns in `benchmarks-pr17/nvfp4_stub.py`:
- W4A16 with FP32 per-block scale + Cast to bf16: builds, but tactic is
  `sm80_xmma_gemm_bf16bf16` (Ampere BF16), not NVFP4. 217 TFLOPS, same
  as bf16 baseline.
- W4A4 with FP32 scale + Cast: 189 TFLOPS, slower than bf16. Q-DQ
  overhead.
- ModelOpt 2-DQ pattern (FP8E4M3 block × FP32 per-tensor): parses, same
  BF16 tactic, 213 TFLOPS.
- All combinations with `BuilderFlag.FP4` (non-strongly-typed): same
  result.

TRT loads `nvinfer_builder_resource_sm120_10.dll` (Blackwell builder
resource present), but Myelin's tactic generator does not emit NVFP4
GEMM kernels for our ONNX patterns. The Q-DQ chains get fused into
`__myl_CastReplDivCastCastMulCast` ops that round-trip FP4 through bf16
before MatMul.

### ModelOpt 0.43.0's ONNX `quantize` does not implement NVFP4

`modelopt.onnx.quantization.quantize(quantize_mode="nvfp4", ...)` raises
`RuntimeError: Invalid quantization mode choice: nvfp4`. The dispatcher
only handles int8/fp8/int4. The NVFP4 exporter
(`modelopt/onnx/export/nvfp4_exporter.py`) exists but isn't wired into
the end-to-end ONNX quantize pipeline yet.

The torch-side `modelopt.torch.quantization` may have NVFP4 (untried).

### Attention FP8 quantization works but ceiling is 1.15x

`acestep/engine/trt/fp8_onnx.py` was extended with
`--quantize-attention` (full implementation present — both fp8_onnx.py
patcher logic and CLI plumbing in `run_fp8_patch.py` and
`fp8_build_variant.py`).

Best result: **1.65x speedup at cos 0.93** (hardcoded `generic_max=500`)
or **1.61x at cos 0.92** with traced-scale calibration. The
attention-amax tracing logic (find source Linear via producer graph
BFS) is implemented in `fp8_onnx.py:_trace_to_source`.

Theoretical ceiling: 1.15x over current FP8 (12 ms saved on 25 ms attn
budget). Empirical: 1.07-1.09x. **Not a step function.**

### Quality issues with attention quant

Per-tensor activation scale doesn't fit 128 attention MatMuls cleanly.
Per-layer attention output absmax ranges from ~10 (early cross-attn Q)
to ~2700 (cross-attn K outputs). Even traced per-MatMul scales hit a
quality ceiling around cos 0.93. To recover further would need
per-block FP8 activation scales (untried) or quantize-skip on the worst
outlier attention layers.

## Calibration state

- **cal1** (archived): 8 prompts × 16 calls × batch=4 = 64 samples.
  Backed up as `~/.daydream-scope/models/demon/calibration/decoder_xl_fp8/calibration.npz.v1` and `activation_absmax.json.v1`.
- **cal2** (current): 25 prompts × 200 calls × batch=4 = 800 samples.
  Generated by extended `scripts/collect_decoder_calibration.py` (the
  PROMPTS list was tripled to cover vocal-forward, transient-extreme,
  low-freq-dominant, high-freq-dominant, dense-polyphony, sparse content).
  Wider distribution found 2-10x larger activation outliers than cal1
  (layer 15 mlp.down_proj absmax: 4128 → 22272). Plus per-Linear
  `output_absmax` added via forward hook in
  `scripts/collect_activation_absmax.py`.

## Quality metrics

`benchmarks-pr17/score_variants.py` computes:
- **cos sim** (latent, from `fp8_vs_bf16_validate.py`)
- **text-CLAP** (LAION CLAP music-and-speech, n=6 prompts, noisy)
- **audio-CLAP vs bf16** (perceptual fidelity, noise floor ~0.05 due to
  CLAP's stochastic chunking)
- **mel-L1 vs bf16** (deterministic, lowest variance — primary signal)
- **MR-STFT vs bf16** (multi-resolution STFT, tracks mel-L1 1:1)

Results in `benchmarks-pr17/variants/scores.json`.

## Disk situation

C: drive filled during attention quantization iterations. Moved to
`/d/daydream-scope-overflow/`:
- `trt_engines-10.16/` (old TRT version snapshot)
- `decoder_xl-turbo_bf16mix_dynbatch_b8_60s_trt10.13_backup/`
- `decoder_base_mixed_b8_60s/`, `decoder_base_mixed_refit_b8_60s/`
- `decoder_mixed_b8_60s/`, `decoder_mixed_b8_240s/`
- `decoder_mixed_refit_b8_60s/`, `decoder_mixed_refit_b8_120s/`,
  `decoder_mixed_refit_b8_240s/`

About 100+ GB moved. If more space needed during plugin work, candidate
moves: `decoder_xl-turbo_mixed_refit_b8_60s/` (we use b4),
`decoder_xl-turbo_bf16mix_dynbatch_b8_60s/`, and older `dreamvae_*`
engines.

## Iteration commands

### Build a new variant
```
python benchmarks-pr17/fp8_build_variant.py \
    --tag <name> \
    --description "<text>" \
    --smoothquant-alpha 0.0 \
    --outlier-skip-ratio <float> \
    --percentile absmax
```
(Plus optional `--quantize-attention --attention-softmax-max 1.05 --attention-generic-max 200`.)

Single command runs: patch → build → validate → listen → archive (~60s).
Adds row to `INDEX.md`. Engine + audio + manifest + metrics persisted
to `variants/<tag>/`.

### Re-score all variants
```
python benchmarks-pr17/score_variants.py
```

### Re-run calibration (regenerate cal2)
```
python scripts/collect_decoder_calibration.py --num-prompts 25 --max-calls 200
python scripts/collect_activation_absmax.py
```

### Profile a variant
```
python benchmarks-pr17/profile_engine.py --engine <tag> [--iters N]
```

## Files extended this session

- `acestep/engine/trt/fp8_onnx.py`:
  - Added `quantize_attention`, `attention_softmax_max`, `attention_generic_max` kwargs to `patch_bf16_onnx_to_fp8`
  - Added `output_amax_lookup` build from JSON
  - Added attention Q-DQ insertion section (after main W8A8 act loop)
  - Added `_trace_to_source` backward-traversal helper that walks producer chain to find source Linear and apply scale_factor from intervening Mul ops
  - Added `attention_log` to manifest
- `scripts/collect_decoder_calibration.py`: tripled PROMPTS to 25.
- `scripts/collect_activation_absmax.py`: added forward hook capturing
  `output_absmax` per Linear (alongside existing input absmax).
- `benchmarks-pr17/run_fp8_patch.py`: added `--quantize-attention`,
  `--attention-softmax-max`, `--attention-generic-max` flags.
- `benchmarks-pr17/fp8_build_variant.py`: same flags plumbed through.

## Files created this session

- `benchmarks-pr17/score_variants.py` — CLAP + mel-L1 + MR-STFT scorer
- `benchmarks-pr17/profile_engine.py` — TRT IProfiler per-layer timing
- `benchmarks-pr17/inspect_engine_tactics.py` — engine tactic
  inspector via `IEngineInspector`
- `benchmarks-pr17/inspect_attn_matmuls.py` — finds the 128 dynamic
  MatMuls
- `benchmarks-pr17/inspect_attn_amax.py` — reads QKV output_absmax
- `benchmarks-pr17/nvfp4_stub.py` — feasibility stub for NVFP4 ONNX
  patterns
- `benchmarks-pr17/nvfp4_modelopt_test.py` — ModelOpt quantize test
- `benchmarks-pr17/fp8_stub.py` — verified TRT FP8 GEMM at 414 TFLOPS
- `benchmarks-pr17/torch_scaled_mm_test.py` — verified torch._scaled_mm
  FP8 at 423 TFLOPS
- `scripts/collect_attention_amax.py` — ORT-based attention tensor
  capture (FAILED: bf16 Conv not supported by ORT)
- `benchmarks-pr17/nvfp4_stub_out/FINDINGS.md` — detailed NVFP4 stub
  findings

## The only remaining step-function lever

**Custom TRT plugin wrapping cuBLASLt NVFP4 GEMM.**

Rationale:
- TRT 10.16 won't pick NVFP4 tactics for any ONNX pattern (verified)
- ModelOpt's ONNX quantize doesn't implement NVFP4 yet (verified)
- Compute is the bottleneck; only lower-precision GEMM unlocks more throughput
- Blackwell NVFP4 peak: ~660 TFLOPS dense (2x FP8's 330)
- Expected at 80% utilization: ~528 TFLOPS achieved
- Tick savings: 75 ms → 37.5 ms (MatMul halved) → 55 ms total
- Speedup over current FP8: 92/55 = **1.67x**
- Speedup over bf16: 140/55 = **2.55x**

### Recommended approach

1. **Verify cuBLASLt NVFP4 GEMM exists and delivers on RTX 5090**
   (the verification spike — this is what's running next in the
   session that produced this handoff).
2. Build a TRT plugin (Python via `tensorrt.IPluginV3` or C++) that:
   - Accepts bf16 input
   - Quantizes activations to NVFP4 with per-block FP8E4M3 scale (block size 16)
   - Holds pre-quantized FP4 weights with their per-block FP8 + per-tensor FP32 scale
   - Calls cuBLASLt NVFP4 GEMM
   - Returns bf16
3. Modify `fp8_onnx.py` (probably better as `nvfp4_onnx.py` fork) to
   emit the plugin op in place of the Linear MatMul + DequantizeLinear
   pattern.
4. Calibrate per-block FP8 scales from cal2 data (the existing
   `activation_absmax.json` has per-channel absmax; per-block (size 16)
   can be derived).
5. Apply existing outlier-skip mechanism to keep the worst layers on
   FP8 W8A8 fallback (the cal2 absmax data already identifies them).
6. Build, validate, score.

### Risk

- If cuBLASLt NVFP4 doesn't deliver claimed throughput on RTX 5090
  (possible if Blackwell support is preview/partial in CUDA 12.8), the
  entire NVFP4 bet falls. Verification spike is the gate.
- If verified, plugin engineering is well-bounded (~5-7 days).

## What NOT to retry

- **NVFP4 via standard ONNX QuantizeLinear / DequantizeLinear** — no
  combination of opset, scale dtype, builder flags, strongly-typed
  vs not, or 2-DQ patterns triggers NVFP4 tactic selection in TRT
  10.16 (extensively verified).
- **Percentile mode (p99_9)** as activation scale — catastrophically
  clips signal (cos crashes to 0.74). Per-tensor absmax is the only
  viable activation scale unless you go per-block.
- **SmoothQuant alpha sweeps** — the entire SmoothQuant family is
  dominated by w8a8 + outlier-skip on spectral metrics. The technique
  works mathematically but the per-output-channel weight quant
  absorbs the smoothing factor in a way the VAE then amplifies as
  spectral artifact.
- **Step count reduction** (8 → 4 via different sampler) — explicitly
  ruled out by the user: trades quality for speed.
- **Hardcoded attention scales** — hit cos 0.93 ceiling. Either
  per-block scales or quantize-skip on outlier layers is needed.

## Auxiliary context references

- Original FP8 patch rationale and history:
  `acestep/engine/trt/fp8_onnx.py` docstring (top of file).
- Massive activation pattern: layer 15-20 `mlp.down_proj` have channel
  outliers up to 22000. Structural to this DiT.
- Per-channel outlier residual decomposition was discussed as another
  step-function lever (split outlier channels to bf16 side-path, rest
  W8A8/W4A4). Not pursued this session. Would be the natural follow-up
  if NVFP4 plugin path falls through.
- Hadamard rotation (QuaRot/SpinQuant) was discussed. Doesn't deliver
  step function on its own with FP8; useful as quality enabler for
  W4A4 to flatten outliers. Layer it on top of NVFP4 plugin if quality
  recovery is needed.

## Memory state pointers (in `~/.claude/projects/C---dev-projects-DEMON/memory/`)

- `project_demon_attribution.md` — pay homage to ACE-Step, no AI credit
- `feedback_scope_strictly.md` — exact scope, no silent siblings
- `user_role_daydreamlive.md` — CTO track, biweekly with Doug
- `project_demon_company_arrangement.md` — promotion arrangement
- `MEMORY.md` index

## Verification spike result (RESOLVED, 2026-05-13)

The cuBLASLt NVFP4 GEMM verification spike now PASSES via direct ctypes
binding to `cublasLt64_12.dll` (CUDA 12.8) bundled with torch 2.9.1+cu128.
See `benchmarks-pr17/cublaslt_nvfp4_spike/RESULT.md` for full detail and
`nvfp4_gemm.py` for the wrapper.

Measured throughput on RTX 5090 (Blackwell SM_120):
- 6144 x 3072 x 3072 (production shape): **1409 TFLOPS**
- 8192 x 8192 x 8192: **1517 TFLOPS** (peak)
- 2048 x 2048 x 2048: 832 TFLOPS

Ratio over FP8 (422 TFLOPS at same shape): **3.4x**.
Gate (>= 500 TFLOPS) cleared by ~2.8x.

Critical configuration: NVFP4 GEMM requires K-contiguous "TN" layout for
both operands. A as `(M, K)` row-major, B as `(N, K)` row-major with
`TransB=T`, plus `CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3` scale mode on
both A and B. Per-block scales are FP8 E4M3, one per 16 K-elements.

Updated projection (replaces the conservative 1.67x in the section below):
- New MatMul budget: 75 ms -> 22 ms (FP8 414 TF -> NVFP4 1409 TF on same shape)
- Total tick: 92 ms -> ~39 ms
- Speedup over bf16: **3.59x** (vs. current FP8 best of 1.52x)
- Speedup over current FP8 frontier: **2.36x**

The plugin work is now justified to pursue. Risk is engineering, not
feasibility.

## Plugin built + end-to-end engines built (UPDATED, 2026-05-13 EOD)

The C++ TRT plugin and the parallel ONNX patcher are both done. Engines
build end-to-end and produce audio. Per-Linear correctness matches the
Python reference at cos 1.000. Single-GEMM throughput is 738 TFLOPS at
the production shape (1.70× over `torch._scaled_mm` FP8). The full
end-state lives in `acestep/engine/trt/plugins/nvfp4_linear/STATUS.md` —
read that for the canonical state. Quick pointers below.

Code:
- `acestep/engine/trt/plugins/nvfp4_linear/nvfp4_linear_plugin.cu` — C++ IPluginV3 plugin (~500 lines)
- `acestep/engine/trt/plugins/nvfp4_linear/build.bat` — nvcc + MSVC build
- `acestep/engine/trt/nvfp4_onnx.py` — bf16→NVFP4 patcher (parallel to `fp8_onnx.py`, does not modify it)
- `acestep/engine/trt/nvfp4_build.py` — end-to-end driver (patch + build)
- `benchmarks-pr17/nvfp4_vs_bf16_validate.py`, `nvfp4_listen_test.py`, `score_nvfp4_quick.py`

Headline end-to-end ticks (DiT decoder, batch=4, seq=1500, bf16 = 133.5 ms):

| Engine | tick (ms) | vs bf16 |
|---|---|---|
| NVFP4 build 1 (no-skip, fixed batch profile) — overwritten | 58.9 | 2.27× |
| NVFP4 build 4 (skip-ratio 20, currently on disk @ `decoder_xl-turbo_nvfp4_refit_b4_60s`) | 68.2 | 1.98× |
| NVFP4 build 5 (no-skip rebuilt, currently on disk @ `decoder_xl-turbo_nvfp4_noskip_b4_60s`) | ~59 expected | ~2.27× expected |
| `w8a8_absmax_cal2` (fp8 frontier, archived) | 87.8 | 1.52× |
| `w8a8_skip3_cal2` (slowest fp8 anchor, archived) | 130.9 | 1.02× |

Audio quality (Build 4 measured; Build 5 generated, not scored):

| | mel-L1 vs bf16 | MR-STFT vs bf16 |
|---|---|---|
| NVFP4 build 4 (skip-20) | 0.59 | 1.18 |
| `w8a8_absmax_cal2` (fp8 frontier) | 0.50 | 0.77 |
| `w8a8_skip3_cal2` (slowest fp8) | 0.44 | 0.72 |

So: faster than every fp8 variant ever built, but quality is below the fp8 frontier. Quality recovery (Hadamard, p99 percentile, hybrid dynamic scale on outlier layers) is the open work.

### Gotchas worth remembering
1. CUTLASS SF scale swizzle is non-negotiable (`torchao.prototype.mx_formats.utils.to_blocked`) plus TN layout.
2. Plugin inputs cannot be UINT8 — use INT8 with raw bytes reinterpreted.
3. Plugin attributes can't carry the FP4 weights for full XL DiT (~4.7 GB of weight bytes) due to protobuf 2 GiB serialization limit. Weights flow as ONNX initializers, plugin reads them as `inputs[1]`/`inputs[2]`.
4. `out_tensor.dtype = trt.DataType.BF16` is required on the network output binding; without it TRT inserts an implicit cast that misreads bf16 bytes as FP32.
5. Dynamic seq triggers Myelin "no tactics" on the embedding conv1d when MatMuls are plugins. Engines built with fixed seq=1500. Investigating that interaction is open work.
6. The fp8 engine sitting at the production path `decoder_xl-turbo_fp8_refit_b4_60s` is patch-config-identical to `w8a8_absmax_cal2_attn` (FAILED quality, cos 0.638). The actual fp8 frontier is archived under `benchmarks-pr17/variants/w8a8_absmax_cal2/`.

## Start here (fresh-conversation prompt)

> Continue the NVFP4 work. Plugin is built and producing engines end-to-end.
> Read `acestep/engine/trt/plugins/nvfp4_linear/STATUS.md` for the canonical
> end-state (engines + ticks + audio quality + gotchas + open work). Two
> NVFP4 engines are on disk: `decoder_xl-turbo_nvfp4_refit_b4_60s` (skip-20,
> 68.2 ms, audio scored at mel-L1 0.59 / MR-STFT 1.18) and
> `decoder_xl-turbo_nvfp4_noskip_b4_60s` (no-skip, ~59 ms, audio generated
> but not scored). NVFP4 is faster than every fp8 variant ever built, but
> quality is below the fp8 frontier. Next focus is quality recovery without
> losing speed — try Hadamard rotation pre-pass, p99 percentile for
> `act_global_scale`, or hybrid dynamic-scale on the worst-outlier layers
> only. Score build 5's audio first (`benchmarks-pr17/listen/nvfp4_noskip/`)
> to see if no-skip is actually worse than skip-20 on audio metrics or just
> on the latent cos that we don't fully trust.
