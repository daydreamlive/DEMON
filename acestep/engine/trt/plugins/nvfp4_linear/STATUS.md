# NVFP4 Linear TRT Plugin — Status

End-of-day 2026-05-13. WIP.

## What got built (end-to-end)

1. **C++ TRT plugin** (`nvfp4_linear_plugin.cu`, ~500 lines incl. CUDA kernels)
   - Implements `IPluginV3` + V3OneCore/Build/Runtime
   - Registered via `REGISTER_TENSORRT_PLUGIN(NVFP4LinearPluginCreator)`
   - Builds via `build.bat` → `nvfp4_linear_plugin.dll`
   - Toolchain: MSVC 14.44 BuildTools + CUDA 12.8 nvcc + TRT 10.16 headers (vendored, gitignored, regenerable via sparse clone of github.com/NVIDIA/TensorRT @ release/10.16)
   - Local `nvinfer_10.lib` import lib generated from `nvinfer_10.dll` exports (gitignored under `../_build/`)

2. **ONNX patcher** (`acestep/engine/trt/nvfp4_onnx.py`)
   - Walks the bf16 decoder ONNX, replaces each Linear MatMul node with an `NVFP4Linear` plugin op
   - Quantizes weights to packed FP4 + CUTLASS-swizzled FP8 block scales at patch time
   - Bakes a static `act_global_scale` per Linear from cal2 `activation_absmax.json`
   - Outlier-skip mechanism (keep bf16 MatMul on layers above ratio threshold), same shape+L2 lookup as `fp8_onnx.py`
   - Does NOT touch `fp8_onnx.py` — coexists as a parallel pipeline

3. **Engine build driver** (`acestep/engine/trt/nvfp4_build.py`)
   - Loads the plugin DLL → patches bf16 ONNX → builds TRT engine
   - CLI: `--bf16-onnx`, `--absmax-json`, `--engine-dir`, `--outlier-skip-ratio`, `--force`
   - Fixed-seq profile (seq_min=seq_opt=seq_max=1500), dynamic batch (1..4), dynamic encoder (32..512). Fixed seq is required: with dynamic seq the embedding conv1d hits a Myelin "no tactics" failure when the surrounding Linear MatMuls are plugins instead of native ops. Diagnosing why is open work.

4. **Test + bench harnesses** under `acestep/engine/trt/plugins/nvfp4_linear/`:
   - `test_plugin.py`: programmatic-network plugin numerical test (cos 0.9991 vs ctypes ref)
   - `test_patcher.py`: smoke test of patcher → ONNX parser → engine build → execution (cos 0.991)
   - `bench_plugin.py`: single-GEMM throughput at production shape vs torch FP8 and cuBLASLt-direct
   - `STATUS.md`: this file

5. **Validation + listen** in `benchmarks-pr17/`:
   - `nvfp4_vs_bf16_validate.py`: end-to-end DiT tick latency + latent cos
   - `nvfp4_listen_test.py`: 60s clip generation, mirrors `fp8_listen_test.py`
   - `score_nvfp4_quick.py`: mel-L1 + MR-STFT vs bf16 (auraloss-free; MR-STFT identical algorithm to `score_variants.py`)

## Plugin surface (current)

Inputs (3):
- `inputs[0]`: bf16 activation, shape `(..., K)`, M dynamic
- `inputs[1]`: packed FP4 weight bytes, INT8 ONNX initializer with shape `(N, K/2)` (raw uint8 reinterpreted; INT8 chosen because TRT plugin inputs don't accept UINT8)
- `inputs[2]`: CUTLASS-swizzled FP8 E4M3 weight scale bytes, INT8 initializer, flat 1D

Outputs (1):
- `outputs[0]`: bf16 result, shape `(..., N)`

Plugin attributes:
- `K` (INT32), `N` (INT32)
- `weight_global_scale` (FLOAT32) — per-tensor FP32 global scale baked from weight absmax
- `act_global_scale` (FLOAT32) — per-tensor FP32 global scale baked from cal2 per-Linear activation absmax

Weights flow through ONNX initializers (external_data) rather than plugin attributes because the all-Linears-per-attributes path exceeded protobuf's 2 GiB serialization limit for the XL DiT (350 Linears × ~13.5 MB each).

## Forward path (one Linear call)

1. Single-pass per-block FP4 quant + CUTLASS scale swizzle CUDA kernel (vectorized 2× `uint4` bf16 loads, 16 elements per thread). No per-tensor reduction; uses the baked `act_global_scale`.
2. cuBLASLt NVFP4 GEMM (TN layout: A row-major K-contiguous, B row-major K-contiguous via TransB=T, both `CUDA_R_4F_E2M1` with `VEC16_UE4M3` per-block scales, compute `CUBLAS_COMPUTE_32F`, output `CUDA_R_16BF`).
3. Output bf16 written directly to TRT's output tensor.

`alpha` = `act_global_scale * weight_global_scale` computed once at plugin construction, passed as HOST scalar. No per-call sync, no device alpha buffer.

## Throughput (single-GEMM, production shape M=6144 N=3072 K=3072)

Measured by `bench_plugin.py`:

| Path | ms/iter | TFLOPS | vs torch FP8 |
|---|---|---|---|
| FP8 baseline (`torch._scaled_mm` e4m3) | 0.267 | 435 | 1.00× |
| **NVFP4 plugin (final, opt 1+2)** | **0.157** | **738** | **1.70×** |
| cuBLASLt direct NVFP4 (ceiling) | 0.082 | 1410 | 3.24× |

Plugin captures 52% of the cuBLASLt ceiling. The remaining gap is activation-quant kernel + TRT/cuBLASLt dispatch. Closing it would require a CUTLASS-fused activation-quant-and-GEMM (3-5 days, not done).

## End-to-end engine builds (DiT decoder tick @ batch=4, seq=1500)

bf16 reference measured at 133.5 ms today.

| Build | Config | tick (ms) | vs bf16 | engine path | status |
|---|---|---|---|---|---|
| 1 | no-skip, profile batch=4 fixed | 58.9 | 2.27× | `decoder_xl-turbo_nvfp4_refit_b4_60s` | OVERWRITTEN |
| 2 | skip-5, dynamic batch, fixed enc | 107.4 | 1.26× | (same path) | OVERWRITTEN |
| 3 | skip-5, dynamic batch + encoder | 99.9 | 1.34× | (same path) | OVERWRITTEN |
| 4 | skip-20, dynamic batch + encoder | 68.2 | 1.98× | `decoder_xl-turbo_nvfp4_refit_b4_60s` | currently on disk |
| 5 | no-skip rebuild, dynamic batch + encoder | (expected ~59) | (expected ~2.27×) | `decoder_xl-turbo_nvfp4_noskip_b4_60s` | currently on disk |

Build 1's engine is gone (overwritten by build 2). Build 5 is the rebuilt no-skip into a distinct path so it can be preserved.

vs the slowest fp8 archived (`w8a8_skip3_cal2`, 1.02× bf16 ≈ 130.9 ms):
- Build 4 (currently on disk): 1.92× over slowest fp8
- Build 5 (currently on disk): ~2.22× over slowest fp8 (projected)

vs the fp8 frontier (`w8a8_absmax_cal2`, 1.52× bf16 ≈ 87.8 ms):
- Build 4: 1.29× over fp8 frontier
- Build 5: ~1.49× over fp8 frontier (projected)

## Audio quality (single config measured)

Build 4 (skip-20) listen-test audio scored against bf16:

| | mel-L1 | MR-STFT |
|---|---|---|
| **NVFP4 build 4 (skip-20)** | **0.59** | **1.18** |
| fp8 frontier `w8a8_absmax_cal2` (INDEX.md) | 0.50 | 0.77 |
| slowest fp8 `w8a8_skip3_cal2` (INDEX.md) | 0.44 | 0.72 |

NVFP4 audio is below fp8 frontier quality at this config. **MR-STFT 1.18 is the cleanest signal here** (same algorithm as `score_variants.py`, directly comparable to INDEX.md). The mel-L1 metric uses a hand-rolled mel filterbank because `torchaudio.transforms.MelSpectrogram` hits a broken `torchcodec` install on this machine, so mel-L1 isn't bit-exact to INDEX.md — only directionally informative.

Build 5 (no-skip) audio has been generated to `benchmarks-pr17/listen/nvfp4_noskip/` but not scored yet.

## Open work

- Audio quality of build 5 (no-skip) — generated, not scored
- Quality recovery without losing speed. Speed is monotonic vs skip ratio (more skip = slower), so improving cos via skip is a poor trade. Better levers:
  - Hadamard rotation pre-pass (weights at build time + activations in plugin) — classic W4A4 quality enabler, untried
  - p99 percentile instead of absmax for `act_global_scale` (smaller G → finer block-scale grid for bulk values, outliers clip)
  - Per-Linear dynamic global scale on the worst-outlier layers only (hybrid: static for most, dynamic for ~10 layers) — recovers per-call quant adaptivity for outlier-heavy layers without paying the reduction cost on every Linear
- Confirming the on-disk fp8 prod-path engine is not the fp8 frontier (`decoder_xl-turbo_fp8_refit_b4_60s.engine` built 2026-05-13 11:55 UTC is `w8a8_absmax_cal2_attn` config = cos 0.638 FAILED quality). The actual fp8 frontier (`w8a8_absmax_cal2`) is archived under `benchmarks-pr17/variants/w8a8_absmax_cal2/`.
- Refit support: plugin INT8 inputs aren't refittable; LoRA refit currently disabled for NVFP4 engines (just logs a warning). If LoRA support is needed, weights would need to flow through bf16 initializers that get quantized at engine build time, not at patch time.
- Dynamic seq profile: currently fixed at 1500 because dynamic seq triggers a Myelin "no tactics" failure on the embedding conv1d. Reverse-engineering the Myelin fusion that breaks would unlock dynamic-duration support.

## Gotchas (keep in mind for the next pass)

1. **Anonymous namespace + `REGISTER_TENSORRT_PLUGIN` conflict**: nvcc generates a `_GLOBAL__N_...` symbol for anonymous namespaces, which collides with TensorRT's own internal anonymous namespace symbol. Use a named namespace (`nvfp4_plugin`) and a top-level `using` alias for the macro.
2. **`__nv_fp8_e4m3(uint8_t)` ctor converts the integer value**, doesn't bit-interpret. To construct an FP8 with specific bits, set `.__x` directly.
3. **`attachToContext` returns a clone**: any per-instance device state must live on the clone.
4. **`PluginField` is unowned**: numpy arrays passed to `trt.PluginField` must remain alive for the lifetime of the build, or TRT reads garbage.
5. **`out_tensor.dtype = trt.DataType.BF16` is required** when marking the network output. Without it, TRT defaults to FP32 output and inserts an implicit cast that misinterprets the plugin's bf16 bytes as FP32 (manifests as zero-interleaved data in `out_bf16`).
6. **Plugin inputs cannot be UINT8** (TRT rejects); use INT8 with raw bytes reinterpreted.
7. **Plugin attributes can't carry the FP4 weights** for a full XL DiT — total bytes (~4.7 GB) exceeds protobuf's 2 GiB serialization limit. Weights must flow through ONNX initializers (`external_data` eligible) as plugin inputs.
8. **CUTLASS SF scale swizzle is non-negotiable**: cuBLASLt NVFP4 expects scales in a 128×4 → 32×16 tiled layout, not row-major. Without the swizzle the GEMM runs and returns garbage (cos ~0.89 with NaN at some shapes). Implementation: `torchao.prototype.mx_formats.utils.to_blocked`. TN layout also required (K-contiguous for both operands).
