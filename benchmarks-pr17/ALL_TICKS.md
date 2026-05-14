# All decoder tick measurements (DEMON XL turbo)

Per-tick latency at batch=4, seq=1500 (60s audio). Lower is better.
ms numbers in `[brackets]` are derived from a measured speedup × bf16 ms;
others are directly measured. bf16 measured today at 133.5 ms.

## bf16 reference (baseline)

| Engine | tick (ms) | vs bf16 | Source |
|---|---|---|---|
| bf16 mixed (`decoder_xl-turbo_mixed_refit_b4_60s`) | 133.5 | 1.00× | measured today (validation) |

## FP8 variants from `benchmarks-pr17/variants/INDEX.md` (archived from previous work)

Speedup column is verbatim from INDEX.md; ms = 133.5 / speedup.

| Variant | speedup vs bf16 | tick [ms] | cos | mel-L1 | Notes |
|---|---|---|---|---|---|
| `w8a8_skip3_cal2` | 1.02× | [130.9] | 0.9786 | 0.441 | **slowest fp8 ("original") — user-chosen anchor** |
| `w8a8_skip3` | 1.04× | [128.4] | 0.9749 | 0.566 | |
| `w8a8_skip5_p999_cal2` | 1.13× | [118.1] | 0.7382 | 1.669 | FAILED (clipping) |
| `sq_a05_skip5_cal2` | 1.14× | [117.1] | 0.9742 | 1.143 | |
| `w8a8_skip5_cal2` | 1.15× | [116.1] | 0.9735 | 0.483 | |
| `w8a8_skip5` | 1.23× | [108.5] | 0.9735 | 0.716 | |
| `w8a8_skip8_cal2` | 1.27× | [105.1] | 0.9702 | 0.476 | |
| `sq_a05_skip10` | 1.34× | [99.6] | 0.9726 | 1.218 | |
| `sq_a07_skip10` | 1.34× | [99.6] | 0.9694 | 1.457 | |
| `w8a8_skip8` | 1.36× | [98.2] | 0.9713 | 0.560 | |
| `w8a8_skip20_cal2` | 1.40× | [95.4] | 0.9703 | 0.478 | |
| `w8a8_skip20` | 1.41× | [94.7] | 0.9687 | 0.714 | |
| `w8a8_absmax` | 1.51× | [88.4] | 0.9695 | 0.621 | |
| `w8a8_absmax_cal2` | 1.52× | [87.8] | 0.9718 | 0.499 | **fp8 quality+speed frontier** |
| `sq_a05_full` | 1.52× | [87.8] | 0.9708 | 1.239 | |
| `w8a8_absmax_cal2_attn` | 1.61× | [82.9] | 0.6380 | — | FAILED (cos crash) |
| `w8a8_absmax_cal2_attn_calib` | 1.61× | [82.9] | 0.6331 | — | FAILED (cos crash) |
| `w8a8_attn_g1000` | 1.62× | [82.4] | 0.9124 | — | |
| `w8a8_absmax_cal2_attn_g100` | 1.63× | [81.9] | 0.9180 | — | |
| `w8a8_attn_g200` | 1.64× | [81.4] | 0.9378 | — | |
| `w8a8_absmax_cal2_attn_g500` | 1.65× | [80.9] | 0.9304 | — | |
| `w8a8_absmax_cal2_attn_g10000` | 1.73× | [77.2] | 0.6487 | — | FAILED (cos crash) |

## FP8 engine currently sitting at the prod path on disk

`decoder_xl-turbo_fp8_refit_b4_60s.engine`, built 2026-05-13 11:55 UTC.
Per its `engine.metadata.json` + the patch manifest at
`_onnx_acestep-v15-xl-turbo/decoder_refit/decoder_refit_dynbatch_fp8_manifest.json`,
this engine was built with the SAME patch config as the archived
`w8a8_absmax_cal2_attn` variant (W8A8 absmax + attention_quantized,
softmax_max=1.05, generic_max=10.0). All 193 activation amax values
match the archived variant byte-for-byte; engines differ by 4 KB due to
TRT non-deterministic tactic selection. Effectively the same engine
config, rebuilt.

| Engine | tick (ms) | vs bf16 | Notes |
|---|---|---|---|
| fp8 on prod path = `w8a8_absmax_cal2_attn` rebuild | 83.0 | 1.61× | FAILED quality (cos 0.638 per archived variant) |

## Single-GEMM bench (production shape 6144 × 3072 × 3072) — earlier today

Per-GEMM throughput, NOT full engine tick. Not comparable to the table above.

| Path | ms/iter | TFLOPS | vs torch FP8 |
|---|---|---|---|
| FP8 (`torch._scaled_mm`) | 0.267 | 435 | 1.00× |
| NVFP4 plugin (initial) | 0.188 | 617 | 1.42× |
| NVFP4 plugin (opt 1 + 2) | 0.157 | 738 | 1.70× |
| cuBLASLt direct NVFP4 (ceiling) | 0.082 | 1410 | 3.24× |

## NVFP4 engines built today

Builds 1-4 all overwrote the engine at `decoder_xl-turbo_nvfp4_refit_b4_60s`. Build 5 was written to a distinct path so it survives. Tick measured via `benchmarks-pr17/nvfp4_vs_bf16_validate.py`.

| Build # | Config | tick (ms) | vs bf16 | vs slowest-fp8 (1.02×) | vs fp8 frontier (1.52×) | Status |
|---|---|---|---|---|---|---|
| **1** | no-skip, profile fixed B=4, seq=1500, enc=200, opt_lvl=3 | **58.9** | **2.27×** | **2.22×** | **1.49×** | OVERWRITTEN |
| 2 | skip-5, dynamic B=1..4, fixed seq, fixed enc=200, opt_lvl=3 | 107.4 | 1.26× | 1.22× | 0.83× | overwritten |
| 3 | skip-5, dynamic B=1..4, fixed seq, dynamic enc=32..512, opt_lvl=3 | 99.9 | 1.34× | 1.31× | 0.88× | overwritten |
| 4 | skip-20, dynamic B=1..4, fixed seq, dynamic enc=32..512, opt_lvl=3 | 68.2 | 1.98× | 1.94× | 1.29× | currently on disk @ `decoder_xl-turbo_nvfp4_refit_b4_60s` |
| 5 | no-skip rebuild, dynamic B=1..4, fixed seq, dynamic enc=32..512, opt_lvl=3 | ~59 expected | ~2.27× expected | ~2.22× expected | ~1.49× expected | currently on disk @ `decoder_xl-turbo_nvfp4_noskip_b4_60s` |

## What we know about quality

Latent cos vs bf16 (from `nvfp4_vs_bf16_validate.py`):
- **Build 1**: 0.825
- **Build 3**: 0.868
- **Build 4**: 0.831
- Build 5: not yet measured (engine on disk)

Audio quality scored only for Build 4 via `score_nvfp4_quick.py` against
`listen/bf16_60s/`:
- **Build 4 (skip-20)**: mel-L1 0.59 / MR-STFT 1.18
- Reference: fp8 frontier `w8a8_absmax_cal2`: mel-L1 0.50 / MR-STFT 0.77
- Reference: slowest fp8 `w8a8_skip3_cal2`: mel-L1 0.44 / MR-STFT 0.72

NVFP4 (Build 4) audio is below fp8 frontier quality. MR-STFT is the
cleanest signal here (identical algorithm to INDEX.md's
`score_variants.py`). mel-L1 uses a hand-rolled mel filterbank (torchaudio
hits a broken torchcodec install on this machine), so it's only
directionally informative.

Note: the validation script's latent cos shows the fp8 prod-path engine
at 0.574 vs bf16, while INDEX.md shows fp8 variants at 0.97+. The
validation cos metric here is not directly comparable to INDEX.md's cos
(different bf16 reference / different cal samples) — only mel-L1 /
MR-STFT are directly comparable.

## Key headline numbers

- **Fastest engines on disk**: Build 4 (skip-20) at 68.2 ms / 1.98× bf16, and Build 5 (no-skip rebuild) at ~59 ms / ~2.27× bf16.
- NVFP4 is faster than every fp8 variant ever built (including the failed-quality attention-quantized variants at 77-83 ms).
- Build 4 audio quality: mel-L1 0.59 / MR-STFT 1.18 vs bf16 — below fp8 frontier (0.50 / 0.77). Build 5 audio: generated to `benchmarks-pr17/listen/nvfp4_noskip/`, not yet scored.
- Open work: quality recovery without losing speed. See `acestep/engine/trt/plugins/nvfp4_linear/STATUS.md` "Open work" section.
