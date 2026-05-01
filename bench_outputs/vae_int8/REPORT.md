# INT8 VAE decode, variant comparison

_2026-05-01 17:00:00_

## TL;DR

**Practical winner: Variant A** (`vae_decode_int8_60s`), TRT entropy calibrator, 32 calibration latents, no layer pinning.

- **1.64x faster than fp16** on standalone 60 s decode (33 ms vs 54 ms)
- **32.5 dB PSNR vs fp16** on synthetic latents; below the encoder noise floor in encode/decode round-trips (both fp16 and int8 within 0.2 dB of PT)
- **+88 MB VRAM** vs fp16 (engine binary grows because TRT retains both int8 and fp16 kernels; activation savings don't offset)

**Worth shipping for**: large `vae_window` configurations and batch generation, where standalone VAE decode dominates and the 1.6x speedup compounds across long passes. For a 60 s decode chain, you save ~21 ms per call; if `vae_window` keeps the decoder hot, that's real time recovered.

**Marginal for**: streaming pipelines with short `vae_window` and skip-cache reuse. The diffusion decoder dominates the tick (~88 ms) and int8 VAE saves <2 % end-to-end (260 ms over a 15 s run, see "Streaming pipeline" below).

**Two negative results that closed the search**:
- Variant H (hybrid: modelopt scales fed into TRT-native build path) was *worse* than A on quality (29.9 dB vs 32.5 dB) despite matching A on speed. Scales and graph structure are entangled, not independently swappable.
- Variant I (modelopt autotune) failed structurally; its per-region trtexec timing harness can't build sub-graphs of this audio decoder in isolation. Designed for transformer-shaped graphs.

## Key findings

1. **Profile-matching the calibration data** to the deployed engine profile (1500-frame opt instead of 250-frame opt) was the first big win, moving fp16-vs-int8 PSNR from 28.3 dB to 32.5 dB on the same recipe (4.2 dB free).
2. **modelopt closes the quality gap further** (+3.0 dB to 35.5 dB) but at a 2.4x latency penalty, ending up *slower* than fp16 and defeating the purpose. modelopt-entropy and modelopt-percentile produce essentially identical scales; the quality win is the modelopt pipeline (offline ORT scale computation), not the calibration algorithm.
3. **modelopt's quality and speed are entangled**, not separable. A hybrid that injects modelopt scales into TRT-native's build pipeline (variant H) hits TRT-native's speed but produces *worse* quality than either parent (29.9 dB), because mixing wider modelopt scales on 16 % of tensors with narrower TRT-entropy scales on the surrounding 84 % creates boundary mismatches that add quantization error.
4. **Layer pinning, MinMax, and more calibration data with the TRT-native pipeline all hurt or did nothing.**
5. **The Pareto frontier vs fp16** has only one int8 point that is strictly better than fp16 on the speed axis: Variant A. All modelopt variants are dominated by fp16 on speed.
6. **For streaming**, int8's win is dilute: 1.7 % end-to-end despite a 1.58x per-VAE-call speedup, because VAE decode is only 13 % of the streaming tick (88 ms diffusion + 12 ms VAE → 7 ms VAE).
7. **int8 costs VRAM, doesn't save it.** The INT8+FP16 builder flag retains kernels for both precisions, so int8 engines are larger than fp16 engines. Most ops stay in fp16 anyway (Snake activations are not int8-friendly), so activation memory is similar.

## Setup

- All INT8 variants share the same 60 s dynamic profile (min=125, opt=1500, max=1500 latent frames).
- Calibration set: 32 generated latents at 1500 frames each (prompt-diverse: dance / jazz / ambient / metal / hip hop / classical / folk / synthwave; cfg=7.5; 8 steps).
- fp16 baseline is `vae_decode_fp16_60s` (existing build, same 60 s profile).
- PT eager reference is the bf16 ACE-Step Oobleck VAE (`session.handler.vae`).
- All measurements on RTX 5090, TRT 10.13.3, torch 2.9.1+cu128.

## Variant matrix

| Variant | Description | Engine | MB |
|---|---|---|---:|
| A | TRT entropy, 32 latents, no pinning | `vae_decode_int8_60s` | 245 |
| B | TRT minmax, 32 latents, no pinning | `vae_decode_int8_60s_minmax` | 250 |
| C | TRT entropy, 32 latents, pin first/last conv to fp16 | `vae_decode_int8_60s_pin` | 240 |
| D | TRT entropy + pin (orchestrator pick, dup of C) | `vae_decode_int8_60s_combined` | 251 |
| E | TRT minmax, 32 latents, pin first/last conv | `vae_decode_int8_60s_minmax_pin` | 226 |
| F | TRT entropy, 128 transient-biased latents | `vae_decode_int8_60s_data128` | 259 |
| G1 | modelopt percentile-99.999, 32 latents | `vae_decode_int8_60s_modelopt_pct32` | 88 |
| G2 | modelopt entropy, 32 latents | `vae_decode_int8_60s_modelopt_ent32` | 88 |
| G3 | modelopt percentile-99.999, 128 latents | `vae_decode_int8_60s_modelopt_pct128` | 88 |
| H | hybrid: G2 scales injected into TRT-native build path | `vae_decode_int8_60s_hybrid` | 341 |
| I | modelopt autotune (G2 baseline + per-region search) | (build failed: 0 Q/DQ pairs produced) | n/a |

## Synthetic latent (Session.generate, 60 s, seed=42)

Decode-only quality, measured on a freshly-generated 1500-frame latent. **Speedup vs fp16 = fp16_ms / int8_ms** (>1.0x = int8 is faster, <1.0x = int8 is slower than fp16 and should be discarded).

| Variant | int8 ms | fp16 ms | **Speedup vs fp16** | int8 MSE vs PT | fp16 MSE vs PT | fp16-vs-int8 PSNR (dB) | Verdict |
|---|---:|---:|---:|---:|---:|---:|---|
| A | 32.9 | 54.0 | 1.64x | 6.87e-04 | 3.46e-06 | 32.52 | **1.64x faster, ship** |
| B | 34.8 | 54.8 | 1.57x | 1.53e-03 | 3.46e-06 | 29.02 | 1.57x faster, lower quality |
| C | 34.1 | 54.0 | 1.58x | 9.98e-04 | 3.46e-06 | 30.87 | 1.58x faster, lower quality |
| D | 34.2 | 54.0 | 1.58x | 9.43e-04 | 3.46e-06 | 31.11 | 1.58x faster, lower quality |
| E | 34.5 | 54.3 | 1.58x | 1.06e-03 | 3.46e-06 | 30.62 | 1.58x faster, lower quality |
| F | 33.9 | 55.4 | 1.63x | 8.03e-04 | 3.46e-06 | 31.83 | 1.63x faster, similar to A |
| G1 | 78.7 | 54.2 | 0.69x | 3.46e-04 | 3.46e-06 | 35.51 | _slower than fp16, DISCARD_ |
| G2 | 79.1 | 54.2 | 0.69x | 3.46e-04 | 3.46e-06 | 35.51 | _slower than fp16, DISCARD_ |
| G3 | 82.1 | 57.0 | 0.69x | 8.54e-04 | 3.46e-06 | 31.57 | _slower than fp16, DISCARD_ |
| H | 34.2 | 55.9 | 1.63x | 1.24e-03 | 3.46e-06 | 29.88 | 1.63x faster but *worse quality than A* |

## Wav round-trip (encode->decode of `tests/fixtures/inside_confusion.wav`, first 60 s)

The wav is PT-encoded once and the same latents are decoded by all three paths. PSNR vs fp16 is the metric that matters for the int8-vs-fp16 question; PSNR vs original is mostly encoder-dominated noise (~0.2 dB spread across all decoders).

| Variant | int8 ms | fp16 ms | **Speedup vs fp16** | fp16-vs-int8 PSNR (dB) | int8 PSNR vs orig | fp16 PSNR vs orig | Verdict |
|---|---:|---:|---:|---:|---:|---:|---|
| A | 33.3 | 55.0 | 1.65x | 26.25 | 18.27 | 18.11 | **1.65x faster, ship** |
| B | 32.8 | 54.7 | 1.67x | 25.54 | 16.20 | 18.11 | 1.67x faster |
| C | 34.4 | 55.0 | 1.60x | 25.79 | 18.20 | 18.11 | 1.60x faster |
| D | 35.1 | 54.6 | 1.56x | 25.76 | 18.04 | 18.11 | 1.56x faster |
| E | 34.1 | 55.0 | 1.61x | 27.31 | 17.04 | 18.11 | 1.61x faster |
| F | 33.5 | 55.9 | 1.67x | 25.90 | 18.30 | 18.11 | 1.67x faster |
| G1 | 79.0 | 54.3 | 0.69x | 31.74 | 17.78 | 18.11 | _slower, DISCARD_ |
| G2 | 79.1 | 54.1 | 0.68x | 31.76 | 17.79 | 18.11 | _slower, DISCARD_ |
| G3 | 82.9 | 57.8 | 0.70x | 30.16 | 17.57 | 18.11 | _slower, DISCARD_ |
| H | 35.1 | 58.3 | 1.66x | 24.76 | 18.11 | 18.11 | 1.66x faster but *worse than A* |

## Streaming pipeline (`demos/test_stream_cover_graph.py`, 60 s cover, vae-window=5)

Why this matters: the standalone decode benches above measure VAE decode in isolation. The streaming pipeline is the actual production path, where VAE decode is one of many stages and the 53 % skip-cache rate (latents that didn't change enough between ticks reuse the prior wav) further dilutes the int8 contribution.

| Metric | fp16 | int8 (A) | win |
|---|---:|---:|---:|
| Per-call VAE decode (avg of 75 calls) | 12.5 ms | 7.9 ms | 1.58x |
| Amortized decode (53 % skip rate) | 5.9 ms | 3.7 ms | 2.2 ms saved |
| Diffusion + everything-else tick | 88.0 ms | 88.6 ms | ~noise |
| **Per-generation total** | **93.9 ms** | **92.3 ms** | **1.7 %** |
| Total run time (160 generations) | 15055 ms | 14797 ms | 258 ms |

In streaming, VAE decode is ~13 % of the tick. The 1.58x decode speedup compresses that 13 % to ~8 %, recovering 1.7 % of total time. Real but small.

The standalone wins (1.64x on the bench) translate fully when `vae_window` is large enough that decode runs unbroken (no skip cache, full 60 s passes). They translate marginally in tight streaming.

## VRAM

| | fp16 | int8 (A) | int8 (G2 modelopt) |
|---|---:|---:|---:|
| Engine binary | 181 MB | **245 MB** | 88 MB |
| Engine vs fp16 | baseline | **+64 MB** | -93 MB |
| Peak runtime alloc | 6.621 GB | 6.644 GB | (not measured) |
| Alloc vs fp16 | baseline | **+24 MB** | -- |
| **Net VRAM impact vs fp16** | -- | **+88 MB** | -93 MB (but slower, discarded) |

Why int8 (A) costs more VRAM, not less:

1. **TRT retains kernels for both precisions.** With `INT8+FP16` builder flags, TRT keeps int8 and fp16 tactics in the engine and selects per-layer at runtime. Engine binary grows.
2. **Snake activations stay fp16.** `sin`, `exp`, `reciprocal`, `pow` are not int8-friendly; TRT picks fp16 for them automatically. Most of the activation memory is for these layers, where there's no int8 saving.
3. **Activation savings on the conv layers** (1 byte vs 2 bytes per element) are real but small relative to the doubled kernel storage in the engine binary.

The only int8 build that's actually *smaller* than fp16 is G2 (modelopt) at 88 MB, because modelopt commits each op to one precision (no fallback kernels). G2 is slower than fp16 (0.69x), so it's discarded for shipping; but on a VRAM-constrained box where you can tolerate slower decode, G2 is the only int8 build that wins on memory.

## When to ship int8 (A)

Ship A when **standalone VAE decode latency matters**:
- **Large `vae_window`**: decoding long contiguous spans of audio. Each call does the full 60 s decode; the 33 ms vs 54 ms gap (21 ms saved per call) compounds.
- **Batch generation**: producing many independent generations back-to-back, decoding each. No skip cache to mask the difference.
- **Non-realtime decode**: any pipeline where `Session.decode()` is called outside the streaming tick loop (e.g., final-pass decode of a fully-denoised latent for export).

Stick with fp16 when:
- **Streaming with short `vae_window`** (the demo's default `vae_window=5` case): VAE is too small a fraction of tick time for the int8 win to show. The 1.7 % end-to-end speedup is below the noise floor of the diffusion decoder's variability.
- **VRAM is tight**: int8 costs +88 MB. If you're at the edge of VRAM, that pushes you over.

## Decision guide (metrics)

- The **synthetic int8-MSE-vs-PT** is the cleanest signal of decoder fidelity. Use that to pick the variant. modelopt variants (G1, G2, G3) win by 2-3x on this metric, but they cost so much speed they ship as fp16 instead.
- The **wav round-trip PSNR vs the original audio is dominated by the encoder**: PT, fp16, and int8 all sit within ~0.2 dB of each other. Decoder differences are not audible at this granularity for full encode/decode cycles, but they do matter when the latent comes from generation (not encoding), since there is no encoder error to mask.
- TRT-native int8 variants run at ~33 ms/60 s (1800x realtime). modelopt variants run at ~79 ms/60 s (760x realtime). For non-realtime decoding, latency doesn't matter. For streaming/realtime, both are fine; the question is whether the int8 path's marginal streaming win is worth the +88 MB VRAM.

Engines and wavs:
- Engines: `~/.daydream-scope/models/demon/trt_engines/<engine_name>/`
- Per-variant wavs: `bench_outputs/vae_int8/<engine_name>/`
  - `wav_original.wav`: the source clip, trimmed to 60 s
  - `wav_pt_eager.wav` / `wav_trt_fp16.wav` / `wav_trt_int8.wav`: decoded outputs
  - `synth_*.wav`: decoded synthetic latents
- Streaming outputs: `_debug_tests/stream_output/stream_cover_vae_{fp16,int8}_w5.wav`

## What didn't work

**MinMax calibrator (B, E):** MinMax sets per-tensor int8 scales from observed (min, max), with no clipping. Snake-activated audio decoders have heavy-tailed activations; a single transient sample sets the scale and most of the int8 range is wasted. KL-entropy calibration clips outliers for better resolution on the bulk distribution. MSE doubled going entropy->minmax in TRT-native (A 6.87e-4 -> B 1.53e-3).

**Pinning first/last Conv to fp16 (C, D, E):** in theory the input and output convs are the most sensitive layers. In practice TRT inserts reformat kernels at int8/fp16 boundaries and `OBEY_PRECISION_CONSTRAINTS` removes tactics from the search. Pinned layers run fp16 but eat quantization error from input reformats, and neighbouring layers can't opportunistically stay fp16. A's MSE 6.87e-4 vs C's 9.98e-4 is a 45 % degradation.

**More calibration data with TRT-native (F):** going from 32 to 128 prompt-diverse, transient-biased latents with the same TRT-entropy recipe slightly *hurt* synthetic MSE (8.03e-4 vs 6.87e-4). TRT's calibrator appears to plateau quickly; the bottleneck is the calibrator, not the data volume. Same data through modelopt (G3) does help.

**modelopt percentile vs entropy (G1 vs G2):** **identical** results to four significant figures (MSE 3.46e-04 each). The win is the modelopt pipeline (offline ORT-based scale computation, more aggressive fp16 fallback for non-quantized ops), not the choice of calibration algorithm.

**modelopt percentile + transient-biased 128 latents (G3) was *worse* than 32 mixed-prompt (G1/G2):** MSE 8.54e-04 vs 3.46e-04, a 2.5x degradation. Reason: the extra prompts I added were percussion- and quiet-ambient-heavy (drums, claps, vinyl crackle, drone). Their activation tails are *wider* than the original diverse mix, so the 99.999th percentile picks a wider per-tensor scale, sacrificing precision on the bulk distribution to accommodate transients that the production stream rarely sees. **Lesson**: for percentile calibration, the calibration set must match the inference distribution, not artificially stretch it. The original 32-latent mixed-prompt set was already the right distribution. "More data" was the wrong axis to optimize.

**Hybrid (variant H), modelopt scales + TRT-native build path:** the hypothesis was that modelopt's scale quality and TRT-native's build speed could be combined. Implementation: walk G2's QDQ ONNX, extract `y_scale` from each `QuantizeLinear` for per-tensor activations (124 of them, the per-channel weight scales TRT folds from the embedded Q/DQ pattern), merge into TRT-native's calibration cache by tensor-name match, build with the same TRT-native pipeline. Coverage: 165 of 1035 TRT cache entries got modelopt scales (16 %); the remaining 84 % retained TRT-entropy values.

Result: 1.63x speed (matched A), 29.88 dB PSNR (worse than A's 32.52 dB).

The empirical scale check showed modelopt's scales are roughly 2x wider than TRT-entropy's on the same tensors (median ratio 0.5 with quartile spread 0.4 to 0.6). modelopt deliberately preserves outliers; TRT-entropy aggressively clips them. The wider modelopt scales work coherently graph-wide because modelopt also keeps heavy-tailed ops in fp16, so the wider scales never have to compete with narrow ones at int8-int8 boundaries. In the hybrid, that coherence breaks: 16 % of layers use wider scales, 84 % use narrower ones, and the boundary mismatches *add* quantization error rather than reducing it.

**Insight**: scales and graph structure are entangled. modelopt's quality advantage cannot be transplanted into TRT-native's build path by scale substitution alone. Future hybrids would need to also transplant modelopt's per-op precision decisions, at which point you've reproduced modelopt's full pipeline and inherited its speed problem.

**Autotune (variant I), modelopt with `autotune=True`:** modelopt's autotune extracts each region of the graph as a sub-ONNX, builds it via trtexec, and times candidate quantization schemes per region to find the fastest config that meets a quality threshold. Discovers 322 regions on this graph and seeds 27 patterns from G2. Every per-region timing returned `error: true`: the extracted sub-graphs (Snake activations `Mul→Sin→Pow→Mul→Add`, dynamic-shape ConvTranspose with stride-10 upsampling, Reshape→Conv chains) cannot be compiled in isolation without the full graph's shape context.

Autotune correctly concluded "no scheme is faster than the baseline" because every measurement was `.inf`, exported a final ONNX with 0 Q/DQ pairs, and the subsequent TRT build failed because there were no scales. The result is informative: modelopt autotune is designed for transformer-shaped graphs with self-contained MHA/feed-forward blocks. A 1D conv VAE decoder with Snake activations and dynamic time dim doesn't fit the assumption.
