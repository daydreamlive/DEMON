# FP8 engine variants

Each row below corresponds to a folder under `variants/<tag>/` containing:

- `engine.engine` — the actual TRT engine binary
- `engine.metadata.json` — TRT build metadata (version, GPU, ONNX SHA)
- `fp8_patch_manifest.json` — exact FP8 quant config and per-Linear log
- `config.json` — exact patch + build + listen commands to reproduce
- `metrics.json` — cos sim, speedup, MatMul counts
- `validation_results.json` — full benchmark JSON
- `audio/01_dance.wav` … `06_folk.wav` — listen-test renders (30 s @ seed 42)

The bf16 reference engine is archived at `variants/bf16/` for safekeeping
and its audio at `variants/bf16/audio/`. All FP8 variants are compared
against bf16 using identical seeds, prompts, and durations.

## Variants

Sorted by speedup, fastest first. Cos sim is vs the bf16 reference on
the shared calibration .npz; speedup is per-tick latency at batch=4,
60 s seq.

| tag | cos | speedup | engine MB | W8A8 MatMuls | W8A16 fallback | SQ α | description |
|---|---|---|---|---|---|---|---|
| sq_a05_full   | **0.9708** | **1.52x** | 4360 | 350 |   3 | 0.5 | SmoothQuant α=0.5, full coverage (no skip). Every W8A8 MatMul gets smoothing. Fastest fp8. |
| w8a8_absmax   | **0.9695** | **1.51x** | 4349 | 193 | 160 | 0.0 | Pure W8A8 absmax (no SmoothQuant, no skip) — the original engine that first produced audio. |
| sq_a05_skip10 | **0.9726** | **1.34x** | 5222 | 306 |  47 | 0.5 | SmoothQuant α=0.5 + outlier-skip ratio=10. Massive-activation layers (mlp.down_proj of layers 6-20) fall back to W8A16; rest gets SmoothQuant. |
| sq_a07_skip10 | **0.9694** | **1.34x** | 5222 | 306 |  47 | 0.7 | Same skip rule as α=0.5 but more aggressive smoothing. Cos worse than α=0.5 — α=0.5 was the sweet spot. |
| w8a8_skip5    | **0.9735** | **1.23x** | 6346 | 104 | 249 | 0.0 | W8A8 absmax + aggressive outlier-skip (ratio=5, no SmoothQuant). Highest cos observed, slowest fp8. |
| bf16          |   1.0000   |   1.00x   | 7989 | — | — | — | Reference. No quantization. ~135 ms/tick. |

`W8A8 MatMuls` is the count of MatMuls where both inputs reach via FP8
Q-DQ chains (i.e. TRT picks FP8 GEMM). `W8A16 fallback` is the count
where the weight is FP8 but the activation stays bf16. Both numbers
exclude the 129 MatMuls that have non-initializer weight inputs
(attention Q×K^T etc).

## Listening

Audio is at `variants/<tag>/audio/<NN>_<prompt>.wav`. To A/B against
bf16, compare with `variants/bf16/audio/<NN>_<prompt>.wav` — same seed
(42), same prompts, same duration (30 s). Prompts in order:

1. `01_dance.wav` — four-on-the-floor electronic
2. `02_jazz.wav` — piano trio with brushed drums
3. `03_ambient.wav` — slow evolving pads, no drums
4. `04_metal.wav` — fast double kick, distorted guitar
5. `05_orch.wav` — strings + brass + timpani
6. `06_folk.wav` — fingerpicked guitar + harmonica

## Reproducing a variant

Each variant's `config.json` has the exact `patch`, `build`, and
`listen` commands. Or use the all-in-one builder:

```
python benchmarks-pr17/fp8_build_variant.py \
    --tag <tag> \
    --description "<text>" \
    --smoothquant-alpha <float> \
    --outlier-skip-ratio <float> \
    --percentile absmax
```

That patches the ONNX, builds the engine, runs validation + listen
test, and copies all artifacts into `variants/<tag>/` while appending
a row to this INDEX.md. Engines are never deleted — every variant
keeps its own folder with the engine binary preserved.

## Observations

- Cos sim spans **0.969 → 0.974** across all FP8 configs. The 0.005
  range is near the audible-difference threshold; whether configs sound
  different requires actually listening — you can't tell from cos.
- The speed-quality tradeoff is roughly linear: every ~0.0015 of cos
  improvement costs ~0.1x of speedup, by routing more MatMuls onto the
  bf16-activation path.
- SmoothQuant **α=0.5 is the sweet spot.** α=0.7 (sq_a07_skip10) gave
  lower cos than α=0.5 at identical speed. α=0.0 (no SQ) is on the
  speed-favoring end of the spectrum.
- The "massive activation" pattern in `mlp.down_proj` (per-channel
  activation absmax up to 10752 vs typical ~30) is structural for this
  DiT. SmoothQuant handles it cleanly at the activation level (max
  amax reduction 207x) but per-output-channel weight quant absorbs
  some of the smoothing factor as weight magnitude growth, capping
  the quality gain.
- **Top recommendation if you want speed:** `sq_a05_full` (1.52x, cos 0.971).
- **Top recommendation if you want quality:** `w8a8_skip5` (1.23x, cos 0.974).
- **Best balanced:** `sq_a05_skip10` (1.34x, cos 0.973).

If after listening none of these are quality-acceptable, the next
realistic moves are SmoothQuant with α tuned per-layer (much more
work, modest expected gain) or moving from per-tensor to per-token
activation quantization (untested with TRT 10.16 FP8 GEMM tactics —
might lose speedup entirely).
