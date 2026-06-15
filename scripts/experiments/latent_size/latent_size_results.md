# Minimum acceptable latent size for the loop-focused workflow

**Task:** "We currently loop over one section of 60 second latent. Does
it need to be 60 seconds?" Smaller window ⇒ less DiT compute per
(re)diffusion ⇒ lower param-update latency (knob-to-ear), and a smaller
activation/workspace footprint ⇒ VRAM headroom.

**TL;DR:** No, 60 s is not required — but the gains have *two different
knees* and they meet around **20–30 s**:

- **Latency payoff saturates at ~20 s.** Generate (DiT) time falls 60→30 s
  (0.58→0.45 s) then *plateaus at ~0.38 s* for any window ≤20 s, because
  at small T the DiT is launch/overhead-bound, not compute-bound. Going
  below 20 s buys essentially nothing more on generate.
- **Quality degrades below ~15 s.** Dead-space (`silence_frac`) climbs
  steadily as the window shrinks — 0.095 at 60 s → 0.20 at 10 s → 0.27 at
  6 s. Short windows can't sustain continuous musical content.

**Recommendation:** prototype the live loop window at **30 s** (safe) or
**20 s** (aggressive). 20 s roughly halves total offline compute vs 60 s
and shrinks the would-be TRT workspace, while the quality proxies stay at
the 60 s baseline. Do **not** drop below ~15 s without careful listening —
dead-space risk rises sharply and the latency curve is already flat there.
**Confirm by ear** on the saved clips before committing a window size.

## Method

- Harness: [`latent_size_sweep.py`](./latent_size_sweep.py).
- Pure text-to-music (the loop-focused generation case). Only `duration`
  varies; prompt/seed/knobs fixed. 2 prompts (techno, lo-fi) × 8 windows
  × 3 seeds = **48 renders**. Eager backend, 8 steps, shift 3.0, CFG 7.5,
  seeds `[1528, 42, 9999]`.
- Timing brackets `torch.cuda.synchronize()` so wall time reflects GPU
  work, not async launch.
- Proxies as in the loop-prompting study: `seam_spec_dist` (loop-seam
  texture continuity), `silence_frac` (near-silent 50 ms frames — the
  dead-space / degeneracy signal), plus `rms_db` / `centroid_hz` sanity.

## Results (mean across prompts × seeds)

| window | frames | gen_s | gen speedup | dec_s | gen+dec | seam_spec_dist | silence_frac |
|---|---|---|---|---|---|---|---|
| 60 s | 1500 | 0.582 | 1.00× | 0.209 | 0.791 | 0.582 | 0.095 |
| 45 s | 1125 | 0.549 | 1.06× | 0.147 | 0.696 | 0.519 | 0.105 |
| 30 s | 750 | 0.448 | 1.30× | 0.093 | 0.541 | 0.673 | 0.124 |
| **20 s** | **500** | **0.391** | **1.49×** | **0.052** | **0.443** | **0.500** | **0.105** |
| 15 s | 375 | 0.387 | 1.50× | 0.039 | 0.426 | 0.634 | 0.152 |
| 10 s | 250 | 0.379 | 1.54× | 0.025 | 0.404 | 0.439 | 0.201 |
| 8 s | 200 | 0.378 | 1.54× | 0.021 | 0.399 | 0.491 | 0.164 |
| 6 s | 150 | 0.380 | 1.53× | 0.016 | 0.396 | 0.504 | 0.265 |

## Interpretation

- **Generate is compute-bound only 30–60 s, then launch-bound.** The
  ~0.38 s floor is fixed per-call overhead (kernel launches, Python, fixed
  setup) that T can't shrink. So the DiT latency win is real but *capped*
  — ~1.5× by 20 s and flat thereafter.
- **Decode scales ~linearly with T** (0.21→0.016 s) because VAE decode is
  length-bound. NB: this matters for *offline* render only — the live
  engine uses a windowed VAE (fixed ~1 s decode), so decode latency is
  already decoupled from the loop-window size in production.
- **The quality limiter is dead-space, not seam quality.**
  `seam_spec_dist` is noisy with no monotonic trend (shorter ≠ obviously
  worse seams). But `silence_frac` roughly doubles by 10 s and rises
  sharply below 15 s: the model leaves more empty space when asked to
  fill a very short section. 20–30 s holds near the 60 s baseline.
- **Net:** the latency knee (~20 s) and the quality knee (~15 s) bracket a
  comfortable operating point at **20–30 s**.

## What this means for the live engine (and VRAM)

The transferable result is the **DiT compute scaling**, not the absolute
eager ms. In the live path the window is the `walk_window_s` /
`walk_window_T` the backend feeds the DiT (`acestep/streaming/ace_backend.py`),
and the DiT runs on TRT. Two consequences:

- **Param latency:** a smaller window means proportionally less DiT work
  per re-diffusion when a knob/prompt changes — the direct payoff for the
  "chip away at param latency" goals. The eager launch-bound floor here is
  pessimistic; lower-overhead TRT may keep scaling a bit past 20 s.
- **VRAM (ties to #242):** TRT engine workspace is reserved by the
  profile's max T. Today only **60/120/240 s** profiles ship
  (`acestep/paths.py`). Realizing a 20–30 s window live therefore requires
  **building a sub-60 s TRT profile** — which itself reserves less
  workspace, a direct VRAM win on top of the latency one.

## Caveats / threats to validity

- Objective proxies, not perceptual truth. The decision-relevant signals
  (`gen_s` scaling, `silence_frac`) are solid; final musical acceptability
  is human-judged — **listen to the saved WAVs**.
- Eager single-shot timings. Absolute ms differ on TRT/windowed; the
  *shape* of the curve is the result.
- Text-to-music; a source-driven cover/walk run may behave differently
  (the source supplies content, possibly lowering the dead-space risk at
  short windows). Natural follow-up.
- 2 prompts / 3 seeds shows the trend, not tight error bars. Widen
  `--seeds` / `--durations` to sharpen the knee.

## Reproduce

```
.venv/bin/python scripts/experiments/latent_size/latent_size_sweep.py
# add e.g. --durations 25 22 20 18  to zoom in on the knee
# WAVs + metrics.json -> test_output/experiments/latent_size/ (gitignored)
```
