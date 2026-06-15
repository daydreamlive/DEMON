# Does loop-focused prompting affect quality?

**Task:** "Do science and determine if loop-focused prompting has any
bearing on quality" — e.g. prepending `a short perfect loop of <prompt>`.

**TL;DR:** Yes, but it's a *small, free* win, not a transformation.
Prepending a loop phrase lowers the loop-seam spectral discontinuity by
~12–14% on average, and the gain concentrates exactly where loop seams
are audible — dense rhythmic material (techno −22%, deep house −18%) —
while being neutral on already-smooth textures (ambient −2%, lo-fi ~0).
There is no measured downside. Recommendation: make
`a short perfect loop of …` the default prompt prefix for the
loop-focused workflow, and surface it as a one-click toggle rather than
baking it in. **Confirm by ear** — the WAVs are saved for A/B listening;
the metrics only rank candidates.

## Method

- Harness: [`loop_prompting_quality.py`](./loop_prompting_quality.py).
- Pure text-to-music (no source audio) so prompt wording is the *only*
  variable. Fully paired: every base prompt × template rendered at the
  same fixed seeds, duration, and diffusion knobs.
- 4 base prompts (lo-fi, deep house, ambient, techno) × 3 templates ×
  3 seeds = **36 renders**, 60 s each, eager backend, 8 steps, shift 3.0,
  CFG 7.5, seeds `[1528, 42, 9999]`.
- Templates: `baseline` = `{p}` (control), `loop_perfect` =
  `a short perfect loop of {p}` (the phrasing the task names),
  `loop_seamless` = `seamless repeating loop of {p}` (does wording
  matter?).
- Proxies (objective, *not* ground truth — ears decide):
  - `seam_spec_dist` — L2 between the time-averaged log-STFT profiles of
    the first vs. last 0.5 s. Low ⇒ texture matches across the wrap ⇒
    smoother loop. **The headline metric.**
  - `seam_rms_jump_db` — loudness step at the wrap. *Confounded* here:
    many renders fade toward silence at one end, which inflates this to
    40+ dB regardless of prompt, so it's reported but not relied on.
  - `silence_frac` — fraction of near-silent 50 ms frames (dropout guard).

## Results

Aggregate, mean across all prompts × seeds (lower seam = smoother loop):

| template | seam_spec_dist | Δ vs baseline | seam_rms_jump_db | silence_frac |
|---|---|---|---|---|
| baseline | 0.528 | — | 40.4 | 0.079 |
| loop_perfect | 0.452 | **−14.4%** | 38.4 | 0.077 |
| loop_seamless | 0.470 | −11.0% | 38.8 | 0.076 |

Per-prompt `seam_spec_dist`, baseline → `loop_perfect` (mean of 3 seeds):

| prompt | baseline | loop_perfect | Δ |
|---|---|---|---|
| techno | 0.894 | 0.694 | **−22%** |
| deep house | 0.562 | 0.459 | **−18%** |
| ambient | 0.385 | 0.376 | −2% |
| lo-fi | 0.270 | 0.278 | +3% |

## Interpretation

- The effect is **real and directionally consistent in aggregate**, but
  **per-seed variance is large** — on individual seeds the baseline
  sometimes wins. This is a distributional nudge, not a guarantee on any
  single render.
- The win **scales with how much loop seams matter**: percussive,
  high-energy material (techno, house) where a hard wrap is obvious gets
  the biggest improvement; ambient/lo-fi pads already cross the seam
  smoothly, so there's little left to gain.
- `loop_perfect` slightly beats `loop_seamless` — the exact phrasing the
  task suggested is the better of the two. Wording matters a little.
- No downside observed (silence/dropout rate unchanged), so it's a
  zero-risk default.

## Caveats / threats to validity

- Objective proxies, not perceptual ground truth. `seam_spec_dist`
  measures texture continuity, which correlates with — but is not — a
  "good loop". **Listen to the saved WAVs before shipping.**
- Text-to-music isolates the prompt cleanly but the production
  loop-focused workflow often runs over *source audio* (cover/walk),
  where the source constrains content and may shrink the prompt's
  leverage. A follow-up cover-path A/B would close that gap.
- 4 prompts / 3 seeds is enough to see a trend, not to put error bars on
  it. Re-run with `--seeds` widened for a tighter estimate.

## How this feeds the latent-size study

If loop-prompting reliably tightens the seam, shorter looped sections
(which wrap more often, so seam quality matters more) become more
viable — the two experiments compound. See the companion
`min-latent-size` study.

## Reproduce

```
.venv/bin/python scripts/experiments/loop_prompting/loop_prompting_quality.py
# WAVs + metrics.json -> test_output/experiments/loop_prompting/ (gitignored)
```
