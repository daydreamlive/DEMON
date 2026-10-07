# v3 smoke batch (2026-10-07, local 5090 authorised by Ryan, run concurrently)

Builds on v3_smoke_runbook.md and status_v3_smoke.md (first smoke: ridge descriptor directions pass on the up side,
beat the tag knobs 2 to 8x at matched LPAPS, fail on the down side; onset 0.88 / lra 0.90 Spearman on 5 alphas, 1 seed).
Same rules as v3_smoke_runbook.md: code in C:\_dev\projects\DEMON-steer-bench scripts/steering_bench/, data under
E:\Projects\DEMON\steering-bench\v3_smoke\<smoke>\, no commits, no box, never touch the backend on :1318, mv not rm,
no attribution, status to notes/steering_pr/status_v3_smoke.md (append a section per smoke), report <= 10 lines at terminal
state or FLAG only. Several smokes share the one GPU (32 GB, ~10 GB taken by the demo backend): check nvidia-smi before
loading a model, keep each process under 6 GB, FLAG on OOM instead of retrying. Ryan allowed about 30 extra GPU minutes
per smoke to use a larger set where it sharpens the verdict; prefer more prompts and seeds over more alphas.

## S1. Per-sign ridge, and 3 seeds (one agent)
- Fit per-sign vectors for onset_rate, centroid, lra: fit ridge on the rows in the low tercile (down vector) and the high
  tercile (up vector) separately, versus the middle; also try the alternative "ridge toward the low tail" = ridge fit on
  the bottom half only. Keep the first-smoke symmetric ridge as the comparison row.
- Render with smoke_drive.py on 24 prompts (the 12 screening plus 12 more from the 50 protocol prompts) x seeds 0,1,2,
  same alpha logic. Report per descriptor and sign: Spearman median, LPAPS/cutoff, delta at matched LPAPS vs control, for
  symmetric / per-sign / low-tail. Verdict: does per-sign fitting give a working down side, and do onset and lra clear 0.9
  with 3 seeds.

## S2. Minimal pairs from A's recipe grammar (one agent)
- Find the prompt grammar / generator in the instrument-controlnet repo (read-only there; Ryan said the corpus work lives
  somewhere in that repo; scripts\corpus_preprocess_chunk.py is the preprocessing entry). Do not modify that repo.
- Add per-sample seeding to scripts/steering_bench/capture_resid.py (noise seeded per clip inside the batch, so a pair
  shares its seed exactly); verify two single-clip renders with the same seed are bit-identical to the batched ones.
- Pick 3 concepts that are a single grammar field AND have a scorer we already have: suggested tempo (descriptor tempo /
  onset_rate), drums present vs absent (perc_ratio), and one instrument swap scored by CLAP anchors. 150 pairs each: same
  recipe, same seed, one field changed. 10 s clips, capture all 24 blocks (means), same corpus format as v2.
- Direction = mean of per-pair activation differences (per block; pick the block by held-out pair separation). Render
  through smoke_drive.py (extend it for a CLAP-scored concept if needed) on 24 prompts x 2 seeds against the v2/ship tag
  knob for the same concept (percussive for drums, the closest instrument knob, tempo knob if shipped else PCI only).
- Report: pair-separation AUC per block, Spearman / LPAPS / delta at matched LPAPS vs control, both signs. Verdict: do
  minimal-pair directions beat tag-vs-rest directions, and are they two-sided.

## S3. Shared-norm stacking rule (one agent)
- The v2 stacking pass tried 1/n and shipped-only; the shared-norm-budget rule was never run. Define: when k knobs are
  active, scale each knob's shift so the summed shift norm equals the single-knob budget at its calibrated gain (and a
  variant: sqrt(k) scaling). Reuse the v2 stacking code and the 300 combos (set_tools.py / many_knobs_v2 stacking stage;
  read status_many_knobs_v2.md for where it lives), on the shipped bundle's 97 knobs at shipped gains, 12 prompts, seed 0,
  plus seed 1 if time allows.
- Report: fraction of combos under the PCI cutoff per rule (1/n, shipped-only, shared-norm, sqrt(k)) and the mean own-label
  effect retained per knob under each rule. Verdict: which rule ships in v3.

## S4. Perceptual knob mapping (one agent, CPU only)
- From the ship pass probe data (E:\Projects\DEMON\steering-bench\many_knobs_v2\ship\, LPAPS vs alpha per knob, sign,
  seed), fit a per-knob monotone response curve and build a lookup that maps UI knob position (0 to 1 per sign) to alpha
  such that LPAPS grows linearly in knob position up to the cutoff. Report curvature statistics (how far from linear the
  raw mapping is, median and worst knobs) and write the map as a sidecar JSON next to the bundle format proposal (do not
  change the bundle or the runtime yet). Include a 5-knob listening list for Ryan (knob, prompt, two positions) only if the
  curvature is large enough to matter; otherwise say it is not.

## L. Literature review (one agent, no GPU)
- 30 to 45 min with web search. Questions: (1) is "regression-fit descriptor directions, per-sign, calibrated to a perceptual
  budget, with minimal-pair prompts from a recipe grammar and stacking rules, on a music diffusion model" a small
  contribution on its own, and what is the closest prior work for each ingredient (concept activation vectors, representation
  engineering, TADA arXiv 2602.11910, SAE audio steering, sliders / concept sliders in image diffusion, any music steering
  work 2025-2026); (2) does anything in the review suggest a smoke test we are not running (list concretely, with cost).
- Write notes/steering_pr/lit_review_v3_2026-10-07.md: per ingredient, closest prior work with citation, what differs,
  honest verdict; then the suggested smokes. No praise, no hedging where a paper is explicit.
