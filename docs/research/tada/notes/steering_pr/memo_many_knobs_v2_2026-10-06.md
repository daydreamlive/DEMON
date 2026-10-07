# Memo: many-knobs v2 outcome and how it compares with TADA (2026-10-06)

Sources: results_many_knobs_v2.md (Executive summary, Comparison with TADA), status_many_knobs_v2.md,
E:\Projects\DEMON\steering-bench\many_knobs_v2\tables\, v1 eval at E:\Projects\DEMON\steering-bench\many_knobs\eval\.

## Summary
- 107 packs ship: 63 new v2 knobs (a 20, d 23, b 13, c 7), 39 v1 packs kept by the regression gate, 5 legacy.
  140 v2 knobs were evaluated; 44 failed the quality bar (CLAP and MuQ intended-sign fraction >= 0.6 on some sign).
- Coverage is the gain: 2.3x the v1 knob count, a new articulation category, more rhythm, space and dynamics knobs,
  every new knob reaches the PCI cutoff on both signs on every seed, gains are 2 to 3 seed medians.
- Per-knob strength did not improve: the new knobs' effect at calibrated gain is 0.95x (MuQ) / 1.02x (CLAP) of v1's.
- On the TADA metric the shipped set (where measurable) sits at 1.22x PCI on MuQ, level with TADA's CAA(loc) on
  ACE-Step (1.27x) and far above our earlier TADA-recipe replication on SA3 (0.61x).

## TADA comparison (AUC of sign-corrected alignment delta vs LPAPS up to the PCI cutoff, mean of both directions)
Ratio = mean steer AUC / mean PCI AUC over knobs; frac > 1 = knobs whose AUC beats their own PCI.

| row set | n | MuQ AUC med / mean | MuQ ratio vs PCI | frac > 1 | CLAP AUC med / mean | CLAP ratio vs PCI | frac > 1 |
|---|---|---|---|---|---|---|---|
| v2 shipped with local AUC (39 v1-kept + 5 legacy) | 44 | 0.052 / 0.058 | 1.22 | 0.68 | 0.079 / 0.070 | 3.13 | 0.91 |
| v2 shipped new knobs | 63 | not on this PC | n/a | n/a | not on this PC | n/a | n/a |
| v2 new, proxy (endpoint effect, not AUC) | 63 | 0.043 / 0.053 | 0.95x of v1 | n/a | 0.066 / 0.067 | 1.02x of v1 | n/a |
| v1, all knobs | 46 | 0.046 / 0.054 | 1.20 | 0.63 | 0.073 / 0.065 | 3.09 | 0.91 |
| SA3 earlier TADA replication (CAA, 7 concepts) | 7 | 0.021 mean | 0.61 loc / 0.71 all | 2/7 | 0.006 mean | 0.67 | n/a |
| Paper ACE-Step PCI (7 non-vocal) | 7 | 0.094 mean | 1.00 | | 0.045 mean | 1.00 | |
| Paper ACE-Step CAA (loc) | 7 | 0.119 mean | 1.27 | 6/7 | 0.065 mean | 1.44 | n/a |
| Paper ACE-Step SAE (loc) | 7 | 0.132 mean | 1.40 | 6/7 | 0.071 mean | 1.58 | n/a |

- Missing: v2 per-knob auc.json and PCI results are only in B2 (many_knobs_v2_2026-10-05/eval/); no B2 client here.
- Caveats: different model and site, LPAPS calibrated per model, so only ratios compare. 50 prompts and one seed vs
  their 100; our concepts are screened survivors of our own catalogue (inflates the ratio) vs their fixed 7.
  SA3 has no vocals (paper rows are non-vocal); CLAP PCI is tiny on SA3 and many knobs used CLAP labels: trust MuQ.

## Regression gate (46 shared concepts) and legacy five
- v1 kept: 39 (28 lost on intended-sign fraction or LPAPS/cutoff > 1.1x v1's; 11 not re-found by v2).
- v2 won: 7 (aggressive, black_metal, jet, polyrhythmic, rain_on_surface, ukulele, sfx_laden). sfx_laden then
  fails the quality bar, so neither its v1 nor its v2 pack ships.
- Legacy five (bright, dense_arrangement, percussive, rough, warm) all ship as legacy packs (status log: they beat
  their v2 counterparts on effect at cutoff; the results table says "v2 counterpart not evaluated", unreconciled).
  TADA AUC from the 2026-10-05 eval: MuQ ratio 1.13 (4/5 above PCI), CLAP 2.71. Rough is below PCI on both
  (0.57 / 0.66). In the v2 protocol with v2 anchors, warm moves the wrong way on MuQ on both signs
  (intended-sign fraction 0.36 / 0.30) and would fail the quality bar if it were not legacy.

## Product negatives
- Stacking: 300 combos (200 pairs, 100 triples) at calibrated gain, LPAPS vs the strictest member's cutoff:
  no scaling 0%, 1/sqrt(n) 4.3%, 1/n 35.3% under cutoff. No rule reaches the 95% target; 1/n ships as the default.
  The shared-norm-budget rule was not run.
- Hold drift: 63 of 135 knobs lose more than 30% of their 0-10 s effect by 20-30 s (44 of 91 shipped knobs with
  hold data); median drift 0.29. Example: aggressive 0.12 to 0.03.
- SFX coverage: SFX expansion of 40 rows / 57 candidates gave 14 survivors and 9 shipped knobs; 20 of the 107
  shipped knobs are sound effects. On SFX prompts median intended-sign fraction is MuQ 0.54 / CLAP 0.75
  (music 0.63 / 0.83). MuQ cannot see SFX, so SFX applicability is CLAP-only.
- Also: cross-effect, own label moves most on only 40 of 270 knob-signs (median |own z| 1.13 vs median max
  off-target 1.84); closed loop (10 knobs) gave no consistent gain over open loop.

## Recommended next three actions
1. Pull the v2 eval auc.json files (results/, results_s1/) from B2 via a box agent and fill the "v2 shipped new
   knobs" row: CPU only, minutes. Until then the v2 TADA claim rests on the 44 knobs with local AUC.
2. Stacking: test the shared-norm budget (sum of active alphas scaled to one knob's cutoff norm) and a
   per-pair LPAPS-fitted scale on the 300 saved combos; ship whichever reaches 95%, else cap active knobs at 2
   with 1/n and say so in the UI.
3. Hold drift: split the 44 drifting shipped knobs by cause (does the 20-30 s window lose the effect at the
   same LPAPS, or does the music move away from the concept) on the saved hold renders, and flag drift in pack
   metadata until fixed. SFX scoring (PaSST in the protocol in place of MuQ for SFX knobs) comes after these.
