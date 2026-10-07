# Durable follow-ups from the many-knobs v2 run (2026-10-06)

Sources: inventory_many_knobs_v2_2026-10-06.md, astra_second_opinion_many_knobs_v2_2026-10-06.md, results_many_knobs_v2.md.
Not for the ship pass (ship_uniform_rigor_runbook.md); these are the larger items to pick up later.

## A. Step-function opportunities, ranked by expected payoff per GPU-hour

1. Calibrate to a distortion budget, not "first probe at or above cutoff". Median LPAPS/cutoff at the shipped gain is 1.19;
   only 8.7% of new knob-signs sit at or under the cutoff. Interpolate the gain from the existing LPAPS curves (CPU), validate
   with one render per sign (small GPU). Everything downstream (fractions, stacking bar, TADA comparison) is taken past the
   cutoff by a knob-dependent amount today. This is also the first move on stacking: no rule keeps 95% of combos under cutoff
   (1/n: 35%; combos of shipped-only vectors: 32%); the shared-norm-budget rule was never run.
2. Direction specificity: only 24 of 126 shipped new knob-signs move their own label more than every other label. Residualise
   descriptors against the known confounds (catalogue_v2_notes: sub_db confound cos 0.92 -> 0.74, offbeat 0.81 -> 0.61 with
   residualisation) and dedupe within every variant, not only a. CPU to build, GPU only to validate replacements. Also carry
   two variants per concept through P2 and choose after, as the runbook said; the 12-prompt screening choice is unvalidated.
3. SFX scoring: the bar uses MuQ, applicability uses CLAP only, and the 50 protocol prompts are music. MuQ is blind to SFX
   (fan: CLAP 0.94 / MuQ 0.54; glass_clink 0.88 / 0.54). Define one SFX criterion (CLAP plus PaSST/descriptor evidence) and
   score SFX knobs on SFX prompts. Reanalysis of existing scores is CPU; new renders only for the SFX set (20 packs).
4. Hold drift: 63 of 135 flagged (signed: 47 decrease, 16 increase, 11 cross sign). Separate loss, growth, reversal and
   wrong-initial-sign before designing a fix; no mechanism is established. CPU first.
5. Benchmark accounting (no GPU): publish steer AUC, PCI AUC, difference, ratio and seed spread per fixed cohort. The new-63
   MuQ ratio 1.52x vs kept-v1 1.24x comes from a smaller PCI denominator (0.034 vs 0.045), not a larger numerator (0.052 vs
   0.056); all 140 evaluated knobs sit at 1.18x. The memo's "level with TADA's CAA(loc)" compares different concept sets.
6. Specificity as a gate: cross-effect is measured (24/126 shipped knob-signs move their own label most) but not applied.
   Gate per sign, clamp the knob to passing sides. No listening grid; spot listening by ear stays informal.
7. Closed loop: deprioritise. Ten knobs with ridge R^2 > 0.5 gave no consistent gain; the prototype adjusts gain between
   renders and does not address drift.

## B. Mistakes of method or reporting (not of effort), and the rule each one leaves

- The regression gate compared v1-protocol numbers with v2-protocol numbers and exempted kept_v1 and legacy packs from the
  bar (4 kept_v1 + warm fail it and shipped; sfx_laden failed it and shipped nowhere). Rule: the bar applies to every
  candidate before any replacement choice; a measurement names its vector, anchors, gain, prompts and seeds.
- "Legacy 5/5 beat their v2 counterparts" in the status log and "stronger by effect at cutoff" in the manifest were fixed
  strings; gate.csv says "v2 counterpart not evaluated" for all five. Rule: outcome text is generated from the comparison
  record; "retained by default" is its own state.
- 28 kept_v1 packs carry hold and applies_to measured on the v2 vector that lost. Rule: never join measurements to packs by
  concept name; join by vector hash and gain.
- Departures from the runbook (variant chosen at screening, dedupe on a only, residualisation skipped, shared-norm stacking
  skipped, 4 SFX concepts resolved by last-writer-wins) were not recorded in a deviations section. Rule: results files carry
  "asked / done / deviation" per stage.
- Hold drift was reported as "lose more than 30%" while the statistic is two-sided. Rule: report signed drift and sign
  crossings with the ratio.
- Legacy pack headers say 32 pairs; the notes said 50. Rule: cite provenance from the header, not from memory.
- Three metadata shapes shipped side by side (flat v1, dict v2, legacy dict) and broke the demo ticks for 68 packs. Rule: one
  pack metadata shape per release; the installer validates it.
- Inventory corrections (from Astra, checked against p4_report.json): sfx-only tags from the CLAP asymmetry are 13 of 15
  measured cases, not 16 (sfx_telephone's tag is a fallback, plate_reverb passes both SFX scorers); hold "44 lost / 19 grew"
  counted magnitudes, signed counts are 47 / 16.

## C. Listening observations (Ryan, 2026-10-06, demo with the 107-pack set, prompt "guitars")
Three random examples from a few minutes of listening, NOT an exhaustive list and not to be over-indexed on; the point is that
metric-only knobs have audible failures. The fix is in the measurement, not in listening: specificity (own label moves most,
per sign, at the shipped gain) becomes a gate, and one-sided concepts get per-sign vectors. No listening grid.
- close_mic, negative side: becomes distorted guitar, not a far or room mic. (Fails the bar on its v1 numbers.)
- wide_stereo, negative side: becomes a full band, not more mono. (Fails the bar on its v1 numbers.)
- arpeggiated: off-target (detail not recorded).
Pattern: the negative side of a one-sided concept is the corpus anticorrelate, not the absence of the concept. Feeds A.2
(residualisation, per-sign vectors). Packaging mitigation available from the ship pass data: per-sign bar flag in the manifest,
UI clamps the knob to the passing side(s).
