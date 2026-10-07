**Q1. Minimum to ship at uniform rigor**

The proposal is close, but “evaluate all 107 uniformly” cannot guarantee “ship 107 passing knobs.” It also repeats some completed work and misses measurements needed to validate the final gains.

- **Keep the vectors; standardize their evaluation.** Rebuilding the extraction corpus, re-extracting directions, or repeating preliminary screening is unnecessary for uniform measurement. Preserve provenance and use identical evaluation prompts, anchors, scorers, seeds, calibration rules and reference distributions.
- **Reuse completed legacy evaluations.** The [inventory](C:/_dev/projects/DEMON/notes/steering_pr/inventory_many_knobs_v2_2026-10-06.md), §§0–3, records full v2 protocol results for four legacy vectors on seeds 0/1/2 and `warm` on 0/1. Complete `warm`’s missing seed; add hold, applicability and cross-effect for all five. Rerunning their existing protocol results is unnecessary unless the measurement definition changes.
- **Measure effects at the gain that ships.** Taking the median of effects measured at three different seed-specific gains does not measure performance at the median gain. Evaluate every seed at the fixed shipped gain, then report median effects/fractions and seed spread. Reuse matching renders where available. Keep AUC separately defined as integration over the curve, not endpoint strength.
- **Check the 63 new vectors’ product measurements too.** `status_many_knobs_v2.md` places P4 completion at 07:45Z and cross-effect completion by 08:25Z, before the third-seed extension finished at 10:03Z. Verify those measurements’ gains against final headers; repeat affected cases. Measuring only the 44 old vectors leaves this unresolved.
- **Apply the absolute quality bar to every origin; apply regression comparisons only where counterparts exist.** A like-for-like rerun of the gate also requires the six displaced v1 vectors—`aggressive`, `black_metal`, `jet`, `polyrhythmic`, `rain_on_surface`, `ukulele`—outside the proposed 44, plus complete comparable measurements for competing v2 vectors. Include `sfx_laden` if claiming coverage of all 46 previous concepts. Missing counterparts mean “not compared,” not “won.”
- **Freeze the decision rules first.** Specify seed aggregation, same-sign CLAP/MuQ requirements, reach requirements and LPAPS tolerance. Hold drift and cross-effect currently have no documented shipping rejection threshold. Preserve empty applicability results instead of substituting catalogue defaults.
- **Then rebuild metadata and final-set checks.** Bind measurements to the actual vector and gain; recompute cross-effect summaries against a common label universe. Rerun stacking on final vectors and gains. Standardize c-pack units and verify physical perturbation/sign mapping: `acid_house`’s negative gain is 28.99 in the header versus 88.33 in `knob_table.csv`. `results_many_knobs_v2.md` explicitly describes conversion into the up-vector magnitude units; unequal numbers alone do not establish a runtime gain bug.

Existing numbers identify five immediate exclusion candidates:

| Pack | Recorded failure |
|---|---|
| `close_mic` | CLAP fractions 0.36 / 0.26 |
| `live_recording` | CLAP 0.26 / 0.54; MuQ 0.34 / 0.26 |
| `sensual` | CLAP 0.56 / 0.52 |
| `wide_stereo` | MuQ 0.40 / 0.46 |
| `warm` | MuQ 0.36 / 0.30 |

These fail the existing bar; their future three-seed results are unknown. No documented bar automatically drops drifting or entangled packs. Listening remains missing for all 107; add a listening check for shipping confidence, while distinguishing it from protocol uniformity.

**Q2. Step-function opportunities**

Ranked by expected payoff per GPU-hour; this is a prioritization judgment—the files provide no GPU-hour estimates for these proposed changes.

1. **Repair benchmark accounting and denominator reporting — no GPU.** Publish steer AUC, PCI AUC, their difference, ratio and seed spread together, with fixed cohorts and survivor counts. Inventory §5 gives new-v2 MuQ means **0.052/0.034 = 1.52×**, versus kept-v1 **0.056/0.045 = 1.24×**: the larger ratio accompanies a smaller numerator. All 140 evaluated knobs yield **1.18×**. Neither survivor selection nor small denominators demonstrates stronger steering.

2. **Calibrate to an actual distortion budget, then solve stacking.** First-at-or-above calibration gives median LPAPS/cutoff **1.19**, with only **8.7%** of new-v2 signs under cutoff. Use existing curves to propose interpolated/bracketed gains, then validate with targeted renders. CPU fitting needs no GPU; validation does. Afterward test the omitted shared-norm budget and fitted attenuation on the final set. Current **35.3%** success for `1/n` is far from 95%; even the 60 combinations containing only shipped vectors achieve **31.7%**. This addresses both misleading comparisons and an actual product limitation.

3. **Validate scorers by domain and make the rules consistent.** The bar requires MuQ for SFX while applicability drops it. `fan` passes positive CLAP **0.94** but fails MuQ **0.54**; `glass_clink` is **0.88/0.54**. Establish a supported SFX criterion using CLAP plus validated PaSST/descriptor evidence, and evaluate SFX on SFX prompts—the 50 protocol prompts are music. Reanalysis of existing scores costs no GPU; missing scoring and new renders do. Do not merely remove whichever scorer rejects a candidate.

4. **Improve direction specificity before expanding the catalogue.** Only **24/126** shipped-new knob-signs move their own label most. Residualisation and dedupe within each variant target this directly. `catalogue_v2_notes.md` reports residualisation reducing confound cosine from **0.92→0.74** for `sub_db` and **0.81→0.61** for offbeat. Direction construction/dedupe are CPU work; validate promising replacements on GPU. Also advance competing variants beyond the 12-prompt screen, as the runbook required, rather than committing early to one winner.

5. **Diagnose long-duration behavior before designing corrections.** P4 flags **63/135**, including **33/63** new shipped vectors. Separate intended-effect loss, growth, reversal and initially wrong-sign effects; inspect `dark`, `pulsing_synth`, `solo_build` and `solo_crest`. Existing numeric reanalysis needs no GPU. Mechanism tests and correction validation require renders; the files do not establish the cause.

6. **Deprioritize further closed-loop work.** Ten tested knobs produced no consistent gain despite strong readout predictability. The tested system was three-pass, between-render gain adjustment, not a demonstrated remedy for temporal drift. Fix measurement, calibration and specificity first.

**Q3. Mistakes and durable notes**

- **The gate compared different protocols and exempted origins from the bar.** Durable rule: eligibility applies to every candidate before replacement selection; measurements identify vector, anchors, gain, prompts and seeds. Reconcile “never regress a knob” with absolute rejection explicitly. `sfx_laden` disappeared while four failing v1 packs remained.

- **Legacy victories were invented by reporting logic.** The status log says “legacy 5/5 beat their v2 counterparts”; [gate.csv](E:/Projects/DEMON/steering-bench/many_knobs_v2/eval/gate.csv) says “v2 counterpart not evaluated” for all five. Durable rule: generate outcome language from actual comparison records, with a separate default-retention state.

- **Product metadata described other vectors.** Twenty-eight retained v1 packs inherited their rejected v2 counterparts’ hold/applicability results. Durable rule: never join experimental measurements to packs by concept name alone.

- **Selection and implementation departed from the plan.** Variant choice occurred before P2; dedupe covered only a; residualisation and shared-norm stacking were omitted. Four overlapping SFX concepts used whichever main-corpus pack wrote last (`status_many_knobs_v2.md`, 01:00Z). Record deviations and make candidate resolution deterministic.

- **Hold reporting misdescribed the statistic.** Results says “63 of 135 knobs lose more than 30%”; [p4_report.json](E:/Projects/DEMON/steering-bench/many_knobs_v2/eval/p4_report.json) contains two-sided drift. The inventory’s “44 lost effect, 19 grew” describes **absolute magnitudes**; signed effects decrease in **47** and increase in **16** flagged cases. Report direction, sign crossings and initial effect alongside the ratio.

- **The inventory overattributes SFX-only tags to scorer asymmetry.** Its claim that “16 new_v2 packs … missed the music bar on MuQ but passed the laxer SFX rule” is contradicted by `p4_report.json`: only **15** have measured SFX-only applicability. `sfx_telephone` has an empty list and SFX CLAP **0.50**; its tag is a fallback. `shimmering_r2` fails music **CLAP 0.583**, while MuQ is **0.75**. `plate_reverb` passes both SFX scorers. Only **13/15** require the CLAP-only exception.

- **Headline comparisons exceeded their evidence.** Results’ “63 new knobs have no AUC on this PC” is superseded by inventory §5. Its “SFX knobs are judged on CLAP alone” contradicts `knob_table.csv`’s MuQ quality gate. The memo’s “level with TADA’s CAA(loc)” is not established by different selected concept sets. Durable rule: publish comparable cohorts, numerator/denominator values and selection provenance; paper ratios provide context, not equivalence.
