# Inventory: many-knobs v2 steering run (SA3 medium), 2026-10-06

Read-only audit for the re-measure decision. Every count below was recomputed from the files, not copied from the
results or memo. Where a claim in results_many_knobs_v2.md, the memo or the status log disagrees with the data, the
data wins and the disagreement is stated.

Sources read: notes/steering_pr/{results_many_knobs_v2.md, status_many_knobs_v2.md, memo_many_knobs_v2_2026-10-06.md,
many_knobs_v2_runbook.md, results_many_knobs.md, status_many_knobs.md, catalogue_v2_notes.md};
E:\Projects\DEMON\steering-bench\many_knobs_v2\{tables\, eval\, packs_final\}; E:\Projects\DEMON\steering-bench\many_knobs\eval\
(v1 table1.csv, gain_table.csv, results\*\auc.json); E:\Projects\DEMON\steering-bench\2026-10-05\eval\auc.json (legacy five);
the 107 installed pack headers (~/.daydream-scope/models/demon/steering_packs/sa3/medium) and the 51 old packs in
~/.claude-trash/steering_packs_v1_2026-10-06; the box's gate/pack code as copied to this session's scratchpad (v2box/final.py,
which already carries the 09:10Z LPAPS gate fix; the box's own final copy was not read).

Missing: many_knobs_v2_runbook.v1-timeline.md does not exist in notes/steering_pr (the runbook says it was kept; it was not).
No v2 listen WAVs, corpus, directions or screening audio are on this PC (B2 only). The eval tree is otherwise complete for
what ran: auc.json for 153 packs at seed s0, 153 at s1, 111 at s2 (98 = every pack of every quality-bar knob, plus 13
tie-break packs). Only legacy_warm lacks s2, and it never ran s2.

## 0. Origin counts (verified)

| origin | packs | how identified | calibrated_gain shape in header | gain seeds |
|---|---|---|---|---|
| new_v2 | 63 | no provenance.kept_from; vector from corpus_v2 (54) or corpus_sfx (9) | dict {median,min,max,reached,seeds,cutoff} | 0,1,2 on all 63 |
| kept_v1 | 39 | provenance.kept_from = "v1"; vector byte-identical to the v1 pack in the trash (39/39) | flat numbers (pos, neg, pos_reached, ...) | 0 only (v1 phase 6) |
| legacy | 5 | provenance.kept_from = "legacy"; vector byte-identical to the old pack in the trash (5/5) | dict | 0,1,2 (warm: 0,1) |

The UI agent's split reconciles exactly: 68 dict = 63 new_v2 + 5 legacy (gains re-measured under the v2 P2/P3/P5 protocol);
39 flat = the kept_v1 packs (gains carried over from v1 phase 6). The legacy packs re-used their old vectors but got v2-protocol
gains.

Correction to the brief: the legacy headers say provenance.pairs = 32 (plus patch_pairs 8, 8 base prompts, discover.py), not 50.
The old trash packs carry the same 32. Nothing in the headers supports "50-pair vectors".

Category mix: new_v2 sound_effect 15, rhythm 11, space 7, timbre 7, genre 6, instrument 6, articulation 6, mood 2, dynamics 2,
abstract 1; kept_v1 genre 10, instrument 7, production 6, sound_effect 5, space 3, mood 3, abstract 2, timbre 2, dynamics 1;
legacy timbre 4, rhythm 1. Variants: new_v2 a 20, d 23, b 13, c 7; kept_v1 all a (v1 single block); legacy single block.

## 1. Shipped packs (107)

Column notes:
- vector corpus: corpus = v1 5000-clip capture; corpus_v2 = 20000-clip merged v2 capture; corpus_sfx = SFX supplement capture.
  `*` = the header's capture field is empty (all b and c packs); the corpus is read from the directions path instead.
- gain = the header's calibrated_gain (knob value at which mean LPAPS vs unsteered reaches the PCI-all cutoff; the first probe at
  or above it). new_v2 and legacy: median over the listed seeds. kept_v1: v1 phase 6, seed 0 only. c packs: the neg gain is in the
  up vector's magnitude units (knob_table shows a different number for the dn pack; see section 5).
- quality bar = some sign with CLAP and MuQ intended-sign fraction >= 0.6 (50 TADA test prompts, seed 0). new_v2 and legacy:
  v2 knob_table.csv. kept_v1: v1 table1.csv (v1 phase 6, v1 anchors, v1 gain). The v2 bar was never applied to kept_v1 packs.
- TADA AUC: steer/PCI, each the mean of the two directions' sign-corrected AUC (sa3_tada_score auc), computed here from the
  local auc.json. It is now available for all 107 (the memo said 63 were "not on this PC"; the eval tree has since arrived).
  For c packs the + side is the up pack's + direction and the - side is the dn pack's + direction (as final.py builds the
  knob table); this pairing is my reading of the code, not a stored number.
- hold drift: header hold_30s. `*` on kept_v1 = measured on the v2 counterpart vector, not on the shipped v1 vector.
- applies_to: header. kept_v1 `*` = copied from the v2 counterpart's applicability run (different vector); "catalogue default"
  = no applicability run at all.

| # | pack | pedal category | origin | variant | blocks | vector corpus | gain seeds | gain + | gain - | cutoff reached + / - (per seed) | quality bar: CLAP/MuQ intended-sign frac per sign, pass? | TADA AUC local? steer/PCI (source) | hold drift | applies_to | regression gate compared |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | abs_forest | abstract | new_v2 | d | 16 | corpus_v2 | 0,1,2 | 592.97 | 296.49 | yyy / yyy | + 0.78/0.84, - 0.84/0.44: PASS | yes: MuQ 0.011/0.010, CLAP 0.020/0.005 (v2 s0) | DRIFT 0.94 | music,sfx | nothing (no v1 counterpart) |
| 2 | accelerando | rhythm | new_v2 | a | 11 | corpus_v2 | 0,1,2 | 73.09 | 146.18 | yyy / yyy | + 0.52/0.66, - 0.94/0.76: PASS | yes: MuQ 0.028/0.017, CLAP 0.013/0.008 (v2 s0) | ok 0.08 | music,sfx,abstract | nothing (no v1 counterpart) |
| 3 | acid_house | genre | new_v2 | c | 12 | corpus_v2* | 0,1,2 | 57.98 | 28.99 | yyy / yyy | + 0.62/0.76, - 0.22/0.58: PASS | yes: MuQ 0.068/0.090, CLAP 0.049/0.035 (v2 s0) | ok 0.22 | music,sfx,abstract | nothing (no v1 counterpart) |
| 4 | aggressive | mood | new_v2 | a | 13 | corpus_v2 | 0,1,2 | 55.61 | 27.81 | yyy / yyy | + 0.84/0.84, - 0.94/0.84: PASS | yes: MuQ 0.081/0.023, CLAP 0.062/0.020 (v2 s0) | DRIFT 0.73 | music,sfx,abstract | v2 vs v1: v2 won |
| 5 | arpeggiated | rhythm | new_v2 | b | 11,12,10 | corpus_v2* | 0,1,2 | 17.44 | 17.44 | yyy / yyy | + 0.86/0.60, - 0.84/0.48: PASS | yes: MuQ 0.018/0.023, CLAP 0.073/0.016 (v2 s0) | ok 0.05 | music,sfx,abstract | nothing (no v1 counterpart) |
| 6 | bedroom_intimate | space | new_v2 | d | 19 | corpus_v2 | 0,1,2 | 430.41 | 430.41 | yyy / yyy | + 1.00/0.80, - 0.68/0.82: PASS | yes: MuQ 0.133/0.026, CLAP 0.114/0.027 (v2 s0) | DRIFT 0.58 | music,sfx,abstract | nothing (no v1 counterpart) |
| 7 | black_metal | genre | new_v2 | a | 11 | corpus_v2 | 0,1,2 | 54.63 | 27.32 | yyy / yyy | + 0.98/1.00, - 0.90/0.82: PASS | yes: MuQ 0.137/0.119, CLAP 0.103/0.063 (v2 s0) | DRIFT 0.39 | music,sfx,abstract | v2 vs v1: v2 won |
| 8 | christmas | genre | new_v2 | d | 16 | corpus_v2 | 0,1,2 | 417.84 | 417.84 | yyy / yyy | + 0.88/0.68, - 0.64/0.62: PASS | yes: MuQ 0.036/0.086, CLAP 0.072/0.030 (v2 s0) | DRIFT 0.42 | abstract | nothing (no v1 counterpart) |
| 9 | clave_pattern | rhythm | new_v2 | b | 11,13,12 | corpus_v2* | 0,1,2 | 16.55 | 16.55 | yyy / yyy | + 0.76/0.84, - 0.52/0.82: PASS | yes: MuQ 0.063/0.005, CLAP 0.015/0.004 (v2 s0) | DRIFT 0.34 | sfx | nothing (no v1 counterpart) |
| 10 | cumbia | genre | new_v2 | b | 11,13,12 | corpus_v2* | 0,1,2 | 25.54 | 6.39 | yyy / yyy | + 0.78/0.68, - 0.86/0.42: PASS | yes: MuQ -0.009/0.095, CLAP 0.058/0.040 (v2 s0) | DRIFT 0.36 | sfx,abstract | nothing (no v1 counterpart) |
| 11 | dark | mood | new_v2 | a | 11 | corpus_v2 | 0,1,2 | 26.56 | 26.56 | yyy / yyy | + 0.82/0.80, - 0.74/0.82: PASS | yes: MuQ 0.043/0.037, CLAP 0.055/0.022 (v2 s0) | DRIFT 2.12 | abstract | nothing (no v1 counterpart) |
| 12 | dishes | sound_effect | new_v2 | b | 15,16,13 | corpus_sfx* | 0,1,2 | 22.84 | 22.84 | yyy / yyy | + 1.00/1.00, - 0.66/0.58: PASS | yes: MuQ 0.129/0.053, CLAP 0.073/-0.005 (v2 s0) | ok 0.11 | music,sfx,abstract | nothing (no v1 counterpart) |
| 13 | drum_machine | instrument | new_v2 | a | 18 | corpus_v2 | 0,1,2 | 44.49 | 22.24 | yyy / yyy | + 0.58/0.94, - 0.82/0.74: PASS | yes: MuQ 0.009/0.039, CLAP 0.066/0.011 (v2 s0) | DRIFT 0.74 | music | nothing (no v1 counterpart) |
| 14 | dub_space_echo | space | new_v2 | d | 12 | corpus_v2 | 0,1,2 | 250.32 | 250.32 | yyy / yyy | + 0.74/0.76, - 0.44/0.50: PASS | yes: MuQ 0.071/0.103, CLAP 0.023/0.056 (v2 s0) | ok 0.26 | music,sfx,abstract | nothing (no v1 counterpart) |
| 15 | electric_piano | instrument | new_v2 | d | 16 | corpus_v2 | 0,1,2 | 266.76 | 266.76 | yyy / yyy | + 0.98/0.90, - 0.80/0.58: PASS | yes: MuQ 0.091/0.051, CLAP 0.139/0.031 (v2 s0) | ok 0.15 | music,abstract | nothing (no v1 counterpart) |
| 16 | erhu | instrument | new_v2 | d | 19 | corpus_v2 | 0,1,2 | 588.67 | 294.33 | yyy / yyy | + 0.80/0.74, - 0.60/0.60: PASS | yes: MuQ 0.021/-0.002, CLAP 0.019/0.001 (v2 s0) | DRIFT 0.35 | sfx | nothing (no v1 counterpart) |
| 17 | explosion | sound_effect | new_v2 | b | 12,13,15 | corpus_v2* | 0,1,2 | 30.32 | 30.32 | yyy / yyy | + 1.00/0.90, - 0.60/0.26: PASS | yes: MuQ -0.007/0.018, CLAP 0.071/0.017 (v2 s0) | ok 0.07 | music,sfx | nothing (no v1 counterpart) |
| 18 | fizzy_highs | timbre | new_v2 | a | 22 | corpus_v2 | 0,1,2 | 25.10 | 50.20 | yyy / yyy | + 0.94/0.68, - 0.96/0.52: PASS | yes: MuQ 0.025/0.066, CLAP 0.120/0.025 (v2 s0) | ok 0.14 | music,sfx,abstract | nothing (no v1 counterpart) |
| 19 | frogs | sound_effect | new_v2 | a | 14 | corpus_v2 | 0,1,2 | 128.34 | 64.17 | yyy / yyy | + 0.62/0.96, - 0.86/0.44: PASS | yes: MuQ 0.055/0.050, CLAP 0.038/0.014 (v2 s0) | DRIFT 0.53 | music,sfx,abstract | nothing (no v1 counterpart) |
| 20 | fuzzy | timbre | new_v2 | d | 16 | corpus_v2 | 0,1,2 | 263.20 | 526.40 | yyy / yyy | + 0.84/0.70, - 0.46/0.50: PASS | yes: MuQ 0.062/0.021, CLAP 0.027/0.019 (v2 s0) | ok 0.01 | abstract | nothing (no v1 counterpart) |
| 21 | glitch_sfx | sound_effect | new_v2 | b | 12,13,11 | corpus_sfx* | 0,1,2 | 24.12 | 12.06 | yyy / yyy | + 0.98/0.98, - 0.70/0.56: PASS | yes: MuQ 0.099/0.120, CLAP 0.076/0.051 (v2 s0) | ok 0.04 | music,sfx,abstract | nothing (no v1 counterpart) |
| 22 | hammered_notes | articulation | new_v2 | d | 16 | corpus_v2 | 0,1,2 | 388.40 | 776.80 | yyy / yyy | + 0.80/0.80, - 0.44/0.82: PASS | yes: MuQ 0.101/-0.015, CLAP -0.023/-0.002 (v2 s0) | DRIFT 0.38 | music,sfx,abstract | nothing (no v1 counterpart) |
| 23 | handclaps | instrument | new_v2 | a | 11 | corpus_v2 | 0,1,2 | 79.98 | 39.99 | yyy / yyy | + 0.70/0.60, - 0.76/0.54: PASS | yes: MuQ 0.047/0.034, CLAP 0.015/0.011 (v2 s0) | DRIFT 0.80 | music | nothing (no v1 counterpart) |
| 24 | jet | sound_effect | new_v2 | a | 15 | corpus_v2 | 0,1,2 | 56.27 | 56.27 | yyy / yyy | + 0.98/0.76, - 0.46/0.18: PASS | yes: MuQ -0.015/0.003, CLAP 0.082/0.003 (v2 s0) | ok 0.17 | sfx | v2 vs v1: v2 won |
| 25 | laser_zap | sound_effect | new_v2 | b | 15,13,11 | corpus_v2* | 0,1,2 | 24.62 | 24.62 | yyy / yyy | + 0.96/0.88, - 0.88/0.42: PASS | yes: MuQ 0.043/0.080, CLAP 0.079/0.025 (v2 s0) | ok 0.05 | sfx | nothing (no v1 counterpart) |
| 26 | legato_phrasing | articulation | new_v2 | d | 19 | corpus_v2 | 0,1,2 | 289.42 | 289.42 | yyy / yyy | + 1.00/0.90, - 1.00/0.76: PASS | yes: MuQ 0.120/-0.001, CLAP 0.193/0.013 (v2 s0) | ok 0.13 | music,sfx,abstract | nothing (no v1 counterpart) |
| 27 | locked_groove | rhythm | new_v2 | a | 13 | corpus_v2 | 0,1,2 | 119.62 | 119.62 | yyy / yyy | + 0.78/0.40, - 0.90/0.98: PASS | yes: MuQ 0.009/-0.029, CLAP 0.090/0.007 (v2 s0) | DRIFT 0.42 | sfx | nothing (no v1 counterpart) |
| 28 | magic_sparkle | sound_effect | new_v2 | a | 15 | corpus_sfx | 0,1,2 | 34.22 | 34.22 | yyy / yyy | + 0.92/0.96, - 0.90/0.78: PASS | yes: MuQ 0.129/0.087, CLAP 0.076/0.021 (v2 s0) | ok 0.05 | music,sfx,abstract | nothing (no v1 counterpart) |
| 29 | marcato | articulation | new_v2 | b | 11,9,10 | corpus_v2* | 0,1,2 | 24.85 | 24.85 | yyy / yyy | + 0.88/0.60, - 0.82/0.56: PASS | yes: MuQ 0.050/0.004, CLAP 0.095/0.017 (v2 s0) | ok 0.18 | music,sfx,abstract | nothing (no v1 counterpart) |
| 30 | muted_horn | timbre | new_v2 | b | 11,9,23 | corpus_v2* | 0,1,2 | 13.68 | 13.68 | yyy / yyy | + 0.88/0.70, - 0.84/0.64: PASS | yes: MuQ 0.049/0.046, CLAP 0.066/0.011 (v2 s0) | ok 0.16 | music,sfx | nothing (no v1 counterpart) |
| 31 | new_age | genre | new_v2 | c | 13 | corpus_v2* | 0,1,2 | 57.00 | 28.50 | yyy / yyy | + 0.90/0.92, - 0.26/0.64: PASS | yes: MuQ 0.039/0.015, CLAP 0.012/0.028 (v2 s0) | DRIFT 0.57 | music,sfx,abstract | nothing (no v1 counterpart) |
| 32 | nylon_soft | timbre | new_v2 | a | 11 | corpus_v2 | 0,1,2 | 64.00 | 64.00 | yyy / yyy | + 0.92/0.72, - 0.98/0.72: PASS | yes: MuQ 0.141/0.071, CLAP 0.113/0.043 (v2 s0) | ok 0.02 | sfx | nothing (no v1 counterpart) |
| 33 | palm_muted | articulation | new_v2 | d | 16 | corpus_v2 | 0,1,2 | 259.61 | 259.61 | yyy / yyy | + 0.90/0.76, - 0.92/0.78: PASS | yes: MuQ 0.056/-0.019, CLAP 0.085/-0.006 (v2 s0) | DRIFT 0.37 | music,sfx,abstract | nothing (no v1 counterpart) |
| 34 | plate_reverb | space | new_v2 | d | 16 | corpus_v2 | 0,1,2 | 461.36 | 461.36 | yyy / yyy | + 0.90/0.32, - 0.68/0.82: PASS | yes: MuQ 0.049/0.000, CLAP 0.048/0.015 (v2 s0) | DRIFT 0.39 | sfx | nothing (no v1 counterpart) |
| 35 | polyrhythmic | rhythm | new_v2 | a | 13 | corpus_v2 | 0,1,2 | 71.92 | 71.92 | yyy / yyy | + 0.64/0.82, - 0.64/0.48: PASS | yes: MuQ 0.038/0.025, CLAP 0.003/-0.005 (v2 s0) | DRIFT 0.40 | music,sfx,abstract | v2 vs v1: v2 won |
| 36 | pulsing_synth | rhythm | new_v2 | d | 16 | corpus_v2 | 0,1,2 | 644.94 | 322.47 | yyy / yyy | + 0.84/0.94, - 0.94/0.66: PASS | yes: MuQ 0.082/0.005, CLAP 0.123/0.016 (v2 s0) | DRIFT 1.23 | sfx | nothing (no v1 counterpart) |
| 37 | quantized | rhythm | new_v2 | d | 19 | corpus_v2 | 0,1,2 | 1206.25 | 603.12 | yyy / yyy | + 0.80/0.70, - 0.22/0.26: PASS | yes: MuQ 0.010/0.054, CLAP 0.023/0.017 (v2 s0) | ok 0.28 | sfx | nothing (no v1 counterpart) |
| 38 | rain_on_surface | sound_effect | new_v2 | a | 15 | corpus_v2 | 0,1,2 | 43.09 | 43.09 | yyy / yyy | + 0.96/0.66, - 0.52/0.18: PASS | yes: MuQ -0.025/0.015, CLAP 0.067/0.015 (v2 s0) | ok 0.02 | sfx,abstract | v2 vs v1: v2 won |
| 39 | revving | sound_effect | new_v2 | c | 15 | corpus_sfx* | 0,1,2 | 63.65 | 31.83 | yyy / yyy | + 0.88/0.90, - 0.74/0.52: PASS | yes: MuQ 0.035/0.010, CLAP 0.044/0.003 (v2 s0) | ok 0.30 | sfx | nothing (no v1 counterpart) |
| 40 | sfx_applause | sound_effect | new_v2 | b | 12,13,11 | corpus_sfx* | 0,1,2 | 11.02 | 11.02 | yyy / yyy | + 0.96/0.70, - 0.94/0.78: PASS | yes: MuQ 0.028/0.018, CLAP 0.083/0.033 (v2 s0) | ok 0.07 | music,sfx,abstract | nothing (no v1 counterpart) |
| 41 | sfx_baby_cry | sound_effect | new_v2 | a | 13 | corpus_sfx | 0,1,2 | 28.45 | 56.90 | yyy / yyy | + 0.78/0.76, - 0.86/0.56: PASS | yes: MuQ 0.046/-0.038, CLAP 0.083/0.003 (v2 s0) | ok 0.01 | sfx,abstract | nothing (no v1 counterpart) |
| 42 | sfx_event_rate | rhythm | new_v2 | a | 15 | corpus_v2 | 0,1,2 | 69.66 | 139.32 | yyy / yyy | + 0.78/0.72, - 0.50/0.76: PASS | yes: MuQ 0.042/0.003, CLAP 0.064/0.014 (v2 s0) | DRIFT 1.00 | music | nothing (no v1 counterpart) |
| 43 | sfx_far_away | space | new_v2 | a | 13 | corpus_v2 | 0,1,2 | 68.55 | 68.55 | yyy / yyy | + 0.84/0.82, - 0.86/0.38: PASS | yes: MuQ 0.025/0.005, CLAP 0.062/0.016 (v2 s0) | DRIFT 0.89 | sfx,abstract | nothing (no v1 counterpart) |
| 44 | sfx_indoor_hall | space | new_v2 | d | 15 | corpus_v2 | 0,1,2 | 537.21 | 537.21 | yyy / yyy | + 0.82/0.68, - 0.48/0.28: PASS | yes: MuQ 0.018/0.030, CLAP 0.063/0.007 (v2 s0) | DRIFT 0.56 | music,sfx | nothing (no v1 counterpart) |
| 45 | sfx_insects | sound_effect | new_v2 | a | 13 | corpus_sfx | 0,1,2 | 61.65 | 61.65 | yyy / yyy | + 1.00/0.98, - 0.40/0.42: PASS | yes: MuQ 0.050/0.075, CLAP 0.096/-0.001 (v2 s0) | ok 0.12 | music,sfx,abstract | nothing (no v1 counterpart) |
| 46 | sfx_ocean_waves | sound_effect | new_v2 | c | 13 | corpus_sfx* | 0,1,2 | 29.62 | 29.62 | yyy / yyy | + 0.86/0.84, - 1.00/0.94: PASS | yes: MuQ 0.094/0.067, CLAP 0.089/0.028 (v2 s0) | ok 0.06 | music,sfx,abstract | nothing (no v1 counterpart) |
| 47 | sfx_sub_weight | space | new_v2 | c | 15 | corpus_v2* | 0,1,2 | 68.76 | 68.76 | yyy / yyy | + 1.00/0.82, - 0.84/0.44: PASS | yes: MuQ 0.065/0.073, CLAP 0.161/0.044 (v2 s0) | ok 0.06 | music,sfx | nothing (no v1 counterpart) |
| 48 | sfx_telephone | sound_effect | new_v2 | d | 23 | corpus_sfx | 0,1,2 | 212.24 | 212.24 | yyy / yyy | + 0.64/0.70, - 0.44/0.26: PASS | yes: MuQ -0.012/0.012, CLAP -0.016/0.015 (v2 s0) | DRIFT 0.63 | sfx | nothing (no v1 counterpart) |
| 49 | shaker_pulse | rhythm | new_v2 | d | 12 | corpus_v2 | 0,1,2 | 325.65 | 325.65 | yyy / yyy | + 0.62/0.86, - 0.84/0.80: PASS | yes: MuQ 0.117/-0.005, CLAP 0.086/0.006 (v2 s0) | DRIFT 0.70 | music,abstract | nothing (no v1 counterpart) |
| 50 | shimmer_reverb | space | new_v2 | c | 12 | corpus_v2* | 0,1,2 | 85.16 | 85.16 | yyy / yyy | + 0.74/0.94, - 0.04/0.64: PASS | yes: MuQ 0.098/0.050, CLAP -0.016/0.017 (v2 s0) | ok 0.29 | music,sfx,abstract | nothing (no v1 counterpart) |
| 51 | shimmering | timbre | new_v2 | d | 16 | corpus_v2 | 0,1,2 | 327.38 | 327.38 | yyy / yyy | + 0.78/0.94, - 0.60/0.72: PASS | yes: MuQ 0.069/0.017, CLAP 0.030/0.038 (v2 s0) | ok 0.18 | sfx | nothing (no v1 counterpart) |
| 52 | sitar | instrument | new_v2 | d | 19 | corpus_v2 | 0,1,2 | 633.48 | 316.74 | yyy / yyy | + 0.86/0.86, - 0.98/0.74: PASS | yes: MuQ 0.075/0.050, CLAP 0.091/0.028 (v2 s0) | ok 0.21 | music,sfx,abstract | nothing (no v1 counterpart) |
| 53 | smash | sound_effect | new_v2 | b | 15,13,12 | corpus_v2* | 0,1,2 | 39.71 | 39.71 | yyy / yyy | + 0.66/0.62, - 0.82/0.24: PASS | yes: MuQ -0.033/-0.023, CLAP 0.053/0.019 (v2 s0) | ok 0.29 | sfx | nothing (no v1 counterpart) |
| 54 | solo_build | dynamics | new_v2 | c | 12 | corpus_v2* | 0,1,2 | 435.36 | 217.68 | yyy / yyy | + 0.92/0.22, - 0.86/0.82: PASS | yes: MuQ 0.032/0.102, CLAP 0.071/0.054 (v2 s0) | DRIFT 4.05 | music,sfx,abstract | nothing (no v1 counterpart) |
| 55 | solo_crest | dynamics | new_v2 | d | 23 | corpus_v2 | 0,1,2 | 92.52 | 46.26 | yyy / yyy | + 0.70/0.64, - 0.46/0.48: PASS | yes: MuQ 0.014/0.024, CLAP 0.038/-0.002 (v2 s0) | DRIFT 1.95 | music | nothing (no v1 counterpart) |
| 56 | spiccato | articulation | new_v2 | d | 16 | corpus_v2 | 0,1,2 | 393.20 | 393.20 | yyy / yyy | + 0.76/0.84, - 0.96/0.52: PASS | yes: MuQ 0.087/0.015, CLAP 0.080/0.010 (v2 s0) | DRIFT 0.40 | music,sfx,abstract | nothing (no v1 counterpart) |
| 57 | swing | rhythm | new_v2 | b | 11,12,13 | corpus_v2* | 0,1,2 | 16.98 | 16.98 | yyy / yyy | + 0.66/0.72, - 0.74/0.66: PASS | yes: MuQ 0.051/-0.012, CLAP 0.028/0.013 (v2 s0) | ok 0.07 | sfx | nothing (no v1 counterpart) |
| 58 | thick_unison | timbre | new_v2 | b | 13,11,15 | corpus_v2* | 0,1,2 | 37.15 | 37.15 | yyy / yyy | + 0.72/0.70, - 0.50/0.42: PASS | yes: MuQ 0.003/0.013, CLAP 0.044/0.027 (v2 s0) | DRIFT 0.52 | sfx | nothing (no v1 counterpart) |
| 59 | tom_patterns | rhythm | new_v2 | d | 16 | corpus_v2 | 0,1,2 | 286.20 | 286.20 | yyy / yyy | + 0.88/0.72, - 0.78/0.58: PASS | yes: MuQ 0.051/0.056, CLAP 0.084/0.071 (v2 s0) | DRIFT 0.32 | sfx | nothing (no v1 counterpart) |
| 60 | ukulele | instrument | new_v2 | a | 11 | corpus_v2 | 0,1,2 | 59.23 | 29.61 | yyy / yyy | + 0.96/0.90, - 0.78/0.58: PASS | yes: MuQ 0.093/0.103, CLAP 0.089/0.051 (v2 s0) | DRIFT 0.38 | music,sfx,abstract | v2 vs v1: v2 won |
| 61 | vaporwave | genre | new_v2 | a | 16 | corpus_v2 | 0,1,2 | 46.65 | 23.32 | yyy / yyy | + 0.64/0.38, - 0.64/0.70: PASS | yes: MuQ 0.003/-0.019, CLAP 0.041/0.015 (v2 s0) | DRIFT 0.75 | music | nothing (no v1 counterpart) |
| 62 | virtuosic_runs | articulation | new_v2 | d | 13 | corpus_v2 | 0,1,2 | 290.25 | 290.25 | yyy / yyy | + 0.86/0.88, - 0.98/0.54: PASS | yes: MuQ 0.061/0.081, CLAP 0.098/0.039 (v2 s0) | DRIFT 0.37 | music,sfx,abstract | nothing (no v1 counterpart) |
| 63 | woody_body | timbre | new_v2 | d | 18 | corpus_v2 | 0,1,2 | 303.23 | 303.23 | yyy / yyy | + 0.92/0.74, - 0.88/0.64: PASS | yes: MuQ 0.053/0.005, CLAP 0.092/0.007 (v2 s0) | DRIFT 0.58 | music,sfx,abstract | nothing (no v1 counterpart) |
| 64 | abs_cinematic | abstract | kept_v1 | a | 13 | corpus | 0 | 46.23 | 23.12 | y / y | + 0.94/0.80, - 0.72/0.46: PASS (v1 phase 6) | yes: MuQ 0.030/0.016, CLAP 0.089/0.013 (v1 run) | not measured | music (catalogue default) | nothing (v2 did not re-find it; v1 kept by default) |
| 65 | abs_industrial | abstract | kept_v1 | a | 11 | corpus | 0 | 66.15 | 33.08 | y / y | + 0.94/0.84, - 0.60/0.62: PASS (v1 phase 6) | yes: MuQ 0.061/0.033, CLAP 0.067/0.025 (v1 run) | not measured | music (catalogue default) | nothing (v2 did not re-find it; v1 kept by default) |
| 66 | acid_303 | instrument | kept_v1 | a | 11 | corpus | 0 | 55.56 | 55.56 | n / y | + 0.84/0.82, - 0.94/0.56: PASS (v1 phase 6) | yes: MuQ 0.074/0.091, CLAP 0.115/0.036 (v1 run) | not measured | music (catalogue default) | nothing (v2 did not re-find it; v1 kept by default) |
| 67 | alarm_clock | sound_effect | kept_v1 | a | 15 | corpus | 0 | 72.28 | 72.28 | n / y | + 0.88/0.90, - 0.92/0.70: PASS (v1 phase 6) | yes: MuQ 0.082/0.043, CLAP 0.085/0.023 (v1 run) | ok 0.11* | music,sfx,abstract * | v2 vs v1: v1 kept (+ LPAPS/cutoff 1.23 > 1.1 x v1 0.88; - clap 0.50 < v1 0.92 - 0.05; - muq 0.48 < v1 0.70 - 0.05) |
| 68 | analog | production | kept_v1 | a | 15 | corpus | 0 | 49.41 | 49.41 | y / n | + 0.70/0.60, - 0.84/0.44: PASS (v1 phase 6) | yes: MuQ 0.010/-0.016, CLAP 0.093/0.016 (v1 run) | DRIFT 1.09* | sfx * | v2 vs v1: v1 kept (+ clap 0.62 < v1 0.70 - 0.05; - clap 0.66 < v1 0.84 - 0.05; - muq 0.36 < v1 0.44 - 0.05) |
| 69 | childrens | genre | kept_v1 | a | 13 | corpus | 0 | 45.78 | 22.89 | y / y | + 0.72/0.72, - 0.90/0.58: PASS (v1 phase 6) | yes: MuQ 0.028/0.041, CLAP 0.057/0.024 (v1 run) | not measured | music (catalogue default) | nothing (v2 did not re-find it; v1 kept by default) |
| 70 | close_mic | space | kept_v1 | a | 11 | corpus | 0 | 59.21 | 59.21 | y / n | + 0.36/0.74, - 0.26/0.68: FAIL (v1 phase 6) | yes: MuQ 0.042/-0.009, CLAP -0.027/-0.004 (v1 run) | DRIFT 1.41* | music,sfx * | v2 vs v1: v1 kept (+ clap 0.14 < v1 0.36 - 0.05; + muq 0.38 < v1 0.74 - 0.05; - clap 0.12 < v1 0.26 - 0.05; - muq 0.44 < v1 0.68 - 0.05) |
| 71 | dissonant | timbre | kept_v1 | a | 13 | corpus | 0 | 53.03 | 53.03 | y / y | + 0.86/0.68, - 0.92/0.52: PASS (v1 phase 6) | yes: MuQ 0.033/0.003, CLAP 0.061/0.016 (v1 run) | ok 0.25* | music,sfx,abstract * | v2 vs v1: v1 kept (+ clap 0.66 < v1 0.86 - 0.05; - clap 0.78 < v1 0.92 - 0.05) |
| 72 | distant | space | kept_v1 | a | 22 | corpus | 0 | 57.50 | 28.75 | y / y | + 0.80/0.84, - 0.54/0.44: PASS (v1 phase 6) | yes: MuQ 0.027/0.010, CLAP 0.018/0.018 (v1 run) | not measured | music (catalogue default) | nothing (v2 did not re-find it; v1 kept by default) |
| 73 | dreamy | mood | kept_v1 | a | 13 | corpus | 0 | 40.93 | 40.93 | y / y | + 0.98/0.68, - 0.98/0.56: PASS (v1 phase 6) | yes: MuQ 0.016/0.007, CLAP 0.103/0.025 (v1 run) | ok 0.28* | music,sfx,abstract * | v2 vs v1: v1 kept (+ muq 0.42 < v1 0.68 - 0.05) |
| 74 | drone | genre | kept_v1 | a | 13 | corpus | 0 | 46.46 | 46.46 | y / y | + 0.92/0.94, - 0.70/0.70: PASS (v1 phase 6) | yes: MuQ 0.051/0.011, CLAP 0.078/0.022 (v1 run) | DRIFT 0.38* | abstract * | v2 vs v1: v1 kept (+ muq 0.76 < v1 0.94 - 0.05; - clap 0.56 < v1 0.70 - 0.05; - muq 0.60 < v1 0.70 - 0.05) |
| 75 | future_bass | genre | kept_v1 | a | 13 | corpus | 0 | 47.86 | 47.86 | y / y | + 0.92/0.72, - 0.86/0.64: PASS (v1 phase 6) | yes: MuQ 0.069/0.035, CLAP 0.100/0.045 (v1 run) | DRIFT 0.76* | music,sfx,abstract * | v2 vs v1: v1 kept (+ clap 0.84 < v1 0.92 - 0.05; + LPAPS/cutoff 1.12 > 1.1 x v1 1.01; - LPAPS/cutoff 1.21 > 1.1 x v1 1.08) |
| 76 | glitchy | production | kept_v1 | a | 19 | corpus | 0 | 41.97 | 41.97 | y / y | + 1.00/0.98, - 0.82/0.42: PASS (v1 phase 6) | yes: MuQ 0.158/0.065, CLAP 0.165/0.040 (v1 run) | ok 0.11* | music,sfx,abstract * | v2 vs v1: v1 kept (+ muq 0.90 < v1 0.98 - 0.05; + LPAPS/cutoff 1.33 > 1.1 x v1 1.20; - muq 0.22 < v1 0.42 - 0.05) |
| 77 | gritty | timbre | kept_v1 | a | 11 | corpus | 0 | 57.08 | 57.08 | y / y | + 0.96/0.96, - 0.88/0.90: PASS (v1 phase 6) | yes: MuQ 0.150/0.082, CLAP 0.087/0.023 (v1 run) | not measured | music (catalogue default) | nothing (v2 did not re-find it; v1 kept by default) |
| 78 | hardstyle | genre | kept_v1 | a | 13 | corpus | 0 | 45.89 | 45.89 | y / y | + 0.82/0.76, - 0.98/0.84: PASS (v1 phase 6) | yes: MuQ 0.120/0.037, CLAP 0.094/0.022 (v1 run) | ok 0.25* | sfx,abstract * | v2 vs v1: v1 kept (+ muq 0.66 < v1 0.76 - 0.05; + LPAPS/cutoff 1.32 > 1.1 x v1 1.12) |
| 79 | horror_score | genre | kept_v1 | a | 11 | corpus | 0 | 48.32 | 48.32 | y / y | + 1.00/0.88, - 0.50/0.56: PASS (v1 phase 6) | yes: MuQ 0.052/0.061, CLAP 0.061/0.020 (v1 run) | not measured | music (catalogue default) | nothing (v2 did not re-find it; v1 kept by default) |
| 80 | impact_boom | sound_effect | kept_v1 | a | 15 | corpus | 0 | 57.37 | 57.37 | n / y | + 0.82/0.94, - 0.60/0.36: PASS (v1 phase 6) | yes: MuQ 0.066/0.078, CLAP 0.048/0.031 (v1 run) | DRIFT 1.00* | sfx * | v2 vs v1: v1 kept (+ clap 0.40 < v1 0.82 - 0.05; + muq 0.62 < v1 0.94 - 0.05; + LPAPS/cutoff 1.33 > 1.1 x v1 0.84; - clap 0.34 < v1 0.60 - 0.05; - muq 0.30 < v1 0.36 - 0.05) |
| 81 | intense | dynamics | kept_v1 | a | 10 | corpus | 0 | 29.94 | 59.89 | y / y | + 0.94/0.88, - 0.96/0.92: PASS (v1 phase 6) | yes: MuQ 0.106/0.053, CLAP 0.086/0.020 (v1 run) | ok 0.09* | music,sfx,abstract * | v2 vs v1: v1 kept (+ LPAPS/cutoff 1.38 > 1.1 x v1 1.07) |
| 82 | jingle_bells | instrument | kept_v1 | a | 9 | corpus | 0 | 67.67 | 33.83 | y / y | + 0.98/0.98, - 0.90/0.52: PASS (v1 phase 6) | yes: MuQ 0.076/0.161, CLAP 0.069/0.048 (v1 run) | ok 0.19* | music,sfx,abstract * | v2 vs v1: v1 kept (- clap 0.82 < v1 0.90 - 0.05) |
| 83 | koto | instrument | kept_v1 | a | 13 | corpus | 0 | 53.01 | 53.01 | y / y | + 0.46/0.42, - 0.96/0.62: PASS (v1 phase 6) | yes: MuQ 0.028/0.006, CLAP 0.018/0.008 (v1 run) | not measured | music (catalogue default) | nothing (v2 did not re-find it; v1 kept by default) |
| 84 | live_recording | production | kept_v1 | a | 13 | corpus | 0 | 68.88 | 68.88 | y / y | + 0.26/0.34, - 0.54/0.26: FAIL (v1 phase 6) | yes: MuQ -0.028/0.018, CLAP -0.016/0.006 (v1 run) | ok 0.10* | sfx * | v2 vs v1: v1 kept (- clap 0.40 < v1 0.54 - 0.05) |
| 85 | lofi_hip_hop | genre | kept_v1 | a | 11 | corpus | 0 | 65.78 | 32.89 | n / y | + 0.62/0.84, - 0.62/0.70: PASS (v1 phase 6) | yes: MuQ 0.084/0.151, CLAP 0.016/0.057 (v1 run) | ok 0.12* | music,abstract * | v2 vs v1: v1 kept (+ clap 0.48 < v1 0.62 - 0.05; + LPAPS/cutoff 1.39 > 1.1 x v1 0.89; - clap 0.44 < v1 0.62 - 0.05; - muq 0.64 < v1 0.70 - 0.05) |
| 86 | mains_hum | production | kept_v1 | a | 22 | corpus | 0 | 112.51 | 112.51 | y / y | + 0.90/0.84, - 0.88/0.38: PASS (v1 phase 6) | yes: MuQ 0.033/0.006, CLAP 0.098/0.000 (v1 run) | ok 0.16* | sfx,abstract * | v2 vs v1: v1 kept (- clap 0.56 < v1 0.88 - 0.05) |
| 87 | phone_ring | sound_effect | kept_v1 | a | 15 | corpus | 0 | 67.72 | 67.72 | n / y | + 0.98/0.76, - 0.90/0.32: PASS (v1 phase 6) | yes: MuQ 0.040/0.069, CLAP 0.121/-0.005 (v1 run) | ok 0.05* | sfx * | v2 vs v1: v1 kept (+ LPAPS/cutoff 1.20 > 1.1 x v1 0.93; - clap 0.76 < v1 0.90 - 0.05) |
| 88 | playful | mood | kept_v1 | a | 11 | corpus | 0 | 51.27 | 25.64 | y / y | + 0.60/0.70, - 0.90/0.64: PASS (v1 phase 6) | yes: MuQ 0.007/0.007, CLAP 0.038/-0.003 (v1 run) | not measured | music (catalogue default) | nothing (v2 did not re-find it; v1 kept by default) |
| 89 | post_rock | genre | kept_v1 | a | 11 | corpus | 0 | 55.68 | 27.84 | y / y | + 1.00/0.94, - 0.84/0.84: PASS (v1 phase 6) | yes: MuQ 0.175/0.095, CLAP 0.102/0.038 (v1 run) | DRIFT 0.40* | music,abstract * | v2 vs v1: v1 kept (+ clap 0.82 < v1 1.00 - 0.05; + LPAPS/cutoff 1.34 > 1.1 x v1 1.03; - LPAPS/cutoff 1.28 > 1.1 x v1 1.08) |
| 90 | radio_filtered | production | kept_v1 | a | 16 | corpus | 0 | 68.80 | 68.80 | y / y | + 0.96/0.80, - 0.68/0.76: PASS (v1 phase 6) | yes: MuQ 0.053/0.037, CLAP 0.110/-0.012 (v1 run) | ok 0.21* | music * | v2 vs v1: v1 kept (- clap 0.58 < v1 0.68 - 0.05) |
| 91 | riser | sound_effect | kept_v1 | a | 15 | corpus | 0 | 58.86 | 58.86 | n / y | + 0.66/0.76, - 0.56/0.36: PASS (v1 phase 6) | yes: MuQ 0.035/0.063, CLAP 0.040/0.038 (v1 run) | DRIFT 0.59* | sfx * | v2 vs v1: v1 kept (+ muq 0.48 < v1 0.76 - 0.05; + LPAPS/cutoff 1.30 > 1.1 x v1 0.91; - clap 0.44 < v1 0.56 - 0.05) |
| 92 | samba | genre | kept_v1 | a | 11 | corpus | 0 | 52.41 | 26.20 | y / y | + 0.94/0.84, - 0.86/0.48: PASS (v1 phase 6) | yes: MuQ 0.041/0.088, CLAP 0.086/0.038 (v1 run) | not measured | music (catalogue default) | nothing (v2 did not re-find it; v1 kept by default) |
| 93 | sensual | mood | kept_v1 | a | 13 | corpus | 0 | 48.36 | 24.18 | y / y | + 0.56/0.80, - 0.52/0.72: FAIL (v1 phase 6) | yes: MuQ 0.073/0.037, CLAP 0.012/0.010 (v1 run) | not measured | music (catalogue default) | nothing (v2 did not re-find it; v1 kept by default) |
| 94 | solo_instrument | production | kept_v1 | a | 12 | corpus | 0 | 57.19 | 57.19 | y / y | + 0.82/0.86, - 0.72/0.58: PASS (v1 phase 6) | yes: MuQ 0.039/0.024, CLAP 0.030/0.006 (v1 run) | ok 0.29* | music,sfx,abstract * | v2 vs v1: v1 kept (+ clap 0.76 < v1 0.82 - 0.05; - muq 0.48 < v1 0.58 - 0.05) |
| 95 | strummed_guitar | instrument | kept_v1 | a | 13 | corpus | 0 | 43.35 | 43.35 | y / y | + 0.90/0.94, - 0.88/0.28: PASS (v1 phase 6) | yes: MuQ 0.036/0.033, CLAP 0.098/0.020 (v1 run) | ok 0.16* | abstract * | v2 vs v1: v1 kept (+ LPAPS/cutoff 1.09 > 1.1 x v1 1.00; - clap 0.80 < v1 0.88 - 0.05; - LPAPS/cutoff 1.44 > 1.1 x v1 1.29) |
| 96 | surf_rock | genre | kept_v1 | a | 13 | corpus | 0 | 47.54 | 23.77 | y / y | + 0.72/0.82, - 0.80/0.66: PASS (v1 phase 6) | yes: MuQ 0.047/0.085, CLAP 0.045/0.028 (v1 run) | DRIFT 0.39* | sfx,abstract * | v2 vs v1: v1 kept (+ muq 0.74 < v1 0.82 - 0.05; - muq 0.58 < v1 0.66 - 0.05) |
| 97 | synth_pad | instrument | kept_v1 | a | 15 | corpus | 0 | 41.16 | 41.16 | n / y | + 1.00/0.94, - 0.80/0.60: PASS (v1 phase 6) | yes: MuQ 0.074/0.047, CLAP 0.116/0.055 (v1 run) | ok 0.16* | music,sfx,abstract * | v2 vs v1: v1 kept (+ clap 0.94 < v1 1.00 - 0.05; - clap 0.64 < v1 0.80 - 0.05) |
| 98 | synthesizer | instrument | kept_v1 | a | 16 | corpus | 0 | 47.81 | 47.81 | n / y | + 0.64/0.74, - 0.96/0.66: PASS (v1 phase 6) | yes: MuQ 0.039/0.020, CLAP 0.080/0.023 (v1 run) | ok 0.13* | music * | v2 vs v1: v1 kept (+ clap 0.50 < v1 0.64 - 0.05; + LPAPS/cutoff 1.22 > 1.1 x v1 0.97) |
| 99 | tango | genre | kept_v1 | a | 13 | corpus | 0 | 40.80 | 40.80 | n / y | + 0.66/0.70, - 0.80/0.54: PASS (v1 phase 6) | yes: MuQ 0.046/0.091, CLAP 0.030/0.012 (v1 run) | ok 0.29* | music * | v2 vs v1: v1 kept (+ clap 0.10 < v1 0.66 - 0.05; + muq 0.60 < v1 0.70 - 0.05; + LPAPS/cutoff 1.34 > 1.1 x v1 0.90; - clap 0.66 < v1 0.80 - 0.05) |
| 100 | timpani | instrument | kept_v1 | a | 15 | corpus | 0 | 47.80 | 47.80 | y / y | + 0.86/0.94, - 0.78/0.62: PASS (v1 phase 6) | yes: MuQ 0.069/0.027, CLAP 0.084/0.028 (v1 run) | DRIFT 1.68* | music,abstract * | v2 vs v1: v1 kept (+ clap 0.24 < v1 0.86 - 0.05; + muq 0.78 < v1 0.94 - 0.05; + LPAPS/cutoff 1.19 > 1.1 x v1 1.00; - clap 0.60 < v1 0.78 - 0.05; - LPAPS/cutoff 1.39 > 1.1 x v1 1.19) |
| 101 | ui_click | sound_effect | kept_v1 | a | 12 | corpus | 0 | 63.54 | 63.54 | n / y | + 0.72/0.82, - 0.84/0.80: PASS (v1 phase 6) | yes: MuQ 0.070/0.037, CLAP 0.038/0.014 (v1 run) | DRIFT 0.31* | music,sfx,abstract * | v2 vs v1: v1 kept (+ LPAPS/cutoff 1.28 > 1.1 x v1 0.98) |
| 102 | wide_stereo | space | kept_v1 | a | 12 | corpus | 0 | 27.84 | 55.68 | y / y | + 0.60/0.40, - 0.28/0.46: FAIL (v1 phase 6) | yes: MuQ -0.027/0.016, CLAP 0.003/-0.017 (v1 run) | DRIFT 0.82* | sfx * | v2 vs v1: v1 kept (- muq 0.36 < v1 0.46 - 0.05) |
| 103 | bright | timbre | legacy | legacy | 23 | discover.py, 32 pairs | 0,1,2 | 8.62 | 8.62 | yyy / yyy | + 0.96/0.58, - 1.00/0.90: PASS | yes: MuQ 0.032/0.034, CLAP 0.112/0.028 (v2 s0); MuQ 0.020/0.018, CLAP 0.141/0.055 (10-05 eval) | not measured | music (default, not measured) | nothing (no v2 knob of that name; legacy kept by default) |
| 104 | dense_arrangement | rhythm | legacy | legacy | 1 | discover.py, 32 pairs | 0,1,2 | 24.81 | 24.81 | yyy / yyy | + 1.00/0.96, - 0.92/0.80: PASS | yes: MuQ 0.069/0.052, CLAP 0.110/0.024 (v2 s0); MuQ 0.129/0.117, CLAP 0.131/0.075 (10-05 eval) | not measured | music (default, not measured) | nothing (no v2 knob of that name; legacy kept by default) |
| 105 | percussive | timbre | legacy | legacy | 15 | discover.py, 32 pairs | 0,1,2 | 31.88 | 15.94 | yyy / yyy | + 1.00/0.98, - 0.96/0.80: PASS | yes: MuQ 0.055/0.010, CLAP 0.161/0.021 (v2 s0); MuQ 0.085/0.064, CLAP 0.067/0.004 (10-05 eval) | not measured | music (default, not measured) | nothing (no v2 knob of that name; legacy kept by default) |
| 106 | rough | timbre | legacy | legacy | 2 | discover.py, 32 pairs | 0,1,2 | 20.34 | 20.34 | yyy / yyy | + 0.44/0.80, - 0.64/0.72: PASS | yes: MuQ 0.042/0.071, CLAP 0.002/0.027 (v2 s0); MuQ 0.070/0.123, CLAP 0.017/0.026 (10-05 eval) | not measured | music (default, not measured) | nothing (no v2 knob of that name; legacy kept by default) |
| 107 | warm | timbre | legacy | legacy | 15 | discover.py, 32 pairs | 0,1 | 21.68 | 21.68 | yy / yy | + 0.26/0.36, - 0.68/0.30: FAIL | yes: MuQ -0.029/-0.006, CLAP 0.016/0.012 (v2 s0); MuQ 0.060/0.001, CLAP 0.118/0.015 (10-05 eval) | not measured | music (default, not measured) | nothing (no v2 knob of that name; legacy kept by default) |

Per-origin summary of the table:
- Cutoff reached: new_v2 63/63 on both signs on every seed (claim verified). kept_v1: 27 both signs, 12 knob-signs never reached
  (acid_303 +, alarm_clock +, analog -, close_mic -, impact_boom +, lofi_hip_hop +, phone_ring +, riser +, synth_pad +,
  synthesizer +, tango +, ui_click +): those gains are the v1 probe ceiling (alpha 64), a lower bound. legacy: all reached.
- Quality bar: new_v2 63/63 pass (by construction). kept_v1 35/39 pass on their own v1 numbers; close_mic, live_recording,
  sensual and wide_stereo fail it and ship. legacy 4/5; warm fails (+ 0.26/0.36, - 0.68/0.30) and ships.
- Seed spread: on 37 of the 126 new_v2 knob-signs the three seeds' gains differ (min != max), usually by a factor of 2 (one
  probe step).

## 2. Candidates that did not ship (44)

All 44 have the same recorded reason. 43 are from the main corpus, 1 (fan) from the SFX supplement. 34 ran 2 seeds, 10 ran 3
(tie-break). 32 of the 44 would pass on CLAP alone; 11 miss by one scorer at 0.52 to 0.58 (amateur, bass_808, boxy_tone,
brassy_blare, crowd, driving_eighths, modular_synth_r2, sfx_level, wall_of_sound, wide_vibrato, wind). Four are sound effects
(crowd, fan, glass_clink, wind) judged partly on MuQ, which does not see SFX.

| # | candidate | category | set | variant | seeds run | CLAP/MuQ intended-sign frac per sign (s0) | hold | shared with v1? | reason not shipped |
|---|---|---|---|---|---|---|---|---|---|
| 1 | amateur | production | main | d | 0,1 | + 1.00/0.52, - 0.88/0.58 | ok | no | no sign with CLAP and MuQ intended-sign fraction >= 0.6 |
| 2 | bass_808 | instrument | main | d | 0,1 | + 0.52/0.82, - 0.56/0.72 | ok | no | no sign with CLAP and MuQ intended-sign fraction >= 0.6 |
| 3 | bluegrass | genre | main | d | 0,1 | + 0.78/0.18, - 0.86/0.52 | DRIFT | no | no sign with CLAP and MuQ intended-sign fraction >= 0.6 |
| 4 | boomy | timbre | main | d | 0,1 | + 0.36/0.96, - 0.82/0.36 | ok | no | no sign with CLAP and MuQ intended-sign fraction >= 0.6 |
| 5 | boxy_tone | timbre | main | b | 0,1 | + 0.78/0.40, - 0.66/0.56 | ok | no | no sign with CLAP and MuQ intended-sign fraction >= 0.6 |
| 6 | brassy_blare | timbre | main | b | 0,1 | + 0.46/0.54, - 0.78/0.56 | ok | no | no sign with CLAP and MuQ intended-sign fraction >= 0.6 |
| 7 | breakbeat_feel | rhythm | main | d | 0,1 | + 0.48/0.68, - 0.18/0.02 | ok | no | no sign with CLAP and MuQ intended-sign fraction >= 0.6 |
| 8 | chord_stabs | articulation | main | a | 0,1 | + 0.52/0.68, - 0.64/0.50 | ok | no | no sign with CLAP and MuQ intended-sign fraction >= 0.6 |
| 9 | crowd | sound_effect | main | c | 0,1 | + 0.64/0.58, - 0.68/0.46 | ok | no | no sign with CLAP and MuQ intended-sign fraction >= 0.6 |
| 10 | damped_strings | articulation | main | d | 0,1 | + 0.48/0.76, - 0.50/0.82 | DRIFT | no | no sign with CLAP and MuQ intended-sign fraction >= 0.6 |
| 11 | deep_r2 | timbre | main | d | 0,1,2 | + 0.50/0.68, - 0.58/0.34 | DRIFT | no | no sign with CLAP and MuQ intended-sign fraction >= 0.6 |
| 12 | djembe | instrument | main | d | 0,1 | + 0.56/0.54, - 0.88/0.50 | DRIFT | no | no sign with CLAP and MuQ intended-sign fraction >= 0.6 |
| 13 | dotted_rhythm | rhythm | main | d | 0,1 | + 0.82/0.38, - 0.86/0.46 | DRIFT | no | no sign with CLAP and MuQ intended-sign fraction >= 0.6 |
| 14 | driving_eighths | rhythm | main | c | 0,1 | + 0.68/0.58, - 0.30/0.76 | DRIFT | no | no sign with CLAP and MuQ intended-sign fraction >= 0.6 |
| 15 | echo_delay | space | main | a | 0,1,2 | + 0.54/0.82, - 0.58/0.58 | DRIFT | no | no sign with CLAP and MuQ intended-sign fraction >= 0.6 |
| 16 | enjoyment | production | main | b | 0,1,2 | + 0.76/0.28, - 0.98/0.34 | ok | no | no sign with CLAP and MuQ intended-sign fraction >= 0.6 |
| 17 | fan | sound_effect | sfx | a | 0,1 | + 0.94/0.54, - 0.08/0.16 | ok | no | no sign with CLAP and MuQ intended-sign fraction >= 0.6 |
| 18 | funny | mood | main | b | 0,1,2 | + 0.86/0.54, - 0.74/0.50 | ok | no | no sign with CLAP and MuQ intended-sign fraction >= 0.6 |
| 19 | glass_clink | sound_effect | main | a | 0,1,2 | + 0.88/0.54, - 0.64/0.14 | ok | no | no sign with CLAP and MuQ intended-sign fraction >= 0.6 |
| 20 | halftime_r2 | rhythm | main | d | 0,1 | + 0.22/0.08, - 0.72/0.34 | ok | no | no sign with CLAP and MuQ intended-sign fraction >= 0.6 |
| 21 | hihat_rolls | rhythm | main | d | 0,1 | + 0.56/0.56, - 0.54/0.80 | DRIFT | no | no sign with CLAP and MuQ intended-sign fraction >= 0.6 |
| 22 | honky_mids | timbre | main | a | 0,1 | + 0.64/0.32, - 0.40/0.34 | ok | no | no sign with CLAP and MuQ intended-sign fraction >= 0.6 |
| 23 | laid_back | rhythm | main | c | 0,1,2 | + 0.36/0.66, - 0.40/0.58 | ok | no | no sign with CLAP and MuQ intended-sign fraction >= 0.6 |
| 24 | lush_r2 | timbre | main | d | 0,1 | + 0.10/0.12, - 0.14/0.70 | ok | no | no sign with CLAP and MuQ intended-sign fraction >= 0.6 |
| 25 | majestic | mood | main | b | 0,1 | + 0.32/0.40, - 0.34/0.50 | DRIFT | no | no sign with CLAP and MuQ intended-sign fraction >= 0.6 |
| 26 | modular_synth_r2 | instrument | main | b | 0,1 | + 0.68/0.54, - 0.86/0.56 | DRIFT | no | no sign with CLAP and MuQ intended-sign fraction >= 0.6 |
| 27 | no_percussion | rhythm | main | d | 0,1,2 | + 0.44/0.64, - 0.36/0.52 | DRIFT | no | no sign with CLAP and MuQ intended-sign fraction >= 0.6 |
| 28 | noise_floor | production | main | b | 0,1 | + 0.44/0.70, - 0.32/0.36 | ok | no | no sign with CLAP and MuQ intended-sign fraction >= 0.6 |
| 29 | odd_meter | rhythm | main | b | 0,1 | + 0.86/0.10, - 0.94/0.26 | DRIFT | no | no sign with CLAP and MuQ intended-sign fraction >= 0.6 |
| 30 | ostinato_r2 | rhythm | main | c | 0,1 | + 0.52/0.80, - 0.78/0.38 | ok | no | no sign with CLAP and MuQ intended-sign fraction >= 0.6 |
| 31 | pitch_bends | articulation | main | b | 0,1 | + 0.94/0.46, - 0.26/0.36 | DRIFT | no | no sign with CLAP and MuQ intended-sign fraction >= 0.6 |
| 32 | sfx_close_foley | space | main | d | 0,1 | + 0.68/0.46, - 0.38/0.08 | DRIFT | no | no sign with CLAP and MuQ intended-sign fraction >= 0.6 |
| 33 | sfx_laden | production | main | d | 0,1 | + 0.44/0.70, - 0.46/0.60 | DRIFT | yes: gate picked v2, v1 pack also dropped | no sign with CLAP and MuQ intended-sign fraction >= 0.6 |
| 34 | sfx_level | dynamics | main | d | 0,1,2 | + 0.96/0.18, - 0.84/0.58 | ok | no | no sign with CLAP and MuQ intended-sign fraction >= 0.6 |
| 35 | sfx_level_motion | dynamics | main | a | 0,1 | + 0.14/0.48, - 0.62/0.18 | ok | no | no sign with CLAP and MuQ intended-sign fraction >= 0.6 |
| 36 | sharp | timbre | main | b | 0,1 | + 0.90/0.48, - 0.96/0.32 | ok | no | no sign with CLAP and MuQ intended-sign fraction >= 0.6 |
| 37 | slurred_brass | articulation | main | b | 0,1,2 | + 0.72/0.46, - 0.82/0.50 | DRIFT | no | no sign with CLAP and MuQ intended-sign fraction >= 0.6 |
| 38 | timbral_warmth_r2 | timbre | main | d | 0,1 | + 0.42/0.44, - 0.70/0.46 | ok | no | no sign with CLAP and MuQ intended-sign fraction >= 0.6 |
| 39 | tongued_flute | articulation | main | d | 0,1 | + 0.78/0.32, - 0.86/0.28 | DRIFT | no | no sign with CLAP and MuQ intended-sign fraction >= 0.6 |
| 40 | trills_ornaments | articulation | main | d | 0,1 | + 0.38/0.72, - 0.98/0.40 | ok | no | no sign with CLAP and MuQ intended-sign fraction >= 0.6 |
| 41 | video_game | genre | main | a | 0,1 | + 0.94/0.54, - 0.88/0.40 | DRIFT | no | no sign with CLAP and MuQ intended-sign fraction >= 0.6 |
| 42 | wall_of_sound | production | main | d | 0,1,2 | + 0.58/0.72, - 0.88/0.48 | ok | no | no sign with CLAP and MuQ intended-sign fraction >= 0.6 |
| 43 | wide_vibrato | articulation | main | a | 0,1 | + 0.74/0.46, - 0.58/0.64 | DRIFT | no | no sign with CLAP and MuQ intended-sign fraction >= 0.6 |
| 44 | wind | sound_effect | main | b | 0,1 | + 0.84/0.56, - 0.52/0.30 | ok | no | no sign with CLAP and MuQ intended-sign fraction >= 0.6 |

Not in this table but also not shipped: the 6 v2 counterparts that lost the gate fail the bar themselves (close_mic,
impact_boom, live_recording, lofi_hip_hop, riser, wide_stereo); their v1 packs ship instead. legacy_warm fails the bar and ships.
Of the 46 v1 knobs: 39 ship as v1 vectors, 6 ship as v2 vectors, sfx_laden ships in no form.

## 3. Runbook versus what ran

| item | asked for | actually done | evidence |
|---|---|---|---|
| Phase 1 capture + labels | 20000-clip corpus, 7 scorers, CLAP/MuQ re-score for v2 anchors | done 22:41Z/22:44Z; 20000 clips x 3046 columns merged | status 22:27Z-22:50Z; eval/p3_status.txt |
| P0 filler | alpha-0 renders: 12 screen x 3 seeds, 36 applicability, 6 x 30 s hold | done 22:40Z (10 items; 3 cuDNN failures re-run) | status 22:38Z, 22:58Z |
| P1 screen | every candidate of every variant, two-family rule; finalize per variant group, survivors to P2 | main: 182 clusters -> 125 survivors; SFX: 57 -> 14. All four variants screened, but the variant was CHOSEN at screening (slope per LPAPS, ties to a), not after P2 | eval/selected.csv, selected_sfx.csv; results Notes |
| P2 protocol seed EVAL_SEED | every survivor | done 03:40Z: 153 bench packs (140 knobs incl. 5 legacy; the 13 c knobs as up/dn pairs) | eval/results_s0 (153 auc.json) |
| P3 protocol seed +1 | every survivor | done 07:05Z: 153 | eval/results_s1 (153) |
| P4 stacking | 200 pairs + 100 triples, rules 1/sqrt(n), 1/n, shared norm budget; pick rule keeping 95% | 300 combos x 3 rules (none, 1/sqrt n, 1/n). Shared norm budget NOT run. No rule reached 95% | eval/p4/stack (900 json), p4_report.json |
| P4 applicability | each knob, both signs, 36 prompts, CLAP+MuQ per category | done for 135 v2 knobs; SFX category scored on CLAP only | eval/p4/app, p4_report.json |
| P4 30 s hold | 6 prompts, 0-10 s vs 20-30 s, flag drift > 30% | done for 135 v2 knobs | eval/p4/hold (270 json) |
| P5 seed +2 | tie-break only (gain spread > 30% or reach mismatch) | 39 tie-break packs, then extended at 08:25Z to all 73 quality-bar packs lacking it: 111 packs | eval/results_s2 (111) |
| P6 ridge probes | per block, per label | done: 629 labels, 454 with R^2 > 0.5 | eval/ridge_r2.csv, ridge_best.json |
| P6 closed loop | top 10 by R^2 | done, 10 knobs; no consistent gain. Only 5 of the 10 ship with the vector tested (glitch_sfx, laser_zap, magic_sparkle, revving, sfx_applause); riser and ui_click ship as v1 vectors; boxy_tone, brassy_blare, honky_mids do not ship | eval/p6/*; results Closed loop |
| Variant a | single best block, cross-fitted | built (659 rows) | p3_status.txt |
| Variant b | top-3 blocks | built | p3_status.txt |
| Variant c | per-sign quartile vs median | built (1314 one-sided) | p3_status.txt |
| Variant d | not in the runbook's variant list (catalogue notes listed within-prompt centring as "truly missing") | added: within-prompt centring; chosen for 57 of the 125 main survivors; 23 shipped packs | status 22:50Z; selected.csv |
| Dedupe | abs cos > 0.8 within each variant | ran on variant a only (659 -> 182; SFX 118 -> 57); b, c, d inherit a's keepers by concept | status 22:45Z; results Notes |
| Residualised descriptors | catalogue controls column + build_directions support | NOT done | status 22:50Z; results Notes |
| SFX supplement | (added 22:58Z by Ryan) 40 rows, supplement prompts for thin classes | done: 1216 prompts, 57 candidates, 14 survivors, 9 shipped from corpus_sfx | eval/sfx_status.txt, sfx_counts.json |
| Regression gate | 46 shared concepts, v2 (seed 0) vs v1 phase 6; v2 wins on fraction >= v1 - 0.05, LPAPS <= cutoff, reach >= v1 | ran on 35 names with a v2 knob; LPAPS rule changed at 09:10Z to LPAPS/cutoff <= 1.1 x v1's; 11 v1 names had no v2 knob (kept by default); legacy five never compared | eval/gate.csv, final.py gate() |
| Cross-effect | on the final set, reusing v1 scripts | new scripts; ran on the 135 v2 bench knobs (270 signs), not on the final set: none of the 39 kept_v1 vectors or 5 legacy vectors | eval/crossfx_summary.csv |
| Knob table with seed spread, gate table, stacking, hold, closed loop in results | asked | done; "what the queue did NOT reach" section not written | results_many_knobs_v2.md |
| B2 mirror after phase 1 | yes | done 22:51Z, rclone check OK | status 22:51Z |
| B2 mirror after P1 | yes | done 00:33Z, check 0 differences | status 00:33Z |
| B2 mirror after P2 | yes | done 03:52Z; check listed 80 differences, all P3 files written during the copy | status 03:52Z |
| B2 final mirror | yes | done 10:16Z; check: only the mirror's live logs differ | status 10:16Z |
| Local copy | packs + tables to E: | done; eval tree arrived later (not in the memo) | E:\...\many_knobs_v2\ |
| Pack format v2 | a byte-compatible; b vectors+blocks; c vectors_neg | done: a/d/legacy/kept_v1 format 1 `vector`; b format 2 [3,1536]; c format 2 [1,1536] + vectors_neg | pack headers |

## 4. Product checks with numbers

Stacking (recomputed from eval/p4/stack, pass = mean LPAPS over 12 prompts <= the strictest member's cutoff):

| rule | all 300 | 200 pairs | 100 triples | 60 combos whose members all ship as that vector | median LPAPS / cutoff (all) |
|---|---|---|---|---|---|
| none | 0 (0%) | 0% | 0% | 0% | 1.56 |
| 1/sqrt(n) | 13 (4.3%) | 5.5% | 2.0% | 3.3% | 1.30 |
| 1/n (shipped default) | 106 (35.3%) | 32.0% | 42.0% | 31.7% | 1.06 |
| shared norm budget | not run | | | | |

Caveat that changes the reading: a single knob at its own calibrated probe already sits above its cutoff (new_v2 knob-signs:
median LPAPS/cutoff 1.19, only 8.7% at or under 1.0), because the gain is the first probe at or above the cutoff. The stacking
test compares combos against a bar most single knobs fail. Combos were drawn from the 135 bench knobs, so 240 of 300 include a
knob that does not ship as that vector.

Hold drift (drift = |effect 0-10 s - effect 20-30 s| / |effect 0-10 s|, flag > 0.3; effect = steered minus unsteered anchor
margin, CLAP and MuQ mean, 6 prompts):
- 63 of 135 v2 knobs flagged; median drift 0.29. Of the 63: 44 lost effect, 19 grew (the flag is two-sided), 11 changed sign.
- Shipped: 44 flagged of the 91 shipped packs with a hold record = 33 of 63 new_v2 (25 lost, 8 grew) + 11 of 28 kept_v1,
  whose record belongs to the v2 counterpart vector, not the shipped one. 16 shipped packs have no hold record (11 kept_v1, 5 legacy).
- 4 shipped new_v2 knobs have a 0-10 s effect at or below zero on the hold prompts (plate_reverb, sfx_far_away, solo_build,
  solo_crest), so their drift ratio is not meaningful.

SFX survival: 40 SFX catalogue rows -> 57 candidates after dedupe -> 14 P1 survivors -> 9 shipped from corpus_sfx (dishes,
glitch_sfx, magic_sparkle, revving, sfx_applause, sfx_baby_cry, sfx_insects, sfx_ocean_waves, sfx_telephone). Of the other 5
survivors, jet, laser_zap, riser and ui_click were evaluated from the main corpus entry (screened in both), fan failed the bar.
Sound-effect category in the shipped set: 20 (15 new_v2, 5 kept_v1). Applicability medians over 135 knobs: SFX prompts MuQ 0.54
/ CLAP 0.75, music prompts 0.63 / 0.83. Because SFX applicability is CLAP-only, 16 new_v2 packs (several instruments and
articulations, e.g. erhu, plate_reverb, quantized) are tagged applies_to = sfx only: they missed the music bar on MuQ but passed
the laxer SFX rule.

Cross-effect (own label moves more than every other label): 40 of 270 knob-signs over the 135 v2 knobs (21 of 135 on +, 19 on
-); 24 of the 126 shipped new_v2 knob-signs (median |own z| 1.21, median max off-target |z| 1.95); 5 of 56 for the v2 vectors
of the 28 kept_v1 names. No cross-effect exists for the 39 shipped kept_v1 vectors or the 5 legacy vectors in v2 (v1 had its
own 12-prompt crossfx for the 46 v1 knobs).

## 5. Metric definitions in use, and what does not compare across origins

- effect_clap / effect_muq (knob_table.csv): per sign, the sign-corrected mean anchor-alignment delta vs alpha 0 at alpha_end
  (the seed-0 probe at or above the cutoff), 50 TADA test prompts. Not an AUC. The memo's "0.95x MuQ / 1.02x CLAP of v1" is this
  number averaged over signs; it is taken where LPAPS overshoots the cutoff by different amounts per knob.
- intended-sign fraction (frac_clap / frac_muq): fraction of the 50 prompts whose anchor alignment moves the intended way at
  alpha_end, seed 0. v1's frac_*_50 is the same statistic at the v1 phase 6 gain.
- quality bar: some sign with both fractions >= 0.6 at seed 0, and the cutoff reached on some sign at seed 0.
- TADA AUC and ratio to PCI: per direction, sign-corrected MuQ-text or CLAP-text alignment delta integrated against LPAPS up to
  that sign's PCI-all cutoff; knob AUC = mean of the two directions; ratio = mean steer AUC / mean PCI AUC over a set of knobs.
- calibrated gain: knob value (alpha in knob_unit 0.1 of the pack's magnitude) at the first probe whose mean LPAPS reaches the
  PCI-all cutoff. Depends on the vector norm: d vectors are small (abs_forest magnitude 0.22 vs 1.38 for a v1 a vector), so d and
  some c gains run into the hundreds (up to 1206) against roughly 6 to 130 for a and b.

Recomputed TADA rows (MuQ ratio / CLAP ratio vs PCI; frac of knobs above their own PCI on MuQ):

| set | n | steer MuQ mean | PCI MuQ mean | MuQ ratio | CLAP ratio | frac > 1 MuQ |
|---|---|---|---|---|---|---|
| new_v2 shipped, s0 | 63 | 0.052 | 0.034 | 1.52 | 3.20 | 0.62 |
| new_v2 shipped, s1 / s2 | 63 | 0.052 / 0.047 | 0.033 / 0.029 | 1.58 / 1.61 | 3.13 / 3.21 | 0.63 / 0.63 |
| new_v2 shipped, s0, excluding the 7 c packs | 56 | | | 1.63 | 3.44 | |
| kept_v1 (v1 run, matches memo) | 39 | 0.056 | 0.045 | 1.24 | 3.23 | 0.67 |
| legacy, 2026-10-05 eval (matches memo) | 5 | 0.073 | 0.065 | 1.13 | 2.71 | 0.80 |
| legacy, v2 protocol s0 | 5 | 0.034 | 0.032 | 1.05 | 3.53 | 0.40 |
| all 107 shipped, mixed sources | 107 | 0.054 | 0.039 | 1.38 | 3.17 | 0.64 |
| all 140 v2-evaluated knobs, s0 | 140 | 0.037 | 0.031 | 1.18 | 3.05 | 0.54 |
| 28 gate losers: their v1 vector (v1 run) | 28 | 0.058 | | 1.25 | 3.28 | |
| 28 gate losers: their v2 vector (v2 s0) | 28 | 0.053 | | 1.16 | 2.66 | |

The new knobs' higher ratio comes mostly from smaller PCI denominators (0.034 vs 0.045), not from larger steer AUC (0.052 vs
0.056); and the 63 are quality-bar survivors of 135 (all 140 sit at 1.18).

Not comparable across the three origins, and why:
1. Gain: kept_v1 = one seed, v1 probe grid capped at alpha 64 (12 signs are lower bounds); new_v2 and legacy = 3-seed (warm
   2-seed) median on the v2 grid (reaches 128 and above). Different vector norms make raw gains incomparable across variants
   in any case.
2. Cutoff reached: kept_v1 has one flag per sign; the others have one per seed.
3. Intended-sign fractions and the quality bar: kept_v1 numbers come from the v1 run with v1 catalogue anchors at the v1 gain;
   new_v2 and legacy from the v2 run with v2 anchors (minimal pairs, re-scored). The 6 v2 winners and 28 v1 keepers were
   decided by comparing these two different measurements.
4. TADA AUC: same 50 prompts, seed and PCI descriptor text for the 33 names whose descriptors I could match across v1 and v2,
   but separate runs; kept_v1 values are v1-run values. Legacy AUC differs by source (1.13 in the 2026-10-05 eval with its own
   PCI pair and cutoffs, e.g. bright + cutoff 4.16, vs 1.05 in the v2 run, cutoff 3.85).
5. Hold and applies_to: for 28 kept_v1 packs these describe the v2 vector they lost to; for 11 kept_v1 and 5 legacy they were
   never measured (applies_to is a default).
6. Stacking and cross-effect: measured only on v2 bench vectors, so they say nothing about the 44 kept_v1 and legacy vectors.
7. Gain vs measurement seed: the shipped gain is a 3-seed median but the fraction, effect and AUC are seed-0 numbers at the
   seed-0 probe; on 37 new_v2 knob-signs the seeds disagree, so the measured strength is not the shipped strength.
8. c packs: the header's neg gain is in the up vector's units (results Notes); knob_table's dn-pack gain is in the dn
   vector's units (acid_house: 28.99 shipped vs 88.33 in knob_table).

## 6. Gaps, ranked by how much each undermines "107 knobs measured under one protocol"

| rank | gap | packs touched |
|---|---|---|
| 1 | 39 kept_v1 packs were never measured under the v2 protocol: gain, reach, fractions and AUC are v1 phase 6 numbers (one seed, capped grid, v1 anchors). 12 of their knob-signs carry lower-bound gains; 4 fail the quality bar on their own numbers and ship | 39 |
| 2 | Product metadata on the wrong vector or none: 28 kept_v1 packs carry hold_30s and applies_to measured on the v2 vector that lost the gate; 11 kept_v1 and 5 legacy have none. Stacking and cross-effect never touched any of these 44 shipped vectors | 44 (28 wrong vector, 16 none) |
| 3 | Legacy five bypass screening, the gate and the bar: built by a different method (discover.py, 32 pairs), "won" a comparison that never ran, warm fails the bar and has only 2 seeds | 5 |
| 4 | Measured strength is seed 0 at the seed-0 probe while the shipped gain is a 3-seed median; seeds disagree on 37 knob-signs | 63 (37 signs) |
| 5 | Variant chosen at screening (12 prompts, a few alphas), not after P2 as the runbook required; only one variant per concept got the full protocol, so the variant column is unvalidated | 63 |
| 6 | No listening check: every pack is metric-only; v2 audio is not on this PC | 107 |
| 7 | SFX knobs are scored with MuQ in the bar but CLAP-only in applicability; 16 new_v2 music knobs are tagged sfx-only by that asymmetry | 20 SFX packs; 16 sfx-only tags |
| 8 | Calibrated gain overshoots the cutoff (median 1.19x), so every downstream check (fractions, stacking bar) is taken past the cutoff by a knob-dependent amount | 107 |
| 9 | Residualisation not done and dedupe only on variant a: confounded directions (drum/electronic, tilt, level) may survive under several names | 63 |
| 10 | c-pack neg gain unit mismatch between header and knob_table; AUC pairing for c is inferred | 7 |

## 7. The two known inconsistencies

Legacy five: RESOLVED from code and data. final.py gate() looks up knob_table rows named bright, dense_arrangement,
percussive, rough and warm. knob_table.csv (140 knobs) has none of them (the v1 catalogue rows bright/warm/percussive/rough were
absorbed in dedupe or failed screening; the nearest v2 knobs, timbral_warmth_r2 and sharp, fail the bar). So every legacy row
takes the "v2 counterpart not evaluated" branch and the legacy pack wins by default. No effect-at-cutoff comparison ran. The
manifest's "legacy pack stronger by effect at cutoff" is a fixed string final.py writes for any legacy winner, and the status
lines at 09:10Z and 10:03Z ("legacy 5/5 beat their v2 counterparts") restate the tally, not a measurement. gate.csv is correct;
the status log and manifest wording are wrong. Separately, legacy warm fails the v2 quality bar and ships because the bar is
not applied to legacy winners.

sfx_laden: RESOLVED from data. Gate: v2 sfx_laden (variant d) vs v1 on seed 0: + CLAP 0.44 vs 0.32, MuQ 0.70 vs 0.62; - CLAP
0.46 vs 0.38, MuQ 0.60 vs 0.64 (within the 0.05 margin); LPAPS rule passed; so v2 "won". The bar was then applied to the v2
pack and failed it (CLAP < 0.6 on both signs). The v1 pack fails the same bar on its own numbers (CLAP 0.32 / 0.38), so dropping
the knob is consistent with the bar. What is inconsistent is the policy: the bar is applied to v2 winners but not to kept_v1
packs (close_mic, live_recording, sensual, wide_stereo fail it and ship), and the runbook's "never regress a knob" is broken
for sfx_laden, which was a shipped v1 knob and now ships in no form.
