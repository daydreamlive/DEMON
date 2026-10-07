# many-knobs v2 results (SA3 medium, post_block_residual)

Shipped 107 packs; not shipped 44. Variants shipped: {'a': 59, 'd': 23, 'b': 13, 'c': 7, 'legacy': 5}. Kept from v1: 39; legacy winners: 5.

## Executive summary

- Shipped 107 packs from 140 evaluated v2 knobs: 63 new v2 knobs (variant a 20, d 23, b 13, c 7), 39 v1 packs kept by the regression gate, 5 legacy packs. 44 knobs not shipped (no sign with CLAP and MuQ intended-sign fraction >= 0.6).
- Better than v1: 2.3x the knob count, a new articulation category and more rhythm, space and dynamics knobs (spiccato, accelerando, plate_reverb, solo_crest, ...), every new knob reaches the PCI cutoff on both signs on every seed run, calibrated gains are a 2 to 3 seed median, and multi-block (b) and per-sign (c) packs exist.
- Not better than v1 per knob: the new knobs' effect at calibrated gain is 0.95x (MuQ) / 1.02x (CLAP) of the v1 knobs'. The gate kept v1 for 39 of 46 shared concepts (28 lost on intended-sign fraction or LPAPS, 11 not re-found); v2 won 7. Closed loop gave no consistent gain over open loop.
- TADA metric (shipped knobs with a local AUC, 44 of 107): 1.22x PCI on MuQ, 3.13x on CLAP; v1 was 1.20x / 3.09x. TADA's CAA(loc) on ACE-Step is 1.27x / 1.44x. Our earlier TADA-recipe replication on SA3 was 0.61x MuQ. The 63 new knobs have no AUC on this PC (see Comparison with TADA).
- Negative 1, stacking: no attenuation rule keeps 95% of pairs and triples under the strictest member's cutoff (none 0%, 1/sqrt(n) 4%, 1/n 35%; 1/n ships as the default).
- Negative 2, hold drift: 63 of 135 knobs lose more than 30% of their 0-10 s effect by 20-30 s (44 of the 91 shipped knobs with hold data); median drift 0.29.
- Negative 3, SFX coverage: the SFX expansion (40 catalogue rows, 57 screening candidates) gave 14 screening survivors and 9 shipped knobs. MuQ is blind to SFX (median intended-sign MuQ fraction on SFX prompts 0.54 vs 0.63 on music), so SFX knobs are judged on CLAP alone.

## Comparison with TADA

Metric, same as v1 (`sa3_tada_score.py auc`): per knob and direction, the sign-corrected MuQ-text or CLAP-text alignment delta integrated against LPAPS up to the PCI-all cutoff of that sign (`compute_alignment_auc_direction`), averaged over the two directions. PCI = the knob's own PCI-all prompt pair on the same 50 TADA test prompts, seed 2115. Ratio vs PCI = mean steer AUC / mean PCI AUC over knobs (TADA's convention; ratio of medians in brackets). Frac > 1 = fraction of knobs whose steer AUC exceeds their own PCI AUC.

| row set | n | MuQ AUC median / mean | PCI MuQ median / mean | MuQ ratio vs PCI | frac > 1 MuQ | CLAP AUC median / mean | PCI CLAP median / mean | CLAP ratio vs PCI | frac > 1 CLAP |
|---|---|---|---|---|---|---|---|---|---|
| v2 shipped, AUC available (39 v1-kept + 5 legacy) | 44 | 0.052 / 0.058 | 0.037 / 0.047 | **1.22** (1.38) | 0.68 (30/44) | 0.079 / 0.070 | 0.022 / 0.022 | **3.13** (3.62) | 0.91 (40/44) |
| of which v1-kept | 39 | 0.047 / 0.056 | 0.037 / 0.045 | 1.24 (1.27) | 0.67 (26/39) | 0.078 / 0.067 | 0.022 / 0.021 | 3.23 (3.57) | 0.92 (36/39) |
| of which legacy five | 5 | 0.070 / 0.073 | 0.064 / 0.065 | 1.13 (1.09) | 0.80 (4/5) | 0.118 / 0.095 | 0.026 / 0.035 | 2.71 (4.61) | 0.80 (4/5) |
| v2 shipped, new knobs | 63 | not local | not local | n/a | n/a | not local | not local | n/a | n/a |
| v2 new knobs, proxy: endpoint effect (see below) | 63 | 0.043 / 0.053 | none | 0.95x of v1 (mean) | n/a | 0.066 / 0.067 | none | 1.02x of v1 (mean) | n/a |
| v1, all 46 knobs | 46 | 0.046 / 0.054 | 0.036 / 0.045 | **1.20** (1.28) | 0.63 (29/46) | 0.073 / 0.065 | 0.020 / 0.021 | **3.09** (3.61) | 0.91 (42/46) |
| v1, proxy: endpoint effect, same statistic | 46 | 0.050 / 0.056 | 0.018 / 0.024 (full swap) | 2.28 | n/a | 0.069 / 0.066 | 0.009 / 0.011 (full swap) | 6.05 | n/a |
| SA3, earlier TADA-recipe replication (CAA, cross-attn blocks 3,5,6,7, 7 concepts) | 7 | 0.021 (loc) / 0.024 (all) mean | 0.034 mean | 0.61 loc / 0.71 all | 2/7 (loc) | 0.006 mean | 0.009 mean | 0.67 | n/a |
| Paper, ACE-Step, PCI (7 non-vocal) | 7 | 0.094 mean | | 1.00 | | 0.045 mean | | 1.00 | |
| Paper, ACE-Step, CAA (loc) | 7 | 0.119 mean | | 1.27 | 6/7 (1 tie) | 0.065 mean | | 1.44 | n/a |
| Paper, ACE-Step, SAE (loc) | 7 | 0.132 mean | | 1.40 | 6/7 | 0.071 mean | | 1.58 | n/a |

What the rows are:
- The per-knob v2 AUC files (`auc.json` under results/ and results_s1/) were not copied to this PC; they are only in B2 (`sa3/steering_bench/many_knobs_v2_2026-10-05/eval/`) and this PC has no B2 client. So the only shipped knobs with a TADA AUC are the 39 v1 packs (v1's local auc.json, same packs, prompts and seed) and the legacy five (2026-10-05 eval/auc.json: production packs, same 50 prompts and seed, but that run's own anchor text, not the v2 catalogue anchors).
- The proxy for the 63 new knobs is `effect_clap` / `effect_muq` from tables/knob_table.csv: the sign-corrected mean anchor-alignment delta vs alpha 0 at the calibrated probe alpha (seed 1), averaged over the two signs. I recomputed the same statistic from v1's protocol CSVs (jet: 0.069 CLAP from v1 CSVs vs 0.066 in the v2 table) so the v1 and v2 proxy rows match. It is NOT an AUC (one strength, not integrated against LPAPS), it is measured where LPAPS is about 1.3x the cutoff, and it has no per-concept PCI because the v2 PCI results are also only in B2. On v1 the proxy ratio ranks knobs like the AUC ratio (Spearman 0.75 MuQ, 0.82 CLAP), but its scale is not the AUC scale (2.28 vs 1.20 on MuQ), so only the v2-new vs v1 multiplier (0.95x / 1.02x) is read from it.
- Paper fractions are from the per-concept MuQ tables (non-vocal concepts); per-concept CLAP was not tabulated. The earlier SA3 replication ratio is the ratio of 7-concept means, as here.

Caveats (apples to oranges):
- Different model and site (SA3 medium post-block residual vs ACE-Step cross-attention CAA or SAE), and LPAPS cutoffs are calibrated per model, so only ratios vs PCI compare, never absolute AUCs.
- 50 TADA test prompts and one seed vs their 100 tracks; our 107 (46 in v1) concepts were chosen and screened by us (survivors of 182 / 136 candidates, which inflates the ratio) vs their 7 fixed concepts.
- SA3 generates no vocals, so the paper rows use its 7 non-vocal concepts; CLAP-text PCI is small on SA3 and many knobs were estimated from CLAP labels, so CLAP ratios are inflated and MuQ is the fairer column.

## Knob table

| pack | knob | category | variant | blocks | gain + (min-max) | gain - (min-max) | reached +/- (per seed) | applies_to | kept_from |
|---|---|---|---|---|---|---|---|---|---|
| abs_cinematic | abs_cinematic | abstract | a | [13] | 46.23 | 23.12 | y/y | music | v1 |
| abs_forest | abs_forest | abstract | d | [16] | 592.97 (592.97-592.97) | 296.49 (148.24-296.49) | yyy/yyy | music,sfx |  |
| abs_industrial | abs_industrial | abstract | a | [11] | 66.15 | 33.08 | y/y | music | v1 |
| accelerando | accelerando | rhythm | a | [11] | 73.09 (73.09-73.09) | 146.18 (73.09-146.18) | yyy/yyy | music,sfx,abstract |  |
| acid_303 | acid_303 | instrument | a | [11] | 55.56 | 55.56 | n/y | music | v1 |
| acid_house | acid_house | genre | c | [12] | 57.98 (57.98-115.96) | 28.99 (28.99-57.98) | yyy/yyy | music,sfx,abstract |  |
| aggressive | aggressive | mood | a | [13] | 55.61 (55.61-55.61) | 27.81 (27.81-55.61) | yyy/yyy | music,sfx,abstract |  |
| alarm_clock | alarm_clock | sound_effect | a | [15] | 72.28 | 72.28 | n/y | music,sfx,abstract | v1 |
| analog | analog | production | a | [15] | 49.41 | 49.41 | y/n | sfx | v1 |
| arpeggiated | arpeggiated_r2 | rhythm | b | [11, 12, 10] | 17.44 (17.44-17.44) | 17.44 (17.44-17.44) | yyy/yyy | music,sfx,abstract |  |
| bedroom_intimate | bedroom_intimate | space | d | [19] | 430.41 (215.20-430.41) | 430.41 (215.20-430.41) | yyy/yyy | music,sfx,abstract |  |
| black_metal | black_metal | genre | a | [11] | 54.63 (54.63-54.63) | 27.32 (27.32-54.63) | yyy/yyy | music,sfx,abstract |  |
| bright | bright | timbre | legacy | [23] | 8.62 (8.62-8.62) | 8.62 (8.62-17.24) | yyy/yyy | music | legacy |
| childrens | childrens | genre | a | [13] | 45.78 | 22.89 | y/y | music | v1 |
| christmas | christmas | genre | d | [16] | 417.84 (417.84-417.84) | 417.84 (208.92-417.84) | yyy/yyy | abstract |  |
| clave_pattern | clave_pattern | rhythm | b | [11, 13, 12] | 16.55 (16.55-33.09) | 16.55 (16.55-33.09) | yyy/yyy | sfx |  |
| close_mic | close_mic | space | a | [11] | 59.21 | 59.21 | y/n | music,sfx | v1 |
| cumbia | cumbia | genre | b | [11, 13, 12] | 25.54 (25.54-25.54) | 6.39 (6.39-6.39) | yyy/yyy | sfx,abstract |  |
| dark | dark | mood | a | [11] | 26.56 (26.56-26.56) | 26.56 (26.56-53.13) | yyy/yyy | abstract |  |
| dense_arrangement | dense_arrangement | rhythm | legacy | [1] | 24.81 (24.81-24.81) | 24.81 (12.41-24.81) | yyy/yyy | music | legacy |
| dishes | dishes | sound_effect | b | [15, 16, 13] | 22.84 (22.84-22.84) | 22.84 (11.42-22.84) | yyy/yyy | music,sfx,abstract |  |
| dissonant | dissonant | timbre | a | [13] | 53.03 | 53.03 | y/y | music,sfx,abstract | v1 |
| distant | distant | space | a | [22] | 57.50 | 28.75 | y/y | music | v1 |
| dreamy | dreamy | mood | a | [13] | 40.93 | 40.93 | y/y | music,sfx,abstract | v1 |
| drone | drone | genre | a | [13] | 46.46 | 46.46 | y/y | abstract | v1 |
| drum_machine | drum_machine | instrument | a | [18] | 44.49 (44.49-44.49) | 22.24 (22.24-22.24) | yyy/yyy | music |  |
| dub_space_echo | dub_space_echo | space | d | [12] | 250.32 (250.32-250.32) | 250.32 (125.16-250.32) | yyy/yyy | music,sfx,abstract |  |
| electric_piano | electric_piano_r2 | instrument | d | [16] | 266.76 (266.76-533.53) | 266.76 (266.76-266.76) | yyy/yyy | music,abstract |  |
| erhu | erhu | instrument | d | [19] | 588.67 (588.67-588.67) | 294.33 (294.33-294.33) | yyy/yyy | sfx |  |
| explosion | explosion | sound_effect | b | [12, 13, 15] | 30.32 (15.16-30.32) | 30.32 (15.16-30.32) | yyy/yyy | music,sfx |  |
| fizzy_highs | fizzy_highs | timbre | a | [22] | 25.10 (25.10-25.10) | 50.20 (50.20-50.20) | yyy/yyy | music,sfx,abstract |  |
| frogs | frogs | sound_effect | a | [14] | 128.34 (64.17-128.34) | 64.17 (64.17-128.34) | yyy/yyy | music,sfx,abstract |  |
| future_bass | future_bass | genre | a | [13] | 47.86 | 47.86 | y/y | music,sfx,abstract | v1 |
| fuzzy | fuzzy | timbre | d | [16] | 263.20 (263.20-263.20) | 526.40 (263.20-526.40) | yyy/yyy | abstract |  |
| glitch_sfx | glitch_sfx | sound_effect | b | [12, 13, 11] | 24.12 (24.12-24.12) | 12.06 (12.06-12.06) | yyy/yyy | music,sfx,abstract |  |
| glitchy | glitchy | production | a | [19] | 41.97 | 41.97 | y/y | music,sfx,abstract | v1 |
| gritty | gritty | timbre | a | [11] | 57.08 | 57.08 | y/y | music | v1 |
| hammered_notes | hammered_notes | articulation | d | [16] | 388.40 (388.40-388.40) | 776.80 (388.40-776.80) | yyy/yyy | music,sfx,abstract |  |
| handclaps | handclaps | instrument | a | [11] | 79.98 (79.98-79.98) | 39.99 (39.99-39.99) | yyy/yyy | music |  |
| hardstyle | hardstyle | genre | a | [13] | 45.89 | 45.89 | y/y | sfx,abstract | v1 |
| horror_score | horror_score | genre | a | [11] | 48.32 | 48.32 | y/y | music | v1 |
| impact_boom | impact_boom | sound_effect | a | [15] | 57.37 | 57.37 | n/y | sfx | v1 |
| intense | intense | dynamics | a | [10] | 29.94 | 59.89 | y/y | music,sfx,abstract | v1 |
| jet | jet | sound_effect | a | [15] | 56.27 (56.27-56.27) | 56.27 (56.27-56.27) | yyy/yyy | sfx |  |
| jingle_bells | jingle_bells | instrument | a | [9] | 67.67 | 33.83 | y/y | music,sfx,abstract | v1 |
| koto | koto | instrument | a | [13] | 53.01 | 53.01 | y/y | music | v1 |
| laser_zap | laser_zap | sound_effect | b | [15, 13, 11] | 24.62 (24.62-24.62) | 24.62 (12.31-24.62) | yyy/yyy | sfx |  |
| legato_phrasing | legato_phrasing | articulation | d | [19] | 289.42 (289.42-289.42) | 289.42 (289.42-289.42) | yyy/yyy | music,sfx,abstract |  |
| live_recording | live_recording | production | a | [13] | 68.88 | 68.88 | y/y | sfx | v1 |
| locked_groove | locked_groove | rhythm | a | [13] | 119.62 (59.81-119.62) | 119.62 (119.62-119.62) | yyy/yyy | sfx |  |
| lofi_hip_hop | lofi_hip_hop | genre | a | [11] | 65.78 | 32.89 | n/y | music,abstract | v1 |
| magic_sparkle | magic_sparkle | sound_effect | a | [15] | 34.22 (34.22-34.22) | 34.22 (34.22-34.22) | yyy/yyy | music,sfx,abstract |  |
| mains_hum | mains_hum | production | a | [22] | 112.51 | 112.51 | y/y | sfx,abstract | v1 |
| marcato | marcato | articulation | b | [11, 9, 10] | 24.85 (24.85-24.85) | 24.85 (12.42-24.85) | yyy/yyy | music,sfx,abstract |  |
| muted_horn | muted_horn | timbre | b | [11, 9, 23] | 13.68 (13.68-13.68) | 13.68 (13.68-13.68) | yyy/yyy | music,sfx |  |
| new_age | new_age | genre | c | [13] | 57.00 (57.00-57.00) | 28.50 (28.50-28.50) | yyy/yyy | music,sfx,abstract |  |
| nylon_soft | nylon_soft | timbre | a | [11] | 64.00 (64.00-64.00) | 64.00 (32.00-64.00) | yyy/yyy | sfx |  |
| palm_muted | palm_muted | articulation | d | [16] | 259.61 (259.61-259.61) | 259.61 (259.61-519.22) | yyy/yyy | music,sfx,abstract |  |
| percussive | percussive | timbre | legacy | [15] | 31.88 (15.94-31.88) | 15.94 (15.94-15.94) | yyy/yyy | music | legacy |
| phone_ring | phone_ring | sound_effect | a | [15] | 67.72 | 67.72 | n/y | sfx | v1 |
| plate_reverb | plate_reverb | space | d | [16] | 461.36 (230.68-461.36) | 461.36 (461.36-461.36) | yyy/yyy | sfx |  |
| playful | playful | mood | a | [11] | 51.27 | 25.64 | y/y | music | v1 |
| polyrhythmic | polyrhythmic | rhythm | a | [13] | 71.92 (71.92-71.92) | 71.92 (71.92-71.92) | yyy/yyy | music,sfx,abstract |  |
| post_rock | post_rock | genre | a | [11] | 55.68 | 27.84 | y/y | music,abstract | v1 |
| pulsing_synth | pulsing_synth | rhythm | d | [16] | 644.94 (322.47-644.94) | 322.47 (322.47-644.94) | yyy/yyy | sfx |  |
| quantized | quantized | rhythm | d | [19] | 1206.25 (603.12-1206.25) | 603.12 (603.12-603.12) | yyy/yyy | sfx |  |
| radio_filtered | radio_filtered | production | a | [16] | 68.80 | 68.80 | y/y | music | v1 |
| rain_on_surface | rain_on_surface | sound_effect | a | [15] | 43.09 (43.09-43.09) | 43.09 (43.09-43.09) | yyy/yyy | sfx,abstract |  |
| revving | revving | sound_effect | c | [15] | 63.65 (63.65-63.65) | 31.83 (31.83-31.83) | yyy/yyy | sfx |  |
| riser | riser | sound_effect | a | [15] | 58.86 | 58.86 | n/y | sfx | v1 |
| rough | rough | timbre | legacy | [2] | 20.34 (20.34-20.34) | 20.34 (20.34-20.34) | yyy/yyy | music | legacy |
| samba | samba | genre | a | [11] | 52.41 | 26.20 | y/y | music | v1 |
| sensual | sensual | mood | a | [13] | 48.36 | 24.18 | y/y | music | v1 |
| sfx_applause | sfx_applause | sound_effect | b | [12, 13, 11] | 11.02 (11.02-11.02) | 11.02 (11.02-11.02) | yyy/yyy | music,sfx,abstract |  |
| sfx_baby_cry | sfx_baby_cry | sound_effect | a | [13] | 28.45 (28.45-28.45) | 56.90 (56.90-56.90) | yyy/yyy | sfx,abstract |  |
| sfx_event_rate | sfx_event_rate | rhythm | a | [15] | 69.66 (69.66-69.66) | 139.32 (139.32-139.32) | yyy/yyy | music |  |
| sfx_far_away | sfx_far_away | space | a | [13] | 68.55 (68.55-137.11) | 68.55 (68.55-137.11) | yyy/yyy | sfx,abstract |  |
| sfx_indoor_hall | sfx_indoor_hall | space | d | [15] | 537.21 (537.21-537.21) | 537.21 (537.21-537.21) | yyy/yyy | music,sfx |  |
| sfx_insects | sfx_insects | sound_effect | a | [13] | 61.65 (61.65-61.65) | 61.65 (61.65-61.65) | yyy/yyy | music,sfx,abstract |  |
| sfx_ocean_waves | sfx_ocean_waves | sound_effect | c | [13] | 29.62 (29.62-29.62) | 29.62 (29.62-29.62) | yyy/yyy | music,sfx,abstract |  |
| sfx_sub_weight | sfx_sub_weight | space | c | [15] | 68.76 (68.76-68.76) | 68.76 (68.76-137.51) | yyy/yyy | music,sfx |  |
| sfx_telephone | sfx_telephone | sound_effect | d | [23] | 212.24 (212.24-212.24) | 212.24 (212.24-212.24) | yyy/yyy | sfx |  |
| shaker_pulse | shaker_pulse | rhythm | d | [12] | 325.65 (325.65-325.65) | 325.65 (325.65-325.65) | yyy/yyy | music,abstract |  |
| shimmer_reverb | shimmer_reverb | space | c | [12] | 85.16 (85.16-85.16) | 85.16 (85.16-85.16) | yyy/yyy | music,sfx,abstract |  |
| shimmering | shimmering_r2 | timbre | d | [16] | 327.38 (327.38-327.38) | 327.38 (327.38-327.38) | yyy/yyy | sfx |  |
| sitar | sitar_r2 | instrument | d | [19] | 633.48 (316.74-633.48) | 316.74 (316.74-316.74) | yyy/yyy | music,sfx,abstract |  |
| smash | smash | sound_effect | b | [15, 13, 12] | 39.71 (39.71-39.71) | 39.71 (39.71-39.71) | yyy/yyy | sfx |  |
| solo_build | solo_build | dynamics | c | [12] | 435.36 (217.68-435.36) | 217.68 (217.68-217.68) | yyy/yyy | music,sfx,abstract |  |
| solo_crest | solo_crest | dynamics | d | [23] | 92.52 (92.52-92.52) | 46.26 (46.26-92.52) | yyy/yyy | music |  |
| solo_instrument | solo_instrument | production | a | [12] | 57.19 | 57.19 | y/y | music,sfx,abstract | v1 |
| spiccato | spiccato | articulation | d | [16] | 393.20 (393.20-393.20) | 393.20 (393.20-393.20) | yyy/yyy | music,sfx,abstract |  |
| strummed_guitar | strummed_guitar | instrument | a | [13] | 43.35 | 43.35 | y/y | abstract | v1 |
| surf_rock | surf_rock | genre | a | [13] | 47.54 | 23.77 | y/y | sfx,abstract | v1 |
| swing | swing_r2 | rhythm | b | [11, 12, 13] | 16.98 (16.98-16.98) | 16.98 (16.98-33.95) | yyy/yyy | sfx |  |
| synth_pad | synth_pad | instrument | a | [15] | 41.16 | 41.16 | n/y | music,sfx,abstract | v1 |
| synthesizer | synthesizer | instrument | a | [16] | 47.81 | 47.81 | n/y | music | v1 |
| tango | tango | genre | a | [13] | 40.80 | 40.80 | n/y | music | v1 |
| thick_unison | thick_unison | timbre | b | [13, 11, 15] | 37.15 (37.15-37.15) | 37.15 (37.15-37.15) | yyy/yyy | sfx |  |
| timpani | timpani | instrument | a | [15] | 47.80 | 47.80 | y/y | music,abstract | v1 |
| tom_patterns | tom_patterns | rhythm | d | [16] | 286.20 (286.20-286.20) | 286.20 (286.20-286.20) | yyy/yyy | sfx |  |
| ui_click | ui_click | sound_effect | a | [12] | 63.54 | 63.54 | n/y | music,sfx,abstract | v1 |
| ukulele | ukulele | instrument | a | [11] | 59.23 (59.23-59.23) | 29.61 (29.61-29.61) | yyy/yyy | music,sfx,abstract |  |
| vaporwave | vaporwave_r2 | genre | a | [16] | 46.65 (46.65-46.65) | 23.32 (23.32-23.32) | yyy/yyy | music |  |
| virtuosic_runs | virtuosic_runs | articulation | d | [13] | 290.25 (290.25-290.25) | 290.25 (290.25-290.25) | yyy/yyy | music,sfx,abstract |  |
| warm | warm | timbre | legacy | [15] | 21.68 (21.68-21.68) | 21.68 (21.68-21.68) | yy/yy | music | legacy |
| wide_stereo | wide_stereo | space | a | [12] | 27.84 | 55.68 | y/y | sfx | v1 |
| woody_body | woody_body | timbre | d | [18] | 303.23 (303.23-303.23) | 303.23 (303.23-303.23) | yyy/yyy | music,sfx,abstract |  |

## Not shipped

| knob | variant | why |
|---|---|---|
| amateur | d | no sign with CLAP and MuQ intended-sign fraction >= 0.6 |
| bass_808 | d | no sign with CLAP and MuQ intended-sign fraction >= 0.6 |
| bluegrass | d | no sign with CLAP and MuQ intended-sign fraction >= 0.6 |
| boomy | d | no sign with CLAP and MuQ intended-sign fraction >= 0.6 |
| boxy_tone | b | no sign with CLAP and MuQ intended-sign fraction >= 0.6 |
| brassy_blare | b | no sign with CLAP and MuQ intended-sign fraction >= 0.6 |
| breakbeat_feel | d | no sign with CLAP and MuQ intended-sign fraction >= 0.6 |
| chord_stabs | a | no sign with CLAP and MuQ intended-sign fraction >= 0.6 |
| crowd | c | no sign with CLAP and MuQ intended-sign fraction >= 0.6 |
| damped_strings | d | no sign with CLAP and MuQ intended-sign fraction >= 0.6 |
| deep_r2 | d | no sign with CLAP and MuQ intended-sign fraction >= 0.6 |
| djembe | d | no sign with CLAP and MuQ intended-sign fraction >= 0.6 |
| dotted_rhythm | d | no sign with CLAP and MuQ intended-sign fraction >= 0.6 |
| driving_eighths | c | no sign with CLAP and MuQ intended-sign fraction >= 0.6 |
| echo_delay | a | no sign with CLAP and MuQ intended-sign fraction >= 0.6 |
| enjoyment | b | no sign with CLAP and MuQ intended-sign fraction >= 0.6 |
| fan | a | no sign with CLAP and MuQ intended-sign fraction >= 0.6 |
| funny | b | no sign with CLAP and MuQ intended-sign fraction >= 0.6 |
| glass_clink | a | no sign with CLAP and MuQ intended-sign fraction >= 0.6 |
| halftime_r2 | d | no sign with CLAP and MuQ intended-sign fraction >= 0.6 |
| hihat_rolls | d | no sign with CLAP and MuQ intended-sign fraction >= 0.6 |
| honky_mids | a | no sign with CLAP and MuQ intended-sign fraction >= 0.6 |
| laid_back | c | no sign with CLAP and MuQ intended-sign fraction >= 0.6 |
| lush_r2 | d | no sign with CLAP and MuQ intended-sign fraction >= 0.6 |
| majestic | b | no sign with CLAP and MuQ intended-sign fraction >= 0.6 |
| modular_synth_r2 | b | no sign with CLAP and MuQ intended-sign fraction >= 0.6 |
| no_percussion | d | no sign with CLAP and MuQ intended-sign fraction >= 0.6 |
| noise_floor | b | no sign with CLAP and MuQ intended-sign fraction >= 0.6 |
| odd_meter | b | no sign with CLAP and MuQ intended-sign fraction >= 0.6 |
| ostinato_r2 | c | no sign with CLAP and MuQ intended-sign fraction >= 0.6 |
| pitch_bends | b | no sign with CLAP and MuQ intended-sign fraction >= 0.6 |
| sfx_close_foley | d | no sign with CLAP and MuQ intended-sign fraction >= 0.6 |
| sfx_laden | d | no sign with CLAP and MuQ intended-sign fraction >= 0.6 |
| sfx_level | d | no sign with CLAP and MuQ intended-sign fraction >= 0.6 |
| sfx_level_motion | a | no sign with CLAP and MuQ intended-sign fraction >= 0.6 |
| sharp | b | no sign with CLAP and MuQ intended-sign fraction >= 0.6 |
| slurred_brass | b | no sign with CLAP and MuQ intended-sign fraction >= 0.6 |
| timbral_warmth_r2 | d | no sign with CLAP and MuQ intended-sign fraction >= 0.6 |
| tongued_flute | d | no sign with CLAP and MuQ intended-sign fraction >= 0.6 |
| trills_ornaments | d | no sign with CLAP and MuQ intended-sign fraction >= 0.6 |
| video_game | a | no sign with CLAP and MuQ intended-sign fraction >= 0.6 |
| wall_of_sound | d | no sign with CLAP and MuQ intended-sign fraction >= 0.6 |
| wide_vibrato | a | no sign with CLAP and MuQ intended-sign fraction >= 0.6 |
| wind | b | no sign with CLAP and MuQ intended-sign fraction >= 0.6 |

## Regression gate (v2 vs v1 phase 6) and legacy five

| knob | v1 / legacy | winner | why |
|---|---|---|---|
| aggressive | aggressive | v2 |  |
| alarm_clock | alarm_clock | v1 | + LPAPS/cutoff 1.23 > 1.1 x v1 0.88; - clap 0.50 < v1 0.92 - 0.05; - muq 0.48 < v1 0.70 - 0.05 |
| analog | analog | v1 | + clap 0.62 < v1 0.70 - 0.05; - clap 0.66 < v1 0.84 - 0.05; - muq 0.36 < v1 0.44 - 0.05 |
| black_metal | black_metal | v2 |  |
| close_mic | close_mic | v1 | + clap 0.14 < v1 0.36 - 0.05; + muq 0.38 < v1 0.74 - 0.05; - clap 0.12 < v1 0.26 - 0.05; - muq 0.44 < v1 0.68 - 0.05 |
| dissonant | dissonant | v1 | + clap 0.66 < v1 0.86 - 0.05; - clap 0.78 < v1 0.92 - 0.05 |
| dreamy | dreamy | v1 | + muq 0.42 < v1 0.68 - 0.05 |
| drone | drone | v1 | + muq 0.76 < v1 0.94 - 0.05; - clap 0.56 < v1 0.70 - 0.05; - muq 0.60 < v1 0.70 - 0.05 |
| future_bass | future_bass | v1 | + clap 0.84 < v1 0.92 - 0.05; + LPAPS/cutoff 1.12 > 1.1 x v1 1.01; - LPAPS/cutoff 1.21 > 1.1 x v1 1.08 |
| glitchy | glitchy | v1 | + muq 0.90 < v1 0.98 - 0.05; + LPAPS/cutoff 1.33 > 1.1 x v1 1.20; - muq 0.22 < v1 0.42 - 0.05 |
| hardstyle | hardstyle | v1 | + muq 0.66 < v1 0.76 - 0.05; + LPAPS/cutoff 1.32 > 1.1 x v1 1.12 |
| impact_boom | impact_boom | v1 | + clap 0.40 < v1 0.82 - 0.05; + muq 0.62 < v1 0.94 - 0.05; + LPAPS/cutoff 1.33 > 1.1 x v1 0.84; - clap 0.34 < v1 0.60 - 0.05; - muq 0.30 < v1 0.36 - 0.05 |
| intense | intense | v1 | + LPAPS/cutoff 1.38 > 1.1 x v1 1.07 |
| jet | jet | v2 |  |
| jingle_bells | jingle_bells | v1 | - clap 0.82 < v1 0.90 - 0.05 |
| live_recording | live_recording | v1 | - clap 0.40 < v1 0.54 - 0.05 |
| lofi_hip_hop | lofi_hip_hop | v1 | + clap 0.48 < v1 0.62 - 0.05; + LPAPS/cutoff 1.39 > 1.1 x v1 0.89; - clap 0.44 < v1 0.62 - 0.05; - muq 0.64 < v1 0.70 - 0.05 |
| mains_hum | mains_hum | v1 | - clap 0.56 < v1 0.88 - 0.05 |
| phone_ring | phone_ring | v1 | + LPAPS/cutoff 1.20 > 1.1 x v1 0.93; - clap 0.76 < v1 0.90 - 0.05 |
| polyrhythmic | polyrhythmic | v2 |  |
| post_rock | post_rock | v1 | + clap 0.82 < v1 1.00 - 0.05; + LPAPS/cutoff 1.34 > 1.1 x v1 1.03; - LPAPS/cutoff 1.28 > 1.1 x v1 1.08 |
| radio_filtered | radio_filtered | v1 | - clap 0.58 < v1 0.68 - 0.05 |
| rain_on_surface | rain_on_surface | v2 |  |
| riser | riser | v1 | + muq 0.48 < v1 0.76 - 0.05; + LPAPS/cutoff 1.30 > 1.1 x v1 0.91; - clap 0.44 < v1 0.56 - 0.05 |
| sfx_laden | sfx_laden | v2 |  |
| solo_instrument | solo_instrument | v1 | + clap 0.76 < v1 0.82 - 0.05; - muq 0.48 < v1 0.58 - 0.05 |
| strummed_guitar | strummed_guitar | v1 | + LPAPS/cutoff 1.09 > 1.1 x v1 1.00; - clap 0.80 < v1 0.88 - 0.05; - LPAPS/cutoff 1.44 > 1.1 x v1 1.29 |
| surf_rock | surf_rock | v1 | + muq 0.74 < v1 0.82 - 0.05; - muq 0.58 < v1 0.66 - 0.05 |
| synth_pad | synth_pad | v1 | + clap 0.94 < v1 1.00 - 0.05; - clap 0.64 < v1 0.80 - 0.05 |
| synthesizer | synthesizer | v1 | + clap 0.50 < v1 0.64 - 0.05; + LPAPS/cutoff 1.22 > 1.1 x v1 0.97 |
| tango | tango | v1 | + clap 0.10 < v1 0.66 - 0.05; + muq 0.60 < v1 0.70 - 0.05; + LPAPS/cutoff 1.34 > 1.1 x v1 0.90; - clap 0.66 < v1 0.80 - 0.05 |
| timpani | timpani | v1 | + clap 0.24 < v1 0.86 - 0.05; + muq 0.78 < v1 0.94 - 0.05; + LPAPS/cutoff 1.19 > 1.1 x v1 1.00; - clap 0.60 < v1 0.78 - 0.05; - LPAPS/cutoff 1.39 > 1.1 x v1 1.19 |
| ui_click | ui_click | v1 | + LPAPS/cutoff 1.28 > 1.1 x v1 0.98 |
| ukulele | ukulele | v2 |  |
| wide_stereo | wide_stereo | v1 | - muq 0.36 < v1 0.46 - 0.05 |
| abs_cinematic | abs_cinematic | v1 | v2 dropped it (screening, dedupe or no survivor) |
| abs_industrial | abs_industrial | v1 | v2 dropped it (screening, dedupe or no survivor) |
| acid_303 | acid_303 | v1 | v2 dropped it (screening, dedupe or no survivor) |
| childrens | childrens | v1 | v2 dropped it (screening, dedupe or no survivor) |
| distant | distant | v1 | v2 dropped it (screening, dedupe or no survivor) |
| gritty | gritty | v1 | v2 dropped it (screening, dedupe or no survivor) |
| horror_score | horror_score | v1 | v2 dropped it (screening, dedupe or no survivor) |
| koto | koto | v1 | v2 dropped it (screening, dedupe or no survivor) |
| playful | playful | v1 | v2 dropped it (screening, dedupe or no survivor) |
| samba | samba | v1 | v2 dropped it (screening, dedupe or no survivor) |
| sensual | sensual | v1 | v2 dropped it (screening, dedupe or no survivor) |
| bright | legacy_bright | legacy_bright | v2 counterpart not evaluated |
| dense_arrangement | legacy_density | legacy_density | v2 counterpart not evaluated |
| percussive | legacy_percussive | legacy_percussive | v2 counterpart not evaluated |
| rough | legacy_rough | legacy_rough | v2 counterpart not evaluated |
| warm | legacy_warm | legacy_warm | v2 counterpart not evaluated |

## Stacking (pairs and triples at calibrated gain, LPAPS vs the strictest member's cutoff)

| rule | combos | fraction under cutoff |
|---|---|---|
| none | 300 | 0.0 |
| inv_sqrt | 300 | 0.043333333333333335 |
| inv_n | 300 | 0.35333333333333333 |

Chosen: inv_n (no rule reached 95%; the best observed is chosen). The shared-norm-budget candidate was not run.

## 30 s hold (effect = steered minus unsteered anchor margin, CLAP and MuQ mean)

| knob | 0-10 s | 20-30 s | drift | flag |
|---|---|---|---|---|
| abs_forest | 0.0284 | 0.0018 | 0.94 | DRIFT |
| accelerando | 0.1497 | 0.1380 | 0.08 |  |
| acid_house | 0.1447 | 0.1123 | 0.22 |  |
| aggressive | 0.1205 | 0.0330 | 0.73 | DRIFT |
| alarm_clock | 0.2522 | 0.2248 | 0.11 |  |
| amateur | 0.1061 | 0.1257 | 0.18 |  |
| analog | 0.0500 | -0.0044 | 1.09 | DRIFT |
| arpeggiated_r2 | 0.1258 | 0.1317 | 0.05 |  |
| bass_808 | 0.0837 | 0.0620 | 0.26 |  |
| bedroom_intimate | 0.0794 | 0.0330 | 0.58 | DRIFT |
| black_metal | 0.1228 | 0.0754 | 0.39 | DRIFT |
| bluegrass | 0.0492 | -0.0058 | 1.12 | DRIFT |
| boomy | 0.0831 | 0.0683 | 0.18 |  |
| boxy_tone | 0.0392 | 0.0418 | 0.07 |  |
| brassy_blare | 0.0702 | 0.0620 | 0.12 |  |
| breakbeat_feel | 0.1566 | 0.1208 | 0.23 |  |
| chord_stabs | 0.1058 | 0.0926 | 0.12 |  |
| christmas | 0.1369 | 0.0791 | 0.42 | DRIFT |
| clave_pattern | 0.0407 | 0.0547 | 0.34 | DRIFT |
| close_mic | 0.0255 | 0.0614 | 1.41 | DRIFT |
| crowd | 0.0683 | 0.0697 | 0.02 |  |
| cumbia | 0.1762 | 0.1124 | 0.36 | DRIFT |
| damped_strings | 0.0340 | -0.0049 | 1.14 | DRIFT |
| dark | 0.0316 | -0.0353 | 2.12 | DRIFT |
| deep_r2 | 0.0039 | -0.0318 | 9.09 | DRIFT |
| dishes | 0.2779 | 0.3095 | 0.11 |  |
| dissonant | 0.0840 | 0.0626 | 0.25 |  |
| djembe | 0.0191 | -0.0132 | 1.69 | DRIFT |
| dotted_rhythm | 0.0827 | 0.0192 | 0.77 | DRIFT |
| dreamy | 0.1123 | 0.0806 | 0.28 |  |
| driving_eighths | 0.0728 | 0.1013 | 0.39 | DRIFT |
| drone | 0.0368 | 0.0227 | 0.38 | DRIFT |
| drum_machine | 0.0670 | 0.0176 | 0.74 | DRIFT |
| dub_space_echo | 0.2212 | 0.1639 | 0.26 |  |
| echo_delay | 0.0756 | 0.0321 | 0.58 | DRIFT |
| electric_piano_r2 | 0.1429 | 0.1212 | 0.15 |  |
| enjoyment | 0.1011 | 0.1246 | 0.23 |  |
| erhu | 0.1023 | 0.0666 | 0.35 | DRIFT |
| explosion | 0.1378 | 0.1469 | 0.07 |  |
| fan | 0.0742 | 0.0943 | 0.27 |  |
| fizzy_highs | 0.1813 | 0.1556 | 0.14 |  |
| frogs | 0.0428 | 0.0654 | 0.53 | DRIFT |
| funny | 0.0709 | 0.0738 | 0.04 |  |
| future_bass | 0.0711 | 0.0173 | 0.76 | DRIFT |
| fuzzy | 0.1641 | 0.1625 | 0.01 |  |
| glass_clink | 0.1158 | 0.1403 | 0.21 |  |
| glitch_sfx | 0.2113 | 0.2033 | 0.04 |  |
| glitchy | 0.2346 | 0.2597 | 0.11 |  |
| halftime_r2 | 0.0418 | 0.0455 | 0.09 |  |
| hammered_notes | 0.1364 | 0.0843 | 0.38 | DRIFT |
| handclaps | 0.0314 | 0.0064 | 0.80 | DRIFT |
| hardstyle | 0.1363 | 0.1025 | 0.25 |  |
| hihat_rolls | -0.0139 | -0.0226 | 0.62 | DRIFT |
| honky_mids | 0.0974 | 0.0972 | 0.00 |  |
| impact_boom | 0.0387 | -0.0000 | 1.00 | DRIFT |
| intense | 0.2137 | 0.1953 | 0.09 |  |
| jet | 0.1745 | 0.1451 | 0.17 |  |
| jingle_bells | 0.3206 | 0.2584 | 0.19 |  |
| laid_back | 0.0574 | 0.0737 | 0.28 |  |
| laser_zap | 0.1657 | 0.1576 | 0.05 |  |
| legato_phrasing | 0.0611 | 0.0530 | 0.13 |  |
| live_recording | 0.0315 | 0.0347 | 0.10 |  |
| locked_groove | 0.0599 | 0.0348 | 0.42 | DRIFT |
| lofi_hip_hop | 0.1025 | 0.0906 | 0.12 |  |
| lush_r2 | 0.1357 | 0.1665 | 0.23 |  |
| magic_sparkle | 0.3279 | 0.3126 | 0.05 |  |
| mains_hum | 0.0432 | 0.0500 | 0.16 |  |
| majestic | 0.0256 | -0.0146 | 1.57 | DRIFT |
| marcato | 0.1304 | 0.1537 | 0.18 |  |
| modular_synth_r2 | 0.1186 | 0.0355 | 0.70 | DRIFT |
| muted_horn | 0.1284 | 0.1073 | 0.16 |  |
| new_age | 0.1102 | 0.0477 | 0.57 | DRIFT |
| no_percussion | 0.0063 | 0.0391 | 5.21 | DRIFT |
| noise_floor | 0.0382 | 0.0318 | 0.17 |  |
| nylon_soft | 0.0407 | 0.0417 | 0.02 |  |
| odd_meter | 0.0319 | 0.0097 | 0.70 | DRIFT |
| ostinato_r2 | 0.1154 | 0.0905 | 0.22 |  |
| palm_muted | 0.1253 | 0.0794 | 0.37 | DRIFT |
| phone_ring | 0.1944 | 0.1846 | 0.05 |  |
| pitch_bends | 0.0525 | 0.0904 | 0.72 | DRIFT |
| plate_reverb | -0.0253 | -0.0155 | 0.39 | DRIFT |
| polyrhythmic | 0.1192 | 0.0714 | 0.40 | DRIFT |
| post_rock | 0.1131 | 0.0682 | 0.40 | DRIFT |
| pulsing_synth | 0.0106 | -0.0025 | 1.23 | DRIFT |
| quantized | 0.1074 | 0.0774 | 0.28 |  |
| radio_filtered | 0.1032 | 0.0820 | 0.21 |  |
| rain_on_surface | 0.1612 | 0.1579 | 0.02 |  |
| revving | 0.1015 | 0.0715 | 0.30 |  |
| riser | 0.0865 | 0.0354 | 0.59 | DRIFT |
| sfx_applause | 0.2160 | 0.2311 | 0.07 |  |
| sfx_baby_cry | 0.0663 | 0.0672 | 0.01 |  |
| sfx_close_foley | 0.0424 | 0.0555 | 0.31 | DRIFT |
| sfx_event_rate | 0.0245 | 0.0490 | 1.00 | DRIFT |
| sfx_far_away | -0.0312 | -0.0036 | 0.89 | DRIFT |
| sfx_indoor_hall | 0.0688 | 0.0304 | 0.56 | DRIFT |
| sfx_insects | 0.2265 | 0.2543 | 0.12 |  |
| sfx_laden | -0.0040 | -0.0134 | 2.33 | DRIFT |
| sfx_level | 0.0761 | 0.0635 | 0.17 |  |
| sfx_level_motion | -0.0274 | -0.0315 | 0.15 |  |
| sfx_ocean_waves | 0.1645 | 0.1553 | 0.06 |  |
| sfx_sub_weight | 0.0900 | 0.0951 | 0.06 |  |
| sfx_telephone | 0.0552 | 0.0204 | 0.63 | DRIFT |
| shaker_pulse | 0.0625 | 0.0189 | 0.70 | DRIFT |
| sharp | 0.0823 | 0.1027 | 0.25 |  |
| shimmer_reverb | 0.3060 | 0.2177 | 0.29 |  |
| shimmering_r2 | 0.1950 | 0.2304 | 0.18 |  |
| sitar_r2 | 0.1327 | 0.1055 | 0.21 |  |
| slurred_brass | 0.0330 | 0.0434 | 0.32 | DRIFT |
| smash | 0.1401 | 0.0993 | 0.29 |  |
| solo_build | -0.0029 | 0.0089 | 4.05 | DRIFT |
| solo_crest | -0.0089 | -0.0263 | 1.95 | DRIFT |
| solo_instrument | 0.1749 | 0.1237 | 0.29 |  |
| spiccato | 0.0420 | 0.0253 | 0.40 | DRIFT |
| strummed_guitar | 0.0727 | 0.0614 | 0.16 |  |
| surf_rock | 0.2201 | 0.1354 | 0.39 | DRIFT |
| swing_r2 | 0.0696 | 0.0650 | 0.07 |  |
| synth_pad | 0.2207 | 0.1857 | 0.16 |  |
| synthesizer | 0.0819 | 0.0709 | 0.13 |  |
| tango | -0.0638 | -0.0824 | 0.29 |  |
| thick_unison | 0.0900 | 0.0436 | 0.52 | DRIFT |
| timbral_warmth_r2 | -0.0308 | -0.0234 | 0.24 |  |
| timpani | 0.0269 | -0.0184 | 1.68 | DRIFT |
| tom_patterns | 0.0799 | 0.0545 | 0.32 | DRIFT |
| tongued_flute | 0.0087 | 0.0271 | 2.11 | DRIFT |
| trills_ornaments | 0.1630 | 0.1431 | 0.12 |  |
| ui_click | 0.1121 | 0.0769 | 0.31 | DRIFT |
| ukulele | 0.2904 | 0.1792 | 0.38 | DRIFT |
| vaporwave_r2 | 0.0907 | 0.0225 | 0.75 | DRIFT |
| video_game | 0.0797 | 0.0342 | 0.57 | DRIFT |
| virtuosic_runs | 0.1229 | 0.1684 | 0.37 | DRIFT |
| wall_of_sound | 0.0847 | 0.0865 | 0.02 |  |
| wide_stereo | 0.0066 | 0.0120 | 0.82 | DRIFT |
| wide_vibrato | 0.1306 | 0.0849 | 0.35 | DRIFT |
| wind | 0.1142 | 0.1365 | 0.19 |  |
| woody_body | 0.0572 | 0.0902 | 0.58 | DRIFT |

## Cross-effect (screen_music, calibrated gain; z = mean label delta / corpus std; full matrices in crossfx_matrix_plus.csv / _minus.csv)

270 knob-signs; own label moves by more than any other knob label on 40; median |own z| 1.13, median max |off-target z| 1.84.

| knob | sign | own label | own z | largest off-target label | z | labels moved more than own |
|---|---|---|---|---|---|---|
| abs_forest | + | clap_music.abs_forest | 1.18 | clap_music.lush | 3.49 | 34 |
| abs_forest | - | clap_music.abs_forest | -1.05 | clap_music.wide_stereo | -1.47 | 5 |
| accelerando | + | clap_music.accelerando | 1.32 | passt.Mains hum | 2.23 | 9 |
| accelerando | - | clap_music.accelerando | -2.38 | clap_music.laid_back | 1.88 | 0 |
| acid_house | + | clap_music.acid_house | 1.23 | clap_music.hardstyle | 1.50 | 5 |
| acid_house | - | clap_music.acid_house | -0.71 | clap_music.tango | -0.91 | 5 |
| aggressive | + | clap_music.aggressive | 1.71 | clap_music.legato_phrasing | -2.11 | 5 |
| aggressive | - | clap_music.aggressive | -0.68 | clap_music.black_metal | -0.75 | 1 |
| alarm_clock | + | passt.Alarm clock | 0.26 | clap_music.shimmering | 2.57 | 83 |
| alarm_clock | - | passt.Alarm clock | -0.01 | clap_music.lush | 1.68 | 122 |
| amateur | + | clap_music.amateur | 2.73 | dyn.rms_db | -8.63 | 9 |
| amateur | - | clap_music.amateur | -1.34 | dyn.crest_db | -1.36 | 1 |
| analog | + | clap_music.analog | 1.54 | clap_music.fuzzy | 1.93 | 3 |
| analog | - | clap_music.analog | -0.50 | timbral.sharpness | 1.30 | 29 |
| arpeggiated_r2 | + | clap_music.arpeggiated | 1.26 | clap_music.glitchy | 1.76 | 5 |
| arpeggiated_r2 | - | clap_music.arpeggiated | -1.40 | passt.Echo | 1.16 | 0 |
| bass_808 | + | clap_music.bass_808 | 0.03 | dyn.rms_db | -4.73 | 119 |
| bass_808 | - | clap_music.bass_808 | -0.93 | clap_music.wide_stereo | -1.37 | 4 |
| bedroom_intimate | + | clap_music.bedroom_intimate | 1.73 | dyn.rms_db | -5.11 | 21 |
| bedroom_intimate | - | clap_music.bedroom_intimate | -2.28 | clap_music.fizzy_highs | 2.72 | 2 |
| black_metal | + | clap_music.black_metal | 1.78 | clap_music.pulsing_synth | -2.11 | 8 |
| black_metal | - | clap_music.black_metal | -0.77 | clap_music.spiccato | 0.65 | 0 |
| bluegrass | + | passt.Bluegrass | 0.34 | dyn.crest_db | -1.69 | 50 |
| bluegrass | - | passt.Bluegrass | -0.06 | desc.floor_db | -1.23 | 98 |
| boomy | + | timbral.boominess | 1.55 | clap_music.lush | 1.60 | 1 |
| boomy | - | timbral.boominess | -1.43 | passt.Mains hum | 1.96 | 8 |
| boxy_tone | + | clap_music.boxy_tone | 1.58 | passt.Video game music | 2.79 | 3 |
| boxy_tone | - | clap_music.boxy_tone | -1.83 | clap_music.dub_space_echo | 2.34 | 7 |
| brassy_blare | + | clap_music.brassy_blare | 0.97 | clap_music.fuzzy | 1.18 | 6 |
| brassy_blare | - | clap_music.brassy_blare | -1.41 | clap_music.solo_instrument | -1.19 | 0 |
| breakbeat_feel | + | clap_music.breakbeat_feel | 1.20 | dyn.crest_db | -1.72 | 5 |
| breakbeat_feel | - | clap_music.breakbeat_feel | -2.00 | dyn.rms_db | -3.55 | 1 |
| chord_stabs | + | clap_music.chord_stabs | 1.87 | passt.Funny music | 2.56 | 1 |
| chord_stabs | - | clap_music.chord_stabs | -1.97 | clap_music.muted_horn | 2.08 | 4 |
| christmas | + | passt.Christmas music | 2.88 | passt.Video game music | 2.01 | 0 |
| christmas | - | passt.Christmas music | -0.08 | dyn.rms_db | -1.29 | 97 |
| clave_pattern | + | clap_music.clave_pattern | 0.60 | passt.Drum machine | 1.13 | 20 |
| clave_pattern | - | clap_music.clave_pattern | -0.84 | clap_music.damped_strings | -1.21 | 4 |
| close_mic | + | clap_music.close_mic | 0.27 | clap_music.enjoyment | -1.42 | 68 |
| close_mic | - | clap_music.close_mic | -1.35 | clap_music.wide_stereo | 1.40 | 1 |
| crowd | + | passt.Crowd | 0.92 | clap_music.legato_phrasing | -1.43 | 13 |
| crowd | - | passt.Crowd | -0.03 | clap_music.muted_horn | 2.06 | 118 |
| cumbia | + | clap_music.cumbia | 2.01 | clap_music.polyrhythmic | 1.80 | 0 |
| cumbia | - | clap_music.cumbia | -0.92 | clap_music.boxy_tone | -0.98 | 1 |
| damped_strings | + | clap_music.damped_strings | 1.15 | dyn.rms_db | -2.90 | 20 |
| damped_strings | - | clap_music.damped_strings | -2.21 | clap_music.fizzy_highs | 2.11 | 0 |
| dark | + | clap_music.dark | 0.77 | passt.Echo | 1.14 | 5 |
| dark | - | clap_music.dark | -0.84 | passt.Funny music | 0.94 | 1 |
| deep_r2 | + | clap_music.deep_r2 | 0.61 | timbral.warmth | 1.48 | 41 |
| deep_r2 | - | clap_music.deep_r2 | -1.88 | clap_music.fizzy_highs | 1.75 | 0 |
| dishes | + | passt.Dishes, pots, and pans | 1.02 | passt.Chink, clink | 3.67 | 31 |
| dishes | - | passt.Dishes, pots, and pans | -0.05 | passt.New-age music | 3.35 | 112 |
| dissonant | + | clap_music.dissonant | 1.28 | clap_music.wall_of_sound | 1.50 | 8 |
| dissonant | - | clap_music.dissonant | -0.97 | passt.Echo | 1.33 | 8 |
| djembe | + | clap_music.djembe | 0.12 | clap_music.abs_forest | 1.37 | 88 |
| djembe | - | clap_music.djembe | -0.59 | clap_music.fuzzy | 1.03 | 17 |
| dotted_rhythm | + | clap_music.dotted_rhythm | 1.08 | passt.Radio | 1.73 | 7 |
| dotted_rhythm | - | clap_music.dotted_rhythm | -1.30 | clap_music.shimmering | 1.82 | 5 |
| dreamy | + | clap_music.dreamy | 1.40 | clap_music.synth_pad | 1.21 | 0 |
| dreamy | - | clap_music.dreamy | -1.09 | clap_music.lush | 1.48 | 9 |
| driving_eighths | + | clap_music.driving_eighths | 1.22 | passt.Funny music | 1.93 | 2 |
| driving_eighths | - | clap_music.driving_eighths | -1.36 | clap_music.tongued_flute | -1.75 | 10 |
| drone | + | clap_music.drone | 1.02 | clap_music.lush | 1.63 | 11 |
| drone | - | clap_music.drone | -0.38 | timbral.warmth | -1.41 | 35 |
| drum_machine | + | passt.Drum machine | 0.06 | clap_music.fizzy_highs | 2.84 | 112 |
| drum_machine | - | passt.Drum machine | -0.16 | clap_music.shimmer_reverb | 0.95 | 74 |
| dub_space_echo | + | clap_music.dub_space_echo | 2.14 | passt.Mains hum | 8.56 | 11 |
| dub_space_echo | - | clap_music.dub_space_echo | -1.09 | dyn.crest_db | -2.13 | 18 |
| echo_delay | + | passt.Echo | 0.81 | passt.New-age music | 1.29 | 11 |
| echo_delay | - | passt.Echo | -0.34 | clap_music.shimmering | 1.89 | 73 |
| electric_piano_r2 | + | clap_music.electric piano | 1.38 | passt.Christmas music | 1.55 | 1 |
| electric_piano_r2 | - | clap_music.electric piano | -1.07 | clap_music.glitchy | 1.24 | 6 |
| enjoyment | + | clap_music.enjoyment | 0.90 | passt.New-age music | 1.15 | 2 |
| enjoyment | - | clap_music.enjoyment | -1.64 | clap_music.pitch_bends | 1.40 | 0 |
| erhu | + | clap_music.erhu | 1.73 | clap_music.synth_pad | 3.28 | 15 |
| erhu | - | clap_music.erhu | -0.66 | clap_music.tango | -0.82 | 8 |
| explosion | + | passt.Explosion | 5.03 | clap_general.impact_boom | 2.64 | 0 |
| explosion | - | passt.Explosion | -0.04 | clap_music.ostinato | -1.76 | 112 |
| fan | + | passt.Mechanical fan | 0.31 | clap_music.slurred_brass | 1.95 | 85 |
| fan | - | passt.Mechanical fan | -0.00 | desc.floor_db | -1.27 | 133 |
| fizzy_highs | + | clap_music.fizzy_highs | 1.55 | timbral.warmth | -1.34 | 0 |
| fizzy_highs | - | clap_music.fizzy_highs | -0.93 | clap_music.marcato | -2.11 | 32 |
| frogs | + | passt.Frog | 0.41 | clap_music.fizzy_highs | 2.95 | 80 |
| frogs | - | passt.Frog | -0.01 | clap_music.wide_stereo | 1.19 | 124 |
| funny | + | passt.Funny music | 1.55 | passt.Video game music | 1.25 | 0 |
| funny | - | passt.Funny music | -0.32 | clap_music.boxy_tone | -0.65 | 31 |
| future_bass | + | clap_music.future_bass | 1.82 | clap_music.slurred_brass | 2.63 | 6 |
| future_bass | - | clap_music.future_bass | -0.89 | passt.Strum | 1.32 | 3 |
| fuzzy | + | clap_music.fuzzy | 1.77 | clap_music.pitch_bends | 1.85 | 2 |
| fuzzy | - | clap_music.fuzzy | 0.65 | clap_music.slurred_brass | 2.92 | 52 |
| glass_clink | + | passt.Chink, clink | 0.78 | clap_music.shimmering | 2.01 | 28 |
| glass_clink | - | passt.Chink, clink | -0.04 | clap_music.fuzzy | 1.83 | 115 |
| glitch_sfx | + | clap_general.glitch_sfx | 3.06 | clap_music.fizzy_highs | 3.10 | 1 |
| glitch_sfx | - | clap_general.glitch_sfx | -0.68 | passt.New-age music | 1.50 | 19 |
| glitchy | + | clap_music.glitchy | 2.70 | clap_music.fizzy_highs | 2.70 | 1 |
| glitchy | - | clap_music.glitchy | -1.18 | passt.Mains hum | 3.60 | 25 |
| halftime_r2 | + | clap_music.halftime | 1.45 | clap_music.cumbia | -1.62 | 4 |
| halftime_r2 | - | clap_music.halftime | -1.69 | timbral.warmth | -2.26 | 4 |
| hammered_notes | + | clap_music.hammered_notes | 1.77 | dyn.rms_db | -2.00 | 1 |
| hammered_notes | - | clap_music.hammered_notes | -2.02 | clap_music.woody_body | -2.35 | 6 |
| handclaps | + | passt.Clapping | 0.10 | clap_music.pitch_bends | 1.45 | 100 |
| handclaps | - | passt.Clapping | -0.00 | clap_music.tongued_flute | -0.83 | 128 |
| hardstyle | + | clap_music.hardstyle | 2.16 | passt.Jet engine | 3.29 | 7 |
| hardstyle | - | clap_music.hardstyle | -1.06 | clap_music.electric piano | 0.98 | 0 |
| hihat_rolls | + | clap_music.hihat_rolls | 1.45 | dyn.rms_db | -3.31 | 11 |
| hihat_rolls | - | clap_music.hihat_rolls | -1.08 | clap_music.woody_body | -1.24 | 3 |
| honky_mids | + | clap_music.honky_mids | 2.19 | clap_music.fizzy_highs | 2.39 | 3 |
| honky_mids | - | clap_music.honky_mids | -0.62 | clap_music.marcato | -1.47 | 35 |
| impact_boom | + | clap_general.impact_boom | 0.28 | dyn.crest_db | -2.88 | 86 |
| impact_boom | - | clap_general.impact_boom | -0.87 | dyn.rms_db | -1.98 | 7 |
| intense | + | clap_music.intense | 2.24 | clap_music.honky_mids | 2.10 | 0 |
| intense | - | clap_music.intense | -1.56 | clap_music.trills_ornaments | -1.79 | 5 |
| jet | + | passt.Jet engine | 1.07 | clap_music.fizzy_highs | 2.73 | 36 |
| jet | - | passt.Jet engine | -0.01 | dyn.rms_db | -2.14 | 124 |
| jingle_bells | + | passt.Jingle bell | 1.23 | passt.Christmas music | 3.09 | 10 |
| jingle_bells | - | passt.Jingle bell | -0.01 | passt.Echo | 1.18 | 114 |
| laid_back | + | clap_music.laid_back | 1.85 | clap_music.lush | 1.97 | 1 |
| laid_back | - | clap_music.laid_back | -2.03 | clap_music.fizzy_highs | 2.16 | 1 |
| laser_zap | + | clap_general.laser_zap | 1.39 | clap_music.fizzy_highs | 2.52 | 13 |
| laser_zap | - | clap_general.laser_zap | -1.83 | clap_music.lush | 3.30 | 6 |
| legato_phrasing | + | clap_music.legato_phrasing | 0.86 | clap_music.nylon_soft | 1.49 | 24 |
| legato_phrasing | - | clap_music.legato_phrasing | -2.39 | clap_music.fizzy_highs | 2.14 | 0 |
| live_recording | + | clap_music.live_recording | 1.59 | clap_music.slurred_brass | 2.47 | 6 |
| live_recording | - | clap_music.live_recording | -1.02 | clap_music.wide_stereo | -1.65 | 2 |
| locked_groove | + | clap_music.locked_groove | 2.34 | clap_music.lush | 2.85 | 4 |
| locked_groove | - | clap_music.locked_groove | -1.80 | clap_music.pitch_bends | 1.58 | 0 |
| lofi_hip_hop | + | clap_music.lofi_hip_hop | 1.26 | clap_music.lush | 3.23 | 29 |
| lofi_hip_hop | - | clap_music.lofi_hip_hop | -0.50 | dyn.momentary_std | 0.72 | 4 |
| lush_r2 | + | clap_music.lush | 2.35 | clap_music.brassy_blare | 2.90 | 3 |
| lush_r2 | - | clap_music.lush | -0.75 | dyn.rms_db | -3.15 | 23 |
| magic_sparkle | + | clap_general.magic_sparkle | 1.47 | clap_music.woody_body | -2.13 | 4 |
| magic_sparkle | - | clap_general.magic_sparkle | -1.25 | clap_music.lush | 2.24 | 18 |
| mains_hum | + | passt.Mains hum | 1.11 | passt.New-age music | 1.62 | 3 |
| mains_hum | - | passt.Mains hum | -0.02 | clap_music.synth_pad | 1.09 | 115 |
| majestic | + | clap_music.majestic | 0.07 | clap_music.intense | -0.77 | 85 |
| majestic | - | clap_music.majestic | -1.12 | clap_music.tango | -1.16 | 1 |
| marcato | + | clap_music.marcato | 1.27 | clap_music.fizzy_highs | 2.69 | 23 |
| marcato | - | clap_music.marcato | -1.99 | clap_music.lush | 2.87 | 3 |
| modular_synth_r2 | + | clap_music.modular_synth | 1.43 | clap_music.ostinato | -1.62 | 4 |
| modular_synth_r2 | - | clap_music.modular_synth | -0.87 | passt.Mains hum | 2.01 | 13 |
| muted_horn | + | clap_music.muted_horn | 2.03 | clap_music.thick_unison | -1.66 | 0 |
| muted_horn | - | clap_music.muted_horn | -1.33 | clap_music.palm_muted | -1.51 | 1 |
| new_age | + | passt.New-age music | 8.94 | clap_music.shimmer_reverb | 2.57 | 0 |
| new_age | - | passt.New-age music | -0.58 | timbral.warmth | -0.60 | 2 |
| no_percussion | + | clap_music.no_percussion | 0.63 | clap_music.virtuosic_runs | -1.05 | 19 |
| no_percussion | - | clap_music.no_percussion | -0.65 | clap_music.wide_stereo | -1.27 | 21 |
| noise_floor | + | desc.floor_db | 1.27 | dyn.crest_db | -1.97 | 3 |
| noise_floor | - | desc.floor_db | -2.35 | dyn.rms_db | -3.45 | 1 |
| nylon_soft | + | clap_music.nylon_soft | 2.62 | clap_music.chord_stabs | -2.31 | 0 |
| nylon_soft | - | clap_music.nylon_soft | -1.65 | clap_music.fizzy_highs | 2.24 | 8 |
| odd_meter | + | clap_music.odd_meter | 1.54 | passt.Jet engine | 2.95 | 17 |
| odd_meter | - | clap_music.odd_meter | -1.49 | clap_music.laid_back | 1.31 | 0 |
| ostinato_r2 | + | clap_music.ostinato | 1.86 | passt.Funny music | 2.07 | 2 |
| ostinato_r2 | - | clap_music.ostinato | -1.70 | clap_music.deep_r2 | -1.72 | 1 |
| palm_muted | + | clap_music.palm_muted | 1.77 | clap_music.fuzzy | 1.19 | 0 |
| palm_muted | - | clap_music.palm_muted | -1.86 | passt.Jet engine | 2.25 | 1 |
| phone_ring | + | passt.Telephone bell ringing | 0.32 | clap_music.damped_strings | -2.79 | 81 |
| phone_ring | - | passt.Telephone bell ringing | -0.01 | clap_music.dub_space_echo | 2.04 | 122 |
| pitch_bends | + | clap_music.pitch_bends | 2.01 | clap_music.glitchy | 1.38 | 0 |
| pitch_bends | - | clap_music.pitch_bends | -1.34 | clap_music.wide_stereo | 2.46 | 17 |
| plate_reverb | + | clap_music.plate_reverb | 1.18 | clap_music.woody_body | -3.62 | 39 |
| plate_reverb | - | clap_music.plate_reverb | -3.16 | passt.Jet engine | 6.87 | 1 |
| polyrhythmic | + | clap_music.polyrhythmic | 1.33 | clap_music.damped_strings | -1.40 | 2 |
| polyrhythmic | - | clap_music.polyrhythmic | -1.11 | clap_music.laid_back | 1.28 | 2 |
| post_rock | + | clap_music.post_rock | 1.40 | clap_music.lush | 3.40 | 25 |
| post_rock | - | clap_music.post_rock | -0.97 | clap_music.glitchy | 1.98 | 10 |
| pulsing_synth | + | clap_music.pulsing_synth | 0.04 | passt.Mains hum | 4.04 | 120 |
| pulsing_synth | - | clap_music.pulsing_synth | -1.79 | clap_music.lush | 1.55 | 0 |
| quantized | + | clap_music.quantized | 0.57 | clap_music.shimmering | 3.93 | 73 |
| quantized | - | clap_music.quantized | -1.39 | clap_music.slurred_brass | 1.44 | 4 |
| radio_filtered | + | passt.Radio | 0.32 | clap_music.amateur | -1.16 | 50 |
| radio_filtered | - | passt.Radio | -0.28 | dyn.rms_db | -2.22 | 81 |
| rain_on_surface | + | passt.Rain on surface | 3.00 | passt.Ocean | 2.70 | 0 |
| rain_on_surface | - | passt.Rain on surface | -0.03 | clap_music.boxy_tone | -1.25 | 109 |
| revving | + | passt.Accelerating, revving, vroom | 3.72 | passt.Jet engine | 4.62 | 1 |
| revving | - | passt.Accelerating, revving, vroom | -0.19 | passt.Telephone | 3.30 | 90 |
| riser | + | clap_general.riser | 0.57 | dyn.crest_db | -2.77 | 59 |
| riser | - | clap_general.riser | -0.27 | dyn.rms_db | -1.94 | 63 |
| sfx_applause | + | passt.Applause | 7.40 | passt.Crowd | 6.24 | 0 |
| sfx_applause | - | passt.Applause | -0.00 | clap_music.bass_808 | 1.47 | 133 |
| sfx_baby_cry | + | passt.Baby cry, infant cry | 1.84 | clap_music.muted_horn | 1.13 | 0 |
| sfx_baby_cry | - | passt.Baby cry, infant cry | -0.03 | passt.Jet engine | 3.52 | 120 |
| sfx_close_foley | + | clap_general.sfx_close_foley | 0.41 | clap_music.fizzy_highs | 1.86 | 72 |
| sfx_close_foley | - | clap_general.sfx_close_foley | -0.10 | timbral.sharpness | -1.40 | 98 |
| sfx_event_rate | + | dyn.onset_rate | 0.95 | clap_music.lush | 1.75 | 14 |
| sfx_event_rate | - | dyn.onset_rate | -0.29 | dyn.rms_db | -3.93 | 82 |
| sfx_far_away | + | clap_general.sfx_far_away | 0.29 | clap_music.wide_stereo | 1.39 | 59 |
| sfx_far_away | - | clap_general.sfx_far_away | -0.75 | clap_music.glitchy | 1.73 | 23 |
| sfx_indoor_hall | + | clap_general.sfx_indoor_hall | 0.56 | clap_music.fizzy_highs | 1.88 | 50 |
| sfx_indoor_hall | - | clap_general.sfx_indoor_hall | -0.69 | clap_music.lush | 1.90 | 48 |
| sfx_insects | + | passt.Insect | 9.14 | clap_music.no_percussion | 2.29 | 0 |
| sfx_insects | - | passt.Insect | -0.02 | clap_music.dark | 1.31 | 117 |
| sfx_laden | + | passt.Sound effect | 0.33 | dyn.rms_db | -1.58 | 38 |
| sfx_laden | - | passt.Sound effect | -0.07 | dyn.crest_db | -2.28 | 108 |
| sfx_level | + | dyn.rms_db | 1.51 | dyn.crest_db | -1.67 | 2 |
| sfx_level | - | dyn.rms_db | -2.58 | desc.floor_db | -1.52 | 0 |
| sfx_level_motion | + | dyn.momentary_std | 0.57 | dyn.rms_db | -5.20 | 73 |
| sfx_level_motion | - | dyn.momentary_std | -0.76 | clap_music.legato_phrasing | -1.60 | 20 |
| sfx_ocean_waves | + | passt.Ocean | 4.08 | passt.Wind | 3.77 | 0 |
| sfx_ocean_waves | - | passt.Ocean | -0.01 | dyn.rms_db | -1.69 | 122 |
| sfx_sub_weight | + | desc.sub_db | 1.07 | clap_music.lush | 2.15 | 22 |
| sfx_sub_weight | - | desc.sub_db | -0.80 | desc.floor_db | -1.34 | 8 |
| sfx_telephone | + | passt.Telephone | 0.02 | clap_music.muted_horn | 1.39 | 105 |
| sfx_telephone | - | passt.Telephone | -0.00 | clap_music.shimmering | 1.82 | 133 |
| shaker_pulse | + | clap_music.shaker_pulse | 0.79 | dyn.rms_db | -1.86 | 16 |
| shaker_pulse | - | clap_music.shaker_pulse | -1.81 | clap_music.amateur | -1.97 | 1 |
| sharp | + | timbral.sharpness | 1.25 | clap_music.shimmering | 1.31 | 1 |
| sharp | - | timbral.sharpness | -2.12 | passt.Echo | 2.64 | 1 |
| shimmer_reverb | + | clap_music.shimmer_reverb | 2.71 | clap_music.dub_space_echo | 2.71 | 0 |
| shimmer_reverb | - | clap_music.shimmer_reverb | -1.06 | clap_music.laid_back | -1.46 | 4 |
| shimmering_r2 | + | clap_music.shimmering | 1.91 | clap_music.fizzy_highs | 2.71 | 9 |
| shimmering_r2 | - | clap_music.shimmering | 0.40 | passt.Mains hum | 3.42 | 64 |
| sitar_r2 | + | clap_music.anchor:sitar | 1.98 | clap_music.woody_body | -3.48 | 9 |
| sitar_r2 | - | clap_music.anchor:sitar | -0.81 | clap_music.lush | 1.49 | 18 |
| slurred_brass | + | clap_music.slurred_brass | 2.48 | clap_music.lush | 3.08 | 2 |
| slurred_brass | - | clap_music.slurred_brass | -1.00 | passt.Strum | 1.73 | 4 |
| smash | + | passt.Smash, crash | 1.41 | clap_music.dissonant | 1.62 | 3 |
| smash | - | passt.Smash, crash | -0.07 | clap_music.muted_horn | 1.63 | 105 |
| solo_build | + | dyn.loudness_slope | 0.49 | clap_music.lush | 2.32 | 69 |
| solo_build | - | dyn.loudness_slope | -1.31 | dyn.rms_db | -2.94 | 8 |
| solo_crest | + | dyn.crest_db | 2.04 | dyn.rms_db | -4.69 | 4 |
| solo_crest | - | dyn.crest_db | -2.14 | dyn.rms_db | 1.84 | 0 |
| solo_instrument | + | clap_music.solo_instrument | 0.70 | clap_music.shimmer_reverb | 1.71 | 42 |
| solo_instrument | - | clap_music.solo_instrument | -1.45 | clap_music.glitchy | 1.62 | 4 |
| spiccato | + | clap_music.spiccato | 1.10 | dyn.rms_db | -2.12 | 9 |
| spiccato | - | clap_music.spiccato | -1.01 | passt.New-age music | 1.85 | 13 |
| strummed_guitar | + | passt.Strum | 1.57 | dyn.crest_db | -1.60 | 1 |
| strummed_guitar | - | passt.Strum | -0.31 | dyn.rms_db | -2.68 | 79 |
| surf_rock | + | clap_music.surf_rock | 2.15 | clap_music.black_metal | 2.41 | 5 |
| surf_rock | - | clap_music.surf_rock | -0.36 | desc.floor_db | -0.90 | 11 |
| swing_r2 | + | clap_music.swing | 1.32 | passt.Funny music | 2.08 | 2 |
| swing_r2 | - | clap_music.swing | -1.05 | passt.New-age music | 1.82 | 6 |
| synth_pad | + | clap_music.synth_pad | 2.17 | passt.New-age music | 2.64 | 1 |
| synth_pad | - | clap_music.synth_pad | -0.67 | passt.Funny music | 1.05 | 3 |
| synthesizer | + | passt.Synthesizer | 1.33 | clap_music.close_mic | -1.66 | 6 |
| synthesizer | - | passt.Synthesizer | -0.12 | clap_music.woody_body | -0.80 | 86 |
| tango | + | clap_music.tango | -0.39 | clap_music.shimmer_reverb | 2.19 | 67 |
| tango | - | clap_music.tango | -1.78 | clap_music.fizzy_highs | 1.61 | 0 |
| thick_unison | + | clap_music.thick_unison | 2.31 | passt.Timpani | 2.35 | 1 |
| thick_unison | - | clap_music.thick_unison | -1.58 | clap_music.ostinato | -2.00 | 2 |
| timbral_warmth_r2 | + | timbral.warmth | 1.55 | timbral.sharpness | -1.33 | 0 |
| timbral_warmth_r2 | - | timbral.warmth | -2.05 | timbral.sharpness | 1.58 | 0 |
| timpani | + | passt.Timpani | 0.46 | dyn.rms_db | -2.17 | 59 |
| timpani | - | passt.Timpani | -0.01 | clap_music.fizzy_highs | 1.62 | 123 |
| tom_patterns | + | clap_music.tom_patterns | 1.05 | clap_music.fuzzy | 1.75 | 8 |
| tom_patterns | - | clap_music.tom_patterns | -2.11 | clap_music.woody_body | -2.11 | 0 |
| tongued_flute | + | clap_music.tongued_flute | 0.93 | clap_music.woody_body | -1.90 | 26 |
| tongued_flute | - | clap_music.tongued_flute | -2.20 | passt.Mains hum | 4.42 | 2 |
| trills_ornaments | + | clap_music.trills_ornaments | 1.45 | clap_music.amateur | -1.81 | 3 |
| trills_ornaments | - | clap_music.trills_ornaments | -1.56 | dyn.rms_db | -1.92 | 1 |
| ui_click | + | clap_general.ui_click | 1.19 | clap_music.fizzy_highs | 2.81 | 29 |
| ui_click | - | clap_general.ui_click | -1.07 | clap_music.lush | 1.45 | 6 |
| ukulele | + | passt.Ukulele | 3.59 | passt.Strum | 1.92 | 0 |
| ukulele | - | passt.Ukulele | -0.01 | clap_music.legato_phrasing | -0.83 | 126 |
| vaporwave_r2 | + | clap_music.vaporwave | 1.06 | clap_music.palm_muted | -1.68 | 9 |
| vaporwave_r2 | - | clap_music.vaporwave | -0.68 | desc.floor_db | -0.87 | 1 |
| video_game | + | passt.Video game music | 2.49 | clap_music.muted_horn | -1.47 | 0 |
| video_game | - | passt.Video game music | -0.10 | desc.floor_db | -0.69 | 73 |
| virtuosic_runs | + | clap_music.virtuosic_runs | 1.32 | clap_music.fizzy_highs | 2.42 | 26 |
| virtuosic_runs | - | clap_music.virtuosic_runs | -1.59 | passt.Mains hum | 4.04 | 9 |
| wall_of_sound | + | clap_music.wall_of_sound | 0.93 | timbral.warmth | -1.41 | 9 |
| wall_of_sound | - | clap_music.wall_of_sound | -1.04 | clap_music.marcato | -2.11 | 24 |
| wide_stereo | + | clap_music.wide_stereo | 0.72 | clap_music.chord_stabs | -1.10 | 18 |
| wide_stereo | - | clap_music.wide_stereo | -2.20 | clap_music.glitchy | 2.00 | 0 |
| wide_vibrato | + | clap_music.wide_vibrato | 2.16 | clap_music.dub_space_echo | 2.47 | 2 |
| wide_vibrato | - | clap_music.wide_vibrato | -1.07 | passt.Bluegrass | 2.45 | 14 |
| wind | + | passt.Wind | 1.15 | clap_music.slurred_brass | 1.67 | 5 |
| wind | - | passt.Wind | -0.01 | desc.floor_db | -1.09 | 123 |
| woody_body | + | clap_music.woody_body | 1.29 | clap_music.palm_muted | 1.63 | 7 |
| woody_body | - | clap_music.woody_body | -2.55 | timbral.warmth | -2.02 | 0 |

## Closed loop (P6 prototype: per-prompt alpha rescaled between renders toward the open loop's mean predicted effect, 3 passes, ridge readout)

| knob | label | R^2 | readout | delta open (sd) | delta closed (sd) | reached open | reached closed | LPAPS open | LPAPS closed |
|---|---|---|---|---|---|---|---|---|---|
| boxy_tone | clap_music.boxy_tone | 0.56 | b11 step 7 | 0.1393 (0.1110) | 0.1377 (0.0751) | 0.67 | 0.67 | 4.49 | 4.81 |
| brassy_blare | clap_music.brassy_blare | 0.60 | b12 step 7 | 0.1046 (0.0654) | 0.0987 (0.0695) | 0.75 | 0.83 | 4.61 | 4.61 |
| glitch_sfx | clap_general.glitch_sfx | 0.63 | b12 step 7 | 0.3357 (0.1450) | 0.3364 (0.1776) | 0.83 | 0.83 | 6.94 | 6.94 |
| honky_mids | clap_music.honky_mids | 0.58 | b11 step 7 | 0.2386 (0.0877) | 0.1719 (0.0854) | 0.83 | 0.75 | 5.41 | 5.60 |
| laser_zap | clap_general.laser_zap | 0.67 | b12 step 7 | 0.3244 (0.1719) | 0.3076 (0.1400) | 0.75 | 0.92 | 6.32 | 6.30 |
| magic_sparkle | clap_general.magic_sparkle | 0.72 | b12 step 7 | 0.2755 (0.1059) | 0.2574 (0.0944) | 0.92 | 0.92 | 5.24 | 5.74 |
| revving | passt.Accelerating, revving, vroom | 0.60 | b13 step 7 | 0.0537 (0.1008) | 0.0723 (0.0473) | 0.83 | 1.00 | 5.75 | 5.78 |
| riser | clap_general.riser | 0.66 | b13 step 7 | 0.2213 (0.1336) | 0.2117 (0.1965) | 0.83 | 0.67 | 6.89 | 7.04 |
| sfx_applause | passt.Applause | 0.63 | b12 step 7 | 0.2750 (0.2062) | 0.2146 (0.1874) | 0.67 | 0.50 | 6.23 | 6.14 |
| ui_click | clap_general.ui_click | 0.64 | b11 step 7 | 0.2631 (0.1984) | 0.1825 (0.1430) | 0.75 | 0.67 | 6.44 | 5.63 |

## Notes

- Residualised descriptors were not done (not a one-line change in build_directions).
- Quality bar: intended-sign fraction >= 0.6 on CLAP and on MuQ at calibrated gain, seed 1, and the PCI cutoff reached on at least one sign. MuQ is blind to sound effects, so SFX knobs mostly fail the MuQ half.
- Dedupe ran on variant (a); variants b, c and d inherit the keepers by base concept.
- The variant was chosen at screening (slope per LPAPS, two-family rule, ties within 10% to a), not after P2.
- Applicability: applies_to = categories with CLAP and MuQ intended-sign fraction >= 0.6 averaged over both signs (sfx: CLAP only); when none pass, the catalogue screen set is used.
- Calibrated gain = median over the seeds done (spread in brackets). c packs: the neg gain is in the up vector's magnitude units (the reader uses one magnitude for both rows).
