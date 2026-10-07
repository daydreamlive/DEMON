# Many-knobs results (SA3 medium, box 51480126, 2026-10-05)

Pipeline: 5000-clip capture -> 7 scorers (2569 label columns) -> 480 directions -> 136 dedupe keepers -> screening (46 accepted) -> full protocol for all 46 (probe, PCI-all, calibrate, 15-point sweep, protocol --catalogue, auc) -> packs rewritten with the phase 6 gain -> cross-effect renders -> seed-2 rerun for the top 10.
Local copies: E:\Projects\DEMON\steering-bench\many_knobs\ (packs\, listen\ with index.csv, corpus\, eval\ with results\, results_s2\, crossfx\ and the CSV tables). Status log: status_many_knobs.md.

## Caveats
- Gate 0 (label validity) is not in the code, so screening ran without it.
- The 50 TADA test prompts are all music, so the music/sfx/hybrid split comes only from the 12 cross-effect prompts (6 TADA holdout music, 3 sfx holdout, 3 corpus hybrid, since there is no hybrid holdout set).
- The phase 6 alignment scorers are CLAP and MuQ against the pos anchor. For the 13 knobs whose primary is a PaSST column, the primary scorer is not in the protocol, so their table 1 primary ratio is n/a and they were ranked on CLAP. The own-primary fractions (frac_own_*) come from the cross-effect renders, which scored with the full label set.
- A ratio is steer alignment divided by PCI alignment at the cutoff. When the PCI value is near 0, the ratio blows up or flips sign (for example mains_hum 50 at seed 1 and 8 at seed 2). Read ratios together with steer_* and pci_*.
- Calibration: the probe grid is powers of 2 up to 64, so the phase 6 gain is the first probe at or above the PCI cutoff.
- The block mini-sweep (phase 6 pass 3) has no command, so it was not run.
- Failures: none final. 13 knobs failed at probe from an OOM with 3 drivers per GPU (a generator holds about 15 GB). They were requeued and all 46 completed.

## Calibration: screening gain vs phase 6 gain (knob value at the PCI cutoff)

Overall, phase 6 / screening median 1.39 (range 0.41 to 2.38). Cutoff reached on 80/92 knob-signs.

**Calibration warnings (rain_on_surface type).** These 12 knob-signs never reached the PCI cutoff within the probe grid. Their gain is the largest probe (alpha 64), so it is a lower bound: acid_303 +, alarm_clock +, analog -, close_mic -, impact_boom +, lofi_hip_hop +, phone_ring +, riser +, synth_pad +, synthesizer +, tango +, ui_click +. (rain_on_surface itself reached it at 64 on both signs.)

| knob | sign | screening gain | phase 6 gain | ratio | alpha | PCI cutoff | reached |
|---|---|---|---|---|---|---|---|
| abs_cinematic | + | 32.32 | 46.23 | 1.43 | 64.00 | 3.87 | True |
| abs_cinematic | - | 32.32 | 23.12 | 0.72 | -32.00 | 3.26 | True |
| abs_industrial | + | 46.91 | 66.15 | 1.41 | 64.00 | 3.91 | True |
| abs_industrial | - | 46.91 | 33.08 | 0.71 | -32.00 | 3.26 | True |
| acid_303 | + | 36.71 | 55.56 | 1.51 | 64.00 | 4.50 | False |
| acid_303 | - | 36.71 | 55.56 | 1.51 | -64.00 | 3.26 | True |
| aggressive | + | 32.70 | 55.56 | 1.70 | 64.00 | 3.69 | True |
| aggressive | - | 32.70 | 55.56 | 1.70 | -64.00 | 3.26 | True |
| alarm_clock | + | 67.35 | 72.28 | 1.07 | 64.00 | 4.14 | False |
| alarm_clock | - | 67.35 | 72.28 | 1.07 | -64.00 | 3.68 | True |
| analog | + | 35.25 | 49.41 | 1.40 | 64.00 | 4.17 | True |
| analog | - | 35.25 | 49.41 | 1.40 | -64.00 | 4.19 | False |
| black_metal | + | 33.39 | 59.42 | 1.78 | 64.00 | 4.38 | True |
| black_metal | - | 33.39 | 59.42 | 1.78 | -64.00 | 3.26 | True |
| childrens | + | 56.08 | 45.78 | 0.82 | 64.00 | 3.75 | True |
| childrens | - | 56.08 | 22.89 | 0.41 | -32.00 | 3.26 | True |
| close_mic | + | 47.95 | 59.21 | 1.23 | 64.00 | 4.04 | True |
| close_mic | - | 47.95 | 59.21 | 1.23 | -64.00 | 4.00 | False |
| dissonant | + | 35.35 | 53.03 | 1.50 | 64.00 | 3.82 | True |
| dissonant | - | 35.35 | 53.03 | 1.50 | -64.00 | 4.01 | True |
| distant | + | 26.51 | 57.50 | 2.17 | 64.00 | 4.16 | True |
| distant | - | 26.51 | 28.75 | 1.08 | -32.00 | 3.88 | True |
| dreamy | + | 27.43 | 40.93 | 1.49 | 64.00 | 3.63 | True |
| dreamy | - | 27.43 | 40.93 | 1.49 | -64.00 | 3.26 | True |
| drone | + | 32.47 | 46.46 | 1.43 | 64.00 | 3.74 | True |
| drone | - | 32.47 | 46.46 | 1.43 | -64.00 | 3.26 | True |
| future_bass | + | 34.39 | 47.86 | 1.39 | 64.00 | 4.03 | True |
| future_bass | - | 34.39 | 47.86 | 1.39 | -64.00 | 3.26 | True |
| glitchy | + | 29.80 | 41.97 | 1.41 | 64.00 | 4.19 | True |
| glitchy | - | 29.80 | 41.97 | 1.41 | -64.00 | 4.15 | True |
| gritty | + | 37.13 | 57.08 | 1.54 | 64.00 | 3.88 | True |
| gritty | - | 37.13 | 57.08 | 1.54 | -64.00 | 4.11 | True |
| hardstyle | + | 34.80 | 45.89 | 1.32 | 64.00 | 4.04 | True |
| hardstyle | - | 34.80 | 45.89 | 1.32 | -64.00 | 3.26 | True |
| horror_score | + | 20.28 | 48.32 | 2.38 | 64.00 | 4.06 | True |
| horror_score | - | 20.28 | 48.32 | 2.38 | -64.00 | 3.26 | True |
| impact_boom | + | 40.00 | 57.37 | 1.43 | 64.00 | 4.45 | False |
| impact_boom | - | 40.00 | 57.37 | 1.43 | -64.00 | 3.68 | True |
| intense | + | 29.74 | 29.94 | 1.01 | 32.00 | 3.60 | True |
| intense | - | 29.74 | 59.89 | 2.01 | -64.00 | 4.00 | True |
| jet | + | 26.88 | 56.29 | 2.09 | 64.00 | 4.11 | True |
| jet | - | 26.88 | 56.29 | 2.09 | -64.00 | 3.68 | True |
| jingle_bells | + | 36.84 | 67.67 | 1.84 | 64.00 | 4.27 | True |
| jingle_bells | - | 36.84 | 33.83 | 0.92 | -32.00 | 3.26 | True |
| koto | + | 46.16 | 53.01 | 1.15 | 64.00 | 3.94 | True |
| koto | - | 46.16 | 53.01 | 1.15 | -64.00 | 3.26 | True |
| live_recording | + | 47.31 | 68.88 | 1.46 | 64.00 | 4.28 | True |
| live_recording | - | 47.31 | 68.88 | 1.46 | -64.00 | 3.69 | True |
| lofi_hip_hop | + | 57.29 | 65.78 | 1.15 | 64.00 | 4.45 | False |
| lofi_hip_hop | - | 57.29 | 32.89 | 0.57 | -32.00 | 3.26 | True |
| mains_hum | + | 87.24 | 112.51 | 1.29 | 64.00 | 4.07 | True |
| mains_hum | - | 87.24 | 112.51 | 1.29 | -64.00 | 3.89 | True |
| phone_ring | + | 46.19 | 67.72 | 1.47 | 64.00 | 4.48 | False |
| phone_ring | - | 46.19 | 67.72 | 1.47 | -64.00 | 3.68 | True |
| playful | + | 33.36 | 51.27 | 1.54 | 64.00 | 3.69 | True |
| playful | - | 33.36 | 25.64 | 0.77 | -32.00 | 3.26 | True |
| polyrhythmic | + | 52.03 | 70.16 | 1.35 | 64.00 | 4.00 | True |
| polyrhythmic | - | 52.03 | 70.16 | 1.35 | -64.00 | 3.66 | True |
| post_rock | + | 34.67 | 55.68 | 1.61 | 64.00 | 4.25 | True |
| post_rock | - | 34.67 | 27.84 | 0.80 | -32.00 | 3.26 | True |
| radio_filtered | + | 41.97 | 68.80 | 1.64 | 64.00 | 3.99 | True |
| radio_filtered | - | 41.97 | 68.80 | 1.64 | -64.00 | 3.89 | True |
| rain_on_surface | + | 20.31 | 42.91 | 2.11 | 64.00 | 4.54 | True |
| rain_on_surface | - | 20.31 | 42.91 | 2.11 | -64.00 | 3.68 | True |
| riser | + | 43.32 | 58.86 | 1.36 | 64.00 | 4.53 | False |
| riser | - | 43.32 | 58.86 | 1.36 | -64.00 | 3.68 | True |
| samba | + | 45.40 | 52.41 | 1.15 | 64.00 | 4.10 | True |
| samba | - | 45.40 | 26.20 | 0.58 | -32.00 | 3.26 | True |
| sensual | + | 39.05 | 48.36 | 1.24 | 64.00 | 3.60 | True |
| sensual | - | 39.05 | 24.18 | 0.62 | -32.00 | 3.26 | True |
| sfx_laden | + | 35.26 | 32.59 | 0.92 | 32.00 | 3.76 | True |
| sfx_laden | - | 35.26 | 65.19 | 1.85 | -64.00 | 3.89 | True |
| solo_instrument | + | 30.13 | 57.19 | 1.90 | 64.00 | 4.07 | True |
| solo_instrument | - | 30.13 | 57.19 | 1.90 | -64.00 | 3.36 | True |
| strummed_guitar | + | 36.82 | 43.35 | 1.18 | 64.00 | 3.96 | True |
| strummed_guitar | - | 36.82 | 43.35 | 1.18 | -64.00 | 3.26 | True |
| surf_rock | + | 34.80 | 47.54 | 1.37 | 64.00 | 3.97 | True |
| surf_rock | - | 34.80 | 23.77 | 0.68 | -32.00 | 3.26 | True |
| synth_pad | + | 36.04 | 41.16 | 1.14 | 64.00 | 4.11 | False |
| synth_pad | - | 36.04 | 41.16 | 1.14 | -64.00 | 3.26 | True |
| synthesizer | + | 46.17 | 47.81 | 1.04 | 64.00 | 4.16 | False |
| synthesizer | - | 46.17 | 47.81 | 1.04 | -64.00 | 3.26 | True |
| tango | + | 44.13 | 40.80 | 0.92 | 64.00 | 4.11 | False |
| tango | - | 44.13 | 40.80 | 0.92 | -64.00 | 3.26 | True |
| timpani | + | 38.90 | 47.80 | 1.23 | 64.00 | 4.03 | True |
| timpani | - | 38.90 | 47.80 | 1.23 | -64.00 | 3.26 | True |
| ui_click | + | 40.10 | 63.54 | 1.58 | 64.00 | 4.32 | False |
| ui_click | - | 40.10 | 63.54 | 1.58 | -64.00 | 3.68 | True |
| ukulele | + | 48.17 | 60.10 | 1.25 | 64.00 | 4.08 | True |
| ukulele | - | 48.17 | 30.05 | 0.62 | -32.00 | 3.26 | True |
| wide_stereo | + | 26.13 | 27.84 | 1.07 | 32.00 | 3.91 | True |
| wide_stereo | - | 26.13 | 55.68 | 2.13 | -64.00 | 4.30 | True |

## Table 1: per knob and sign, fraction of prompts moving the intended way at the calibrated gain, and steer/PCI ratios

frac_clap_50 and frac_muq_50: the 50 TADA test prompts, pos anchor delta vs alpha 0, intended sign. frac_own_*: the own primary label on the 12 cross-effect prompts (music 6, sfx 3, hybrid 3). ratio_primary is the CLAP ratio when the primary is CLAP, else n/a.

Medians: frac_clap_50 0.84, frac_muq_50 0.70, frac_own_12 0.83 (music 1.00, sfx 1.00, hybrid 1.00).

| knob | sign | primary | gain (alpha) | frac CLAP 50 | frac MuQ 50 | frac own 12 | music | sfx | hybrid | ratio primary | ratio CLAP | ratio MuQ | cutoff |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| rain_on_surface | + | passt.Rain on surface | 64.00 | 0.96 | 0.68 | 1.00 | 1.00 | 1.00 | 1.00 |  | 6.72 | -0.03 | 4.54 |
| rain_on_surface | - | passt.Rain on surface | -64.00 | 0.54 | 0.16 | 0.83 | 0.83 | 0.67 | 1.00 |  | 2.10 | 3.27 | 3.68 |
| glitchy | + | clap_music.glitchy | 64.00 | 1.00 | 0.98 | 1.00 | 1.00 | 1.00 | 1.00 | 2.74 | 2.74 | 1.38 | 4.19 |
| glitchy | - | clap_music.glitchy | -64.00 | 0.82 | 0.42 | 0.83 | 0.83 | 0.67 | 1.00 | 14.48 | 14.48 | -2.08 | 4.15 |
| radio_filtered | + | passt.Radio | 64.00 | 0.96 | 0.80 | 0.83 | 0.83 | 1.00 | 0.67 |  | -8.42 | 1.15 | 3.99 |
| radio_filtered | - | passt.Radio | -64.00 | 0.68 | 0.76 | 0.75 | 0.83 | 0.33 | 1.00 |  | -10.24 | 10.82 | 3.89 |
| childrens | + | passt.Music for children | 64.00 | 0.72 | 0.72 | 1.00 | 1.00 | 1.00 | 1.00 |  | 1.14 | 0.68 | 3.73 |
| childrens | - | passt.Music for children | -32.00 | 0.90 | 0.58 | 0.75 | 0.67 | 0.67 | 1.00 |  | -8.53 | -0.26 | 3.26 |
| distant | + | clap_music.distant | 64.00 | 0.80 | 0.84 | 1.00 | 1.00 | 1.00 | 1.00 | 0.60 | 0.60 | 1.70 | 4.16 |
| distant | - | clap_music.distant | -32.00 | 0.54 | 0.44 | 0.75 | 0.67 | 1.00 | 0.67 | 2.88 | 2.88 | -0.22 | 3.88 |
| hardstyle | + | clap_music.hardstyle | 64.00 | 0.82 | 0.76 | 0.83 | 1.00 | 0.33 | 1.00 | 4.20 | 4.20 | 2.07 | 4.04 |
| hardstyle | - | clap_music.hardstyle | -64.00 | 0.98 | 0.84 | 0.83 | 0.83 | 0.67 | 1.00 | 4.46 | 4.46 | -13.34 | 3.26 |
| strummed_guitar | + | passt.Strum | 64.00 | 0.90 | 0.94 | 1.00 | 1.00 | 1.00 | 1.00 |  | 4.01 | 1.37 | 3.96 |
| strummed_guitar | - | passt.Strum | -64.00 | 0.88 | 0.28 | 0.75 | 0.83 | 0.33 | 1.00 |  | 8.11 | -1.27 | 3.26 |
| gritty | + | clap_music.gritty | 64.00 | 0.96 | 0.96 | 1.00 | 1.00 | 1.00 | 1.00 | 7.01 | 7.01 | 1.30 | 3.88 |
| gritty | - | clap_music.gritty | -64.00 | 0.88 | 0.90 | 0.83 | 0.67 | 1.00 | 1.00 | 2.60 | 2.60 | 5.45 | 4.11 |
| sensual | + | clap_music.sensual | 64.00 | 0.56 | 0.80 | 1.00 | 1.00 | 1.00 | 1.00 | 0.37 | 0.37 | 1.22 | 3.60 |
| sensual | - | clap_music.sensual | -32.00 | 0.52 | 0.72 | 0.75 | 0.83 | 0.33 | 1.00 | 2.93 | 2.93 | -3.44 | 3.26 |
| aggressive | + | clap_music.aggressive | 64.00 | 0.86 | 0.80 | 0.83 | 1.00 | 0.33 | 1.00 | 2.48 | 2.48 | 1.92 | 3.69 |
| aggressive | - | clap_music.aggressive | -64.00 | 0.98 | 0.82 | 0.75 | 0.67 | 0.67 | 1.00 | 4.24 | 4.24 | -5.98 | 3.26 |
| drone | + | clap_music.drone | 64.00 | 0.92 | 0.94 | 1.00 | 1.00 | 1.00 | 1.00 | 3.01 | 3.01 | 3.49 | 3.74 |
| drone | - | clap_music.drone | -64.00 | 0.70 | 0.70 | 0.75 | 0.83 | 1.00 | 0.33 | 5.07 | 5.07 | 42.41 | 3.26 |
| synth_pad | + | clap_music.synth_pad | 64.00 | 1.00 | 0.94 | 1.00 | 1.00 | 1.00 | 1.00 | 1.68 | 1.68 | 1.30 | 4.11 |
| synth_pad | - | clap_music.synth_pad | -64.00 | 0.80 | 0.60 | 0.83 | 0.83 | 1.00 | 0.67 | 8.41 | 8.41 | 3.98 | 3.26 |
| close_mic | + | clap_music.close_mic | 64.00 | 0.36 | 0.74 | 0.92 | 0.83 | 1.00 | 1.00 | 0.28 | 0.28 | -7.08 | 4.04 |
| close_mic | - | clap_music.close_mic | -64.00 | 0.26 | 0.68 | 0.92 | 0.83 | 1.00 | 1.00 | 1.74 | 1.74 | -3.43 | 4.00 |
| tango | + | clap_music.tango | 64.00 | 0.66 | 0.70 | 0.92 | 0.83 | 1.00 | 1.00 | 0.89 | 0.89 | 0.34 | 4.11 |
| tango | - | clap_music.tango | -64.00 | 0.80 | 0.54 | 0.83 | 1.00 | 0.67 | 0.67 | -11.06 | -11.06 | -1.38 | 3.26 |
| ukulele | + | passt.Ukulele | 64.00 | 0.92 | 0.90 | 1.00 | 1.00 | 1.00 | 1.00 |  | 1.40 | 0.71 | 4.08 |
| ukulele | - | passt.Ukulele | -32.00 | 0.78 | 0.60 | 0.83 | 1.00 | 0.67 | 0.67 |  | 7.40 | 2.41 | 3.26 |
| playful | + | clap_music.playful | 64.00 | 0.60 | 0.70 | 1.00 | 1.00 | 1.00 | 1.00 | -10.92 | -10.92 | 0.62 | 3.69 |
| playful | - | clap_music.playful | -32.00 | 0.90 | 0.64 | 0.67 | 0.67 | 0.67 | 0.67 | -13.28 | -13.28 | 0.14 | 3.26 |
| dissonant | + | clap_music.dissonant | 64.00 | 0.86 | 0.68 | 0.83 | 1.00 | 0.33 | 1.00 | 4.53 | 4.53 | 19.29 | 3.82 |
| dissonant | - | clap_music.dissonant | -64.00 | 0.92 | 0.52 | 1.00 | 1.00 | 1.00 | 1.00 | 3.39 | 3.39 | 7.75 | 4.01 |
| surf_rock | + | clap_music.surf_rock | 64.00 | 0.72 | 0.82 | 1.00 | 1.00 | 1.00 | 1.00 | 0.96 | 0.96 | 0.37 | 3.97 |
| surf_rock | - | clap_music.surf_rock | -32.00 | 0.80 | 0.66 | 0.58 | 0.50 | 0.33 | 1.00 | 4.16 | 4.16 | -2.61 | 3.26 |
| intense | + | clap_music.intense | 32.00 | 0.94 | 0.88 | 1.00 | 1.00 | 1.00 | 1.00 | 2.04 | 2.04 | 1.71 | 3.60 |
| intense | - | clap_music.intense | -64.00 | 0.96 | 0.92 | 0.92 | 0.83 | 1.00 | 1.00 | -75.80 | -75.80 | 2.52 | 4.00 |
| post_rock | + | clap_music.post_rock | 64.00 | 1.00 | 0.94 | 1.00 | 1.00 | 1.00 | 1.00 | 2.10 | 2.10 | 1.29 | 4.25 |
| post_rock | - | clap_music.post_rock | -32.00 | 0.84 | 0.84 | 0.58 | 0.67 | 0.00 | 1.00 | -306848.31 | -306848.31 | -8.87 | 3.26 |
| wide_stereo | + | clap_music.wide_stereo | 32.00 | 0.60 | 0.40 | 0.83 | 0.83 | 0.67 | 1.00 | -3.44 | -3.44 | -4.88 | 3.91 |
| wide_stereo | - | clap_music.wide_stereo | -64.00 | 0.28 | 0.46 | 1.00 | 1.00 | 1.00 | 1.00 | 0.83 | 0.83 | -1.18 | 4.30 |
| live_recording | + | clap_music.live_recording | 64.00 | 0.26 | 0.34 | 0.83 | 1.00 | 0.33 | 1.00 | -1.65 | -1.65 | -0.13 | 4.28 |
| live_recording | - | clap_music.live_recording | -64.00 | 0.54 | 0.26 | 0.92 | 1.00 | 1.00 | 0.67 | 22.45 | 22.45 | 1.36 | 3.69 |
| polyrhythmic | + | clap_music.polyrhythmic | 64.00 | 0.66 | 0.80 | 0.75 | 0.83 | 0.33 | 1.00 | 1.17 | 1.17 | 0.96 | 4.00 |
| polyrhythmic | - | clap_music.polyrhythmic | -64.00 | 0.66 | 0.46 | 1.00 | 1.00 | 1.00 | 1.00 | 0.91 | 0.91 | -0.51 | 3.66 |
| samba | + | clap_music.samba | 64.00 | 0.94 | 0.84 | 1.00 | 1.00 | 1.00 | 1.00 | 1.58 | 1.58 | 0.42 | 4.10 |
| samba | - | clap_music.samba | -32.00 | 0.86 | 0.48 | 0.83 | 0.83 | 0.67 | 1.00 | -35.96 | -35.96 | -0.21 | 3.26 |
| koto | + | clap_music.koto | 64.00 | 0.46 | 0.42 | 1.00 | 1.00 | 1.00 | 1.00 | 0.09 | 0.09 | 2.76 | 3.94 |
| koto | - | clap_music.koto | -64.00 | 0.96 | 0.62 | 0.92 | 1.00 | 0.67 | 1.00 | 65.13 | 65.13 | 6.06 | 3.26 |
| solo_instrument | + | clap_music.solo_instrument | 64.00 | 0.82 | 0.86 | 0.83 | 0.67 | 1.00 | 1.00 | 2.16 | 2.16 | 1.43 | 4.07 |
| solo_instrument | - | clap_music.solo_instrument | -64.00 | 0.72 | 0.58 | 0.92 | 1.00 | 1.00 | 0.67 | 0.17 | 0.17 | 2.29 | 3.36 |
| riser | + | clap_general.riser | 64.00 | 0.66 | 0.76 | 1.00 | 1.00 | 1.00 | 1.00 | 0.70 | 0.70 | 0.21 | 4.53 |
| riser | - | clap_general.riser | -64.00 | 0.56 | 0.36 | 0.92 | 1.00 | 1.00 | 0.67 | 11.90 | 11.90 | -2.12 | 3.68 |
| dreamy | + | clap_music.dreamy | 64.00 | 0.98 | 0.68 | 0.92 | 0.83 | 1.00 | 1.00 | 2.16 | 2.16 | 1.63 | 3.63 |
| dreamy | - | clap_music.dreamy | -64.00 | 0.98 | 0.56 | 0.83 | 1.00 | 0.33 | 1.00 | -232.78 | -232.78 | 5.10 | 3.26 |
| black_metal | + | clap_music.black_metal | 64.00 | 1.00 | 0.92 | 1.00 | 1.00 | 1.00 | 1.00 | 1.25 | 1.25 | 0.76 | 4.38 |
| black_metal | - | clap_music.black_metal | -64.00 | 0.92 | 0.70 | 0.75 | 0.83 | 0.33 | 1.00 | -611.25 | -611.25 | -3.45 | 3.26 |
| analog | + | clap_music.analog | 64.00 | 0.70 | 0.60 | 1.00 | 1.00 | 1.00 | 1.00 | 1.98 | 1.98 | -0.18 | 4.17 |
| analog | - | clap_music.analog | -64.00 | 0.84 | 0.44 | 0.83 | 0.83 | 1.00 | 0.67 | -4.99 | -4.99 | -0.93 | 4.19 |
| horror_score | + | clap_music.horror_score | 64.00 | 1.00 | 0.88 | 1.00 | 1.00 | 1.00 | 1.00 | 2.07 | 2.07 | 0.64 | 4.06 |
| horror_score | - | clap_music.horror_score | -64.00 | 0.50 | 0.56 | 0.67 | 0.83 | 0.67 | 0.33 | -12.75 | -12.75 | -4.60 | 3.26 |
| lofi_hip_hop | + | clap_music.lofi_hip_hop | 64.00 | 0.62 | 0.84 | 0.83 | 0.67 | 1.00 | 1.00 | 0.23 | 0.23 | 0.41 | 4.45 |
| lofi_hip_hop | - | clap_music.lofi_hip_hop | -32.00 | 0.62 | 0.70 | 0.67 | 0.67 | 0.33 | 1.00 | -3.03 | -3.03 | -40.19 | 3.26 |
| ui_click | + | clap_general.ui_click | 64.00 | 0.72 | 0.82 | 0.83 | 0.67 | 1.00 | 1.00 | 5.14 | 5.14 | 1.23 | 4.32 |
| ui_click | - | clap_general.ui_click | -64.00 | 0.84 | 0.80 | 1.00 | 1.00 | 1.00 | 1.00 | 1.88 | 1.88 | 16.82 | 3.68 |
| impact_boom | + | clap_general.impact_boom | 64.00 | 0.82 | 0.94 | 1.00 | 1.00 | 1.00 | 1.00 | 1.12 | 1.12 | 0.61 | 4.45 |
| impact_boom | - | clap_general.impact_boom | -64.00 | 0.60 | 0.36 | 0.83 | 1.00 | 1.00 | 0.33 | 4.99 | 4.99 | -0.19 | 3.68 |
| abs_industrial | + | clap_music.abs_industrial | 64.00 | 0.94 | 0.84 | 0.92 | 1.00 | 0.67 | 1.00 | 2.28 | 2.28 | 1.26 | 3.91 |
| abs_industrial | - | clap_music.abs_industrial | -32.00 | 0.60 | 0.62 | 0.33 | 0.50 | 0.00 | 0.33 | -16.72 | -16.72 | -1.28 | 3.26 |
| jet | + | passt.Jet engine | 64.00 | 0.98 | 0.78 | 1.00 | 1.00 | 1.00 | 1.00 |  | 12.39 | 0.60 | 4.11 |
| jet | - | passt.Jet engine | -64.00 | 0.42 | 0.14 | 0.42 | 0.33 | 0.33 | 0.67 |  | -24.08 | 2.36 | 3.68 |
| mains_hum | + | passt.Mains hum | 64.00 | 0.90 | 0.84 | 0.83 | 0.83 | 0.67 | 1.00 |  | 50.01 | 1.80 | 4.07 |
| mains_hum | - | passt.Mains hum | -64.00 | 0.88 | 0.38 | 0.75 | 0.83 | 0.33 | 1.00 |  | -48.29 | 0.38 | 3.89 |
| abs_cinematic | + | clap_music.abs_cinematic | 64.00 | 0.94 | 0.80 | 1.00 | 1.00 | 1.00 | 1.00 | 5.75 | 5.75 | 1.33 | 3.87 |
| abs_cinematic | - | clap_music.abs_cinematic | -32.00 | 0.72 | 0.46 | 0.67 | 1.00 | 0.33 | 0.33 | 17.12 | 17.12 | -1.24 | 3.26 |
| synthesizer | + | passt.Synthesizer | 64.00 | 0.64 | 0.74 | 0.83 | 1.00 | 0.67 | 0.67 |  | 3.24 | 2.79 | 4.16 |
| synthesizer | - | passt.Synthesizer | -64.00 | 0.96 | 0.66 | 0.92 | 1.00 | 0.67 | 1.00 |  | 3.76 | 1.14 | 3.26 |
| future_bass | + | clap_music.future_bass | 64.00 | 0.92 | 0.72 | 0.92 | 1.00 | 0.67 | 1.00 | 1.50 | 1.50 | 1.33 | 4.03 |
| future_bass | - | clap_music.future_bass | -64.00 | 0.86 | 0.64 | 0.83 | 0.83 | 0.67 | 1.00 | 62.20 | 62.20 | 14.10 | 3.26 |
| acid_303 | + | clap_music.acid_303 | 64.00 | 0.84 | 0.82 | 0.92 | 1.00 | 0.67 | 1.00 | 3.04 | 3.04 | 0.63 | 4.50 |
| acid_303 | - | clap_music.acid_303 | -64.00 | 0.94 | 0.56 | 0.83 | 0.83 | 0.67 | 1.00 | 3.38 | 3.38 | -3.52 | 3.26 |
| sfx_laden | + | passt.Sound effect | 32.00 | 0.32 | 0.62 | 0.75 | 0.83 | 0.33 | 1.00 |  | -9.03 | 0.46 | 3.76 |
| sfx_laden | - | passt.Sound effect | -64.00 | 0.38 | 0.64 | 1.00 | 1.00 | 1.00 | 1.00 |  | -1.66 | 0.69 | 3.89 |
| jingle_bells | + | passt.Jingle bell | 64.00 | 0.98 | 0.98 | 1.00 | 1.00 | 1.00 | 1.00 |  | 0.91 | 0.37 | 4.27 |
| jingle_bells | - | passt.Jingle bell | -32.00 | 0.90 | 0.52 | 0.58 | 0.83 | 0.33 | 0.33 |  | -2.79 | -1.60 | 3.26 |
| timpani | + | passt.Timpani | 64.00 | 0.86 | 0.94 | 1.00 | 1.00 | 1.00 | 1.00 |  | 1.95 | 1.78 | 4.03 |
| timpani | - | passt.Timpani | -64.00 | 0.78 | 0.62 | 0.92 | 0.83 | 1.00 | 1.00 |  | -2.76 | -1.17 | 3.26 |
| alarm_clock | + | passt.Alarm clock | 64.00 | 0.88 | 0.90 | 0.92 | 0.83 | 1.00 | 1.00 |  | 2.61 | 1.24 | 4.14 |
| alarm_clock | - | passt.Alarm clock | -64.00 | 0.92 | 0.70 | 0.67 | 0.83 | 0.00 | 1.00 |  | 5.45 | 22.26 | 3.68 |
| phone_ring | + | passt.Telephone bell ringing | 64.00 | 0.98 | 0.76 | 0.75 | 0.50 | 1.00 | 1.00 |  | -6.43 | 0.52 | 4.48 |
| phone_ring | - | passt.Telephone bell ringing | -64.00 | 0.90 | 0.32 | 0.75 | 0.67 | 1.00 | 0.67 |  | 11.67 | -0.11 | 3.68 |

## Table 2: cross-effect matrix (12 prompts, calibrated gain)

Files: eval/crossfx/crossfx_matrix_plus.csv and crossfx_matrix_minus.csv (row = steered knob, column = the other knob's primary label, mean change in corpus-SD units, sign-adjusted to that knob's direction) and crossfx_effect.parquet (92 knob-signs x 2567 label columns). + sign: median diagonal 1.50 SD, median |off-diagonal| 0.31 SD.

Entangled = another knob's label moves by more than 0.5 x own effect and more than 0.5 SD (+ sign).

| knob | own effect + (SD) | own effect - (SD) | n entangled (+) | top 3 off-target (+) |
|---|---|---|---|---|
| rain_on_surface | 1.96 | -0.03 | 0 | lofi_hip_hop -0.90, black_metal +0.76, distant -0.75 |
| glitchy | 2.27 | -0.89 | 6 | wide_stereo -2.23, hardstyle +1.45, sensual -1.43 |
| radio_filtered | 2.72 | -0.66 | 1 | analog +1.69, sensual -1.16, wide_stereo -1.04 |
| childrens | 2.90 | -0.11 | 0 | dreamy +1.26, tango +1.16, playful +1.12 |
| distant | 2.93 | -0.40 | 5 | close_mic -2.48, aggressive -1.99, playful -1.73 |
| hardstyle | 1.65 | -1.21 | 7 | tango -1.31, lofi_hip_hop -1.15, sensual -1.15 |
| strummed_guitar | 2.57 | -0.19 | 1 | glitchy -1.30, post_rock +1.26, surf_rock +1.06 |
| gritty | 1.70 | -1.19 | 16 | jet +4.80, sensual -1.88, glitchy +1.83 |
| sensual | 2.42 | -0.55 | 7 | dissonant -1.81, glitchy -1.62, synth_pad +1.44 |
| aggressive | 1.17 | -1.00 | 11 | jet +3.38, black_metal +1.76, riser +1.31 |
| drone | 2.02 | -0.69 | 7 | wide_stereo +1.80, distant +1.63, horror_score +1.39 |
| synth_pad | 2.12 | -0.55 | 3 | drone +1.96, abs_cinematic +1.25, wide_stereo +1.22 |
| close_mic | 0.91 | -1.49 | 4 | live_recording -0.85, abs_industrial -0.73, wide_stereo -0.69 |
| tango | 1.31 | -1.00 | 5 | hardstyle -1.09, dreamy +0.92, samba +0.72 |
| ukulele | 3.59 | -0.01 | 0 | dissonant -1.76, dreamy +1.72, sensual +1.67 |
| playful | 1.50 | -0.64 | 7 | tango +1.22, koto +1.22, polyrhythmic +1.14 |
| dissonant | 0.83 | -1.75 | 12 | jet +5.77, hardstyle +1.43, black_metal +1.09 |
| surf_rock | 1.80 | -0.37 | 4 | post_rock +1.69, black_metal +1.36, strummed_guitar +1.08 |
| intense | 1.11 | -1.54 | 10 | aggressive +1.03, lofi_hip_hop -0.91, polyrhythmic +0.74 |
| post_rock | 1.93 | -0.30 | 4 | glitchy -1.60, gritty -1.22, black_metal +1.16 |
| wide_stereo | 1.07 | -2.23 | 12 | playful -1.02, surf_rock -0.91, distant +0.85 |
| live_recording | 1.19 | -0.88 | 22 | wide_stereo +2.08, horror_score +1.36, close_mic -1.32 |
| polyrhythmic | 1.54 | -1.34 | 2 | intense +1.04, sensual -0.84, lofi_hip_hop -0.77 |
| samba | 1.88 | -0.47 | 4 | tango +1.55, playful +1.17, close_mic +1.05 |
| koto | 1.16 | -0.95 | 4 | childrens +0.96, dreamy +0.96, playful +0.66 |
| solo_instrument | 0.84 | -0.97 | 20 | mains_hum +2.31, playful -1.53, radio_filtered +1.42 |
| riser | 0.76 | -1.26 | 7 | black_metal +0.89, analog +0.72, jet +0.68 |
| dreamy | 1.51 | -0.91 | 10 | synth_pad +1.24, drone +1.20, dissonant -1.18 |
| black_metal | 1.85 | -1.08 | 7 | jet +2.90, playful -1.35, tango -1.12 |
| analog | 1.93 | -1.05 | 1 | radio_filtered +2.45, abs_cinematic -0.83, playful -0.80 |
| horror_score | 1.47 | -0.36 | 9 | drone +1.68, playful -1.36, synth_pad +1.35 |
| lofi_hip_hop | 0.87 | -0.26 | 9 | dissonant -0.95, intense -0.88, sensual +0.78 |
| ui_click | 0.98 | -1.77 | 6 | wide_stereo -0.88, glitchy +0.88, koto +0.63 |
| impact_boom | 1.40 | -0.45 | 9 | wide_stereo +1.24, ui_click -1.08, distant +1.03 |
| abs_industrial | 1.65 | 0.03 | 16 | black_metal +1.78, aggressive +1.68, intense +1.51 |
| jet | 3.03 | -0.05 | 0 | sensual -1.24, samba -1.21, black_metal +1.18 |
| mains_hum | 1.50 | -0.11 | 5 | gritty +1.59, radio_filtered +0.94, playful -0.89 |
| abs_cinematic | 1.40 | -0.26 | 13 | synth_pad +1.77, drone +1.73, horror_score +1.59 |
| synthesizer | 0.77 | -0.21 | 16 | distant +1.53, close_mic -1.29, wide_stereo +1.10 |
| future_bass | 1.27 | -0.67 | 13 | hardstyle +1.41, synth_pad +1.19, acid_303 +1.03 |
| acid_303 | 1.47 | -0.66 | 10 | hardstyle +1.54, lofi_hip_hop -1.31, future_bass +1.10 |
| sfx_laden | 0.21 | -0.48 | 6 | close_mic -0.77, riser -0.73, distant +0.70 |
| jingle_bells | 0.84 | -0.01 | 15 | glitchy +0.96, dreamy +0.88, polyrhythmic +0.88 |
| timpani | 1.62 | -0.08 | 12 | wide_stereo +1.97, horror_score +1.77, solo_instrument +1.27 |
| alarm_clock | 0.06 | -0.00 | 1 | koto +0.74, ui_click +0.50, samba +0.49 |
| phone_ring | 0.02 | -0.00 | 4 | jet +1.32, wide_stereo -0.72, analog +0.62 |

## Table 3: seed 2 (EVAL_SEED+1) vs seed 1, top 10 by + sign primary ratio

Seed 2 = PCI-all plus sweep at the seed-1 calibrated range, with protocol and auc. Files: eval/results_s2/<knob>/.

| knob | sign | s1 ratio CLAP | s2 ratio CLAP | s1 ratio MuQ | s2 ratio MuQ | s1 cutoff | s2 cutoff |
|---|---|---|---|---|---|---|---|
| mains_hum | + | 50.01 | 8.01 | 1.80 | 3.38 | 4.07 | 4.30 |
| mains_hum | - | -48.29 | -6.18 | 0.38 | -0.51 | 3.89 | 3.77 |
| jet | + | 12.39 | 11.18 | 0.60 | 1.61 | 4.11 | 4.18 |
| jet | - | -24.08 | 3.32 | 2.36 | 1.18 | 3.68 | 3.81 |
| gritty | + | 7.01 | 1.77 | 1.30 | 1.25 | 3.88 | 3.96 |
| gritty | - | 2.60 | 4.66 | 5.45 | 20.59 | 4.11 | 3.87 |
| rain_on_surface | + | 6.72 | 3.40 | -0.03 | 0.08 | 4.54 | 4.40 |
| rain_on_surface | - | 2.10 | 0.95 | 3.27 | 1.46 | 3.68 | 3.81 |
| abs_cinematic | + | 5.75 | 7.35 | 1.33 | 2.37 | 3.87 | 3.79 |
| abs_cinematic | - | 17.12 | 8.71 | -1.24 | -4.37 | 3.26 | 3.12 |
| ui_click | + | 5.14 | 1.03 | 1.23 | 1.50 | 4.32 | 4.26 |
| ui_click | - | 1.88 | 2.10 | 16.82 | -5.25 | 3.68 | 3.81 |
| dissonant | + | 4.53 | 6.58 | 19.29 | -4.49 | 3.82 | 4.03 |
| dissonant | - | 3.39 | 8.50 | 7.75 | 3.16 | 4.01 | 3.96 |
| hardstyle | + | 4.20 | 2.66 | 2.07 | 1.70 | 4.04 | 3.94 |
| hardstyle | - | 4.46 | 7.46 | -13.34 | -8.52 | 3.26 | 3.12 |
| strummed_guitar | + | 4.01 | 2.92 | 1.37 | 2.16 | 3.96 | 4.22 |
| strummed_guitar | - | 8.11 | 4.30 | -1.27 | 0.96 | 3.26 | 3.12 |
| synthesizer | + | 3.24 | 2.15 | 2.79 | 1.62 | 4.16 | 4.21 |
| synthesizer | - | 3.76 | -9.93 | 1.14 | 7.33 | 3.26 | 3.12 |

On the + sign, 10/10 knobs keep a CLAP ratio above 1 at seed 2. Magnitudes are not stable where the PCI denominator is small.

