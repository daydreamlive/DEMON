# Ship pass results: every shipped SA3 steering knob under one protocol (2026-10-06)

Rules: ship/rules.md (frozen before any render). Gain = interpolated at the PCI-all cutoff, median over seeds 0,1,2; fractions, LPAPS and reach measured AT that gain on 50 test prompts x 3 seeds (fixed-gain pass).

Shipped 97; vectors measured 143; bar pass 121. Origins shipped: {'new': 57, 'new_v2': 8, 'legacy': 4, 'v1': 28}.
Gate outcomes: {'not compared (no counterpart)': 67, 'compared': 24, 'decided by the bar': 6, 'neither vector passes the bar': 5, 'not compared (no counterpart); fails the bar': 6}.

## Knob table (all measured vectors)

| pack | vector | origin | variant | blocks | gain + median (min-max) | gain - median (min-max) | reached +/- | CLAP + / - | MuQ + / - | LPAPS/cut + / - | bar | gate | hold drift (signed) | cross-effect own z +/- (n off > own) | applies_to | ships |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| abs_cinematic | v1_abs_cinematic | v1 | a |  | 31.98 (30.94-32.69) | 17.56 (16.42-22.80) | yyy/yyy | 0.94 / 0.80 | 0.72 / 0.54 | 0.96 / 0.93 | pass | not compared (no counterpart) | -0.48 | 0.80 (17) / -0.44 (8) | sfx | yes |
| abs_forest | abs_forest | new | d |  | 303.29 (297.18-305.52) | 149.10 (147.74-179.69) | yyy/yyy | 0.64 / 0.68 | 0.76 / 0.44 | 0.98 / 0.94 | pass | not compared (no counterpart) | -1.18 x | 0.83 (24) / -0.60 (6) | music | yes |
| abs_industrial | v1_abs_industrial | v1 | a |  | 47.62 (44.19-49.64) | 22.30 (20.55-31.53) | yyy/yyy | 0.92 / 0.64 | 0.80 / 0.60 | 1.00 / 0.96 | pass | not compared (no counterpart) | -0.31 | 1.22 (12) / -0.32 (16) | abstract | yes |
| accelerando | accelerando | new | a |  | 52.62 (47.86-60.26) | 76.74 (67.15-79.82) | yyy/yyy | 0.50 / 0.72 | 0.66 / 0.66 | 1.04 / 1.00 | pass | not compared (no counterpart) | -0.37 | 0.73 (27) / -1.39 (9) | music,sfx,abstract | yes |
| acid_303 | v1_acid_303 | v1 | a |  | 62.65 (60.81-64.56) | 27.12 (27.05-31.53) | yyy/yyy | 0.88 / 0.82 | 0.86 / 0.70 | 1.02 / 0.98 | pass | not compared (no counterpart) | +0.10 | 4.18 (5) / -0.93 (3) | music,sfx,abstract | yes |
| acid_house | acid_house | new | c |  | 57.78 (50.82-58.32) | 79.26 (75.71-103.38) | yyy/yyy | 0.88 / 0.40 | 0.76 / 0.60 | 1.03 / 0.95 | pass | not compared (no counterpart) | +0.04 | 3.10 (0) / -1.97 (0) | music,sfx,abstract | yes |
| aggressive | aggressive | new | a |  | 31.11 (30.30-32.93) | 26.41 (25.51-35.86) | yyy/yyy | 0.76 / 0.86 | 0.80 / 0.76 | 0.95 / 1.05 | pass | compared: v2 wins | -0.85 | 0.86 (11) / -0.54 (11) | sfx,abstract | yes |
| aggressive | v1_aggressive | v1 | a |  | 29.42 (29.34-31.85) | 26.05 (24.98-34.26) | yyy/yyy | 0.72 / 0.86 | 0.80 / 0.74 | 0.95 / 1.06 | pass | compared: v2 wins | -0.98 | 0.75 (12) / -0.52 (15) | sfx | no |
| alarm_clock | alarm_clock | v2gate | c |  | 113.14 (106.79-114.25) | 164.61 (148.94-170.07) | yyy/yyy | 0.86 / 0.56 | 0.92 / 0.48 | 0.97 / 0.97 | pass | compared: v1 kept | -0.17 | 4.31 (4) / -0.47 (61) | music,sfx,abstract | no |
| alarm_clock | v1_alarm_clock | v1 | a |  | 85.46 (76.74-86.84) | 58.15 (57.84-58.61) | yyy/yyy | 0.90 / 0.90 | 0.90 / 0.68 | 0.96 / 0.96 | pass | compared: v1 kept | -0.12 | 3.24 (4) / -0.52 (50) | music,sfx,abstract | yes |
| analog | analog | v2gate | d |  | 158.65 (157.04-175.87) | 247.15 (237.15-275.92) | yyy/yyy | 0.68 / 0.72 | 0.46 / 0.32 | 0.97 / 0.97 | FAIL | decided by the bar: only v1 passes | -0.70 | 1.60 (4) / -0.87 (13) | sfx | no |
| analog | v1_analog | v1 | a |  | 34.10 (33.28-36.97) | 52.81 (51.10-57.76) | yyy/yyy | 0.80 / 0.84 | 0.60 / 0.46 | 0.98 / 0.98 | pass | decided by the bar: only v1 passes | -0.28 | 1.62 (2) / -1.43 (6) | sfx | yes |
| arpeggiated | arpeggiated_r2 | new | b |  | 14.64 (13.83-15.65) | 13.25 (11.84-13.26) | yyy/yyy | 0.90 / 0.84 | 0.66 / 0.56 | 1.00 / 0.99 | pass | not compared (no counterpart) | +0.18 | 1.01 (11) / -0.92 (2) | music,sfx,abstract | yes |
| bedroom_intimate | bedroom_intimate | new | d |  | 230.17 (205.63-234.80) | 218.66 (208.92-229.58) | yyy/yyy | 0.96 / 0.84 | 0.90 / 0.90 | 1.04 / 0.98 | pass | not compared (no counterpart) | -0.04 | 1.00 (13) / -1.13 (10) | music,sfx,abstract | yes |
| black_metal | black_metal | new | a |  | 37.98 (36.38-43.70) | 25.50 (23.54-29.03) | yyy/yyy | 0.92 / 0.86 | 0.90 / 0.78 | 0.98 / 1.06 | pass | compared: v2 wins | -0.71 | 1.05 (20) / -0.48 (8) | sfx,abstract | yes |
| black_metal | v1_black_metal | v1 | a |  | 40.09 (35.01-41.88) | 26.09 (24.36-32.47) | yyy/yyy | 0.94 / 0.86 | 0.90 / 0.80 | 0.99 / 1.06 | pass | compared: v2 wins | -0.65 | 0.73 (25) / -0.52 (8) | music,sfx,abstract | no |
| bright | legacy_bright | legacy | legacy |  | 4.75 (4.56-4.80) | 8.43 (7.58-8.64) | yyy/yyy | 0.92 / 1.00 | 0.60 / 0.74 | 0.98 / 1.05 | pass | not compared (no counterpart) | +0.01 | 1.54 (7) / -1.46 (11) | sfx,abstract | yes |
| childrens | v1_childrens | v1 | a |  | 47.19 (45.70-50.14) | 19.57 (17.69-22.52) | yyy/yyy | 0.78 / 0.82 | 0.76 / 0.46 | 0.98 / 0.88 | pass | not compared (no counterpart) | +0.30 | 11.12 (0) / -0.47 (20) | music,sfx,abstract | yes |
| christmas | christmas | new | d |  | 388.39 (376.07-399.91) | 214.86 (178.86-235.61) | yyy/yyy | 0.88 / 0.80 | 0.62 / 0.64 | 0.97 / 0.95 | pass | not compared (no counterpart) | -0.49 | 18.65 (0) / -0.53 (14) | music,abstract | yes |
| clave_pattern | clave_pattern | new | b |  | 15.82 (15.79-17.56) | 13.88 (13.75-17.04) | yyy/yyy | 0.70 / 0.54 | 0.78 / 0.70 | 0.99 / 0.97 | pass | not compared (no counterpart) | -0.23 | 0.65 (19) / -0.77 (15) | music,sfx,abstract | yes |
| close_mic | close_mic | v2gate | d |  | 236.84 (209.40-247.10) | 264.30 (220.84-268.13) | yyy/yyy | 0.18 / 0.10 | 0.46 / 0.34 | 0.98 / 0.99 | FAIL | neither vector passes the bar: ships nothing | -0.02 | 0.63 (29) / -1.81 (2) | music,sfx | no |
| close_mic | v1_close_mic | v1 | a |  | 48.77 (44.40-50.75) | 54.21 (52.86-59.94) | yyy/yyy | 0.42 / 0.26 | 0.76 / 0.66 | 0.99 / 0.97 | FAIL | neither vector passes the bar: ships nothing | -0.29 | 1.82 (6) / -1.91 (4) | music,sfx | no |
| cumbia | cumbia | new | b |  | 15.44 (15.00-15.59) | 4.93 (4.90-5.21) | yyy/yyy | 0.88 / 0.76 | 0.54 / 0.38 | 0.95 / 0.90 | FAIL | not compared (no counterpart); fails the bar: ships nothing | -0.26 | 1.19 (11) / -0.54 (8) | sfx,abstract | no |
| dark | dark | new | a |  | 23.58 (19.10-25.50) | 18.74 (17.25-27.02) | yyy/yyy | 0.82 / 0.72 | 0.72 / 0.58 | 0.99 / 0.99 | pass | not compared (no counterpart) | -1.69 x | 0.66 (14) / -0.59 (5) | abstract | yes |
| dense_arrangement | legacy_density | legacy | legacy |  | 18.83 (16.83-20.20) | 13.15 (12.09-14.52) | yyy/yyy | 0.96 / 0.88 | 0.82 / 0.74 | 1.05 / 1.03 | pass | not compared (no counterpart) | -0.07 | 0.10 (91) / 0.13 (83) | sfx,abstract | yes |
| dishes | dishes | new | b |  | 14.98 (13.03-15.65) | 12.55 (11.28-12.83) | yyy/yyy | 0.90 / 0.72 | 0.94 / 0.64 | 0.96 / 1.03 | pass | not compared (no counterpart) | +0.13 | 7.53 (0) / -0.60 (27) | music,sfx,abstract | yes |
| dissonant | dissonant | v2gate | d |  | 240.05 (227.37-280.94) | 273.94 (247.06-280.52) | yyy/yyy | 0.62 / 0.80 | 0.66 / 0.50 | 0.97 / 1.02 | pass | compared: v1 kept | -0.28 | 1.58 (4) / -1.42 (2) | music,sfx,abstract | no |
| dissonant | v1_dissonant | v1 | a |  | 36.79 (33.98-40.95) | 43.41 (40.44-46.11) | yyy/yyy | 0.76 / 0.88 | 0.66 / 0.62 | 0.98 / 1.04 | pass | compared: v1 kept | -0.24 | 1.98 (4) / -1.58 (2) | music,sfx,abstract | yes |
| distant | v1_distant | v1 | a |  | 29.81 (26.06-32.80) | 19.55 (19.48-20.71) | yyy/yyy | 0.72 / 0.58 | 0.76 / 0.46 | 1.04 / 0.97 | pass | not compared (no counterpart) | -0.28 | 2.28 (3) / -0.63 (23) | music,sfx | yes |
| dreamy | dreamy | v2gate | d |  | 209.65 (191.72-214.83) | 160.10 (147.34-176.99) | yyy/yyy | 0.98 / 0.88 | 0.46 / 0.56 | 1.00 / 1.03 | FAIL | decided by the bar: only v1 passes | -0.15 | 0.90 (5) / -0.71 (7) | music,sfx,abstract | no |
| dreamy | v1_dreamy | v1 | a |  | 25.20 (22.00-26.10) | 18.11 (17.19-21.49) | yyy/yyy | 0.92 / 0.90 | 0.60 / 0.56 | 1.00 / 1.00 | pass | decided by the bar: only v1 passes | -0.14 | 0.64 (13) / -0.64 (13) | music,sfx,abstract | yes |
| drone | drone | v2gate | d |  | 174.74 (164.39-184.02) | 133.08 (117.80-143.54) | yyy/yyy | 0.84 / 0.58 | 0.66 / 0.54 | 0.94 / 1.00 | pass | compared: v1 kept | -0.55 | 0.88 (12) / -0.33 (24) | abstract | no |
| drone | v1_drone | v1 | a |  | 27.21 (26.93-27.86) | 19.46 (18.97-27.80) | yyy/yyy | 0.92 / 0.80 | 0.72 / 0.60 | 0.95 / 0.91 | pass | compared: v1 kept | -0.34 | 1.10 (3) / -0.58 (5) | music,sfx,abstract | yes |
| drum_machine | drum_machine | new | a |  | 25.08 (23.36-25.58) | 12.54 (12.52-14.32) | yyy/yyy | 0.74 / 0.74 | 0.68 / 0.62 | 1.04 / 0.97 | pass | not compared (no counterpart) | -0.01 | 0.20 (77) / -0.45 (18) | abstract | yes |
| dub_space_echo | dub_space_echo | new | d |  | 135.41 (126.81-140.40) | 154.33 (124.60-159.63) | yyy/yyy | 0.72 / 0.48 | 0.74 / 0.64 | 0.97 / 0.97 | pass | not compared (no counterpart) | -0.47 | 1.11 (9) / -1.62 (6) | music,sfx,abstract | yes |
| electric_piano | electric_piano_r2 | new | d |  | 264.51 (236.78-267.85) | 151.17 (147.53-171.27) | yyy/yyy | 0.94 / 0.78 | 0.92 / 0.66 | 1.01 / 1.01 | pass | not compared (no counterpart) | -0.15 | 1.58 (4) / -0.63 (11) | music,abstract | yes |
| erhu | erhu | new | d |  | 310.57 (303.76-318.79) | 196.24 (185.04-250.78) | yyy/yyy | 0.72 / 0.68 | 0.64 / 0.54 | 0.98 / 1.01 | pass | not compared (no counterpart) | -0.67 | 1.31 (6) / -0.60 (4) | sfx | yes |
| explosion | explosion | new | b |  | 16.96 (14.23-19.39) | 16.89 (14.32-17.95) | yyy/yyy | 0.86 / 0.58 | 0.62 / 0.34 | 0.91 / 0.97 | pass | not compared (no counterpart) | -0.25 | 6.88 (1) / -0.15 (72) | sfx | yes |
| fizzy_highs | fizzy_highs | new | a |  | 21.56 (21.06-23.36) | 29.30 (29.23-34.51) | yyy/yyy | 0.90 / 0.96 | 0.72 / 0.58 | 0.95 / 0.99 | pass | not compared (no counterpart) | -0.09 | 1.31 (11) / -0.92 (23) | music,sfx,abstract | yes |
| frogs | frogs | new | a |  | 67.53 (63.72-68.42) | 60.93 (56.20-69.23) | yyy/yyy | 0.64 / 0.80 | 0.90 / 0.38 | 0.96 / 0.96 | pass | not compared (no counterpart) | -0.18 | 24.93 (0) / -0.57 (35) | music,sfx,abstract | yes |
| future_bass | future_bass | v2gate | b |  | 16.65 (16.12-18.32) | 11.34 (11.32-14.03) | yyy/yyy | 0.86 / 0.74 | 0.74 / 0.70 | 0.97 / 0.98 | pass | compared: v2 wins | -0.68 | 2.65 (2) / -0.99 (2) | music,sfx,abstract | yes |
| future_bass | v1_future_bass | v1 | a |  | 44.45 (43.50-48.56) | 31.35 (29.17-39.25) | yyy/yyy | 0.88 / 0.78 | 0.74 / 0.70 | 0.98 / 1.01 | pass | compared: v2 wins | -0.33 | 2.92 (2) / -0.81 (2) | music,abstract | no |
| fuzzy | fuzzy | new | d |  | 203.93 (187.00-231.40) | 267.42 (239.39-274.47) | yyy/yyy | 0.74 / 0.52 | 0.74 / 0.62 | 0.94 / 1.03 | pass | not compared (no counterpart) | +0.05 | 2.34 (2) / -1.38 (7) | music,abstract | yes |
| glitch_sfx | glitch_sfx | new | b |  | 13.17 (12.52-13.69) | 9.22 (8.09-10.02) | yyy/yyy | 0.90 / 0.60 | 0.88 / 0.66 | 0.97 / 0.97 | pass | not compared (no counterpart) | +0.08 | 1.16 (17) / -0.51 (29) | music,sfx,abstract | yes |
| glitchy | glitchy | v2gate | d |  | 134.21 (131.74-144.31) | 145.42 (135.10-150.82) | yyy/yyy | 0.90 / 0.72 | 0.88 / 0.50 | 0.99 / 1.03 | pass | compared: v1 kept | -0.16 | 1.13 (17) / -0.73 (35) | music,sfx,abstract | no |
| glitchy | v1_glitchy | v1 | a |  | 28.21 (27.07-31.04) | 29.44 (28.16-29.72) | yyy/yyy | 0.96 / 0.84 | 0.96 / 0.64 | 0.99 / 1.03 | pass | compared: v1 kept | -0.04 | 1.34 (14) / -0.50 (52) | music,sfx,abstract | yes |
| gritty | v1_gritty | v1 | a |  | 30.75 (26.11-33.59) | 32.51 (30.21-34.01) | yyy/yyy | 0.88 / 0.84 | 0.88 / 0.88 | 1.02 / 1.04 | pass | not compared (no counterpart) | +0.08 | 1.07 (7) / -1.02 (5) | music,sfx,abstract | yes |
| hammered_notes | hammered_notes | new | d |  | 301.93 (298.94-310.50) | 400.70 (379.34-440.38) | yyy/yyy | 0.70 / 0.32 | 0.80 / 0.82 | 0.98 / 0.98 | pass | not compared (no counterpart) | -0.31 | 1.54 (3) / -1.57 (6) | music,sfx,abstract | yes |
| handclaps | handclaps | new | a |  | 61.37 (50.19-68.90) | 28.57 (26.74-28.87) | yyy/yyy | 0.68 / 0.58 | 0.64 / 0.60 | 0.96 / 0.95 | pass | not compared (no counterpart) | -0.36 | 11.30 (2) / -0.52 (8) | sfx | yes |
| hardstyle | hardstyle | v2gate | b |  | 11.68 (11.12-12.52) | 9.16 (9.08-9.88) | yyy/yyy | 0.84 / 0.88 | 0.78 / 0.80 | 0.98 / 1.01 | pass | compared: v2 wins | -0.21 | 1.58 (10) / -0.91 (9) | sfx,abstract | yes |
| hardstyle | v1_hardstyle | v1 | a |  | 34.12 (30.39-34.73) | 27.08 (25.74-29.59) | yyy/yyy | 0.88 / 0.92 | 0.70 / 0.82 | 0.98 / 1.03 | pass | compared: v2 wins | -0.17 | 1.60 (9) / -0.76 (12) | sfx,abstract | no |
| horror_score | v1_horror_score | v1 | a |  | 27.51 (24.53-30.51) | 18.20 (16.43-26.06) | yyy/yyy | 0.90 / 0.58 | 0.72 / 0.60 | 1.00 / 1.00 | pass | not compared (no counterpart) | -1.07 x | 0.87 (9) / -0.55 (5) | abstract | yes |
| impact_boom | impact_boom | v2gate | d |  | 208.49 (195.10-225.31) | 176.90 (174.15-185.46) | yyy/yyy | 0.38 / 0.26 | 0.64 / 0.32 | 0.95 / 1.03 | FAIL | decided by the bar: only v1 passes | -0.35 | 0.45 (41) / -1.35 (6) | (none measured >= 0.6) | no |
| impact_boom | v1_impact_boom | v1 | a |  | 79.10 (73.69-80.27) | 53.09 (52.29-53.80) | yyy/yyy | 0.90 / 0.72 | 0.92 / 0.46 | 1.00 / 1.00 | pass | decided by the bar: only v1 passes | -0.19 | 2.40 (6) / -1.07 (8) | music,sfx,abstract | yes |
| intense | intense | v2gate | d |  | 189.65 (162.96-200.14) | 213.21 (204.90-230.53) | yyy/yyy | 0.92 / 0.94 | 0.86 / 0.84 | 0.99 / 1.02 | pass | compared: v2 wins | -0.09 | 1.23 (7) / -1.05 (7) | music,sfx,abstract | yes |
| intense | v1_intense | v1 | a |  | 25.17 (24.43-29.77) | 32.77 (31.67-33.47) | yyy/yyy | 0.88 / 0.84 | 0.80 / 0.84 | 1.00 / 1.03 | pass | compared: v2 wins | +0.03 | 1.02 (8) / -0.95 (4) | music,sfx,abstract | no |
| jet | jet | new | a |  | 35.97 (35.78-36.86) | 32.19 (31.72-33.10) | yyy/yyy | 0.96 / 0.66 | 0.66 / 0.26 | 0.97 / 1.00 | pass | compared: v1 kept | +0.02 | 2.73 (2) / -0.42 (40) | sfx | no |
| jet | v1_jet | v1 | a |  | 36.07 (35.70-37.17) | 32.76 (31.53-32.91) | yyy/yyy | 0.92 / 0.72 | 0.64 / 0.26 | 0.96 / 1.00 | pass | compared: v1 kept | -0.00 | 2.53 (3) / -0.45 (38) | sfx | yes |
| jingle_bells | jingle_bells | v2gate | a |  | 46.38 (40.02-47.72) | 17.23 (15.69-19.00) | yyy/yyy | 0.94 / 0.76 | 0.88 / 0.62 | 0.97 / 0.95 | pass | compared: v2 wins | -0.20 | 15.71 (0) / -0.48 (17) | music,sfx,abstract | yes |
| jingle_bells | v1_jingle_bells | v1 | a |  | 42.28 (41.60-43.34) | 17.47 (15.74-20.02) | yyy/yyy | 0.84 / 0.72 | 0.86 / 0.62 | 0.97 / 0.98 | pass | compared: v2 wins | -0.33 | 18.01 (0) / -0.49 (15) | music,sfx,abstract | no |
| koto | v1_koto | v1 | a |  | 50.33 (49.24-54.06) | 26.78 (26.60-31.96) | yyy/yyy | 0.44 / 0.72 | 0.44 / 0.62 | 0.96 / 0.93 | pass | not compared (no counterpart) | -0.45 | 1.31 (7) / -0.56 (17) | music,sfx,abstract | yes |
| laser_zap | laser_zap | new | b |  | 16.37 (15.30-16.51) | 12.94 (11.33-13.10) | yyy/yyy | 0.90 / 0.80 | 0.76 / 0.40 | 0.97 / 0.99 | pass | not compared (no counterpart) | -0.03 | 0.61 (50) / -0.78 (21) | music,sfx,abstract | yes |
| legato_phrasing | legato_phrasing | new | d |  | 216.42 (206.39-240.06) | 173.59 (162.05-179.77) | yyy/yyy | 0.94 / 0.96 | 0.96 / 0.78 | 1.04 / 0.97 | pass | not compared (no counterpart) | +0.06 | 0.68 (24) / -1.26 (8) | music,sfx,abstract | yes |
| live_recording | live_recording | v2gate | a |  | 59.13 (57.27-65.06) | 60.19 (55.93-67.04) | yyy/yyy | 0.30 / 0.40 | 0.44 / 0.20 | 0.95 / 0.99 | FAIL | neither vector passes the bar: ships nothing | -0.86 | 2.46 (2) / -1.97 (3) | sfx | no |
| live_recording | v1_live_recording | v1 | a |  | 62.00 (61.37-71.60) | 58.46 (46.23-67.02) | yyy/yyy | 0.32 / 0.52 | 0.40 / 0.24 | 0.95 / 0.99 | FAIL | neither vector passes the bar: ships nothing | -0.71 | 2.68 (2) / -1.96 (2) | sfx | no |
| locked_groove | locked_groove | new | a |  | 60.68 (57.60-63.96) | 68.31 (65.45-71.93) | yyy/yyy | 0.80 / 0.90 | 0.38 / 0.74 | 1.00 / 0.99 | pass | not compared (no counterpart) | +0.13 | 1.31 (13) / -1.02 (7) | sfx | yes |
| lofi_hip_hop | lofi_hip_hop | v2gate | b |  | 35.47 (30.17-37.80) | 12.52 (11.95-13.06) | yyy/yyy | 0.64 / 0.48 | 0.96 / 0.64 | 1.06 / 1.07 | pass | compared: v1 kept | -0.08 | 2.07 (3) / -0.44 (12) | music,abstract | no |
| lofi_hip_hop | v1_lofi_hip_hop | v1 | a |  | 88.87 (79.92-94.07) | 28.05 (25.18-31.99) | yyy/yyy | 0.64 / 0.52 | 0.88 / 0.70 | 1.01 / 1.06 | pass | compared: v1 kept | -0.17 | 1.69 (5) / -0.37 (10) | music,abstract | yes |
| magic_sparkle | magic_sparkle | new | a |  | 32.93 (32.47-33.36) | 23.13 (22.44-23.51) | yyy/yyy | 0.88 / 0.92 | 0.96 / 0.82 | 1.01 / 1.01 | pass | not compared (no counterpart) | -0.07 | 1.72 (12) / -1.18 (10) | music,sfx,abstract | yes |
| mains_hum | mains_hum | v2gate | d |  | 634.32 (599.45-653.01) | 665.99 (624.12-666.33) | yyy/yyy | 0.94 / 0.66 | 0.76 / 0.44 | 0.97 / 0.98 | pass | compared: v1 kept | -0.49 | 4.32 (0) / -0.23 (63) | sfx | no |
| mains_hum | v1_mains_hum | v1 | a |  | 82.26 (81.88-83.93) | 82.60 (78.15-84.26) | yyy/yyy | 0.88 / 0.84 | 0.78 / 0.48 | 0.96 / 0.99 | pass | compared: v1 kept | -0.04 | 2.83 (2) / -0.60 (23) | (none measured >= 0.6) | yes |
| marcato | marcato | new | b |  | 13.56 (13.02-13.74) | 13.20 (11.69-14.26) | yyy/yyy | 0.86 / 0.78 | 0.68 / 0.62 | 1.02 / 0.98 | pass | not compared (no counterpart) | +0.18 | 1.20 (13) / -1.16 (9) | music,sfx,abstract | yes |
| muted_horn | muted_horn | new | b |  | 10.24 (9.58-10.78) | 8.93 (8.73-9.00) | yyy/yyy | 0.80 / 0.74 | 0.70 / 0.66 | 1.04 / 0.96 | pass | not compared (no counterpart) | -0.17 | 1.62 (3) / -1.12 (12) | music,sfx,abstract | yes |
| new_age | new_age | new | c |  | 30.91 (28.65-33.51) | 30.03 (28.18-36.90) | yyy/yyy | 0.78 / 0.36 | 0.82 / 0.60 | 0.99 / 0.92 | pass | not compared (no counterpart) | -0.56 | 0.97 (5) / -0.22 (36) | music,abstract | yes |
| nylon_soft | nylon_soft | new | a |  | 37.84 (34.34-38.01) | 35.04 (30.63-35.13) | yyy/yyy | 0.88 / 0.84 | 0.86 / 0.66 | 1.05 / 1.00 | pass | not compared (no counterpart) | -0.04 | 1.90 (3) / -1.02 (17) | sfx,abstract | yes |
| palm_muted | palm_muted | new | d |  | 217.91 (197.75-221.47) | 249.96 (246.00-262.24) | yyy/yyy | 0.90 / 0.84 | 0.70 / 0.74 | 1.00 / 1.00 | pass | not compared (no counterpart) | -0.13 | 2.52 (1) / -2.64 (3) | sfx | yes |
| percussive | legacy_percussive | legacy | legacy |  | 16.07 (15.37-16.51) | 12.06 (11.15-12.06) | yyy/yyy | 0.98 / 0.98 | 0.76 / 0.74 | 1.01 / 1.02 | pass | not compared (no counterpart) | -0.10 | 1.16 (12) / -0.59 (35) | music,sfx,abstract | yes |
| phone_ring | phone_ring | v2gate | c |  | 123.67 (119.70-131.42) | 65.39 (63.94-68.59) | yyy/yyy | 0.98 / 0.76 | 0.86 / 0.56 | 0.93 / 0.99 | pass | compared: v1 kept | -0.16 | 7.10 (2) / 0.41 (42) | music,sfx | no |
| phone_ring | v1_phone_ring | v1 | a |  | 66.82 (66.51-70.18) | 46.35 (43.58-47.85) | yyy/yyy | 0.92 / 0.92 | 0.80 / 0.38 | 0.93 / 1.02 | pass | compared: v1 kept | +0.01 | 1.21 (8) / -0.78 (18) | music,sfx | yes |
| plate_reverb | plate_reverb | new | d |  | 232.33 (223.90-248.88) | 242.08 (233.99-256.97) | yyy/yyy | 0.52 / 0.76 | 0.64 / 0.76 | 1.05 / 0.96 | pass | not compared (no counterpart) | -0.41 | 1.65 (10) / -2.04 (4) | sfx | yes |
| playful | v1_playful | v1 | a |  | 30.96 (30.87-32.60) | 14.35 (14.22-16.59) | yyy/yyy | 0.70 / 0.72 | 0.60 / 0.58 | 0.99 / 0.88 | pass | not compared (no counterpart) | -0.10 | 1.12 (7) / -0.52 (6) | music,sfx,abstract | yes |
| polyrhythmic | polyrhythmic | new | a |  | 61.88 (61.57-66.55) | 43.27 (38.51-51.95) | yyy/yyy | 0.62 / 0.56 | 0.78 / 0.56 | 0.99 / 1.00 | pass | compared: v2 wins | -0.21 | 1.02 (17) / -0.62 (11) | music,sfx,abstract | yes |
| polyrhythmic | v1_polyrhythmic | v1 | a |  | 58.52 (56.67-62.05) | 40.82 (34.61-46.89) | yyy/yyy | 0.66 / 0.52 | 0.80 / 0.58 | 0.99 / 1.01 | pass | compared: v2 wins | -0.35 | 1.08 (16) / -0.64 (11) | music,abstract | no |
| post_rock | post_rock | v2gate | d |  | 365.69 (346.10-368.12) | 203.14 (182.50-213.60) | yyy/yyy | 0.90 / 0.80 | 0.96 / 0.80 | 0.98 / 1.02 | pass | compared: v2 wins | -0.62 | 1.53 (5) / -0.76 (10) | music,abstract | yes |
| post_rock | v1_post_rock | v1 | a |  | 41.96 (41.11-46.68) | 19.77 (18.63-25.14) | yyy/yyy | 0.94 / 0.82 | 0.92 / 0.78 | 0.93 / 0.98 | pass | compared: v2 wins | -0.58 | 1.57 (2) / -0.50 (7) | music,abstract | no |
| pulsing_synth | pulsing_synth | new | d |  | 334.86 (297.30-341.99) | 306.76 (298.58-325.87) | yyy/yyy | 0.70 / 0.92 | 0.84 / 0.62 | 1.03 / 0.97 | pass | not compared (no counterpart) | -6.91 x | 0.72 (29) / -1.37 (9) | sfx,abstract | yes |
| quantized | quantized | new | d |  | 605.37 (569.16-631.44) | 497.39 (440.00-510.69) | yyy/yyy | 0.80 / 0.38 | 0.70 / 0.40 | 1.00 / 1.02 | pass | not compared (no counterpart) | -0.33 | 0.56 (40) / -1.15 (19) | (none measured >= 0.6) | yes |
| radio_filtered | radio_filtered | v2gate | b |  | 14.70 (13.69-14.71) | 18.86 (18.75-19.47) | yyy/yyy | 0.92 / 0.72 | 0.82 / 0.78 | 0.94 / 1.03 | pass | compared: v1 kept | -0.32 | 0.65 (16) / -0.49 (43) | music,abstract | no |
| radio_filtered | v1_radio_filtered | v1 | a |  | 40.60 (40.43-40.65) | 44.87 (43.92-49.80) | yyy/yyy | 0.92 / 0.78 | 0.70 / 0.68 | 0.93 / 1.01 | pass | compared: v1 kept | -0.43 | 0.51 (32) / -0.57 (32) | music,abstract | yes |
| rain_on_surface | rain_on_surface | new | a |  | 34.48 (32.72-35.85) | 30.61 (28.13-31.01) | yyy/yyy | 0.86 / 0.58 | 0.54 / 0.20 | 0.93 / 1.02 | FAIL | neither vector passes the bar: ships nothing | -0.08 | 28.26 (1) / -0.40 (40) | sfx | no |
| rain_on_surface | v1_rain_on_surface | v1 | a |  | 34.59 (33.39-35.81) | 30.78 (28.05-30.97) | yyy/yyy | 0.84 / 0.56 | 0.56 / 0.20 | 0.94 / 1.02 | FAIL | neither vector passes the bar: ships nothing | -0.10 | 21.97 (1) / -0.40 (39) | sfx | no |
| revving | revving | new | c |  | 41.16 (40.09-42.23) | 45.11 (44.83-45.25) | yyy/yyy | 0.80 / 0.72 | 0.82 / 0.56 | 0.99 / 1.01 | pass | not compared (no counterpart) | -0.25 | 3.08 (1) / -0.30 (58) | music,sfx | yes |
| riser | riser | v2gate | d |  | 226.97 (199.60-240.18) | 163.12 (161.25-180.57) | yyy/yyy | 0.54 / 0.48 | 0.56 / 0.48 | 0.98 / 0.99 | FAIL | decided by the bar: only v1 passes | -0.31 | 1.10 (11) / -0.75 (13) | music,sfx | no |
| riser | v1_riser | v1 | a |  | 63.30 (51.92-66.17) | 36.00 (35.24-38.75) | yyy/yyy | 0.68 / 0.58 | 0.80 / 0.54 | 1.00 / 0.96 | pass | decided by the bar: only v1 passes | -0.55 | 2.10 (5) / -0.41 (25) | music,sfx,abstract | yes |
| rough | legacy_rough | legacy | legacy |  | 13.74 (11.56-16.10) | 19.25 (17.43-19.68) | yyy/yyy | 0.48 / 0.62 | 0.66 / 0.72 | 0.91 / 1.05 | pass | not compared (no counterpart) | +0.03 | 0.03 (99) / 0.25 (62) | sfx | yes |
| samba | v1_samba | v1 | a |  | 45.13 (43.17-50.57) | 15.81 (15.77-17.44) | yyy/yyy | 0.88 / 0.76 | 0.82 / 0.44 | 0.99 / 0.90 | pass | not compared (no counterpart) | -0.12 | 1.05 (12) / -0.58 (9) | music,sfx,abstract | yes |
| sensual | v1_sensual | v1 | a |  | 28.87 (27.56-30.66) | 20.19 (19.61-23.55) | yyy/yyy | 0.44 / 0.54 | 0.78 / 0.76 | 1.00 / 0.99 | FAIL | not compared (no counterpart); fails the bar: ships nothing | -0.04 | 0.87 (7) / -0.79 (10) | music,sfx,abstract | no |
| sfx_applause | sfx_applause | new | b |  | 6.97 (6.76-7.19) | 8.13 (7.58-8.19) | yyy/yyy | 0.80 / 0.92 | 0.74 / 0.72 | 0.91 / 1.02 | pass | not compared (no counterpart) | +0.01 | 394.89 (0) / -0.95 (10) | music,sfx,abstract | yes |
| sfx_baby_cry | sfx_baby_cry | new | a |  | 22.34 (20.32-22.41) | 33.31 (33.12-39.26) | yyy/yyy | 0.78 / 0.92 | 0.72 / 0.66 | 0.95 / 0.96 | pass | not compared (no counterpart) | -0.07 | 32.68 (0) / -0.69 (32) | sfx | yes |
| sfx_event_rate | sfx_event_rate | new | a |  | 59.85 (52.95-60.26) | 82.68 (78.48-83.56) | yyy/yyy | 0.82 / 0.74 | 0.72 / 0.74 | 0.97 / 0.97 | pass | not compared (no counterpart) | +0.11 | 0.84 (26) / -0.52 (45) | music,sfx,abstract | yes |
| sfx_far_away | sfx_far_away | new | a |  | 65.50 (61.17-77.42) | 62.86 (60.78-71.46) | yyy/yyy | 0.84 / 0.72 | 0.78 / 0.36 | 0.96 / 0.93 | pass | not compared (no counterpart) | -0.88 | 0.44 (57) / -1.14 (16) | music,sfx | yes |
| sfx_indoor_hall | sfx_indoor_hall | new | d |  | 420.46 (404.05-422.48) | 445.34 (444.21-492.21) | yyy/yyy | 0.76 / 0.54 | 0.80 / 0.44 | 0.97 / 1.02 | pass | not compared (no counterpart) | -0.64 | 0.63 (42) / -0.45 (57) | music,sfx | yes |
| sfx_insects | sfx_insects | new | a |  | 36.29 (35.28-37.41) | 35.53 (34.31-38.32) | yyy/yyy | 0.92 / 0.58 | 0.80 / 0.54 | 0.97 / 0.93 | pass | not compared (no counterpart) | +0.22 | 40.77 (0) / -0.13 (76) | music,sfx | yes |
| sfx_laden | sfx_laden | v2gate | d |  | 200.74 (187.88-216.36) | 273.47 (245.63-273.78) | yyy/yyy | 0.44 / 0.54 | 0.66 / 0.52 | 0.94 / 0.98 | FAIL | neither vector passes the bar: ships nothing | -3.32 x | 2.09 (0) / -0.13 (76) | (none measured >= 0.6) | no |
| sfx_laden | v1_sfx_laden | v1 | a |  | 28.51 (27.96-31.49) | 30.50 (28.10-32.74) | yyy/yyy | 0.44 / 0.38 | 0.60 / 0.56 | 0.97 / 0.98 | FAIL | neither vector passes the bar: ships nothing | -0.87 | 2.27 (2) / -0.69 (14) | (none measured >= 0.6) | no |
| sfx_ocean_waves | sfx_ocean_waves | new | c |  | 25.34 (23.56-27.83) | 59.20 (52.60-64.78) | yyy/yyy | 0.80 / 0.94 | 0.78 / 0.92 | 0.93 / 1.01 | pass | not compared (no counterpart) | -0.35 | 42.85 (0) / -0.30 (57) | music,sfx | yes |
| sfx_sub_weight | sfx_sub_weight | new | c |  | 62.13 (55.13-63.17) | 76.58 (70.61-81.17) | yyy/yyy | 1.00 / 0.86 | 0.82 / 0.52 | 1.00 / 1.03 | pass | not compared (no counterpart) | +0.01 | 1.59 (9) / -1.24 (9) | music,sfx,abstract | yes |
| sfx_telephone | sfx_telephone | new | d |  | 195.81 (167.70-198.84) | 192.03 (167.58-200.39) | yyy/yyy | 0.58 / 0.44 | 0.66 / 0.32 | 1.01 / 0.95 | FAIL | not compared (no counterpart); fails the bar: ships nothing | -0.57 | 0.28 (43) / 0.04 (100) | (none measured >= 0.6) | no |
| shaker_pulse | shaker_pulse | new | d |  | 220.32 (214.80-221.40) | 185.45 (179.25-228.37) | yyy/yyy | 0.68 / 0.84 | 0.84 / 0.82 | 1.00 / 0.90 | pass | not compared (no counterpart) | -0.41 | 0.80 (12) / -0.98 (6) | abstract | yes |
| shimmer_reverb | shimmer_reverb | new | c |  | 72.24 (69.01-75.39) | 123.17 (117.02-130.17) | yyy/yyy | 0.74 / 0.18 | 0.88 / 0.60 | 1.02 / 0.97 | pass | not compared (no counterpart) | -0.20 | 2.48 (4) / -0.85 (14) | music,sfx | yes |
| shimmering | shimmering_r2 | new | d |  | 181.27 (170.09-188.23) | 187.39 (165.71-215.88) | yyy/yyy | 0.76 / 0.48 | 0.88 / 0.74 | 1.00 / 1.01 | pass | not compared (no counterpart) | -0.04 | 1.69 (5) / -0.30 (64) | music,sfx,abstract | yes |
| sitar | sitar_r2 | new | d |  | 327.18 (311.30-328.55) | 214.42 (205.09-253.44) | yyy/yyy | 0.84 / 0.94 | 0.82 / 0.74 | 1.00 / 1.00 | pass | not compared (no counterpart) | -0.58 | 2.67 (2) / -0.91 (8) | music,sfx,abstract | yes |
| smash | smash | new | b |  | 29.44 (27.97-33.93) | 23.41 (23.11-23.49) | yyy/yyy | 0.62 / 0.76 | 0.52 / 0.28 | 0.97 / 0.98 | FAIL | not compared (no counterpart); fails the bar: ships nothing | -0.39 | 11.10 (0) / -0.58 (19) | sfx | no |
| solo_build | solo_build | new | c |  | 218.10 (215.25-263.45) | 35.31 (33.24-40.00) | yyy/yyy | 0.80 / 0.68 | 0.36 / 0.88 | 0.97 / 1.01 | pass | not compared (no counterpart) | -0.50 | 0.38 (56) / -1.61 (2) | music,sfx,abstract | yes |
| solo_crest | solo_crest | new | d |  | 49.10 (47.18-51.65) | 46.07 (40.77-48.40) | yyy/yyy | 0.76 / 0.50 | 0.68 / 0.52 | 1.01 / 0.96 | pass | not compared (no counterpart) | -5.91 | 1.82 (4) / -2.82 (2) | sfx | yes |
| solo_instrument | solo_instrument | v2gate | a |  | 31.35 (31.26-32.38) | 30.53 (26.76-32.34) | yyy/yyy | 0.72 / 0.66 | 0.80 / 0.68 | 0.98 / 1.02 | pass | compared: v2 wins | -0.22 | 0.44 (42) / -1.08 (7) | music,sfx,abstract | yes |
| solo_instrument | v1_solo_instrument | v1 | a |  | 32.91 (32.26-33.80) | 31.82 (28.93-33.30) | yyy/yyy | 0.76 / 0.62 | 0.84 / 0.68 | 0.98 / 1.01 | pass | compared: v2 wins | -0.05 | 0.57 (24) / -1.10 (5) | music,sfx,abstract | no |
| spiccato | spiccato | new | d |  | 346.58 (339.30-384.60) | 337.12 (310.71-344.49) | yyy/yyy | 0.84 / 0.88 | 0.86 / 0.64 | 1.00 / 1.00 | pass | not compared (no counterpart) | -0.58 | 0.73 (20) / -0.68 (25) | music,sfx,abstract | yes |
| strummed_guitar | strummed_guitar | v2gate | d |  | 373.44 (357.73-385.61) | 227.91 (202.65-232.97) | yyy/yyy | 0.92 / 0.68 | 0.78 / 0.34 | 0.98 / 1.04 | pass | compared: v1 kept | -0.33 | 3.06 (1) / -0.59 (14) | (none measured >= 0.6) | no |
| strummed_guitar | v1_strummed_guitar | v1 | a |  | 40.88 (40.03-42.95) | 21.62 (21.01-21.83) | yyy/yyy | 0.96 / 0.82 | 0.92 / 0.44 | 0.97 / 0.97 | pass | compared: v1 kept | -0.63 | 4.93 (1) / -0.67 (9) | abstract | yes |
| surf_rock | surf_rock | v2gate | b |  | 12.08 (11.94-13.17) | 5.95 (5.77-6.19) | yyy/yyy | 0.80 / 0.76 | 0.72 / 0.58 | 0.95 / 0.93 | pass | compared: v2 wins | -0.42 | 1.10 (11) / -0.38 (16) | abstract | yes |
| surf_rock | v1_surf_rock | v1 | a |  | 35.68 (33.42-36.40) | 16.76 (16.10-17.06) | yyy/yyy | 0.78 / 0.68 | 0.76 / 0.50 | 0.96 / 0.95 | pass | compared: v2 wins | -0.33 | 1.21 (9) / -0.23 (40) | (none measured >= 0.6) | no |
| swing | swing_r2 | new | b |  | 15.91 (13.05-16.73) | 15.17 (12.56-18.28) | yyy/yyy | 0.68 / 0.70 | 0.74 / 0.68 | 1.00 / 0.98 | pass | not compared (no counterpart) | -0.28 | 1.15 (16) / -0.85 (14) | sfx | yes |
| synth_pad | synth_pad | v2gate | d |  | 275.53 (273.14-278.27) | 167.99 (155.92-209.66) | yyy/yyy | 0.94 / 0.74 | 0.98 / 0.78 | 1.00 / 0.98 | pass | compared: v1 kept | -0.15 | 2.37 (4) / -0.54 (5) | music,sfx,abstract | no |
| synth_pad | v1_synth_pad | v1 | a |  | 41.16 (38.82-42.66) | 19.70 (19.63-27.25) | yyy/yyy | 0.98 / 0.84 | 0.94 / 0.70 | 1.00 / 0.95 | pass | compared: v1 kept | -0.41 | 2.02 (6) / -0.65 (0) | music,sfx,abstract | yes |
| synthesizer | synthesizer | v2gate | b |  | 18.82 (17.37-21.06) | 12.09 (11.07-13.71) | yyy/yyy | 0.76 / 0.88 | 0.62 / 0.62 | 0.96 / 1.01 | pass | compared: v1 kept | -0.03 | 9.71 (1) / -0.88 (8) | music,sfx,abstract | no |
| synthesizer | v1_synthesizer | v1 | a |  | 52.83 (44.77-56.46) | 31.17 (30.95-36.44) | yyy/yyy | 0.66 / 0.92 | 0.76 / 0.58 | 1.04 / 0.98 | pass | compared: v1 kept | -0.35 | 5.68 (0) / -0.73 (12) | music,sfx | yes |
| tango | tango | v2gate | d |  | 392.93 (386.63-450.61) | 193.10 (175.99-209.42) | yyy/yyy | 0.28 / 0.62 | 0.66 / 0.58 | 0.99 / 1.01 | FAIL | decided by the bar: only v1 passes | -3.22 x | 0.72 (26) / -0.71 (13) | music,sfx,abstract | no |
| tango | v1_tango | v1 | a |  | 48.44 (47.18-49.56) | 17.83 (17.66-21.37) | yyy/yyy | 0.70 / 0.68 | 0.78 / 0.52 | 0.97 / 0.89 | pass | decided by the bar: only v1 passes | -0.47 | 0.95 (11) / -0.67 (11) | music,sfx,abstract | yes |
| thick_unison | thick_unison | new | b |  | 28.95 (28.56-32.06) | 32.75 (29.31-36.45) | yyy/yyy | 0.72 / 0.62 | 0.58 / 0.46 | 0.97 / 0.98 | FAIL | not compared (no counterpart); fails the bar: ships nothing | -0.56 | 1.52 (16) / -1.37 (10) | sfx | no |
| timpani | timpani | v2gate | d |  | 238.24 (231.83-266.18) | 177.18 (168.52-191.05) | yyy/yyy | 0.34 / 0.54 | 0.84 / 0.66 | 0.99 / 1.06 | FAIL | decided by the bar: only v1 passes | -1.28 x | 8.11 (0) / -0.46 (37) | music | no |
| timpani | v1_timpani | v1 | a |  | 41.40 (40.42-52.43) | 27.23 (26.20-31.66) | yyy/yyy | 0.88 / 0.74 | 0.90 / 0.58 | 0.94 / 0.99 | pass | decided by the bar: only v1 passes | -0.44 | 32.89 (0) / -0.53 (13) | music,sfx,abstract | yes |
| tom_patterns | tom_patterns | new | d |  | 223.40 (193.76-225.52) | 195.72 (190.42-196.24) | yyy/yyy | 0.84 / 0.70 | 0.66 / 0.66 | 1.02 / 1.02 | pass | not compared (no counterpart) | -0.08 | 1.70 (9) / -2.11 (5) | sfx | yes |
| ui_click | ui_click | v2gate | b |  | 18.60 (17.68-20.99) | 13.87 (12.45-14.59) | yyy/yyy | 0.74 / 0.74 | 0.92 / 0.80 | 0.95 / 0.98 | pass | compared: v2 wins | -0.13 | 0.64 (39) / -0.82 (22) | music,sfx,abstract | yes |
| ui_click | v1_ui_click | v1 | a |  | 58.46 (54.81-65.60) | 39.24 (35.91-42.12) | yyy/yyy | 0.72 / 0.64 | 0.84 / 0.78 | 0.96 / 0.95 | pass | compared: v2 wins | +0.01 | 0.52 (39) / -0.90 (10) | music,sfx,abstract | no |
| ukulele | ukulele | new | a |  | 55.94 (48.22-56.47) | 27.64 (24.45-27.87) | yyy/yyy | 0.94 / 0.76 | 0.84 / 0.58 | 1.02 / 1.01 | pass | compared: v2 wins | -0.35 | 261.48 (0) / -0.79 (13) | music,sfx,abstract | yes |
| ukulele | v1_ukulele | v1 | a |  | 53.47 (48.55-56.05) | 28.12 (24.19-28.88) | yyy/yyy | 0.92 / 0.74 | 0.80 / 0.62 | 1.01 / 1.02 | pass | compared: v2 wins | -0.28 | 185.30 (0) / -0.77 (12) | music,sfx,abstract | no |
| vaporwave | vaporwave_r2 | new | a |  | 36.58 (35.76-38.29) | 19.00 (17.25-21.43) | yyy/yyy | 0.68 / 0.68 | 0.40 / 0.68 | 0.95 / 0.90 | pass | not compared (no counterpart) | -0.79 | 0.97 (15) / -0.62 (6) | (none measured >= 0.6) | yes |
| virtuosic_runs | virtuosic_runs | new | d |  | 167.14 (161.54-172.59) | 165.45 (160.45-174.01) | yyy/yyy | 0.84 / 0.92 | 0.82 / 0.56 | 0.98 / 1.01 | pass | not compared (no counterpart) | +0.01 | 1.31 (11) / -1.04 (13) | music,sfx,abstract | yes |
| warm | legacy_warm | legacy | legacy |  | 15.80 (15.79-16.39) | 20.34 (18.32-21.13) | yyy/yyy | 0.38 / 0.62 | 0.38 / 0.34 | 1.00 / 0.99 | FAIL | not compared (no counterpart); fails the bar: ships nothing | +0.06 | 1.14 (7) / -1.30 (10) | (none measured >= 0.6) | no |
| wide_stereo | v1_wide_stereo | v1 | a |  | 24.54 (23.25-30.89) | 37.41 (37.31-42.69) | yyy/yyy | 0.64 / 0.48 | 0.46 / 0.38 | 0.93 / 0.96 | FAIL | neither vector passes the bar: ships nothing | +0.36 | 0.65 (14) / -1.38 (6) | sfx | no |
| wide_stereo | wide_stereo | v2gate | a |  | 23.46 (21.92-29.38) | 35.17 (34.90-40.17) | yyy/yyy | 0.68 / 0.48 | 0.40 / 0.38 | 0.94 / 0.97 | FAIL | neither vector passes the bar: ships nothing | -0.07 | 0.67 (14) / -1.29 (7) | sfx | no |
| woody_body | woody_body | new | d |  | 235.05 (224.30-261.00) | 220.70 (202.01-225.67) | yyy/yyy | 0.90 / 0.86 | 0.78 / 0.66 | 1.00 / 0.98 | pass | not compared (no counterpart) | +1.17 | 1.30 (9) / -2.39 (4) | music,sfx | yes |

## Regression gate

| concept | v1 | v2 | legacy | outcome | why |
|---|---|---|---|---|---|
| abs_cinematic | v1_abs_cinematic (bar) |   |  | not compared (no counterpart) |  |
| abs_forest |   | abs_forest (bar) |  | not compared (no counterpart) |  |
| abs_industrial | v1_abs_industrial (bar) |   |  | not compared (no counterpart) |  |
| accelerando |   | accelerando (bar) |  | not compared (no counterpart) |  |
| acid_303 | v1_acid_303 (bar) |   |  | not compared (no counterpart) |  |
| acid_house |   | acid_house (bar) |  | not compared (no counterpart) |  |
| aggressive | v1_aggressive (bar) | aggressive (bar) |  | compared: v2 wins |  |
| alarm_clock | v1_alarm_clock (bar) | alarm_clock (bar) |  | compared: v1 kept | neg clap 0.56 < v1 0.90 - 0.05; neg muq 0.48 < v1 0.68 - 0.05 |
| analog | v1_analog (bar) | analog  |  | decided by the bar: only v1 passes | pos clap 0.68 < v1 0.80 - 0.05; pos muq 0.46 < v1 0.60 - 0.05; neg clap 0.72 < v1 0.84 - 0.05; neg muq 0.32 < v1 0.46 - 0.05 |
| arpeggiated |   | arpeggiated_r2 (bar) |  | not compared (no counterpart) |  |
| bedroom_intimate |   | bedroom_intimate (bar) |  | not compared (no counterpart) |  |
| black_metal | v1_black_metal (bar) | black_metal (bar) |  | compared: v2 wins |  |
| bright (legacy) |   |   | legacy_bright | not compared (no counterpart) |  |
| childrens | v1_childrens (bar) |   |  | not compared (no counterpart) |  |
| christmas |   | christmas (bar) |  | not compared (no counterpart) |  |
| clave_pattern |   | clave_pattern (bar) |  | not compared (no counterpart) |  |
| close_mic | v1_close_mic  | close_mic  |  | neither vector passes the bar: ships nothing | pos clap 0.18 < v1 0.42 - 0.05; pos muq 0.46 < v1 0.76 - 0.05; neg clap 0.10 < v1 0.26 - 0.05; neg muq 0.34 < v1 0.66 - 0.05 |
| cumbia |   | cumbia  |  | not compared (no counterpart); fails the bar: ships nothing |  |
| dark |   | dark (bar) |  | not compared (no counterpart) |  |
| dense_arrangement (legacy) |   |   | legacy_density | not compared (no counterpart) |  |
| dishes |   | dishes (bar) |  | not compared (no counterpart) |  |
| dissonant | v1_dissonant (bar) | dissonant (bar) |  | compared: v1 kept | pos clap 0.62 < v1 0.76 - 0.05; neg clap 0.80 < v1 0.88 - 0.05; neg muq 0.50 < v1 0.62 - 0.05 |
| distant | v1_distant (bar) |   |  | not compared (no counterpart) |  |
| dreamy | v1_dreamy (bar) | dreamy  |  | decided by the bar: only v1 passes | pos muq 0.46 < v1 0.60 - 0.05 |
| drone | v1_drone (bar) | drone (bar) |  | compared: v1 kept | pos clap 0.84 < v1 0.92 - 0.05; pos muq 0.66 < v1 0.72 - 0.05; neg clap 0.58 < v1 0.80 - 0.05; neg muq 0.54 < v1 0.60 - 0.05; neg LPAPS/cutoff 1.00 > 1.1 x v1 0.91 |
| drum_machine |   | drum_machine (bar) |  | not compared (no counterpart) |  |
| dub_space_echo |   | dub_space_echo (bar) |  | not compared (no counterpart) |  |
| electric_piano |   | electric_piano_r2 (bar) |  | not compared (no counterpart) |  |
| erhu |   | erhu (bar) |  | not compared (no counterpart) |  |
| explosion |   | explosion (bar) |  | not compared (no counterpart) |  |
| fizzy_highs |   | fizzy_highs (bar) |  | not compared (no counterpart) |  |
| frogs |   | frogs (bar) |  | not compared (no counterpart) |  |
| future_bass | v1_future_bass (bar) | future_bass (bar) |  | compared: v2 wins |  |
| fuzzy |   | fuzzy (bar) |  | not compared (no counterpart) |  |
| glitch_sfx |   | glitch_sfx (bar) |  | not compared (no counterpart) |  |
| glitchy | v1_glitchy (bar) | glitchy (bar) |  | compared: v1 kept | pos clap 0.90 < v1 0.96 - 0.05; pos muq 0.88 < v1 0.96 - 0.05; neg clap 0.72 < v1 0.84 - 0.05; neg muq 0.50 < v1 0.64 - 0.05 |
| gritty | v1_gritty (bar) |   |  | not compared (no counterpart) |  |
| hammered_notes |   | hammered_notes (bar) |  | not compared (no counterpart) |  |
| handclaps |   | handclaps (bar) |  | not compared (no counterpart) |  |
| hardstyle | v1_hardstyle (bar) | hardstyle (bar) |  | compared: v2 wins |  |
| horror_score | v1_horror_score (bar) |   |  | not compared (no counterpart) |  |
| impact_boom | v1_impact_boom (bar) | impact_boom  |  | decided by the bar: only v1 passes | pos clap 0.38 < v1 0.90 - 0.05; pos muq 0.64 < v1 0.92 - 0.05; neg clap 0.26 < v1 0.72 - 0.05; neg muq 0.32 < v1 0.46 - 0.05 |
| intense | v1_intense (bar) | intense (bar) |  | compared: v2 wins |  |
| jet | v1_jet (bar) | jet (bar) |  | compared: v1 kept | neg clap 0.66 < v1 0.72 - 0.05 |
| jingle_bells | v1_jingle_bells (bar) | jingle_bells (bar) |  | compared: v2 wins |  |
| koto | v1_koto (bar) |   |  | not compared (no counterpart) |  |
| laser_zap |   | laser_zap (bar) |  | not compared (no counterpart) |  |
| legato_phrasing |   | legato_phrasing (bar) |  | not compared (no counterpart) |  |
| live_recording | v1_live_recording  | live_recording  |  | neither vector passes the bar: ships nothing | neg clap 0.40 < v1 0.52 - 0.05 |
| locked_groove |   | locked_groove (bar) |  | not compared (no counterpart) |  |
| lofi_hip_hop | v1_lofi_hip_hop (bar) | lofi_hip_hop (bar) |  | compared: v1 kept | neg muq 0.64 < v1 0.70 - 0.05 |
| magic_sparkle |   | magic_sparkle (bar) |  | not compared (no counterpart) |  |
| mains_hum | v1_mains_hum (bar) | mains_hum (bar) |  | compared: v1 kept | neg clap 0.66 < v1 0.84 - 0.05 |
| marcato |   | marcato (bar) |  | not compared (no counterpart) |  |
| muted_horn |   | muted_horn (bar) |  | not compared (no counterpart) |  |
| new_age |   | new_age (bar) |  | not compared (no counterpart) |  |
| nylon_soft |   | nylon_soft (bar) |  | not compared (no counterpart) |  |
| palm_muted |   | palm_muted (bar) |  | not compared (no counterpart) |  |
| percussive (legacy) |   |   | legacy_percussive | not compared (no counterpart) |  |
| phone_ring | v1_phone_ring (bar) | phone_ring (bar) |  | compared: v1 kept | neg clap 0.76 < v1 0.92 - 0.05 |
| plate_reverb |   | plate_reverb (bar) |  | not compared (no counterpart) |  |
| playful | v1_playful (bar) |   |  | not compared (no counterpart) |  |
| polyrhythmic | v1_polyrhythmic (bar) | polyrhythmic (bar) |  | compared: v2 wins |  |
| post_rock | v1_post_rock (bar) | post_rock (bar) |  | compared: v2 wins |  |
| pulsing_synth |   | pulsing_synth (bar) |  | not compared (no counterpart) |  |
| quantized |   | quantized (bar) |  | not compared (no counterpart) |  |
| radio_filtered | v1_radio_filtered (bar) | radio_filtered (bar) |  | compared: v1 kept | neg clap 0.72 < v1 0.78 - 0.05 |
| rain_on_surface | v1_rain_on_surface  | rain_on_surface  |  | neither vector passes the bar: ships nothing |  |
| revving |   | revving (bar) |  | not compared (no counterpart) |  |
| riser | v1_riser (bar) | riser  |  | decided by the bar: only v1 passes | pos clap 0.54 < v1 0.68 - 0.05; pos muq 0.56 < v1 0.80 - 0.05; neg clap 0.48 < v1 0.58 - 0.05; neg muq 0.48 < v1 0.54 - 0.05 |
| rough (legacy) |   |   | legacy_rough | not compared (no counterpart) |  |
| samba | v1_samba (bar) |   |  | not compared (no counterpart) |  |
| sensual | v1_sensual  |   |  | not compared (no counterpart); fails the bar: ships nothing |  |
| sfx_applause |   | sfx_applause (bar) |  | not compared (no counterpart) |  |
| sfx_baby_cry |   | sfx_baby_cry (bar) |  | not compared (no counterpart) |  |
| sfx_event_rate |   | sfx_event_rate (bar) |  | not compared (no counterpart) |  |
| sfx_far_away |   | sfx_far_away (bar) |  | not compared (no counterpart) |  |
| sfx_indoor_hall |   | sfx_indoor_hall (bar) |  | not compared (no counterpart) |  |
| sfx_insects |   | sfx_insects (bar) |  | not compared (no counterpart) |  |
| sfx_laden | v1_sfx_laden  | sfx_laden  |  | neither vector passes the bar: ships nothing |  |
| sfx_ocean_waves |   | sfx_ocean_waves (bar) |  | not compared (no counterpart) |  |
| sfx_sub_weight |   | sfx_sub_weight (bar) |  | not compared (no counterpart) |  |
| sfx_telephone |   | sfx_telephone  |  | not compared (no counterpart); fails the bar: ships nothing |  |
| shaker_pulse |   | shaker_pulse (bar) |  | not compared (no counterpart) |  |
| shimmer_reverb |   | shimmer_reverb (bar) |  | not compared (no counterpart) |  |
| shimmering |   | shimmering_r2 (bar) |  | not compared (no counterpart) |  |
| sitar |   | sitar_r2 (bar) |  | not compared (no counterpart) |  |
| smash |   | smash  |  | not compared (no counterpart); fails the bar: ships nothing |  |
| solo_build |   | solo_build (bar) |  | not compared (no counterpart) |  |
| solo_crest |   | solo_crest (bar) |  | not compared (no counterpart) |  |
| solo_instrument | v1_solo_instrument (bar) | solo_instrument (bar) |  | compared: v2 wins |  |
| spiccato |   | spiccato (bar) |  | not compared (no counterpart) |  |
| strummed_guitar | v1_strummed_guitar (bar) | strummed_guitar (bar) |  | compared: v1 kept | pos muq 0.78 < v1 0.92 - 0.05; neg clap 0.68 < v1 0.82 - 0.05; neg muq 0.34 < v1 0.44 - 0.05 |
| surf_rock | v1_surf_rock (bar) | surf_rock (bar) |  | compared: v2 wins |  |
| swing |   | swing_r2 (bar) |  | not compared (no counterpart) |  |
| synth_pad | v1_synth_pad (bar) | synth_pad (bar) |  | compared: v1 kept | neg clap 0.74 < v1 0.84 - 0.05 |
| synthesizer | v1_synthesizer (bar) | synthesizer (bar) |  | compared: v1 kept | pos muq 0.62 < v1 0.76 - 0.05 |
| tango | v1_tango (bar) | tango  |  | decided by the bar: only v1 passes | pos clap 0.28 < v1 0.70 - 0.05; pos muq 0.66 < v1 0.78 - 0.05; neg clap 0.62 < v1 0.68 - 0.05; neg LPAPS/cutoff 1.01 > 1.1 x v1 0.89 |
| thick_unison |   | thick_unison  |  | not compared (no counterpart); fails the bar: ships nothing |  |
| timpani | v1_timpani (bar) | timpani  |  | decided by the bar: only v1 passes | pos clap 0.34 < v1 0.88 - 0.05; pos muq 0.84 < v1 0.90 - 0.05; neg clap 0.54 < v1 0.74 - 0.05 |
| tom_patterns |   | tom_patterns (bar) |  | not compared (no counterpart) |  |
| ui_click | v1_ui_click (bar) | ui_click (bar) |  | compared: v2 wins |  |
| ukulele | v1_ukulele (bar) | ukulele (bar) |  | compared: v2 wins |  |
| vaporwave |   | vaporwave_r2 (bar) |  | not compared (no counterpart) |  |
| virtuosic_runs |   | virtuosic_runs (bar) |  | not compared (no counterpart) |  |
| warm (legacy) |   |   | legacy_warm | not compared (no counterpart); fails the bar: ships nothing |  |
| wide_stereo | v1_wide_stereo  | wide_stereo  |  | neither vector passes the bar: ships nothing | pos muq 0.40 < v1 0.46 - 0.05 |
| woody_body |   | woody_body (bar) |  | not compared (no counterpart) |  |

## Stacking (300 combos over the final set, LPAPS vs the strictest member's cutoff)

| rule | combos | fraction under cutoff | median scale |
|---|---|---|---|
| none | 300 | 0.04 | 1.0 |
| inv_sqrt | 300 | 0.2733333333333333 | 0.7071067811865476 |
| inv_n | 300 | 0.8133333333333334 | 0.5 |
| norm_budget | 300 | 0.12333333333333334 | 0.8088613659577752 |

Chosen: inv_n (no rule reached 95%; the best observed is named). norm_budget: the combined perturbation's Frobenius norm over all steps and blocks is held to the largest single member's.

## SFX prompt pass (12 SFX prompts x 3 seeds at the shipped gain; median intended-sign fraction; recorded, not gating)

| vector | CLAP + | CLAP - | MuQ + | MuQ - |
|---|---|---|---|---|
| alarm_clock | 0.83 | 1.00 | 0.75 | 0.75 |
| dishes | 0.83 | 0.83 | 0.67 | 0.92 |
| explosion | 0.92 | 0.83 | 0.67 | 0.67 |
| frogs | 0.50 | 0.92 | 0.50 | 0.75 |
| glitch_sfx | 1.00 | 0.67 | 0.75 | 0.83 |
| impact_boom | 0.33 | 0.83 | 0.67 | 0.50 |
| jet | 0.92 | 0.92 | 0.25 | 0.67 |
| laser_zap | 0.92 | 0.92 | 0.75 | 0.08 |
| magic_sparkle | 0.92 | 0.92 | 1.00 | 0.42 |
| phone_ring | 0.92 | 0.75 | 0.42 | 0.83 |
| rain_on_surface | 0.83 | 0.83 | 0.75 | 0.67 |
| revving | 0.83 | 0.83 | 0.33 | 0.58 |
| riser | 0.67 | 0.75 | 0.58 | 0.75 |
| sfx_applause | 1.00 | 0.75 | 0.83 | 0.50 |
| sfx_baby_cry | 0.92 | 0.75 | 1.00 | 0.75 |
| sfx_insects | 1.00 | 0.50 | 0.50 | 0.75 |
| sfx_ocean_waves | 0.92 | 0.67 | 0.83 | 0.67 |
| sfx_telephone | 0.92 | 0.08 | 0.50 | 0.25 |
| smash | 0.92 | 0.83 | 0.33 | 0.50 |
| ui_click | 0.67 | 0.75 | 0.67 | 0.83 |
| v1_alarm_clock | 0.83 | 1.00 | 0.67 | 0.67 |
| v1_impact_boom | 1.00 | 0.75 | 1.00 | 0.67 |
| v1_jet | 0.92 | 0.92 | 0.33 | 0.67 |
| v1_phone_ring | 0.92 | 0.92 | 0.33 | 0.83 |
| v1_rain_on_surface | 0.92 | 0.75 | 0.67 | 0.67 |
| v1_riser | 0.92 | 0.83 | 0.67 | 0.67 |
| v1_ui_click | 0.58 | 0.83 | 0.67 | 0.75 |

## Notes and deviations

- C parts reused from last night (gain within 5%): applicability/hold for 0 new knobs, cross-effect for 0; reused cross-effect rows lack label columns added to the universe for the old vectors (n_cols shows it).
- v2gate: the v2 vectors that lost last night's gate (and sfx_laden v2) were measured too, so the gate compares like with like; their missing seed-2 protocol runs were added to P0.
- v1 vectors were measured with their v2 counterpart's PCI descriptors and anchors (PCI results reused per seed where last night rendered them); the 11 v1 concepts without a v2 counterpart use the catalogue anchors.
- Hold drift and cross-effect are reported, not gating. applies_to is empty when no category passes.
- c packs: neg gain in the dn vector's own magnitude units.
