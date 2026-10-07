# TADA (arXiv 2602.11910) numbers: SAO vs ACE-Step

Source: tada.pdf in a prior session scratchpad (50044f95-.../scratchpad/tada.pdf). Page numbers = PDF page.
Headline metric: AUC of sign-corrected alignment delta (MuQ-text or CLAP-text similarity) vs LPAPS preservation, steering both directions, mean over concepts.
LPAPS cutoff: no fixed number. Each method's strength range is calibrated so its max LPAPS distortion equals the LPAPS distance between two audios generated from different text prompts (= PCI at max strength) (Sec 5.2, p.6). PQ/Audio Quality is Audiobox Aesthetics.
Note: the paper's metric is MuQ / CLAP text alignment, NOT "MuQ-MuLan".

## Key finding: SAO has NO method comparison in the paper
Stable Audio Open appears only in (a) localization (layers {12,13,14}, Fig 8) and (b) CAA all vs localized vs ablated: Table 23, p.39, 4 concepts only.
There is no PCI / SAE / Concept Sliders / AUSteer row for SAO, no per-method ratio vs PCI is computable, and no SAO human study.
The human study (Fig 4 p.7, App J, Table 21) was ACE-Step only: 32 listeners, 1279 ratings, 15 trials, concepts piano, female vocals, tempo.
Seamless Edit means: CAA(loc) 3.32, AUSteer(loc) 3.24, SAE 3.22 (text p.7). Table 21 (trained vs casual, loc variants): PCI 3.20/2.80, TextEmb 2.69/2.32, TokenEmb 2.64/2.58, FreeSliders 3.38/2.71, CAA 3.23/3.40, AUSteer 3.10/3.42, SAE 3.27/3.17.

## Table 23 (p.39): SAO, CAA, 4 concepts (piano, tempo, mood, female vocal)
| Concept | all MuQ | all CLAP | loc MuQ | loc CLAP | ablated MuQ | ablated CLAP |
|---|---|---|---|---|---|---|
| Piano | 0.792 | 0.228 | 0.742 | 0.220 | 0.383 | 0.094 |
| Tempo | 0.268 | 0.108 | 0.323 | 0.216 | 0.183 | 0.031 |
| Mood | 0.074 | 0.241 | 0.119 | 0.246 | 0.034 | 0.163 |
| Female vocal | 0.106 | 0.122 | 0.153 | 0.097 | 0.030 | 0.087 |
| Average | 0.310 | 0.175 | 0.334 | 0.195 | 0.158 | 0.094 |
Localization on SAO: +8% MuQ, +11% CLAP vs all layers (CAA only). Vocal concept is in SAO's list, and SAO AUC values are on a different scale from ACE (different LPAPS calibration per model), so do not compare absolute numbers across models.

## Table 24 (p.39): ACE-Step, CAA, same 4 concepts
| Concept | all MuQ/CLAP | loc MuQ/CLAP |
|---|---|---|
| Piano | 0.060/0.026 | 0.141/0.061 |
| Tempo | 0.095/0.022 | 0.111/0.023 |
| Mood | 0.043/0.049 | 0.049/0.046 |
| Female vocal | 0.020/0.104 | 0.029/0.116 |
| Average | 0.055/0.051 | 0.083/0.061 |
(+51% MuQ, +20% CLAP localization gain on these 4.)

## ACE-Step, 9 concepts: Table 1 (p.7); per-concept Tables 26-34 (pp.41-43, avg of +/- directions)
Concepts (9): piano (T26), tempo (T27), mood (T28), vocal gender (T29), vocal style rap/sing (T30), violin (T31), guitar type acoustic/electric (T32), jazz genre (T33), classical genre (T34). N=100 30-s tracks, 15 pos + 15 neg strengths.
Per-concept AUC-MuQ avg (parsed from Tables 26-34; means reproduce Table 1 exactly):
| Method | piano | tempo | mood | vox gender | vox style | violin | guitar | jazz | classical | MEAN MuQ | MEAN CLAP | MuQ ratio vs PCI | CLAP ratio vs PCI |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| PCI | .054 | .063 | .027 | .027 | .073 | .083 | .079 | .206 | .144 | 0.084 | 0.049 | 1.00 | 1.00 |
| Text Emb | .030 | .013 | .004 | -.017 | -.010 | .041 | .009 | .021 | .034 | 0.014 | 0.011 | 0.17 | 0.22 |
| Token Emb | .084 | .018 | .001 | -.007 | .026 | .087 | .044 | .010 | .050 | 0.035 | 0.024 | 0.42 | 0.49 |
| FreeSliders | .067 | .073 | .030 | .035 | .068 | .077 | .087 | .130 | .141 | 0.079 | 0.049 | 0.94 | 1.00 |
| Concept Sliders | .078 | .062 | .051 | .076 | .089 | .106 | .089 | .152 | .073 | 0.086 | 0.067 | 1.02 | 1.37 |
| AUSteer (all) | .049 | .092 | .026 | .019 | .069 | .032 | .099 | .187 | .156 | 0.081 | 0.058 | 0.96 | 1.18 |
| CAA (all layers) | .052 | .084 | .046 | .012 | .073 | .048 | .060 | .168 | .134 | 0.075 | 0.046 | 0.89 | 0.94 |
| AUSteer (loc) | .102 | .080 | .053 | .031 | .072 | .104 | .077 | .185 | .159 | 0.096 | 0.063 | 1.14 | 1.29 |
| CAA (loc) | .107 | .105 | .049 | .018 | .091 | .108 | .090 | .206 | .165 | 0.104 | 0.065 | 1.24 | 1.33 |
| SAE (loc) | .140 | .110 | .046 | .031 | .110 | .130 | .137 | .222 | .137 | 0.118 | 0.070 | 1.40 | 1.43 |
Table 1 PCI CLAP is 0.049; CLAP ratios use Table 1 values (CLAP column of the per-concept sums gives 0.048 for PCI, rounding). SAE exists only localized in the benchmark (no all-layer SAE row). Per-concept CLAP not tabulated here; see Tables 26-34 for the CLAP columns.
Table 1 smoothness/quality omitted. Best: SAE(loc) 0.118 MuQ / 0.070 CLAP; Concept Sliders is CLAP runner-up (0.067).
Text parse caveat: pdftotext row labels in Tables 1 and 26-34 are shifted by one line; mapping was validated by reproducing every Table 1 mean.

## Caveat for SA3 (vocals)
Paper itself says concepts must be ones the model can generate (App, p.~22). Vocal concepts (vocal gender, vocal style; female vocal on SAO) are 2 of the 9 ACE concepts and 1 of 4 SAO concepts. They are the weakest ACE concepts for every method (vox gender CAA .012-.018 MuQ). Dropping them for SA3 shifts means upward and compresses method differences; recompute means over the 7 non-vocal ACE concepts before comparing to SA3 results.

## ACE-Step recomputed on the 7 non-vocal concepts (my arithmetic from Tables 26-34; drops vocal gender + vocal style)
PCI is 0.094 MuQ / 0.045 CLAP. Ratios vs PCI (MuQ / CLAP): TextEmb 0.23/0.33, TokenEmb 0.45/0.62, FreeSliders 0.91/1.04, Concept Sliders 0.93/1.47, AUSteer 0.98/1.13, CAA(all) 0.90/0.96, AUSteer(loc) 1.16/1.40, CAA(loc) 1.27/1.44, SAE(loc) 1.40/1.58 (absolute SAE 0.132/0.071, CAA loc 0.119/0.065).
