# Prior SAE / concept-steering work in instrument-controlnet (archaeology, 2026-10-03)

Read-only survey of `C:\_dev\projects\instrument-controlnet`. Nothing there was modified.

## Where the material lives (read this first)

- **All SAE code and specs are on the local branch `feat/synesthesia-lane` only.** It is not
  checked out anywhere (its worktree `C:\_dev\projects\instrument-controlnet-synesthesia` no
  longer exists) and it has no remote counterpart (origin has `master` only). Tip `20d4b746`
  (2026-09-01, "Knob stage 3 artifacts: editor probe KILL ..."). A pre-secret-purge copy of the
  same tip exists as `refs/backup/pre-secret-purge/feat/synesthesia-lane` = `32730f4c` (memory
  files cite 32730f4). Read files with `git show feat/synesthesia-lane:<path>`. All `file:line`
  references below to specs, scripts and results point at that branch.
- **The narrative (results, verdicts, numbers) is on `master`** in
  `orchestration/legacy/comms/*synesthesia*` and `*concept*` (also mirrored at
  `C:\datasets\orchestration\comms\`), plus `orchestration/legacy/status/concept.json`.
  Comms references below are `comms/<file>:<line>`, relative to `orchestration/legacy/`.
- Claude memory for that project:
  `C:\Users\ryanf\.claude\projects\C---dev-projects-instrument-controlnet\memory\`
  (`sae2-launched.md`, `sae2-kill-test-outcome.md`, `subspace-probe-outcome.md`,
  `concept-knob-editor-kill.md`, `concept-lane-b2-archive.md`).
- Funding-review summary of the closure: `orchestration/legacy/funding/harvest/F_closed_lanes.md:23-34` (master).
- Lane names: "synesthesia" (2026-08-20 to 09-01), renamed "concept knob" / `concept` on
  2026-09-01T13:14Z (`comms/2026-09-01T1314Z_concept_FLAG-lane-renamed...md:5-15`). Same session
  (`525fcbc4`), same branch.
- **Model in every attempt: Stable Audio 3 medium-base** (`sa3-medium-base`, 24 trunk layers,
  width 1536, 323-token audio tail; `scripts/sae_harvest.py:58-64`). No ACE-Step, MiniMax, YuE or
  other DiT was ever SAE'd: `git grep` over every local branch for
  "sparse auto|BatchTopK|JumpReLU|dictionary learning" hits only `feat/synesthesia-lane` and one
  comms file. BatchTopK and JumpReLU were never tried; every SAE was plain TopK.

## 1. Inventory

| date | what | where (branch/commit) | one line |
|---|---|---|---|
| 08-20 | Activation transplant smoke (rung 1, not an SAE) | `TRANSPLANT_SPEC.md`, `TRANSPLANT_RESULTS.md`, `scripts/transplant_generate.py` (1921fce..6838e44) | Graft trunk activations from gen A into gen B mid-sampling. Worked: CLAP possession -0.7 -> +0.3..+0.5 |
| 08-20 | Transplant battery 2 | `BATTERY2_SPEC.md`, `BATTERY2_RESULTS.md`, `scripts/transplant_battery2.py` (e2379791..ba2a367a) | Release-point frontier: musical up to release step 24, cliff at 24-29 |
| 08-20 | SAE spec v1 + v1 harvest (sigma 0.05) | `SAE_SPEC.md` (332449d3/242f9f88), `scripts/sae_harvest.py`, `sae_train.py`, `sae_browse.py` (27dbca55/b4bb37a9) | TopK 12,288 x k32 on L11/17/22. Harvest was off-policy. Kept only as a code shakedown (inferred: no v1 dictionary result is recorded anywhere) |
| 08-21 | SAE plan v2 + SAE2 run | `SAE_PLAN_V2.md`, `SAE2_RUNBOOK.md`, `scripts/sae_onpolicy_capture.py`, `sae_eval.py`, `scripts/box/sae2_driver.sh` (cc81babc, acec1c0a, 8a38ebb1 = 7f2e866 pre-purge) | euler50-sigma harvest, L17+L22 TopK 12,288 x k32, 50k steps. FVU 0.41: quality gate FAIL |
| 08-30 | SAE2H harvest regen + k-sweep | `SAE2H_RUNBOOK.md`, `scripts/box/sae2h_driver.sh`, `sae2h_ksweep_driver.sh` (e7d798ba..262b06f9) | 128 tok/window, 63.3 GB. dict 24,576, k 64/96/128 (and 192). k=128 reaches FVU 0.16-0.19 |
| 08-30 | Phase R triage + single-feature causal screen (kill test) | `TRIAGE_SPEC.md` + Amendment 1, `scripts/sae_features.py`, `sae_contexts.py`, `sae_descriptors.py`, `sae_regress.py`, `sae_steer.py`, `sae_screen_score.py`, `scripts/box/sae2h_triage_driver.sh` (fecc85fd, e598d154, d19913e0..5ee709f) | KILL: 0/45 playable on both k=128 dictionaries; ear concurs |
| 08-30 | Subspace (learned-direction) steering probe | `SUBSPACE_PROBE_SPEC.md`, `SUBSPACE_RUNBOOK.md`, `scripts/sae_direction_fit.py`, `sae_subspace_summary.py`, `scripts/box/sae2h_subspace_driver.sh` (cd7b24a, b47031a, 598382a..77f038b) | DEAD: context-general but only 0.35-0.62 std |
| 08-30/31 | Vision-texture stitch probe | `STITCH_PROBE_SCOPE.md`, `STITCH_PREREG.md`, `scripts/stitch_*.py`, `results/stitch_screen/` (f5149ee..417a5ec) | STOP at the CLAP screen (13/43 concepts render). No SAE involved; the lane's vocabulary/render tooling comes from here |
| 09-01 | Concept knob: SAE editor probe | `CONCEPT_KNOB_PREREG.md` + Amendments A-C, `scripts/knob_*.py`, `results/knob_screen/`, `results/knob_stage2/`, `results/knob_stage3/` (bdb9e7e8..20d4b746) | Scale word-linked SAE features where active. KILL: audible but sign-inverted, non-specific, destructive |
| 08-11 | (related, separate repo) LoRA concept-slider lane | `C:\_dev\projects\underfit`; memory `slider-lane-ledger.md`; status `orchestration/legacy/status/lora-slider.json` | Weight-space sliders, not SAE. Room LoRA shipped (ear-confirmed), articulation went training-free (embedding blend). Not examined further here |

## 2. Each SAE attempt in detail

Common to all: hook = `register_forward_hook` on `model.model.model.transformer.layers[i]`, i.e.
the **residual-stream output of trunk block i**, keeping only the last 323 positions (the audio
tail; the 64 memory/prepend tokens are dropped) (`scripts/sae_harvest.py:224-253`, slice at
`:248`). Activations are raw fp32 block outputs: no normalization or whitening; `b_dec` is set to
the first batch mean (`scripts/sae_train.py:240-243`). Corpus for harvests = SAME-L latent caches
of real audio, noised and pushed through one trunk forward under a neutral prompt "instrumental
band recording", cfg 1.0 (`sae_harvest.py:5-13,62`, noise blend `:261-283`).

SAE class (`scripts/sae_train.py:1-8,51-84`): TopK, ReLU on the top-k values, unit-norm decoder
rows renormalized after each step, tied init (W_enc = W_dec at init), pre-encoder `b_dec`
subtraction. **Loss is plain MSE, with no AuxK or L1 term** (`:6`, `:246`). Dead features are
handled by Anthropic-style resampling toward high-error rows every 4,000 steps, with dead
meaning no fire in 800 steps (`:86-108`, `:191-194`). Adam lr 3e-4, betas (0.9, 0.999), batch
8,192 (`:183-184,208`). Metrics: FVU, L0, dead fraction (`:256-269`).

Eval (`scripts/sae_eval.py:1-21,36-37,130-142`): FVU overall and per sigma bucket
(t_idx // 10, so b0..b4, where b0 is highest sigma). The deployment gate fails if on-policy FVU
exceeds real-val FVU + 0.05 overall or in any bucket.

### 2a. SAE v1 (2026-08-20): spec'd, harvested, superseded

- Spec: L11/17/22, sigma fixed 0.05, 64 tokens/window, ~26k windows x 2 latents x 64 tokens
  ~= 3.3M rows/layer, ~10 GB/layer fp16 (`SAE_SPEC.md:9-30`). TopK d_in 1536, dict 12,288 (8x),
  k=32, ~50k steps, batch 8192. Gates: FVU <= 0.15, dead < 20%, 10/20 browse coherence
  (`SAE_SPEC.md:32-46`). MoisesDB only (79,315 files / 17.6 GB cache, `SAE_SPEC.md:11-15`).
- **Why it failed (recorded):** "the sampler's euler-50 grid under medium-base's dist_shift spans
  sigma 1.0000 .. 0.1375 ... The trunk never sees sigma 0.05 during generation, so the v1 harvest
  was even further off-policy than the plan assumed" (`SAE_PLAN_V2.md:18-21`). Other named flaws:
  rock/pop-only corpus, no causal screen, global triage (`SAE_PLAN_V2.md:44-57`). The v1 harvest
  was kept "as a shakedown corpus for the training code, nothing more" (`SAE_PLAN_V2.md:65-67`).

### 2b. SAE2 (2026-08-21): L17 + L22, 12,288 x k32

- Corpus: MoisesDB (train 24,286 / val 2,152 filelist lines) + Jamendo transfer-v3 SAME-L
  (34,377 clips; train 32,731 / val 1,646 lines), Suno rejected (`SAE_PLAN_V2.md:6-16`). Each
  unit gets one sigma drawn from the sampler's own euler-50 grid (`sae_harvest.py:210-218`,
  `--sigma-mode euler50`). Layers 11/17/22, 64 tokens/window, fp16. ~43 GB of shards
  (`comms/2026-08-22T0935Z_synesthesia_shutdown-ready.md:10-12`). On-policy set: 300 real
  euler-50 cfg-7 generations, both CFG rows, captured at a stride of steps, never trained on
  (`scripts/sae_onpolicy_capture.py:1-20`; `SAE_PLAN_V2.md:69-72`).
- Training: L17 and L22, dict 12,288, k=32, 50k steps, batch 8,192, 3.65M rows (L22), 112 epochs
  (`comms/2026-08-30T1226Z_synesthesia_acclimated...md:19-21`). Whole SAE2 sequence (4 harvests,
  on-policy capture, 2 trainings, evals) = **3.3 h on one RTX 5090 (gpu3, box 48209840)**
  (`comms/2026-08-22T0935Z...:5`).
- Results, verbatim: "both dictionaries trained to 50k steps, dead features 0.0%; deployment gate
  (on-policy vs real-val FVU, margin 0.05) PASSES for L22 in every sigma bucket, L17 fails only
  the highest-sigma bucket. But absolute FVU is ~0.38-0.45 vs the preregistered <= 0.15 quality
  bar, so the dictionary-quality gate FAILS" (`comms/2026-08-22T0935Z...:20-24`). L17 b0 = 0.506
  vs 0.429 (memory `sae2-launched.md:32-33`). "L22 train FVU flat at 0.386 by step 12.5k and only
  0.381 at 50k ... MORE STEPS IS DEAD; the levers are k and dict size"
  (`comms/2026-08-30T1226Z...:19-21`).
- Recorded reason it did not work: dictionary quality (FVU 0.41) with k=32 too sparse. Its
  activation shards were lost when box 48209840 was deleted
  (`comms/2026-08-30T1226Z...:5-11`).

### 2c. SAE2H + k-sweep (2026-08-30): L22 (and L17), 24,576 x k{64,96,128,192}

- Harvest regenerated on cluster 49257231: layers 17+22 only, 128 tokens/window, same euler50
  recipe. **48 min wall on 2x RTX 5090, 63.3 GB** (`comms/2026-08-30T1555Z...:5-8`). Runbook
  `SAE2H_RUNBOOK.md`. Val rows used later for fitting: ~530k real-audio rows
  (`TRIAGE_SPEC.md:187-188`).
- Training: dict 24,576, 20k steps each, ~40 min each (`comms/2026-08-30T1226Z...:23-25`;
  `scripts/box/sae2h_ksweep_driver.sh:2-3`). Results table, verbatim
  (`comms/2026-08-30T1716Z_synesthesia_ksweep-readout...md:7-15`):

  | k | val FVU moises | val FVU jamendo | on-policy FVU | gate |
  |---|---|---|---|---|
  | 32* (old 12,288, 50k) | 0.410 | 0.415 | 0.429 | PASS |
  | 64 | | 0.283 | 0.296 | PASS |
  | 96 | 0.213 | 0.226 | 0.238 | PASS |
  | 128 | 0.158 | 0.174 | 0.185 | PASS |

  L17 k=96: 0.223 / 0.230 / 0.258, failing only the highest-sigma bucket. Dead features 0.0%
  everywhere. "FVU falls almost linearly in k ... Steps were never the issue (curves flat by
  ~12k)" (`:17-19`). Whole sweep cost ~3 GPU-h (`:21`).
- k=192 **diverged**: "train FVU 0.05 but val FVU 4-33x variance in the two lowest-sigma buckets;
  the on-policy gate cannot see it because it compares on-policy to real val (both broken). Any
  SAE gate needs an absolute bucket check too" (`comms/2026-08-30T1813Z...:40-43`).
- Final dictionaries kept: L22 and L17 at 24,576 x k128 (L22 val FVU 0.16/0.17, L17 0.18/0.19,
  `comms/2026-08-30T1813Z...:33-34`). L22 checkpoint identity: step 20000, sha256
  `752c237a...331fd` (`results/knob_stage3/sae_info.json`).
- The dictionaries themselves "reconstruct well" and passed every gate. **What failed was
  everything built on top of them (2d-2f).**

### 2d. Phase R triage + single-feature causal screen = the preregistered kill test (2026-08-30)

- Labelling funnel (`TRIAGE_SPEC.md:43-160`): encode every row; per-stratum firing stats (stratum =
  corpus x kind x stem class x sigma bucket); top-64 contexts per feature; browse smoke (20 random
  features, KL concentration permutation tests on stem class / instruments / t-bucket /
  descriptor, Bonferroni p < 0.01, PASS at >= 10/20); decode every unit to audio and compute
  per-token descriptor banks (SPECTRAL: centroid, rolloff, flatness, rms_db, bandwidth, zcr,
  hi/lo ratio; STRUCTURAL: onset rate, IOI entropy, swing, chroma movement, flux CV, section
  proximity, local tempo); classify features with ridge R2 (DEMON_REDUX if R2_spec > 0.5,
  STRUCTURAL if R2_struct > 0.3, and so on).
- Browse smoke PASSED (L22 15/20, L17 14/20) "but ... 83-90% of features have their
  top-activating contexts in the on-policy (generation) root, which carries no labels or
  descriptors, so their coherence is sigma-bucket only"
  (`comms/2026-08-30T1716Z...:23-30`).
- **The R2 classifier was degenerate:** "Max R2_spec was ~0.17 and no feature reached R2_struct
  0.3 ... STRUCTURAL 0 / DEMON_REDUX 0 / UNKNOWN 8783 / NOISE 3505 ... Features fire on ~0.2%
  of tokens (fire_rate_pooled_median 0.0024), so a linear R2 against a 99.8%-zero log1p target
  is bounded near zero by construction" (`TRIAGE_SPEC.md:187-194`). Amendment 1 replaced it with
  effect-size selection (firing-vs-not Cohen's d >= 0.5, |z| >= 4; `TRIAGE_SPEC.md:197-229`). On
  k=128: L22 467 STRUCTURAL_ALIGNED / 969 SPECTRAL / 130 UNKNOWN / 23,007 NOISE; L17
  295/501/80/23,695; top d +1.6 to +1.9 (`comms/2026-08-30T1815Z...:8-17`).
- Steering screen (`TRIAGE_SPEC.md:118-160`; `scripts/sae_steer.py:9,56-90`):
  `h[:, -323:, :] += lam * p90_f * W_dec[f]` on both CFG rows, layer L, steps 0-24 (early) or
  25-49 (late), lam {1,2,4}, 2 prompts, euler 50, cfg 7.0; lam=0 bit-exact null. 45 features
  (25 structural, 15 unknown, 5 spectral controls) x 3 lam x 2 windows x 2 prompts = 540
  renders per layer. Gates: garble <= 1.2, comb <= 3.0 dB at k x 172.27 Hz, musicality, CLAP
  >= base - 0.05. PLAYABLE = monotone same-sign dose on both prompts and |delta| >= 1.0 std at
  the largest gate-passing lambda.
- Results, verbatim (`comms/2026-08-30T1915Z_synesthesia_KILL-test-verdict...md:8-25`): "k=128
  L22: KILL, 0/45 playable. k=128 L17: KILL, 0/45 playable. Old k=32 dicts ... L22 UNDECIDED
  (1/15, f9719 centroid), L17 KILL (0/15)." "Gates were NOT the killer: 436/540 (L22) and
  502/540 (L17) steered renders pass all four gates." "single-feature steering at natural
  magnitude moves the feature's own descriptor 2-3 std with every gate passing in isolated
  (feature, prompt) pairs: L22 f7338 section_proximity -2.6 std at lam1 on prompt 0; L17 f1259
  -2.5 std at lam2/lam4 on prompt 0; L22 f8346 +1.6 std on prompt 1. The same decoder direction
  then does something different, nothing, or breaks the render on the other prompt." "Late-window
  steering at L17 frequently does nothing."
- Old k=32 extra detail: at lam 4 "Several ... renders went near-silent (NaN descriptors)"; the
  two extra playables found without CLAP "move the audio away from the prompt"
  (`comms/2026-08-30T1813Z...:14-23`).
- Ear: Ryan on the 6-file package of the best all-gates-passing near-misses: "no, i dont really
  hear anything recognizable." Calibration lesson: "descriptor deltas of 1.5-2.6 std ... at
  gate-passing steering strengths are BELOW audibility"
  (`comms/2026-08-30T2028Z...:5-16`).
- **Recorded reason:** no context generality. Single features have context-conditional authority
  only, and the effect sizes that do pass are sub-audible.

### 2e. Subspace / learned-direction probe (2026-08-30, Ryan: "do the multi feature rung")

- Prereg `SUBSPACE_PROBE_SPEC.md`. L22 k=128 only. 12 directions: Variant A = 8 ridge fits
  SAE codes -> descriptor z (5-fold CV over alpha {1e-2..1e3}), `d = W_dec^T w` normalized; Variant
  B = 4 pooled top-16 features by effect size, signed sum of decoder rows (`:51-67`). Fit on
  jamendo_val + moisesdb_val rows (Amendment 1.1, `:133-137`). Hook identical to the kill test,
  `h += lam * s * d`, s = p90 of centred projection, early window only (`:69-75`). 4 prompts,
  lam +-{0.5,1,2} + 0: 288 steered + 4 baselines (`:79-87`). Bars: |rho| >= 0.8 dose on >= 3/4
  prompts same sign, AND one lambda where >= 3/4 prompts pass all gates with |dz| >= 1.0;
  audibility tier 3.0 (`:89-114`).
- Ran 35 min on one 5090 (`comms/2026-08-30T2147Z...:4-6`).
- Results, verbatim (`comms/2026-08-30T2147Z_synesthesia_FLAG-subspace-verdict-DEAD...md:9-25`):
  "DEAD ... no direction passed bars (a)+(b) ... 275/288 steered renders pass all four gates,
  lam=0 bit-exact." "Four learned directions (A_rms_db, A_centroid, B_rms_db, B_hilo_ratio_db;
  B_flux_cv at 3/4) achieved monotone, sign-correct dose response on 4/4 prompts with gates
  passing." "peak gate-passing |delta| 0.35-0.62 std vs the 1.0 bar-(b) floor and the 3.0
  ear-anchored audibility tier." "Read-side fits were strong (CV R2 rms_db .56, centroid .49 vs
  the .17 single-feature ceiling)." ioi_entropy/section_proximity were again "big-but-inconsistent
  ... at 1.8-2.2 std."
- **Recorded reason:** "Pooling many features bought consistency and paid in magnitude" and
  "open-loop injection just cannot move the render enough at gate-passing strengths."

### 2f. Concept knob stage 1-3: SAE "editor" (2026-09-01)

- Prereg `CONCEPT_KNOB_PREREG.md` (goal: dial a concept word up or down on any prompt, replacing
  LECO-style LoRAs and per-attribute adapters, `:3-6`). Stage 1: 69-word borrowed vocabulary
  (Zacharakis 30, AudioCommons 8, Grill, SocialFX top-60), 8 renders/word, CLAP self-rank screen
  (`:9-48`). Result: 16/69 survive: mellow, heavy, dirty, cold, tinny, loud, dark, boomy, cool,
  strong, clean, spacious, warm, hard, soothing, rounded
  (`comms/2026-09-01T1429Z...:5-8`).
- Stage 2: capture L22 at sigma 0.1375, neutral prompt, cfg 1 (Amendment B, `:110-111`); encode
  with k=128; per word, top-16 features by word-vs-pool effect size (`:52-61,113`). "220 unique
  features across 256 slots", top effects mellow +1.23, heavy +0.81, hard +0.73, warm +0.64, boomy
  +0.53 (`comms/2026-09-01T1454Z...:5-9`). Shared features across words: 4342
  (warm/hard/heavy), 13914 (boomy/mellow), 21506 (warm/mellow) (Amendment C.14, `:134`).
- Stage 3: during sampling at L22, both CFG rows, scale active target features by g:
  `h + (g-1) * sum_{active i in set} c_i W_dec[i]` (Amendment C.1, `:121`;
  `scripts/knob_edit.py:17,338-365`). g in {0, 0.5, 2, 4} plus a bit-exact g=1; matched random-16
  and cross-word null arms; 494 renders in 37 min plus 24 min measuring on one 5090
  (`comms/2026-09-01T1616Z...:3`). Referees: AudioCommons timbral_models regressors (warm, hard,
  boomy), CLAP rank (mellow, heavy).
- Results, verbatim (`results/knob_stage3/edit_measure/summary.md:7-13`):

  | word | target | z@g0 | z@g0.5 | z@g2 | z@g4 | monotone | verdict |
  |---|---|---|---|---|---|---|---|
  | boomy | ac:boominess | 0.30 | 0.57 | -2.26 | -3.78 | False | bar cleared but gates fail |
  | warm | ac:warmth | 0.93 | 0.79 | -3.55 | -2.94 | False | bar cleared but gates fail |
  | hard | ac:hardness | -0.40 | -0.92 | 1.63 | 0.64 | False | bar cleared but gates fail |
  | mellow | clap_rank:mellow | -0.38 | 0.11 | 0.59 | -1.94 | False | bar cleared but gates fail |
  | heavy | clap_rank:heavy | 0.45 | 0.19 | -0.43 | -0.46 | True | no effect >= 1.0 |

  Worst non-target |z| at g2: boomy 12.97 (`summary.md:20`). boomy's set applied to "hard" moved
  hardness +4.93 at g2 (`:22`). At g4 warm was newly broken 0.71 and beat F1 0.37 (`:28`); mellow
  0.86 and 0.41 (`:42`).
- **Recorded reason** (`comms/2026-09-01T1616Z...:10-24`): "Sign inversion: scaling UP the
  'boomy'/'warm' feature sets makes the audio LESS boomy/warm ... ablation (g0) leaves the concept
  intact ... The top-16 contrast features are neither necessary nor sufficient for the word."
  "Non-specific ... The edit is a global timbre push, not a concept dial." "Destructive at g4."
  "Dose non-monotone for all 4 words with effect." Summary: "additive steering was consistent but
  sub-audible; multiplicative editing is audible but blunt. At L22 with this k=128 dictionary,
  neither handle isolates a concept." Memory adds: "Word-vs-pool contrast picks correlates, not
  causes" (`concept-knob-editor-kill.md:6`).

### 2g. Not SAE, but the steering that DID work: activation transplant (2026-08-20)

- Hooks on layers 4/11/17/22, euler 50. A graft replaces host activations with donor
  activations. 168 renders, ~14 min on the local 5090 (`TRANSPLANT_RESULTS.md:3-6`). CLAP
  possession: e.g. piano->techno host -0.321 -> best graft +0.494 vs donor baseline +0.555
  (`:12-16`). "Layer profile: 17 and 22 dominate" (`:34-36`). Ear: "graft early, release late"
  is musical (`:49-59`). Battery 2 (294 renders, 23.5 min): the musicality cliff sits between
  release steps 24 and 29 in every pair, with the knee at R19-R24
  (`BATTERY2_RESULTS.md:17-25`; `comms/2026-08-20T2120Z...:8-15`). Euler-50 generations were
  sha256-identical across the local 5090 and the box (`comms/2026-08-20T2120Z...:17-20`).
  Whole-trajectory activation replacement carries content; small additive directions do not.

## 3. Steering with the learned features: everything tried, and what happened

| # | mechanism | where | outcome |
|---|---|---|---|
| 1 | Additive single decoder row, `h += lam*p90_f*W_dec[f]`, early or late window, lam 1/2/4, k=32 dicts (15 features/layer) | `sae_steer.py`; comms 1813Z | L22 1/15 playable (f9719 centroid), L17 0/15. lam 4 renders near-silent; extra playables broke prompt fidelity |
| 2 | Same, k=128 dicts, effect-size-selected 45 features/layer, 540 renders/layer | comms 1915Z | 0/45 both layers. Own-descriptor moves of 2-3 std occur in single (feature, prompt) pairs only; late window at L17 often inert; ear: nothing recognizable |
| 3 | Additive learned direction (ridge in code space -> `W_dec^T w`), plus pooled top-16 signed sums, 4 prompts, lam +-0.5/1/2 | `sae_direction_fit.py`; comms 2147Z | Context-general (4/4 prompts monotone for rms_db, centroid, hilo) but 0.35-0.62 std, below the 1.0 floor and far below the 3.0 audibility tier |
| 4 | Multiplicative edit-where-active of word-linked feature sets (g 0/0.5/2/4), with random and cross-word nulls | `knob_edit.py`; summary.md | Moves of 1.6-3.8 null-std, but sign-inverted, non-specific (off-target up to 12.97), non-monotone, destructive at g4 |
| - | Never tried (recorded as remaining mechanism classes, `status/concept.json` "next"): causal feature selection by ablation screen; steering with the L17 k=128 dict in the editor; on-manifold edits (re-encode after edit); injection into absence (explicitly out of scope, `CONCEPT_KNOB_PREREG.md:75`); LECO-style per-concept training; per-token time automation and XYZ pad (Phase P2, never reached, `SAE_PLAN_V2.md:133-142`) | | Lane state `STAGE3_KILL_AWAITING_RYAN_NEXT_MECHANISM` since 2026-09-08 |

## 4. What is reusable now

**Code** (all on `feat/synesthesia-lane`, SA3-specific hook path `model.model.model.transformer.layers`):
- `scripts/sae_harvest.py`: resumable sharded residual-stream capture. Uses sampler-grid sigma
  assignment (`--sigma-mode euler50`) and per-row metadata (stem set, token idx, sigma, t_idx).
  The pattern ports to any DiT; the 323-token tail and 64-token memory constants are SA3's.
- `scripts/sae_onpolicy_capture.py`: on-policy capture from real cfg generations, both CFG rows.
- `scripts/sae_train.py`: TopK SAE trainer (streamed shards, resampling, atomic resume). No AuxK.
- `scripts/sae_eval.py`: FVU per sigma bucket plus the on-policy gate. Needs an absolute
  per-bucket check added (the k=192 lesson).
- `scripts/sae_features.py`, `sae_contexts.py`, `sae_descriptors.py` (per-token
  spectral/structural descriptor banks), `sae_regress.py` (use `--select effect --no-r2`),
  `sae_browse.py`.
- `scripts/sae_steer.py` (`Steerer`: additive hook, lam=0 bit-exact check, direction mode),
  `scripts/knob_edit.py` (`SAEEditor`: closed-form edit-where-active, g=1 bit-exact),
  `scripts/sae_direction_fit.py` (GPU sparse ridge with CV, smoke-tested against planted
  directions per memory `subspace-probe-outcome.md:32-33`).
- Screening: `scripts/sae_screen_score.py` (garble + 172.27 Hz comb + musicality + CLAP
  four-gate), `scripts/knob_measure.py` (AudioCommons timbral_models with the two numpy/librosa
  shims, onset/beat F1 in pure numpy), `scripts/knob_screen.py` (CLAP self-rank vocabulary
  screen).
- Box drivers `scripts/box/sae2*_driver.sh`, `knob_*_driver.sh`.
- Tests `tests/test_rf_steering.py` on master are unrelated (E2 inversion-steering stack).

**Data and checkpoints.** Nothing local: the scratchpad dirs `sae2_b2/` and `sae2h_box/` under
`C:\Users\ryanf\AppData\Local\Temp\claude\C---dev-projects-instrument-controlnet\525fcbc4-...\scratchpad\`
are empty, and `E:\Projects\instrument-controlnet\sa3_synesthesia\` holds only
`transplant_smoke` (2.1 GB). The box tree was torn down 2026-09-08. Everything is on B2, bucket
`mucket`, endpoint us-east-005, prefix `sa3/synesthesia/`. Listing as of 2026-09-08
(`comms/2026-09-08T1948Z...:6-12`; memory `concept-lane-b2-archive.md:4-9`). Not re-verified
today: no B2 creds were used.

| prefix | objects / size | content |
|---|---|---|
| `dicts_keep_2026-09-01/` | 4 / 1.81 GB | **k128 d24576 SAE finals, L22 + L17** (latest.pt + config.json). Pull first |
| `ksweep_2026-08-30/` | 58 / 16.3 GB | all k-sweep dictionaries + evals |
| `sae2/` | 19 / 0.9 GB | SAE2 k=32 ckpts L17/L22, metrics.jsonl, eval jsons, manifests |
| `sae2_ckpts_2026-08-30/` | 2 / 0.91 GB | two non-duplicate SAE2 intermediates |
| `sae2h/` | 88 / 63.3 GB | L17+L22 activation shards, 128 tok/window, euler50 sigmas, incl. on-policy + meta |
| `data_same_l_2026-09-01/` | 79,387 / 23.1 GB | SAME-L latent caches used by the harvest |
| `triage_k128_2026-08-30/` | 5,493 / 8.3 GB | k128 triage + all 1,080 screen renders |
| `triage_2026-09-01/` | 11,115 / 8.7 GB | descriptor banks, decoded audio (old-dict triage) |
| `triage_old/` | 730 / 1 MB | old-dict screen outputs |
| `subspace_2026-08-30/` | 1,479 / 1.6 GB | directions, fit report, screen csv, summary |
| `knob_2026-09-01/` | 3,222 / 6.54 GB | concept-knob stage 1-3 renders, features, measures |
| `stitch_2026-08-31/` | 905 / 2.4 GB | stitch renders + screen |
| `listen_sae_2026-08-30/` | 6 / 32 MB | the ear-failed calibration package |
| `box_tail_2026-09-08/` | 31 / 73 MB | venv freeze files, push/pull scripts, `synrepo_sae2h_v3.tar.gz` |

Related caches: `sa3/cache/jamendo_v3_same_l/` (71 objects, 5.57 GB) and
`sa3/cache/moisesdb_same_l/pairs_same_l/` (9 tars, 17.7 GB) (`SAE2H_RUNBOOK.md:92-96`).
Local listening copy: `E:\Projects\instrument-controlnet\listening\2026-08-30_sae-concept_sa3_synesthesia_sae_screen_2026-08-30`.
Committed small results on the branch: `results/knob_screen/`, `results/knob_stage2/features/*.json`
(16 words x top-16 features), `results/knob_stage3/` (selection, null sets, measures.csv,
summary.md, sae_info.json), `results/stitch_screen/`.

Resume recipe (memory `concept-lane-b2-archive.md:13`): pull `dicts_keep_2026-09-01/`, rebuild
venvs from the freeze files in `box_tail_2026-09-08/`, check out the lane branch. Trap: a
`git archive` tarball of the repo ships a pruned vendor tree, so graft a complete
`stable-audio-3` vendor before the trunk loads (`SAE2H_RUNBOOK.md:75-79`; memory
`sae2-launched.md:46-49`).

## 5. Do not repeat (dead ends with evidence)

1. **Harvesting at a sigma the sampler never visits.** SA3 medium-base euler-50 bottoms out at
   sigma 0.1375, so the v1 0.05 harvest was off-policy (`SAE_PLAN_V2.md:18-21`). Harvest on the
   sampler's own grid and gate with an on-policy capture.
2. **Training longer to fix FVU.** Flat by ~12k steps; 50k bought 0.386 -> 0.381. k is the lever
   (`comms/2026-08-30T1226Z...:19-21`; `1716Z...:17-19`).
3. **Trusting a relative on-policy-vs-val gate alone.** k=192 hit train FVU 0.05 while val blew up
   4-33x in the low-sigma buckets, and the relative gate passed (`comms/2026-08-30T1813Z...:40-43`).
4. **R2 / linear-regression labelling of sparse TopK features.** ~0.2% fire rate bounds R2 near
   0; max 0.17; zero features classified (`TRIAGE_SPEC.md:187-194`). Use firing-vs-not effect
   sizes.
5. **Browse-coherence as a usefulness signal.** It passed (15/20, 14/20), but 83-90% of features
   were coherent only by sigma bucket (`comms/2026-08-30T1716Z...:23-27`).
6. **Single-feature additive steering (fixed decoder row), any layer tested.** 0/45 + 0/45
   playable, prompt-inconsistent (`comms/2026-08-30T1915Z...:8-23`). Rule: "never propose
   reopening" (memory `sae2-kill-test-outcome.md:10`).
7. **Learned linear directions in SAE code space, open-loop additive.** Consistent but 0.35-0.62
   std (`comms/2026-08-30T2147Z...:9-27`).
8. **Scaling word-linked SAE features where active (gain or M re-sweeps).** Sign-inverted,
   non-specific, destructive; the kill clause forbids gain/M rescue sweeps
   (`results/knob_stage3/edit_measure/summary.md:3`; `CONCEPT_KNOB_PREREG.md:89-92`).
9. **Selecting features by word-vs-pool contrast.** Picks correlates, not causes. Ablation (g0)
   left concepts intact (`comms/2026-09-01T1616Z...:11-14`).
10. **Quoting descriptor-std effects as audible.** 1.5-2.6 std was ear-verified inaudible; the
    provisional audibility floor is 3.0 std (`comms/2026-08-30T2028Z...:11-16`;
    `SUBSPACE_PROBE_SPEC.md:100-102`).
11. **Late-window steering.** At L17 it was frequently inert. Early steps (0-24) are where
    authority lives, and past release ~24 grafts tear (`comms/2026-08-30T1915Z...:24-25`;
    `BATTERY2_RESULTS.md:17-25`).
12. **Ops traps that cost hours:** a fork `multiprocessing.Pool` after CUDA init deadlocked the
    descriptor stage; librosa/numba segfaults under numpy 2.4 (separate venv); CLAP silently
    "unavailable" without `HF_HOME` set on the box; `pkill -f` over ssh kills its own shell
    (`comms/2026-08-30T1555Z...:15-23`; `1813Z...:44-47`). Never left a box without pushing
    activation shards: the SAE2 shards were lost with box 48209840.

What the record says survives as fact (not a proposal): k=128 dictionaries at L17/L22
reconstruct well (FVU 0.16-0.19, all on-policy gates); the information is linearly present on
the read side (CV R2 .56/.49); context-conditional authority exists. The failure is on the write
side: open-loop residual edits at L22/L17 of SA3 medium-base.
