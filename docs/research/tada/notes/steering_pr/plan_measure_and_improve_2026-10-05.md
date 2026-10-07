# SA3 steering knobs: provenance, measurement plan, improvement candidates (2026-10-05)

Sources (read once, this session): knob_provenance.md, tada_bench_reuse.md, literature_scan.md (same folder).
Worktrees: production = C:\_dev\projects\DEMON-steer (HEAD 83f5b77b); bench = DEMON-tada-sa3 (974ab049, read-only).

## 1. How the five vectors were made (facts, refs in DEMON-steer)

One recipe for all five. Differences: prompt poles, proxy, chosen block, raw norm.
- Site: output of ONE whole transformer block (post-residual), plain add to every token incl. the 64 memory tokens,
  on both CFG passes, every step. layout.py:28-30; stream.py:1910-1912 (eager), 946-963 (TRT fill);
  sa3_steering_onnx.py:117-160; stream.py:1408,1422.
- Layers: one block per pack, no per-layer weights. packs.py:337-344. Block = argmax over 24 of the MEDIAN
  residual-patching effect on 8 pairs, scored by a librosa proxy. discover.py:184-193, 403-428.
- Steps: policy range 0..1 = weight 1 at all 8 steps. packs.py:129-132; discover.py:544.
- Estimator: diff of means (CAA). Token mean over batch+tokens (discover.py:178-183), mean over 8 steps (197-203),
  mean over 32 runs per pole (395-397), unit-normalised, raw L2 kept as `norm` (431-434). No PCA/probe/SAE/whitening.
- Data: 32 same-seed prompt pairs = 8 neutral bases x 4 (discover.py:107-116, 363-369), poles at discover.py:71-103,
  54 s, 8 steps, pingpong, denoise 1.0, eager DiT via StreamPipeline (211-252). No labelled audio, no listening test.
- Scale: shift = knob x (norm x 0.1) x unit vector; knob 10 = one raw mean difference; clamp +-30; no norm matching.
  knobs.py:224,257-290,435-441; packs.py:334; stream.py:1881; discover.py:483-485,543.
- Per knob: bright b23 norm 18.6 (centroid); warm b15 29.5 (low/high dB); percussive b15 40.2 (HPSS ratio);
  rough b2 15.7 (flatness); density b1 12.9 (onset rate, POSITIVE = sparse).
- NOT IN REPO: the pack files (only ~/.daydream-scope/models/demon/steering_packs/sa3/medium), the exact
  discover.py that wrote them (header method_source "TADA, arXiv 2602.11910 (CAA + activation patching)" never
  existed in git; committed writer emits a different string, discover.py:547), the report JSON (calibration,
  per-block norms), per-block vectors for unchosen blocks. docs/STEERING.md:122-124 claims calibration is stored; it is not.
- Red flags in the headers: rough's proxy gap has the WRONG sign (gritty prompts LESS flat: 0.00135 vs 0.00360);
  density's per-block effects swing +1.30 (b1) / -1.60 (b2) on a 0.59 onsets/s gap (noise); bright's best patch
  recovers 18% of the gap. docs/FAMILIES.md:119-120 already says rough/density proxies do not track the concept.
- Why these five: discover.py:69-70 ("the four ACE demo axes ... plus one extra"). No other rationale recorded.
- Bright vs warm: identical recipe; warm's negative pole ("thin, tinny, no bass, harsh brittle highs") overlaps
  bright's positive pole, so in prompt space warm is partly minus bright; but they sit at different blocks (15 vs 23),
  so they are not opposite vectors. Cross-block cosine -0.23 is not like-for-like. Nothing measured yet.
- Were the knobs ever measured as knobs? No. Proxies only scored the prompt-pair gap and picked the block.

## 2. Measurement plan (about 45 GPU-min on ARC medium, eager, 8 steps, 10 s, one seed)

Harness: TADA bench from DEMON-tada-sa3 (scripts/tada/sa3_tada_run.py + sa3_tada_score.py), run from a NEW worktree
cut from tada-sa3 HEAD 974ab049 (branch ryanontheinside/spike/steer-bench, dir C:\_dev\projects\DEMON-steer-bench)
so the tada worktrees stay untouched. Shim (about 60 lines):
  1. `--method pack --pack <safetensors>` in `_load_vectors` (RUN:664-679): load_pack, return
     {s: {pack.block: v * policy_weight[s]}}.
  2. block-output variant of `steer_offline` hooking `sa3_blocks(sam)[b]` (production site) instead of cross_attn.
  3. PCI triples for the five concepts from discover.py:71-103 (neutral "{p}", pos "{p}, <pos>", neg "{p}, <neg>").
  4. anchor dict for MuQ ("This is a music of a bright track" etc.) passed as eval_prompt=.
  5. descriptor column: scripts/steering/proxies.py (centroid in semitones, lowhigh_db, perc_ratio, onset_rate,
     flatness) plus AudioCommons timbral_models warmth/roughness/hardness if the package installs in evalenv.
     Computed on CPU for EVERY clip (all five descriptors per clip). Alignment for the AUC = descriptor delta vs
     alpha 0, divided by the PCI full-swap descriptor gap for that concept (dimensionless, so steer/PCI ratio
     is comparable across knobs). MuQ delta kept as a secondary column.
Protocol per concept (cost figures from tada_bench_reuse.md section 3):
  - probe: 20 holdout prompts x 13 alphas, LPAPS only (260 clips, about 1 min)
  - calibrate: alpha range = smallest |alpha| reaching the PCI-all cutoff (max mean LPAPS of the full swap)
  - sweep: 50 prompts x 7/side + 0 (750 clips, about 1.5 min)
  - PCI-all: --pci-ks 2 4 6 8 both signs (450 clips + record passes, about 2 min)
  - score: LPAPS + MuQ + descriptors (about 1200 clips x 0.2 s = 4 min); no aesthetics, no CLAP
  - about 8 min serial per concept; 5 concepts about 40 min; GPU_LOCK set; demo backend (PID 58000) stopped first
    (bench + scorer page at 32 GB).
Outputs per knob, per sign: AUC(descriptor)/PCI, AUC(MuQ)/PCI, alpha at cutoff mapped to knob value at cutoff
  (alpha / magnitude: tells whether +-30 is sane), knob-to-descriptor curve (monotone? symmetric?), and the
  5x5 cross-effect matrix (each knob's effect on all five descriptors: selectivity; answers bright vs warm).
  Per-prompt spread grouped by genre answers "behaves differently on different material".
Gate before trusting MuQ: on the PCI endpoints check MuQ separates real pos vs neg prompts (as done for
  tempo/piano/mood); if |delta| is at noise, the descriptor column is primary and MuQ is reported only.
Caveats to print with the numbers: one seed, 50 prompts (ratio differences < 0.1 are noise); bright centroid
  is register-confounded (Friberg: spectral features explain at most 31% of perceived brightness, r 0.90 with pitch),
  report HF-energy ratio beside centroid; roughness descriptors track perception at r 0.44 to 0.56 only.
Data to E:\Projects\steering-bench\<date>\; status to notes/steering_pr/status_bench.md.

## 3. Improvements, each gated on the section-2 numbers

P1 Re-estimate directions from labelled activations at the production site. One recapture at post_block_residual
   (sa3_e5_capture.py, one-line hook change, 5521 captions, all 24 blocks, 8 steps: about 9 GPU-min, 3.1 GB) plus a
   self-labelled set: about 800 neutral-prompt renders with activations captured in the same pass, descriptors
   measured on the audio, direction = mass-mean of top vs bottom quartile per block per step (about 3 GPU-min).
   MusicCaps classes: bright 105/51, warm 59/82, percussive 366, rough-group 394/565, density only via broad proxy
   238/726. Rescore with section 2 (about 40 min). Cost about 55 GPU-min. Falsified if no knob gains > 0.1 ratio
   (E4 precedent: +0.25 for tempo and piano).
P2 Block choice by AUC, not 8-pair patching median. With P1's per-block vectors: 24 blocks x 20 holdout x 5 alphas,
   descriptor + LPAPS only (about 2400 clips, 6 min per knob, 30 min for five), or skip selection and use all-blocks
   per-step vectors (E5: loc gave -13% vs all on SA3). Falsified if the best single block is within 0.1 of all-blocks.
P3 Per-step vectors and audio-token-only means (drop the 64 memory tokens from the mean). Pack format v2 with
   `vector [steps, 1536]`; TRT input is already [1,24,1536] per row filled per step, so host-side only, no rebuild.
   Zero extra GPU beyond P1. Falsified if the per-step ratio is within noise of the single-vector ratio.
P4 Calibrated knob axis: std-normalised alpha, monotone map knob to descriptor per sign, range ending at the LPAPS
   cutoff. Fit from the section-2 sweep, zero GPU. Falsified if the curves are already linear and symmetric within 10%.
P5 Closed-loop strength (SMITIN): forward-only linear readout of the descriptor from the hooked hidden state,
   alpha raised toward a target and decayed once reached. Readout trained on the P1 capture (CPU);
   live smoke on the demo backend about 10 min; bench about 40 min. Falsified if readout R^2 < 0.5 at the site,
   or AUC gain < 0.1 at matched LPAPS.
P6 Step function: trained conditioner (Sketch2Sound recipe) for bright and density first: per-frame centroid
   (semitones) and onset-density curves, one linear projection added to the noisy latent, or an AdaLN vector
   (MuseTimbre proved AdaLN injection on SA3 medium). Frozen DiT, about 40k steps, control dropout, median-filtered
   controls. Needs audio data (self-generated SA3 clips with measured descriptors, or a corpus on E:). Estimate
   6 to 12 GPU-h training plus a new TRT input. The knob becomes a target in physical units and time-varying.
   Falsified if held-out centroid error is not below what P1 steering achieves at the cutoff, or LPAPS/FAD at
   matched effect is worse than steering.
P7 Concept set: from the section-2 renders (zero GPU) compute descriptor correlations among the five; consider
   replacing rough (no validated scorer on mixes) and density (positive = sparse, noisy proxy) with concepts that
   have validated scorers: hardness/attack (AudioCommons r 0.87), speed in onsets/s (r about 0.7), loudness/dynamics,
   reverb/depth.
P0 (hygiene, no GPU): commit the pack writer state, store calibration as the docs claim, exclude memory tokens.

Ruled out for streaming: FreeSliders (+2 NFE per step), DITTO (seconds per clip), gradient guidance through the DiT.
