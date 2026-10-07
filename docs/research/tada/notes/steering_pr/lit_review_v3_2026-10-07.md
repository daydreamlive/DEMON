# Literature review for v3 steering (item L of v3_smoke_batch_runbook.md), 2026-10-07

Scope: question (1) is "regression-fit descriptor directions, per-sign, calibrated to a perceptual budget, with minimal-pair
prompts from a recipe grammar and stacking rules, on a music diffusion model" a contribution on its own, and what is the
closest prior work per ingredient; question (2) which smokes the literature suggests that we are not running.
Companion: literature_scan.md (2026-10-05) already covers TADA in depth, ITI, SMITIN, CAA, ReFT, Concept Sliders, h-space,
Facchiano et al. (MusicGen bright/dark, tempo), MusicRFM, SDXL-Turbo SAEs, FreeSliders, descriptor validity (Friberg,
AudioCommons). Not repeated here except where a fact bears on v3.

Reading depth: "full" = read the paper text; "html summary" = read through a fetch summary of the arXiv HTML, numbers
quoted as returned; "abstract" = abstract only. Our own facts come from status_v3_smoke.md, results_ship.md,
followups_many_knobs_v2.md, tada_sao_vs_ace_numbers.md.

---

## 1. Per ingredient: closest prior work, what differs, verdict

### 1a. Regression-fit directions to a continuous descriptor (ridge from block means to centroid / onset_rate / lra)

Closest prior work:
- Graziani, Andrearczyk, Müller. Regression Concept Vectors for Bidirectional Explanations in Histopathology. iMIMIC 2018,
  arXiv 1904.04520. https://arxiv.org/abs/1904.04520 (abstract). The RCV is the least-squares regression direction from layer
  activations to a continuous concept measure. This is exactly our fit (ours adds a ridge penalty). Used for explanation
  (directional derivatives), never for generation.
- Koh. Probing-Based Test-Time Steering of Music Diffusion Transformers. ICML 2026 Workshop on ML for Audio.
  https://mlforaudioworkshop.github.io/accepted_submissions_2026/CameraReadys%204-83/47/CameraReady/Probing_Based_Test_Time_Steering_of_Music_Diffusion_Transformers.pdf
  (full). Stable Audio Open DiT (24 blocks, d 1536, the same width as SA3 medium). Class-balanced logistic probes on
  time-pooled block activations at t = 0.5, 150 clips, 11 binary concepts; the l2-normalised probe coefficients are
  injected at one block at every step. Results: monotone probe-score response for 5/5 concepts at block 12; specificity
  on/off-diagonal 13.6 / 3.8 (ratio 3.6); block 6 slope 24.0 vs 18.2 (block 12) and 17.5 (block 18). Independent CLAP
  check: negative steering at -30 drives concept-text similarity to about zero for all 5 concepts; positive steering
  helps at +10 (electronic +0.03, bass +0.06) and turns negative at +30 while the probe score keeps rising. A matched SAE
  baseline was "substantially weaker". The same pipeline on ACE-Step v1.5 turbo kept decodability but lost causal
  specificity.
- Spanio, Rodà. A Quantized Native Runtime for On-Device Semantic Audio Generation (aria). arXiv 2607.08526, July 2026.
  https://arxiv.org/abs/2607.08526 (html summary). Steers SA3 small and medium (the model we steer). DiffMean directions
  from (i) caption pairs differing only in a taste word and (ii) top/bottom-k of 377 clips ranked by a continuous human
  taste rating (a ranked-continuous-label contrast, the nearest thing to our tercile fits). Block chosen by a logistic
  probe, then a per-block sweep at alpha 0.15; alpha in units of the block's mean residual norm. Four oracles (a learned
  wav2taste regressor, CLAP, FAD, CLAP drift). Sweet +0.119 at alpha 0.3 (block 16), sour +0.419 and bitter +0.314 at
  alpha 0.1 (block 19); salty and spicy weak. Past a narrow window the target regressor keeps rising while CLAP collapses,
  which they call metric gaming.
- Marks, Tegmark. The Geometry of Truth. arXiv 2310.06824 (abstract): difference-in-mean probes "identify directions which
  are more causally implicated in model outputs" than other probes. Li et al. ITI (in literature_scan.md B1): mass-mean
  direction 42.3% vs probe-weight direction 34.8% True*Info. Im, Li. A Unified Understanding and Evaluation of Steering
  Methods. arXiv 2502.02716 (abstract): mean of differences recovers the optimal steering vector in their framework where
  PCA- and classifier-based directions do not.
- Chen, Wei, Yin. Measured Sliders: Learning Continuous Controls from Differentiable Image Measurements. arXiv 2609.05234,
  Sep 2026. https://arxiv.org/abs/2609.05234 (html summary). Image diffusion. Slider axes defined by differentiable image
  measurements; an "observability test" before training keeps a measurement only when SNR = robust decoded range (p95-p5)
  / preview RMSE > 1; LoRA branches trained with measurement-guided loss; lighting rho 0.995, 98.9% monotone sweeps;
  selectivity 2.59 vs 1.50 baseline. This is the closest image-side analogue of "descriptor-defined knobs", but it trains
  per-axis LoRAs instead of reading a direction off activations.

What differs in ours: a ridge (not logistic, not class means) direction to a continuous audio descriptor, fit on 14,800
corpus clips' block means, used directly as an additive steering vector in a music DiT, scored by the same descriptor at a
fixed LPAPS budget against a tag-knob control. Nobody above does exactly this. But the parts are each published (RCV for
the fit, Koh and aria for probe/ranked-label directions on SAO/SA3, Facchiano for descriptor scoring of bright/dark).

What the literature predicts about our result: the LLM evidence (Marks and Tegmark, ITI, Im and Li) says prediction-optimal
directions steer worse than mean-difference directions. A ridge weight is (Sigma + lambda I)^-1 X^T y; at the CV-chosen
lambda it re-weights toward low-variance dimensions, which is what status_v3_smoke.md measured (ridge projection std 0.4 to
1.3 vs 4 to 13 for the class-mean packs; magnitudes 5 to 15x smaller). The first smoke (2/12 pass, only centroid up, onset
0.883, lra 0.900 just under) is the expected outcome, not a surprise. The continuous-label analogue of the mass-mean
direction is the covariance vector X^T y (ridge with lambda to infinity); we have not tested it (smoke Q1 below).

Verdict: not novel as a method. Novelty would rest on an empirical finding (e.g. covariance or ridge directions beat
tag-vs-rest knobs on descriptor-scored effect at matched LPAPS across many knobs), which our data does not yet show.

### 1b. Per-sign vectors

Closest prior work:
- TADA (Staniszewski et al., arXiv 2602.11910, html read for this question): "positive (0, alpha_max] and negative
  [-alpha_max, 0) directions evaluated independently". It evaluates per sign; it does not fit per sign.
- Facchiano et al. 2025 (literature_scan.md B7): dark moved centroid +40.1% vs bright +20%, i.e. sign asymmetry on the very
  descriptor we steer.
- Koh 2026 (above): negative steering consistently suppresses, positive steering saturates and degrades early. Same
  asymmetry on SAO, opposite in direction to ours (our down side fails).
- Verdini et al. Controlling Speaking Rate in Autoregressive TTS via Activation Steering. arXiv 2609.33810, Sep 2026.
  https://arxiv.org/html/2609.33810v1 (html summary). Three classes (slow / neutral / fast, the same utterance
  time-stretched), class-mean axis; projection clamped to k * Delta instead of additive. At k = -2 additive gave 86.1% WER
  and 92% non-termination on Qwen3-TTS, clamping 8.6% WER and 0%. Side asymmetry: CosyVoice2 clamp fast 17.6% WER vs slow 1.7%.
- Torop, Masoomi, Dy. Inverted Detection and Control in Steering Vectors. arXiv 2608.02957, Aug 2026 (html summary).
  Highly discriminative DiffMean vectors (one with AUC 0.97) can steer in the opposite direction ("inverted steering
  vectors"); a forward-pass-only "representation response" detects them with 0.91 AUC; targeted sign flips improved 27/30
  experiments.
- Our own v2 already shipped variant (c): pos = top quartile vs median, neg = bottom quartile vs median, two stored vectors
  (many_knobs_v2_runbook.md, Variants). S1's per-sign ridge is the regression version of that.

What differs: fitting a separate regression for each tail. Minor.

Verdict: not novel. Per-sign evaluation is standard in TADA; per-sign fitting exists in our own v2 (variant c). The
literature adds two specific things we do not do: clamping the projection instead of adding (Verdini), and checking for
inverted directions per sign (Torop).

### 1c. Calibration to a perceptual budget (PCI LPAPS cutoff; S4 linear-in-LPAPS knob map)

Closest prior work:
- TADA: alpha_max per method set so the max LPAPS equals PCI's LPAPS at max strength. This is our cutoff definition
  (tada_sao_vs_ace_numbers.md). Not ours.
- FreeSliders (arXiv 2511.00103, html summary): saturation detection by the ratio of alignment to
  perceptual deviation over eta in {0, 0.5, 1, 2, 4, 8, 16}, then a per-concept low-degree polynomial reparameterisation
  so equal steps give uniform change; tested on Stable Audio Open with LPAPS and CLAP; audio overall score 1.83 -> 2.58.
- Serrano-Lozano et al. UniSlider: Perceptually Uniform Sliders for Continuous Image Editing. arXiv 2610.06831, 5 Oct 2026
  (html summary). Defines uniformity as D(x_src, x_s) = s * D(x_src, x_edit) with DreamSim; this is S4's target with LPAPS
  in place of DreamSim. They also apply an inference-time strength remap to training-free baselines: it improves
  uniformity but cannot repair non-monotone trajectories. CV of consecutive distances 0.652 vs 1.199 (SliderEdit).
- Measured Sliders: calibrated endpoint alpha_k* = inverse of the running-max mean response at tau * natural range,
  tau shared across attributes. Positive ladder only.
- aria: alpha in units of the block's mean residual norm (a scale normalisation, not a perceptual one).

What differs: we calibrate per knob and per sign on LPAPS against a PCI reference, interpolated from probe curves, on 97
knobs. The method is FreeSliders' or UniSlider's remap with a different distance.

Verdict: not novel. S4 is a direct application of the UniSlider definition with a training-free remap, which UniSlider
itself shows is limited by monotonicity. S4 should therefore also report monotonicity violations of the raw LPAPS-vs-alpha
curves per knob-sign, since a remap cannot fix those.

### 1d. Minimal pairs from a recipe grammar (S2)

Closest prior work:
- CAA (Panickssery et al., 2312.06681): A/B answers to the same question, hundreds of pairs; the canonical minimal-pair
  contrast in LLMs.
- TADA (html, this session): MusicCaps captions containing the concept; GPT-4 rewrites "by replacing concept-associated
  terms with their counterparts", up to 256 pairs per concept; tempo is binary (slow vs fast).
- aria: caption pairs differing only in a taste descriptor, on SA3.
- Verdini 2026: matched content by construction (the same recording at three tempi), "isolates rate from pitch, timbre,
  lexical content, and vocal affect by construction". The cleanest minimal-pair design in audio I found.
- Braun et al., Understanding (Un)Reliability of Steering Vectors in Language Models, ICLR 2025 workshop; thesis arXiv
  2602.17881 (abstract): higher cosine similarity between per-sample training activation differences, and better
  separation along the direction, predict more reliable steering.

What differs: programmatic grammar (one field changed), shared seed per pair at capture time, per-block selection by
held-out pair AUC. Shared seed is the only design element I did not find stated in the music papers (TADA's text does not
say pairs share noise).

Verdict: not novel. TADA already steers a music diffusion model with one-attribute-changed prompt pairs. S2 is a cleaner
instance, worth running for our knobs, not a contribution.

### 1e. Stacking rules (1/n, shipped-only, shared-norm budget, sqrt(k))

Closest prior work:
- TADA: multi-concept steering (two instruments, vocal gender plus tempo); localized methods beat global ones in the
  multi-concept setting. No scaling rule.
- Measured Sliders: compose by plain superposition of individually calibrated endpoints, alpha(r) = (r_1 alpha_1*, ...);
  all requested directions preserved in 96.7% of 2-attribute and 86.1% of 3-attribute compositions. No norm budget.
- Concept Sliders (2311.12092): up to 50 sliders composed, qualitative. CompSlider (arXiv 2509.01028, abstract): trains
  for disentangled multi-attribute control because independently trained sliders interfere.
- LLM side: interference between non-orthogonal vectors is reported widely; I found no paper that defines a shared
  perturbation-norm budget across stacked activation vectors and measures the fraction of combinations under a perceptual
  cutoff.

What differs: our stacking question is posed against a distortion budget (fraction of 300 combos under the strictest
member's LPAPS cutoff) with candidate scaling rules. results_ship.md already ran a norm_budget rule (Frobenius norm over all
steps and blocks held to the largest member's): 12.3% under cutoff vs inv_n 81.3%; S3's "shared-norm" is a different
definition (sum of shift norms equal to the single-knob budget).

Verdict: the most original of the five ingredients, because the literature has no budgeted stacking rule. It is still an
engineering result, and the measured numbers so far (no rule reaches 95%) are not a positive finding. Measured Sliders'
metric (all requested directions preserved per combination) is a better-defined success criterion than "own-label effect
retained" and costs nothing extra to compute.

### 1f. SAE steering in audio

- TADA: SAE(loc) is the best ACE-Step method (MuQ 0.118, ratio 1.40 to PCI); SAE only localised; no SAO SAE row.
- Singh, Cherep, Maes. Discovering and Steering Interpretable Concepts in Large Generative Music Models. ICLR 2026,
  arXiv 2505.18186 (search summary). k-sparse SAEs on MusicGen residual streams over about 160k clips; 15 to 35% of tested
  features (by configuration) improved CLAP alignment under steering.
- Koh 2026: on SAO a matched SAE baseline was "substantially weaker" than probe directions.
- Ours: two SA3 SAE attempts failed the reconstruction gate (memory: project_sae_on_sa3_attempts).
- Hiramatsu et al. Disentangling Steering Vectors. arXiv 2609.07037, Sep 2026 (html summary): nearest-neighbour pair the
  positive and negative activations, train an SAE on the differences, keep decoder atoms that are semantically consistent.
  This is a much smaller SAE problem (differences of one concept) than a full-stream dictionary.

Verdict: no reason to restart full SAEs for v3. Difference-SAEs (Hiramatsu) on S2's minimal-pair differences is the one
SAE variant that fits our data; it is a research item, not a smoke.

### 1g. Music and audio steering 2025-2026 not covered on 2026-10-05

- Koh 2026 (SAO, probe CAVs) and aria 2026 (SA3, DiffMean, multi-oracle): both above. aria is the only other published
  steering of SA3 that I found; it reports single knobs only, alpha 0.1 to 0.3 residual-norm units, and metric gaming.
- Ye, Zheng, Zang. Pitch-class Steering for Diffusion-based Music Generation via Latent-space Probes. IEEE MLSP 2026,
  arXiv 2609.04516 (abstract): a 125k-parameter conv probe on SAO's VAE latent used as a guidance loss; melodic coherence
  2.4x over unguided. Gradient guidance, not activation addition; it costs backprop per step, out of our latency budget.
- Chang et al. AnchorSteer. arXiv 2605.31053 (third-party review summary only): label-free concept vectors by masked
  reconstruction with L1 and orthogonality, for structure-preserving editing. No numbers available to me.
- Torop 2026, Braun 2025, Billa (Predicting Where Steering Vectors Succeed, arXiv 2604.15557, abstract: a logit-lens
  accessibility score predicts steering success at rho 0.86 to 0.91) are LLM-only.
- TADA's citing papers found: aria (2607.08526) and Verdini (2609.33810). No citing paper reports a method that beats
  TADA's benchmark on a music model.

---

## 2. Verdict on question (1)

No. Every ingredient has direct prior work: regression concept vectors (Graziani 2018), probe-coefficient steering of a
music DiT of our width (Koh 2026, SAO), ranked-continuous-label DiffMean steering of SA3 itself (aria 2026), per-sign
evaluation and PCI-LPAPS budget calibration (TADA 2026), perceptual-uniform remapping (FreeSliders 2025 on SAO, UniSlider
2026), measurement-defined calibrated and composable sliders (Measured Sliders 2026, images), and one-attribute prompt
pairs on a music diffusion model (TADA). The combination on SA3 with 97 knobs is an engineering and benchmark contribution,
not a method. The only element without a close published precedent is a budgeted stacking rule measured as the fraction of
combinations under a perceptual cutoff, and our numbers there are not yet positive. A workshop note becomes defensible
only with a positive empirical result the literature does not already predict. Our first smoke (regression directions
mostly lose to class-mean knobs) is what the literature predicts.

---

## 3. Suggested smokes we are not running (cost on one 5090)

Cost basis: the first v3 smoke rendered and scored 2100 clips of 10 s in 1256 s, about 100 clips per minute with LPAPS,
CLAP and descriptor scoring overlapped. CPU items use existing corpus or ship-pass data.

Q1. Ridge lambda path to the covariance direction. Motivation: Marks and Tegmark, ITI, Im and Li (prediction-optimal
directions steer worse than mean-difference); our ridge picked low-variance directions. Fit per descriptor at lambda =
CV, 10x CV, 100x CV and infinity (X^T y on standardised y, the continuous mass-mean), report cos to the tag control and to
the CV ridge, render as the first smoke. 3 descriptors x 3 new lambdas x 2 signs x 5 alphas x 12 prompts = 1080 clips,
about 11 min GPU plus 5 min CPU. Fold into S1 if S1 is not yet launched; it shares the drive.

Q2. Independent-scorer check for metric gaming. Motivation: aria (target regressor rises while CLAP collapses), Koh (probe
score overstates perceptual gain). Our pass criterion scores the descriptor we fit to. On the existing v3_smoke renders
and S1 renders, add an independent check per descriptor: centroid against CLAP text anchors "bright" / "dark" and a
noise-floor check (spectral flatness of the top band, so hiss cannot pass as brightness); onset_rate against a beat
tracker BPM or the MuQ "fast tempo" anchor; lra against short-term loudness range from a second implementation. Rescoring
only: about 5 to 10 min GPU for CLAP/MuQ, no new renders.

Q3. Clamp versus add on the down side. Motivation: Verdini 2026 (clamping the projection to k * Delta held quality where
additive steering broke; side asymmetry). Implement a set-projection hook (h + (target - <h, v>) v at the block) next to
the additive hook; for onset_rate, centroid, lra on the down side, 5 targets spanning the corpus 10th to 50th percentile of
the projection, 12 prompts, seed 0. 3 x 5 x 12 = 180 renders x the clip batch; with matching additive rows about 360
clips, about 4 min GPU plus about an hour of code. Directly tests whether the down-side failure is a magnitude/overshoot
problem.

Q4. Inverted-direction census from existing data. Motivation: Torop 2026. The ship pass already records cross-effect own z
per sign (results_ship.md); count knob-signs whose own z has the wrong sign (example: dense_arrangement own z +0.10 / +0.13,
so its negative side moves the label up). CPU only, minutes. If the count is non-trivial, add Torop's representation
response (steer at block b, read the projection on v at blocks > b, slope vs alpha) as a forward-pass-only gate: one
forward pass per knob-sign-alpha with no decode, about 5 min GPU for 97 knobs.

Q5. Pair-difference consistency as a predictor (CPU, on S2's capture). Motivation: Braun 2025. For each S2 concept and
block, mean pairwise cosine of the 150 per-pair differences and the separation along the mean difference; check whether it
picks the same block as held-out pair AUC and whether it ranks the three concepts as the render does. CPU only, under 10
min. If it predicts, it becomes a pre-render filter for any future minimal-pair knob (cheaper than screening renders).

Q6. Block depth for descriptor directions. Motivation: Koh 2026 (on SAO, block 6 steered 30% harder than 12 or 18; probe
AUC peaks near block 5); TADA's SAO bottleneck {12, 13, 14}; aria's chosen blocks 16 and 19 on SA3. Our descriptor blocks
(16, 22, 11) were inherited from tag-knob variant (a). Corpus has all 24 block means, so fitting is CPU. Render centroid
and onset_rate covariance (Q1) directions at blocks {4, 6, 8, 11, 16, 22}, 3 alphas, both signs, 12 prompts, seed 0: 6 x
2 x 3 x 2 x 12 = 864 clips, about 9 min.

Q7. Composition success as "all requested directions preserved" in S3. Motivation: Measured Sliders (96.7% for pairs,
86.1% for triples). Score each S3 combo on whether every member's own label moves in its requested sign, alongside the
LPAPS fraction. Zero extra GPU if S3 already scores own labels per member; otherwise a CLAP/MuQ rescoring pass of S3's
renders, about 5 min.

Q8. Monotonicity violations before the S4 remap. Motivation: UniSlider (a remap improves uniformity, cannot fix a
non-monotone trajectory). From the ship probe data, count knob-signs whose LPAPS or own-label curve is non-monotone in
alpha, and exclude or flag them in the S4 map. CPU only, minutes.

Q9. Conditional-branch-only steering, only if the SA3 runtime steers both CFG branches. Motivation: TADA Table 12
(cond-only 0.109 vs cond+null 0.100 holdout AUC on ACE-Step). 6 shipped knobs x 2 signs x 3 alphas x 12 prompts x 2
modes = 864 clips, about 9 min. Skip if the runtime already steers only the conditional branch or SA3 is run without CFG.

---

## 4. Addendum: steering in a continuously generating, real-time streaming model (search 2026-10-07, about 15 min)

Context: knobs change mid-stream in the ring on SA3 and ACE-Step; hold drift measured over 30 s; knob-to-ear latency;
stacking live.

Closest prior work found:
- Live Music Models (Magenta RealTime / Lyria RealTime), arXiv 2508.04651, 2025. Streaming chunked generation with mid-stream
  control by mixing style embeddings (text and audio, MusicCoCa). Control lives in the conditioning space, not internal
  activations. Not a steering-vector paper.
- Real-Time Interactive Music Generation via Data-Free Streaming Consistency Distillation, arXiv 2606.24307, June 2026
  (html summary). ACE-Step 1.5 XL-Turbo plus LoRA, streaming chunk by chunk. Mid-stream controls are prompt interpolation,
  semantic prompt modifiers (energy, density, brightness, tension) and temperature, again in conditioning space; "interventions
  bend the continuation from the current trajectory rather than resetting it". Control latency is a scheduling estimate
  equal to chunk length plus about 0.04 s (0.543 s at 0.5 s chunks, 2.043 s at 2 s chunks). The summary reports no measured
  control-to-ear latency and no drift of a held control. This is the nearest system (same model family, same use), but it
  does not do activation steering.
- Live Music Diffusion Models, arXiv 2605.22717, 2026 (abstract): KV caching over steps and time for streaming diffusion
  music; the abstract does not mention activation steering or mid-stream control latency.
- MusicRFM (arXiv 2510.19127, ICLR 2026): time-varying injection schedules within one offline MusicGen generation; it names
  real-time steering as a possibility only. Genre Controlled Music Generation via Activation Steering (arXiv 2506.10225,
  MusicGen, probe weights) is offline.
- aria (arXiv 2607.08526): on-device SA3 runtime with built-in steering, but fixed-length offline clips, one knob at a time.
- DEMON (Fosdick, arXiv 2605.28657): the abstract covers live denoise controls (per-slot schedules, shared mutable state,
  source blending), not steering vectors.
- Image/video streaming (StreamDiffusion 2312.12491 and successors): no activation-steering knob with a measured
  change-to-output latency or hold drift found.

Status: in about 15 minutes of search I found no published work that applies activation steering vectors to a streaming
or real-time audio or music model with knobs changed mid-stream. I also found none that measures, for a steering vector,
knob-to-ear latency, drift of a held knob over tens of seconds, or live stacking. All the real-time music systems found
steer through conditioning (prompt or style embeddings). This is absence after a short search, not proof.

Effect on the verdict for question (1): the verdict on the recipe stands. Regression-fit, per-sign, budget-calibrated,
minimal-pair, stacked descriptor directions are still not a method contribution, and running them in a stream does not
make the recipe new. The streaming context does change what the defensible contribution is. Live activation steering of a
streaming music diffusion model, measured for knob-to-ear latency, hold drift and live stacking under a perceptual budget,
appears unpublished. That is a systems and measurement contribution, a natural extension of the DEMON paper, where the v3
recipe would be one component, not the claim.

---

## Sources

- Live Music Models, arXiv 2508.04651: https://arxiv.org/abs/2508.04651
- Streaming consistency distillation, arXiv 2606.24307: https://arxiv.org/abs/2606.24307
- Live Music Diffusion Models, arXiv 2605.22717: https://arxiv.org/abs/2605.22717
- Genre control via activation steering, arXiv 2506.10225: https://arxiv.org/abs/2506.10225
- DEMON, arXiv 2605.28657: https://arxiv.org/abs/2605.28657
- StreamDiffusion, arXiv 2312.12491: https://arxiv.org/abs/2312.12491

- TADA, arXiv 2602.11910: https://arxiv.org/abs/2602.11910
- Koh 2026, ML4Audio @ ICML 2026: https://mlforaudioworkshop.github.io/accepted_submissions_2026/CameraReadys%204-83/47/CameraReady/Probing_Based_Test_Time_Steering_of_Music_Diffusion_Transformers.pdf
- Spanio, Rodà 2026 (aria), arXiv 2607.08526: https://arxiv.org/abs/2607.08526
- Verdini et al. 2026, arXiv 2609.33810: https://arxiv.org/abs/2609.33810
- Torop, Masoomi, Dy 2026, arXiv 2608.02957: https://arxiv.org/abs/2608.02957
- Chen, Wei, Yin 2026 (Measured Sliders), arXiv 2609.05234: https://arxiv.org/abs/2609.05234
- Serrano-Lozano et al. 2026 (UniSlider), arXiv 2610.06831: https://arxiv.org/abs/2610.06831
- FreeSliders, arXiv 2511.00103: https://arxiv.org/abs/2511.00103
- CompSlider, arXiv 2509.01028: https://arxiv.org/abs/2509.01028
- Concept Sliders, arXiv 2311.12092: https://arxiv.org/abs/2311.12092
- Graziani et al. 2018 (RCV), arXiv 1904.04520: https://arxiv.org/abs/1904.04520
- Marks, Tegmark 2023, arXiv 2310.06824: https://arxiv.org/abs/2310.06824
- Im, Li 2025, arXiv 2502.02716: https://arxiv.org/abs/2502.02716
- Braun 2026 thesis, arXiv 2602.17881: https://arxiv.org/abs/2602.17881
- Billa 2026, arXiv 2604.15557: https://arxiv.org/abs/2604.15557
- Hiramatsu et al. 2026, arXiv 2609.07037: https://arxiv.org/abs/2609.07037
- Singh, Cherep, Maes, ICLR 2026, arXiv 2505.18186: https://arxiv.org/abs/2505.18186
- Ye, Zheng, Zang 2026, arXiv 2609.04516: https://arxiv.org/abs/2609.04516
- AnchorSteer, arXiv 2605.31053: https://arxiv.org/abs/2605.31053
- CAA, arXiv 2312.06681: https://arxiv.org/abs/2312.06681
