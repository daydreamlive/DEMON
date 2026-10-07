# Literature scan: improving the five SA3 steering knobs (bright, density, percussive, rough, warm)

Date: 2026-10-05. Web only, no repo reads. Every entry below was read from the primary source (arXiv HTML or the PDF itself) unless marked "abstract-level". Numbers are the papers' own measured values, not abstract claims. Where the source only reported a claim with no number, that is stated.

Reference protocol: TADA (Staniszewski et al. 2026, arXiv 2602.11910). That is the paper our AUC-vs-LPAPS benchmark follows. It is summarised first because every later "implication" is relative to it.

---

## 0. The benchmark paper we follow

**TADA! Tuning Audio Diffusion Models through Activation Steering.** Łukasz Staniszewski, Katarzyna Zaleska, Mateusz Modrzejewski, Kamil Deja. arXiv 2602.11910 (v2), Feb 2026. https://arxiv.org/abs/2602.11910 , code https://github.com/luk-st/steer-audio

What it did: activation patching on AudioLDM2, Stable Audio Open (SAO) and ACE-Step to find a "semantic bottleneck" (ACE-Step cross-attention layers {7,8}; SAO {12,13}; AudioLDM2 layers 45 to 51). Then compared steering methods on ACE-Step only: CAA (mean difference of cross-attention outputs, time-averaged, averaged over N prompt pairs, roughly CASteer), AUSteer (sparse signed dims), TopK SAE features (TF-IDF scored, top-20 per layer, decoder columns summed), against prompt interpolation (PCI), text/token embedding steering, FreeSliders (score space) and Concept Sliders (LoRA rank 8, 500 iters, eta 7). Alpha swept over 31 values; alpha_max calibrated per method so the max LPAPS distortion equals PCI's max distortion. Alignment = CLAP or MuQ text similarity delta; preservation p(alpha) = LPAPS(alpha_max) - LPAPS(alpha); score = AUC. Smoothness = std of consecutive normalised alignment gaps. ACE-Step, 30 Euler steps, CFG 5, 30 s, 100 test prompts.

Measured (Table 1, ACE-Step, MuQ / CLAP AUC): localized SAE 0.106 / 0.059; localized CAA 0.098 / 0.058; localized AUSteer 0.090 / 0.058; global CAA 0.067 / 0.039; global AUSteer 0.073 / 0.055; Concept Sliders 0.081 / 0.058; FreeSliders 0.073 / 0.047; PCI 0.074 / 0.042. Localization gave CAA +46% / +49% AUC; it hurt Concept Sliders (-21% AUC, -75% smoothness). Appendix details I read in the PDF: steering only the conditional branch of CFG beats steering cond+null (holdout AUC avg 0.109 vs 0.100, Table 12). Delaying the start step (0, 3, 5, 10 of 30) moves AUC by at most about 0.01 for activation methods (localized CAA avg 0.121 to 0.124; Table 15). SAE top-k per layer 20 to 50 is the sweet spot (avg 0.120 to 0.125). Concept Sliders hyperparameters (rank 4 to 16, iters 500 to 1500) barely matter (0.101 to 0.105). Human study: localized CAA highest "seamless edit" (3.32/5). Concepts were vocal gender, tempo, mood, instruments, genres. No timbre-descriptor concepts. SAO steering numbers are not in the main tables.

Implication for our knobs: our protocol inherits TADA's weakest link, a text-similarity alignment axis (MuQ/CLAP), which was never validated against perceived brightness/warmth/roughness; the cheap wins TADA itself measured are localization (+46%) and cond-only steering (+9%), and start-step tuning is worth little.

---

## A. Perceptual descriptors with validated computational models (the "ground truth" scorer question)

### A1. AudioCommons timbral models (Pearce, Safavi, Brookes, Mason, Wang, Plumbley)
**Deliverable D5.7: Evaluation report on the second prototypes of the timbral characterisation tools.** AudioCommons, 30 Nov 2018. https://audiocommons.github.io/assets/files/AC-WP5-Surrey-D5.7%20Evaluation%20report%20on%20the%20second%20prototypes%20of%20the%20timbral%20characterisation%20tools.pdf . Package: https://github.com/AudioCommons/timbral_models (v0.4.1, May 2024, "no longer actively maintained").

What it did: listener ratings (13 to 17 Surrey Tonmeister undergraduates per attribute) on about 165 to 212 freesound.org sound-effect clips per attribute (selected from freesound search history, loudness-normalised to -24 LUFS), then fitted or re-fitted a model per attribute. Validation also on the earlier D5.3 within-source and between-source sets (no audio overlap with training).

Feature definitions (from the released code, which I read):
- Brightness: whole-file, power-weighted log10 ratio of energy above 2000 Hz to total, plus power-weighted spectral centroid of the signal high-passed at 100 Hz (cascaded high-pass filters), combined by a 3-coefficient linear regression.
- Warmth: onset-segmented; log10 spectral centroid, RMS-weighted high-frequency content (bark-weighted region), mean energy in a low-mid "warmth region", an R^2 "high-frequency decay" score, and an RMS-weighted low-frequency ratio (energy between f0 and 260 Hz over total, for onsets with f0 < 260 Hz); 6-coefficient linear regression.
- Roughness: a tuned implementation of Vassilakis (spectral peak picking, sum over peak pairs, log transform).
- Sharpness: direct Fastl/Zwicker-style (Klippel) sharpness, no fitting.
- Hardness: multilinear regression over iteratively selected onset/attack features.

Measured: brightness r = 0.86 (rho 0.83) on training data, r = 0.81 between-source held-out (old model 0.69), within-source r 0.96 to 0.99 except voice 0.57. Warmth r = 0.79 (rho 0.79), training-set fit only, no held-out test. Roughness r = 0.70 training; between-source held-out r = 0.56, worse than the original Vassilakis implementation (0.63). Sharpness r = 0.78 (validation only, not fitted). Hardness r = 0.87 training; within-source held-out r 0.75 to 1.0. Boominess r = 0.67. Reverb is a 2-class classifier, 75.25%.

Polyphonic music: not tested. All stimuli are single-source sound effects (bell, piano, swoosh, guitar, engine, etc.).

Implication: these are the best-validated descriptor models available for brightness, warmth, hardness (percussive) and roughness, but their validation is on single sound effects, so use them as a measured-property scorer, not as proof of perceived brightness in a full mix.

### A2. Friberg et al., perceptual features of polyphonic music (the key caveat)
**Using perceptually defined music features in music information retrieval.** Anders Friberg, Erwin Schoonderwaldt, Anton Hedblad, Marco Fabiani, Anders Elowsson. arXiv 1403.7923, 2014 (JASA 136(4) 1951, 2014). https://arxiv.org/abs/1403.7923

What it did: 20 and 21 listeners rated nine perceptual features (speed, rhythmic clarity, rhythmic complexity, articulation, dynamics, modality, pitch, harmonic complexity, brightness/timbre) on 100 polyphonic ringtones (MIDI-synthesised pop, loudness-normalised) and 110 film-music clips. Then predicted each rating from 54 MIRToolbox / VAMP audio features with PLS and SVR, 10-fold CV.

Measured: rater reliability for brightness/timbre is acceptable (Cronbach alpha 0.88 / 0.90) but mean inter-subject r only 0.27 / 0.32. Rated brightness correlates r = 0.90 with rated pitch in both experiments (listeners do not separate pitch from brightness in mixes). Prediction from spectral audio features (Table 9, R^2 CV): brightness 0.04 to 0.31; timbre 0.33 to 0.41; speed 0.69 to 0.74 (ringtones), 0.37 to 0.56 (film); dynamics 0.48 to 0.74; articulation 0.43 to 0.49; rhythmic complexity 0.0 to 0.19. From MIDI, notes per second of all voices correlates 0.72 with rated speed, drums nps 0.71. Authors: dedicated features beat the "brute force" set; speed (not tempo or note density) was the perceptually clean concept.

Implication: on full mixes, spectral-centroid-type features explain at most about 30% of perceived brightness and are confounded with register, so "bright" must be scored with a register control (pitch-matched comparisons or a pitch covariate), while "density" has a usable scorer in onset rate / notes per second (r about 0.7 with perceived speed).

### A3. Aljanaki and Soleymani, mid-level perceptual features
**A data-driven approach to mid-level perceptual musical feature modeling.** Anna Aljanaki, Mohammad Soleymani. arXiv 1806.04903, June 2018 (ISMIR 2018). https://arxiv.org/abs/1806.04903

What it did: 5000 15-s music excerpts (Jamendo/Magnatune plus emotion sets), rated by 115 musically screened crowd workers on 7 features including articulation (staccato vs legato) and dissonance ("noisier timbre / more dissonant intervals"). Compared fine-tuned Inception on mel spectrograms against classic algorithms.

Measured: Cronbach alpha articulation 0.80, dissonance 0.74, rhythmic complexity 0.27 (0.47 after retest). Best net (pretrain + finetune) Pearson r on test (read from Fig. 3): articulation about 0.76, dissonance about 0.74. Algorithms: MIRToolbox/Essentia sensory dissonance (roughness) about 0.44 against perceived dissonance; MIRToolbox "leap" attack feature about 0.41 against articulation; inharmonicity 0.22.

Implication: for "rough" a Sethares/Vassilakis roughness on a mix tracks perceived dissonance only at r about 0.44, and for "percussive" an attack/onset algorithm tracks perceived staccato at r about 0.41; a small learned regressor on the released ratings (osf.io/5aupt) roughly doubles agreement and is the cheapest route to a validated "rough" and "percussive" scorer on music.

### A4. Vassilakis roughness (SRA)
**SRA: A web-based research tool for spectral and roughness analysis of sound signals.** Pantelis N. Vassilakis. SMC 2007. http://acousticslab.org/learnmoresra/files/vassilakis2007smc.pdf

Definition: for every pair of spectral peaks, R = X^0.1 * 0.5 * Y^3.11 * Z, with X = Amin*Amax, Y = 2Amin/(Amin+Amax), Z = exp(-b1 s (fmax-fmin)) - exp(-b2 s (fmax-fmin)), s = 0.24/(s1 fmin + s2), b1 = 3.5, b2 = 5.75, s1 = 0.0207, s2 = 18.96 (Z is Sethares' fit to Plomp-Levelt curves). Total roughness = sum over pairs. Requires reassigned-STFT peak picking with a 14% amplitude threshold.
Measured: r = 0.98 against the author's own roughness ratings, versus 0.73 and 0.87 for older models. Stimuli were controlled tones, not mixes.

Implication: good on tones, but on mixes the pair sum is dominated by dense partials and noise, so A1 (r 0.56 held-out) and A3 (r about 0.44 vs perceived dissonance) are the honest numbers for "rough".

### A5. Timbre Toolbox (abstract-level for validation)
**The Timbre Toolbox: extracting audio descriptors from musical signals.** Geoffroy Peeters, Bruno Giordano, Patrick Susini, Nicolas Misdariis, Stephen McAdams. JASA 130(5) 2902 to 2916, 2011. https://eprints.gla.ac.uk/68491/ , code https://github.com/MPCL-McGill/TimbreToolbox-R2021a

What it is: descriptor extraction (spectral centroid, spread, flatness, rolloff, log-attack time, temporal centroid, harmonic descriptors, ERB-model descriptors) from STFT, harmonic, ERB and energy-envelope representations. The paper's analysis is a redundancy study (correlation plus hierarchical clustering into about 10 independent descriptor classes) on isolated instrument notes. It is not a listening-test validation.

Implication: use it for exact descriptor definitions (centroid, attack time), but it supplies no evidence that any descriptor matches perception in a polyphonic mix.

### A6. HPSS percussiveness and onset density
- **Harmonic/Percussive Separation using Median Filtering.** Derry FitzGerald. DAFx 2010. https://www.researchgate.net/publication/254583990 (abstract-level). Median filter along time keeps harmonic ridges, along frequency keeps percussive columns; soft masks. No perceptual validation of a percussive-energy ratio exists in this paper; it is a separation method.
- Onset/event density: MIRToolbox "event density" was one of the 54 features in A2; per-second onset counts were the main predictors of perceived speed (A2 Table 9 R^2 up to 0.74 on ringtones, worse on film music).

Implication: a percussive/total energy ratio from HPSS is a defensible objective measurement for "percussive" but has no listening-test anchor; an onset-rate or transcribed notes-per-second measure is the best-anchored scorer we have for "density".

### A7. Sketch2Sound's centroid convention (scorer unit)
Sketch2Sound (C1 below) expresses spectral centroid as a MIDI-like semitone value (Hz to MIDI, /127) and reports control error in semitones. Facchiano et al. (B7) report raw centroid in Hz.

Implication: report "bright" effect in semitones of centroid per unit LPAPS; it makes knob effects comparable across prompts with very different base centroids.

---

## B. Direction estimation beyond mean difference

### B1. ITI, inference-time intervention
**Inference-Time Intervention: Eliciting Truthful Answers from a Language Model.** Kenneth Li, Oam Patel, Fernanda Viégas, Hanspeter Pfister, Martin Wattenberg. arXiv 2306.03341, June 2023 (NeurIPS 2023). https://arxiv.org/abs/2306.03341

What it did: logistic probes on every attention head output; intervene on top-K heads by probe accuracy, shifting by alpha times the activation std along the direction, at every token. Measured (LLaMA-7B, Table 3): mass-mean-shift direction True*Info 42.3% at alpha 20, probe-weight direction 34.8% at alpha 15, CCS 33.4%. Mass mean shift paid more CE (2.41 vs 2.21) and KL (0.27 vs 0.06). Default K = 48, alpha = 15. Alpaca truthfulness 32.5% to 65.1%.

Implication: a direction from class means (what CAA already does) beat the probe weight vector; the improvement to borrow is head-level selection by probe accuracy and std-normalised alpha, not probe weights.

### B2. SMITIN (directly relevant: music, probes, self-monitored alpha)
**SMITIN: Self-Monitored Inference-Time INtervention for Generative Music Transformers.** Junghyun Koo, Gordon Wichern, François G. Germain, Sameer Khurana, Jonathan Le Roux. arXiv 2404.02252 (v2 Feb 2025, IEEE OJSP). https://arxiv.org/abs/2404.02252 , code github.com/merlresearch/smitin

What it did: MusicGen-large (48 layers x 32 heads = 1536 heads). Logistic probes per head on last-step self-attention head outputs, trained on MUSDB/MoisesDB mixes with vs without a stem (8 to 13 h per instrument; 3-s segments; unconditional mode). Direction = probe weight theta, normalised by std of projections. Soft weights for all heads w = ((acc - acc_min)/(acc_max - acc_min))^3 instead of top-K. Sparse intervention every s = 5 tokens. Self-monitoring: compute median probe probability C over top-16 heads; threshold tau = median(acc) - std(acc); if C < tau, keep intervening with weight decayed by (1 - delta C); if C >= tau, switch off. alpha = 5.

Measured: probe accuracy drums 94.3%, bass 89.1%, guitar 81.8%, piano 75.3%. Continuation success rate (avg): unconditioned 7.4%, text "add inst" 12.8%, original ITI 49.9% (FAD 0.420, similarity 0.896), SMITIN 23.3% (FAD 0.336, similarity 0.913), SMITIN + text 19.8%. Text-to-music: text+"add" 30.8%, ITI 52.0% (FAD 0.538), SMITIN alpha 5 29.7% (FAD 0.487), SMITIN alpha 10 39.5% (FAD 0.497). Internal-probe success correlates with external taggers (CLAP rho 0.75, ConvNeXt 0.75) and with humans (rho 0.58 avg). Even n = 10 paired examples (1 minute of audio) gives 31.0% full-set success. MOS: no intervention 3.35, SMITIN 3.34, ITI 3.28.

Implication: the closed-loop trick (stop steering once an internal probe says the property is present) bought large FAD and similarity gains at a cost in success rate, which maps onto our knob as "steer until the measured descriptor reaches the knob's target, then hold", and it needs only a linear readout per step.

### B3. CAA
**Steering Llama 2 via Contrastive Activation Addition.** Nina Panickssery, Nick Gabrieli, Julian Schulz, Meg Tong, Evan Hubinger, Alexander Matt Turner. arXiv 2312.06681, Dec 2023 (ACL 2024). https://arxiv.org/abs/2312.06681

What it did: residual-stream mean difference at the answer token over hundreds of A/B contrast pairs, added at all post-prompt positions. Measured: best layer 13 for Llama-2-7B, 14 to 15 for 13B; multipliers -1..+1 for multiple choice, larger multipliers degraded open-ended text; MMLU roughly unchanged (about 0.63). Authors stress that hundreds of diverse pairs reduce noise.

Implication: the layer sweep matters more than the estimator, and pair count/diversity is a direct lever on our vectors (our prompt sets should be varied in genre and register so the direction does not absorb those).

### B4. ReFT / LoReFT
**ReFT: Representation Finetuning for Language Models.** Zhengxuan Wu, Aryaman Arora, Zheng Wang, Atticus Geiger, Dan Jurafsky, Christopher D. Manning, Christopher Potts. arXiv 2404.03592, Apr 2024 (NeurIPS 2024). https://arxiv.org/abs/2404.03592

What it did: learned intervention Phi(h) = h + R^T (W h + b - R h), R in R^{r x d} with orthonormal rows, applied at a few prefix/suffix positions of chosen layers. Measured: 0.025% to 0.031% of LLaMA-13B parameters (15x to 65x fewer than LoRA), and it matched or beat LoRA-family PEFT on commonsense, arithmetic, instruction-tuning and GLUE.

Implication: a rank-1 to rank-4 LoReFT per knob, trained against a descriptor target with our knob value folded into b (or a scalar times the edit), is a trainable generalisation of our fixed vector with per-step cost of one small matmul.

### B5. Concept Sliders
**Concept Sliders: LoRA Adaptors for Precise Control in Diffusion Models.** Rohit Gandikota, Joanna Materzynska, Tingrui Zhou, Antonio Torralba, David Bau. arXiv 2311.12092, Nov 2023 (ECCV 2024). https://arxiv.org/abs/2311.12092

What it did: train a LoRA so that eps*(x, c_t, t) = eps(x, c_t, t) + eta (eps(x, c+, t) - eps(x, c-, t)), with optional preservation prompts; image-pair variant trains on paired images. SDEdit-style: base model for the first steps, slider for the rest. Measured: Delta CLIP 3.93 vs Prompt-to-Prompt 1.10; LPIPS 0.06 vs 0.15; interference 0.10 vs 0.33; distorted hands 62% to 22%. In TADA on ACE-Step, Concept Sliders AUC 0.081 (MuQ), below localized CAA 0.098, and localization hurt it.

Implication: as a LoRA, its cost at a fixed scale is zero after merge, but a live knob needs the LoRA scale applied per step (non-merged path) and TADA measured it below localized CAA on audio, so it is a baseline rather than an upgrade.

### B6. h-space / Asyrp (abstract-level for numbers)
**Diffusion Models already have a Semantic Latent Space.** Mingi Kwon, Jaeseok Jeong, Youngjung Uh. arXiv 2210.10960, Oct 2022 (ICLR 2023). https://arxiv.org/abs/2210.10960

What it did: shows the U-Net bottleneck ("h-space") is a semantic latent with homogeneity, linearity and consistency across timesteps; learns delta-h via CLIP directional loss, applied only in an editing interval [T, t_edit], with a separate quality-boosting interval near t = 0. I did not extract numeric results.

Implication: the "edit only in an early interval, leave late steps alone" design is the main transferable idea; TADA's start-step sweep (Table 15) suggests the gain on audio is small, so test end-step cut-off (stop steering in the last steps) rather than start-step.

### B7. Activation patching / DiffMean on MusicGen, including bright vs dark timbre
**Activation Patching for Interpretable Steering in Music Generation.** Simone Facchiano, Giorgio Strano, Donato Crisostomi, Irene Tallini, Tommaso Mencattini, Fabio Galasso, Emanuele Rodolà. arXiv 2504.04479, Apr 2025. https://arxiv.org/abs/2504.04479

What it did: MusicGen (melody variant). DiffMean vectors from four sets of 100 LLM-written prompts (fast/slow, bright/dark), layer sweep, then inject one layer's vector into all layers ("one-to-all"); layer 16 for tempo, layer 10 for timbre. Scored with BeatThis BPM and spectral centroid, quality with FAD. Measured (lambda 1.25): bright centroid 2161 Hz (+20% relative), dark 1077 Hz (+40.1% relative), FAD 1.78 / 3.49; slow 85.9 BPM (+36.1%), fast 176.0 BPM (+30.1%), FAD 6.76 / 4.64. FAD rises sharply past lambda about 1.5. A clear brightness shift appears with only 10 prompts; it stabilises around 25.

Implication: the only paper steering "bright/dark" directly, and it scored it with spectral centroid, not text similarity; it also shows the dark direction moves centroid twice as far as the bright direction, so our bright knob should be evaluated per sign.

### B8. MusicRFM (Recursive Feature Machines)
**Steering Autoregressive Music Generation with Recursive Feature Machines.** Daniel Zhao, Daniel Beaglehole, Taylor Berg-Kirkpatrick, Julian McAuley, Zachary Novack. arXiv 2510.19127 (v2), ICLR 2026. https://arxiv.org/abs/2510.19127

What it did: MusicGen-large; RFM probes on mean-pooled residual states trained on the synthetic SynTheory set; the AGOP matrix's top eigenvectors are the directions; injection h' = h + eta_l(t) q with eta_l(t) = eta_0 w_l phi(t) psi_p(t) (layer weights, deterministic rise/decay/sinusoid schedules, Bernoulli gating). Measured: note accuracy 0.23 to 0.82 at eta_0 = 0.60 with CLAP within about 0.02 of unsteered; FD and MMD rise with strength; listening scores 72.9 to 73.5 vs lower for naive variants; RFM probes beat FFN probes (scales 0.956 vs 0.905). Multi-direction: notes 0.92 but CLAP drops to 0.13.

Implication: the AGOP top eigenvector is a nonlinear-probe-derived direction that beat linear probes on music concepts, and per-layer weights plus a time schedule were part of the winning recipe; both are cheap to adopt.

### B9. SAEs in diffusion: SDXL Turbo
**Unpacking SDXL Turbo: Interpreting Text-to-Image Models with Sparse Autoencoders.** Viacheslav Surkov, Chris Wendler, Mikhail Terekhov, Justin Deschenaux, Robert West, Caglar Gulcehre. arXiv 2410.22366, Oct 2024. https://arxiv.org/abs/2410.22366

What it did: k-sparse SAEs (k = 10, 5120 features) on residual updates of 4 cross-attention blocks, trained on 1.5M LAION-COCO prompts (384M vectors per block). Steering adds beta times a decoder column. Measured: block up.0.1 specialises in colour/illumination/style, up.0.0 local detail, down.2.1 composition; feature specificity CLIP 0.71 vs 0.50 random; sensitivity 0.60 vs 0.06 baseline. SAEs trained on 1-step Turbo transfer to 4-step and to base SDXL.

Implication: block specialisation suggests "texture" properties like brightness and roughness may sit in different blocks from "structure" properties like density, so run the localization sweep per knob, not once.

### B10. SAEs on audio latents (Paek et al.)
**Learning Interpretable Features in Audio Latent Spaces via Sparse Autoencoders.** N. Paek, Y. Zang, Q. Yang, R. Leistikow. arXiv 2510.23802, Oct 2025 (NeurIPS 2025 MechInterp workshop). https://arxiv.org/abs/2510.23802

What it did: SAEs on VAE latents of SAO and DiffRhythm plus EnCodec/WavTokenizer; linear maps from SAE features to binned pitch, loudness and spectral centroid ("timbre"). Measured (DiffRhythm VAE): pitch probe accuracy 0.75 to 0.87, loudness 0.17 to 0.49, centroid 0.17 to 0.46. Interventions with alpha in {1,10,20,30} changed the targeted property. On 500 MusicCaps prompts over 32 steps: pitch converges first (about step 21), then timbre, loudness last and still unresolved at the final step.

Implication: centroid is only weakly linearly decodable from the VAE latent, and timbre settles late in the trajectory, which argues for keeping our brightness/warmth steering active into the later steps and for reading descriptors from decoded audio (or a nonlinear probe), not from a linear latent readout.

### B11. Latent-space probe guidance on Stable Audio Open (pitch class)
**Pitch-class Steering for Diffusion-based Music Generation via Latent-space Probes.** Yushi Ye, Wilson Zheng, Yongyi Zang. arXiv 2609.04516, Sep 2026. https://arxiv.org/abs/2609.04516

What it did: a 125k-parameter 2-layer 1-D conv probe on SAO VAE latents (trained on MAESTRO with noise augmentation sigma up to 0.7) predicts 12 pitch-class activations. At inference the probe's BCE gradient updates z_t on steps 20 to 50 of 50, every 2 steps, with an RMS-normalised, clamped step (alpha 0.05 of rms(z_t)). Measured: probe micro-F1 0.663 clean, 0.579 at sigma 1.0. Pitch-class match 0.112 baseline to 0.274 guided (2.4x, p < 1e-6); CLAP 0.274 vs 0.279 (n.s.); FAD-OpenL3 267.7 vs 260.4. Cost not quantified ("minimal").

Implication: a tiny probe on the VAE latent avoids decoding and backprop through the DiT, so descriptor guidance is affordable per step; for our knobs it is the cheapest differentiable-descriptor route, provided the probe for a given descriptor actually decodes (B10 warns centroid is weakly linear).

### B12. FreeSliders (score-space sliders, tested on SAO)
**FreeSliders: Training-Free, Modality-Agnostic Concept Sliders for Fine-Grained Diffusion Control in Images, Audio, and Video.** arXiv 2511.00103, Oct 2025. https://arxiv.org/abs/2511.00103

What it did: eps = eps(x, c_base) + eta (eps(x, c+) - eps(x, c-)) after k initial steps (k = 4 of 36 on SAO, Heun), so two extra forward passes per step (reported about 40% overhead). ASTD: detect saturation of an alignment/preservation ratio over eta in {0, 0.5, ..., 16}, then fit a monotone reparameterisation so equal knob steps give equal perceived change. Audio concepts were sound classes (cat meow, rain, choir...). Measured: authors report ASTD nearly doubles their overall score on images and audio; TADA measured FreeSliders at 0.073 MuQ AUC on ACE-Step, equal to PCI.

Implication: the two extra NFEs rule out the score-space method for streaming, but ASTD's monotone reparameterisation of the knob axis is free at inference and directly fixes a non-uniform knob.

### B13. Steering in rectified-flow DiTs (FlowChef, SHIFT, SteeringDiffusion)
- **FlowChef: Steering Rectified Flow Models in the Vector Field for Controlled Image Generation.** Maitreya Patel, Song Wen, Dimitris N. Metaxas, Yezhou Yang. arXiv 2412.00100, Nov 2024. https://arxiv.org/abs/2412.00100 (abstract-level). Gradient skipping: apply the loss gradient to the state without backprop through the model. No numbers extracted.
- **SHIFT: Steering Hidden Intermediates in Flow Transformers.** Nina Konovalova, Andrey Kuznetsov, Aibek Alanov. arXiv 2604.09213, Apr 2026. https://arxiv.org/abs/2604.09213 . FLUX.1; directions from SVM normal or mean difference on text tokens after shared attention; "a single time-independent vector works across timesteps", early steps most effective; strength modulated by a classifier confidence. Measured: I2P nudity 97 detections vs 364 best baseline; FID 33.9 to 34.5 vs 31.6.
- **SteeringDiffusion: A Bottlenecked Activation Control Interface for Diffusion Models.** Fangzheng Wu, Brian Summa. arXiv 2605.01653, May 2026. https://arxiv.org/abs/2605.01653 . SD1.5: text to k-dim code via 2-layer MLP, FiLM/AdaGN on normalised activations, zero-init, sigmoid gate suppressing early steps; trained on about 250 images per style. Measured: 0.88M params, about 3 ms overhead, 33% to 80% higher style shift than LoRA r = 4 at matched content preservation, strictly monotone where LoRA collapsed; removing the timestep gate caused non-monotonicity at high scale.

Implication: for a flow DiT, a time-independent vector is acceptable (SHIFT), confidence-modulated strength is the shared pattern (SHIFT, SMITIN), and a trained FiLM-style bottleneck with a timestep gate is a measured way to get monotone knobs at about 1M params.

---

## C. One-shot or signal-conditioned timbre control for text-to-audio diffusion

### C1. Sketch2Sound (directly relevant: brightness as a control)
**Sketch2Sound: Controllable Audio Generation via Time-Varying Signals and Sonic Imitations.** Hugo Flores García, Oriol Nieto, Justin Salamon, Bryan Pardo, Prem Seetharaman. arXiv 2412.08550, Dec 2024 (ICASSP 2025). https://arxiv.org/abs/2412.08550

What it did: on a latent DiT T2A (VAE 64 ch at 40 Hz), adds per-frame loudness (A-weighted), spectral centroid (Hz to MIDI /127) and CREPE pitch probabilities (zeroed below 0.1). Each control goes through one linear projection and is added to the noisy latent z before the DiT. 40k fine-tuning steps; 20% per-control dropout plus 20% all-drop; random median filters (1 to 25 frames) on controls during training. CFG with separate s_text = 5, s_ctrl = 1.
Measured (median filter 10 at test, Table I): text-only CLAP 0.27 FAD 2.57; loudness+centroid: centroid error 4.39 semitones (vs 10.37 without centroid control), RMS error 3.60 dB, CLAP 0.306; full: centroid 4.43 st, pitch 1.49 st, CLAP 0.312, FAD 2.51. Without median filtering control error drops (centroid 3.21 st) but FAD worsens to 3.53.

Implication: a single linear layer per descriptor, trained 40k steps, makes brightness (and loudness) a calibrated, time-varying input in semitone units with no per-step cost beyond CFG; this is the strongest evidence for an amortised "bright" (and, by the same recipe, onset-density) knob.

### C2. Music ControlNet
**Music ControlNet: Multiple Time-varying Controls for Music Generation.** Shih-Lun Wu, Chris Donahue, Shinji Watanabe, Nicholas J. Bryan. arXiv 2311.07069, Nov 2023. https://arxiv.org/abs/2311.07069

What it did: spectrogram diffusion with ControlNet-style zero-conv branch; controls are top-1 chroma (melody), dB energy smoothed with Savitzky-Golay (dynamics), RNN beat/downbeat probabilities (rhythm); per-control MLP to frequency bins; masking for partially specified controls. 41M params, about 1800 h, 5 days on 32 A100 pretrain plus 3 days on 8 A100. Measured: melody accuracy 47.1% vs MusicGen 41.3% (extracted), 82.6% vs 55.2% (created).

Implication: confirms smoothed energy and beat curves are learnable control signals, but the ControlNet copy-branch cost is far above Sketch2Sound's linear layer for the same kind of signal.

### C3. SAO ControlNet (melody)
**Editing Music with Melody and Text: Using ControlNet for Diffusion Transformer.** Siyuan Hou, Shansong Liu, Ruibin Yuan, Wei Xue, Ying Shan, Mangsuo Zhao, Chao Zhang. arXiv 2410.05151, Oct 2024. https://arxiv.org/abs/2410.05151

What it did: copies the first N SAO DiT blocks as a control branch with zero-init linear outputs; top-4 CQT per channel melody; curriculum masking; 2240 h instrumental; 4 V100 for about 3 days. Measured: melody accuracy 56.6% vs MusicGen 44.7%; editing CLAP 0.396 vs 0.354; FD-openl3 97.7 vs 190.8.

Implication: the SAO-family DiT accepts a copy-branch control at modest cost, but it doubles the cost of the copied blocks every step, which is the wrong trade for a knob.

### C4. MuseControlLite
**MuseControlLite: Multifunctional Music Generation with Lightweight Conditioners.** Fang-Duo Tsai et al. arXiv 2506.18729, June 2025. https://arxiv.org/abs/2506.18729

What it did: decoupled cross-attention adapters with RoPE on SAO for melody, rhythm, dynamics and audio inpainting; zero-init conv sums; multiple CFG. 85M trainable (vs 572M for SAO ControlNet), 1.7k h, 40k steps on one RTX 3090. Measured: melody accuracy 61.1% vs 56.6%; without RoPE melody accuracy collapses 58.6% to 10.7%.

Implication: a time-varying descriptor curve (centroid, onset density, HPSS ratio) can be trained into SA3-class DiTs on one consumer GPU, and positional encoding of the control stream is essential.

### C5. Audio ControlNet / T2A-Adapter
**Audio ControlNet for Fine-Grained Audio Generation and Editing.** Haina Zhu et al. arXiv 2602.04680, Feb 2026. https://arxiv.org/abs/2602.04680

What it did: on FluxAudio (MMDiT), loudness, pitch and sound-event controls via a 38M-param 1-D conv adapter into cross-attention (vs 410M ControlNet), 120k steps on 4 x 48 GB GPUs. Measured: loudness MAE 1.40 vs 13.36 baseline; pitch MAE 148 vs 252; event F1 54.4 vs 10.0.

Implication: another data point that a small adapter gives near-exact energy-curve adherence, the closest analogue of a density/percussive control.

### C6. MuseTimbre (on SA3 medium)
**MuseTimbre: Zero-Shot Timbre Transfer by Controlling a Frozen Music Generator.** Yuan-Chiao Cheng, Zhiyao Duan. arXiv 2609.30548, Sep 2026. https://arxiv.org/abs/2609.30548

What it did: frozen Stable Audio 3 Medium; pitch module (Basic Pitch piano roll, 1-D CNN, cross-attention with rotary embeddings) plus timbre module (fine-tuned LAION-CLAP audio embedding via AdaLN in every layer). 250k steps on 2 GPUs. Measured: timbre match 57.6% (PaSST family level) vs 48.7% CTD and 45.4% SS-VQ-VAE; pitch frame-F1 0.316 (source ceiling 0.540); MUSHRA timbre 2.6/5. About 1 s on an RTX 5090 for 5 s audio at 25 Euler steps, CFG 2.

Implication: AdaLN injection of a global vector into every SA3 layer is proven to steer timbre on this exact backbone, which is the natural home for a trained per-knob "warm/bright" embedding if fixed vectors plateau.

### C7. MusicGen-Style (abstract-level)
**Audio Conditioning for Music Generation via Discrete Bottleneck Features.** Simon Rouard, Yossi Adi, Jade Copet, Axel Roebel, Alexandre Défossez. arXiv 2407.12563, July 2024. https://arxiv.org/abs/2407.12563 . Frozen feature extractor (EnCodec/MERT/MusicFM) plus transformer, RVQ bottleneck and downsampling on 1.5 to 4.5 s excerpts, as an extra conditioner; autoregressive model. No numbers extracted.

Implication: style-from-reference is a different control axis from our scalar knobs; the bottleneck design (quantise so the conditioner cannot copy) is the relevant lesson if we ever condition a knob on a reference clip.

### C8. DITTO and DITTO-2 (inference-time optimisation)
- **DITTO: Diffusion Inference-Time T-Optimization for Music Generation.** Zachary Novack, Julian McAuley, Taylor Berg-Kirkpatrick, Nicholas J. Bryan. arXiv 2401.12179, Jan 2024 (ICML 2024). https://arxiv.org/abs/2401.12179 . Optimises the initial noise x_T through the full 20-step DDIM sampler with gradient checkpointing against any differentiable feature loss; 70 optimisation steps for intensity, 150 for melody/structure. Measured (intensity, Table 3): MSE 4.758 (DITTO) vs 4.785 DOODL, 23.29 FreeDoM, 38.41 ControlNet; FAD 0.682; CLAP 0.433. About 1.86 s per optimisation step; 82 s per intensity example (from DITTO-2's table).
- **DITTO-2: Distilled Diffusion Inference-Time T-Optimization for Music Generation.** Same authors. arXiv 2405.20289, May 2024 (ISMIR 2024). https://arxiv.org/abs/2405.20289 . Optimise through a 1-step consistency/CTM-distilled surrogate, then decode with T in [1, 8] steps. Measured: intensity 5.47 s vs 82.19 s, MSE 3.311, FAD 0.640; melody 22.5 s vs 230.8 s, accuracy 85.2%; structure 11.75 s vs 245.3 s.

Implication: even the distilled version costs seconds per clip and optimises a whole clip, so DITTO is unusable per step in streaming; its value for us is offline, to generate high-precision descriptor-matched training targets for an amortised knob.

---

## D. Whole-model or step-function alternatives, compared for a streaming budget

Per-step cost budget: our current knob costs one vector add at a few blocks (effectively zero). The table is per denoising step, relative to one DiT forward.

| Approach | Example paper | Training | Per-step inference cost | Measured control quality |
|---|---|---|---|---|
| Fixed vector add (current) | TADA, Facchiano | none (prompt pairs) | about 0 | TADA localized CAA 0.098 MuQ AUC; bright centroid +20%, dark +40% (Facchiano) |
| Per-step self-monitored alpha | SMITIN | linear probes, 1 min to hours of audio | probe dot products, about 0 | ITI 49.9% success / FAD 0.420 vs SMITIN 23.3% / 0.336 |
| Learned low-rank intervention | LoReFT, SteeringDiffusion | small, per knob | one rank-r matmul or FiLM, about 3 ms | SteeringDiffusion +33% to 80% style shift vs LoRA, monotone |
| Descriptor linear-projection adapter | Sketch2Sound | 40k steps fine-tune | one linear layer, plus CFG pass if s_ctrl separate | centroid error 4.4 st, loudness 3.6 dB |
| Cross-attn / AdaLN adapter | MuseControlLite, MuseTimbre, T2A-Adapter | 40k to 250k steps, 1 to 4 GPUs | extra cross-attn or AdaLN per layer, small | melody 61.1%; loudness MAE 1.40; timbre 57.6% |
| LoRA slider | Concept Sliders | 500 iters per concept | LoRA matmuls if scale must vary live | TADA 0.081 MuQ AUC (below localized CAA) |
| Prompt-embedding slider | Prompt Sliders (Sridhar, Vasconcelos, arXiv 2409.16535, Sep 2024, abstract-level) | textual inversion per concept | about 0 (one embedding) | 30% faster than LoRA, 3 KB per concept; TADA's text/token-embedding steering 0.099 to 0.120 holdout AUC |
| Score-space slider | FreeSliders | none | +2 forward passes | about PCI level on ACE-Step |
| Probe-gradient guidance on latent | Ye et al. (SAO), Readout Guidance | tiny probe/head | backprop through probe only (latent probe) or through DiT features (readout) | pitch-class 0.112 to 0.274; Readout Guidance 3x to 5x slower |
| Inference-time noise optimisation | DITTO, DITTO-2 | none | 2x model calls per opt step, tens of steps | best MSE, but 5 to 230 s per clip |

Readout Guidance (Grace Luo, Trevor Darrell, Oliver Wang, Dan B Goldman, Aleksander Holynski, arXiv 2312.02150, Dec 2023, CVPR 2024, https://arxiv.org/abs/2312.02150): timestep-conditioned readout heads (5.9M to 8.5M params) on frozen U-Net decoder features, trained on as few as 100 images; guidance eps <- eps + w grad d(r, f(x_t)). Measured: 350x less data than T2I-Adapter at matched performance, but 3x to 5x slower inference. Implication: a readout head on our DiT features is the right *monitor* for SMITIN-style alpha control (forward only, no gradient), but using its gradient breaks the streaming budget.

Measured Sliders (Yijia Chen, Boyu Wei, Xuanhua Yin, arXiv 2609.05234, Sep 2026, https://arxiv.org/abs/2609.05234): multi-branch LoRA sliders trained against closed-form differentiable image measurements with an "effect and stillness" loss, then a post-hoc decoded calibration from raw coefficient to natural-variation units. Measured on SDXL: lighting rho 0.995, 98.9% monotone sweeps, selectivity 1.76 vs 0.54 best baseline; five-attribute average selectivity 2.59 vs 1.50; 96.7% of pairs compose. Image only. Implication: this is the closest published recipe to "train each knob against its measured descriptor and penalise movement in the other four", and its calibration step costs nothing at inference.

---

## Ranked: five most actionable ideas

1. **Score the knobs with measured descriptors on decoded audio, not MuQ/CLAP text similarity.** Use centroid in semitones and HF-energy ratio (bright), the AudioCommons warmth regression or low-mid ratio (warm), onset rate or transcribed notes per second (density), HPSS percussive energy ratio plus AudioCommons hardness (percussive), and Vassilakis/AudioCommons roughness (rough). Report effect per unit LPAPS, per sign, and add a pitch/register covariate for "bright". Evidence: Facchiano et al. scored bright/dark with centroid (+20% / +40%); Sketch2Sound measures centroid error in semitones; AudioCommons brightness r 0.81 held-out, warmth r 0.79, hardness r 0.87; but Friberg et al. show spectral features explain only 4% to 31% of perceived brightness in polyphonic music and brightness is r 0.90 confounded with pitch; Aljanaki shows algorithmic roughness tracks perceived dissonance at only r about 0.44 (a fine-tuned net reaches about 0.74). Keep MuQ/CLAP as a secondary axis, and treat "rough" scores as the least trustworthy.

2. **Re-estimate each direction from descriptor labels on self-generated audio, then localize per knob.** Generate a few hundred clips, measure the descriptor, and take the mass-mean difference between top and bottom descriptor quantiles (or the top AGOP eigenvector of a small nonlinear probe) at each block, rather than prompt-pair CAA; pick blocks by AUC per knob, steer the conditional branch only, and scale alpha by the activation std along the direction. Evidence: ITI found the mass-mean direction (42.3%) beats probe weights (34.8%); MusicRFM's AGOP directions beat linear probes on music concepts; TADA localization +46% AUC and cond-only 0.109 vs 0.100; Surkov et al. show texture and composition live in different blocks; CAA notes many diverse pairs reduce noise.

3. **Closed-loop (self-monitored) strength using a forward-only readout.** Train one linear or tiny probe per knob on DiT hidden states (or on x0-hat latents) that predicts the measured descriptor; each step (or each streamed window), raise alpha while the readout is below the knob's target and decay it once reached. This turns the knob into a target value in descriptor units at near-zero cost. Evidence: SMITIN (median probe vs threshold, decaying weights) cut FAD 0.420 to 0.336 and raised similarity 0.896 to 0.913 versus constant ITI, with probes trained on as little as 1 minute of audio; SHIFT modulates strength by classifier confidence; Readout Guidance heads need about 100 examples. Caveat from Paek et al.: centroid is only weakly linearly decodable from VAE latents (0.17 to 0.46), so read from hidden states or use a nonlinear head.

4. **Amortise the strongest knobs into a tiny trained conditioner (Sketch2Sound recipe).** For bright and density first: one linear projection per descriptor curve (centroid in semitones, onset density) added to the noisy latent, or an AdaLN/FiLM vector, trained about 40k steps on SA3 with control dropout and median-filtered controls, with CFG on the control. Per-step cost is one linear layer (plus a CFG pass only if a separate control scale is wanted). Evidence: Sketch2Sound centroid error 4.4 semitones vs 10.4 without the control, at 40k steps; MuseControlLite trained 85M params on one RTX 3090; T2A-Adapter loudness MAE 1.40 vs 13.36; SteeringDiffusion's 0.88M FiLM bottleneck was monotone where LoRA collapsed, at about 3 ms; MuseTimbre proves AdaLN injection works on SA3 medium. DITTO-2 can generate offline, descriptor-matched targets if self-labelled data is too noisy.

5. **Calibrate the knob axis and the step window per knob.** Fit a monotone map from knob position to measured descriptor change (FreeSliders' ASTD; Measured Sliders' decoded calibration) so equal knob moves give equal effect, and choose the active step window per knob: keep timbre knobs (bright, warm, rough) active into the late steps, test end-step cut-offs rather than start-step delays. Evidence: Measured Sliders 98.9% monotone sweeps after calibration at zero inference cost; FreeSliders reports ASTD nearly doubling its score; Paek et al. show timbre settles after pitch (pitch about step 21 of 32) and loudness last; TADA's start-step sweep moved AUC by about 0.01 at most; SteeringDiffusion's ablation shows an ungated schedule turns non-monotone at high scale.

Not recommended for streaming: score-space sliders (FreeSliders, +2 NFEs per step), gradient guidance through the DiT (Readout Guidance 3x to 5x slower), and DITTO-style optimisation (5 s to 230 s per clip). Concept Sliders LoRA is a reasonable baseline but TADA measured it below localized CAA on audio (0.081 vs 0.098 MuQ AUC).
