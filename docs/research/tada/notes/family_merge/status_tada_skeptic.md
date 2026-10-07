# TADA skeptic audit (runbook 20_tada_skeptic.md)
- 2026-10-04 07:55 started. Worktree read: DEMON-tada-sa3 @ 206f3198. GPU_LOCK held by sa3-tada (E2 resid patching sweep running, ~97% util); non-GPU checks first, metric check A on CPU from cached PCI endpoint audio.
- 07:58 coordinator: GPU checks may run shared beside the sa3-tada E2 sweep without taking the lock (eager, small batches, no TRT builds). Any timing in this file is from a SHARED GPU and is not representative.

## A. Metric positive control
- 08:20 Source: cached PCI-all endpoints, eval50 (50 benchmark prompts, seed 2115, 8 steps): alpha +8 = positive prompt at every block for every step (= rendering e.g. "fast song, {p}"; text enters SA3 only through cross-attn context), -8 = negative prompt, 0 = neutral ("a song, {p}"). Scored with the reference compute_muqt_alignment / compute_clap_alignment (the exact functions the AUC uses). E:/Projects/tada-replication/sa3/skeptic_check_a.json.
  | concept | metric | pos | neutral | neg | paired pos-neg (sd) | frac pos>neg |
  |---|---|---|---|---|---|---|
  | tempo | MuQ | 0.127 | 0.079 | 0.023 | 0.104 (0.095) | 0.86 |
  | tempo | CLAP | 0.117 | 0.120 | 0.102 | 0.015 (0.035) | 0.60 |
  | piano | MuQ | -0.031 | -0.074 | -0.096 | 0.065 (0.113) | 0.76 |
  | piano | CLAP | 0.049 | 0.026 | 0.024 | 0.025 (0.050) | 0.66 |
  | mood | MuQ | 0.094 | 0.081 | 0.069 | 0.024 (0.050) | 0.68 |
  | mood | CLAP | 0.107 | 0.102 | 0.099 | 0.008 (0.046) | 0.60 |
- Reading: MuQ separates the real prompt for tempo and piano, weakly for mood. CLAP barely separates on SA3 for any of the three (0.008 to 0.025, 60 to 66 percent), so every SA3 CLAP AUC near 0 is uninterpretable: the metric cannot see the concept even when the prompt itself says it.
- Scale: the WHOLE prompt swap moves MuQ by +0.048/-0.056 (tempo, pos/neg vs neutral), +0.043/-0.022 (piano), +0.012/-0.012 (mood). CAA loc tempo at the cutoff already reaches +0.040 (pos) and +0.059 (neg, sign-corrected): i.e. on tempo CAA moves MuQ about as far as the real prompt does at the same LPAPS budget. The AUC is a raw area (not normalised), so SA3's small attainable delta caps every method's AUC; 0.334 (SAO) is not a reachable reference on this model. Cross-check on the reference ACE data (steer-audio/results/steering_metrics.csv, my reimplementation of auc.py): ACE PCI-all tempo MuQ AUC 0.052/0.075 and ACE CAA loc tempo 0.111/0.099, consistent with Table 1 scale, so the AUC code is not the problem.
- Running: PCI-all through the full protocol + AUC on SA3 (tempo, piano, mood) = the real-prompt AUC ceiling (results_subdir protocol_results_skeptic, lane files untouched).

- 08:25 PCI-all (the REAL prompt, switched in for the last k of 8 steps at every block) through the full reference protocol (LPAPS, MuQ, CLAP; eval50; SA3's own cutoffs; results_subdir protocol_results_skeptic, E:/.../sa3/skeptic_check_a2.json). This is the within-model reference (what the real prompt gets on SA3; not an upper bound):
  | concept | PCI-all MuQ AUC pos/neg (avg) | PCI-all CLAP pos/neg | CAA loc MuQ avg (status) | CAA all MuQ avg |
  |---|---|---|---|---|
  | tempo | 0.070 / 0.050 (0.060) | 0.012 / 0.009 | 0.050 | 0.050 |
  | piano | 0.050 / 0.008 (0.029) | 0.017 / 0.003 | 0.007 | 0.020 |
  | mood | 0.015 / 0.002 (0.008) | 0.010 / -0.006 | 0.002 | 0.002 |
  | mean of 3 | 0.032 | 0.008 | 0.020 | 0.024 |
- For contrast, on the reference ACE data the real prompt (PCI-all) gets tempo 0.052/0.075 and CAA loc beats it (0.111/0.099). On SA3, CAA reaches about 60 to 80 percent of the real prompt's AUC (tempo 83 percent), but the real prompt's own AUC is only about 0.03 (MuQ) and 0.008 (CLAP).
- Caveat (from G, confirmed at source): PCI renders start from a neutral wrapper ('a song, {p}', '{p}, with instrument', sa3_tada_run.py _pci_triple) while CAA starts from the raw benchmark prompt {p} (cmd_sweep), as the reference does; so the CAA/PCI ratio compares two different zero points.
VERDICT: the metric sees the concept on SA3 for MuQ (tempo, piano; mood weakly) but barely for CLAP, and its range is small: the real prompt itself (PCI-all) scores MuQ AUC 0.032 mean over tempo/piano/mood (tempo 0.060), CAA 0.020 to 0.024 on the same three. PCI is a within-model REFERENCE, not a ceiling (on ACE, CAA beats it; corrected after G). So the 0.011 vs 0.334 comparison is not a like-for-like effect size, CLAP AUCs near 0 are uninterpretable on SA3, and the honest headline is 'CAA on SA3 moves MuQ about two thirds as much as the real prompt does at the same LPAPS budget, both small', not 'TADA fails'.

## B. Oracle vs patch consistency (tempo, CAA pairs 0-24, seed 2115, batch 25, strength 1.0; E:/.../sa3/skeptic/B/)
- Cached oracle diff vs a fresh pos-minus-neg render: max abs difference 0.0 (bit-identical; cache, token alignment (238 tokens per row, memory tokens included on both sides), dtype all fine).
- Step 0, rel L2 vs the clean (pos) run's cross_attn_output: oracle add on the NEG prompt: block 3 0.0002 (fp16 rounding of the stored diff: exact), block 6 0.36, block 7 0.36; output patch (exact clean outputs) 0.0 everywhere; K/V patch 0.38 / 0.61 / 0.38; unsteered neg 0.85 / 0.97 / 0.67. Later steps diverge (step 7: oracle 0.81 to 1.23).
- Why it does not match the K/V patch, and why that is not a bug: the oracle diff is (clean run - corrupted run) output, so it reconstructs the clean OUTPUT, and only at the first patched block of the first step; blocks 6/7 sit downstream of a residual that blocks 0-2, 4-5 still compute from the negative prompt, so their raw output no longer equals the cached negative one. The K/V patch is a different quantity by construction: the clean context read by the corrupted query (rel 0.38 from the clean output at block 3 step 0). Oracle == K/V patch is not an identity.
- Impact (eq. 2, joint blocks 3/6/7, 25 pairs, s_none = neg render, s_all = pos render):
  | intervention | MuQ "a fast track" | CLAP fusion "fast song" (localisation score) | LPAPS vs neg |
  |---|---|---|---|
  | oracle add, alpha 1 | 0.16 | -0.03 | 3.58 |
  | output patch (exact clean outputs, every step) | 0.30 | 0.19 | 3.74 |
  | K/V patch (clean context) | 0.47 | 0.65 | 3.83 |
- What eq. 2 measures in our code: CLAP (laion_clap fusion, default ckpt) text-audio cosine against "This is a music of {clean concept prompt}" (e.g. "fast song"), mean over clips, then (s_l - s_none)/(s_all - s_none) floored at 0 (acestep/tada/patching.py:39-45; scripts/tada/sa3_tada_score.py cmd_patch). Not a latent distance. The per-concept denominators are small (happy 0.011, sad 0.039, slow 0.049 CLAP), so single-concept impacts are noisy.
- FINDING (design flaw in E2, not a code bug): the E2 "oracle" sweep adds the diff of CAA pair i ("fast song, a song" - "slow song, a song") to the render of BENCHMARK prompt i ("Upbeat indie pop track ..."), sa3_tada_run.py cmd_oracle (pairs from C.prompt_pairs) vs cmd_sweep (prompts from C.benchmark_prompts). It is not the exact activation difference of the rendered prompt and trajectory, so "oracle FAIL" over-reads: it is a per-token vector from a different prompt. The correct oracle on its own pair (above) still carries only 0.16 (MuQ) of the concept, versus 0.30 for exact output substitution and 0.47 to 0.65 for the K/V patch.
VERDICT: the add is numerically correct (block 3 step 0 exact, cache bit-identical); the mismatch with the K/V patch is structural. Mechanism: on SA3 the concept rides on the trajectory-adaptive cross-attention READ (K/V patch at 3 blocks recovers 47 to 65 percent) and is lost when frozen into any fixed additive output (exact outputs 19 to 30 percent, oracle add 0 to 16 percent). E2's oracle was mislabelled (wrong prompt), but a correct oracle confirms the direction of its conclusion. "Not a lever" should read "a weak lever for fixed additive vectors; K/V (context) is the strong one".

## C. The cutoff
- 08:25 Reproduced CAA loc tempo eval50 AUC from the raw per-strength CSVs with an independent trapezoid (E:/.../sa3/eval50/caa_loc_tempo/protocol_results): MuQ pos 0.0273 / neg 0.0727, CLAP -0.0009 / -0.0038; auc.json says 0.027309 / 0.072746 / -0.000897 / -0.003829. Exact match.
- Cutoff is SA3's own: min(PCI-all, PCI-loc) max LPAPS at the full-swap endpoints of SA3's own PCI renders on the same 50 prompts and seed (tempo pos 3.211 = PCI-loc, neg 3.400 = PCI-loc). No constants carried from the paper (reference ACE cutoffs 3.2 to 4.5 are similar in size).
- Raw tempo CAA loc curve, MuQ (alpha, LPAPS, sign-corrected delta), no cutoff: pos 1.6 (1.24, -0.001), 3.2 (1.73, 0.005), 4.8 (2.10, 0.007), 6.4 (2.50, 0.014), 8.0 (2.72, 0.018), 9.6 (2.85, 0.033), 11.2 (3.00, 0.041), 12.8 (3.20, 0.040) | 14.4 (3.34, 0.051), 16.0 (3.52, 0.060). neg -3.2 (1.66, 0.013), -6.4 (2.41, 0.029), -9.6 (2.88, 0.048), -12.8 (3.28, 0.059) | -16 (3.65, 0.066) ... -32 (4.58, 0.135). CLAP flat within +-0.006 across the whole range both ways.
- Admitted: pos 8 strengths (1.6..12.8), neg 4 strengths (-3.2..-12.8) plus the interpolated cutoff point. Not a one-or-two-strength artifact. Beyond the cutoff MuQ keeps rising monotonically (neg reaches 0.135 at LPAPS 4.6), so the steer is a real, monotone lever whose effect per unit LPAPS is small; the cutoff truncates it where the real prompt swap would.
- The LPAPS jump at the smallest strength (1.24 at alpha 1.6) is not SA3-specific: ACE reference CAA loc tempo jumps to 1.70 at its first strength too.
VERDICT: AUCs reproduce exactly from the raw tables and the cutoff is SA3's own PCI with 4 to 8 admitted strengths; the low AUC is not a cutoff artifact.

## D. Localisation control (shared GPU)
- Control pair set: 32 MusicCaps captions from the patching CSV, clean "This is a recording. {caption}" vs corrupted "Here is an audio clip. {caption}" (musically identical text change, concept-free), seeds 222 and 223, same K/V patch sweep (none, all, each of 24 blocks), 8 steps, 10 s. E:/.../sa3/skeptic/D_patch/control.
- Score: eq. 2 needs a concept text, which a concept-free pair has none of (s_all - s_none would be ~0 and the ratio undefined), so I used eq. 2 with an AUDIO target: s = CLAP-fusion cosine of each clip to its own clean (all-patched) render, so s_all = 1; impact = (s_l - s_none)/(1 - s_none), floored at 0. Same score applied to cached real-concept patch audio (seeds 0-1, 64 clips) for a like-for-like comparison. E:/.../sa3/skeptic/D_impact.json.
  | pair set | s_none | top blocks (impact) | blocks 3 / 6 / 7 |
  |---|---|---|---|
  | control (neutral) | 0.857 | tf4 0.072, tf0 0.038, tf3 0.037 | 0.037 / 0.018 / 0.000 |
  | fast | 0.655 | tf6 0.150, tf3 0.043 | 0.043 / 0.150 / 0.021 |
  | slow | 0.744 | tf6 0.191, tf3 0.107 | 0.107 / 0.191 / 0.024 |
  | female | 0.696 | tf7 0.309, tf1 0.116 | 0.048 / 0.000 / 0.309 |
  | flute | 0.594 | tf7 0.320, tf11 0.064 | 0.040 / 0.010 / 0.320 |
- The control curve is flat (max 0.07, nothing above tau 0.10, blocks 6 and 7 at 0.02 and 0.00), while the concepts peak where the text-target localisation put them and at different blocks per concept family (tempo at tf6, vocal/timbre at tf7). The audio-target score reproduces the text-target per-concept tops (fast tf6, slow tf6, female tf7, flute tf7).
VERDICT: localisation is concept-specific, not generic sensitivity; the 3/6/7 set is a union of concept-dependent peaks (tf6 tempo/mood, tf7 timbre/vocal, tf3 mood), which also means a single "loc" set of 3/6/7 applies two off-target blocks to every concept.

## E. Hook position at source
- Vendor block (C:/Users/ryanf/.daydream-scope/models/demon/sa3/vendor/stable-audio-3/stable_audio_3/models/transformer.py, adaLN branch used by SA3 medium): self-attn is gated (1027-1029: x*sigmoid(1-gate_self), self_attn_scale, + residual); cross-attn at 1033/1035 is `x = x + self.cross_attn_scale(self.cross_attn(self.cross_attend_norm(x), context=...))` with NO adaLN gate or scale; cross_attn_scale = nn.Identity unless layer_scale (926; SA3 medium config has no layer_scale). FF gated after (1047-1049). So the module output IS what enters the residual.
- Eager capture and add: acestep/engine/sa3_tada.py:87-96 forward hook on blocks[i].cross_attn (module output, before cross_attn_scale and the residual add); recorder (acestep/tada ActivationRecorder via sa3_target :49-55) and oracle (_oracle_context, scripts/tada/sa3_tada_run.py:517) and output patching (_OutputPatcher) hook the same module output. Step counter = forward pre-hook on block 0 (sa3_tada.py:81-85), one DiT forward per sampler step at cfg 1 (oracle cache has exactly 8 calls per block).
- TRT: acestep/engine/trt/sa3_steering_onnx.py:122-157 find_cross_attn_outputs takes the single tensor produced inside /transformer/layers.i/cross_attn/ and consumed outside it (to_out MatMul output, read by the block's residual Add), and 221-240 rewires every consumer to out_i + steering_xattn[:, i]. Same point as the eager hook. Caveat: steering_xattn is [1, blocks, hidden], one vector per block broadcast over tokens and batch, so per-token or per-pair (oracle) steering is eager-only; every offline table (CAA, AUSteer, E1, E2) ran eager (sa3_tada_run._load_sam = eager fp16), so the TRT path does not enter any AUC.
- Text entry, at source: model_config.json conditioning: cross_attention_cond_ids [prompt, seconds_total], global_cond_ids [seconds_total], no prepend/input_concat ids; dit.py:197-198 cross_attn_cond -> to_cond_embed -> context of every block; dit.py:243-247/270-271 adaLN global = seconds_total + timestep only. Cross-attn conditioning mask is force-disabled (dit.py:414). Confirmed: text enters only as cross-attn K/V.
- Side findings (minor, do not move the headline): (1) T5Gemma padding_mode is "learned" (model_config + conditioners.py:38), so padded prompt tokens are a learned NON-zero embedding; the caakv variant's "non-padding" mask (sa3_tada_run.py:599 and :630, abs-sum > 0) is all ones, so the K/V-site vector is averaged over and added to all 257 tokens including ~240 padding tokens and the seconds token. The status claim "SA3 masks padding by replacing context tokens" is right in substance (no attention mask) but padding is learned, not zero. (2) The CAA vector is also added to the 64 memory-token rows (SAO has none); harmless for the K/V patch, a small deviation for steering.
VERDICT: eager and TRT add at the same point (cross_attn module output, ungated, straight into the residual); text enters only via cross-attn K/V; no hook-position bug.

## F. Ear (tempo)
- I cannot listen; I measured the ear files and the 50-prompt sweep with acoustic proxies (librosa onset rate, beat-tracker tempo, spectral flatness and centroid). Not a substitute for the user's ears.
- Ear files (listen_sa3, prompt 27 breakbeat): zero 103 BPM, flatness 0.051, centroid 2902 Hz; +12.8 129 BPM, flatness 0.091, centroid 3574 Hz (brighter and noisier); -12.8 99 BPM, flatness 0.062, centroid 3082 Hz.
- 50-prompt CAA loc tempo sweep (eval50) vs alpha 0: +12.8 onset rate 3.97 -> 4.34/s (62 percent of clips up), flatness 0.025 -> 0.032, centroid +150 Hz; -12.8 onset rate 3.97 -> 3.40/s (only 16 percent of clips up), centroid -176 Hz. Reference, the real prompt (PCI full swap): "slow song" 2.90/s, "fast song" 3.75/s (the neutral render 3.83/s). Beat-tracker tempo barely moves for any of them (tempo median 117 to 129, octave-error prone).
VERDICT: tempo steering changes the sound in the intended direction on event density (negative clearly sparser and darker; positive denser but also brighter and noisier, i.e. part texture/noise), at a smaller effect than the real "slow song" prompt; it steers weakly rather than only degrading.

## G. Second opinion (Codex, gpt-6-astra)
- 08:33 `codex exec -m gpt-6-astra --cd C:\_dev\projects\DEMON-tada-sa3 -s read-only` (flags accepted; prompt = the Astra brief verbatim + a read-only note + status_tada_sa3.md + findings A to F, via stdin). Full answer: notes/family_merge/astra_tada_review.md. Each objection, checked at source:
- "A state-dependent output add A(q,c0)+[A(q,c+)-A(q,c0)] equals the K/V patch, so the output site IS causally a lever; what fails is compressing it into fixed vectors": CONFIRMED (algebraic identity; consistent with B, where exact frozen outputs recover less than the adaptive K/V read). "Site is not a lever" must become "fixed additive vectors are a weak lever there".
- "E2 oracle is not an oracle for the evaluated prompts" (pairs from C.prompt_pairs, applied to C.benchmark_prompts): CONFIRMED (sa3_tada_run.py cmd_oracle vs _render_alpha/cmd_sweep; B found it independently).
- "Even a same-pair frozen difference is not trajectory-adaptive": CONFIRMED (B: exact at block 3 step 0 only).
- "PCI is a baseline, not a ceiling": CONFIRMED (ACE reference data: CAA loc tempo 0.111/0.099 beats PCI-all 0.052/0.075). A's wording corrected.
- "PCI and CAA start from different base prompts": CONFIRMED (_pci_triple neutral wrapper vs raw prompt in cmd_sweep; same as reference design). Caveat added to A.
- "Endpoint-only PCI cutoff assumption is false in one curve (PCI-all piano neg max at -7: 3.3753 vs -8: 3.3683)": CONFIRMED from my full PCI-all LPAPS (protocol_results_skeptic), but IMMATERIAL for the reported cutoffs here: piano/tempo/mood cutoffs come from PCI-loc (2.41 to 3.40), far below PCI-all's 3.37 to 3.81; PCI-loc interior points were never scored, so the same issue there stays OPEN (size bounded by the 0.007 seen on PCI-all).
- "Compare to the 4 SAO-table concepts, not 9": CONFIRMED, SA3 CAA loc MuQ over piano/tempo/mood/vocal_gender = 0.0163 (both directions) and 0.0034 (positive only, the SAO configs are positive-only), still far below 0.334.
- "Localisation scored with CLAP for every concept; the paper uses MuQ for non-vocal concepts": CONFIRMED deviation (sa3_tada_score.py cmd_patch is CLAP-only; DIGEST says MuQ for mood/tempo/instruments/genres). OPEN whether MuQ scoring changes the block set; D's audio-target score reproduces the per-concept tops, which argues it would not move much.
- "64 memory tokens are 27 percent of the CAA token mean": CONFIRMED (acestep/tada/target.py time_mean averages every row; 64 of 238 rows). Effect on AUC unmeasured; a deviation from SAO (no memory tokens).
- "caakv padding mask invalid with learned padding, includes seconds token": CONFIRMED (same as E side finding).
- "Local renorm lacks the reference clamp(min=1e-8)": CONFIRMED (acestep/tada/caa.py:92-95 vs reference controller.py:259); harmless in practice (main tables renorm off; no zero-norm rows), cosmetic fix.
- "Renorm off is consistent with the SAO reference path": CONFIRMED (steer-audio stable_audio_caa/method.py:52 renorm_after_steer=False).
- "Released SAO loc config is tf4tf11tf12tf13tf18, 100 steps, CFG 7, positive-only": CONFIRMED (already in DIGEST); SA3's 3-block set is not the same protocol shape.
- "30-step run uses ordinally remapped 8-step vectors and 8-step cutoffs": CONFIRMED (_steps_for floor(s*n/steps); status already caveats it). So E1 does not rule out a sampler/time-window effect; OPEN.
- "Guidance mostly reparameterises alpha (range scaled 1/g)": CONFIRMED by the status's own table (g3/g5/g7 within 0.005 of g1).
- "fp16 may erase small additions": REFUTED for these runs (the smallest strength already moves LPAPS to 1.2 to 1.7 and MuQ moves monotonically; the add is not lost to rounding).
- "TRT delta accuracy / binding issues": AGREED but irrelevant to the offline tables (eager fp16 only, E); deployment-only.
- "Audio serialisation (stereo mean, int16 clip) and same-query-both-directions": CONFIRMED as implemented, both match the reference protocol layout; not a deviation.
- "Resume could mix configs (skip-if-exists)": OPEN in general; sweep.json records guidance/renorm/range per directory and E1 used separate subdirectories, so no evidence of mixing.
- "50-prompt subset, fixed AUSteer budgets, 10 strengths per side, no bootstrap CIs": CONFIRMED deviations already stated in the status; no CI means "no effect" is unbounded.
- Astra's top three: (1) matched positive control from identical base prompts with all PCI points scored and human labels, (2) adaptive output-difference identity test vs K/V patching on the benchmark prompts, (3) sampler-native 8 vs 30 step vectors with their own PCI and time-window pulses.
VERDICT: Astra found no new code bug; it confirms the E2 oracle mislabel and overturns two over-reads (mine: PCI as a "ceiling"; the lane's: "site is not a lever"). Supported claim: fixed additive vectors at cross_attn_output give small, monotone MuQ gains on SA3 (about two thirds of the real prompt's own small AUC), localisation gives no steering advantage, and the stronger negative claims are unproven.

## Summary
- No code bug found that would change any table; no commit made. Two labelling/design errors: E2 "oracle" used the wrong prompt (CAA pair diff applied to benchmark prompts); "not a lever" and "0.011 vs 0.334" over-read a metric whose within-model range on SA3 is small (real prompt MuQ AUC 0.032 mean on tempo/piano/mood).
SKEPTIC DONE none
