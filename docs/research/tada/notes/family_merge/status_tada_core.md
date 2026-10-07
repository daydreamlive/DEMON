# Lane CORE status (TADA replication)

STATE: DONE. core 1e90846c, docs 20db2cce, AUSteer + combine cab0451c (ryanontheinside/feat/tada-core on 64e565c5)

## DIGEST

Sources (local): E:/Projects/tada-replication/tada_2602.11910.pdf (arXiv 2602.11910v3, 27 Sep 2026; text dump tada.txt
alongside), E:/Projects/tada-replication/steer-audio (clone of github.com/luk-st/steer-audio @ 3516910, MIT,
Copyright (c) 2026 Lukasz Staniszewski). Paper: "TADA! Tuning Audio Diffusion Models through Activation Steering",
Staniszewski, Zaleska, Modrzejewski, Deja (Warsaw Univ. of Technology, IDEAS). Paper and repo win over our notes.

### BLOCK INDEXING (read this first)
The PDF numbers cross-attention layers 1-BASED (figures 1..24). The repo uses 0-BASED block names `tf<i>` =
`transformer_blocks.<i>`. Paper ACE-Step {7,8} == repo `tf6tf7` == transformer_blocks.6 and .7. Paper Stable Audio
Open {12,13,14} == repo tf11, tf12, tf13. The published impact table (results/tables_localization.md) confirms:
ACE tf7 0.488, tf6 0.292, tf5 0.094, rest < 0.04. SAO tf12 0.511, tf11 0.222, tf13 0.117, tf4 0.095, tf5 0.068, tf18 0.066.
AudioLDM2 (U-Net, 64 attn): paper layers 45-51, I(attn51)=0.36, I(attn45)=0.33, I(attn46)=0.15.

### Models (confirmed from PDF Sec. 4 and App. K)
- ACE-Step v1 ("Ace-Step [24]"; the repo vendors ACE-Step v1 code under src/models/ace_step/ACE): flow-matching
  Linear DiT, mT5 text encoder, 24 blocks, cross-attn output width 2560 (AUSteer: "two active layers expose 2d = 5120").
- Stable Audio Open (1.0, diffusers StableAudioPipeline; DiT, T5 text encoder, 24 blocks, cross-attn module `attn2`).
- AudioLDM2 (U-Net, FLAN-T5). Not a DEMON family; context only.
DEMON: our ACE is v1.5 and our SA3 is Stable Audio 3 medium; neither is a paper checkpoint.

### Hook point (the activation everything uses)
OUTPUT OF THE CROSS-ATTENTION MODULE, before it is added to the residual stream (Eq. 1: h_l = h_{l-1} + CrossAttn(h_{l-1}, c)).
- ACE: inside LinearTransformerBlock.forward, `attn_output = self.cross_attn(...)`; steer that tensor, then
  `hidden_states = attn_output + hidden_states`. Self-attn and FF untouched. NOT the post-block residual.
  (src/models/ace_step/ace_steering/controller.py register_vector_control.)
- Stable Audio Open: forward hook on `transformer.transformer_blocks.<i>.attn2` output (diffusers CFG batch = [uncond, cond]).
- Patching point: the cross-attn K and V, implemented as replacing the cross-attn INPUTS (ACE: encoder_hidden_states,
  encoder_attention_mask, rotary_freqs_cis_cross; SAO: attn2 inputs) with the clean run's. Since K = c W_K and
  V = c W_V, this is "layer l sees the clean prompt". Config variants also patch only `.to_k` or `.to_v` outputs.
- Paper Limitations: residual stream, self-attn and FF are explicitly NOT studied.
=> The steering-generic slot (post-block residual) is NOT TADA's hook. Family lanes must add a `cross_attn_out`
   hook point (eager + TRT engine input) and a cross-attn conditioning patch point (per-layer text context).

### Technique 1: activation patching (layer localisation), Sec. 3-4, App. E, F
- Counterfactual pairs (P_c clean containing the concept, P_~c corrupted). The clean run caches each layer's
  cross-attn K,V (every call, every step); the corrupted run with the SAME initial latents substitutes layer l's K,V
  with the clean ones.
- Impact (Eq. 2): I(l,c) = [s(l<-c, rest<-~c) - s(all<-~c)] / [s(all<-c) - s(all<-~c)], s = audio-text similarity
  between the generation and the concept text. Per concept: L single-layer patches + 2 references (none patched =
  corrupted, all patched = clean); N_gen = C*(L+2)*P*S (Eq. 3).
  The published localization_impact.csv has min 0.0 and max 1.3125: values are floored at 0, NOT capped at 1.
  The impact-aggregation script is not in the repo; the formula is from the PDF.
- I(l) = mean over concepts of I(l,c); functional set = {l : I(l) >= tau}, tau = 0.10 (Table 7: ACE {7,8} stable
  for tau in [0.10, 0.20]; tau 0.05 and 0.075 give {6,7,8}).
- Settings ACE (configs/patch_config/ace.yaml): 30 s, 50 steps, Euler, cfg_type apg, guidance_scale 7.0,
  guidance_scale_text 6.0, guidance_scale_lyric 1.0, omega 7.0, guidance_interval 1.0, lyrics "" (mood and tempo
  experiments set lyrics "[inst]"), seed 222, n_latents_per_prompt 8 (S = 8 seeds), up to 256 pairs per concept
  (230 average), text padded to 128 tokens. SAO: 9 s in config (paper says 10 s), 100 steps, guidance 7.0, negative
  prompt "Low quality, average quality.", fp16, seed 222, 8 latents per prompt.
- Prompt budget: Table 6, the ACE bottleneck is recovered exactly (Jaccard top-2 = 1.00) with 34 pairs per concept;
  12 pairs gives 0.98. Cost at P=34: 70,720 generations, about 33 GH200 GPU-hours.
- Similarity: MuQ-MuLan ("OpenMuQ/MuQ-MuLan-large", template "A {p} track", audio resampled to 24 kHz) for mood,
  tempo, instruments, genres; CLAP for vocal gender (LAION-CLAP music ckpt music_audioset_epoch_15_esc_90.14.pt,
  HTSAT-base, template "This is a music of {p}"). Metric per concept in the published CSV (ACE):
  female, male clap_1; drums, fast, happy, jazz, reggae, violin muqt_1; sad, slow muqt_2.
- Localisation concepts: ACE (10) female, male, fast, slow, happy, sad, drums, violin, reggae, jazz.
  SAO (12) female, male, fast, slow, happy, sad, violin, flute, maracas, guitar, techno, reggae.
- Dataset: HF lukasz-staniszewski/patching-music-musiccaps-prompts (original_feature, clean_prompt, corrupted_prompt),
  from MusicCaps captions; keywords and replacements in src/preprocess/features.py (Table 5). Eval prompts per concept
  in configs/patch_data/musiccaps/<concept>_ace.yaml (violin: clap "Violin music", muqt "violin-based"; female: clap
  ["A female vocal singing", "A male vocal singing"], muqt ["female singer", "male singer"]). CORE copies these.

### Technique 2: CAA (ported first), Sec. 3, App. I.1.4 and I.2
- Vector per diffusion step t and layer l: v = mean_i(hbar_c^(i) - hbar_~c^(i)), hbar = cross-attn output averaged over
  batch and time frames, then L2-NORMALISED (Eq. 6; compute_standard_steering_vectors divides by the norm).
  Per step (30 for ACE), per layer.
- Collection: one generation per prompt, seed 10, 30 steps, guidance_scale 5.0, 30 s, lyrics "[inst]" (vocal concepts
  lyrics ""), guidance_scale_text 0 and guidance_scale_lyric 0 (2 CFG passes), cond-pass activations used.
  N = 50 pairs per concept.
- Application: h_l <- h_l + alpha * v_{t,l} broadcast over time frames, ONLY on the conditional CFG pass
  (cond_only; Table 17: cond+null no better). Negative direction = negative alpha.
  The released eval configs pass `renorm: true`: after adding, each time frame is rescaled back to its original
  L2 norm. The controller default is renorm False; the paper benchmark used renorm true.
  "Localized" = tf6tf7; "all" = all 24 blocks; "ablated" = all except tf6tf7.
- ACE eval grid: 15 negative + 0 + 15 positive, alpha_k = range * k / 15, per-concept calibrated ranges (loc):
  piano [-132.0726, 121.6698], mood [-115.1507, 68.9759], tempo [-61.9471, 66.6927], vocal_gender [-173.2238, 124.7203],
  violin [-136.9527, 150.0571], vocal_style [-63.7567, 98.2866], guitar_electronic [-76.7974, 78.7766],
  rock_genre [-101.5888, 120.7404], electronic_music [-147.1525, 184.9562]. Eval seed 2115, 30 steps, CFG 5.0, 30 s.
- SAO CAA (configs/steering/stable_audio): 100 steps, guidance 7.0, 10 s, compute seed 10, fp16, layers all,
  normalize_sv true; eval 16-point positive-only grids (piano 0..12.12 step 0.808), eval seed 2115, localized
  target_layers tf4tf11tf12tf13tf18. This does NOT match the paper's {12,13,14} = tf11..tf13 (it is the tau=0.05
  set minus tf5). Released SAO vectors only for piano, mood, tempo, vocal_gender.

### Concept sets (verbatim; CORE copies the data into acestep/tada/data/)
Steering benchmark (9; repo keys): piano, mood, tempo, vocal_gender, vocal_style, guitar_electronic, violin,
rock_genre, electronic_music. Templates (Table 8; 50 base prompts each, src/steering/methods/sae/lib/configs/steer_prompts.py):
  piano "{base}, with piano" vs "{base}"; violin "{base}, with violin" vs "{base}";
  guitar_electronic "{base}, with acoustic guitar" vs "{base}, with electric guitar";
  mood "happy song, {base}" vs "sad song, {base}"; tempo "fast song, {base}" vs "slow song, {base}";
  vocal_gender "{base}, with female vocal" vs "{base}, with male vocal" (lyrics "");
  vocal_style "{base}, with rap vocal" vs "{base}, with sing vocal" (lyrics "");
  electronic_music "classical song, {base}" vs "electronic song, {base}"; rock_genre "jazz song, {base}" vs "rock song, {base}".
Alignment queries (Table 9, configs/eval.py): piano "a piano song", mood "a cheerful track", tempo "a fast track",
vocal_gender "female vocal singing" (code: "This is music with female vocal singing"), vocal_style "rap vocal"
(code: "This is music with rap vocal"), guitar "a song with acoustic guitar", violin "a song with violin",
rock_genre "a jazz song", electronic_music "a classical song"; MuQ uses "This is a music of <query>" for non-vocal ones.
Benchmark prompts: 100 test prompts (src/steering/eval/test_prompts.py), disjoint from the base prompts.

### Evaluation protocol (Sec. 5.2, src/steering/eval/auc.py, eval_steering_protocol.py)
- Preservation: LPAPS (Manor and Michaeli; CLAP HTSAT-base music ckpt; 4 Swin stage features, normalised, squared
  diff summed over channels, averaged, summed over stages; 10 s windows, 0.1 overlap, mean) vs the alpha=0 output of
  the same prompt and seed.
- Alignment: sign-corrected delta a(alpha) = sign(alpha) * (sim(alpha) - sim(0)), sim = windowed music CLAP and MuQ-MuLan.
- AUC: per direction, trapezoid of delta over LPAPS, points in increasing |alpha| order, truncated at the cutoff with
  one interpolated point exactly at the cutoff; cutoff = PCI max LPAPS (min over PCI-all and PCI-loc), i.e. the
  distance of a full prompt swap. Reported pos, neg, avg=(pos+neg)/2. (The paper writes p = LPAPS_max - LPAPS; the
  code integrates over LPAPS; same area.)
- Smoothness (CSM): within the cutoff, delta / max(clip(delta, 0)), sorted by |alpha|, std of consecutive gaps (lower better).
- Audio quality: Audiobox Aesthetics CE/CU/PC/PQ interpolated at 15 uniform LPAPS points per direction, averaged.
- Extra preservation axes (App. H.3): harmony (CQT chroma frame cosine), rhythm (mir_eval beat F), melody (pYIN raw
  pitch accuracy), structure (SSM Pearson clamped [0,1]).
- Sample counts: N = 100 prompts x 31 alphas, 30 s, per concept per method. Hyperparameters chosen on 20 held-out
  prompts x 9 strengths. User study: 32 listeners, 1279 ratings, 15 trials (piano, female vocal, tempo).

### Headline numbers (ACE-Step, Table 1, mean over 9 concepts, both directions)
CAA all: AUC MuQ 0.075, CLAP 0.046. CAA loc: 0.104 / 0.065. AUSteer loc 0.096 / 0.063. SAE loc 0.118 / 0.070.
Concept Sliders 0.086 / 0.067. CAA localisation gain +39% / +40% AUC.
CAA all / loc / ablated, 4 concepts (App. L Tables 22-24): ACE avg AUC MuQ 0.055 / 0.083 / 0.021; SAO 0.310 / 0.334 / 0.158.

### Other techniques (lane SAE scope; CORE does not port)
- AUSteer: per-dim sign-consistency score over pairs x frames, global top-s over active layers per step, additive
  (Eq. 9; multiplicative Eq. 8 gave about 0). s in {256..16384} all, {256..5120} loc.
- SAE: BatchTopK SAEs on cross-attn outputs of ACE tf6 (m,k)=(2,32) 10 epochs and tf7 (4,64) 15 epochs; activations
  every 5th of 30 steps on MusicCaps prompts; TF-IDF feature score (Eq. 12); v = sum of top-k decoder columns
  (per-concept k pairs in App. I.2); h += alpha*v; negative = -alpha.
- Concept Sliders: LoRA on W_Q, W_K, W_V, W_O of every block, 500 iters, AdamW 1e-4, eta 7, rank per concept (Table 13).
- Non-activation baselines: PCI, Text Embeddings, Token Embeddings, FreeSliders. Multi-concept: uniform sum of vectors.

### Released artifacts and licences
Code MIT. HF user lukasz-staniszewski ("TADA Steering Collection"): ace-step-caa-<9 concepts> (sv.pkl,
pos_vectors.pkl, neg_vectors.pkl, config.json), ace-step-austeer-*-{all,tf6tf7}, ace-step-cs-*-r{4,8,16}-{all,tf6tf7}
(Concept-Slider LoRAs, apache-2.0 tag), ace-step-sae-tf6-cross-attn, ace-step-sae-tf7-cross-attn,
ace-step-sae-scores-<9>, ace-step-tokemb-*, ace-step-rfm-*, stable-audio-caa-{piano,mood,tempo,vocal-gender},
audioldm2-caa-*, dataset patching-music-musiccaps-prompts. CAA, SAE and dataset cards declare no licence (code is MIT).
All ACE artifacts are for ACE-Step v1 (2560-wide cross-attn); SAO artifacts for Stable Audio Open. They do not apply
to ACE-Step v1.5 or SA3 weights; usable only as format and reference fixtures.
Results CSVs: steer-audio/results/. LPAPS code: steer-audio/editing/AudioEditingCode/evals (MIT Hila Manor; LPAPS
licence inherited from LPIPS, BSD-style).

DIGEST READY

## Core port (2026-10-03)

acestep/tada/: target.py (ActivationTarget, ModuleTarget, ActivationRecorder,
ConditioningPatcher, forward_counter), patching.py (impact score, sweep,
localize, tau 0.10), caa.py (caa_vectors, stack_vectors -> [K,S,H],
steer_activation, alphas_from_range), packs.py (caa_pack / write_caa_pack:
hook cross_attn_output, blocks, cond_only, renorm via ACE's save_pack),
metrics.py (AUC, smoothness, cutoff, quality; CLAP/MuQ/mir_eval lazy),
concepts.py + data/ (verbatim reference data). A family plugs in by giving
ModuleTarget its cross-attention modules and conditioning kwarg names.
Tests: tests/unit/test_tada_core.py 17 pass; with steering seam/packs/xattn
hook/sa3 engine tests 57 pass. CPU only.

CORE READY 1e90846c

## AUSteer + multi-concept (2026-10-03)
acestep/tada/austeer.py (austeer_scores, select_top_s, DEFAULT_TOP_S 256), packs.austeer_pack (method auscore), caa.combine(vectors, negate), recorder reduce=frames, austeer_eval data. Tests 22 pass (62 with steering suites). Note: vector is sparse signed beta (paper Eq. 7 and reference code), not a 0/1 indicator.

CORE READY2 cab0451c
