# SA3 medium steering knobs: how each vector was made (read-only audit, 2026-10-05)

Scope: worktree C:\_dev\projects\DEMON-steer, branch ryanontheinside/feat/steering-generic, HEAD 83f5b77b.
All file:line refs are in that worktree unless stated otherwise. Nothing was modified, no GPU used.

## 0. Where the packs are

- NO pack file is in the worktree or in git. `git ls-files | grep safetensors` is empty and
  `git log --all -- '*.safetensors'` is empty. NOT IN REPO.
- The five packs exist only in the default packs dir (outside the repo):
  `C:\Users\ryanf\.daydream-scope\models\demon\steering_packs\sa3\medium\{bright,density,percussive,rough,warm}.safetensors`
  (resolved by `steering_packs_dir()`, acestep/paths.py:219-230: `$DEMON_STEERING_PACKS_DIR` else `<models>/steering_packs`).
  I read them read-only on CPU (8-byte length prefix + JSON header, numpy) to get headers, norms and cosines.
- File mtimes (local): bright 14:50:04, warm 14:53:10, rough 14:56:18, density 14:59:15, percussive 15:15:39, all 2026-10-03.
- Commits that matter (all 2026-10-03, RyanOnTheInside): 1885aeec 14:29 slot; 12f08551 14:34 pack format;
  4581c568 14:42 SA3 steer TRT engine; 7bf6e260 15:19 discover.py/proxies.py/sanity tool; 4beb08d4 docs.
  So the packs were written before discover.py was committed. No commit message explains why these five
  concepts were chosen beyond discover.py:69-70 ("the four ACE demo axes ... plus one extra", the extra = percussive).

### Provenance mismatch (important)
Every pack header says `"method_source": "TADA, arXiv 2602.11910 (CAA + activation patching)"`.
The committed writer emits a different string: `"prompt-pair difference of means + residual patching block choice"`
(scripts/steering/discover.py:547). `git log --all -S"TADA, arXiv 2602.11910 (CAA"` finds nothing:
the packs were produced by an uncommitted earlier version of discover.py. The exact code that produced them is
NOT IN REPO. Everything else in the header (keys, pos/neg strings, bases, defaults) matches the committed
script, so the committed script is the best available description, but it is not proven identical.
docs/STEERING.md:10-19 and discover.py:7-9 now say explicitly "not a replication of TADA".

Also: commit 7bf6e260 and docs/STEERING.md:122-124 say calibration is "stored in provenance". It is not:
discover.py:545-557 omits it (calibration goes only to the optional `--report` JSON, discover.py:561-564),
and no pack header has a calibration key. No `--report` JSON exists in the worktree: NOT IN REPO.

## 1. Summary table

All five share one construction; they differ only in prompts, proxy, chosen block and norm.

| knob | hook site | block (of 24) | steps | estimator | data (pos / neg descriptors) | proxy (block choice only) | raw norm | magnitude (per knob unit) |
|---|---|---|---|---|---|---|---|---|
| bright | post-block residual, additive, all tokens | 23 | all (range 0..1) | CAA diff of means; token-mean, step-mean, pair-mean; unit-normalised | "bright, crisp, sparkling highs, airy, shimmering treble" / "dark, muffled, dull, lowpassed, murky, no treble" | centroid (Hz) | 18.564 | 1.8564 |
| density | same | 1 | all | same | "sparse, minimal, few instruments, lots of space and silence" / "dense, busy, layered, many instruments, wall of sound" | onset_rate (/s) | 12.896 | 1.2896 |
| percussive | same | 15 | all | same | "heavy drums, punchy percussion, driving beat" / "no drums, beatless, ambient sustained pads" | perc_ratio (HPSS) | 40.154 | 4.0154 |
| rough | same | 2 | all | same | "gritty, distorted, noisy, saturated, rough lo-fi texture" / "clean, smooth, pure, polished, pristine" | flatness | 15.732 | 1.5732 |
| warm | same | 15 | all | same | "warm, deep round bass, mellow full low end" / "thin, tinny, no bass, harsh brittle highs" | lowhigh_db | 29.518 | 2.9518 |

Descriptor strings: discover.py:71-103 (match headers verbatim). Density's POSITIVE pole is "sparse"
(positive knob thins the texture, blurb discover.py:94-95).

## 2. Per-question detail (applies to all five unless stated)

### 2.1 Hook site
- Hook = `post_block_residual`: the output of a whole transformer block (after its residual Add), broadcast to
  every token. acestep/steering/layout.py:28-30; header `"hook": "post_block_residual"`.
- Operation: pure ADD (no affine, no projection, no norm-matching): `hs[rows] = hs[rows] + scale * v.view(1,1,-1)`
  (acestep/engine/stream.py:1910-1912; eager forward hook on each block, stream.py:1918-1921).
  TRT/adapter path fills `buf[rows, layer, :] += scale * v` (stream.py:946-963), then the engine adds it.
- Eager blocks hooked: `wrapper.model.transformer.layers` (acestep/engine/sa3_internals.py:169-185, via
  acestep/engine/sa3_adapter.py:80-89, 116-117). Hidden = block `.dim` (sa3_adapter.py:111-114) = 1536.
- TRT: ONNX surgery adds `Add(out_b, Gather(steering, b))` right after block b's residual Add output, every
  consumer rewired to the steered tensor (acestep/engine/trt/sa3_steering_onnx.py:117-160); "memory tokens
  included" (sa3_steering_onnx.py:11-13).
- Applied on the positive and (if CFG is active) the negative pass: `_current_step_per_row` is set for both
  (stream.py:1408, 1422).

### 2.2 Layers
- ONE block per pack (`block` in header): single-layer injection; no per-layer weights exist.
  Config `"layer": int(p.block)` (acestep/steering/packs.py:337-344); fill skips out-of-range layers (stream.py:955-957).
- Block chosen by residual patching (discover.py:403-428): for the first `patch_pairs` = 8 pairs, rerun the
  negative prompt with block b's contribution (output minus input) replaced by the positive run's at every step
  (BlockRecorder patch mode, discover.py:184-193), decode, measure the proxy,
  effect = (proxy_patched - proxy_neg) / pooled_gap (discover.py:410-421); block = argmax of the MEDIAN effect
  over the 8 pairs (discover.py:427-428). Per-block effects are in each header (section 4).

### 2.3 Denoising steps
- Policy `{"kind":"range","start":0.0,"end":1.0}` in every header = weight 1.0 at every step
  (packs.py:129-132; written at discover.py:544). Curve length n = session steps
  (`prep["steps"]` = `steps_override`, default 8; acestep/streaming/sa3_backend.py:1523, 1568-1571; acestep/streaming/knobs.py:151).
- Per-row gate: the weight is indexed by each ring row's own step index (stream.py:937-944).
- Estimation used all 8 steps of a full denoise (denoise 1.0), averaged equally (discover.py:197-203).

### 2.4 Estimator
- Contrastive activation addition, difference of means. Per (block, step): token mean of the block output over
  dims (batch, tokens) `hs.float().mean(dim=(0,1))` (discover.py:178-183); per run: mean over steps
  (discover.py:197-203); per pole: mean over the 32 runs (discover.py:395-396); `diffs = pos_m - neg_m` (discover.py:397).
- Vector = `diffs[block] / ||diffs[block]||` (unit L2); raw L2 stored as `norm` (discover.py:431-434).
  No PCA, no probe, no SAE, no per-token direction, no centering or whitening. Saved as fp32 [1536] (packs.py:208-212).
- Activations recorded on the EAGER DiT (SA3Driver, discover.py:211-252: SA3Context dit, StreamPipeline depth 1,
  sde/pingpong, noise_on_cpu, seeded renoise); production applies them on the fp16 TRT residual.
- Caveat: the exact producing code is NOT IN REPO (method_source mismatch, section 0).

### 2.5 Data
- Prompt pairs only; no labelled audio set, no AudioCommons/timbral model, no CLAP (discover.py:35-37).
- 32 pairs per concept: pair k = (`"{base}, {pos}"`, `"{base}, {neg}"`), the same seed `1000+k` for both
  (discover.py:363-369); base rotates over 8 neutral prompts (discover.py:107-116): lofi hip hop beat, cinematic
  synthwave, jazz piano trio, deep house groove, acoustic folk song, orchestral film score, indie rock band,
  ambient electronic music. Each base appears 4 times; the 8 patch pairs are seeds 1000-1007, one per base.
- Render: 54 s, 8 steps, sampler pingpong, denoise 1.0 (headers; defaults discover.py:477-482).
- Spectral proxies (scripts/steering/proxies.py, librosa STFT n_fft 2048 hop 512, proxies.py:19-20):
  centroid (power-weighted spectral centroid, l.35-40), lowhigh_db (energy <250 Hz over >2.5 kHz in dB, l.43-48),
  flatness (mean spectral flatness, l.51-55), onset_rate (librosa onsets per second, l.58-62),
  perc_ratio (HPSS percussive energy share, l.65-72). The proxy does NOT shape the vector; it only scores the
  pos/neg gap and picks the block.
- Pooled proxy means (header proxy_pos_mean / proxy_neg_mean):
  bright 829.4 / 218.3 Hz; warm 34.34 / 8.39 dB; percussive 0.3075 / 0.0634; density 4.171 / 4.762 onsets/s
  (sparse lower, consistent); rough 0.001355 / 0.003595 (the "gritty" prompts measured LESS flat than "clean":
  the proxy disagrees with the label, so rough's block choice was normalised by a gap of the wrong sign).
  docs/FAMILIES.md:119-120: "The proxies for rough and density do not track those concepts."

### 2.6 Scale
- Knob spec: default 0, range -30..+30, group "steering" (acestep/streaming/knobs.py:224, 257-290).
  Out-of-range values are clamped (knobs.py:435-441, coerce_knob_values).
- alpha = raw knob value (packs.py:334); zero alpha is dropped (packs.py:335-336; stream.py:1874-1876).
- Shift = knob x magnitude x policy_weight(step) x unit vector: scale = alpha*magnitude (stream.py:1881), times
  the step weight (stream.py:944). magnitude = norm x knob_unit, knob_unit 0.1 (discover.py:483-485, 543):
  knob 10 = one full raw mean difference at that block, knob 30 = 3x. No norm-matching to the residual,
  no clamp on the shift. Several packs on the same block sum (stream.py:1866-1867).
- Knob comment: "useful magnitude roughly 2..15 by ear; breakage above that" (knobs.py:221-223), written for the ACE axes.

### 2.7 Headers verbatim (common fields)
format 1; family "sa3"; checkpoint "medium"; hook "post_block_residual"; hidden_size 1536;
method "caa_diff_means"; policy {"end": 1.0, "kind": "range", "start": 0.0};
provenance: tool "scripts/steering/discover.py", method_source "TADA, arXiv 2602.11910 (CAA + activation patching)",
date "2026-10-03", pairs 32, patch_pairs 8, seed0 1000, steps 8, duration_s 54.0, sampler "pingpong",
denoise 1.0, knob_unit 0.1, bases (the 8 above). No author, no checkpoint hash, no n_samples beyond pairs,
no calibration, no per-block diff norms. Tensors: exactly one, `vector`, F32, shape [1536], offsets [0, 6144],
L2 = 1.0000 in all five. Header JSON lengths: bright 1496, density 1568, percussive 1488, rough 1512, warm 1480 bytes.

Per pack (name / label / block / norm / magnitude / blurb):
- bright / "Bright" / 23 / 18.564006805419922 / 1.8564006805419924 / "positive brightens (spectral centroid up)"
- density / "Density" / 1 / 12.896434783935547 / 1.2896434783935549 / "positive thins the texture toward sparse/minimal (same direction as the ACE steer_density axis)"
- percussive / "Percussive" / 15 / 40.15378952026367 / 4.015378952026367 / "positive pushes toward drums and percussion"
- rough / "Rough" / 2 / 15.732086181640625 / 1.5732086181640625 / "positive adds grit and noise (spectral flatness up)"
- warm / "Warm" / 15 / 29.51801109313965 / 2.951801109313965 / "positive tilts the spectrum toward bass (warmer)"

## 3. Bright vs warm

- Same tool, estimator (token-mean CAA diff of means, unit-normalised), hook type, step policy, knob unit and data
  recipe (32 pairs, same 8 bases, same seeds 1000-1031).
- Different: prompts (treble vs bass descriptors; warm's negative includes "harsh brittle highs", i.e. partly the
  bright pole), proxy (centroid vs lowhigh_db), and block (bright 23 = the last block, warm 15). They share NO
  layer, so no same-layer cosine exists. Cross-block cosine (residuals at blocks 23 vs 15, not like-for-like) = -0.2289.
- Raw norms: bright 18.56, warm 29.52, so per knob unit warm pushes 1.59x harder in absolute terms.

## 4. Cosines, norms, patching

Only one vector per pack is stored (its chosen block). The only shared layer is block 15 (percussive, warm):
cos(percussive, warm) at block 15 = **-0.2222**. Every other pair sits on different blocks.

Cosine matrix of the stored unit vectors (blocks differ except percussive/warm):

|            | bright b23 | density b1 | percussive b15 | rough b2 | warm b15 |
|---|---|---|---|---|---|
| bright     | 1.0000 | -0.1034 | 0.0500 | -0.0010 | -0.2289 |
| density    | -0.1034 | 1.0000 | -0.2083 | -0.1231 | 0.3256 |
| percussive | 0.0500 | -0.2083 | 1.0000 | 0.1341 | -0.2222 |
| rough      | -0.0010 | -0.1231 | 0.1341 | 1.0000 | -0.0586 |
| warm       | -0.2289 | 0.3256 | -0.2222 | -0.0586 | 1.0000 |

Per-layer vector norms: only the chosen block's raw norm is stored (section 2.7). Per-block diff norms
(`diff_norms`, discover.py:440) went only to the report JSON: NOT IN REPO.

Median patching effect per block, b0..b23 (header `patching_effect_per_block`; chosen block in brackets):
- bright: -0.0235 0.0533 0.1302 -0.006 -0.0158 -0.0134 -0.0513 -0.0322 0.0058 -0.0318 -0.003 0.0299 0.0472 0.0038 0.0324 0.0126 0.0909 0.064 0.0113 0.0418 0.1416 0.0421 0.0484 [0.1766]
- warm: -0.1181 0.3648 0.202 0.1493 -0.0047 -0.0297 -0.1237 0.0328 0.0155 0.1047 -0.0015 0.0465 0.0306 0.1698 -0.0828 [0.3648] -0.0491 0.0453 0.2061 0.0725 -0.1213 0.0084 0.0124 -0.1583 (b1 ties b15 at 4 dp; argmax took b15, so b15 was larger at full precision)
- percussive: -0.0168 0.197 0.2717 0.1897 -0.0137 0.1235 0.0812 0.0632 0.1841 0.0984 0.1128 0.0536 0.0603 0.0631 0.0082 [0.4659] -0.0077 0.0584 0.0318 -0.025 0.0894 0.2599 0.2918 0.2253
- rough: -0.0115 0.1446 [0.1894] -0.0438 -0.0318 -0.1107 -0.1769 0.1216 -0.0179 -0.0201 0.0051 0.1305 0.1075 0.0051 0.0276 0.1631 -0.2214 0.1018 -0.0146 -0.0674 -0.3013 -0.5466 -0.1177 -1.7946
- density: -0.7992 [1.3007] -1.5984 -0.9246 0.0 -0.6738 -0.5015 -0.2821 -0.2194 -0.4858 -0.2037 -0.3291 -0.5955 -0.9873 0.0784 0.1724 -0.7679 -1.285 0.3761 -0.6112 0.3604 -0.1254 -0.047 0.2977

Reading: bright's best patch recovers 18% of the gap (small, spread over blocks); density swings past +/-1
between adjacent blocks (b1 +1.30, b2 -1.60), i.e. noise on a small gap (0.59 onsets/s); rough's gap has the wrong sign.

## 5. TRT engine sa3_m_dit_steer_l1_646_646

- Name = `sa3_m_dit_steer_l{min_latents}_{opt_latents}_{max_latents}` (acestep/engine/trt/sa3_build.py:237-259):
  l1 = min 1 latent frame, 646 = opt, 646 = max (the TRT optimisation profile on the latent length). Profile
  (1, 646, 646) "covers the default 54 s session (60 s padded window)" (sa3_build.py:147-159). Build:
  `python -m acestep.engine.trt.sa3_build --all --dit-only --steer` (sa3_build.py:59-60, 489-505); surgery
  version 1 (sa3_steering_onnx.py:35, sa3_build.py:253).
- Input: `steering`, FLOAT32, static [1, 24, 1536] (sa3_steering_onnx.py:117-119); cast once to the residual
  dtype (fp16 in fp16mixed) (l.120-127); one Gather per block (l.128-136) and one Add after each block's
  residual Add (l.138-160). Weights unchanged; zero input = original graph (docs/STEERING.md:47-57).
- Runtime: selected by regex `^(?P<prefix>.+_dit)_steer_l(?P<lo>\d+)_(?P<opt>\d+)_(?P<hi>\d+)$`
  (acestep/engine/sa3_trt.py:104-105), smallest covering engine, wins when `want_steering`
  (sa3_trt.py:247-256, 289-294); want_steering = packs exist for sa3/<model_id> (acestep/streaming/sa3_backend.py:540-554).
  Bound as a persistent fp32 buffer (sa3_trt.py:418-444), rewritten only while active or dirty (sa3_trt.py:527-533).
  The engine is batch-1, so the adapter slices `steering[i:i+1]` per ring row (acestep/engine/sa3_adapter.py:189-195).
- Parity (docs/FAMILIES.md:111-116): cos 0.99992-0.99997 vs fp16mixed at zero steering; >= 0.99990 vs eager
  with steering; 12.88 vs 12.73 ms/step; session tick 52 ms vs 43 ms fp8.

## 6. NOT IN REPO

- The pack files themselves (only in ~/.daydream-scope/models/demon/steering_packs/sa3/medium).
- The exact discover.py version that wrote them (the header's method_source string never existed in git).
- The discovery report JSON (calibration sweep, per-block diff norms, effects mean/std, paired sign agreement).
- Per-layer vectors for non-chosen blocks (never saved), hence no same-layer cosines except percussive/warm at b15.
- Author, model checkpoint hash or revision, and run logs for the 2026-10-03 discovery run.
- Any rationale for the concept choice beyond discover.py:69-70.
