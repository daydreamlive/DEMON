# Reusing the TADA "effect per distortion" bench to score the production SA3 steering knobs

Read-only fact-finding, 2026-10-05. Nothing was modified, run on the GPU or started/stopped.

Sources (short names used below):
- RUN = `C:\_dev\projects\DEMON-tada-sa3\scripts\tada\sa3_tada_run.py` (branch ryanontheinside/feat/tada-sa3, HEAD 974ab049, 1082 lines; newer than the copy on tada-sae-sa3, 1060 lines)
- SCORE = `C:\_dev\projects\DEMON-tada-sa3\scripts\tada\sa3_tada_score.py`
- ENG = `C:\_dev\projects\DEMON-tada-sa3\acestep\engine\sa3_tada.py`
- CAA = `C:\_dev\projects\DEMON-tada-sa3\acestep\tada\caa.py`; CONC = `...\acestep\tada\concepts.py`; data in `...\acestep\tada\data\{benchmark_prompts,steering_concepts,localization}.json`
- REF = `E:\Projects\tada-replication\steer-audio` (reference code, MIT): `src\steering\eval\eval_steering_protocol.py` (PROT), `src\steering\eval\auc.py` (AUC), `editing\eval.py` (EV), `src\metrics\metrics.py` (MET), `src\steering\methods\sae\lib\configs\eval.py` (EVP), `src\steering\eval\PCI_CUTOFFS.md`
- ST = `C:\_dev\projects\DEMON\notes\family_merge\status_tada_sa3.md`; SAE = `...\status_tada_sae_sa3.md`; OV = `...\19_tada_overnight.md`; E5T = `...\e3\e5_table.md`; logs in `...\family_merge\steer_logs\`
- PROD = `C:\_dev\projects\DEMON-steer` (ryanontheinside/feat/steering-generic, HEAD 83f5b77b)

## 1. Protocol, exactly

Generation (all eager, offline, one python process per subcommand, no server):
- Model: `sa3_reference_generate.load_local_model(checkpoint_dir("medium"), model_half=True)` (RUN:93-98); default checkpoint is the served ARC `medium`; `--checkpoint medium-base --cfg 7` exists for the base control (RUN:85-90, 1061-1065).
- Render: `sa3_tada.generate` = upstream `sam.generate(prompt=list, duration=10.0, steps=8, seed=..., cfg_scale=1.0, batch_size=len(prompts))` (ENG:123-135). Sampler = the upstream default of the ARC checkpoint (status: "8-step ARC sampler", sigmas 1.000, 0.994, 0.984, 0.958, 0.891, 0.746, 0.513, 0.274; ST:123). NOT the production stream's pingpong sampler.
- Constants: `STEPS = 8`, `DURATION = 10.0`, SR 44100 (RUN:69-71). CFG 1, so every row is the conditional pass (ENG:19-21).
- Seeds: `PATCH_SEED = 222` (+k per seed), `CAA_SEED = 10`, `EVAL_SEED = 2115` (RUN:77-82). Same seed and batch size give the same initial latents row by row, which the alpha-0 vs alpha comparison relies on (ENG:126-129). One seed only; no seed repeats anywhere in the eval.
- Batch: `--batch` default 16 (RUN:1007); E5 used `--batch 25` for the 50-prompt sweeps and `--batch 20` for the 20 holdout probes (steer_logs/sa3_e5_chain.sh).
- Prompts: 100 test + 20 holdout benchmark prompts from `benchmark_prompts.json` (CONC:63-69). `--n-prompts` takes the first N (RUN:463-465); every SA3 table used the first 50 test prompts ("eval50"); calibration and E4 used the 20 holdout prompts. Example test prompt: "Upbeat indie pop track with jangly guitars and handclaps, summer road trip vibes".
- Audio saved as int16 mono npz in the reference `alpha_<a>/audios.npz` layout (RUN:101-107).

Strength sweep:
- Grid: `alphas_from_range(min, max, points)` = `points` negative values, 0, `points` positive values, `alpha_k = range * k / points` (CAA:121-130); `--points` default 15 (RUN:1026); E5 ran `--points 14` (29 strengths incl. 0).
- Range: from `calibrate` (RUN:847-891): per method/concept/site/direction, the smallest probed |alpha| whose mean holdout LPAPS reaches that direction's PCI cutoff; the largest probe if none does (`reached: false`). Probes are rendered by `probe` (RUN:828-844) on the 20 holdout prompts; E5 probed alpha 0, +-2, 4, 8, 16, 32, 64 (13 strengths) and scored them LPAPS-only (`--lpaps-only`, SCORE:203-210). Output `ranges_<eval_sub>.json`.
- Application: `steer_offline` adds `alpha * v[step][block]` (unit-norm per-step vector) to the output of `blocks[i].cross_attn` for the chosen blocks, every row, forward hooks; step counted by a pre-hook on block 0 (ENG:58-120; CAA `steer_activation` 82-96, renorm off by default). Sites: `all` = 24 blocks, `loc` = `--loc` list (3,5,6,7 in E3-E5), `ablated` = complement (RUN:453-460).

PCI (prompt-swap baseline) (RUN:472-516, 535-587):
- Prompt triple per concept, hard-coded (RUN:535-548), e.g. tempo `("a song, {p}", "fast song, {p}", "slow song, {p}")`, piano `("{p}, with instrument", "{p}, with piano", "{p}")`.
- "Strength" k in -8..8 (default every step; `--pci-ks` subset, both signs): the neutral prompt for the first 8-|k| steps, then the positive (k>0) or negative (k<0) prompt for the last |k| steps.
- Mechanism: NOT an embedding lerp and NOT the prompt-blend knob. It is conditioning substitution: a 1-step render of the swap prompt records each block's cross-attention `context` kwarg (K/V input); the neutral render then has that `context` replaced from step `switch` on (`_SwitchPatcher`), at all 24 blocks (PCI-all) or only the loc blocks (PCI-loc). Text enters SA3 only through that context (ST:96-99), so PCI-all at k=8 is the full prompt swap.

LPAPS (PROT:247-288, EV:48-77):
- Reference `get_lpaps`: LPAPS with `net="clap"`, LAION-CLAP music checkpoint `music_audioset_epoch_15_esc_90.14.pt` (HTSAT-base, non-fusion; `E:\Projects\tada-replication\steer-audio\res\clap\pretrained\`, 2.35 GB), windowed at 10 s, overlap 0.1, mean over windows.
- Reference audio: the alpha=0 render of the same prompt (same seed, same row) in the same sweep directory; per-clip pairs, mean and std over prompts per alpha.

Cutoff (AUC:112-136, SCORE:237-253):
- cutoff = min(max mean LPAPS of PCI-all, max mean LPAPS of PCI-loc), per concept and direction. The max sits at the full-swap endpoint, so SCORE `cutoff` scores only k in {0, min, max} (`lpaps_endpoints.csv`) and `_pci_cutoff` falls back to those (SCORE:214-253).
- SA3 eval50 cutoffs pos/neg: piano 2.44/2.41, mood 2.28/2.41, tempo 3.21/3.40, vocal_gender 3.10/2.78, vocal_style 3.17/3.09, guitar_electronic 1.85/1.90, violin 2.70/2.41, rock_genre 2.55/2.49, electronic_music 2.40/2.27 (ST:47; ACE reference 3.3 to 4.5, PCI_CUTOFFS.md).

AUC (AUC:87-109, 145-198, 201-232):
- Per direction (pos = alpha >= 0, neg = alpha <= 0): merge mean LPAPS and mean alignment per alpha, keep alphas with LPAPS <= cutoff, append one interpolated point exactly at LPAPS = cutoff (needs a point above the cutoff to bracket it), delta = sign * (alignment - alignment at alpha 0), path integral (trapezoid) of delta over LPAPS in order of increasing |alpha| (backtracking subtracts area). So the x-axis is LPAPS from 0 to the cutoff; the result is "alignment gain per unit distortion" area.
- The tables report the mean of pos and neg AUC; CSM (smoothness) is computed beside it (AUC:239-).

MuQ / CLAP alignment (PROT:121-244, MET:83-123, EV:80-, EVP:1-38):
- One anchor per concept, used for BOTH directions (negative direction scored as a decrease; SCORE calls `protocol(str(d), concept, skip_aesthetics=...)` without `negative_eval_prompt`, SCORE:211). Anchors (EVP): piano "a piano song" / MuQ "This is a music of a piano song"; mood "a cheerful track"; tempo "a fast track"; vocal_gender "This is music with female vocal singing"; vocal_style "This is music with rap vocal"; guitar_electronic "a song with acoustic guitar"; violin "a song with violin"; rock_genre "a jazz song"; electronic_music "a classical song" (MuQ versions prefixed "This is a music of ").
- MuQ-T: `MuQMuLan.from_pretrained("OpenMuQ/MuQ-MuLan-large")`, audio resampled 44.1 kHz -> 24 kHz, cosine to the text embedding (MET:83-123). Cached at `D:\huggingface_cache\hub\models--OpenMuQ--MuQ-MuLan-large` (chains export `HF_HOME=D:/huggingface_cache`).
- CLAP (protocol): reference `get_clap` with the same music checkpoint, 10 s windows (EV:80-). CLAP (patch/localisation scoring only): `laion_clap.CLAP_Module(enable_fusion=True)` default checkpoint, template "This is a music of {p}" (SCORE:66-77, 109-143).
- Aesthetics: audiobox_aesthetics (PQ/CE/CU/PC); E5 ran `--skip-aesthetics`.
- SCORE memoizes the reference's per-strength model constructors (byte-identical outputs; SCORE:162-190).
- Env: scoring runs in `E:\Projects\tada-replication\evalenv` against the REF clone (bare-package shim, SCORE:47-63); generation in the DEMON venv.

Holdout calibration: the 20 holdout prompts render the probe grid; only LPAPS is scored; `calibrate` picks the range per direction (above). The same 20 prompts were also the E4 eval set (E4 deviates: calibration and eval on the same prompts).

Run order (E5, steer_logs/sa3_e5_chain.sh): probe (holdout 20) -> protocol --lpaps-only -> calibrate -> sweep (50 prompts, 14/side) -> protocol --skip-aesthetics (two scorers in parallel) -> auc.

## 2. How the bench injects a direction, and the production packs

- The bench does NOT use the DEMON-steer knob path (`steer_<pack>`) nor the manual slots (`man_src_/man_layer_/man_step_/man_alpha_`, PROD `acestep/steering/controller.py:29-32`). It installs its own offline forward hooks: `sa3_tada.steer_offline` on `blocks[i].cross_attn` (ENG:58-120). Vectors come from `.pt` files under `--vec-dir` (`{"vectors": {step: {block: [1536]}}}`, RUN:664-679), not safetensors packs. (The tada branch also added a production-path hook point `cross_attn_output` plus a TRT engine `sa3_m_dit_steerxa_l1_646_646` for the demo, ST:5-8; steered TRT vs eager hooks min cos 0.999863. The bench does not use it.)
- Hook site on the tada branches: the output of each block's cross-attention module (`HOOK_CROSS_ATTN_OUTPUT = "cross_attn_output"`, tada-branch `acestep/steering/layout.py`); E3-E5 blocks 3,5,6,7 (or all 24). Vectors per step (8) and per block.
- Production site (PROD): `HOOK_POST_BLOCK_RESIDUAL` = block output residual (PROD `acestep/steering/layout.py:28-30`); eager delivery = forward hook on each trunk block adding `knob * magnitude * policy_weight * v` to every token of the block output (PROD `acestep/engine/stream.py:1848-1925`, add at ~1913); TRT delivery = ONNX surgery adding `steering[1,24,1536]` row b to block b's residual Add output (PROD `acestep/engine/trt/sa3_steering_onnx.py:1-20, 94-151`).
- Same tensor? NO. In the vendored `stable_audio_3/models/transformer.py` (rev 960da1f, `C:\Users\ryanf\.daydream-scope\models\demon\sa3\vendor\stable-audio-3`), pre-norm branch lines ~1050-1067: `x = x + self_attn(...)`; `x = x + cross_attn_scale(cross_attn(cross_attend_norm(x), context))` (cross_attn_scale is Identity on medium); conformer (none); local conditioning; `x = x + ff_scale(ff(ff_norm(x)))`; `return x`. cross_attn_output(b) enters the residual mid-block, BEFORE block b's FF and local conditioning; post_block_residual(b) is AFTER the FF (the input of block b+1). Adding v at cross_attn_output(b) = adding v to the residual pre-FF; the two differ by FF(b)'s (nonlinear) response. Both add to every token (incl. the 64 memory tokens); they differ also in estimation: production `discover.py` means over ALL tokens incl. memory tokens, pooled over steps (one [1536] vector), rendered through StreamPipeline + pingpong at 54 s; TADA E3+ means over audio tokens only, per step, ARC `sam.generate` at 10 s.
- Production packs on disk: `C:\Users\ryanf\.daydream-scope\models\demon\steering_packs\sa3\medium\{bright,density,percussive,rough,warm}.safetensors`, each `vector [1536]`, `hook post_block_residual`, one block: bright b23 (norm 18.56, magnitude 1.856), density b1 (12.90/1.290), percussive b15 (40.15/4.015), rough b2 (15.73/1.573), warm b15 (29.52/2.952); policy range 0..1 (every step); magnitude = norm/10 (knob 10 = one raw mean difference). Provenance: 32 pairs, 8 patch pairs, pingpong, 8 steps, 54 s, seed0 1000.
- Can RUN load them unchanged? NO. Needed shim, about 50 to 60 lines, no engine change:
  1. `--method pack --pack <file>` in `_load_vectors` (RUN:664-679): read the safetensors (the tada branch's own `acestep.steering.packs.load_pack` accepts format-1 packs; it is a superset of PROD's), return `{s: {pack.block: vector * policy_weight[s]} for s in range(8)}` (~12 lines).
  2. A block-output variant of `steer_offline` (hook `sa3_blocks(sam)[b]` instead of `cross_attn_modules(sam)[b]`, handle tuple outputs like PROD stream.py) or a `modules=` parameter (~15 lines); `_site_blocks` site `pack` = `[pack.block]` (~3 lines).
  3. PCI triples for the five concepts in `_pci_triple` (RUN:535-548), e.g. neutral `"{p}"`, pos `"{p}, bright, crisp, sparkling highs..."` from PROD `scripts/steering/discover.py:70-103` (~5 lines).
  4. SCORE: the reference anchor table has only the 9 TADA concepts (EVP), so `protocol(d, "bright")` raises KeyError; add a local anchor dict and pass PROT's existing `eval_prompt=` override (SCORE:211; ~10 lines). The `_eval_dirs` name split (`method_site_concept`, SCORE:150) works for single-word concept names.
  5. Optional: PCI-loc at the pack's block (`--loc <block>`) for the min(PCI-all, PCI-loc) cutoff, or cutoff from PCI-all alone (say so).
  Calibration makes the pack `magnitude` irrelevant for the AUC (alpha is re-found against the PCI cutoff on the unit vector); map back to knob units as alpha / magnitude.

## 3. Cost on the 5090 (measured)

Measured rates:
- Generation: "measured SA3 eager batch-16 generation 0.092 s/gen" (ST:20; restated 0.092 s/clip alone, ST:44). E5 logs: 14 sweeps x 1450 clips = 20,300 clips 18:00:20 -> 18:37:05 = 0.109 s/clip at batch 25 incl. save (steer_logs/e5_sweep.log); holdout probes 4,420 clips in ~7 min (~0.1 s/clip).
- PCI render: "PCI generation 6.2 min per concept-site" at 100 prompts x 17 strengths (ST:41) = ~0.22 s/clip (record passes + contention).
- Scoring with aesthetics: "reference protocol scoring 0.45 s/clip beside one generator (2.25 min per 300-clip dir)" (ST:44).
- Scoring without aesthetics (E5 logs): e5_score_b 3 dirs = 4,350 clips 18:48 -> 19:02 = 0.19 s/clip; e3all 7 dirs = 10,150 clips 19:02 -> 19:45 = 0.25 s/clip; two scorers ran in parallel. LPAPS-only: 4,420 clips 17:55 -> 17:59 = ~0.05 s/clip.
- VRAM: "at most one scorer beside one generator" while generating (32 GB paging otherwise; ST:42); two scorers once generation ends.
- Whole E5 (7 concepts x loc+all + 3 probe-direction sweeps + 7 extra scored dirs): "GPU 2.0 h" (ST:152).

Clips per concept at the full SA3 protocol (50 prompts, 14/side, one site, PCI from scratch):
- calibration probes 20 x 13 = 260; sweep 50 x 29 = 1,450; PCI-all full 50 x 17 = 850 (+1-step record renders); PCI-loc endpoints 50 x 3 = 150. Total ~2,710 renders; ~2,300 fully scored + ~410 LPAPS-only.
- GPU minutes: generation ~5.5 min (sweep+probes ~3, PCI ~3.4); scoring ~8.5 min without aesthetics (2,300 x 0.2 + LPAPS), ~17 min with aesthetics. Serial ~14-15 GPU-min per concept (no aesthetics), ~10-11 overlapped (one generator + one scorer), ~25 with aesthetics. Each additional site variant (e.g. loc + all) adds ~1,710 renders + 1,450 scored = ~8 min. Paper scale (100 prompts, 15/side) roughly doubles.

1 GPU hour, 5 concepts (12 min each, one site each = the pack's block):
- Full 50 x 14/side + PCI from scratch: ~50-55 min overlapped. Fits with no slack (model loads ~20 s per process, no rerun room).
- Recommended: 50 prompts, 7/side steering (sweep 750) + PCI-all at `--pci-ks 2 4 6 8` (9 strengths, 450) + PCI-loc endpoints + 13 holdout probes: ~1,610 renders and ~1,200 scored per concept, ~7.5 min serial, ~5-6 overlapped -> ~30-40 min for 5 concepts, room for one rerun.
- Alternative (E4 scale): 20 holdout prompts x 14/side with PCI-all re-rendered on the same 20; E4 did 3 concepts (two sweep sets each + PCI-all + probes) in 17:16 -> 17:35 (ST:139-146), ~6 min per concept.
- Safe reductions:
  - Fewer strengths: safe; the AUC is integrated on the LPAPS axis, so the grid only changes resolution ("strength range scaled 1/g; AUC is on the LPAPS axis so only grid resolution changes", ST:69). Keep at least one point above the cutoff per direction (the boundary interpolation needs a bracket, AUC:181-188) and calibrate first so the grid ends at the cutoff. PCI switch lengths reduce the same way (`--pci-ks`, used in the base test, ST:134).
  - Fewer prompts: costly in variance. "20 prompts only, so per-concept differences of about 0.1 are within noise (no seeds repeated)" (ST:145). Keep 50 for ratios; small PCI denominators blow ratios up regardless (mood PCI 0.009 -> ratio 1.96, ST:152).
  - Shorter clips: not safe and saves little. LPAPS/CLAP use 10 s windows (EV:40-45), the SA3 cutoffs and all vectors are measured at 10 s, and scoring (not generation) is the bottleneck.
  - Drop aesthetics (E5 did) and, if wanted, CLAP (near-blind on SA3, see 7); a MuQ-only mode needs a small change (PROT `only_muqt` requires an existing clap.csv).

## 4. Concepts benchmarked and measured numbers (verbatim)

From `notes\family_merge\e3\e5_table.md` (E5: ARC, 50 benchmark prompts, seed 2115, blocks 3,5,6,7, 14 strengths/side, renorm off, MuQ AUC avg over directions):

```
| Concept | PCI-all | e5 loc | e5 loc/PCI | e5 all | e5 all/PCI | e3 loc/PCI | e3 all/PCI | e5p loc/PCI |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| tempo | 0.060 | 0.062 | 1.03 | 0.069 | 1.15 | 0.92 | 0.81 | 0.72 |
| piano | 0.035 | 0.017 | 0.47 | 0.015 | 0.44 | 0.39 | 0.55 | 0.51 |
| mood | 0.009 | 0.018 | 1.96 | 0.019 | 2.14 | 0.76 | 0.27 | 1.49 |
| violin | 0.036 | 0.009 | 0.24 | 0.016 | 0.44 | 0.30 | 0.35 | n/a |
| guitar_electronic | 0.024 | 0.006 | 0.27 | 0.010 | 0.43 | 0.21 | 0.24 | n/a |
| rock_genre | 0.053 | 0.024 | 0.45 | 0.025 | 0.46 | 0.43 | 0.41 | n/a |
| electronic_music | 0.024 | 0.012 | 0.51 | 0.016 | 0.66 | 0.47 | 0.77 | n/a |
| mean (7) | 0.034 | 0.021 | 0.61 | 0.024 | 0.71 | 0.52 | 0.53 | n/a |
3-concept (tempo, piano, mood) ratio of means: e5 loc 0.93, e5p loc 0.72, e3 loc 0.73
loc vs all (e5): -13%  (paper ACE +39%)
CLAP mean (7): PCI-all 0.009 e5 loc 0.006 e5 all 0.006
```

From `status_tada_sa3.md`:
- ST:53-59 (first eval50, old vectors, AUC MuQ pos/neg/avg | CLAP): "CAA loc (3,6,7): 0.006/0.015/0.011 | 0.004/-0.001/0.001"; "CAA all: 0.012/0.017/0.014 | 0.007/-0.001/0.003"; "CAA ablated (4): 0.008/0.027/0.017 | 0.005/-0.004/0.001"; "AUSteer loc: 0.006/0.012/0.009 | 0.001/-0.002/0.000 ; AUSteer all: 0.009/0.011/0.010 | 0.005/-0.003/0.001"; "N2(c) CAA at K/V site loc: -0.007/0.003/-0.002 | 0.002/-0.004/-0.001 (worse; recorded)".
- ST:130 (E3 interim): "PCI-all MuQ/CLAP AUC 0.036/0.013 (9 concepts); CAA loc (3,5,6,7, corrected vectors) 0.018/0.003, ratio CAA-loc/PCI-all 0.49 MuQ, 0.22 CLAP (paper ACE Table 1: 1.24 / 1.33). Per concept CAA-loc/PCI MuQ: tempo 0.92, vocal_style 0.82, mood 0.76, electronic 0.47, rock 0.43, piano 0.39, violin 0.30, vocal_gender 0.28, guitar 0.21."
- ST:136 (base checkpoint, 50 steps cfg 7): "tempo 0.040 / 0.037 = 1.07 (ARC 0.92); piano -0.027 / 0.025 = -1.08 (ARC 0.39; CAA moves AGAINST piano in both directions on base); mood 0.001 / 0.016 = 0.05 (ARC 0.76)."
- ST:141-144 (E4, 20 holdout prompts, large-N class-mean vectors | matched 50-pair control | 50-pair on eval_e3): "tempo 0.081 / 0.070 = 1.15 | 0.87 | 0.92"; "piano 0.014 / 0.035 = 0.41 | 0.15 | 0.39"; "mood  0.008 / 0.011 = 0.66 | 0.61 | 0.76"; "3-concept ratio of means 0.88 (caa_e3 matched 0.63); paper ACE Table 1 1.24."
- ST:73-88 E1 guidance table (tempo/piano/mood, mean MuQ 0.013 to 0.024 for g in 1,3,5,7; renorm x4 tempo 0.069 loc / 0.075 all).
- Paper reference (ST:108): "ACE avg 9 concepts, MuQ / CLAP: PCI-all 0.084 / 0.049, PCI-loc 0.068 / 0.038, CAA loc 0.104 / 0.065, CAA all 0.075 / 0.046, AUSteer loc 0.096 / 0.063, AUSteer all 0.081 / 0.058 (Table 1 p.7, Table 25 p.40). Ratios CAA loc/PCI-all 1.24 (MuQ) 1.33 (CLAP)".
- vocal_gender and vocal_style are excluded from the 7-concept tables ("SA3 medium generates no vocals", ST:131).

Overlap with bright / density / percussive / rough / warm: NONE of the 9 TADA steering concepts (piano, mood, tempo, vocal_gender, vocal_style, guitar_electronic, violin, rock_genre, electronic_music; CONC:28-31) nor the 10 localisation concepts (female, male, fast, slow, happy, sad, violin, flute, maracas, reggae; RUN:76-77) is one of the five. Nearest neighbours only: tempo (fast/slow) ~ density and percussive (onset-rate family); maracas ~ percussive; guitar_electronic (acoustic vs electric guitar) ~ rough/distorted; electronic_music (classical vs electronic) loosely ~ bright. ST:109 refutes a shared "energy/density" direction across tempo/piano/mood for the old vectors.

## 5. Caches a direction-estimation step could reuse

| Path | Size | Content | Site / blocks / steps | Captions / labels |
|---|---|---|---|---|
| `D:\tada-replication\e5_means\means.npy` (+ `meta.json`) | 3.1 GB (3,256,418,432 B) | float16 `[5521, 8, 24, 1536]` per-step AUDIO-token means (64 memory tokens + padding excluded) | `cross_attn_output`, all 24 blocks, 8 steps (sigmas 1.0 ... 0.274) | 5521 MusicCaps captions, `caption_index` = shuffled order (seed 42); seed 1000 + batch index, batch 16, 10 s, ARC medium cfg 1 |
| `D:\tada-replication\sae_cache\sa3\block_{3,5,6,7}\shard_*.npy/.json` + `config.json` | 89 GB (23 GB per block, 692 files each) | float16 per-token `[128, 172, 1536]` shards (16 prompts x 8 steps), json gives prompt/step/sigma per row | `cross_attn_output`, blocks 3,5,6,7, 8 steps, 172 audio tokens | same 5521 captions / order; "41960 train / 2208 held-out samples per block" (SAE:25) |
| `E:\Projects\tada-replication\sa3\caa\<concept>.acts.pt` (and caa_e3) | 540 MB each dir | per-prompt `[prompt, step, block, hidden]` cross-attn output means, 50 pairs x 2 poles | `cross_attn_output`, 24 blocks, 8 steps (caa = all rows; caa_e3 = audio rows) | TADA prompt pairs, 9 concepts |
| `E:\...\sa3\caa_e5\` (+ `e5_meta.json`), `caa_e4\`, `caa_e5p\` | 8.3 MB / 0.6 MB / 0.6 MB | large-N keyword-class CAA vectors and probe directions | as above | 7 TADA concepts only |
| `E:\...\sa3\oracle\` | 2.5 GB | per-pair per-token diffs `[50, 8, 3, 238, 1536]` | blocks 3,6,7 | tempo/piano/mood |
| `D:\tada-replication\sa3\patch_resid` | 41 GB | E2 resid patch audio (unscored) | n/a | n/a |

Label source: there is no label file; labels are regexes over the MusicCaps CSV `E:\Projects\tada-replication\data\musiccaps-public.csv` (5521 rows; columns include `caption` and `aspect_list`; `audioset_positive_labels` are AudioSet ids). E4/E5 matched the `caption` column only (`sa3_e5_vectors.py` PRESENCE/TWO_SIDED regexes, lines 27-57; `sa3_e4_probe.py` LABELS 29-35).

IMPORTANT for production packs: both caches are at `cross_attn_output`, not `post_block_residual`. For the production site a recapture is needed: `sa3_e5_capture.py` with `sa3_tada.sa3_blocks(sam)` instead of `cross_attn_modules(sam)` (one-line change at line 42); measured "done 5521 prompts 525s" (steer_logs/sae_e5_capture.log), ~9 GPU-min, ~3.1 GB. Note it renders with the ARC `sam.generate` sampler at 10 s, while the production packs came from pingpong at 54 s.

Keyword counts over the 5521 captions (word-boundary regex, lower-cased; caption / aspect_list / either):

| word | caption | aspect | either |
|---|---|---|---|
| bright* | 78 | 70 | 107 |
| warm* | 53 | 38 | 59 |
| dark* | 37 | 45 | 53 |
| dense/density | 9 | 14 | 18 |
| sparse* | 4 | 1 | 4 |
| percussive | 314 | 207 | 366 |
| percussion* | 828 | 918 | 1096 |
| rough* | 6 | 2 | 6 |
| distort* | 281 | 264 | 323 |
| gritty/grit | 9 | 5 | 10 |
| smooth* | 114 | 118 | 124 |
| mellow* | 374 | 350 | 422 |
| (extra) harsh 45, thin 40, muddy 45, fuzz 45, overdriv 48, busy 24, minimal 144, simple 630, complex 105, punchy 500, soft 388, clean 75, crisp 17, airy 19, drum* 1821 (either) | | | |

Pole-exclusive class sizes (caption OR aspect_list):
- bright vs dark: 105 vs 51 (both 2); bright vs dark|muffled|dull|muddy: 100 vs 260.
- warm vs harsh|thin: 59 vs 82; warm vs cold|harsh|thin|bright: 51 vs 184.
- rough group (distort|gritty|rough|fuzz|overdriv) vs smooth|clean|mellow: 394 vs 565 ("rough" itself: 6, unusable alone).
- density: dense|busy vs sparse|minimal: 41 vs 147; dense|busy|full|complex vs sparse|minimal|simple: 238 vs 726 (literal dense 18 / sparse 4: unusable alone).
- percussive: "percussive" 366 vs captions with no percuss*/drum*/beat*/groove* word 2594; any percussion|drum 2769 vs none 2752.
- Usable at E5 class sizes (E5 used 173 to 751 per class): percussive (yes), rough via the distortion group (yes), bright (borderline, ~51 to 105 per class balanced), warm (borderline, ~59), density only with the broad dense/busy/full/complex vs sparse/minimal/simple proxy. Caveat (SAE:40): "keyword labels co-vary with other caption content (genre, instrumentation)". MusicCaps words like "muffled", "noisy", "low quality" mostly describe the recording, so dark/rough classes risk picking recording quality.

## 6. The linear-probe script

- Path: `C:\_dev\projects\DEMON-tada-sae-sa3\scripts\tada\sa3_e4_probe.py` (167 lines, committed 40ece042 on ryanontheinside/feat/tada-sae-sa3).
- Fits, per concept x block x step: (a) mean level: L2 logistic regression on the token-averaged activation, 5-fold CV by caption, with a shuffled-label control; (b) token level: same probe on ~20k individual tokens per class, folds grouped by caption; (c) effect size along the mean difference (class-mean gap of the projection / pooled within-class std); (d) cosine of the mean-difference direction to the `caa_e3` vector (docstring lines 1-16, `logreg_cv` line 53).
- Inputs (args 86-92): `--cache D:/tada-replication/sae_cache/sa3`, `--captions E:/Projects/tada-replication/data/musiccaps-public.csv`, `--caa E:/Projects/tada-replication/sa3/caa_e3`, `--blocks 3 5 6 7`, `--cap 1000` captions per class, `--tokens 20000`. Labels: regexes for piano / tempo / mood only (LABELS 29-35).
- Output: `--out D:/tada-replication/e4_probe.json` (also `e4_probe_b7.json` exists); table copied into SAE:42-139 (mean acc 0.81 to 0.98, shuffled ~0.5).
- Related: `sa3_e5_vectors.py:60-78` `probe_direction` (L-BFGS L2 logistic on standardized means, returns unit weight direction) produced `caa_e5p`; `sa3_e5_vectors.py` also builds the large-N class-mean CAA (`caa_e5`) from `e5_means`.

## 7. Lessons that bear on measuring timbre knobs (verbatim, with paths)

- Estimator at data scale beats 50 pairs: "the large-N estimator lifts every concept over the 50-pair vectors on the same prompts (tempo +0.28, piano +0.26, mood +0.05 in ratio); tempo passes PCI; piano and mood stay below 1." (ST:145). "E4: full-cache CAA vectors raise matched ratios 0.63 -> 0.88 ... Estimator confirmed as the SA3 gap." (OV:239).
- Direction exists linearly, CAA only partly aligned: "mean-level accuracy >= 0.85 for all three concepts (shuffled ~0.5): a linear mean-level direction exists at the site; the CAA vectors are only partly aligned with it (cosines well below 1)." (SAE:144). "alignment tracks success" (OV:233).
- Localisation no gain: "Localised is NOT better than all-blocks or ablated on SA3" (ST:59); "loc vs all (e5): -13%  (paper ACE +39%)" (E5T:12). Localisation itself is robust only at 6, 7: "Blocks 6 and 7 are robust; 1, 3, 5, 11 sit at 0.10 +- 0.01." (ST:118).
- CLAP vs MuQ: "CLAP barely separates the real positive and negative prompts on SA3 (0.008 to 0.025); MuQ does (tempo +0.104, piano +0.065, mood +0.024)." (OV:131-133). "Metric: eq. 2 impact and the AUC alignment term use MuQ as primary on SA3; CLAP is reported in a secondary column, never used for a decision." (OV:146-147). MuQ can also invert: "for "fast" the MuQ clean reference scores BELOW the corrupted one (all 0.171 < none 0.235 on "fast song"), so its eq. 2 ratio has a negative denominator" (ST:119).
- The real prompt is the bar: "The real prompt itself (PCI) scores MuQ AUC only 0.032 averaged over tempo, piano, mood ... The paper's 0.334 is not the comparison; CAA relative to PCI on the same model is." (OV:134-136); "No SA3 number is ever compared with 0.334." (OV:173).
- LPAPS cutoff behaviour: "PCI scored at its full-swap endpoints only (the cutoff; reference PCI_CUTOFFS.md: max LPAPS is at the endpoint)" (ST:45); SA3 cutoffs 1.85 to 3.40 "(reference ACE cutoffs 3.3 to 4.5)" (ST:47); "Every sweep reaches its PCI cutoff (LPAPS max > cutoff), so this is not a range artifact." (ST:70); "the PCI cutoff is SA3's own and admits 4 to 8 strengths" (OV:139-140); "AUC is on the LPAPS axis so only grid resolution changes" (ST:69); a weak site can fail to reach it: "K/V site with the real-token mask never reached the cutoff by 16 (LPAPS 1.6 vs cutoff 2.3 to 3.4 ...), so its probe is extended to 256" (ST:123).
- Small denominators and noise: "20 prompts only, so per-concept differences of about 0.1 are within noise (no seeds repeated)." (ST:145); "mood 1.96 (PCI 0.009, tiny denominator)" (ST:152).
- Fixed vectors vs state: "A correct same-pair oracle recovers 0.16 of the concept on eq. 2 (MuQ), exact output substitution 0.30, the K/V patch 0.47 to 0.65: fixed vectors recover about a third of the patch effect, the rest is state-dependent." (OV:128-131).
- Guidance / renorm / steps do not rescue it: "g3/g5/g7 track g1 within 0.005 (guidance on the steer just rescales the same direction; in LPAPS-normalized AUC it buys nothing). 30 steps at g1 is lower (0.013)" (ST:90); base checkpoint: "distillation is not why CAA underperforms PCI on SA3" (ST:136).
- Token set: "The 64 learned memory tokens are inside the token mean of CAA and AUSteer (about a quarter of the mean)." (OV:137-138) (production `discover.py` still means over all tokens).
- Ear / live check limits: "Live streams are not deterministic across sessions (control: unset vs unset 0/27 identical slices), so zero no-op is proven offline" (ST:61); "Note on the quick piano render: its zero file drew batch-1 noise unlike the steered files, so the heard "tempo change" can be a noise/seed difference" (ST:109).
- Production proxies for two of the five knobs are weak: "rough b2 n15.7 (flatness proxy weak: agree -0.25); density b1 n12.9 (onset proxy weak: agree -0.56)" and "proxies for rough/density do not track the concept" (`notes\family_merge\status_steering.md`, Step 4/5/6 section).
- Practical: "VRAM hit 32 GB (generator + two scorers) and paged ... Rule from here: at most one scorer beside one generator." (ST:42).

Implication for the five timbre knobs (inference, not measured): before spending GPU on sweeps, render the PCI swap pair (`swap`, 2 x 50 clips) per concept and check that the chosen MuQ anchor actually separates the real positive and negative prompts on SA3, as the audit did for tempo/piano/mood; if MuQ does not separate "bright" vs "dark" the AUC cannot measure the knob, and the production DSP proxies (centroid, low/high dB, perc ratio) would be the better alignment column for bright/warm/percussive.
