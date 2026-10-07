# Re-estimated SA3 steering directions (P1, P2, P3): scripts status (2026-10-05)

Worktree C:\_dev\projects\DEMON-steer-bench, branch ryanontheinside/spike/steer-bench.
Commits d119cd8f + 57014efd (scripts/steering_bench/ only, unpushed). Nothing run on a GPU.

## Files (scripts/steering_bench/)

- capture_resid.py: E5 capture (sa3_e5_capture.py) re-hooked on the BLOCK OUTPUT (sa3_blocks(sam)[b], every one
  of 24 blocks, 8 steps), audio tokens only (64 memory tokens + padding excluded, mask class copied from
  tada-sae-sa3 sa3_tada_tokens.py), float16 means [N,8,24,1536] + meta.json. MusicCaps mode: seed-42 shuffle
  (verified identical to D:/tada-replication/e5_means caption_index), seed 1000 + batch. `--self-label`:
  benchmark TEST prompts cycled (holdout 20 untouched), seed 5000 + batch, audio saved as audio/batch_*.npz.
- build_directions.py: class-mean (MusicCaps regexes in concepts.py) or quartile mass-mean (self-label,
  descriptors from proxies.py) per step per block; unit vector, raw norm, pooled within-class projection std
  (`proj_std`, for std-normalised alpha) and std over all rows. Writes vec_<tag>/ (24 blocks), vec_<tag>_bNN/
  (24 dirs), vec_<tag>_prod/, packs_<tag>/sa3/medium/*.safetensors (format 1, step-mean at the production block,
  norm = raw L2, magnitude = 0.1 x norm, provenance incl. method, n per class, capture, date, raw norm per block),
  packs_<tag>_bNN/ (format 1 per block, used by the sweep), `--per-step-pack` packs_<tag>_perstep/ (format 2,
  vector [8,1536]; production does not read format 2), <tag>_summary.json (effect size per block, alpha scale
  per block, cosine to the current production pack).
- concepts.py (regexes, descriptor signs, production blocks), proxies.py (verbatim copy of DEMON-steer 83f5b77b),
  desc_pool.py (parallel descriptors: 1.5 CPU-s per 10 s clip serial), score_descriptors.py (descriptor deltas vs
  alpha 0 + LPAPS -> one summary row per sweep dir), sweep_blocks.sh, chain_directions.sh.

MusicCaps classes with these regexes (caption + aspect_list, exclusive poles), before balancing to min(pos,neg,1000):
bright 100/260, warm 51/185, rough 393/572, density (pos = sparse) 726/238, percussive 2759/2594.

## Smoke (CPU, run)

Synthetic capture [64,8,24,1536] + fake captions csv with planted directions, and a self-label capture with sine +
click-train audio: both build modes end to end; planted direction recovered (cos 0.60 to 0.79, the expected value
for 32/32 rows at that SNR); planted block found; density positive class = lower onset rate (1.0 vs 7.0 /s).
Packs re-read raw: one F32 tensor `vector` [1536], metadata key set identical to the production bright pack
(knob_provenance.md 2.7), family/checkpoint/hook/hidden_size/method/policy equal; bench load_pack OK; the
other agent's `_pack_vectors` (--method pack) reads a per-block pack as {step: {5: unit}} at post_block_residual.
score_descriptors.py run on a fake sweep dir. capture_resid.py: py_compile + --help only (needs the model).

## CLI (Git Bash; P = DEMON venv python, PYTHONPATH = the worktree)

    cd /c/_dev/projects/DEMON-steer-bench; export PYTHONPATH=$PWD PYTHONUTF8=1 CUDA_VISIBLE_DEVICES=0
    P=/c/_dev/projects/DEMON/.venv/Scripts/python.exe; O=E:/Projects/steering-bench/2026-10-05/sa3
    # 1. MusicCaps recapture, 5521 captions                      ~9 GPU-min (E5: 525 s), 3.3 GB
    $P scripts/steering_bench/capture_resid.py --out D:/steer-bench/resid_mc
    # 2. self-label capture, 800 renders + audio                 ~1.5-2 GPU-min, 0.5 GB means + ~0.7 GB audio
    $P scripts/steering_bench/capture_resid.py --self-label --n 800 --out D:/steer-bench/resid_self
    # 3. build directions (CPU): mc ~3 min; self ~3 min + descriptors 800 x 1.5 CPU-s / 30 workers ~1 min
    $P scripts/steering_bench/build_directions.py --capture D:/steer-bench/resid_mc --out $O --per-step-pack
    $P scripts/steering_bench/build_directions.py --capture D:/steer-bench/resid_self --self-label --out $O --per-step-pack
    #    (1-3 in one go: GPU=0 CAP=D:/steer-bench OUT=$O scripts/steering_bench/chain_directions.sh)
    # 4. per-block sweep, one concept, one GPU (K x proj_std alpha, 20 holdout, --points 2, LPAPS + descriptors)
    OUT=$O K=4 scripts/steering_bench/sweep_blocks.sh bright 0 resid_mc $(seq 0 23)
    #    -> $O/blocks_resid_mc_bright/summary.csv, one row per block + the production pack row (_prod)

Sweep cost per concept, 24 blocks: renders 24 x 100 clips x 0.1 s = 4 min, but one python process per block
(the run script takes one --pack per call): ~20 s model load x 25 = ~8 min; LPAPS ~2 min; descriptors ~3 CPU-min.
About 14 GPU-min per concept, 70 for five; a block subset (e.g. 8 blocks) is about a third. Plan section 3 said 6 min
per knob; the gap is the per-process model load.

## Unresolved

- The sweep depends on the other agent's `--method pack` shim (UNCOMMITTED in the worktree when I finished;
  sweep_blocks.sh checks for it and FLAGs). It steers post_block_residual only via packs, i.e. ONE step-mean
  vector at ONE block. `--vec-dir` files (per-step, `all`) are still applied at cross_attn_output by
  sa3_tada_run.py: per-step (P3) and all-blocks (P2 alternative) at the production site need `_hook_kw()` to
  honour a `--hook post_block_residual` flag for --vec-dir (one line). The .pt files already carry
  `"hook": "post_block_residual"`.
- Alpha grid is K x pooled proj std per block (no calibration probe, no PCI cutoff); K=4 is a guess. Check the
  LPAPS column of the first block before running the rest.
- The other agent also added scripts/tada/sa3_descriptor_score.py; overlaps score_descriptors.py (pick one).
- Estimation conditions differ from production (10 s ARC sampler, audio tokens only vs 54 s pingpong, all tokens);
  recorded in every pack's provenance caveat.
- GPU_LOCK / stopping the demo backend before GPU runs is not scripted.
