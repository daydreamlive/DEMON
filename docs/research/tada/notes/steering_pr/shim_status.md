# Steering-bench shim: status (2026-10-05)

Worktree `C:\_dev\projects\DEMON-steer-bench`, branch `ryanontheinside/spike/steer-bench` (cut from tada-sa3 974ab049).
Commits `33750a9c` (shim) and `f75c0563` (--hook, scorer unification). Not pushed. CPU-only work; nothing ran on the GPU.

## Files changed

- `acestep/engine/sa3_tada.py`: `steer_offline(..., hook=)`; `post_block_residual` hooks `sa3_blocks(sam)[b]` (block output,
  tuple outputs steered on element 0 as DEMON-steer stream.py:1900-1925); default stays `cross_attn_output`.
- `scripts/tada/sa3_tada_run.py`: `--method pack --pack <file | dir of <concept>.safetensors>`; `_pack_vectors` checks the
  header has hook/block/policy/norm/magnitude, loads with the branch's `load_pack`, returns
  `{step: {block: unit v * policy_weight(step)}}` for 8 steps; site `pack` = pack's target blocks (`--sites` defaults to
  `pack` under `--method pack`); the steer hook is the pack's hook; `sweep.json` records the pack (path, hook, magnitude);
  `calibrate` knows method/site `pack`; `PACK_DESCRIPTORS` (verbatim discover.py:71-103) give the PCI triples
  `("{p}", "{p}, <pos>", "{p}, <neg>")`. Dir names `pack_pack_<concept>` parse in `_eval_dirs`.
- `scripts/tada/sa3_tada_score.py`: `PACK_ANCHORS` (bright "a bright track", warm "a warm track", percussive "a percussive
  track", rough "a rough, gritty track", density "a sparse, minimal track"), passed as `eval_prompt="This is a music of ..."`.
  Note: the reference `main` reads `CONCEPT_TO_EVAL_PROMPTS[concept]` BEFORE applying `eval_prompt` (PROT:810), so the
  override alone still KeyErrors; the anchors are also registered into that table in memory (`_register_anchors`).
- `scripts/tada/sa3_descriptor_score.py` (new): `score` writes `protocol_results/desc_<name>.csv` (alpha, mean, std, scores:
  the layout `auc.py: load_and_merge_alignment` reads) for centroid_st, lowhigh_db, perc_ratio, onset_rate, flatness
  (+ tm_brightness/warmth/roughness/hardness with `--timbral` if `timbral_models` imports), plus `desc_clips.csv`;
  `auc` (or `score --auc`) writes `desc_<name>_norm.csv` (divided by the signed PCI-all full-swap gap of that descriptor for
  the dir's concept), runs the reference `compute_alignment_auc_direction` against the min(PCI-all, PCI-loc) cutoff
  (lpaps_endpoints.csv accepted), writes `auc_desc.json/csv` with steer/PCI-all ratio per descriptor (+ MuQ when muqt.csv
  exists), and `cross_effect.csv` (each knob dir x descriptor: slope near alpha 0 per alpha and per knob unit, delta at the
  cutoff per direction raw and in units of the descriptor's home-concept PCI gap, alpha and knob value at the cutoff).
- `scripts/tada/proxies.py` (new): verbatim copy of DEMON-steer `scripts/steering/proxies.py` (origin noted at the top).

## Follow-up (coordinator additions)

- `--hook cross_attn_output|post_block_residual` (default cross_attn_output, old runs unchanged) selects the
  `steer_offline` site for `--vec-dir` vectors (caa/austeer, any `--sites` incl. `all`); `--method pack` always uses the
  pack's own hook. `sweep.json` records `hook`. (K/V and oracle paths are unchanged: cross-attention only.)
- Single descriptor scorer = `scripts/tada/sa3_descriptor_score.py` (it already covers per-alpha CSVs in the AUC layout,
  the reference AUC with steer/PCI ratios, and the cross-effect table). Added for the per-block driver: run suffixes
  (`pack_pack_bright_b23`, `_prod`) resolve back to the PCI concept; `auc --pci-root <eval sub>` takes the PCI dirs from
  another sub; cross_effect.csv gains `lpaps_at_max_*` and `delta_per_lpaps_*` (endpoint gain per unit distortion, no PCI
  needed). `scripts/steering_bench/sweep_blocks.sh` now calls it (score + auc, `PCI_SUB` default eval, `WORKERS` 24) and
  ends at `$OUT/$SUB/cross_effect.csv` instead of `summary.csv`. `chain_directions.sh` has no descriptor step (capture +
  build only), so it is unchanged. The other agent's `score_descriptors.py` / `desc_pool.py` stay in the tree, unused by
  the drivers. Caveat: the per-block sweeps use 20 holdout prompts while PCI in `eval` uses 50 test prompts, so their
  cutoff/ratio columns compare different prompt sets; the endpoint columns do not depend on PCI.

## Smoke (run, CPU)

- Real packs through `_load_vectors`: all five load, hook post_block_residual, blocks bright 23, warm 15, rough 2,
  density 1, percussive 15, 8 steps x unit norm, policy weights all 1, magnitudes 1.856/2.952/1.573/1.290/4.015.
- Fake 24-block model: block-output hook adds alpha*v for tensor and tuple outputs. `tests/unit/test_sa3_tada_hooks.py` 3 pass.
- Synthetic eval root (scratchpad `descsmoke/eval`: pci_all_bright, pci_all_warm, pack_pack_bright, pack_pack_warm,
  fake lpaps.csv): score + auc end to end; the reference auc.py is local (E:\Projects\tada-replication\steer-audio), so AUC
  ran too; PCI dirs score ratio 1.00 against themselves as expected.
- Timing: 1500 clips x 10 s in one dir = 100 s with 24 workers.
- `_register_anchors` checked in evalenv. timbral_models is NOT installed locally: that path is untested.

## One concept end to end (bright)

Generation in the DEMON venv from the worktree root; scoring in evalenv. `PACKS=C:/Users/ryanf/.daydream-scope/models/demon/steering_packs/sa3/medium`,
`OUT=E:/Projects/steering-bench/2026-10-05`, `export TADA_REF=E:/Projects/tada-replication/steer-audio HF_HOME=D:/huggingface_cache`.

```
# 1 probe (20 holdout prompts, 13 strengths)
python scripts/tada/sa3_tada_run.py probe --method pack --pack $PACKS --concepts bright --holdout --n-prompts 20 --batch 20 \
  --alphas -64 -32 -16 -8 -4 -2 2 4 8 16 32 64 --calib-sub calib --out $OUT
evalenv/python scripts/tada/sa3_tada_score.py protocol --out $OUT --sub calib --concepts bright --lpaps-only
# 2 PCI-all (50 prompts, k = +-2 4 6 8); add "--pci-sites all loc --loc 23" for PCI-loc at the pack block
python scripts/tada/sa3_tada_run.py pci --concepts bright --n-prompts 50 --batch 25 --pci-ks 2 4 6 8 --pci-sites all --eval-sub eval --out $OUT
evalenv/python scripts/tada/sa3_tada_score.py protocol --out $OUT --sub eval --labels pci_all --concepts bright --skip-aesthetics
# 3 calibrate -> $OUT/ranges_eval.json (ranges.pack.bright.pack)
python scripts/tada/sa3_tada_run.py calibrate --concepts bright --eval-sub eval --calib-sub calib --out $OUT
# 4 sweep (50 prompts, 7/side + 0)
python scripts/tada/sa3_tada_run.py sweep --method pack --pack $PACKS --concepts bright --n-prompts 50 --batch 25 --points 7 \
  --ranges $OUT/ranges_eval.json --eval-sub eval --out $OUT
# 5 score (LPAPS + MuQ + CLAP; no aesthetics)
evalenv/python scripts/tada/sa3_tada_score.py protocol --out $OUT --sub eval --labels pack_pack --concepts bright --skip-aesthetics
# 6 descriptors (CPU; add --timbral if timbral_models is installed)
python scripts/tada/sa3_descriptor_score.py score --root $OUT/eval --workers 24
# 7 AUC: descriptors (steer/PCI ratios + cross_effect.csv) and the reference MuQ/CLAP AUC
python scripts/tada/sa3_descriptor_score.py auc --root $OUT/eval
evalenv/python scripts/tada/sa3_tada_score.py auc --out $OUT --sub eval --concepts bright
```
Optional MuQ gate: `sa3_tada_run.py swap --concepts bright --n-prompts 50 --batch 25 --eval-sub eval --out $OUT` renders the
real pos/neg prompts (`swap_full_bright/{pos,neg}`, no scorer wired for that layout). Knob value = alpha / magnitude
(`knob_at_cut_*` in cross_effect.csv).

## Files to copy to the box (relative to the worktree)

- acestep/engine/sa3_tada.py
- scripts/tada/sa3_tada_run.py
- scripts/tada/sa3_tada_score.py
- scripts/tada/sa3_descriptor_score.py
- scripts/tada/proxies.py
- scripts/steering_bench/sweep_blocks.sh (edited to call the scorer above)
- (packs) ~/.daydream-scope/models/demon/steering_packs/sa3/medium/{bright,density,percussive,rough,warm}.safetensors

## Open

- timbral_models call form (`timbral_x(y, fs=sr)` on an array) is untested; failures go to NaN per clip.
- Descriptor ratios blow up when the PCI-all gap of that descriptor is near 0 (seen on synthetic data); read the
  `pci_gap_own` column before trusting a ratio.
- PCI-loc at the pack block is optional; without it the cutoff is PCI-all alone.
