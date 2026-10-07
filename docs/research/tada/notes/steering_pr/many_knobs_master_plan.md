# Many-knobs master plan (2026-10-05): capture once, label many, build on CPU, screen cheap, measure survivors

Owner: parent. Workers: planner (catalogue, prompts, runbook), code-A (capture + labelling), code-B (directions,
dedupe, screening), box-prep (discriminator installs), launcher-2 (runs everything on the box).
Box: vast.ai 51480126, 4x RTX 5090 32 GB, RAM disk /dev/shm (19 GB free), root disk nearly full. Envs and
paths: box_status.md. Worktree for all code: C:\_dev\projects\DEMON-steer-bench, branch
ryanontheinside/spike/steer-bench, new files under scripts/steering_bench/ (ownership below).

## Timeline (wall clock, from now)

T+0:00  launcher-1 still finishing warm@b23, rough/density sweeps, seed-2 runs (uses all 4 GPUs, ~60 min).
        planner, code-A, code-B, box-prep run in parallel on CPU (~60 to 90 min).
T+1:30  launcher-2 starts phase 1 on GPUs 0-1 (render + capture, ~10 min for 5000 clips) and phase 2
        labelling on GPUs 2-3 as shards land (~30 min). Phase 3+4 on CPU (~15 min).
T+2:15  phase 5 screening on all 4 GPUs: ~300 to 500 candidates, ~0.7 GPU-min each with the ring buffer
        (12 holdout prompts x 3 alphas + 0) => ~1.5 h wall.
T+4:00  phase 6 full protocol for survivors (~8 GPU-min each; 40 survivors => ~80 min wall).
T+5:30  results table, pack files for accepted knobs, cross-effect matrix.

## Interfaces (every worker codes against these; do not change them without updating this file)

Capture corpus layout, root $CAP (box: /dev/shm/steerbench/out/corpus):
- prompts.json: list of {id:int, prompt:str, seed:int, category:str, tags:[str]} (planner writes it, 5000 rows).
- means.npy: float16 [N, 8 steps, 24 blocks, 1536], audio tokens only, row order = prompts.json order.
  meta.json: {"n": N, "steps": [...sigmas], "hook": "post_block_residual", "sr": 44100, "duration_s": 10, ...}.
- audio shards audio_{k:04d}.npz with int16 arrays `wav` [B, T] (mono) and `ids` [B]; B = 32; written as
  rendering proceeds so labelling starts early; a shard is complete when audio_{k:04d}.done exists.
Label matrix: $CAP/labels/<scorer>.parquet, one row per clip id, columns `<scorer>.<label>` float32
  (descriptors: scalar; classifiers: class probabilities; CLAP/MuQ: cosine to each text anchor in
  anchors.json). labels/merged.parquet = outer join on id. Each scorer writes labels/<scorer>.done.
Directions: $CAP/dirs/<concept>.npz with `unit` [8, 4, 1536] fp16 (per step unit vectors at the 4 best blocks
  `blocks` [4], disk budget), `best_block`, `norm` [8, 4], `std` [8, 24] (projection std along the direction),
  `effect` [8, 24] (class gap / pooled std, 2-fold cross-fitted), `n_pos`, `n_neg`, `pos_idx`, `neg_idx`,
  `label_col`, `sign`. Plus dirs/index.csv (code-B, 2026-10-05).
Dedupe: $CAP/dirs/cosine_matrix.npy over concepts at their best block (mean over steps) and
  dirs/clusters.csv (cosine > 0.8 merged, keeper = highest effect).
Screening: $CAP/screen/<concept>.json per candidate: per alpha {lpaps_mean, own_scores{...}, muq, clap},
  slope per unit LPAPS, pass flag. screen/summary.csv.
Packs: $CAP/packs/sa3/medium/<name>.safetensors, production format 1 (one step-mean vector at the chosen
  block, magnitude = raw norm x 0.1, calibrated gain in provenance).

## Acceptance rule (planner refines, code-B implements)
A candidate passes screening when, within the LPAPS cutoff band, its own label column moves the right way
AND an independent second scorer agrees (two different scorer families, e.g. descriptor + classifier, or
classifier + CLAP), both monotone across the 3 alphas, and its best-block cosine to every already-accepted
knob is below 0.8. Abstract (text-anchor-only) concepts additionally need a human spot check before shipping.

## Ownership
- planner: notes/steering_pr/{many_knobs_plan.md, discriminators.md, concept_catalogue.csv,
  render_prompts.json (copied to $CAP/prompts.json), many_knobs_runbook.md}.
- code-A: scripts/steering_bench/capture_resid.py (prompt-file input, sharded audio, .done markers),
  new scripts/steering_bench/label_corpus.py (one scorer per invocation: descriptors, timbral, loudness/
  tempo/key/onsets, passt, clap_music, clap_general, muq, musetimbre; writes parquet + .done), anchors.json.
- code-B: scripts/steering_bench/build_directions.py (any label column, per step per block, std, effect),
  new dedupe.py, new screen_knobs.py (ONE process per GPU looping over many candidates, ring buffer of 8
  audio shards overwritten in place so nothing is ever deleted, in-process LPAPS via CLAP + own scorers),
  new make_packs.py; the `--hook` and multi-pack hooks in scripts/tada/sa3_tada_run.py if needed.
- box-prep: installs on the box only (hear21passt or another AudioSet tagger, pyloudnorm, essentia if it
  installs, laion_clap general checkpoint, parquet libs), smoke each on CPU with one wav, record versions
  and import commands in box_status.md section "discriminators"; no GPU use, no touching /dev/shm/steerbench/out.
- launcher-2: runs phases 1 to 6 per many_knobs_runbook.md, status in status_many_knobs.md, results in
  results_many_knobs.md, packs copied to E:\Projects\steering-bench\many_knobs\.

## Disk budget on the 19 GB RAM disk
means 3.5 GB + audio 4.4 GB (moved to trash after labelling; user purges) + labels < 0.5 GB + dirs ~0.1 GB
per concept at fp16 (500 concepts = 30 GB: NO. Store only `unit` at the best 4 blocks per concept, all
steps: 0.1 GB per 500) + screening ring 8 shards x 32 clips = 0.2 GB + survivors' full-protocol audio
(move to trash per knob, ~0.7 GB each, user purges in batches). Peak < 12 GB.
