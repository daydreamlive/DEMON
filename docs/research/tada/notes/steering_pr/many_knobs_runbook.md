# Many-knobs runbook (box: vast 51480126, 4x RTX 5090, RAM disk /dev/shm)

Companion to many_knobs_plan.md and many_knobs_master_plan.md (interfaces). Envs, ssh and paths: box_status.md.
Capture CLI as it stands: directions_status.md. Status goes to status_many_knobs.md; hand back only on a
state change (launched, failed, done, FLAG), never on a timer tick. No rm anywhere: scored audio is MOVED to
/dev/shm/steerbench/steer-bench-trash/many_knobs/<phase>/. That trash is on the same tmpfs, so a move frees
nothing; only the user can purge it. The budget below assumes NO purge.

Numbers marked (m) are measured on this box; everything else is an estimate from them.
(m): render 0.11 to 0.15 GPU-s per 10 s clip at 8 steps; proxies 1.5 CPU-s per clip; per-process model load
about 20 s; K = 4 x proj_std gave LPAPS 3.1 to 3.7 at b00-b22 and 4.4 to 5.2 at b23; PCI cutoffs 4.0 to 4.8.

## Phase 0: pre-flight (box-prep, CPU only)

- `df -h /dev/shm /` and record. Launch phase 1 only if /dev/shm has >= 18 GB free; otherwise turn on FLAC
  shards (checklist item C3) or ask the user to purge the existing trash first. FLAG either way.
- Installs into evalenv (no deps unless needed): hear21passt (PaSST, ~0.3 GB ckpt), beat_this, pyarrow, demucs
  if missing (84 MB htdemucs), laion_clap general checkpoint 630k-audioset-best.pt (1.86 GB). Optional:
  essentia-tensorflow + Discogs-EffNet / MTG-Jamendo heads (~0.1 GB), basic-pitch. Skip YAMNet and madmom.
- MuseTimbre: upload only the timbre_encoder.* tensors (~0.3 GB), extracted locally on CPU from
  E:\Projects\instrument-controlnet\musetimbre_spike_2026-10-02\ckpt\musetimbre_v1.pt (checklist item L9); the
  CLAP music base checkpoint already exists at /root/.cache/audio_metrics/.
- Smoke each scorer on one wav on CPU; record versions in box_status.md "discriminators". Timbral needs
  /root/sb/tm_shim.py imported first (an unshimmed pass returns all NaN).
- Copy notes/steering_pr/{render_prompts.json -> $CAP/prompts.json, holdout_sfx_prompts.json, proto_prompts.json,
  concept_catalogue.csv} to $CAP = /dev/shm/steerbench/out/corpus.
- Thread caps for every CPU worker: OMP_NUM_THREADS=MKL_NUM_THREADS=OPENBLAS_NUM_THREADS=1, at most 48 workers
  per pool (pids.max 94208 was exhausted before by uncapped workers).

## Phase 1: render + capture (GPUs 0-1)

- 5000 prompts, ARC medium, 8 steps, 10 s, batch 32, hook post_block_residual at all 24 blocks x 8 steps,
  audio tokens only. Two processes over id ranges 0-2495 and 2496-4999 (shard-aligned: 78 x 32) writing rows of one pre-created
  means.npy and globally numbered shards audio_{k:04d}.npz (+ .done). Stereo stats computed inline before the
  mono mean -> labels/stereo.parquet.
- 1b: MuseTimbre prototype renders: 103 instruments x 8 seeds of "solo <instrument>, no other instruments",
  ids 100000+, audio only (no capture), 824 clips. These ids are never in the label matrix.
- Intended CLI (after code-A lands):
  `GPU=0 capture_resid.py --prompts $CAP/prompts.json --id-range 0:2496 --out $CAP --create --shard 32`
  `GPU=1 capture_resid.py --prompts $CAP/prompts.json --id-range 2496:5000 --out $CAP --shard 32`
  then `GPU=1 capture_resid.py --prompts $CAP/proto_prompts.json --no-capture --out $CAP/proto`
- GPU: 5000 x 0.13 s = 11 GPU-min + 1b 2 GPU-min; wall about 7 min.
- Disk: means 2.95 GB (5000 x 8 x 24 x 1536 x 2 B), audio 4.41 GB (int16 mono 44.1 kHz), proto 0.73 GB.
- Done when 157 shards + 26 proto shards have .done and meta.json is written. FLAG if a shard is missing.

## Phase 2: labelling (GPUs 2-3 while phase 1 runs, then all 4; CPU pool)

- One process per scorer per GPU, each polling for new .done shards: `label_corpus.py --scorer <name> --cap $CAP
  --device cuda:N`. GPU order: passt, muq, clap_music, clap_general, audiobox, musetimbre (needs 1b first),
  demucs, beat_this; optional ast, essentia. CPU pool (48 workers): proxies, timbral, level, rhythm, tonal,
  spectral (and basicpitch if installed).
- Output: $CAP/labels/<scorer>.parquet (one row per id, columns <scorer>.<label>, float32) + <scorer>.done;
  merged.parquet = outer join on id. Text scorers read pos/neg anchors from concept_catalogue.csv.
- GPU: passt 2, muq 4, clap x2 4, audiobox 4, musetimbre 3, demucs 13, beat_this 4 GPU-min + about 8 model
  loads = about 41 GPU-min (optional ast +4, essentia +8). CPU: about 21 CPU-h (timbral 14 of it) = about
  26 min wall on 48 workers.
- Disk: labels < 0.1 GB. When every scorer has .done, MOVE audio_*.npz and proto/ to trash (frees nothing on
  shm, but marks them disposable for the user's purge).
- FLAG if any scorer's column is NaN for > 1% of ids.

## Phase 3: label validity + directions (CPU)

Gate 0, label validity (before any direction is used):
- Tagged (sparse) concepts: AUC of the primary column at separating clips whose GENERATED prompt carries the
  concept tag from the rest of the population >= 0.75; the second scorer >= 0.70. Only categories music, solo,
  sfx, hybrid, abstract count (TADA and MusicCaps tags are keyword guesses). If the primary fails and the
  second passes, swap them and FLAG the swap. Also: the top 250 by primary score must contain >= 30% tagged
  clips.
- Dense (descriptor) concepts: Spearman(primary, second) over the population >= 0.2 in the expected sign, or
  tag AUC >= 0.65 where prompt modifiers exist.
- Output dirs/gate0.csv. Concepts that fail are not screened (expected 20 to 35% of rows, mostly text-only).

Directions (build_directions.py generalised):
- Classes per row: dense = top vs bottom quartile of the population; sparse = top 250 by score vs bottom 25%.
  Population masks: music rows use the 3700 non-sfx clips, sfx rows the 1300 sfx clips, "all" rows all 5000.
  The 50 TADA protocol test prompts used in phase 6 are excluded from both classes.
- Per step per block: unit vector (mass-mean difference), raw norm, pooled within-class projection std, std
  over all rows, effect = quartile gap / pooled std. Ridge readout of every label at every cell (5-fold CV R2),
  shared X'X per cell.
- Best block for screening: argmax of step-mean effect over blocks 8-23, ties to the later block (sweep data:
  effect peak was within about 10% of the best steerable block for bright and percussive, warm wanted b23).
- CPU: about 2.8 TFLOP for directions + 2.3 TFLOP for readouts, about 10 min on the box CPUs (seconds on one
  GPU if a GPU is free). Disk: unit [8,24,1536] fp16 = 0.59 MB per concept -> 0.28 GB for 481; readouts the same;
  0.6 GB in all. (The master plan's 0.1 GB per concept was 170x too high; all blocks fit.)

## Phase 4: dedupe + cross-projection (CPU, minutes)

- Cosine between concepts at each concept's best block, step-mean, SIGN-AGNOSTIC: |cos| > 0.8 merges
  (bright and warm are cos -0.95 to -0.99: one axis). Keeper = best label strength (descriptor > classifier >
  text_only), then highest effect. dirs/cosine_matrix.npy, dirs/clusters.csv.
- Cross-projection matrix: predicted change of label j per unit alpha of knob i = readout_j(b_i) . u_i(b_i),
  in units of label j's corpus SD -> dirs/cross_projection.npy. Label correlation (Spearman, all columns) ->
  dirs/label_corr.parquet, to see which labels are the same fact before reading any screening result.
- Candidates for phase 5 = gate-0 passers minus merged duplicates. Expect about 300.

## Phase 5: screening (all 4 GPUs, one process per GPU looping over candidates)

- Holdouts: music and "all" rows use the 20 TADA holdout prompts (never in the corpus); sfx rows use
  holdout_sfx_prompts.json (20). Seed 2115 (EVAL_SEED). Alpha 0 rendered once per holdout set per GPU and shared.
- Vector: step-mean unit at the best block, format-1 pack semantics. Alpha = m x proj_std(b) with
  m in {-6, -3, +3, +6} for b <= 22 and {-4, -2, +2, +4} at b23. After the first 10 knobs, if median LPAPS at
  |m|max is < 3.5, raise m to {±4, ±8}; if > 4.8, lower it. FLAG the change.
- Per candidate: 80 clips, scored in process (LPAPS, primary, second, third if cheap, audiobox PQ) and
  overwritten in a per-GPU ring of 8 shards (0.23 GB per GPU). DSP scorers in the CPU pool.
- Acceptance (all of):
  1. Band: only alphas with mean LPAPS <= 4.3 count (the median of the measured PCI cutoffs).
  2. Own label: at the largest in-band |alpha| of at least one sign, the paired change vs alpha 0 has the
     intended sign, paired t >= 3 over the 20 prompts, and |delta| >= 0.25 corpus SD.
  3. Independent agreement: the second scorer (different family) moves the same way at the same alpha with
     paired t >= 2 and |delta| >= 0.15 of its corpus SD.
  4. Monotone: both scorers' means are ordered 0 < m/2 < m on that sign (no reversal over the 3 points).
  5. Fidelity guard: audiobox PQ drop <= 0.75 at that alpha.
  6. Unique: |cos| < 0.8 at the best block to every knob already accepted (greedy, best second-scorer
     effect per LPAPS first).
  Grades: A = both signs pass and the primary is DSP or classifier; B = one sign, or both scorers are text;
  P = provisional (same-family second, e.g. stereo); text-only and abstract knobs also go on the listening list.
  Noise: with t >= 3 and t >= 2 jointly, about 0.05 false passes expected over 300 candidates.
- Remember the selection bias: the primary column chose the direction, so its delta is optimistic. The second
  scorer is the evidence.
- GPU: about 1.5 GPU-min per knob (80 renders 10 s + scoring + overhead). 300 x 1.5 = 450 GPU-min, about
  1.9 h wall on 4 GPUs. Disk: 0.9 GB of rings + screen/*.json.

## Phase 6: full protocol for survivors (all 4 GPUs)

- Per survivor, in a per-GPU slot dir that is overwritten in place (alpha-indexed dir names, checklist T2),
  run in three passes so a slot never holds more than one pass:
  1. PCI-all on 50 TADA test prompts (pos/neg anchors as prompt modifiers, ks ±2,4,6,8 + full swap, 500 clips)
     -> per-sign LPAPS cutoff and PCI normaliser.
  2. Steering sweep: 15 alphas (0 shared) over the per-sign range to the cutoff, 700 clips. Score LPAPS, own,
     second, third, MuQ and every accepted knob's home label (cross-effect row).
  3. Block mini-sweep: {b*-4, b*+4, 23} at 20 holdout x 4 alphas, 240 clips.
- Ship rule: steer/PCI AUC ratio >= 1.0 on the SECOND scorer and on the own label; monotone to the cutoff on
  both; every off-target |delta| < 0.5 x own effect on the other accepted knobs' labels (else "entangled with X");
  per-sign alpha at cutoff written into the pack provenance (make_packs.py). Grade B and abstract knobs ship only
  after a human listen (4 prompts x ± cutoff).
- Survivors: expected 40 to 80; budget for 60. GPU about 9 GPU-min each (1440 renders + scoring) = 540
  GPU-min, about 2.3 h wall. Disk: slot <= 0.7 GB per GPU (largest pass 700 clips) = 2.8 GB.
- Results: results_many_knobs.md, packs to $CAP/packs/sa3/medium/, copied to E:\Projects\steering-bench\many_knobs\.

## Totals

| phase | GPU-min | wall | new disk on /dev/shm |
|---|---|---|---|
| 0 installs + models | 0 | 20 min | 2.5 GB (clap general 1.86, MuseTimbre encoder 0.3, PaSST 0.3, demucs + beat_this 0.1) |
| 1 render + capture + proto | 13 | 7 min | 8.1 GB (means 2.95, audio 4.41, proto 0.73) |
| 2 labelling | 41 (+12 optional) | 30 min (overlaps 1) | 0.1 GB |
| 3 gate 0 + directions | 0 (CPU) | 15 min | 0.6 GB |
| 4 dedupe | 0 | 5 min | < 0.05 GB |
| 5 screening, ~300 knobs | 450 | 1.9 h | 0.9 GB |
| 6 full protocol, ~60 knobs | 540 | 2.3 h | 2.8 GB |
| total | about 1045 (17.4 GPU-h) | about 5 h | peak about 15.0 GB |

Disk budget (19 GB free, 3 GB floor = 16 GB usable): peak 15.0 GB at phase 6 with the corpus audio still in
trash. Margin 1 GB. Options in order: FLAC shards (about -2 GB, checklist C3); the user purges the phase-1
trash before phase 6 (-5.1 GB, peak 9.9 GB); put model downloads on the overlay root if it still has > 3 GB
free (-2.5 GB). Launcher-1's leftover sweeps shrink the 19 GB before we start: re-measure at phase 0.

## Code-change checklist (worktree C:\_dev\projects\DEMON-steer-bench, branch ryanontheinside/spike/steer-bench)

Capture (code-A), scripts/steering_bench/capture_resid.py:
- [ ] C1 `--prompts FILE` (list of {id, prompt, seed, category, tags}); rows in file order; `--id-range a:b`;
      `--create` pre-creates means.npy (open_memmap w+), later processes open r+.
- [ ] C2 shard writer: audio_{k:04d}.npz {wav int16 [B,T] mono, ids} at B = 32, then audio_{k:04d}.done; resume
      skips done shards; meta.json per the master-plan interface.
- [ ] C3 optional `--flac` shard encoding (soundfile) to save about 2 GB.
- [ ] C4 inline stereo stats (side_mid_db, lr_corr, pan_motion, hf_side_mid_db) from the stereo tensor before
      the mono mean -> labels/stereo.parquet.
- [ ] C5 seeds: per-row seeds if sa3_tada.generate can take per-item noise; else batch seed = first row's seed,
      recorded in meta.json.
- [ ] C6 `--no-capture` (audio only) for the MuseTimbre prototype renders.
Labelling (code-A):
- [ ] L1 new scripts/steering_bench/label_corpus.py: one scorer per invocation, polls .done shards, writes
      labels/<scorer>.parquet + .done, merge step -> merged.parquet.
- [ ] L2 new scripts/steering_bench/scorers/ (dsp.py: proxies, level, rhythm, tonal, spectral ported from
      sa3judge definitions; timbral.py with tm_shim; taggers.py: passt / ast / panns with AudioSet names from
      class_labels_indices.csv; text.py: clap_music / clap_general / muq anchors from concept_catalogue.csv;
      musetimbre.py; audiobox.py; demucs.py; essentia.py optional). One shared module so screening and phase 6
      score exactly as labelling did.
- [ ] L9 new scripts/steering_bench/extract_musetimbre_encoder.py (local, CPU): timbre_encoder.* tensors of
      musetimbre_v1.pt -> one ~0.3 GB file; musetimbre.py loads it over clap_music.pt.
Directions, dedupe, screening, packs (code-B):
- [ ] D1 scripts/steering_bench/build_directions.py: `--labels merged.parquet --catalogue concept_catalogue.csv`;
      any label column + sign; dense quartile vs sparse top-250; population masks; exclusion of the 50 protocol
      prompts; per step per block unit/std/effect; process block by block (memory); dirs/<concept>.npz + index.csv.
- [ ] D2 gate 0 (label validity) in build_directions.py or new label_validity.py -> dirs/gate0.csv.
- [ ] D3 ridge readouts per cell for every label (shared X'X) -> dirs/readouts.npz.
- [ ] D4 new scripts/steering_bench/dedupe.py: sign-agnostic |cos| clusters, cosine_matrix.npy,
      cross_projection.npy, label_corr.parquet.
- [ ] D5 new scripts/steering_bench/screen_knobs.py: one process per GPU, candidate queue, both holdout sets, alpha
      grid rule, shared alpha 0, ring buffer, in-process scorers, acceptance rule, screen/<concept>.json + summary.csv.
- [ ] D6 new scripts/steering_bench/make_packs.py: format-1 packs with provenance (label column, sign, scorers,
      grade, block, per-sign alpha at cutoff).
Run/score harness (DEMON-tada-sa3 tree copied to the box as /root/DEMON-bench):
- [ ] T1 scripts/tada/sa3_tada_run.py: `--hook post_block_residual` honoured for --vec-dir; several packs per
      process (load the model once).
- [ ] T2 scripts/tada/sa3_tada_run.py: alpha-index dir names (a00..a14) and `--slot DIR` so phase 6 overwrites
      in place instead of creating new dirs.
- [ ] T3 scripts/tada/sa3_tada_score.py or scripts/steering_bench/score_descriptors.py: any scorer from
      scorers/ as home and second scorer; ratio on both; cross-effect row over a list of label columns.
- [ ] T4 scripts/steering_bench/sweep_blocks.sh: replace one-process-per-block with the multi-pack process
      (model load was 8 of the 14 GPU-min per concept).
