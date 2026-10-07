# Many-knobs runbook: code CLIs (per worker section)

## code-A (capture + labelling), 2026-10-05

Code: worktree DEMON-steer-bench, branch ryanontheinside/spike/steer-bench (commit in the code-A hand-back).
Box copy: `git archive` of that commit unpacked to /root/DEMON-steerbench (capture_resid.py imports acestep/ and
scripts/sa3/ from its own repo root, so run it from that full tree, not from DEMON-bench).
Files: scripts/steering_bench/{capture_resid.py, label_corpus.py, anchors.json}.

    CAP=/dev/shm/steerbench/out/corpus; B=/root/DEMON-steerbench; mkdir -p $CAP/logs
    cp <planner render_prompts.json> $CAP/prompts.json

### Phase 1: render + capture (demonenv), GPUs 0-1, one line per GPU. Start part 0 first (creates means.npy, owns meta.json).

    source /dev/shm/steerbench/env.sh; cd $B; export PYTHONPATH=$B
    CUDA_VISIBLE_DEVICES=0 nohup /root/demonenv/bin/python scripts/steering_bench/capture_resid.py --prompts-json $CAP/prompts.json --out $CAP --part 0/2 > $CAP/logs/capture_p0.log 2>&1 &
    sleep 30; CUDA_VISIBLE_DEVICES=1 nohup /root/demonenv/bin/python scripts/steering_bench/capture_resid.py --prompts-json $CAP/prompts.json --out $CAP --part 1/2 > $CAP/logs/capture_p1.log 2>&1 &

- Restart after a crash: same lines plus `--resume` (finished shards skipped; part 0 must be given --resume too).
- Output: means.npy fp16 [N,8,24,1536] (5000 rows = 2.9 GB), audio_{k:04d}.npz + .done (32 clips, 10 s int16 mono,
  ~28 MB each, 157 shards = 4.4 GB), meta.json (complete=true only when every shard of both parts is done).
- Defaults kept from the existing script: ARC medium, 8 steps, cfg 1, 10 s, generate batch 16 (2 calls per shard).
- Seeds: one per generate call = seed of the call's first row (meta.json gen_seed lists the effective seed per row).
- If part 1 dies, part 0 waits forever for its shards before writing complete=true: rerun part 1 with --resume.

### Phase 2: labelling (evalenv), starts as soon as shards land; each process polls and exits when all shards are labelled.

Before the first labeller starts (and after any catalogue regeneration), sync the catalogue's anchors and check names
(CPU, no audio; must end `unresolved: 0`):

    python scripts/steering_bench/check_catalogue.py --catalogue $CAP/concept_catalogue.csv --sync-anchors

    source /dev/shm/steerbench/env.sh; export TORCH_HOME=/dev/shm/steerbench/torchhub; cd $B; L="/root/evalenv/bin/python scripts/steering_bench/label_corpus.py --cap $CAP"
    # GPU 2: PaSST then CLAP music
    (CUDA_VISIBLE_DEVICES=2 $L --scorer passt --device cuda:0 && CUDA_VISIBLE_DEVICES=2 $L --scorer clap_music --device cuda:0) > $CAP/logs/label_gpu2.log 2>&1 &
    # GPU 3: CLAP general then MuQ
    (CUDA_VISIBLE_DEVICES=3 $L --scorer clap_general --device cuda:0 && CUDA_VISIBLE_DEVICES=3 $L --scorer muq --device cuda:0) > $CAP/logs/label_gpu3.log 2>&1 &
    # CPU scorers (384 cores; leave headroom for the renders)
    nohup $L --scorer timbral --workers 96 > $CAP/logs/label_timbral.log 2>&1 &
    nohup $L --scorer dyn --workers 64 > $CAP/logs/label_dyn.log 2>&1 &
    nohup $L --scorer desc --workers 32 > $CAP/logs/label_desc.log 2>&1 &
    # when labels/{desc,dyn,timbral,passt,clap_music,clap_general,muq}.done all exist:
    $L --merge        # -> $CAP/labels/merged.parquet (outer join on id)

- Checkpoints (defaults in label_corpus.py; override by env): CLAP_MUSIC_CKPT=/root/.cache/audio_metrics/music_audioset_epoch_15_esc_90.14.pt
  (HTSAT-base), CLAP_GENERAL_CKPT=/dev/shm/steerbench/ckpt/clap/630k-audioset-best.pt (HTSAT-tiny),
  AUDIOSET_LABELS=/dev/shm/steerbench/ckpt/audioset/class_labels_indices.csv, MUQ_MODEL=OpenMuQ/MuQ-MuLan-large
  (HF_HOME=/root/hf from env.sh), PaSST weights from $TORCH_HOME. Anchor texts: scripts/steering_bench/anchors.json
  (182 texts, 7 groups, plus `concepts`: 481 catalogue rows -> 663 columns per text scorer, 763 texts encoded once;
  `--template "This is a music of {}"` optional, default raw text).
- Columns (2026-10-05 catalogue fix): desc 18 (7 old + band shares, tilt, bandwidth, rolloff85, floor_db), dyn 26
  (14 old + loudness_slope, momentary_std, silence_frac, peak_to_lufs, clip_frac, am_tremolo, pulse_clarity,
  tempo_stability, offbeat_share, hf_onset_rate, pitch_centroid, chroma_entropy), timbral 8, passt 527.
- timbral: label_corpus.py applies the numpy 2 / librosa 0.11 shim itself before importing timbral_models.
- Crash/restart: rerun the same line; per-shard parts in labels/.parts/<scorer>/ are reused. `--force` redoes a
  finished scorer's final parquet. `--timeout S` makes a scorer give up after S idle seconds (rc 3, no .done).
- passt without hear21passt: rc 2, error line, nothing written. musetimbre: NotImplementedError stub (MUSETIMBRE_REPO).
- Audio shards may be moved to trash only after every scorer's .done exists.

### Packages the box needs (box-prep reports all present in evalenv, 2026-10-05)
evalenv: pandas, pyarrow, librosa, soundfile, pyloudnorm, timbral_models (+ shim in code), hear21passt + timm,
laion_clap, muq, torchaudio (optional: falls back to librosa resampling). demonenv: nothing new (numpy, torch).

### Local smoke (CPU, run 2026-10-05)
Fake 80-clip corpus (3 shards; sines, noise bursts, click trains, chords) through capture_resid.run_prompts_json with a
fake generator: planned crash after shard 0 + --resume, and a concurrent --part 0/2 + 1/2 run byte-identical to 1 part.
Then desc, dyn, timbral (DEMON venv + pyarrow/pyloudnorm/timbral_models, numpy 2.4 / librosa 0.11, so the shim was
exercised), clap_music and muq on CPU (local evalenv), --merge: 80 rows, unique ids in prompts order, all 29
descriptor columns float32 with no NaN, 394 columns total. Not run locally: passt (hear21passt missing locally,
error path checked), clap_general (no 630k non-fusion ckpt locally), the real SA3 capture (GPU).

## code-B (directions, dedupe, screening, packs), 2026-10-05

Code: worktree DEMON-steer-bench, branch ryanontheinside/spike/steer-bench, commit 31b29a12 (on top of code-A's
a0803111, so one `git archive 31b29a12` to /root/DEMON-steerbench carries both). Files:
scripts/steering_bench/{build_directions.py, dedupe.py, screen_knobs.py, make_packs.py}, scripts/tada/sa3_tada_run.py
(`--pci-descriptors`). Envs: build/dedupe = evalenv (pandas + pyarrow; numpy/torch only), screen = demonenv
generator + evalenv scorer worker, make_packs = demonenv (acestep.steering needs loguru).

    source /dev/shm/steerbench/env.sh; CAP=/dev/shm/steerbench/out/corpus; B=/root/DEMON-steerbench; cd $B
    export PYTHONPATH=$B PYTHONUTF8=1; EV=/root/evalenv/bin/python; DM=/root/demonenv/bin/python
    cp <planner>/concept_catalogue.csv $CAP/concept_catalogue.csv
    cp <planner>/holdout_sfx_prompts.json $CAP/holdout_sfx_prompts.json     # screen_set sfx prompts

### Phase 3: directions (CPU, ~2 to 5 min for 500 concepts x 5000 clips; block by block off the fp16 memmap)

    $EV scripts/steering_bench/build_directions.py --cap $CAP --catalogue $CAP/concept_catalogue.csv > $CAP/logs/dirs.log 2>&1
    # -> $CAP/dirs/<name>.npz (unit [8,4,1536] fp16 at the 4 best blocks, std/effect [8,24], pos_idx/neg_idx) + index.csv
    # one concept: --label-col "passt.Piano" --sign 1 [--name piano --second-col muq.piano]

- Catalogue header is read as is: primary_label (= label_col), sign +/-, second_scorer, third_scorer, population
  (music = every prompt category except sfx, sfx, all), class_rule ("top N by score vs bottom P% of population" or
  "quartile (top P% vs bottom P%)"), screen_set, pos_anchor/neg_anchor, blurb; extra columns pass through to index.csv.
- `effect` is cross-fitted (direction from even rows, scored on odd rows and back); the in-sample gap of 1536-dim
  means is inflated by pure noise and makes every block look alike. best_block = argmax of step-mean effect.
- Read the last log lines: `WARN N concepts have no label column` lists unmatched scorer prefixes (status
  missing_label in index.csv). With the fixed catalogue there are none (check_catalogue.py); `--col-alias` is no
  longer needed.

### Phase 4: dedupe (CPU, ~1 min)

    $EV scripts/steering_bench/dedupe.py --cap $CAP > $CAP/logs/dedupe.log 2>&1
    # -> dirs/cosine_matrix.npy + cosine_names.json, dirs/clusters.csv (keeper = highest effect), dirs/cross_projection.{npy,csv}

### Phase 5: screening, ONE process per GPU (model loads once; candidates = cluster keepers, round-robin over GPUs)

    mkdir -p $CAP/screen
    for g in 0 1 2 3; do CUDA_VISIBLE_DEVICES=$g nohup $DM scripts/steering_bench/screen_knobs.py --cap $CAP \
        --gpu-index $g --gpu-count 4 --eval-python $EV --label-workers 24 > $CAP/screen/log_g$g.txt 2>&1 & done
    # after all four exit:
    $EV scripts/steering_bench/screen_knobs.py --cap $CAP --finalize     # -> screen/accepted.csv (greedy, |cos| < 0.8)

- First: `--dry-run --limit 3` on one GPU (fake generator and scorer, real loop, ring and worker protocol) to check paths.
- Per candidate: 12 holdout prompts (screen_set music = TADA holdout, sfx = $CAP/holdout_sfx_prompts.json) x
  alpha = K x std, K = 2 4 8 (`--k`), at the best block, post_block_residual, step-mean unit vector (what make_packs
  ships). Alpha 0 rendered once per process and prompt set. `--signs both` adds -K (per-sign gain; doubles renders).
- Audio: $CAP/screen/ring/g<i>_s<0..7>.npz (12 clips each, ~10.6 MB, 8 per process = 85 MB; 4 GPUs = 0.34 GB) plus
  g<i>_ref_<set>.npz, overwritten in place, never deleted. Restart = same line: candidates with a screen/<name>.json
  are skipped (`--force` redoes).
- Scorer worker (evalenv, same GPU): reference LPAPS (CLAP music ckpt, 10 s windows) vs alpha 0, MuQ + CLAP cosine to
  pos_anchor, and the own/second/third label columns through label_corpus.make_scorer (each scorer object kept
  loaded; pass `--label-template` if the corpus was labelled with one). The generator renders the next candidate
  while the worker scores the previous one.
- Pass rule (per candidate json, summary.csv `pass_signal`): >= 2 positive alphas with mean LPAPS <= `--lpaps-cut`
  (default 4.0; a catalogue column lpaps_cut overrides), own column sign-adjusted delta > 0 and non-decreasing in band
  with z >= 2 at the largest in-band alpha, and the first second/third scorer column of ANOTHER family (corpus
  |corr| >= 0.1, expected sign = sign of that corr) also > 0 and non-decreasing, else the MuQ pos_anchor.
  Flags: needs_ear (own column is CLAP/MuQ text), second_same_family, second_weak_corr, no_second_scorer,
  too_few_in_band. `pass` adds the cosine gate against what summary.csv held at that moment; accepted.csv is the
  order-independent version.
- Expected cost: ~0.2 to 0.4 GPU-min per candidate (36 renders ~4 s, LPAPS ~2 s, anchors ~4 s, passt/CLAP/MuQ
  columns ~0.05 s/clip; the CPU label scorers dominate: desc ~1.5 CPU-s/clip, timbral several CPU-s/clip, so raise
  `--label-workers` if the `seconds` column climbs). 400 keepers on 4 GPUs: ~20 to 40 min wall. Check the `seconds`
  column of the first 10 rows of screen/summary.csv (it includes pipeline wait) before trusting the estimate.
- Tune after the first ~20 rows: if most candidates show n_band 3 (LPAPS still below the cut at K=8), rerun with
  `--k 4 8 16 --force`; if K=2 already exceeds the cut, `--k 1 2 4`.

### Phase 6a: packs for accepted knobs (CPU)

    $DM scripts/steering_bench/make_packs.py --cap $CAP --pci-descriptors
    # -> $CAP/packs/sa3/medium/<name>.safetensors (format 1, best block, magnitude = norm x 0.1, provenance incl.
    #    screening numbers + calibrated gain = alpha at the LPAPS cut / magnitude), pci_descriptors.json
    #    (pos_anchor/neg_anchor per knob). Each pack is re-read with load_pack. --concepts a b c overrides the list.

### Phase 6b: survivors through the existing full protocol (sa3_tada_run.py --method pack), ~8 GPU-min per knob

Split the accepted names over the 4 GPUs (K1 = names for GPU 0, etc.), same steps as shim_status.md "One concept end
to end", with the many-knobs packs dir and their PCI descriptor texts:

    PK=$CAP/packs/sa3/medium; OUT=/dev/shm/steerbench/out/many_knobs_eval; R="$DM scripts/tada/sa3_tada_run.py"
    CUDA_VISIBLE_DEVICES=0 $R probe --method pack --pack $PK --concepts $K1 --holdout --n-prompts 20 --batch 20 \
        --alphas -64 -32 -16 -8 -4 -2 2 4 8 16 32 64 --calib-sub calib --out $OUT
    $EV scripts/tada/sa3_tada_score.py protocol --out $OUT --sub calib --concepts $K1 --lpaps-only
    CUDA_VISIBLE_DEVICES=0 $R pci --pci-descriptors $PK/pci_descriptors.json --concepts $K1 --n-prompts 50 --batch 25 \
        --pci-ks 2 4 6 8 --pci-sites all --eval-sub eval --out $OUT
    $EV scripts/tada/sa3_tada_score.py cutoff --out $OUT --sub eval --concepts $K1      # PCI LPAPS endpoints only
    $R calibrate --method pack --concepts $K1 --eval-sub eval --calib-sub calib --out $OUT
    CUDA_VISIBLE_DEVICES=0 $R sweep --method pack --pack $PK --concepts $K1 --n-prompts 50 --batch 25 --points 7 \
        --ranges $OUT/ranges_eval.json --eval-sub eval --out $OUT
    $EV scripts/tada/sa3_tada_score.py protocol --out $OUT --sub eval --labels pack_pack --concepts $K1 --lpaps-only
    # full protocol with MuQ/CLAP alignment (pos_anchor of each knob as the eval prompt), then the AUC table:
    $EV scripts/tada/sa3_tada_score.py protocol --out $OUT --sub eval --labels pack_pack --concepts $K1 \
        --catalogue $CAP/concept_catalogue.csv --force
    $EV scripts/tada/sa3_tada_score.py auc --out $OUT --sub eval --concepts $K1

- Knobs without pos/neg text get no PCI (no cutoff); calibrate then falls back to the largest probe (`reached:
  false`); use the screening cut instead (pack provenance calibrated_gain).
- MuQ/CLAP alignment AUC (`protocol` without --lpaps-only, `auc`): `--catalogue` (the catalogue CSV, or a JSON
  {name: [pos, neg]} such as $PK/pci_descriptors.json) gives every knob its pos_anchor as the alignment prompt
  (MuQ form "This is a music of <pos_anchor>"). The five PACK_ANCHORS knobs and the reference TADA concepts (piano,
  violin, ...) keep their own prompts, so old runs score identically. `--force` because the --lpaps-only pass
  already wrote protocol_results/lpaps.csv.
- Audio per knob ~0.7 GB (probe 260 + PCI 450 + sweep 750 clips); move each knob's dirs to trash once scored.

## Label naming (catalogue fix, 2026-10-05)

- Every catalogue label is `<scorer>.<column>` as label_corpus.py writes it: desc.*, dyn.*, timbral.*, passt.<AudioSet
  display name>, and clap_music / clap_general / muq.<concept name> = cos(pos_anchor) - cos(neg_anchor).
- Planned prefixes were mapped in gen_catalogue_and_prompts.py (COL_MAP; proxies/spectral -> desc, level/rhythm/tonal
  -> dyn); stereo (mono shards), audiobox, basicpitch and descriptors not built became text rows (clap_music.<name>).
- Thirds from scorers that do not run (essentia, musetimbre, demucs, audiobox, basicpitch) are empty.
- A raw anchor text equal to a concept name is written as `<scorer>.anchor:<text>` (89 of 182, e.g. muq.anchor:piano).
- After any regeneration: `check_catalogue.py --catalogue <csv> --sync-anchors` must print `unresolved: 0`.
- Code: commit 289da4e3 on the spike branch (on top of 31b29a12); `git archive 289da4e3` for the box. CPU smokes rerun:
  code-B synthetic corpus (identical accepted.csv), code-A 80-clip corpus incl. clap_music + muq on CPU, and
  build_directions on the real catalogue over that corpus: 0 missing_label.
