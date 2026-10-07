
## Deliverable B: smoke_drive.py (2026-10-07, built + CPU dry-run green; GPU run NOT started)

- Code: C:\_dev\projects\DEMON-steer-bench\scripts\steering_bench\smoke_drive.py (uncommitted). No smoke_item.sh: one local process, the box driver pattern is not needed.
- Dry run: E:\Projects\DEMON\steering-bench\v3_smoke\dryrun\ (results.json, results.md, renders/, controls/, stub_packs/): 163 renders, 12 s, end to end.
- Packs read: every provenance.method == "ridge" pack in E:\Projects\DEMON\steering-bench\v3_smoke\packs\*.safetensors (ridge_<d>, ridge_resid_<d>, as ridge_directions.py writes them; --variant-b adds packs\variant_b\). Real run refuses unless all 6 exist.
- Controls: no bundle.safetensors is installed under ~/.daydream-scope/models/demon/steering_packs/sa3/medium/; falls back to the loose percussive / bright / dense_arrangement packs there (DEMON-steer loader), re-saved as format 1 under <out>\controls.
- Blocker for the real run: pyloudnorm is missing in E:\Projects\tada-replication\evalenv (corpus dyn.lra was computed with it). Install it there, or pass --allow-lra-fallback (lra then differs from the corpus definition).
- Estimated wall time on one 5090: about 25 to 35 min (~1 ref + ~65 probe + 90 sweep renders of 12 x 10 s, LPAPS/CLAP/dyn scoring overlapped in the evalenv worker).

Launch (real run, local GPU, only after Ryan's go), from Git Bash:

    cd /c/_dev/projects/DEMON-steer-bench && PYTHONUTF8=1 TADA_ROOT=E:/Projects/tada-replication \
      /c/_dev/projects/DEMON-tada-sa3/.venv/Scripts/python.exe scripts/steering_bench/smoke_drive.py \
      --packs E:/Projects/DEMON/steering-bench/v3_smoke/packs --out E:/Projects/DEMON/steering-bench/v3_smoke \
      --eval-python E:/Projects/tada-replication/evalenv/Scripts/python.exe --tada-root E:/Projects/tada-replication \
      > E:/Projects/DEMON/steering-bench/v3_smoke/smoke_drive.log 2>&1

Outputs: E:\Projects\DEMON\steering-bench\v3_smoke\results.json, results.md (pass table per direction: Spearman median, LPAPS top/cutoff, delta at matched LPAPS vs control).

## Deliverable A: ridge_directions.py (2026-10-07, DONE on CPU; nothing committed)

- Code: C:\_dev\projects\DEMON-steer-bench\scripts\steering_bench\ridge_directions.py (uncommitted). Run (DEMON venv: sklearn yes, pyarrow no, so labels go in as CSV):
  `/c/_dev/projects/DEMON/.venv/Scripts/python.exe scripts/steering_bench/ridge_directions.py --cap E:/Projects/DEMON/steering-bench/many_knobs_v2/corpus --labels E:/Projects/DEMON/steering-bench/v3_smoke/labels_v2_subset.csv --controls E:/Projects/DEMON/steering-bench/many_knobs_v2/packs_final/sa3/medium --out E:/Projects/DEMON/steering-bench/v3_smoke/packs --variant-b` (about 5 min).
- Inputs: v2 corpus pulled from B2 mucket/sa3/steering_bench/many_knobs_v2_2026-10-05/corpus/ to E:\Projects\DEMON\steering-bench\many_knobs_v2\corpus\ (means.npy 6.39 GB size-verified, meta, prompts, labels/{desc,dyn,merged}.parquet, dirs/index.csv + 5 dirs npz). Labels = v2 merged (ids 5000..19999) + v1 merged (ids 0..4999, rows copied from init_means) as labels_v2_subset.csv; music population n = 14800 (matches the v2 quartile n_pos 3700).
- Pack dir layout (format 1, acestep.steering.packs; provenance.method = "ridge", sign +1 = raises the descriptor; norm = projected quartile gap, magnitude = 0.1 x norm; calibrated_gain null):
  E:\Projects\DEMON\steering-bench\v3_smoke\packs\ridge_{onset_rate,centroid,lra}.safetensors, ridge_resid_{onset_rate,centroid,lra}.safetensors (blocks 16 / 22 / 11 = v2 variant a best block of onset_rate / bright / dynamic, same label column);
  packs\variant_b\ same 6 names, 3 blocks each (joint ridge over the v2 top-3 blocks); packs\fit_report.json, fit_report.md.
  All 12 verified with load_pack and with sa3_tada_run._pack_vectors (the bench's --method pack path).
- Descriptor-scoring entrypoint (same definitions as the corpus labels): centroid and onset_rate = desc_pool.measure_clips (proxies.py; score_descriptors.measure_dir wraps it; corpus cols desc.centroid / desc.onset_rate); lra = label_corpus.dyn_clip((clip_int16, sr))["lra" index of DYN_COLS] (corpus col dyn.lra; needs pyloudnorm, else RMS fallback = different definition).
- For the drive: ridge pack magnitudes are 5 to 15x smaller than the v2 class-mean packs (ridge picks low-variance decoding directions: proj std 0.4 to 1.3 vs 4 to 13), so sweep in raw alpha matched by LPAPS, not by knob value. Controls in ~/.daydream-scope packs are bit-identical (cos 1.0) to packs_final's percussive/bright/dense_arrangement.

## Real run (2026-10-07, local 5090, DONE)

- Launched 10:35 ET beside the demo backend on :1318 (GPU had ~28 GB free); finished 10:56 ET, exit 0. Wall ~21 min (175 renders / 2100 clips in 1256.5 s). pyloudnorm True, no --allow-lra-fallback.
- Result: 2/12 ridge vector-directions pass, both centroid pos (ridge and ridge_resid: Spearman 1.0, delta @ matched 1430 / 1100 Hz vs bright 474 / 493). Everything else fails Spearman (onset_rate ridge 0.883 both signs, lra ridge pos 0.900 shown as fail so just under 0.9); centroid neg and lra neg also lose to the control at matched LPAPS. LPAPS budget held everywhere (top/cut 0.95 to 1.03).
- solo_build NOT added as a second lra control: it is not a single-block sign-symmetric format-1 pack (export_controls refuses it), so it needs real render-path changes. smoke_drive.py left as built (a trial edit was reverted).
- Outputs: E:\Projects\DEMON\steering-bench\v3_smoke\results.md, results.json, smoke_drive.log.

## S4: perceptual knob mapping (2026-10-07, CPU only, DONE)

- Code: C:\_dev\projects\DEMON-steer-bench\scripts\steering_bench\knob_response.py (uncommitted). Run with the DEMON venv python, defaults point at the local ship mirror; ~10 s.
- Outputs: E:\Projects\DEMON\steering-bench\v3_smoke\s4\ (knob_response.json = proposed sidecar for the 97 shipped knobs, curvature.json = per vector-sign stats + listening list, s4.md).
- Data: ship-pass probe LPAPS curves for all 143 vectors x 2 signs x seeds 0,1,2 (local mirror has no "done" markers, so presence of the probe csv is used). Stats on the 141 shipped bar-passing sides; interpolant = ship_tools.interp_side after PAV, so u = 1 reproduces the shipped gain exactly.
- Raw slider curvature: u50 (slider position spending half the LPAPS budget) median 0.273 (p10 0.228, p90 0.330); max(f(u) - u) median 0.236, worst 0.346. Most curved: samba pos, rough neg, muted_horn pos, accelerando neg, lofi_hip_hop pos. Least: sfx_event_rate pos, sitar, woody_body neg, dishes pos (u50 ~0.35, still far from 0.5).
- Per-knob tables needed (one global curve errs 5% median / 19% max of the budget); 11-point grid within 1.9%. Bottom ~15% of the perceptual slider lies below the smallest probe (alpha 2): linear-from-zero assumption, not data.
- Listening list (5 items, dry vs A = perceptual midpoint vs B = shipped gain, seed 0) in s4.md: samba pos, rough neg, accelerando neg, lofi_hip_hop pos, bright neg.

## Q2 + Q4 from lit_review_v3 (2026-10-07, DONE; CLAP on local GPU ~3 min, rest CPU; nothing committed)

- Code: C:\_dev\projects\DEMON-steer-bench\scripts\steering_bench\rescore_independent.py (new, uncommitted; subcommands q2-cpu, q2-clap, q2-sum, q4). Outputs: E:\Projects\DEMON\steering-bench\v3_smoke\q2q4\ (q2_clips.csv, q2_clap.csv, q2.json, q2.md, q4.json, q4.md). Smoke renders reused (1092 clips), no new renders. CLAP pass reproduces the smoke's own CLAP column exactly (same ckpt, 10 s window mean), embedding each clip once.
- Q2 verdict: the ridge up-side "wins" do NOT survive independent scorers except lra.
  - centroid pos (the 2 passes): ridge +1430 Hz (bright control +494) but CLAP bright/dark -0.020 (control +0.121); CLAP "radio static" +0.041 and "tape hiss" +0.059; top-band (>8 kHz) spectral flatness +0.019 where bright goes -0.024. Centroid after removing the stationary per-bin floor still +1420 Hz, so it is not a steady hiss bed but noise-like, non-tonal HF ("fizz"). Reads as metric gaming. ridge_resid_centroid same pattern.
  - onset_rate pos: superflux onsets +1.0/s (rho 0.41, weaker than the target), beat_track tempo flat, CLAP percussive -0.068 and fast tempo -0.149 (wrong way; control +0.193 / +0.087). Does not survive.
  - lra pos: ffmpeg ebur128 LRA (second EBU R128 implementation) +4.4 LU, rho 0.94, agrees with pyloudnorm +5.2; CLAP "dynamic" flat (+0.015), "uncompressed" +0.055 (control +0.046). Survives the second implementation, weak perceptual corroboration.
  - beat_track tempo is noisy here (shifts + for every vector and sign); not used for the verdict.
- Q4 (ship pass, 97 shipped vectors x 2 = 194 knob-signs, 141 bar-passing): own-label moves against the intended sign at the shipped gain:
  - bar-passing: 8/141. cross-effect own z wrong sign (seed 2115): legacy_density neg (+0.13, the dense_arrangement example), legacy_rough neg (+0.25). MuQ effect < 0 on one seed only, all tiny (-0.001 to -0.003; CLAP positive on all seeds): drum_machine neg, solo_instrument neg, spiccato neg, v1_analog pos, v1_koto neg, vaporwave_r2 neg.
  - all 194: 51 (cfx 2, CLAP any seed 14, MuQ any seed 40); the other 43 are bar-failing sides. Consistently inverted on all 4 seeds: CLAP acid_house/hammered_notes/new_age/quantized/shimmer_reverb neg; MuQ explosion/frogs/sfx_far_away/sfx_indoor_hall/v1_analog/v1_jet/v1_phone_ring neg, locked_groove/solo_build/vaporwave_r2 pos. Full list in q4.md.
  - Robust inversions among bar-passing sides = 2 (both legacy, neg side), so the Torop representation-response gate is low priority unless bar-failing sides are exposed in the product.

## S1: per-sign ridge + 3 seeds (2026-10-07, IN PROGRESS, FLAG: GPU contention)

- Packs (CPU, done): E:\Projects\DEMON\steering-bench\v3_smoke\s1\packs\ (ridge = symmetric, cos 1.0 to the first smoke's; ridge_up = mid+high tercile; ridge_down = low+mid tercile; ridge_lowtail = bottom half; fit_report.md). CV R^2 up/down/lowtail: onset 0.42/0.65/0.62, centroid 0.69/0.78/0.76, lra 0.66/0.36/0.25. cos(down, symmetric) only 0.64 / 0.25 / 0.39: the down vectors are nearly new directions.
- Extras for lit-review Q1/Q6 (CPU, done): s1\extras\packs\ (ridge CV lambda, 10x, 100x, X^T y limit per descriptor; block sweep 4,6,8,11,16,22 for centroid and onset_rate fit on the v1 corpus, the only local capture with blocks 4-8; v2 corpus holds 9-19, 22, 23 only). CV R^2 falls with lambda (onset 0.51/0.49/0.40); block sweep centroid b4 0.73 > b22 0.69, onset flat 0.42-0.44.
- Code (uncommitted, additive flags): ridge_directions.py --per-sign, --reg-path, --block-sweep; smoke_drive.py --seed-offsets, --n-test-prompts, --probe-small, --render-chunk, --control-sides, --resume, --no-clap (CLAP left to rescore_independent.py). Dry runs green: s1\dryrun, s1\extras\dryrun.
- Main grid (24 prompts = 12 holdout + first 12 TADA test, seeds 2115-2117, 21 vector-sides) is running with --resume, log s1\smoke_drive.log. Rendering runs 12x slower than the first smoke (SA3 step 10-24 s vs ~1 s) because S2 pairs_s2, S3 stacking_rules and a Q3 smoke_drive --clamp run share the card (30+ GB used, 100 % util). 3 of 21 sides done in ~90 min; ETA at this rate 6-8 h, about 40 min if run alone.
- First process was killed externally at 11:39 (SIGTERM, not by S1); the restart resumes from audios.npz on disk.

## Q3: clamp vs add on the descriptor-down side (2026-10-07, FLAG: E: full)

- Q9 dropped (coordinator). Code check: both bench (sa3_tada.generate cfg_scale=1) and runtime (sa3_stream_helpers cond_bundle cfg_scale=1.0; vendored dit.py runs the batched uncond pass only when cfg_scale != 1) run SA3 medium without CFG; the clamp hook saw batch 12 = prompt batch, so one conditional pass per step. Nothing to split.
- Code (uncommitted, additive, flag-gated) in DEMON-steer-bench scripts/steering_bench/smoke_drive.py: --clamp set|min (per audio token at the pack block output: set <h,u> = t, or min(<h,u>, t); memory tokens / padding untouched; every step), --clamp-pcts (default 50..10 percentile of the per-step corpus projection, music rows of many_knobs_v2 corpus), --clamp-match-add (t = corpus mean projection - |add alpha|, the add's own 5 shifts), --corpus, --compare-results; writes an "add vs clamp" section into results.md and q3_compare.json. Runtime untouched.
- Run 1 (spec: percentile targets, 3 symmetric ridge packs, 12 prompts seed 0, both modes) DONE: E:\Projects\DEMON\steering-bench\v3_smoke\q3\results.md. Clamping to in-corpus targets is nearly inert: LPAPS top 0.05 to 0.20 of the cutoff (add: ~0.98), down deltas centroid -0.25 / -2.0 Hz (add -108), onset +0.05 (add -0.68), lra -0.08 / +0.36 LU (add -1.08); Spearman 0 to 0.75. Cause: add sweeps shift the projection by 15 to 100 units, the corpus projection std is 0.4 to 2.2, so the 10th percentile is a tiny move.
- Independent rescoring (rescore_independent.py q2-cpu + q2-sum, no CLAP yet): same picture. centroid_nofloor add -110 [-0.94] vs clamp set -0.45 / min -2.1; onset_superflux add -0.52 [-0.84] vs clamp -0.025; lra_ffmpeg add -1.47 [-0.35] vs clamp set -0.33 [-0.64] / min -0.29 [-0.84] (monotone but tiny). q2_clips.csv in q3\indep\; q2-sum output in the session scratchpad (E: full).
- Run 2 (--clamp-match-add, equal magnitude to the add) FAILED at 12:5x ET: OSError 28, E: has 5.8 MB free (q3 uses 0.9 GB). On disk: centroid set+min, lra set (complete sweeps), lra min partial. Resume after disk is freed and s1\DONE exists: same command with --resume, out q3\match_add (log has the command line). Then q2-clap + q2-sum on q3\renders and q3\match_add\renders, then scratchpad q3_indep_table.py for the side-by-side.
- Verdict so far: at the review's in-corpus targets, clamping does not fix the down side; it barely steers. Whether clamping at add-matched magnitude beats add is open (run 2).
- Run 2 command (Git Bash, from C:\_dev\projects\DEMON-steer-bench): `PYTHONUTF8=1 TADA_ROOT=E:/Projects/tada-replication /c/_dev/projects/DEMON-tada-sa3/.venv/Scripts/python.exe scripts/steering_bench/smoke_drive.py --clamp set min --clamp-match-add --control-sides -1 --only ridge_onset_rate ridge_centroid ridge_lra --packs E:/Projects/DEMON/steering-bench/v3_smoke/packs --out E:/Projects/DEMON/steering-bench/v3_smoke/q3/match_add --eval-python E:/Projects/tada-replication/evalenv/Scripts/python.exe --tada-root E:/Projects/tada-replication --resume`. Side-by-side script: C:\Users\ryanf\AppData\Local\Temp\claude\C---dev-projects-DEMON-notes-steering-pr\a4a78375-a437-4f1c-bcdd-7d7cbe1a990e\scratchpad\q3_indep_table.py. q3\DONE not written (renders unfinished).

## S3: shared-norm stacking rule (2026-10-07, local 5090, IN PROGRESS)

- Code: C:\_dev\projects\DEMON-steer-bench\scripts\steering_bench\stacking_rules.py (uncommitted; export / plan / run / analyse). Out: E:\Projects\DEMON\steering-bench\v3_smoke\s3\ (plan.json, shifts\, res\seed0\{base,single\*,combo\*}.json, results.md/json).
- Set: the ship pass's 300 combos (ship\d\combos.json) over the 97 shipped vectors, member gains = bundle calibrated_gain median (identical to ship gains.json; cutoff_min identical to the ship stack meta). Rules: none, inv_sqrt (1/sqrt k), inv_n, shared_l2 (norm of summed shift = strictest member's single shift norm), shared_l1 (sum of member shift norms = that budget). Phase A = 40 pairs + 20 triples x 5 rules + each member alone (own-label reference); phase B = remaining 240 combos for the shared rules.
- Memory rule: SA3 load peaks 8.8 GiB (fp32 then half), resident 4.35 GiB; renders run in chunks of 2 prompts under a 6 GiB cap (set_per_process_memory_fraction after load). The first launch (11:22-11:40, batch 12) reserved 12.5 GiB: over the rule; killed (my own process only).
- Scorer deps missing locally, fixed without touching evalenv: CLAP 630k-audioset-best.pt + AudioSet class_labels_indices.csv downloaded to s3\ckpt\, hear21passt + timm 0.4.12 + timbral_models installed with uv --target into s3\pylib\ (worker PYTHONPATH only).
- Reproduction: local chunked renders vs the ship pass LPAPS on the same combos: median |rel diff| 0.04-0.06, corr 0.76-0.97.
- 12:57 ET PAUSED on the coordinator's order (card thrashing); resumed 14:05 on the s2 DONE gate, paused again 14:17 (coordinator: card back at 30 GB). 79 singles + 35 combo-rule results on disk; resumes when s1DONE and q3DONE exist.
- SIDE FINDING (runtime, not fixed): variant-c packs in the shipped bundle (acid_house, new_age, revving, sfx_ocean_waves, sfx_sub_weight, shimmer_reverb, solo_build) carry the neg gain in the dn vector's own magnitude units (provenance.dn_magnitude_own), but SteeringPack._row gives the unit neg row the pack's POS magnitude, so a negative knob at the shipped gain applies pos_mag/dn_mag x the calibrated shift: 3.05x acid_house, 2.97x ocean_waves, 2.77x revving, 1.61x shimmer, 1.45x new_age, 1.15x sub_weight, 0.23x solo_build. S3 stacks the calibrated (dn-magnitude) shift.
