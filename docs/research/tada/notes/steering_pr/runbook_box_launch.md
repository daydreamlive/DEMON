# Runbook: 4-GPU benchmark of the SA3 steering knobs on vast.ai 51480126 (2026-10-05)

Inputs (read once): box_status.md (ssh, paths, envs), shim_status.md (bench CLI per concept, files to copy),
directions_status.md (capture, build_directions, block sweep CLIs). Plan: plan_measure_and_improve_2026-10-05.md.
Status file for this run: status_box_run.md (this folder). Results: results_2026-10-05.md (this folder).
Box output root: /root/out/steer-bench/<phase>/... . Copy final CSVs and the summary back to
E:\Projects\steering-bench\2026-10-05\ (not into any repo).

Concepts: bright, warm, percussive, rough, density. Packs on the box in the packs dir from box_status.md.

## Phase A (all four GPUs at once, start immediately)

GPU0: measure current packs bright, warm, percussive (sequential). Per concept: probe (20 holdout x 13 alphas,
      LPAPS only) -> calibrate -> sweep (50 prompts, --points 7) -> PCI-all (--pci-ks 2 4 6 8) -> score
      (LPAPS + MuQ, no aesthetics, no CLAP) -> descriptor score -> AUC on MuQ and on every descriptor column.
GPU1: same for rough, density. Then idle until Phase B.
GPU2: capture_resid.py on all 5521 MusicCaps captions (about 9 min). Then build_directions.py (class-mean, CPU).
GPU3: capture_resid.py --self-label N=800 (about 3 min), then build_directions.py (quartile, CPU, needs the
      descriptors on the saved audio). Then, as soon as directions exist: per-block sweep (24 blocks, 20 holdout
      prompts, 5 alphas, LPAPS + descriptor) for bright, warm, percussive on GPU3 and rough, density on GPU2
      (GPU2 picks these up after its own build finishes). Use the self-label quartile directions for the sweep
      (one estimator only, to save time); record the class-mean sweep only if time allows.

Rules: one generator plus at most one scorer per GPU. Keep PCI eval dirs: Phase B reuses them (same prompts,
seeds, cutoff). Every job writes a log under /root/out/steer-bench/logs/<job>.log. The status file gets one line
per job on launch, finish or failure; nothing on timers.

## Phase B (after Phase A measurement and sweeps are done)

For each concept, build candidate packs: (1) self-label quartile at the production block, (2) self-label
quartile at the best sweep block, (3) MusicCaps class-mean at the best sweep block, (4) all-blocks per-step
vectors (bench --site all with the per-step vector file). Score each with the full Phase A protocol, reusing
the PCI dirs (probe + calibrate + sweep + score only). 20 runs, about 8 min each, spread over 4 GPUs by
concept, longest-first. If the box has fewer than 4 GPUs or time runs short, drop (3) first, then (4).

## Phase C (CPU, while B runs)

From Phase A sweeps: knob -> descriptor curve per sign (monotone? symmetric?), alpha at cutoff mapped to knob
units (alpha / magnitude), 5x5 cross-effect matrix, per-genre spread. Also the MuQ separation gate on the PCI
endpoints per concept (does MuQ separate real pos vs neg prompts?).

## Results table (results_2026-10-05.md)

Per concept, per sign, per pack variant: AUC(descriptor)/PCI, AUC(MuQ)/PCI, LPAPS cutoff, alpha at cutoff,
knob value at cutoff, best block, n clips. Plus the cross-effect matrix and the MuQ gate. State the caveats:
one seed, 50 prompts, ratio differences under 0.1 are noise; brightness register-confounded; roughness
descriptor weak.

## Failure handling

A job that fails twice is recorded as FAILED with the last 20 log lines in the status file and skipped; the
other jobs continue. OOM: retry once with half the batch. Never rm on the box or locally (mv to a trash dir).
Never push anything. Hand back to the parent only when every phase is terminal (done or failed), with at most
10 lines: results path, the headline numbers per knob, failures.

## Phase D (only if Phases A to C are done and GPUs are idle): the next obvious experiment, chosen by the numbers

Decision rule, in order:
1. If any re-estimated pack (Phase B) beats the current pack by more than 0.1 ratio on a knob, rerun that
   winner AND the current pack with a second seed (EVAL_SEED + 1) on the same 50 prompts, so the gain is
   confirmed above noise. About 8 min per run.
2. If the self-label quartile vectors won, run the per-block sweep with the MusicCaps class-mean directions too
   (the sweep skipped it), to see whether the block choice agrees.
3. If bright's centroid effect disappears once the HF-energy ratio is used, score with the register control
   (centroid of the HPSS harmonic part) to separate pitch from brightness.
4. CPU, any time: fit a linear readout per concept from the capture (activation mean at the pack's block ->
   measured descriptor on the self-label audio), report R^2 per block per step. That is the P5 gate.
Record every Phase D run in the results file with the rule number that triggered it.
