# Lane SAE-SA3 (TADA SAE on Stable Audio 3 medium)

- 2026-10-03 started; worktree C:\_dev\projects\DEMON-tada-sae-sa3 branch ryanontheinside/feat/tada-sae-sa3 off tada-sa3 868df565 (already contains CORE READY 1e90846c and the SA3 cross_attn_output hook). Rebase onto the final tada-sa3 HEAD at the end.
- STATE: step 1 (CPU port of TADA's SAE pieces) in progress. Waiting on status_tada_sa3.md for localized blocks + CAA packs.
- Step 1 DONE: f6f980ba feat(tada) acestep/tada/sae (cache, model, train, scoring, packs), tests/unit/test_tada_sae.py 10 pass (30 with core + sa3 hooks). Branch now on tada-sa3 01cfc1c7.
- MusicCaps captions (5521, CC BY-SA 4.0) at E:/Projects/tada-replication/data/musiccaps-public.csv.
- Next: SA3 driver script (CPU), then wait for localized blocks in status_tada_sa3.md before the timed GPU batch.
- Driver scripts/tada/sa3_sae.py (time / cache / train / score; uncommitted, lands with step 2). CPU smoke of train on a synthetic cache passes. SA3 setting: cfg 1, SA3 sampler, steps = SA3 lane's CAA steps (default 8), cache every 2nd step, 10 s, MusicCaps 5521 captions shuffled seed 42.
- WAITING: localized blocks in status_tada_sa3.md; GPU lock (held by sa3-tada).
- Coordinator resume: driver committed ee77d5e2. Waiting (one loop) for the SA3 localize line; then timed batch + epoch under the lock protocol, 6 GPU-h budget gate.
- Coordinator correction: no GPU-hour cap; full run proceeds after the timed batch; ETA line follows the measurement.
- Localize line was a plan line; SA3 lane runs the GPU continuously until its packs (~8 h plan). Waiting (one loop) for its CAA packs written + lock back to user-demo-session.
- Outage: agent stopped by account usage limit ~18:00, resumed 18:21. Nothing was running for this lane (waiting only). Following 19_tada_overnight.md: lock after SA3 DONE or SA3 packs written + lock released (SA3 ETA ~01:00).
- 01:10 SA3 DONE 23e98b84 seen (blocks 3,6,7). GPU_LOCK taken by sae-sa3. Branch rebased onto 23e98b84 (HEAD b45f6eef).
- Timing (measured, E:/Projects/tada-replication/sae_time.json): seq 238 tokens x 1536, 4 samples/prompt (steps 0,2,4,6; sigma 1.0/0.98/0.89/0.51). One warm caching batch of 16 prompts, 3 blocks: 3.80 s. Training: 60 steps of the whole 12-config (m,k) sweep in 10.6 s (0.176 s/step), 1299 steps/epoch -> 0.064 h per block-epoch for the whole sweep. Cache 16.1 GB per block.
ETA: (01:15) cache 5521 captions 0.36 h (~01:40); train sweep 3 blocks x 10 epochs 1.9 h (~03:40); score 9 concepts ~0.1 h; alpha calibration + held-out k_c selection ~2 h; eval vs CAA (50 prompts x 21 strengths x 9, SA3 lane's protocol) ~1.5 h; packs/ear/docs 0.3 h. Total ~6.2 GPU h, SAE DONE ~07:30.
- 01:11 Coordinator resequencing: SA3 lane runs E1 first. Full caching run (started 01:16) stopped before its first shard; timing kept (training measured as a 60-step slice of the whole sweep at 0.176 s/step, extrapolated per epoch above). GPU_LOCK returned to user-demo-session. Waiting (one loop) for `SA3 DONE2` and its MECHANISM: line.
- CPU prep while waiting: sa3_sae.py 'vectors' (pooled + per-step v_SAE per k_c, written where sa3_tada_run.py reads vectors); plan: SAE method in the SA3 lane's driver, same eval50 protocol, PCI cutoffs, calibrate, k_c on 20 holdout x 9 strengths.
- 04:07 SA3 DONE2 983361a4, MECHANISM guidance=1 renorm=no (SAE eval uses the same). Lock taken; rebased (HEAD 53ba2c00); full cache started.
- 04:09 Coordinator hold (E1 negative; SA3 lane runs E2 site search). Cache run stopped before its first shard; timing + ETA already recorded above. GPU_LOCK returned. Waiting (one loop) for SA3 DONE3 and its SITE: line; the full pipeline will run at that site.
- 08:47 E3 corrections done on CPU: 92513262 (audio tokens only via SA3AudioTokens: 64 memory rows + padding excluded from cache and feature means; MuQ primary for all SAE decisions). Tests 13 SAE (38 with core + sa3 hooks). Waiting (one loop) for SA3 DONE4 + SITE: line.
- 09:17 Coordinator correction applied (c: served ARC medium checkpoint, same 8-step CFG-free render as CAA; cache all 8 steps; sigma indexed in cache, scores, vectors, packs): 813c819a. Timing re-scaled: 8 samples/prompt (x2 data): cache ~0.4 h, train ~3.6 h, ETA SAE DONE ~4 h after DONE4 + eval. Still waiting for SA3 DONE4.
- 12:40 SAE GO seen (SITE: cross_attn_output, blocks 3,5,6,7). GPU_LOCK taken (sae-sa3). Branch rebased onto tada-sa3 f46a1900. E: has 21 GB free, so the heavy outputs (cache ~23 GB/block, SAEs, score tables, SAE eval audio) go to D:/tada-replication; packs and the ear package stay on E:. Cache running (steer_logs/sae_e3_cache.sh), measured 1.7 s per 16-prompt batch, ~10 min for 5521 captions x 8 steps x 4 blocks.
- Plan after cache: train (m,k) sweep per block, 10 epochs, lr 3e-5 -> score (TF-IDF, 9 concepts, seed 10) -> per-step vectors for k_c in {5,10,20,50,100,500} -> held-out probes (20 prompts) scored with the reference protocol -> k_c by AUC (MuQ; CLAP for vocal gender) -> calibrate range to the eval50 PCI cutoffs -> eval50 sweep (50 prompts, 10 strengths per side, renorm off) -> SAE/PCI ratios -> packs -> ear package.
- 12:51 Cache done (5521 captions, 573 s, 89 GB on D:, 41960 train / 2208 held-out samples per block, seq 172 audio tokens). Training: 0.197 s per step for the whole 12-config sweep, 18240 steps per block (10 epochs) = ~1.0 h per block. ETA: train ~16:55, score+vectors ~17:05, held-out probes+scoring ~18:15, eval50 sweep+scoring ~19:30, SAE DONE ~20:00.
- 13:55 Block 3 trained (lr 3e-5, 10 epochs). Held-out FVU all-steps: m2_k32 0.392, m4_k64 0.319, m8_k64 0.327, m16_k32 0.408; dead <= 2.2%; fire 0.09-0.92. Absolute per-sigma bucket check FAILS for every config: steps 0-5 (sigma 1.0-0.75) FVU 0.34-0.47, step 6 (sigma 0.51) ~0.70, step 7 (sigma 0.27) 1.02-1.31 (worse than the bucket mean). S1 triggered: retraining block 3 once at lr 1e-5 (sae_s1_lr1e-5, concurrent with blocks 5-7 at 3e-5) to see whether lr is the cause before spending 4 h on all blocks.
- 14:00 Cause measured from the cache (block 3, 320 prompts): per-step token variance falls ~67x from step 0 to step 7 (within-step var 33.5 at sigma 1.0, 4.8 at 0.75, 1.06 at 0.51, 0.50 at 0.27; rms 6.05 -> 0.74), so the MSE objective is dominated by the early steps and the late-step buckets are reconstructed worse than their own mean. Scale effect of the data, not of lr; the lr 1e-5 retrain checks it.
- 14:35 S1 retrain stopped at step 2500 (running it beside the main sweep cost 0.57 s/step each vs 0.197 alone); it reruns alone for block 3 after the eval GPU work, as the recorded S1 check. Since no config passes the bucket gate, the SAE per block falls back to the paper criterion alone (lowest held-out FVU with dead <= 1%, fire <= 25%): block 3 -> m16_k32 (FVU 0.408, dead 0.44%, fire 17.5%). Commit 0d9050e1.
- 15:25 Coordinator changes applied: (1) vocal_gender and vocal_style skipped (SA3 medium generates no vocals; "not generable by SA3, excluded"): 7 concepts in scoring, selection, calibration, eval50. (2) k=128 added (m in {4,8,16} x k 128) for every block, a protocol deviation: the paper grid {16,32,64} was tuned on ACE-Step; the prior SA3 SAE lane (beyond/prior_sae_work.md 2b, 2c) measured FVU falling almost linearly in k on this model (k 32 -> 0.41, k 128 -> 0.16-0.19) and k 192 diverging in the two lowest-sigma buckets. Selection unchanged (paper criterion + absolute per-sigma gate, fallback to the paper criterion alone). (3) S1 lr 1e-5 retrain dropped; replaced by the per-step RMS normalized control at block 7 after eval50 is scored (held-out tempo/piano/mood only). (4) Watching for SITE2 in status_tada_sa3.md before scoring.
- Block 5 (lr 3e-5, 10 epochs): val FVU m2_k32 0.294, m4_k64 0.237, m16_k32 0.301, m16_k64 0.245; step 7 bucket 0.68-0.75, gate fails again. Block 6 started 15:24 (training chain steer_logs/sae_e3_train2.sh, merges into sweep.json).
ETA (revised 15:25): blocks 6,7 paper grid ~17:10; k=128 x3 for 5,6,7,3 ~19:35; score + held-out probes + k_c selection ~20:50; eval50 sweep + scoring + report/packs/ear ~22:10 -> SAE DONE; RMS control ~23:30.
PAUSED 15:46, block 6 killed mid-sweep (step ~3500 of 18240, paper grid), resume point block 6 from scratch (then 7, then k=128 x {4,8,16} for 5,6,7,3; then the 7-concept post chain steer_logs/sae_e3_post.sh). Blocks 3 and 5 (paper grid) saved in D:/tada-replication/sae/sa3/sweep.json; cache kept. SITE2 posted blocks 0,1,3,5,6,7: on resume, cache blocks 0 and 1 too (~10 min) and train them. GPU_LOCK returned.
SAE STOP (parent, 17:00): base test refutes distillation (tempo 1.07, piano -1.08, mood 0.05). No further SAE training. See 19_tada_overnight.md "E4 probe" for the CPU task that follows.
- 17:00 SAE STOP received. Trained (lr 3e-5, 10 epochs, paper grid, ARC medium, 8 steps): block 3 val FVU m2_k32 0.392 / m4_k64 0.319 / chosen-by-paper-criterion m16_k32 0.408; block 5 m2_k32 0.294 / m4_k64 0.237 / m16_k32 0.301; every config fails the absolute per-sigma gate at sigma 0.27 (FVU 0.68-1.31). Blocks 6, 7 and k=128 not trained. No GPU_LOCK held.
SAE FLAG paused by coordinator after base test

## E4 PROBE (CPU, 17:11; scripts/tada/sa3_e4_probe.py, D:/tada-replication/e4_probe.json)
- Cache: ARC medium, 5521 MusicCaps captions x 8 steps, cross_attn_output, audio tokens only. Keyword labels (balanced): piano 751/class (751 pos, 3488 neg), tempo 332/class (652 pos, 332 neg), mood 184/class (463 pos, 184 neg). Piano neg = no piano/keyboard/synth word.
- Mean level: L2 logistic probe on the token mean, 5-fold by caption; shuffled-label control beside it. Token level: same probe on ~20k tokens per class, folds grouped by caption. Effect = class-mean gap of the projection on the mean difference over pooled within-class std. cos CAA = cosine of the mean difference to caa_e3 (same concept/block/step).
- Caveat: keyword labels co-vary with other caption content (genre, instrumentation), so a probe hit means the site linearly separates these caption groups, not that it isolates the concept alone.

| concept | block | step | mean acc | shuffled | token acc | effect | cos CAA |
|---|---|---|---|---|---|---|---|
| piano | 3 | 0 | 0.887 | 0.525 | 0.868 | 1.48 | +0.249 |
| piano | 3 | 1 | 0.886 | 0.511 | 0.874 | 1.47 | +0.232 |
| piano | 3 | 2 | 0.878 | 0.515 | 0.867 | 1.48 | +0.226 |
| piano | 3 | 3 | 0.871 | 0.517 | 0.865 | 1.45 | +0.206 |
| piano | 3 | 4 | 0.860 | 0.511 | 0.864 | 1.43 | +0.180 |
| piano | 3 | 5 | 0.858 | 0.515 | 0.865 | 1.42 | +0.140 |
| piano | 3 | 6 | 0.883 | 0.505 | 0.881 | 1.53 | +0.138 |
| piano | 3 | 7 | 0.903 | 0.508 | 0.892 | 1.65 | +0.189 |
| tempo | 3 | 0 | 0.982 | 0.428 | 0.977 | 3.07 | +0.753 |
| tempo | 3 | 1 | 0.979 | 0.449 | 0.973 | 3.01 | +0.753 |
| tempo | 3 | 2 | 0.983 | 0.434 | 0.976 | 3.06 | +0.745 |
| tempo | 3 | 3 | 0.982 | 0.438 | 0.971 | 3.02 | +0.736 |
| tempo | 3 | 4 | 0.983 | 0.417 | 0.968 | 2.92 | +0.712 |
| tempo | 3 | 5 | 0.973 | 0.425 | 0.960 | 2.76 | +0.685 |
| tempo | 3 | 6 | 0.962 | 0.479 | 0.950 | 2.69 | +0.622 |
| tempo | 3 | 7 | 0.947 | 0.483 | 0.941 | 2.57 | +0.530 |
| mood | 3 | 0 | 0.970 | 0.462 | 0.956 | 3.26 | +0.549 |
| mood | 3 | 1 | 0.970 | 0.459 | 0.955 | 3.21 | +0.541 |
| mood | 3 | 2 | 0.967 | 0.462 | 0.955 | 3.21 | +0.533 |
| mood | 3 | 3 | 0.965 | 0.449 | 0.952 | 3.19 | +0.517 |
| mood | 3 | 4 | 0.965 | 0.446 | 0.947 | 3.07 | +0.464 |
| mood | 3 | 5 | 0.956 | 0.449 | 0.933 | 2.90 | +0.403 |
| mood | 3 | 6 | 0.937 | 0.465 | 0.920 | 2.75 | +0.283 |
| mood | 3 | 7 | 0.924 | 0.449 | 0.910 | 2.58 | +0.142 |
| piano | 5 | 0 | 0.907 | 0.488 | 0.891 | 1.29 | +0.062 |
| piano | 5 | 1 | 0.905 | 0.490 | 0.886 | 1.30 | +0.072 |
| piano | 5 | 2 | 0.899 | 0.502 | 0.884 | 1.30 | +0.057 |
| piano | 5 | 3 | 0.905 | 0.497 | 0.888 | 1.27 | +0.070 |
| piano | 5 | 4 | 0.893 | 0.477 | 0.881 | 1.21 | +0.034 |
| piano | 5 | 5 | 0.904 | 0.483 | 0.886 | 1.16 | +0.018 |
| piano | 5 | 6 | 0.905 | 0.503 | 0.890 | 1.10 | -0.034 |
| piano | 5 | 7 | 0.909 | 0.497 | 0.902 | 1.14 | -0.025 |
| tempo | 5 | 0 | 0.968 | 0.535 | 0.955 | 2.32 | +0.429 |
| tempo | 5 | 1 | 0.958 | 0.527 | 0.957 | 2.34 | +0.432 |
| tempo | 5 | 2 | 0.967 | 0.551 | 0.952 | 2.34 | +0.414 |
| tempo | 5 | 3 | 0.962 | 0.539 | 0.954 | 2.45 | +0.424 |
| tempo | 5 | 4 | 0.958 | 0.514 | 0.949 | 2.58 | +0.422 |
| tempo | 5 | 5 | 0.950 | 0.507 | 0.938 | 2.24 | +0.382 |
| tempo | 5 | 6 | 0.938 | 0.500 | 0.927 | 2.03 | +0.367 |
| tempo | 5 | 7 | 0.934 | 0.488 | 0.926 | 2.32 | +0.409 |
| mood | 5 | 0 | 0.948 | 0.489 | 0.930 | 2.08 | +0.468 |
| mood | 5 | 1 | 0.943 | 0.497 | 0.931 | 2.03 | +0.462 |
| mood | 5 | 2 | 0.927 | 0.484 | 0.923 | 2.04 | +0.460 |
| mood | 5 | 3 | 0.921 | 0.476 | 0.919 | 2.05 | +0.431 |
| mood | 5 | 4 | 0.918 | 0.462 | 0.907 | 2.00 | +0.355 |
| mood | 5 | 5 | 0.916 | 0.448 | 0.885 | 1.69 | +0.236 |
| mood | 5 | 6 | 0.891 | 0.492 | 0.881 | 1.52 | +0.089 |
| mood | 5 | 7 | 0.888 | 0.486 | 0.887 | 1.61 | +0.097 |
| piano | 6 | 0 | 0.921 | 0.498 | 0.892 | 1.75 | +0.230 |
| piano | 6 | 1 | 0.925 | 0.496 | 0.895 | 1.76 | +0.243 |
| piano | 6 | 2 | 0.926 | 0.502 | 0.900 | 1.78 | +0.233 |
| piano | 6 | 3 | 0.924 | 0.488 | 0.896 | 1.78 | +0.192 |
| piano | 6 | 4 | 0.909 | 0.504 | 0.888 | 1.71 | +0.074 |
| piano | 6 | 5 | 0.907 | 0.499 | 0.877 | 1.67 | -0.051 |
| piano | 6 | 6 | 0.890 | 0.511 | 0.863 | 1.58 | -0.090 |
| piano | 6 | 7 | 0.868 | 0.509 | 0.849 | 1.47 | -0.039 |
| tempo | 6 | 0 | 0.934 | 0.474 | 0.907 | 2.67 | +0.576 |
| tempo | 6 | 1 | 0.929 | 0.465 | 0.914 | 2.62 | +0.553 |
| tempo | 6 | 2 | 0.941 | 0.494 | 0.912 | 2.56 | +0.543 |
| tempo | 6 | 3 | 0.941 | 0.480 | 0.915 | 2.53 | +0.562 |
| tempo | 6 | 4 | 0.946 | 0.489 | 0.905 | 2.47 | +0.554 |
| tempo | 6 | 5 | 0.911 | 0.509 | 0.891 | 2.37 | +0.519 |
| tempo | 6 | 6 | 0.905 | 0.468 | 0.890 | 2.15 | +0.525 |
| tempo | 6 | 7 | 0.926 | 0.464 | 0.913 | 2.16 | +0.598 |
| mood | 6 | 0 | 0.910 | 0.519 | 0.905 | 2.43 | +0.414 |
| mood | 6 | 1 | 0.916 | 0.503 | 0.907 | 2.35 | +0.413 |
| mood | 6 | 2 | 0.913 | 0.503 | 0.911 | 2.33 | +0.398 |
| mood | 6 | 3 | 0.905 | 0.533 | 0.897 | 2.31 | +0.383 |
| mood | 6 | 4 | 0.899 | 0.522 | 0.891 | 2.19 | +0.323 |
| mood | 6 | 5 | 0.899 | 0.508 | 0.878 | 1.95 | +0.249 |
| mood | 6 | 6 | 0.899 | 0.470 | 0.878 | 1.94 | +0.180 |
| mood | 6 | 7 | 0.889 | 0.473 | 0.880 | 2.03 | +0.130 |
| piano | 7 | 0 | 0.907 | 0.481 | 0.843 | 1.52 | +0.115 |
| piano | 7 | 1 | 0.907 | 0.495 | 0.846 | 1.44 | +0.096 |
| piano | 7 | 2 | 0.903 | 0.484 | 0.838 | 1.45 | +0.092 |
| piano | 7 | 3 | 0.894 | 0.485 | 0.828 | 1.43 | -0.056 |
| piano | 7 | 4 | 0.913 | 0.509 | 0.835 | 1.54 | -0.009 |
| piano | 7 | 5 | 0.947 | 0.503 | 0.846 | 1.74 | +0.168 |
| piano | 7 | 6 | 0.957 | 0.507 | 0.847 | 1.78 | +0.142 |
| piano | 7 | 7 | 0.960 | 0.506 | 0.851 | 1.86 | +0.380 |
| tempo | 7 | 0 | 0.902 | 0.504 | 0.884 | 1.32 | +0.231 |
| tempo | 7 | 1 | 0.898 | 0.522 | 0.889 | 1.38 | +0.279 |
| tempo | 7 | 2 | 0.896 | 0.498 | 0.882 | 1.34 | +0.225 |
| tempo | 7 | 3 | 0.896 | 0.513 | 0.890 | 1.48 | +0.403 |
| tempo | 7 | 4 | 0.904 | 0.483 | 0.895 | 1.43 | +0.402 |
| tempo | 7 | 5 | 0.898 | 0.500 | 0.889 | 0.97 | +0.279 |
| tempo | 7 | 6 | 0.895 | 0.497 | 0.895 | 0.77 | +0.117 |
| tempo | 7 | 7 | 0.899 | 0.501 | 0.902 | 0.79 | +0.063 |
| mood | 7 | 0 | 0.807 | 0.440 | 0.822 | 1.27 | +0.127 |
| mood | 7 | 1 | 0.818 | 0.449 | 0.820 | 1.24 | +0.108 |
| mood | 7 | 2 | 0.812 | 0.457 | 0.824 | 1.27 | +0.101 |
| mood | 7 | 3 | 0.831 | 0.440 | 0.830 | 1.32 | +0.178 |
| mood | 7 | 4 | 0.826 | 0.448 | 0.824 | 1.45 | +0.131 |
| mood | 7 | 5 | 0.840 | 0.440 | 0.833 | 1.25 | +0.011 |
| mood | 7 | 6 | 0.845 | 0.440 | 0.847 | 1.03 | +0.037 |
| mood | 7 | 7 | 0.856 | 0.470 | 0.857 | 1.04 | +0.045 |

- piano: best mean-level 0.960 (block 7 step 7, effect 1.86, cos CAA +0.38); best token-level 0.902 (block 5 step 7); mean-level acc at every block/step 0.86-0.96; cos CAA -0.09 to +0.38.
- tempo: best mean-level 0.983 (block 3 step 2, effect 3.06, cos CAA +0.74); best token-level 0.977 (block 3 step 0); mean-level acc at every block/step 0.89-0.98; cos CAA +0.06 to +0.75.
- mood: best mean-level 0.970 (block 3 step 0, effect 3.26, cos CAA +0.55); best token-level 0.956 (block 3 step 0); mean-level acc at every block/step 0.81-0.97; cos CAA +0.01 to +0.55.
- Read-out (runbook item 3): mean-level accuracy >= 0.85 for all three concepts (shuffled ~0.5): a linear mean-level direction exists at the site; the CAA vectors are only partly aligned with it (cosines well below 1). Per the runbook this points at steering mechanics / direction, next test = learned affine map (AcT-style). No GPU used.
- 17:14 E4b: caa_e4 written (E:/Projects/tada-replication/sa3/caa_e4/{piano,tempo,mood}.pt, caa_e3 layout, steps 0-7 x blocks 3,5,6,7 only: use with site loc; e4_meta.json). Balanced classes per step: piano 751/751, tempo (fast/slow) 332/332, mood (happy/sad) 184/184. Mean cosine to caa_e3: piano 0.11, tempo 0.48, mood 0.29. Commit d9807108.
E4 VECTORS READY
- 17:45 E5 (a): 24-block per-step audio-token means captured for 5521 captions (D:/tada-replication/e5_means, checked against the SAE cache: cos 1.0 at block 3). GPU_LOCK returned.
- 17:46 E5 (b,c): caa_e5 at E:/Projects/tada-replication/sa3/caa_e5 (7 concepts x 8 steps x 24 blocks, caa_e3 layout, --sites all and loc work; e5_meta.json). Per class: tempo 332, piano 751, mood 184, violin 245, guitar_electronic 444, electronic_music 375, rock_genre 173. Mean cos to caa_e3 at 3,5,6,7: tempo 0.48, piano 0.13, mood 0.29, violin 0.55, guitar 0.39, electronic 0.43, rock 0.49. caa_e5p (tempo/piano/mood, blocks 3,5,6,7 only, unit L2-logistic weight in activation space; e5p_meta.json): cos to caa_e5 0.47/0.39/0.47, to caa_e3 0.27/0.09/0.16. Commit d5f4c63f.
E5 VECTORS READY
