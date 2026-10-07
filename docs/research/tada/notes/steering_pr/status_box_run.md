# Box run status: vast 51480126, SA3 steering bench (2026-10-05)

Outputs: /root/out/steer-bench -> /dev/shm/steerbench/out/steer-bench (/workspace = same overlay as /, 3.8 GB free; shm 25 GB).
Trash: /dev/shm/steerbench/steer-bench-trash (same disk: moving audio there frees nothing). Box event log: $OUT/events.log.

- 15:20Z setup: 13 spike files (974ab049..f75c0563) untarred over /root/DEMON-bench, all parse, --method pack present.
- 15:24Z GPU0 phaseA bright,warm,percussive LAUNCHED; GPU1 phaseA rough,density LAUNCHED; GPU2 capture_mc LAUNCHED; GPU3 capture_self LAUNCHED.
- 15:24Z GPU2 capture_mc FAILED (no pandas in demonenv). 15:27Z GPU0/1 score_pci FAILED (MuQ under transformers 5.17: EasyDict config lacks _attn_implementation, then no hidden_states). Box thread cap (pids.max 94208) hit by build_self workers (510 threads each).
- 15:30Z fixes: pandas 3.0.6 into /root/steerbench_exec/site (shm is noexec); evalenv gets transformers 4.47.1 + tokenizers 0.21.0 + hub 0.27.1 (the pin) prepended from /root/steerbench_exec/evalsite (MuQ text+audio verified); OMP/MKL/BLAS threads capped at 8. Partial PCI protocol_results moved to trash.
- 15:33Z RELAUNCHED all four (probe/PCI audio and capture_self reused).
- 15:36Z GPU3 build_self DONE (pack blocks 23/15/2/1/15; cos to prod +0.74/+0.67/-0.01/+0.42/+0.59). GPU3 first-block check (bright b23 + prod, K=4) LAUNCHED.
- 15:38Z GPU3 first-block check DONE: K=4 at bright b23 gives LPAPS 3.2-3.5 at half and 4.4-5.0 at max (PCI-all cutoff 4.2-4.3): sane, K kept at 4 (no grid change). sweep_blocks.sh fix on the box copy: logs moved to $OUT/logs/<sub> (the logs dir inside the sweep sub crashed the scorer's dir parser).
- 15:38Z alpha 0 renders verified bit-identical across all dirs (holdout-20 and test-50); later dirs hard-link alpha_0.0 from $OUT/shared to save disk.
- 15:39Z DISK: full plan needs ~35-40 GB of audio; shm has 18 GB and the trash is on the same tmpfs (moving frees nothing). Budget to the 10 GB floor covers Phase A + Phase B variant 1 only.
- 15:39Z GPU3 Phase B v1 (self-label quartile @ production block) rough, bright LAUNCHED.
- 15:40Z GPU2 capture_mc + build_mc DONE (MusicCaps class-mean; cos to prod +0.40/+0.32/+0.46/-0.09/+0.49 bright/warm/rough/density/percussive). GPU2 Phase B v1 warm, density LAUNCHED.
- 15:43Z GPU1 Phase A rough, density DONE. 15:46Z GPU3 Phase B v1 rough, bright DONE.
- 15:47Z GPU3 Phase B v1 percussive LAUNCHED (shm 11 GB free; this run takes it to ~10.2 GB, the last launch the floor allows).
- 15:49Z GPU0 Phase A bright, warm, percussive DONE. 15:50Z GPU2 v1 warm, density DONE. 15:53Z GPU3 v1 percussive DONE.
- 15:55Z AUC: reference auc needed lpaps_endpoints.csv (ran `sa3_tada_score.py cutoff`); descriptor auc needs audio present, so it ran with trashed audio hard-linked back, and the links went back to trash/relinks after.
- 15:58Z Phase C DONE (box CPU, $OUT/phaseC). 16:01Z Phase D rule 4 readout DONE (CPU, $OUT/phaseD).
- 15:52Z STOPPED at the disk floor: shm 10.4 GB free, trash 8.3 GB (same tmpfs). NOT RUN: per-block sweeps (only the b23 check), Phase B v2-v4, Phase D rules 1-2 (triggered). Emptying the trash needs the user's OK.
- 16:03Z CSV/JSON summaries copied to E:\Projects\steering-bench\2026-10-05\ (463 files + box_summaries.tgz). No jobs running on the box.
- 16:10Z coordinator: floor lowered to 3 GB, no deletes. GPU0 Phase D rule 1 bright seed 2116 (current + v1, seed-1 ranges/PCI) LAUNCHED.
- 16:14Z GPU0 rule 1 bright seed 2116 DONE (current + v1). 16:15Z GPU0 bright 24-block sweep (resid_self, K=4) LAUNCHED (shm 9.2 GB free).
- 16:30Z GPU0 bright 24-block sweep DONE (first launch FAILED after b00: blocks.sh passed a newline-separated block list; fixed and relaunched 16:20Z; renders reused).
- 16:31Z PURGE NOT DONE. The coordinator relayed a purge authorisation, but the user's standing rule needs their own explicit OK for any permanent delete, and an agent relay is not that. Also a technical blocker: /workspace is the same overlay as / (3.8 GB free), so moving the 8+ GB shm trash to /workspace/trashcan would be a cross-filesystem copy that fills the root disk shared with the other lanes. Nothing was moved or purged.
- 16:31Z DISK BUDGET (no purge, 3 GB floor): block sweep 1.6 GB per concept (23 new blocks x 80 clips x 0.88 MB; alpha 0 hard-linked); Phase B run 0.83 GB; seed-2 sweep 0.6 GB. shm 7.5 GB free now, so warm + percussive block sweeps fit (peak use about 3.3 GB, leaving about 4.2 GB). rough/density blocks, Phase B v2-v4 (about 12 GB) and further Phase D runs do not fit.
- 16:31Z GPU1 warm and GPU2 percussive 24-block sweeps (resid_self, K=4) LAUNCHED.
- 16:35Z bright blocks b01-b22 descriptors RESCORED (stale CSVs from the failed first launch). 16:20Z timbral hardness + 2 onset detectors scored on CPU (phaseD/indep_desc.csv).
- 16:44Z GPU1 warm and GPU2 percussive 24-block sweeps DONE. shm 4.2 GB free: rough/density sweeps (1.6 GB each) would cross the 3 GB floor, so they were NOT launched. Nothing is running on the box. Summaries copied to E:\Projects\steering-bench\2026-10-05\ (box_summaries_2.tgz, extracted).
- 16:48Z RESUME after the user's own purge (shm 18.5 GB free, floor 3 GB; scored audio still goes to steer-bench-trash on the same tmpfs). Budget: 0.83 GB per 50-prompt run, 1.6 GB per block sweep, 0.6 GB per seed-2 sweep; the queue (items 1-6) needs about 19 GB, so expect the floor during item 5 or 6.
- 16:48Z GPU0 warm current-vector@b23 and GPU1 warm v1@b23 (full protocol, PCI reused) LAUNCHED; GPU2 rough and GPU3 density 24-block sweeps LAUNCHED.
- 16:54Z GPU0/1 warm b23 runs DONE. GPU0 queue (seed-2 percussive+density current, then v2 bright b17) and GPU1 queue (seed-2 percussive+density v1, then v2 percussive b17) LAUNCHED. v2 blocks = best d/lp from the sweeps (bright b17, percussive b17; warm b23 = item 1).
- 17:03Z GPU2 rough and GPU3 density block sweeps DONE (best d/lp: rough b22-b23 on self-label flatness, density b11; production b2 / b1). Queues LAUNCHED: GPU2 v2 rough b23, v3 rough b23, v3 bright b17; GPU3 v2 density b11, v3 density b11, v3 percussive b17; GPU0 v3 warm b23 after its seed-2 queue. v4 waits for v3.
- 17:18Z seed-2 percussive/density (current + v1) DONE; v2 bright b17, percussive b17, rough b23, density b11 DONE; v3 (MusicCaps @ best block) all five DONE. shm back up to 13 GB (freed outside this run). 17:19Z GPU0-3 v4 (per-step vectors, all 24 blocks, post_block_residual) LAUNCHED for all five.
- 17:28Z GPU0-3 v4 DONE (all five). 17:30Z AUC recomputed for all new subs (no-audio variant for 5 subs); phaseC/all_variants.csv written. All CSV/JSON copied to E:\Projects\steering-bench\2026-10-05\ (box_summaries_3.tgz, extracted). Floor never hit. Nothing running on the box.
