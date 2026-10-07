# Many-knobs v2 runbook (REVISED 22:40Z 2026-10-05): 4 GPUs until 11:30Z, one priority queue, anytime-valid

Read first: box_status.md, many_knobs_master_plan.md (interfaces), many_knobs_runbook.md + many_knobs_runbook_code.md
(v1 CLIs), results_many_knobs.md + status_many_knobs.md (v1 outcome, gaps), C:\datasets\orchestration\CONTRACT.md lines
84-103 (B2 route, trashcan). Code: C:\_dev\projects\DEMON-steer-bench branch ryanontheinside/spike/steer-bench. Box tree
/root/DEMON-steerbench. Status: status_many_knobs_v2.md. Results: results_many_knobs_v2.md.
Local data root: E:\Projects\DEMON\steering-bench\many_knobs_v2\. B2 prefix: sa3/steering_bench/many_knobs_v2_2026-10-05/.
Previous timeline version kept as many_knobs_v2_runbook.v1-timeline.md. Phase 1 as already launched stands unchanged.

Rules: box is shared, never stop/reboot/destroy it, never kill a process you did not start, never touch trees outside
/dev/shm/steerbench and /root/DEMON-steerbench. Overwrite-in-place slot/pool dirs (no trash growth); the trashcan
/dev/shm/steerbench/steer-bench-trash may be emptied by you at any time, nothing else. 2 drivers per GPU for protocol
work (3 OOM). Hand back only on terminal state or a run-stopping failure; progress to the status file. One permission
refusal ends that approach: record it, go on with the rest.

## Deadline and the anytime principle
HARD FREEZE 10:30Z (06:30 EDT): no new queue claims after this; in-flight items finish; then packs, regression gate,
results, local copy, B2 mirror; hand back by 11:30Z. Everything is ordered so that at ANY moment the finished set is a
complete, shippable deliverable: a knob is never half done across seeds (seed 1 for every survivor completes before
seed 2 starts for any), and the product checks come before the optional third seed and the closed loop.

## Work queue (replaces the H-phases from H2 on)
One queue file many_knobs_v2_eval/queue.txt with lines `<prio> <kind> <args>`, claimed by mkdir claims/<item>; 8 drivers
(2 per GPU) always take the lowest prio number available. CPU-only stages (directions, dedupe, ridge probes, parquet
merges, result tables) run concurrently with the queue so GPUs never wait on CPU. Append items to the queue as each CPU
stage produces them; the drivers never idle while anything is queued.

P0 FILLER (knob-independent; enqueue NOW so it runs during the phase 1 tail and phase 3 CPU time, and any time a driver
   finds nothing else): alpha-0 reference renders + full scoring for (a) the 12 screening holdout prompts x seeds
   EVAL_SEED, +1, +2; (b) the applicability sets: 12 sfx-only, 12 music-only, 12 abstract/hybrid prompts (from
   holdout_sfx_prompts.json, the TADA holdout music prompts, and corpus hybrid prompts; write the lists to
   many_knobs_v2_eval/applicability_prompts.json first) x seed EVAL_SEED; (c) 30 s alpha-0 renders for 6 prompts
   (3 music, 2 sfx, 1 hybrid) x 1 seed; (d) the stacking baseline = (a) at seed EVAL_SEED, nothing extra.
P1 SCREEN: every candidate from every variant (see Variants), 12 holdout x 3 alphas, two-family rule, as v1 phase 5.
   Enqueue in catalogue order with the v2-targeted rows (timbre/rhythm/dynamics, the v1 failure redefinitions) FIRST,
   then v1 rows. Finalize per variant group as soon as its items are done; survivors go straight to P2.
P2 PROTOCOL seed EVAL_SEED for every survivor (probe, LPAPS, PCI, cutoff, calibrate, sweep, protocol --catalogue, auc),
   as v1 phase 6 with the slot design. Order: v2-targeted rows first, then v1 rows by v1 gain-table rank.
P3 PROTOCOL seed EVAL_SEED+1 for every survivor (PCI dirs reused per seed). Same order.
P4 PRODUCT CHECKS on the P2+P3 set (gain = median of the seeds available):
   (1) stacking: 200 random pairs + 100 random triples at calibrated gain, 12 holdout prompts, seed EVAL_SEED; LPAPS vs
       the single-knob cutoff. Fit the simplest rule keeping 95% of combos under cutoff (candidates: per-knob scale
       1/sqrt(n_active), 1/n_active, shared norm budget). Report pass rates for each rule.
   (2) applicability: each knob at calibrated gain, both signs, on the 36 applicability prompts; intended-sign fraction
       by prompt category (music/sfx/abstract) from CLAP+MuQ; applies_to = the categories with fraction >= 0.6.
   (3) 30 s hold: each knob at calibrated gain on the 6 hold prompts, 30 s; descriptor/CLAP delta in 0-10 s vs 20-30 s;
       flag drift > 30% of the 0-10 s effect.
P5 SEED EVAL_SEED+2 ONLY as a tie-break: knobs where the two seeds' calibrated gains differ by more than 30% of their
   mean on either sign, or where cutoff was reached on one seed and not the other.
P6 CLOSED LOOP: ridge probes (CPU, during P2) from each hook block's token-mean to each knob's primary label on the merged
   corpus, R^2 per block; for the 10 knobs with the highest R^2 (> 0.5 required) run the closed-loop prototype on the
   12 holdout prompts: scale alpha toward a label target, report effect reached and LPAPS vs open loop at the same gain.

## Variants (phase 3, CPU, starts the minute labels for the merged corpus are complete)
build_directions on the merged 20000-clip corpus for every catalogue row (v1 catalogue rows now; the new v2 rows the
moment planner-2 delivers, without redoing the v1 rows): (a) single best block, cross-fitted 2-fold; (b) top-3 blocks
summed (one unit vector per block, each applied to its own block); (c) asymmetric per-sign (pos = top quartile vs
median, neg = bottom quartile vs median; stored as two vectors). dedupe |cos|>0.8 within variant, keep the highest
cross-fitted effect. The best variant per concept is chosen AFTER P2 by slope-per-LPAPS with the two-family rule; ties
go to (a) (cheapest, same as production today). Ship one variant per concept.

## Pack format v2 (agreed with the local DEMON-steer side, which is being changed tonight to read it)
safetensors, header metadata as today (provenance, calibrated_gain, screening, category, applies_to, seeds, variant).
Tensors: legacy `vector` [1536] + metadata block (unchanged, variant a); OR `vectors` [K,1536] + `blocks` int64 [K]
(variant b, K=3); optional `vectors_neg` [K,1536] (variant c; when absent, neg = -vectors). make_packs emits exactly
this. Any pack that is variant (a) MUST stay byte-compatible with v1 loading.

## Regression gate (before packs are final)
For every concept that also exists in v1 (46): compare v2 (chosen variant, seed EVAL_SEED protocol) against the v1
phase-6 result on the same prompts and seed: v2 wins only if intended-sign fraction (CLAP and MuQ) >= v1 - 0.05 and
LPAPS at calibrated gain <= cutoff and calibrated gain reached on at least as many signs. Otherwise the v1 pack ships
under the v2 name with provenance.kept_from = v1 and the table says why. Never regress a knob.

## Freeze and hand-back (10:30Z to 11:30Z)
1. Stop claiming; let in-flight items finish (at most ~10 min).
2. make_packs --phase6-results with gain = median over seeds done; regression gate; applies_to and category in metadata.
3. results_many_knobs_v2.md: knob table (name, category, variant, blocks, gain pos/neg with seed spread, cutoff reached,
   applies_to, kept_from), stacking rule table, hold table, cross-effect on the final set (reuse crossfx scripts), closed
   loop table (if reached), and what the queue did NOT reach.
4. Packs + tables to E:\Projects\DEMON\steering-bench\many_knobs_v2\ first (scp, small). Then B2.
5. Hand back with: counts, the regression gate tally, the B2 state, anything unreached. At most 10 lines.

## B2 mirror (incremental, per CONTRACT)
Mirror at each boundary so a mid-night failure loses nothing: after phase 1 (means + labels + embeddings), after P1
(dirs + screen), after P2 (eval seed 1), at the end (everything else). Use the method the B2 agent reports as working
(you will receive a message "B2 free" with the command shape). If a mirror is refused by the classifier, record it in
the status file and continue the run; do not retry the same request. Ring audio is an intermediate (overwritten), not a
retained output; retained outputs are means, labels, embeddings, dirs, screen, eval, listen, packs, tables.

## Disk
/dev/shm: means v2 6.4 GB fp16, labels < 2 GB, eval slots flat by design. If free space nears 4 GB, move v1 eval dirs
that are already on E:\ and on B2 (status_many_knobs.md says which) into the trashcan and empty it.

## GPU accounting (write to the status file once per hour, not more)
Per GPU: items done per kind, idle minutes (a driver with nothing claimable). Target: zero idle minutes while the queue
is non-empty. If a GPU idles because the queue is empty, enqueue P0 filler for an extra seed (+3) rather than idle.
