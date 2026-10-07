# TADA overnight (2026-10-03 evening to 2026-10-04 morning): autonomous rules

The user is away until the morning of 2026-10-04 and expects the TADA replication done. Standing
rules from 18_tada_replication.md hold (no push, no gh writes, no attribution, no em dash, no rm,
no tests/golden, own worktree only, notes/ only in the primary checkout). NO GPU hour cap exists.
On a negative result do not wait for anyone: record it in the status file and start the next
experiment from the tree below. Everything below is pre-authorized.

## Lane SA3 (ryanontheinside/feat/tada-sa3): negative-result tree
N1 Localization: no SA3 cross-attn block exceeds tau 0.10 for a concept. Record the full per-layer
   impact curve, take the top-2 blocks by impact for that concept (say so in docs/TADA.md as a
   deviation), proceed. If NO concept exceeds tau, also run the sweep once at the paper's 30-step
   setting on SA3 (steps 30, cfg 1 since SA3 is CFG-free) to rule out the few-step setting, then
   take top-2 blocks.
N2 CAA at the localized blocks does not beat the no-steer baseline on alignment-preservation AUC
   (MuQ or CLAP) for most concepts: run, in order, each through the same protocol: (a) all-blocks
   application (paper's "all" variant), (b) per-step vectors from the cached per-prompt
   activations, (c) steering the cross-attn K/V site instead of the output (same vectors recomputed
   at that site). Stop at the first that works; record all.
N3 AUSteer worse than CAA: that is a result, record it. If AUSteer is below baseline, check the
   sign convention against the repo once (eq. 7) and rerun only if a bug is found.
N4 Zero-strength is not bit-identical: stop, fix, rerun the affected tables (this is a bug, not a
   result).
At terminal state: packs in E:/Projects/tada-replication/packs/sa3/, 6-file ear package in
E:/Projects/tada-replication/listen_sa3/ (two concepts, zero vs two strengths, README with what to
listen for), docs/TADA.md SA3 rows filled, status file final line `SA3 DONE <sha>` with the
headline table. Then wire the demo (LAUNCH_sa3_tada.sh already written): verify once on the real
server, stop it, release GPU_LOCK to "user-demo-session". Hand back, at most 10 lines.

## Lane SAE-SA3 (ryanontheinside/feat/tada-sae-sa3): negative-result tree
Takes GPU_LOCK after `SA3 DONE` (or after the SA3 lane's CAA packs exist and it releases the lock,
whichever comes first). Timing batch first; write `ETA:` with measured numbers; proceed regardless.
S1 Dictionary bad (FVU above 0.40 at the paper's (m,k), or the absolute per-sigma-bucket FVU
   check blows up): lower lr to 1e-5, cache every step instead of every 2nd, retrain once. Record.
S2 Features found but v_SAE below the no-steer baseline for most concepts: try per-step feature
   choice (the repo variant), then the larger k_c set. Record.
S3 v_SAE steers but is worse than CAA: record it, keep the packs, ship CAA as default. Not a failure.
At terminal state: packs in E:/Projects/tada-replication/packs/sa3_sae/, ear package in
E:/Projects/tada-replication/listen_sae/, docs/TADA.md SAE section, status final line
`SAE DONE <sha>` with FVU per block and the SAE vs CAA headline. Release GPU_LOCK. Hand back.

## Watcher (third party, cheap model, hands back ONLY on a state change)
Every 10 minutes check: mtimes of notes/family_merge/status_tada_sa3.md, status_tada_sae_sa3.md and
the newest file under notes/family_merge/steer_logs/; GPU_LOCK owner; nvidia-smi utilization;
whether a python process from the lane worktrees is alive. Hand back (at most 10 lines, with the
evidence) ONLY when: no status or log change for 45 minutes while the lock is held; GPU utilization
under 5 percent for 20 minutes while the lock is held and the owner is a lane; a line starting
`FLAG` appears in either status file; a Python traceback appears in the newest log; `SA3 DONE` or
`SAE DONE` appears; the lock is released by both lanes. Nothing new = say nothing. Never touch the
lanes' files, never take the lock, never kill anything. Give up and hand back at 09:00 on 2026-10-04
if nothing happened.

## Integration (parent spawns an Opus agent after both DONE lines)
Worktree C:\_dev\projects\DEMON-tada (new), branch ryanontheinside/feat/tada. Rebase order: core
(cab0451c or later CORE READY2 sha) onto steering-generic 4beb08d4, ace 64e565c5 onto core, sa3
final onto ace, sae-sa3 final onto sa3. Full tests/unit (the 4 lora_facade drain failures are
pre-existing on main; anything else is a regression). Gates: no attribution strings, no em dash,
no personal paths, no concept names in demo page source. pr_tada.md body refreshed from the filled
docs/TADA.md. Nothing pushed.

## Morning (parent, one cron wake)
Read both status files once via a haiku agent, write the user's report: localized blocks vs paper,
CAA/AUSteer/SAE headline table, ear package paths, measured GPU hours, integration state, open
FLAGs. No push until the user says so.

## E1 (added 2026-10-04 ~01:15 after `SA3 DONE 23e98b84`): guidance-amplified steering on SA3
Result so far: TADA's CAA and AUSteer reproduce only weakly on SA3 medium (MuQ/CLAP AUC 0.011/0.001
vs paper SAO 0.334/0.195; localized no better than all blocks; K/V site worse; only tempo moves).
Hypothesis to test next (the user pre-authorized the next experiment): TADA applies the vector on
the conditional pass only and renormalizes. Under CFG the final velocity is u + w (c - u), so the
steer's effect is multiplied by the guidance weight (w = 5 or 7 in their runs) while renorm bounds
the raw perturbation. SA3 runs CFG-free (w = 1, ARC-trained, 8 steps), so the same vector yields
roughly 1/w of the paper's effect and cannot be compensated by alpha because of the renorm bound.
Experiment (Lane SA3, branch tada-sa3, same worktree, same engine, no rebuild):
1. Add a family-generic slot option `guidance` (float, default 1.0 = today's behavior) to the
   steering slot: when > 1, each DiT step runs twice through the SAME TensorRT engine, once with
   steering zeros and once steered, and the step uses v = v0 + guidance * (v1 - v0). Batch the two
   in one engine call if the engine's batch dim allows, else two calls. Eager path identical for
   parity. Unit test on the fake adapter: guidance 1.0 bit-identical to today; guidance 3 equals
   the extrapolation formula.
2. Run the scaled protocol (50 prompts, 10 strengths per side, same metric) for CAA localized and
   all-blocks on three concepts (tempo plus two that failed, e.g. piano and mood) at guidance
   {1, 3, 5, 7}. guidance 1 is the control and must reproduce the existing numbers.
3. Same run, one concept, one extra control: no-renorm application at larger alpha (the repo's
   option if it has one, else a flag), to separate "renorm bound" from "no amplification".
4. If any guidance gives MuQ AUC above 0.10 on the failing concepts: run all 9 concepts for CAA
   and AUSteer at the best guidance, update docs/TADA.md (SA3 rows, with a paragraph explaining
   the CFG-free adaptation), set the pack metadata/slot default for SA3 to that guidance, re-verify
   the demo once (sliders move the sound per concept, zero is a no-op), refresh the ear package.
   If nothing beats 0.10: record it as the negative result with the per-guidance table.
5. Write `MECHANISM: guidance=<w> renorm=<yes/no>` and then `SA3 DONE2 <sha>` in
   status_tada_sa3.md, release GPU_LOCK, hand back at most 10 lines.
Budget: this reuses the 18 vectors and the existing drivers; expected 1 to 2 GPU hours. The SAE
lane waits for `SA3 DONE2` before its full run and evaluates with the MECHANISM line's setting.

## E2 (added 2026-10-04 ~04:15 after `SA3 DONE2 983361a4`): is cross_attn_output a lever on SA3 at all, and where does text enter?
E1 result: guidance 1/3/5/7 and 30 steps all leave CAA at MuQ AUC about 0.02; renorm on at 4x
strength lifts tempo to 0.07. So neither CFG-free sampling nor few-step sampling is the cause.
Localization did peak at blocks 3/6/7 under K/V patching, yet a mean-difference vector added at the
output barely moves anything. Two hypotheses, tested together (Lane SA3, same worktree):
1. Oracle test (is the site a lever for ANY additive vector?): for tempo, piano and mood, 50
   prompt pairs, add at cross_attn_output of the localized blocks the PER-PAIR, PER-TOKEN
   difference (activation under the positive prompt minus activation under the negative prompt
   for the same pair, same seed, same step), cond pass, strengths on the usual grid, scored with
   the usual protocol. Also the per-pair MEAN over tokens (one vector per pair) as the middle
   case. If the oracle moves the metric (MuQ AUC above 0.10) and the mean vector does not, the
   site is a lever but SA3 needs per-pair or per-token structure (record which); if even the
   oracle fails, the site is not a lever and the SAE must move.
2. Site sweep (where does the text concept enter SA3 medium?): read the SA3 medium model code in
   your worktree and list every path by which text conditioning reaches the trunk (cross-attn
   K/V, prepended or concatenated conditioning tokens in self-attention, global/pooled text into
   AdaLN or timestep embedding, anything else). Run TADA's activation patching sweep (same
   concepts, same pairs x seeds, same impact score, tau 0.10) at each such entry point and at the
   post-block residual stream for every block. Table: site, block, impact. The best site is the one
   with the highest peak impact; say how it compares to cross-attn's 0.25.
3. If the best site is not cross_attn_output and its peak impact is above 0.30: compute CAA (and
   AUSteer, from the same cached activations) at that site for the 9 concepts, add the hook to the
   layout and the TensorRT export if the slot lacks it (version the engine, keep the old one,
   parity at the bar), run the scaled protocol, write packs under
   E:/Projects/tada-replication/packs/sa3/medium_<site>/, update docs/TADA.md and the ear package
   if MuQ AUC beats 0.10 on most concepts. Otherwise record the table as the negative result.
4. Write `SITE: hook=<name> blocks=<list> peak_impact=<x> oracle=<pass/fail>` and then
   `SA3 DONE3 <sha>` in status_tada_sa3.md, release GPU_LOCK, hand back at most 10 lines.
The SAE lane waits for `SA3 DONE3` and trains at the SITE line's hook and blocks.

## E3 (added 2026-10-04 morning after the skeptic audit, status_tada_skeptic.md and astra_tada_review.md): corrections, then the tables again
Audit findings that bind: (1) the E2 "oracle" added the CAA training-pair difference (pair i) to
benchmark prompt i; it was not a same-pair oracle; the "site is not a lever" conclusion is
RETRACTED. A correct same-pair oracle recovers 0.16 of the concept on eq. 2 (MuQ), exact output
substitution 0.30, the K/V patch 0.47 to 0.65: fixed vectors recover about a third of the patch
effect, the rest is state-dependent. (2) CLAP barely separates the real positive and negative
prompts on SA3 (0.008 to 0.025); MuQ does (tempo +0.104, piano +0.065, mood +0.024). Our eq. 2 is a
CLAP cosine floored at 0, so the localisation and the running E2 site sweep are scored with a
near-blind instrument on SA3. (3) The real prompt itself (PCI) scores MuQ AUC only 0.032 averaged
over tempo, piano, mood (tempo 0.060) under the same protocol; CAA scores 0.020 to 0.024, about two
thirds of PCI. The paper's 0.334 is not the comparison; CAA relative to PCI on the same model is.
(4) The 64 learned memory tokens are inside the token mean of CAA and AUSteer (about a quarter of
the mean). (5) The 30-step row reused 8-step vectors. (6) The K/V-site CAA mask covers padding
tokens. (7) No hook bug (eager and TRT add at the same ungated point); the PCI cutoff is SA3's own
and admits 4 to 8 strengths; the concept-free control sweep is flat (max 0.07) so localisation is
concept-specific. Lane SA3 (same worktree, branch tada-sa3), in order:
1. The E2 site sweep: if sa3_e2_chain.sh keeps the generated audio, let it finish and rescore with
   MuQ; if it keeps only CLAP scores, stop the process now (kill the python, not the box), record
   the partial table as "CLAP-scored, superseded", release GPU_LOCK, take it again for the steps
   below.
2. Metric: eq. 2 impact and the AUC alignment term use MuQ as primary on SA3; CLAP is reported in a
   secondary column, never used for a decision. Say this in docs/TADA.md as the SA3 deviation and
   why (the audit numbers). Re-run localisation with MuQ scoring, tau 0.10, per concept; report the
   per-concept peak blocks and the union; compare with the CLAP-scored 3/6/7.
3. Vectors: recompute CAA and AUSteer at the MuQ-localized blocks with the mean over AUDIO tokens
   only (memory tokens and padding excluded; say so, with the before/after cosine to the old
   vectors). Fix the K/V-site mask to real tokens. Drop the 30-step row (or redo it with 30-step
   vectors if it costs under 30 minutes; say which).
4. Tables: every table gets a PCI row (the real prompt through the same protocol) and a CAA/PCI
   and AUSteer/PCI ratio column. Pull the paper's PCI AUC for Stable Audio Open and ACE-Step from
   the PDF so the ratio can be compared with theirs; cite the table. The headline is the ratio.
5. The all-blocks variant and the K/V-site variant once each with the corrected vectors.
6. Ear package refresh under E:/Projects/tada-replication/listen_sa3/: for piano, mood and tempo,
   four files each: zero, the real prompt (PCI), CAA at the best admitted strength, CAA at the
   largest strength on the grid; README says what to listen for and that the PCI file is the
   reference. Keep the old files in a sub-folder "superseded".
7. docs/TADA.md SA3 section rewritten: localisation (MuQ), CAA and AUSteer vs PCI, the fixed-vector
   versus state-dependent finding (same-pair oracle 0.16, substitution 0.30, patch 0.47 to 0.65),
   the retraction, the deviations list. No speculation beyond those numbers.
8. Write `SITE: hook=cross_attn_output blocks=<MuQ union> peak_impact=<MuQ> oracle=pass(partial)`
   and `SA3 DONE4 <sha>`, release GPU_LOCK, hand back at most 10 lines with the ratio table.
Lane SAE-SA3 waits for `SA3 DONE4` and trains at the SITE line's blocks with the same two
corrections: cache and mean over audio tokens only, MuQ as the primary scoring and selection
metric. Measured plan stands (about 6.2 GPU h); the user approved the spend on 2026-10-03.

### E3 amendments (parent, 2026-10-04 after rereading arXiv 2602.11910 v3 in full)
Binding on Lane SA3 and Lane SAE-SA3; paper line references are to E:/Projects/tada-replication/tada.txt.
1. No SA3 number is ever compared with 0.334. That is Table 23 (Stable Audio Open, CAA only, four
   concepts, LPAPS-axis units, no PCI row). Reference points are RATIOS on the same model: ACE Table 1
   CAA-loc/PCI 0.104/0.084 = 1.24, SAE-loc/PCI 0.118/0.084 = 1.40, CAA-all/PCI 0.075/0.084 = 0.89;
   localisation gain CAA +39% MuQ (Table 2), SAO Table 23 all 0.310 -> loc 0.334 (+8%), AudioLDM2
   Table 22 0.149 -> 0.406 (+172%). Cite the table with every reference number.
2. Localisation = Sec. 4 exactly: K/V patch of one cross-attention layer; impact eq. 2; s = MuQ for
   mood, tempo, instruments, genres and CLAP for vocal gender; I(l) averaged over concepts; tau 0.10 on
   the average; report the per-concept map beside it (App. F.1 style) and the tau sensitivity row
   (Table 7 style, tau 0.05 to 0.30). The E2 xattn_out and resid patch sweeps are NOT part of the
   paper's method: keep their audio on disk, do not spend GPU scoring them with MuQ now, record them
   as "CLAP-scored exploratory, deferred". MuQ rescoring applies to the K/V-patch audio only.
3. Vectors stay per diffusion step (Sec. 5.2 and I.1.4: "we calculate steering vectors for each
   diffusion step separately"), 8 steps on the served ARC checkpoint, sigma recorded per step. CAA
   eq. 6: time-averaged cross-attention outputs, mean of pair differences, unit norm, no renorm
   (H.4: the following norm layer renormalizes). Time average over AUDIO tokens only: the paper's
   models carry no learned memory tokens, so the 64 SA3 memory tokens and any padding are excluded
   from the mean (the vector is still added to every token at the hooked output, as hl = hl + alpha v).
4. Primary rows renorm off; packs.py default renorm=False (done, 9a59ecae). One renorm-on control
   row for tempo only, labelled as such.
5. Protocol deviations list in docs/TADA.md: 50 of the 100 benchmark prompts, 21 strengths (paper
   31), 10 s clips (paper 30 s on ACE, 10 s on SAO/AudioLDM2 in Sec. 4), 8-step CFG-free ARC sampler
   (paper 30 Euler steps CFG 5), alpha_max calibrated to SA3's own PCI max distortion (same rule as
   Sec. 5.2), hold-out 20 prompts for hyperparameters (same as I.2), MuQ primary for all decisions
   on SA3 (paper reports both; audit showed CLAP near-blind on SA3), 32 pairs x 8 seeds in
   localisation (paper up to 256 pairs; Table 6 shows 34 pairs recover ACE's set exactly).
6. Cosine matrix of the 9 CAA vectors, old and corrected, with cosine to tempo (old matrix done:
   piano/tempo -0.02, mood/tempo 0.18; the "one shared density direction" reading is refuted).
7. Ear package: zero and steered files from the same seed and batch layout (the quick piano render
   drew batch-1 noise for zero, so its heard difference is confounded).
8. Agent discipline: the previous SA3 and SAE agents both ENDED THEIR TURN while waiting for GPU
   work and were therefore terminated without reaching DONE. Never end the turn before the terminal
   line is written. Wait for long jobs inside a foreground blocking shell command (an until-loop on
   the pid or the output file, timeout up to 10 minutes per call, repeated as needed); never hand the
   wait to a background job and stop. Before starting any GPU work, inventory running python
   processes from this lane's chains (steer_logs/*.sh) so nothing is started twice.

### E3 reorder (parent, 2026-10-04 mid-morning): SAE takes the GPU right after SITE
Only the MuQ localisation gates the SAE lane. So, Lane SA3: once the MuQ localisation is done,
write `SITE: hook=cross_attn_output blocks=<MuQ union> peak_impact=<MuQ> oracle=pass(partial)`
plus a `SAE GO` line in status_tada_sa3.md, return GPU_LOCK to "user-demo-session", and do your
CPU-only steps while the SAE lane holds the GPU (CAA and AUSteer recompute over audio tokens from the
existing per-prompt caches, cosine matrices, docs text, ear-package README, deviations list). Resume
your GPU steps (calibration probes, sweeps, scoring, ear renders) only after status_tada_sae.md
carries `SAE DONE` or `SAE FLAG`, taking GPU_LOCK by the usual protocol. Lane SAE-SA3: take GPU_LOCK
when `SAE GO` appears and the lock is at "user-demo-session"; your comparison row against CAA uses
the PCI row (already measured, ranges_eval50.json, PCI cutoffs) first and the corrected CAA row when
the SA3 lane posts it; return the lock the moment your GPU work ends. Both lanes: rule 8 of the
amendments (never end the turn while waiting; block in foreground shell commands).

### E3 correction (parent, 2026-10-04 14:45): SA3 medium generates no vocals
vocal_gender and vocal_style are excluded from every SA3 table, average and localisation (paper App. A: concepts the model cannot generate cannot be localized or steered). SA3 benchmark = 7 concepts. Lane SA3: recompute the localisation average without female/male from the existing scores (CPU), post `SITE2:`; if the tau 0.10 set changes, the SAE lane adds or drops blocks accordingly. Lane SAE-SA3: skip both vocal concepts in scoring, selection and eval. Report the two as "not generable by SA3, excluded".

## E4 probe (parent, 2026-10-04 17:00, after BASE TEST DONE)
Base-checkpoint control refuted the distillation hypothesis: SA3 medium-base (50 steps, cfg 7) CAA-loc/PCI MuQ tempo 1.07, piano -1.08, mood 0.05 (ARC 0.92/0.39/0.76; paper ACE 1.24); all sweeps past cutoff. The vector distorts like PCI but does not move the concept: wrong direction or wrong site, not snap-back. Decisions: SAE STOP (fixed dictionary at the same site inherits the failure; dictionaries also fail the low-sigma buckets). ARC E3 tail chain (sa3_e3_final.sh) HELD, no GPU. Next = CPU probe on the SAE cache (D:/tada-replication, 5521 captions x 8 steps x blocks 3,5,6,7, 172 audio tokens):
1. Labels from captions: piano vs not; fast vs slow; happy vs sad; balanced subsample.
2. Per block x step: mean-level logistic probe (5-fold acc) on time-averaged activations + effect size along the mean difference; token-level probe on individual tokens (~20k per class); cosine of the mean-difference direction to caa_e3 CAA vectors (same concept/block/step).
3. Read-out: mean-level acc >= 0.85 -> direction exists, steering mechanics are the problem -> next test is a learned affine map (AcT-style) at the site, fit on the cache, one GPU hour. Token-level high but mean-level near chance -> concept is positional; mean-vector steering (TADA as specified) cannot work on SA3; replication ends as a clean negative with that explanation. Both near chance -> site carries no linear concept information on SA3; same end.
4. No GPU until the probe table is in. Status: status_tada_sae_sa3.md under "E4 PROBE".

## E4b large-N CAA (parent, 2026-10-04 17:20, after E4 PROBE)
Probe: mean-level linear direction exists at cross_attn_output for piano/tempo/mood (acc 0.96 to 0.98, shuffled 0.5). The 50-pair CAA vectors have cosine 0.38 / 0.74 / 0.55 to it and steering ratio 0.39 / 0.92 / 0.76: alignment tracks success. Hypothesis: the estimator, not the site, fails on SA3. Test (one GPU hour, ARC):
1. SAE lane (CPU): export caa_e4 vectors = unit-norm mean difference (positive pole minus negative: piano, fast, happy; same sign convention as caa_e3) over the balanced keyword classes of the cache, time-averaged over the 172 audio tokens, per step (8) x block (3,5,6,7), in the caa_e3 file layout so the SA3 lane's --vec-dir caa_e4 works. Save class counts and cosine to caa_e3 beside them. Write `E4 VECTORS READY` into status_tada_sae_sa3.md. Hand back, 5 lines.
2. SA3 lane (GPU after `E4 VECTORS READY`, GPU_LOCK when free, never preempt the demo session): eval_e4 on ARC, CAA loc 3,5,6,7, --vec-dir caa_e4, tempo/piano/mood, 20 holdout prompts, same seeds as eval_e3, calibrate to the existing eval_e3 PCI cutoffs, 14 strengths/side, MuQ scoring, AUC, ratio vs PCI-all beside the caa_e3 ratios (0.92/0.39/0.76). Write `E4 DONE` in status_tada_sa3.md, hand back 10 lines. No other GPU work (the E3 tail chain stays HELD).
3. Read-out: piano and mood ratios toward 1 -> TADA replicates on SA3 with a data-scale deviation; then finish the 7-concept tables with caa_e4 vectors. Flat -> clean negative, explanation = probe table + cosines; write docs, no more GPU.

## E5 final SA3 table (parent, 2026-10-04 17:45, after E4 DONE)
E4: full-cache CAA vectors raise matched ratios 0.63 -> 0.88 (tempo 0.87 -> 1.15, piano 0.15 -> 0.41, mood 0.61 -> 0.66). Estimator confirmed as the SA3 gap. Last GPU spend (~2.5 h total), then write-up either way. SAE training stays STOPPED (documented: dictionaries fail the sigma 0.27 bucket at paper-grid k; user's prior work says k=128 needed; not pursued).
1. SAE lane: (a) GPU ~10 min (lock when free, never preempt the demo session): capture per-step (8) audio-token-MEAN activations at ALL 24 blocks for the 5521 cached captions (ARC, cond pass, 172 audio tokens, memory tokens and padding excluded), ~4 GB, to D:/tada-replication. (b) CPU: export caa_e5 at E:/Projects/tada-replication/sa3/caa_e5 for the 7 SA3 concepts (tempo, piano, mood, violin, guitar_electronic, electronic_music, rock_genre; keyword classes, balanced, same sign convention as caa_e3) at all 24 blocks x 8 steps in the caa_e3 layout so --sites all works; record class counts and cosine to caa_e3 in e5_meta.json. (c) CPU: caa_e5p = unit-norm logistic-probe weight direction (mean-level probe, blocks 3,5,6,7 only) for tempo/piano/mood, same layout. Write `E5 VECTORS READY` into status_tada_sae_sa3.md, hand back 5 lines.
2. SA3 lane (GPU after `E5 VECTORS READY`): eval_e5 on ARC, the eval50 protocol (50 prompts, same seeds as eval50 PCI-all, 14 strengths/side, calibrated to the existing PCI cutoffs, renorm off): CAA loc 3,5,6,7 and CAA all with caa_e5 for 7 concepts; CAA loc with caa_e5p for tempo/piano/mood. MuQ AUC. Table: per concept CAA-loc/PCI, CAA-all/PCI, 7-concept means, beside the caa_e3 rows and paper ACE (loc/PCI 1.24, all/PCI 0.89, loc vs all +39%). Ear package: tempo and piano, zero vs steered, same seed and batch layout, with PCI reference. Then docs/TADA.md SA3 section (deviations: data-scale estimator, 7 concepts no vocals, 8-step ARC, blocks 3,5,6,7, SAE not reproduced). Write `SA3 DONE5 <sha>`, release lock, hand back 10 lines. GPU budget 2.5 h; FLAG if beyond, do not trim concepts.
3. No other GPU work. Integration of the large-N estimator into the pack builder waits for the user's read of the table.
