# Replicate TADA in DEMON for model families (ACE-Step and Stable Audio 3): FOUR PARALLEL LANES

Common rules: no git push, no gh writes, nothing public, no AI attribution anywhere, no em dashes,
never rm (mv to ~/.claude-trash/), no tests/golden. Each lane has its OWN worktree and branch (below);
never touch another lane's worktree. The primary checkout C:\_dev\projects\DEMON is dirty and never
reference material; write only under its notes/. Python: C:/_dev/projects/DEMON/.venv/Scripts/python.exe,
PYTHONPATH=<your worktree>, PYTHONUTF8=1; confirm acestep.__file__ resolves inside your worktree.
Base for all lanes: the CURRENT HEAD of ryanontheinside/feat/steering-generic (another lane is still
committing demo/docs on it; you rebase onto its final HEAD at the end, which is given in
notes/family_merge/status_steering.md when it says DONE). Create: `git -C /c/_dev/projects/DEMON worktree add -b <branch> <worktree> ryanontheinside/feat/steering-generic`.
GPU: one RTX 5090. notes/family_merge/GPU_LOCK is the mutex: take it only when the owner line is
"user-demo-session" and nothing listens on port 1318; write your lane name as owner; return it to
"user-demo-session" the moment your GPU step ends. While waiting, do every non-GPU step you have.
Status files: notes/family_merge/status_tada_<lane>.md, updated at each state change (this is how the
other lanes learn about you; put the exact line formats below in there). Hand back only at terminal
state, at most 10 lines; HANDOFF in the status file if context runs low.

## The user's instruction, verbatim
"FOR TADA, I AM NOT INTERESTED IN ANY INTERSECTION WITH MY STEERING WORK AT ALL. I WANT TO
REPLICATE THEIR WORK, IN DEMON, PERIOD." "THE TASK IS ONLY TO IMPLEMENT TADA FOR MODEL FAMILIES."
TADA's methods, TADA's concepts, TADA's evaluation, on DEMON's families. Do not read, reuse, compare
with, or mention the repo's existing spectral steering (acestep/steering/ axes) or the prompt-pair
packs on the steering-generic branch. The ONLY things taken from that branch are infrastructure: the
family-agnostic steering slot ([B, L, H] additive tensor per step; eager hooks or the TensorRT
`steering` engine input), the vector pack format and loader, and the knob registration that exposes
packs as `steer_<name>` knobs. If TADA needs a hook point the slot lacks (e.g. cross-attention output
instead of post-block residual), the family lanes extend the layout declaration and the engine export.
Do not bend TADA to fit the slot.

## Source of truth
Paper arXiv 2602.11910; repo github.com/luk-st/steer-audio (MIT). Lane CORE downloads both to
E:/Projects/tada-replication/ (PDF, repo clone) and writes the digest; the family lanes read the digest
but may also read the paper and repo directly. The paper and repo win over
notes/family_merge/beyond/tada_vs_demon_steering.md on any disagreement.

## Lane CORE: algorithm port (no GPU)  branch ryanontheinside/feat/tada-core, worktree C:\_dev\projects\DEMON-tada-core
1. FIRST, within the first part of your run: fetch paper and repo; write notes/family_merge/
   status_tada_core.md section "DIGEST": every technique (activation patching sweep for layer
   localisation, CAA difference-of-means vectors, SAE features, concept-slider LoRAs, anything else),
   the exact concept sets (copy them), the models used (confirm ACE-Step v1 and Stable Audio Open
   from the PDF) and the block numbers they found (from the PDF, not from our notes), the hook points
   (which activation, which module output), hyperparameters (sample counts, steps, guidance, vector
   scaling, strengths evaluated), the evaluation protocol and metrics, released artifacts and their
   licences. End the section with the line `DIGEST READY`. The family lanes block on that line.
2. Port TADA's core as acestep/tada/: activation capture interface (a target object exposes named
   hook points and block count; nothing model-specific inside), patching sweep, CAA vector
   computation, vector application through the generic steering slot, their evaluation metrics as
   functions, and their concept lists as data files verbatim from the repo. Keep their algorithm,
   defaults and naming; cite the paper in docstrings. Unit tests on tiny synthetic activations: sweep
   picks the planted block, CAA vector equals the planted difference, zero strength is a no-op
   through the real StreamPipeline with a fake adapter, pack round-trip. Commit, then write the line
   `CORE READY <sha>` in your status file. The family lanes rebase onto that sha.
3. Then docs/TADA.md skeleton (method, citation, how to run for a new family; results tables left as
   placeholders the family lanes fill), and notes/family_merge/pr_tada.md draft. Hand back.

## Lane ACE: ACE-Step target (GPU)  branch ryanontheinside/feat/tada-ace, worktree C:\_dev\projects\DEMON-tada-ace
Wait for `DIGEST READY`, then before `CORE READY`: map TADA's hook points onto ACE-Step v1.5 (DEMON's
checkpoint; TADA used v1: record the difference) via the steering layout; add any missing hook point
to the layout and the ACE TensorRT export, with the eager path as parity reference; build the engine
only if the export changed (idle GPU needed; version it, keep the old one; parity at the existing
bar). On `CORE READY <sha>`: rebase onto it, run the patching sweep with their concepts and settings,
report the blocks found against the paper; compute CAA vectors for their concept set, write packs to
E:/Projects/tada-replication/packs/acestep/, run their evaluation (their metrics and sample counts,
scaled down only if the GPU budget demands it and say so). Table: concept, block, metric at their
strengths, zero-strength no-op (bit-identical). Verify the packs load as `steer_<concept>` knobs in the
ACE knob manifest. Fill the ACE results into docs/TADA.md. Hand back with the table headline.

## Lane SA3: Stable Audio 3 target (GPU, has priority on the lock when both family lanes are ready)
branch ryanontheinside/feat/tada-sa3, worktree C:\_dev\projects\DEMON-tada-sa3
Same as Lane ACE for SA3 medium (new for TADA; Stable Audio Open is their nearest model: say what
transfers). The SA3 engine already carries the post-block `steering` input (sa3_m_dit_steer_*); add a
hook point and rebuild only if TADA needs one the slot lacks. Packs to E:/Projects/tada-replication/
packs/sa3/. Plus the live path: the /sa3 demo's Steer section renders whatever `steer_*` knobs the
manifest carries; make the server load the TADA pack directory for SA3 via config (no concept names
in the page), verify on the real server (sliders move the sound per their concepts; zero is a no-op),
stop the server, write notes/family_merge/demo_logs/LAUNCH_sa3_tada.sh and HOWTO_sa3_tada.md (pattern:
LAUNCH_minimax.sh; worktree DEMON-tada-sa3). Fill SA3 results into docs/TADA.md. Hand back.

## Lane SAE: SAE features and concept-slider LoRAs (no GPU unless cheap)
branch ryanontheinside/feat/tada-sae, worktree C:\_dev\projects\DEMON-tada-sae
Wait for `DIGEST READY`. Scope exactly what TADA's SAE and slider-LoRA components need (training
data, compute, released artifacts and whether they apply to our checkpoints). If their released
artifacts apply, port the loading and application code onto the slot and test it on CPU with the
artifacts. If replication needs training: write the plan with GPU hours and data in
status_tada_sae.md and build nothing further. If training is a few GPU hours or less and the lock is
free after the family lanes finish, do it. Hand back with what was done versus scoped.

## Integration (done by the parent after all four hand back)
Rebase order: core onto the final steering-generic HEAD, then ace onto core, sa3 onto ace, sae onto
sa3; the result is ryanontheinside/feat/tada. Full tests/unit green; gates (no attribution, no em
dash, no personal paths). PR body pr_tada.md base steering-generic, no attribution footer.

## Hand-back format (each lane, at most 10 lines)
HEAD; what landed; the numbers (blocks vs paper, metric headline, engine parity if rebuilt, test
counts); demo verified yes/no and launch script path (SA3 lane); FLAG with the reason if something in
the paper cannot be reproduced, stopping at the last green step.

## Lane SAE-SA3 (added 2026-10-03 after the cross-check; replaces the scoping-only Lane SAE)
branch ryanontheinside/feat/tada-sae-sa3, worktree C:\_dev\projects\DEMON-tada-sae-sa3, base tada-ace 64e565c5
(rebase onto CORE READY <sha> when it appears). Read FIRST: beyond/sae_crosscheck.md (binding: its
dead-end items are hard constraints), status_tada_sae.md (the recipe), status_tada_ace.md (the hook
API you build on). SA3 medium only; keep every module family-generic through the layout. User
approved the GPU spend (4-6 h estimate) on the condition that TADA's recipe is followed literally.
Steps, one commit each:
1. Port TADA's SAE pieces from the repo onto DEMON: generation-time activation caching at a named
   hook point through DEMON's own pipeline (cond pass, every k-th step, their 30-step CFG 5 setting
   or SA3's equivalent, say which), BatchTopK SAE with AuxK (their hyperparameters), FVU/dead/fire
   metrics PLUS an absolute per-sigma-bucket FVU check, TF-IDF concept scoring (their eq. 12, their
   prompt templates), top-k_c selection on a held-out prompt set by their metric, v_SAE pack writer
   (hook cross_attn_output, single-vector and per-step forms). CPU unit tests on synthetic data:
   planted features recovered, scoring ranks a planted concept feature first, zero alpha is a no-op.
2. Wait for the SA3 lane's localized blocks (status_tada_sa3.md). Time ONE caching batch and ONE
   training epoch on the GPU first and write the measured per-family cost in your status file before
   committing to the full run. Then cache, train (sweep m and k as in the paper, pick by their
   criteria), score, select, write packs to E:/Projects/tada-replication/packs/sa3_sae/.
3. Evaluate with their protocol against the SA3 CAA packs from the SA3 lane, same prompts, same
   metric; add a short ear package (6 files, zero vs alpha, two concepts) under
   E:/Projects/tada-replication/listen_sae/ for the user. Table: concept, k_c, alpha, metric vs CAA,
   zero-alpha no-op (bit-identical).
4. Results into docs/TADA.md (SAE section, SA3 rows; ACE rows as TODO). Hand back: HEAD, measured
   GPU hours, FVU per block, the table headline (SAE vs CAA), test counts, ear package path.
GPU lock protocol as above; the SA3 lane keeps priority until its CAA packs are written.
