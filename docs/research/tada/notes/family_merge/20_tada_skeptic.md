# TADA skeptic audit (2026-10-04 morning): is the SA3 negative real?

Rules from 18/19 hold: no push, no gh writes, no attribution, no em dash, no rm (mv to
~/.claude-trash/), no tests/golden, own worktree only (read DEMON-tada-sa3 and DEMON-tada-sae-sa3,
never the primary checkout except notes/). Python C:/_dev/projects/DEMON/.venv/Scripts/python.exe,
PYTHONUTF8=1. GPU_LOCK protocol as in 18. Hand back at most 10 lines, only at terminal state.

## The claim under audit
Overnight result in status_tada_sa3.md: TADA localisation reproduces on SA3 medium (blocks 3/6/7,
peak impact 0.25 under K/V patching) but CAA, AUSteer, guidance-amplified, 30-step and the oracle
(exact per-pair per-token activation difference) all give MuQ AUC about 0.01 to 0.04 against the
paper's 0.334 on Stable Audio Open. Conclusion drawn: cross_attn_output is not a lever on SA3.
The audit's job is to try to break that conclusion. Default stance: the pipeline is wrong until a
positive control passes.

## Checks, in order (write findings to status_tada_skeptic.md as you go, one dated line each)
A. Positive control for the METRIC (highest priority, cheap). For tempo, piano, mood: take the
   positive-prompt and negative-prompt generations the oracle step cached (same seeds), or
   regenerate 50 pairs if the audio was not kept. Score each with the exact alignment function
   the AUC uses (MuQ and CLAP). Report mean alignment pos vs neg, the paired difference, and the
   fraction of pairs where pos beats neg. If this does not separate, the metric cannot see the
   concept on SA3 and every AUC in the status file is uninterpretable: say so as the headline.
B. Oracle vs patch consistency. At strength 1.0 and step 0, the oracle add at blocks 3/6/7 must
   reproduce the K/V patch's cross_attn_output at those blocks to float tolerance (same pair, same
   seed). Assert it numerically on one pair. If it does not match: find why (hook before or after
   a gate or scale, cached tensor taken at a different point than the add, renorm applied to the
   oracle, token alignment with the 64 memory tokens, dtype). Also report the impact score (eq. 2)
   of the oracle add at strength 1.0 next to the K/V patch's 0.25: if impact matches but AUC does
   not, the disagreement is between the impact score and the alignment metric, and you must say
   what eq. 2 actually measures in our code (latent distance, CLAP delta, or something else).
C. The cutoff. Reproduce one AUC number from the status file from its raw per-strength table.
   Report the raw alignment-vs-strength and preservation-vs-strength curves for tempo (CAA,
   localized) without the cutoff, and state which strengths the PCI cutoff admitted. State how the
   cutoff is computed for SA3 (baseline from SA3's own unsteered pairs or constants carried from
   the paper's models). If the admitted region is one or two strengths, the AUC is a cutoff
   artifact and the headline must say so.
D. Localisation control. Run the K/V patching sweep once with a concept-free control pair set
   (same prompt, different seed, or two neutral prompts) through the same impact score. If the
   control curve also peaks at 3/6/7 near 0.25, the localisation is generic sensitivity, not
   concept localisation.
E. Hook position at source. Read the SA3 medium block code in DEMON-tada-sa3 and state, with
   file:line, where cross_attn_output is captured and where the slot adds, relative to any gate,
   scale or residual add. Same for the TRT engine input steering_xattn. Confirm the eager and TRT
   add points are the same point. Confirm the text entry-point list (cross-attn K/V only) at source.
F. Whether SA3 degrades or steers: the ear. Listen to the existing ear package files for tempo at
   the strengths provided and say in one line what changes (tempo, texture, noise, nothing).
G. Second opinion. Run the Codex CLI with the Astra model from the DEMON-tada-sa3 worktree:
     codex exec -m gpt-6-astra --cd C:\_dev\projects\DEMON-tada-sa3 "<prompt>"
   (if --cd is not accepted, run from that directory). The prompt: the Astra brief below, verbatim,
   plus the current contents of status_tada_sa3.md and your findings A to F so far. Save Astra's
   full answer to notes/family_merge/astra_tada_review.md. Then list, in your status file, each
   Astra objection and whether you confirmed or refuted it at source (one line each). Do not adopt
   an Astra claim without checking it.

## Astra brief (paste verbatim as the start of the Codex prompt)
You are reviewing a replication of the paper TADA (arXiv 2602.11910, "steer-audio" repo) on a
different model, Stable Audio 3 medium (DiT, embed 1536, depth 24, 8-step CFG-free PINGPONG sampler,
text enters only as cross-attention K/V context of every block, 64 learned memory tokens prepended,
AdaLN carries only seconds_total). The replication team reports: TADA's K/V activation-patching
localisation gives blocks 3, 6, 7 with peak impact 0.25 (paper: blocks 11 to 13 on Stable Audio
Open); CAA and AUSteer vectors added at the cross-attention OUTPUT of those blocks give MuQ
alignment-preservation AUC about 0.01 (paper 0.334); guidance amplification (v = v0 + g(v1 - v0),
g in 3, 5, 7), 30 sampler steps, renorm on at 4x strength, and an oracle (exact per-pair per-token
activation difference positive minus negative added at the same site) all stay at or below 0.04.
They conclude the site is not a steering lever on this model. Be maximally pedantic. List every way
this conclusion could be wrong: metric bugs, missing positive controls, cutoff artifacts, hook
position errors, caching and token-alignment errors, sampler-specific effects of PINGPONG renoising
on additive steering, dtype and TensorRT issues, and anything in the paper's protocol that this
replication may have deviated from. For each, say what single check would confirm or rule it out.
Then say which three checks you would run first and what result would convince you the negative is
real. Read the code in this directory (acestep/tada/, scripts/tada/, the SA3 steering hook and
engine export) where it helps; cite file and line.

## Terminal state
status_tada_skeptic.md with sections A to G, each ending in VERDICT: <one line>. Final line
`SKEPTIC DONE <sha of any commit you made, or none>`. Commit nothing to the lane branches unless a
check found a bug; if it did, fix on a new branch ryanontheinside/feat/tada-sa3-fix in the
DEMON-tada-sa3 worktree, say what changed, and do NOT rerun the full tables (the parent decides).
Release GPU_LOCK after any GPU step. Hand back at most 10 lines: the A verdict first.
