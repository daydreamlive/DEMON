# v3 smoke: descriptor-ridge steering directions (drafted 2026-10-07, GPU NOT authorised yet)

Question: does a direction fit by ridge regression from per-block token means to a continuous audio descriptor steer that
descriptor monotonically, within the PCI perceptual budget, and at least as well as the shipped tag knob that proxies it?
Verdict decides whether descriptor knobs go into v3 (notes/steering_pr: v3 discussion 2026-10-07).

## Rules
- Code in C:\_dev\projects\DEMON-steer-bench (branch ryanontheinside/spike/steer-bench), scripts/steering_bench/. Do NOT commit;
  the parent commits. Never touch C:\_dev\projects\DEMON (dirty primary) except notes/.
- Data root E:\Projects\DEMON\steering-bench\v3_smoke\. Inputs are the v2 steering corpus (per-block fp16 token means,
  descriptor scores, labels) under E:\Projects\DEMON\steering-bench\many_knobs_v2\ (or B2 sa3/steering_bench/ if not local;
  B2 download is authorised, read C:\datasets\orchestration\CONTRACT.md first).
- NO GPU use (local or box) until Ryan says go. Build everything, dry-run on CPU with stubs, hand back.
- Reuse existing tools: score_descriptors.py (descriptor definitions), build_directions.py / make_packs.py (pack format),
  ship_tools.py (PCI cutoff, LPAPS, probe grid, prompt sets). Do not reimplement a descriptor or the pack format.
- Reports: <= 10 lines, only at terminal state or FLAG. Status to notes/steering_pr/status_v3_smoke.md.
- Deletion = mv to ~/.claude-trash/. No attribution, no em dashes.

## Design
Descriptors (3), each with a shipped tag knob as control:
  onset_rate   control: percussive
  centroid     control: bright
  lra          control: dense_arrangement (nearest shipped proxy; if a better one exists in the 97, say so)
Directions: ridge (sklearn Ridge, alpha chosen by 5-fold CV on the corpus) from the per-block token means (same block choice
as the v2 variant a packs; also fit the summed top-3 block variant if build_directions.py already supports it) to the
descriptor, standardised. Unit-normalise the coefficient vector; it becomes the pack vector in the existing pack format,
tagged provenance.method = "ridge", with CV R^2, n, blocks in the header. Also residualise each descriptor against the
confound set from catalogue_v2_notes.md (centroid, bass_db, flux_mean, lufs, perc_ratio, excluding itself) and fit a second
vector per descriptor; both go through the render.
Render: 12 protocol prompts (the 12 used for screening in many_knobs_v2), seed 0, alphas = 5 points on the v2 probe grid
spanning 0 to the PCI-all cutoff region (reuse ship_tools probe grid), both signs. Each render scored with the same
descriptor (score_descriptors.py), LPAPS vs alpha-0, CLAP to the control's anchors.
Pass per direction and sign: Spearman(descriptor, alpha) over the 5 alphas >= 0.9 as the median over the 12 prompts, with
LPAPS at the top alpha within 1.1x of the PCI cutoff; and descriptor delta at matched LPAPS >= the control knob's descriptor
delta at matched LPAPS. Report a 3 x (ridge, ridge-residualised, control) table per sign.

## Cost
About 6 vectors x 2 signs x 5 alphas x 12 prompts = 720 renders of 10 s plus 12 alpha-0 refs and the 3 control knobs at
5 alphas (360). ~1100 renders at batch 16 on one 5090: under 30 min including scoring.

## Deliverables
A. ridge_directions.py: fits the 6 vectors from the local corpus, writes packs to E:\...\v3_smoke\packs\ and a fit report
   (CV R^2, cos between ridge and control vector, cos between ridge and ridge-residualised).
B. smoke_drive.py (+ smoke_item.sh if the existing drive pattern needs it): takes the packs dir, renders and scores the grid,
   writes E:\...\v3_smoke\results.json and results.md with the pass table. Must run end to end on CPU with a --dry-run that
   stubs the renderer and scorer, so the pipeline is proven before the GPU word.
