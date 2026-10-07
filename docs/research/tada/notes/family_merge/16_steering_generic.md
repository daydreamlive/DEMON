# Family-generic activation steering (TADA method) with Stable Audio 3 first, on TensorRT

Rules: no git push, no gh writes, nothing public, no AI attribution anywhere (code, commits, docs),
no em dashes, never rm (mv to ~/.claude-trash/), no tests/golden. Work ONLY in a new worktree:
`git -C /c/_dev/projects/DEMON fetch origin && git -C /c/_dev/projects/DEMON worktree add -b ryanontheinside/feat/steering-generic /c/_dev/projects/DEMON-steer origin/main`
The primary checkout C:\_dev\projects\DEMON is dirty and is never reference material; write only
under its notes/. Do not touch the DEMON-fam-* worktrees (another lane is restacking them).
GPU: take notes/family_merge/GPU_LOCK (rewrite owner to "steering-generic"), give it back to
"user-demo-session" when done. TensorRT builds need an idle GPU: do not build while a server runs.
Python: `C:/_dev/projects/DEMON/.venv/Scripts/python.exe` with PYTHONPATH=C:\_dev\projects\DEMON-steer
and PYTHONUTF8=1; confirm `acestep.__file__` resolves inside the worktree before any test.
Status file: notes/family_merge/status_steering.md (every number, every decision, HANDOFF if context
runs low). Hand back only at terminal state, at most 10 lines.

## What the user wants (verbatim intent)
"can we plug their [TADA] work in to acestep and sa3? particularly sa3 is more interesting. i would
like to see it added to the sa3 demo." "the steering port should be done in such a way that it works
on either acestep, sa3, or any model in the future." "we always want it on tensorrt." "as for the
existing acestep steering, lets not lose it."

## Ground truth to read first (do not redesign what exists)
- acestep/steering/ (catalog.py, controller.py, hub.py, policy.py, types.py): the existing ACE-Step
  v1.5 steering (axes steer_bright/warm/rough/density, probe layers, per-step policy). This stays
  byte-for-byte functional. It becomes the first consumer of the generic seam, not a casualty.
- acestep/engine/trt/export.py around lines 337-610 and 1086-1220: the ACE decoder engine carries a
  `steering` input [B, num_layers, hidden_size] added to each block's output residual; the host zeros
  rows for unused layers. StreamPipeline._install_steering_hooks is the eager equivalent.
  acestep/engine/trt/_engine_metadata.py schema 2 and build.py's decoder_onnx_has_steering checks.
- The SA3 adapter and its TRT path: acestep/streaming/ (sa3 adapter / backend), acestep/engine/
  sa3_stream_helpers.py, the SA3 export/build scripts (find them; memory says SA3 DiT parity requires
  fp16mixed STRONGLY_TYPED, never BF16, cos >= 0.9998 per step; engines are versioned, keep the old).
- demos/sa3 (static page on the client SDK; the user said it is fine as is: add, do not restyle).
- notes/family_merge/beyond/tada_vs_demon_steering.md and frontier_capabilities.md (TADA section):
  TADA = arXiv 2602.11910, MIT, github.com/luk-st/steer-audio. Method: activation patching sweep to
  localise the responsible blocks, then CAA difference-of-means vectors (plus SAEs and slider LoRAs,
  which we do NOT port). Cite the paper in docs as the method source; that is proper attribution.

## Design (implement in this order; one commit per step; tests green at each)
1. Generic seam. A ModelAdapter (and the ACE path) declares its steering layout: number of steerable
   blocks, hidden size, hook point name (post_block_residual), and whether the loaded engine carries
   the `steering` input. StreamPipeline owns ONE steering slot that is family-agnostic: a [B, L, H]
   additive tensor per step, filled from active vectors times knob values times the per-step policy
   curve. Eager: hooks on the adapter's block list. TRT: the engine input. ACE keeps its current
   behaviour exactly (all existing steering tests pass unchanged; existing ACE engines load unchanged).
2. Vector packs as data. One file per vector (safetensors + json sidecar, or one json+npz; pick the
   simplest that round-trips): family, checkpoint id, hook, block index, hidden size, name, label,
   method (caa_diff_means), norm, provenance (prompt pairs, sample count, steps, date). A loader finds
   packs under a configured directory, keeps only those matching the booted family and checkpoint, and
   registers each as a knob `steer_<name>` through the same registry the ACE axes use today, so the
   knob manifest and the wire contract carry them with no per-family code. If the ACE axes already
   live in a file format, reuse that format rather than inventing a second one; if they live in code,
   leave them there and make the pack loader additive. Regenerate wire types only if the registry
   shape changes.
3. SA3 engine with the steering input. Add `steering` [B, num_blocks, hidden] to the SA3 DiT ONNX
   export with the same additive post-block convention as ACE. Rebuild the medium engine (fp16mixed
   STRONGLY_TYPED; idle GPU; version it, keep the previous engine in place). Gates: zero steering vs
   the previous engine at the existing parity bar (cos >= 0.9998 per step, or bit-identical if that is
   what the current gate does); nonzero steering TRT vs eager at the same bar; tick time before/after
   (the extra input must cost about nothing; report the number). Record engine path and sha in status.
4. Discovery tool (the portable part of TADA). `scripts/steering/discover.py --family sa3 --checkpoint
   <id> --concept <name> --pos "<prompt>" --neg "<prompt>" ...`: run N paired generations per concept
   (N >= 32, same seeds across the pair), capture post-block residuals at every block at the steps the
   ring actually uses, compute difference-of-means per block, pick the block by the TADA patching
   criterion (patch the positive activation into the negative run at one block, measure the effect
   on a cheap proxy: CLAP similarity to the positive prompt if CLAP is already in the repo, otherwise
   a log-mel spectral statistic appropriate to the concept) and write packs. Must work for any family
   that declares a steering layout; ACE only needs the layout declaration to run it.
   Compute SA3 packs for the four axes the ACE demo has (bright, warm, rough, density) so the user can
   compare like for like, plus up to two more if cheap (e.g. "acoustic vs electronic", "sparse vs
   dense percussion"). Sanity check each vector on the live stream: knob at 0 is a no-op (bit-identical
   output), knob at +1/-1 moves the proxy the right way; report numbers.
5. /sa3 demo. Add a Steer section that renders a slider for every `steer_*` knob in the manifest and
   nothing else (no family-specific code in the page). Match the existing page's look; small change.
   Verify on the GPU with the real server, then stop the server and write
   notes/family_merge/demo_logs/LAUNCH_sa3_steer.sh (same pattern as LAUNCH_minimax.sh, worktree
   DEMON-steer, checkpoint the SA3 medium alias) and HOWTO_sa3_steer.md for the parent to run for the
   user. Idle canvas shows a flat baseline, never fake data.
6. Docs: docs/ steering page (or the FAMILIES/README steering mention) updated: the seam, the pack
   format, how to run discovery for a new family, TADA citation. docs/FAMILIES.md SA3 section gains
   the steering line and numbers.

## Tests
Unit tests through the real StreamPipeline with a tiny fake adapter for the generic slot (zero = no-op
bit-identical; vector applied at the declared block only; policy curve scales per step), pack loader
round-trip and family filtering, manifest exposure of `steer_*` knobs, ACE regression (every existing
steering test passes unmodified), SA3 export carries the input (CPU-checkable on the ONNX graph if an
export fixture exists). Full tests/unit green on the branch.

## Deliverables
Commits on ryanontheinside/feat/steering-generic, status file, notes/family_merge/pr_steering.md (PR
body, base main, no attribution footer), LAUNCH/HOWTO scripts, packs under the configured directory
(report the path; data goes on E: if large, per the repo's data convention). Hand back: HEAD, engine
version and parity numbers, tick before/after, per-axis sanity numbers, test counts, demo verified
yes/no, where the launch script is. If blocked (engine will not build, parity fails), stop at the last
green step, write the finding, hand back FLAG.
