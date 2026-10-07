# Forgotten work inventory (section B, 2026-10-03)

This was read-only archaeology. Sources: `git branch -vv`, `git worktree list`, `git stash list` and
`git rev-list --left-right --count origin/main...<b>` in C:\_dev\projects\DEMON (origin/main = b440920,
2026-10-01); `git status` in every DEMON* checkout and in the DEMON-adjacent repos; `gh pr list` (read-only);
notes/; MEMORY.md leads, each checked against git. "Behind" is the number of commits behind origin/main.
Squash merges make `git cherry` useless, so I decided "merged" from PR state, not from commit
ancestry. All the DEMON* folders are worktrees of the ONE repo C:\_dev\projects\DEMON\.git, so local-only
branches and stashes have exactly one copy on disk.

## 0. Corrections to memory (verified)

- Plugin API is ON MAIN: #331 merged 2026-09-23. feat/plugins-pr and feat/ncas are merged, so DEMON-ncas-wt is no longer
  the only home of the plugin API.
- One-shot MuScriptor transcription is ON MAIN: #348 (Rafal, `midi_transcribe` wire command,
  origin/main:acestep/analysis/midi_transcribe.py). Only the LIVE per-session transcriber (#345) is still a spike.
- Text-only sessions are ON MAIN: #342 (Rafal, feat/text-only-sessions). Ryan's true-t2m 5Hz-LM work is a different thing and
  is NOT on main (it is stash@{0}).
- Audio edit layer is committed and pushed now: PR #311 draft. It is no longer "uncommitted on DEMON_alt".
- MuseTimbre extension is committed and pushed: private-audio-extension feat/reference-demo 72d4b49, which contains demos/musetimbre.
- The TRT output-buffer aliasing bug is STILL ON MAIN: origin/main:acestep/engine/stream.py:889-890 returns
  `out[:, :T, :].to(self._dtype)` / `out.to(self._dtype)`, and `.to()` is a no-op view for bf16. The fix (`out.clone()` when the dtype
  matches) exists only in the local commit 446af4c (stream.py:788-789 there) and in the perf/throughput stack.
- NVFP4 plugin source is not lost. It is tracked on archive/ryanontheinside/xl-fp8-wip (pushed), at
  acestep/engine/trt/plugins/nvfp4_linear.

## 1. Would be LOST if the disk died (local only or uncommitted)

Ranked by how much it would hurt:

1. **instrument-controlnet repo (C:\_dev\projects\instrument-controlnet): NO REMOTE AT ALL.** 22 branches and 1538
   commits, plus 487 untracked files on master. Worktree instrument-controlnet-xattn-mcl (feat/xattn-mcl) has 57 modified and 46
   untracked files; instrument-controlnet-accomp-protocol is on accomp/bed-guidance. This repo holds all the ControlNet and adapter training
   code: groove, melody-transfer, timbre, density, xattn, mix-adapter and the judge protocols. Single biggest risk on the machine.
   Next step: create a private GitHub repo and push all branches (Ryan's call, because it is a public-facing action).
2. **DEMON stash list (32 stashes, local .git only).** The ones worth keeping:
   - stash@{0} "t2m" (2026-07-20): scripts/experiments/text2music, t2m_lm.py plus runners E1 to E6, 1716 lines. The true text-to-music
     work. The working tree has only __pycache__ left in that folder.
   - stash@{1} "playehad pinning" (2026-07-10): pipeline_runner/session changes plus tests/unit/test_playhead_pinning.py, 380 lines.
   - stash@{2} "songswap" (2026-07-01): examples/song_bridge.py (repaint bridge), prompt_enhancer changes plus a test, 595 lines.
   - stash@{3} "sa3 abboration tests": SA3 aberration repro harness (sa3_aberration_repro.py, heard-path repro .js), 9651 lines.
   - stash@{6} "midi input": demos/jam MIDI-input synth demo, 4759 lines.
   - stash@{7} "SA3-SCRIPTS": sa3_fp8_verify_artifacts, latency/throughput/concurrency probes, sa3_t2m_spike, 4498 lines.
   - stash@{12} "PROJECT PAGE": experiments/morph_cover paper evidence plus renders, 2295 lines.
   - stash@{14}: docs/bench plot_all.py plus TRT sweep CSVs. stash@{17}: MelBandRoformer simulated stem extract (483 lines).
   - stash@{18} "post hog mcp": LoRA store plus test_local_config/test_lora_metadata, 1849 lines.
   - The rest (@5, 9, 10, 15, 16, 19 to 23, 25, 26) are config.json or .mcp.json tweaks: dead.
3. **Local-only DEMON branches with unique work** (no remote ref):
   - ryanontheinside/exp/schedule-migration: 2 commits (446af4c shadow migration + TRT alias fix, 2685 lines, 18 CPU + 5 GPU tests). 140 behind.
   - research/walk-mode: 4 commits (chained-repaint walk mode, free-running chain, live-path driver). 114 behind.
     DEMON_walk ALSO has 12 uncommitted files (211 lines: external-GPU-work pause in pipeline_runner, TRT vae_encode selection).
   - ryanontheinside/feat/sa3-sweep-characterization: 11 commits, 5683 lines (smooth_lora training loop + sweep harness). 31 behind.
   - feat/melody-sweep-local: 4 commits (Phase 0 sweep-characterization harness 5456a35 on top of the plugin API). 22 behind.
   - ryanontheinside/feat/plugin: 2 commits (artifact recipe plugin runner, acestep/plugins/runner.py + tests, 2736 lines). 39 behind.
   - ryanontheinside/feat/song-swap-07012026: 4 commits ("X0 MORPH DECENT", song bridge, sa3_song_morph_x0). 41 behind.
   - ryanontheinside/experiments/instrument-lora: 2 commits (3955 lines, recipe plus findings). 84 behind.
   - ryanontheinside/feat/models/M4L-on-rtinput and m4l-split: 17 commits each (Ableton M4L suite; m4l-split is the split-out history
     in DEMON_arp_rebase). 80 and 482 behind.
   - ryanontheinside/feat/models/sa3-controls: 8 commits (sa3 controls page plus TRT builder). Probably superseded by #231 (unverified).
   - ryanontheinside/model-family/startup-behavior: 1 commit (ModelSessionBar / start overlay UI).
   - ryanontheinside/experiments/rave: 1 commit (RAVE tile, rave_diagnostic.py, 545 lines). 364 behind.
   - ryanontheinside/experiments/x0target-morph-4-paper (9), archive/spectral-control (10, research history for merged #232),
     experiments/in-flight-mutable-schedule (2), experiments/scratch/trt-non-determinism (1), feat/lokr-dora-support (1, lora.py +550),
     arch/int-8 + archive/int8-research-2026-05-01 (INT8 VAE variants A to I), demos/arp-demon, demos/wip-motion-graph-unstaged,
     daydream-ambien-oneknob (fun demos), scratch/demon-branding*, scratch/demon-letters, backups/* (pre-rebase copies).
   - perf/latency/01-05 and perf/throughput/01-06 are local, but they are contained in the PUSHED tips 06 and 07, so they are safe.
4. **Family restack (in-flight lane).** docs/readme-platform (36289ee), docs/project-page-families (b5c068b) and
   feat/models/yue2-family (b2f7bf5) have NO remote ref. yue2-ring, mrt2-family and minimax-family are pushed, but the remote copy is
   OLDER than local (local 4137685 / 486343c / 654dd0e vs remote 1e2dccf / 4d5414b / 9cab12e).
5. **Uncommitted files in checkouts:**
   - DEMON primary: SA3 prompt-latency profiling instrumentation (sa3_stream_helpers.py, sa3_backend.py; gated by DEMON_SA3_PROMPT_PROFILE),
     scripts/sa3/sa3_prompt_ablation.py + tests/unit/test_sa3_prompt_ablation.py, scripts/sa3/sa3_depth_knob_sweep.py,
     scripts/spikes/render_midi_video.py, the muscriptor_sim_poc.py diff, and CLAUDE_SA3_RECOVERY_*.md. Not on any branch.
     The YuE2 scripts/docs there ARE identical to origin/ryanontheinside/spike/yue2, so they are safe.
   - DEMON_TUNEJURY: TuneJury corpus, scoring and listening-test package scripts plus docs/DEMON_DJ_CONTROLLER_RESEARCH_NOTE.md (all untracked).
   - private-audio-extension: 5 modified demos/reference files (+399/-68: timeline, app, library).
   - rtmg-vst: 32 unpushed commits across 15 branches (e.g. ryan/m4l-clip-inputs 5, ryan/feat/sa3-vsts 5,
     backup/sampler-on-fixes-on-bpm 6, ryan/feat/plugin 1, pr-186 1). rtmg-vst-capt has STAGED RTMGProcessor changes plus
     capt_proc.patch. Plan files CONSOLIDATION_PLAN.md and SAMPLER_EDIT_MODE_PLAN.md are untracked.
   - underfit (personal repo): 10 unpushed commits (hypernetwork options), 5 modified, 86 untracked.
6. **Never in git:** DEMON/notes (5.5 GB, ignored; includes RECIPE_PLUGIN_PLAN.md, MODEL_SWAP_PLAN.md,
   knob_to_ear_stack_PLAN.md, mirelo-audio-to-midi/00-04 research, musetimbre runbooks), C:\_dev\projects\DEMON_PLAN_CLIENT
   (client-SDK plan series 00-09), C:\_dev\projects\DEMON-staged-backups (sa3-controlnet patch 2026-07-28).

## 2. Inventory by theme

State key: M = merged, PR = open PR (draft = d), P = pushed, no PR, L = local only, U = uncommitted, S = stash.
Each row: item | where | state | last commit | behind | verdict.

### New model families and models
- YuE2 ring family | feat/models/yue2-ring (DEMON-fam-yue2b) | PR #366, local ahead of remote | 10-03 | 0 | in flight (family lane)
- MiniMax-Music3 family | feat/models/minimax-family | P (stale remote) | 10-03 | 0 | in flight. Old PR #332 (feat/minimax-music, 22 behind) = superseded
- MRT2 family | feat/models/mrt2-family | P (stale remote) | 10-03 | 0 | in flight. Old PR #230 (126 behind) = superseded
- YuE2 spike | spike/yue2 | PR #355 d | 09-22 | 8 | research record only
- True t2m (5Hz LM to context slot) | stash@{0} | S | 07-20 | n/a | needs small work: commit to a branch, then compare against #342
- 5Hz LM hints (old) | feat/lm-hints | PR #24 d | 05-06 | 459 | dead (superseded by the t2m stash; memory names its defects)
- RAVE in the demo | experiments/rave + scope-rave repo (pushed) | L | 05-09 | 364 | research record only (diagnostic says RAVE dislikes dense mixes)
- XL / fp8 / NVFP4 | archive/ryanontheinside/xl-fp8-wip, arch/xl, archive/fp8-2b-research-vram | P | 05-04..05-15 | 300+ | research record only
- SA3 fp16 checkpoint converter (Rafal) | sa3-fp16-checkpoint | PR #299 | 07-01 | n/a | not Ryan's; listed for context

### Control features
- Audio edit layer (repaint/extend/SA3 inpaint) | feat/audio-edit-backend-agnostic | PR #311 d | 07-20 | 34 | needs small work (rebase onto FamilySpec, wire verb)
- Shadow migration (exact re-simulation on knob change) | exp/schedule-migration | L | 06-07 | 140 | needs work (heavy rebase over stream.py)
- Walk mode / free-running infinite generation | research/walk-mode + DEMON_walk dirty | L+U | 06-10 | 114 | needs work (commit, golden gate)
- Extend walk mode (older) | feat/extend-walk-mode | P | 05-27 | 208 | dead (superseded by research/walk-mode)
- Playhead pinning | stash@{1} | S | 07-10 | n/a | needs small work (has unit tests)
- Smart looping | feat/smart-looping | PR #179 d | 05-30 | 183 | needs rebase. Might overlap enhanced looping #203 (unverified)
- velocity_ema knob | feat/new-knobs | PR #118 | 05-20 | 243 | dead unless wanted (54 lines; trivial to redo)
- Lyric transcription (Qwen2.5-Omni ACE transcriber) | feat/lyric-conditioning-clean | PR #114 | 05-20 | 247 | research record only (7B model, VRAM)
- x0 song morph / recipe plugin | feat/song-swap-07012026, feat/plugin, notes/RECIPE_PLUGIN_PLAN.md | L | 07-07 | 39-41 | needs work: re-base on the #331 plugin API
- smooth_lora (even denoise dial via LoRA) | feat/sa3-sweep-characterization | L | 08-02 | 31 | research record only (no result recorded in git)
- LoKr/DoRA support | feat/lokr-dora-support | L | 05-09 | 367 | dead unless a LoKr LoRA shows up
- SA3 queue cap configurable | fix/sa3-queue-cap | PR #318 | 08-01 | 31 | ship-ready (1 commit)
- Solo vocabulary guard | fix/solo-guard-vocab | PR #354 | 09-18 | 10 | ship-ready
- Config numeric facts emitter | ryan/feat/config-contract-numeric-facts | PR #301 d | 07-02 | 40 | needs small work
- SA3 denoise product mapping | #321 merged; revert-321 remote branch exists | M | 08-18 | n/a | no revert landed on main. Remote branch is dead

### Performance
- TRT output-buffer aliasing fix (CORRECTNESS bug on main) | inside 446af4c and the perf/throughput stack | L | 06-07 | n/a | ship-ready as a 3-line standalone PR
- Knob-to-ear stack (audible_first 234 to 58 ms) | feat/latency/06-near-playhead-repatch (+01-05) | PR #240 d | 06-09 | 114 | needs rebase + golden (ask first)
- Tick throughput stack (+8.3% CFG tick) | perf/throughput/07-bench-stream-tick (+01-06) | PR #241 d | 06-09 | 114 | needs rebase + golden (ask first)
- SA3 prompt-latency profiling | DEMON primary U + perf/sa3-prompt-update-latency (0 ahead) | U | 08-18 | n/a | research record only
- INT8 VAE decode | feat/vae-decode-int8 PR #1 + arch/int-8 L | PR | 05-01 | 479 | dead (SAME-S decode is ~11 ms flat, so no need)
- StreamA2A feature bank | experiments/streamA2A | P | 05-10 | 364 | research record only (memory marks it historical)
- TRT non-determinism harness | experiments/scratch/trt-non-determinism | L | 05-22 | 230 | research record only

### Integrations
- Ableton M4L suite (DEMON for Live) | feat/models/M4L-on-rtinput, m4l-split (DEMON_arp_rebase) | L | 06-15 | 80 / 482 | needs work. The plan says move it out of DEMON (DEMON_PLAN_CLIENT/02)
- rtmg-vst M4L / VST / sampler branches | rtmg-vst (+capt, m4l, midi, native-bridge, sampler worktrees) | 32 unpushed commits | 06-08..07-20 | n/a | needs triage. Push the backups at least
- DEMON ONE clip/locator preset recall | rtmg-vst ryan/m4l-clip-inputs (5 unpushed) | L | 06-18 | n/a | needs small work (memory: live-verified, scope gap on Timbre/Src/Feed)
- Plugin auto-update endpoint | demon-public-demo feat/plugin-update-endpoints (pushed) + rtmg-vst feat/plugin-auto-update (1 unpushed) | P/L | 07-01 | n/a | needs triage

### Demos
- Backing-track jam pages (5 ideas) | private-audio-extension-jam spike/backing-track | P | 09-30 | n/a | needs server work (bar-accurate chart apply)
- MuseTimbre live demo | private-audio-extension feat/reference-demo | P (+5 U files in demos/reference) | 10-02 | n/a | ship-ready per memory (fast fp16 engine). Commit the 5 files
- Reference adapter v4 painting demo | private-audio-extension feat/reference-adapter (in feat/reference-demo) | P | 09-25 | n/a | needs listening gate
- MIDI-input jam synth | stash@{6} | S | 06-10 | n/a | dead (the arp/rt-input demos shipped in #235)
- Arp hand demo, motion demo, truchet/threejs | demos/arp-demon, demos/wip-motion-graph-unstaged, daydream-ambien-oneknob | L | 06-06..06-12 | 87-149 | dead or fun (family demos should be lightweight static pages, per memory)
- Face warp / DJ controller | docs/tunejury-dj-controller (DEMON_TUNEJURY) | L+U | 06-16 | 75 | research record only
- Project page morph evidence | x0target-morph-4-paper, stash@{12} | L/S | 05-26 | 230 | dead (paper released 2026-05-27)

### Research
- Instrument-isolation LoRA | experiments/instrument-lora | L | 06-13 | 84 | research record only
- ControlNets / adapters training | instrument-controlnet (no remote) | L | 10-03 | n/a | PROTECT FIRST
- TuneJury rater gate | DEMON_TUNEJURY untracked | U | 06-16 | n/a | research record only (underpowered)
- SA3 aberration repro | stash@{3} | S | 07-01 | n/a | research record only. Keep it, because the repro harness is reusable
- Schedule migration in-place operators | in-flight-mutable-schedule, archive/schedule-migration | L/P | 05-24/25 | 223 | dead (measured: in-place buys nothing)
- MelBandRoformer simulated extract | stash@{17}, marco/feat/melbandreformer | S/P | 05-20 | 247 | dead (stems merged via #244)
- Mirelo / MuScriptor landscape | notes/mirelo-audio-to-midi 00-04 | not git | 09-10 | n/a | research record. Feeds sections C and E

### Dead worktrees (safe to retire later; I did not touch them)
DEMON-sa3-denoise-product, DEMON-sa3-schedule, DEMON-sa3-upstream-pin: detached, with dirty pre-merge copies of #319-321. Main already has
sa3_denoise_mapping.py and test_sa3_trt_engine_identity.py.
DEMON-swapfix-wt (#335 merged). DEMON-ncas-wt: its dirty sa3_backend diff duplicates #335. DEMON-prompt-enhancer (#283).
DEMON-example-apps-docs (#282). DEMON_alt2 (#287). DEMON_sa3ref (superseded by #231).
.claude/worktrees/agent-ace4b237... (be4d954, inside backups). DEMON_alt is parked on a pushed archive branch.
**"DEMON_alt - Copy"** is a raw folder copy (around 2026-06-09) whose .git file points at DEMON_alt's worktree gitdir: any git command
run inside it corrupts DEMON_alt's index. I did not check whether it holds unique files (unverified). Move it to the trash only after a diff.

## 3. Top 10 unshipped by value, with the next concrete step

1. **TRT aliasing fix**: main's full-CFG TRT streaming is silently unconditional (stream.py:889-890). Next: cherry-pick only the
   clone hunk of 446af4c plus probe_trt_alias as a unit test. Open a standalone PR. One hour.
2. **Protect instrument-controlnet**: next: a private remote, then push all 22 branches. Push the 15 rtmg-vst branches with unpushed
   commits too. (Not shipping, but it removes the biggest loss risk.)
3. **Live MuScriptor transcription tap** (#345 spike; one-shot #348 is on main): next is section C's stage 1, a server tap that
   reuses acestep/analysis/midi_transcribe.py.
4. **Knob-to-ear + throughput stacks** (#240, #241): measured 234 to 58 ms first-audible and +8.3% CFG tick. Next: rebase both onto main
   (114 behind, stream.py moved), then run the golden gate with the user's OK. Item 1 lands as part of #241 anyway.
5. **MuseTimbre instrument transfer** (private, ship-ready engine): next: commit the 5 demos/reference edits, then merge
   feat/reference-demo into private master behind the plugin API that is now on main (#331).
6. **Audio edit layer** (#311): next: rebase onto FamilySpec (#356-361), then add the wire verb and an SDK method. Gate extend on the ear check.
7. **True t2m via 5Hz LM** (stash@{0}): next: `git stash branch` it onto a new research branch and push it as a backup. Then decide
   whether it replaces or complements #342 text-only sessions.
8. **Shadow migration** (local 446af4c): exact knob changes and 1.24-1.34x faster first-B at depth 4. Next: push the branch as a backup,
   then rebase after #241. Most of the conflict is stream.py.
9. **Walk mode** (research/walk-mode + 12 dirty files): next: commit the DEMON_walk dirty files to the branch, push a backup, then
   golden gate. This is the free-running / long-song story.
10. **Small ship-ready PRs**: #354 solo-guard, #318 queue cap, #301 numeric facts. Next: rebase and request review.
    Close or rebase the stale May PRs (#1, #22, #24, #114, #118, #179) so the open list reflects reality.

What I would NOT do: revive INT8 VAE, in-place schedule operators, the vibe adapter, the old lm-hints, or RAVE in the ACE loop. Each
has a measured negative result. Do not open new PRs from the backups/* branches.

## 4. Other model-family or music-adjacent capabilities found

- **MuScriptor audio-to-MIDI**: live spike #345 (RTF 0.21 on DEMON output; MIDI trails audio by about 2 s;
  notes/mirelo-audio-to-midi/04-spike-results.md). One-shot #348 is on main. Weights are CC BY-NC 4.0.
- **Walk mode / free-running infinite generation**: research/walk-mode (local).
- **Audio edit layer**: repaint, extend, SA3 inpaint (#311).
- **Spectral steering**: SHIPPED (#232/#233 merged 2026-06-09). Nothing pending.
- **Reference-audio adapter v2-v4** (xattn MCL, reveal dial, gauss|shuffle): private-audio-extension (pushed). Training code lives in instrument-controlnet (no remote).
- **ControlNets** (chord/groove/melody/density/accompaniment): private plugin (pushed); training repo has no remote.
- **MuseTimbre zero-shot instrument transfer** on SA3: private extension, fast fp16 engine at 35 ms per tick.
- **True text-to-music via the ACE 5Hz LM**: stash@{0}.
- **Lyric transcription** (ACE transcriber, Qwen2.5-Omni-7B): PR #114, stale.
- **RAVE timbre transfer**: experiments/rave (local) + scope-rave repo. Old CPU-friendly autoencoder line; see section D.
- **StreamA2A feature bank** (style transfer through cached features): experiments/streamA2A (pushed, historical).
- **x0 song morph / song bridge** (A to B transitions): song-swap-07012026, stash@{2}, research/walk-mode examples/song_bridge.py.
- **Recipe plugin system** (out-of-tree recipes callable by any client): feat/plugin + notes/RECIPE_PLUGIN_PLAN.md.
- **Runtime model swap** (model-agnostic server): notes/MODEL_SWAP_PLAN.md (plan only, 2026-07-02). Partly overtaken by FamilySpec.
- **Instrument-isolation LoRA, smooth_lora, underfit hypernetwork LoRA**: training-side research.
- **TuneJury taste rater**: evaluation tooling, underpowered.
- **Stem separation (MelBandRoformer)**: shipped via #244.
