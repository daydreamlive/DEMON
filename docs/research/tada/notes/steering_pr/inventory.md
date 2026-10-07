# Steering PR inventory (read-only, 2026-10-05)

## 1. Branches and worktrees
Primary C:\_dev\projects\DEMON is on ryanontheinside/spike/muscriptor-midi (43f97ae7), dirty. main = b4409206.
Worktree for steering-generic: C:\_dev\projects\DEMON-steer, branch ryanontheinside/feat/steering-generic, HEAD 4beb08d4, clean (git status empty). Branch names carry the prefix `ryanontheinside/feat/`.
Other worktrees: tada-ace 64e565c5, tada-core cab0451c, tada-sa3 974ab049, tada-sae 7bf6e260, tada-sae-sa3 d5f4c63f (each in C:\_dev\projects\DEMON-<name>).

steering-generic commits over main (merge-base = main b4409206), 6 commits, 28 files, +3396/-67:
- 1885aeec feat(steering): family-agnostic steering slot behind the adapter seam
- 12f08551 feat(steering): vector packs as data, registered as steer_* knobs
- 4581c568 feat(sa3): DiT TensorRT engine with the activation-steering input
- 7bf6e260 feat(steering): discovery and live-stream sanity tools
- 2eef961d feat(demos/sa3): Steer section with one slider per steering knob
- 4beb08d4 docs: steering seam, pack format and discovery

Engine files (diff stat): acestep/engine/model_adapter.py (+79), sa3_adapter.py (+56), sa3_context.py, sa3_internals.py (+20), sa3_trt.py (+73), stream.py (+219), trt/sa3_build.py (+104), trt/sa3_steering_onnx.py (new 233), paths.py (+18), steering/layout.py (new 50), steering/packs.py (new 321), streaming/ace_backend.py (+19), diffusion_backend.py (+43), families.py (+65), knobs.py (+35), sa3_backend.py (+38).
Tools: scripts/steering/discover.py (569), proxies.py (85), sanity_sa3_stream.py (190), scripts/sa3/sa3_trt_steer_parity.py (219).
Demo wiring: demos/sa3/index.html (+5), sa3.js (+64), styles.css (+21).
Docs: docs/STEERING.md (new 135), docs/FAMILIES.md (+19).
Tests: tests/unit/test_sa3_steering_engine.py (231), test_steering_packs.py (171), test_steering_seam.py (374).
Notes for the PR: notes/family_merge/pr_steering.md (draft PR body, base main), status_steering.md, 16_steering_generic.md.

Containment (git merge-base --is-ancestor steering-generic <b>):
- tada-core: YES, 6 commits above (ca02dca0 cross-attn hook point, de6a9cd3 ACE build/parity, 64e565c5 ACE runtime, 1e90846c TADA core, 20db2cce docs, cab0451c AUSteer).
- tada-ace: YES, 3 above (ca02dca0, de6a9cd3, 64e565c5).
- tada-sa3: YES, 29 above.
- tada-sae-sa3: YES, 36 above.
- tada-sae: NO, ahead=0; its tip 7bf6e260 is the 4th steering-generic commit (below steering-generic's tip, so steering-generic is not its ancestor).
So steering-generic is fully contained in tada-core, tada-ace, tada-sa3, tada-sae-sa3.

## 2. SA3 demo (route /sa3) in DEMON-steer/demos/sa3/
Files: demo.static.json (4 lines: {"route": "/sa3", "entry": "index.html"}), index.html (75 lines), sa3.js (584), styles.css (495). Mounted by demos/common/static_site.py (tests/unit/test_static_site.py). Client SDK imports at sa3.js:1-5 from "/sdk/demon-client.js": AudioPlayer, RemoteBackend, SLICE_FLAG_DELTA. Served by `python -m demos.realtime_motion_graph_web.server --port 1318 --checkpoint sa3-medium`.
Look: hardware-plugin faceplate (pedal style), no canvas, pure DOM + CSS. .plugin (styles.css ~l.41-60): dark gradient faceplate, max-width 560px, display grid, grid-template-rows auto x5, gap 16px, wood-grain ::before/::after layers. Markup order (index.html): brand-row, #knobs.knob-grid, [steer section l.20-23], .controls (prompts, blend slider, source/duration, Send Prompt), .title "sa3", .transport (power-btn Start/Stop, tick readout, status dot/text). Media query at styles.css:477 (max-width 620px).
Knobs: NOT a reusable component module; plain functions in sa3.js. numericKnob(name, entry) (sa3.js ~l.137-290) builds div.knob-cell > .knob-wrap > .knob (pointer rotor; drag, wheel, keyboard; increments from entry min/max/type) plus .knob-value and .knob-label; enumKnob (~l.292) is select-based; a bool knob follows (~l.325); renderKnobs() at l.355. Knob grid CSS: .knob-grid 3 columns (styles.css:101-107), .knob-wrap 84px (123-). All knobs commit through commitKnobValue(name, entry, value) (sa3.js:125) -> state.values -> sendParamsNow() -> state.remote.sendParams(state.values, positionSec); PARAMS_TICK_MS = 80. valueFromEntry(entry) at sa3.js:76 reads the manifest default.
Steering controls (commit 2eef961d):
- sa3.js:14-17 STEER_PREFIX = "steer_"; els.steer / els.steerSliders (l.25-26); state.steer (l.39).
- steerSlider(name, entry) sa3.js ~l.375-415: label.field-slot.blend-slot with input type=range, min/max from entry (code default -1..1, actual manifest -30..30), step (max-min)/600, label = name minus prefix with underscores to spaces, value readout toFixed(1), dblclick resets to default (0), input event -> commitKnobValue.
- renderSteer() ~l.416; start() splits the manifest at sa3.js ~l.529-537: names not starting with steer_ -> state.knobs (grid), steer_ -> state.steer; stop() clears state.steer.
- index.html:20-23: section#steer.steer-section (hidden) with .steer-head "Steer" and #steer-sliders.
- styles.css:245-264: .steer-section, .steer-head, .steer-sliders (vertical grid).
Binding: manifest from remote.knobManifest.knobs (client SDK RemoteBackend); slider writes knob id steer_<name> into the same params map as every other knob. Section hidden when the manifest has no steer_* knobs.
Packs on this machine for SA3 medium (HOWTO_sa3_steer.md): bright, density, percussive, rough, warm (dir C:\Users\ryanf\.daydream-scope\models\demon\steering_packs via DEMON_STEERING_PACKS_DIR).
Launch/HOWTO: notes/family_merge/demo_logs/LAUNCH_sa3_steer.sh, HOWTO_sa3_steer.md (port 1318, control bus 1319). Measured direction: bright and warm clear; percussive -10 works, +10 weak; rough/density unverified.

## 3. How steering knobs reach the backend
- acestep/streaming/knobs.py (DEMON-steer): STEERING_ALPHA_MAX = 30.0 (l.224); steering_axis_spec (l.227); steering_pack_spec(name, label, block, policy, blurb) (new, ~l.257-290): default 0.0, range -30..30, group "steering"; manual_slot_specs(slot_id, ...) (l.292): knobs man_src_<id> (int catalog index), man_layer_<id> (default 9, DiT inject layer), man_step_<id>, man_alpha_<id> (strength, 0 = off), group "manual".
- Pack knob name is steer_<name>; ACE built-in AUTO_AXES in acestep/steering/policy.py also use steer_* names (e.g. steer_bright); families._attach_steering_packs gets reserved_names=[ax.name for ax in AUTO_AXES] for ACE.
- acestep/streaming/families.py: _attach_steering_packs (~l.241-273), _make_acestep (attaches via ace_engine_steering_layout), _make_sa3 (layout = backend.pipeline.steering_layout()); universes _acestep_knob_universe / _sa3_knob_universe add one placeholder steer_<pack> spec. sa3_backend.sa3_knob_specs(loras, extension_specs, steering_specs) appends pack specs; SA3Backend picks the steering TRT engine via packs_available(family="sa3", checkpoint=model_id) -> prefer_steering.
- Packs: acestep/steering/packs.py (PackSteering, load_session_packs, packs_available); safetensors with `vector` tensor + `steering_pack` JSON metadata (format 1, family, checkpoint, hook post_block_residual, block 23, hidden_size 1536, magnitude, policy range/step/curve). Applied shift = knob x magnitude x policy_weight(step) x vector. Dir: DEMON_STEERING_PACKS_DIR or <models>/steering_packs, convention <family>/<checkpoint>/<name>.safetensors. Doc: docs/STEERING.md.
- Slot: acestep/engine/stream.py StreamPipeline single steering slot; adapter contract in model_adapter.py (steering_layout(), steering_blocks(), accepts_steering); SA3 engine name sa3_m_dit_steer_l{min}_{opt}_{max}, built by `python -m acestep.engine.trt.sa3_build --steer`.
- Manual steering slot API: MCP tools mcp__demon__add_manual_slot, pop_manual_slot, list_manual_steering_vectors (knobs man_*_<id> above). add_manual_slot is not defined in acestep/ python by grep.
- config.json: grep for "steer" in config.json files under DEMON-steer found nothing; no config.json entries for steering.

## 4. Codex CLI usage
- Installed: C:\Users\ryanf\AppData\Roaming\npm\codex (v0.160.0 per logs). `codex exec --help` lists -m/--model <MODEL>, -s/--sandbox <MODE>, -C/--cd <DIR>.
- Sol model id: gpt-6-sol (header "model: gpt-6-sol" in notes/family_merge/demo_logs/sol_mrt2_run1.log, sol_mrt2_run2.log, sol_minimax_run1.log, sol_yue2_run1.log). Log header also shows: provider openai, approval never, sandbox danger-full-access, reasoning effort high, workdir = the worktree, "Reading additional input from stdin" (prompt via stdin). Prompts saved in demo_logs/sol_{mrt2,minimax,yue2}_prompt.txt. The exact Sol command line is not recorded in any note. Run plan: notes/family_merge/12_mrt2_restyle_sol.md (Sol restyles /mrt2; cwd = worktree; Sol forbidden to commit; reviewer gates the diff; element ids/data hooks must remain).
- Astra: `codex exec -m gpt-6-astra` (notes/family_merge/12_mrt2_restyle_sol.md:19); full form `codex exec -m gpt-6-astra --cd C:\_dev\projects\DEMON-tada-sa3 -s read-only`, prompt via stdin (notes/family_merge/20_tada_skeptic.md:48, status_tada_skeptic.md:80).

## 5. Tests and last green state
- Steering: tests/unit/test_steering_seam.py, test_steering_packs.py, test_sa3_steering_engine.py (also test_sa3_adapter.py, test_sa3_backend.py, test_sa3_stream_pipeline.py touch SA3). Static demo mount: tests/unit/test_static_site.py. No web/tests or unit test covers the sa3.js sliders.
- Last recorded (notes/family_merge/status_steering.md:36, pr_steering.md:64): full tests/unit on steering-generic HEAD 4beb08d4: 874 passed, 4 skipped, 4 failed; the 4 failures are test_lora_facade_session.py drain tests, also failing on main (pre-existing).
- Behavioral: knob 0 is a bit-identical no-op (packs at 0 vs no packs, same engine, offline); tick ~52 ms steering engine vs ~43 ms fp8 (HOWTO_sa3_steer.md).
