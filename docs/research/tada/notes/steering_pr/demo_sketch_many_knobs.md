# Demo sketch: a pedalboard for the 46 many-knobs packs

Written 2026-10-05. Read-only survey of DEMON-steer (ryanontheinside/feat/steering-generic, tip 83f5b77b).
Line refs are to that worktree unless marked.

## What exists today
- Demo: demos/sa3 (index.html:67-72 `#steer-pedal` / `#steer-knobs`; sa3.js). Two pedals: main + "STEER".
- Discovery: server scans `steering_packs_dir()` recursively (paths.py:219-230, packs.py:218), keeps packs whose
  family/checkpoint/layout match (packs.py:224-234), one knob `steer_<name>` per pack (packs.py:261-273,
  knobs.py:257-289) appended via sa3_backend.knob_specs (sa3_backend.py:815-818). Client splits the manifest by
  the `steer_` prefix (sa3.js:14-15, 492-493) and builds every steer knob with numericKnob into one grid (sa3.js:357-360).
  catalog.py is the legacy ACE `.pt` catalog; hub.py is only the HF bundle fetcher. Neither touches packs.
- Summing: already N-way. Each non-zero pack becomes one config (packs.py:280-295), merged with other sources
  (diffusion_backend.py:99-109), grouped by block (stream.py:1868-1885), and added into the one
  `[B, num_blocks, H]` tensor (stream.py:946-962, 1005-1030) that feeds the TRT `steering [1,24,1536]` input
  (trt/sa3_steering_onnx.py:9). Packs on the same block simply add.
- Range: every pack knob is -30..+30, default 0 (knobs.py:224, 281-282). The wire carries only type, default,
  min, max, group ("steering" for all), bank, description (knobs.py:368-380). Label/category/gain never reach the client.
- Per-pack provenance written by make_packs.py (DEMON-steer-bench/scripts/steering_bench/make_packs.py:103-131):
  `provenance.category`, `provenance.calibrated_gain {pos, neg, pos_reached, neg_reached, neg_source}`
  (knob value at the LPAPS cut), `provenance.screening {slope_per_lpaps, flags (needs_ear), max_cos_accepted, ...}`,
  plus top-level `label`, `blurb`, `block`, `magnitude` (= norm x 0.1).

## Hard limits found
- No pack slot cap. MANUAL_SLOT_CAP = 16 (policy.py:310) is the ACE manual-catalog slot cap, not packs.
- Per-forward cost scales with ACTIVE (non-zero) packs: stream.py:959-961 does, per pack, a `vector.to(device)`
  (pack vectors live on CPU, packs.py:176, so that is an H2D copy) plus an indexed add. 46 live knobs = ~46 tiny
  copies + ~46 index_put launches per forward, every forward, even when no knob moved. Estimate 0.5-2 ms CPU per
  forward; unmeasured. Knobs at 0 cost nothing.
- Knob values clamp at |30|. 7 of 46 knobs never reached the LPAPS cut in screening (summary.csv cut_reached=0),
  so their gain is "largest probe"; any knob whose gain > 30 cannot reach its cut on the wire.
- Gains were estimated at 10 s ARC, audio tokens only (make_packs.py:132-133); the 54 s stream may differ.

## Pedal layout (accepted.csv x concept_catalogue.csv)
| pedal | knobs |
|---|---|
| GENRE (11) | childrens hardstyle drone tango surf_rock post_rock samba black_metal horror_score lofi_hip_hop future_bass |
| INSTRUMENT (8) | strummed_guitar synth_pad ukulele koto synthesizer acid_303 jingle_bells timpani |
| SFX (7) | rain_on_surface riser ui_click impact_boom jet alarm_clock phone_ring |
| PRODUCTION (7) | glitchy radio_filtered live_recording solo_instrument analog mains_hum sfx_laden |
| MOOD (4) | sensual aggressive playful dreamy |
| SPACE (3) | distant close_mic wide_stereo |
| TONE (5): timbre/dynamics/rhythm | gritty dissonant intense polyrhythmic (+ existing bright warm rough density percussive) |
| ABSTRACT (2) | abs_industrial abs_cinematic |
- Genre (11) and instrument (8) exceed the 1..8 grid; split into two rows on the plate or two pedals (GENRE I/II).
- Each pedal: bypass footswitch (client-side: sends 0 for its knobs, remembers values; LED shows state).
- Knob face is bipolar, 12 o'clock = 0. Full clockwise = +calibrated_gain.pos, full CCW = -calibrated_gain.neg,
  so full scale = the fidelity cutoff. Clamp to 30. Badge knobs with `needs_ear` or `pos_reached=false`.
- Tooltip = blurb. Pedal colour per category. One "ALL BYPASS" master.

## Smallest code changes
1. Manifest carries pack metadata (~0.5 day). Add optional `meta: dict` to KnobSpec (knobs.py:38-64), emit it in
   catalog_from_specs when set (knobs.py:368+), fill it in steering_pack_spec from the pack: label, category
   (provenance.category), gain (calibrated_gain pos/neg, reached flags), flags. Keep group "steering" and range +-30
   so the homonym guard (families.py:447-462, tests/unit/test_knob_homonyms.py) is untouched. Unit test on the manifest.
   (The "catalog: category field" item lands here; catalog.py needs no change.)
2. Engine summing (~0 to 0.5 day). N-way sum already works; nothing is required for correctness. Optional perf:
   move pack vectors to device once at attach (or cache device copies on `_SteeringApply`), and cache the filled
   `[num_blocks, H]` per (snapshot, step) so a no-change tick is one copy. Measure first with 46 knobs non-zero.
   (The "hub" item: hub.py is the HF fetcher; the summing lives in stream.py and needs no new code.)
3. UI pedal groups (~1 to 1.5 days). sa3.js: group `state.steer` by meta.category into a pedal per category
   (reuse the plugin-steer faceplate, index.html:67-72), per-pedal bypass, per-knob display scaling to +-gain,
   fallback to one "STEER" pedal when meta is absent (old packs). CSS: category colours, wrap 2 rows. Phone width check.
4. Install (~10 min). Copy the 46 packs into ~/.daydream-scope/models/demon/steering_packs/sa3/medium/; no name
   collides with the 5 there. With no code change they all load and appear as 51 knobs in the single STEER pedal.
