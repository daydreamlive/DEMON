# Pedalboard status (2026-10-05)

- Commits on DEMON-steer `ryanontheinside/feat/steering-generic` (unpushed): b9ab3540 server range+meta, 686dacf6 demo pedals, c2900f55 CSS centring.
- Server: `knobs.py` adds optional `KnobSpec.meta`, emitted as `meta` by `catalog_from_specs`; `steering_pack_spec(category, gain, flags)` sets range per sign = calibrated_gain x `STEERING_PACK_HEADROOM` (1.25); no gain keeps -30..30. Meta: label, category, blurb, block, calibrated, cutoff {pos,neg}, cutoff_reached {pos,neg} (accepts pos_reached/neg_reached or `reached`; null neg mirrors pos), headroom, flags (screening.flags).
- `packs.py` `PackSteering.knob_specs` reads provenance.category / calibrated_gain / screening.flags. Coercion now clamps to the per-pack range (60 passes for a gain-48 pack). `types/knobs.ts` gains `meta?`.
- Homonym guard untouched (placeholder `steer_<pack>` has no gain, so still +-30).
- Client `demos/sa3`: one pedal per category around the centre SA3 pedal (GENRE, INSTRUMENT, SFX, PRODUCTION, MOOD, SPACE, TONE = timbre+dynamics+rhythm, ABSTRACT, MISC); first four beside the main pedal, rest below; 2 cols <1200px, 1 col <800px.
- The 5 legacy packs have no category field; a client map (`LEGACY_PACK_PEDAL`) puts them on TONE. Other uncategorised packs go to MISC.
- Bypass footswitch + LED per pedal: sends 0 for that pedal's knobs, keeps positions. No ALL-BYPASS master (not built).
- Knobs: bipolar, 0 at 12 o'clock, each sign scaled to its own max; gold tick at 80% (the cutoff) per side; hollow tick + gold dot on label where cutoff_reached is false; tooltip = blurb + description; label = meta.label.
- Point the server at a packs dir: `DEMON_STEERING_PACKS_DIR=<dir>` (recursive scan; default `<models>/steering_packs`). Test packs (5 real copies + tango/riser/dreamy/gritty/koto/mystery, headers edited) are in the agent scratchpad `test_packs/`, not in the real dir.
- Tests: test_steering_packs (2 new tests), test_knob_homonyms, test_steering_seam, test_sa3_steering_engine, test_static_site pass; 4 test_lora_facade_session drain failures are pre-existing (fail with the change stashed).
- UI check: static harness serving demos/sa3 with a stub SDK that replays the manifest built by the real server code (`load_session_packs` + `sa3_knob_specs` + `catalog_from_specs`) under DEMON_STEERING_PACKS_DIR. Screenshot: `demo_pedalboard_screenshot.png`. Probed in headless Chromium: End=60, Home=-7.5 on riser, bypass sends 0 and restores, 10 ticks / 2 hollow / 2 dots, no horizontal scroll at the ~500px headless minimum.
- NOT verified: a live session on the real server/GPU (no SA3 boot run), true phone width below ~500px, audible effect of values past the old +-30 cap, the 46 real packs (not local).
- No web (vitest/Playwright) tests exist for demos/sa3.
