# status: steering-generic (runbook 16)

- 2026-10-03 14:12 worktree C:\_dev\projects\DEMON-steer on ryanontheinside/feat/steering-generic from origin/main b4409206. GPU_LOCK taken (steering-generic).
- Ground truth: ACE axes live in CODE (acestep/steering/policy.py AUTO_AXES) + .pt files from HF bundle. Pack loader will be additive (leave ACE as-is).
- StreamPipeline already has set_steering(configs) with per-layer step-gated applies, eager hooks on self.decoder.layers, TRT fill of bufs["steering"] in _trt_forward (ACE only).
- SA3 TRT engines compile upstream's dit_fp16.onnx (no local export) -> steering input requires ONNX graph surgery on upstream proto.
- Step 1 DONE (commit see git log): SteeringLayout (acestep/steering/layout.py), pipeline slot with per-step weights, adapter-declared delivery (hooks / ACE TRT fill / steering= kwarg). tests/unit: 858 pass + 10 new; 4 fails = test_lora_facade_session drain tests, pre-existing on origin/main (verified by stash).
- Step 2 DONE: acestep/steering/packs.py (one .safetensors per vector, metadata JSON in header), paths.steering_packs_dir() (env DEMON_STEERING_PACKS_DIR, default <models>/steering_packs), knobs.steering_pack_spec (same semantics as steering_axis_spec -> homonym-safe), DiffusionBackend.attach_steering_packs/_sync_steering_slot, factories attach packs filtered by layout. Wire types: registry shape unchanged, regen produced no content diff. tests: 866 pass, same 4 pre-existing fails.
- Step 3 DONE. Engine: C:\Users\ryanf\.daydream-scope\models\demon\sa3\trt_engines\sa3_m_dit_steer_l1_646_646\sa3_m_dit_steer_l1_646_646.trt sha256 d5a7abc6b259c8984b5ca1a494a1f15bf81e3663cb61de7a5db9e21929704fd2 (built 70 s, fp16mixed STRONGLY_TYPED, surgery v1, input steering [1,24,1536]). Previous engines untouched (sa3_m_dit_l1_646_646, fp8, refit).
  Parity (scripts/sa3/sa3_trt_steer_parity.py, L=646, real cond, t=1.0/0.7/0.4/0.1; json notes/family_merge/steer_logs/parity_646.json):
  zero-steer vs previous engine: NOT bit-identical, cos 0.999968/0.999921/0.999953/0.999960.
  zero-steer vs eager: 0.999962/0.999853/0.999957/0.999968 (previous engine vs eager on same inputs: 0.999968/0.999877/0.999967/0.999981). Bar 0.9998 PASS.
  nonzero steer (blocks 5,15, |shift| = 0.25 x token norm; moved output rel 0.33-0.59) TRT vs eager hooks: 0.999966/0.999899/0.999960/0.999967. PASS. re-zero after steering bit-exact.
  Per-step engine time (L=646): previous 12.73 ms, steer engine zero 12.88 ms (+0.15 ms, +1.2%), steer active 12.91 ms; fp8 engine 10.65 ms.
  Decision: discovery prefers steer engine ONLY when packs exist for (sa3, medium); otherwise fp8 stays default. LoRA (refit) keeps precedence.

## Step 4/5/6 progress (2026-10-03)
- Full discovery running (bright, warm, rough, density, percussive; 32 pairs, 8 patch pairs). Log steer_logs/discover_sa3_medium.log.
- bright: centroid pos 829 / neg 218 (sign agreement 1.00); patching -> block 23 (effect 0.177; pairs split 20/23), norm 18.6.
  calibration (alpha x meandiff = knob/10): -2:128 -1:127 0:662 +1:2272 +2:2883 Hz.
- demos/sa3 Steer section written (generic on steer_ prefix, sliders reuse blend-slot style). Not yet GPU-verified.
- LAUNCH_sa3_steer.sh written (DEMON-steer, --checkpoint sa3-medium, port 1318). docs/STEERING.md drafted.
- Discovery done (EXIT 0). Blocks/norms: bright b23 n18.6; warm b15 n29.5 (lowhigh gap 26 dB, agree .94);
  rough b2 n15.7 (flatness proxy weak: agree -0.25); density b1 n12.9 (onset proxy weak: agree -0.56);
  percussive b15 n40.2 (perc gap .244, agree .88). Report steer_logs/discover_sa3_medium.json.
  Packs: C:\Users\ryanf\.daydream-scope\models\demon\steering_packs\sa3\medium\{bright,warm,rough,density,percussive}.safetensors
- Sanity (TRT live backend, steer_logs/sanity.{log,json}): knob0 vs no packs on steer engine BIT-IDENTICAL=True.
  Direction right-way (+/-) of 3 prompts: bright k1 3/3,3/3 k10 3/3,3/3; warm same 3/3 all; percussive 3/3 all;
  density k1 1/3,2/3 k10 3/3,1/3; rough k1 1/3,1/3 k10 3/3,2/3 (proxies for rough/density do not track the concept).
  Tick ms (mean): steer engine knob0 52.1, active 52.8; default engine (no packs) fp8 43.2.
- Coordinator 2026-10-03: no new discovery; do NOT call these vectors TADA (prompt-pair diff-of-means on the generic slot).
- Step 5: server (LAUNCH_sa3_steer.sh) selected sa3_m_dit_steer_l1_646_646, loaded 5 steer_* knobs (manifest -30..30, group steering).
  Chrome extension not connected, so verified over the page's wire with HeadlessClient (steer_logs/server_verify.json), page HTML/JS served with Steer section.
  Heard proxy 0/+10/-10/0: bright 589/1860/169/500 Hz; warm 12.8/58.0/-8.8/17.2 dB; percussive .146/.134/.045/.135;
  density 7.8/8.0/7.8/7.8 onsets/s; rough flatness 2.4e-4/2.9e-5/3.3e-5/7.3e-5. No errors, 4329 slices. Server stopped, GPU 4.1 GB.
- Step 5 commit 2eef961d, step 6 commit 4beb08d4 (HEAD 4beb08d4). Full unit: 874 passed, 4 skipped, 4 failed (lora_facade drain, pre-existing on main).
- pr_steering.md written. GPU_LOCK returned to user-demo-session. DONE.
