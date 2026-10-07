# Steering PR: second pedal for the user's activation-steering knobs

Written 2026-10-05 (parent). Facts from inventory.md in this folder. Scope is exactly this; nothing else.

## Where
- Branch: ryanontheinside/feat/steering-generic, tip 4beb08d4, worktree C:\_dev\projects\DEMON-steer (clean). All work and commits happen ONLY in that worktree, on that branch. Never open, read or modify C:\_dev\projects\DEMON (dirty primary) or any DEMON-tada-* worktree. The tada-* branches sit above this branch; new commits here do not touch them.
- Files to change: demos/sa3/index.html, demos/sa3/sa3.js, demos/sa3/styles.css. No engine, knob-registry, pack or backend changes. No new dependencies, no framework, no canvas, no CDN.

## What the demo does today
- One guitar-pedal faceplate (DOM + CSS). Knobs are rotary, built by numericKnob / enumKnob in sa3.js, committed through commitKnobValue -> sendParams.
- The user's steering knobs (every manifest knob whose id starts with `steer_`; on this machine bright, density, percussive, rough, warm; range -30..30, default 0) were bolted on as faders: steerSlider / renderSteer (~sa3.js 375-415), hidden section #steer (index.html 20-23), CSS styles.css 245-264.

## Target
1. A SECOND pedal on the page, a sibling faceplate of the main one, same construction and visual language (same classes/mixins for the enclosure, screws, label plate), its own name on the plate ("STEER" or similar) and a distinct enclosure colour so the two read as two pedals on a board.
2. Every `steer_*` knob becomes a rotary knob on the second pedal, built with the SAME numericKnob function and the same interaction (drag, wheel, keyboard, reset gesture, value readout) as the main pedal. Bipolar display: 0 sits at 12 o'clock, -30 and +30 at the ends of the same sweep the main knobs use. Label = the pack id without the `steer_` prefix, title-cased.
3. Remove steerSlider, renderSteer, the hidden #steer section and its CSS. One rendering path for steering knobs, no sliders left anywhere.
4. Layout: the two pedals side by side on wide viewports, stacked on narrow ones (phone width must work, no horizontal scroll). The second pedal renders only when the manifest contains at least one `steer_*` knob; otherwise it is absent (not an empty shell).
5. Knob count varies with the installed packs (5 here); the second pedal's knob grid must lay out 1..8 knobs cleanly, wrapping to a second row when needed, matching the main pedal's knob spacing.
6. Behaviour parity: values commit through commitKnobValue exactly as before; reconnect/initial manifest load must populate the second pedal's knobs with current values the same way the main pedal's knobs are populated.

## Verification (all CPU, no GPU)
- `node --check demos/sa3/sa3.js` passes.
- Grep: no occurrence of steerSlider, renderSteer or `id="steer"` remains in demos/sa3.
- tests/unit: test_steering_seam, test_steering_packs, test_sa3_steering_engine still pass (they must be untouched by this change; run them to prove it).
- If a static-demo smoke exists in web/tests or demos, run it; if none exists, do not invent a backend-dependent test. Record in the hand-back that the visual result still needs a live look by the user.

## Commits and PR
- Small commits, conventional prefix (e.g. `demo(sa3): second pedal with rotary steering knobs`). NO Co-Authored-By, NO "Generated with", NO AI or tool attribution anywhere (commit messages, code comments, PR text). This is absolute.
- Write the PR description draft to notes/steering_pr/pr_steering.md (title, summary of the 6 engine commits + the demo change, test evidence, screenshots section left as TODO for the user). Do NOT push, do NOT open a PR; the user does that after review.
