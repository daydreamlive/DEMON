# Golden / latency harness

Black-box regression suite for the realtime streaming stack. It drives a
**live server** over the WebSocket wire protocol (never imports server or
app internals), records every frame, and compares the generated audio
against canonical references captured from a baseline build.

Because it is black-box at the protocol boundary, the same suite measures
any server build — including running a checkout of one branch against a
pod serving another. That is the intended workflow for de-risking large
refactors: capture baselines from the pre-refactor build, then point the
identical suite at the refactored build.

## Pieces

| File | Role |
|------|------|
| `client.py` | Recording WS client (slices, swaps, LoRA, transcripts) |
| `scenarios.py` | Declarative scenario table — add regression cases here |
| `runner.py` | Executes scenarios → result bundles (audio + transcript + timing) |
| `compare.py` | Tier-1 byte identity, tier-2 tolerance metrics |
| `refs_store.py` | Reference bundles on the HF dataset, sha256-pinned manifest |
| `refs.json` | The manifest (committed; bundles themselves live off-repo) |
| `test_golden.py` | Audio matches the canonical reference |
| `test_latency.py` | Coarse latency ceilings + per-build report artifact |

## Running

On a local GPU box the suite is self-contained: it spawns the server
itself on a free port and tears it down afterwards.

```bash
python -m tests.golden.refs_store fetch   # pull canonical refs
pytest tests/golden                       # local GPU (auto-spawn)
pytest tests/golden --pod-url ws://POD:1318   # or a remote pod
```

`DEMON_TEST_ACCEL=eager|compile|tensorrt` overrides the spawned
server's backend (default: the server's own default, tensorrt).
Without a pod URL and without CUDA everything skips, so the suite is
inert in CPU-only environments (`pytest tests/unit` is unaffected).

## Capturing new baselines

Only from a build you intend to be the reference (normally `main`), on
the standard pod class:

```bash
# variance probe FIRST: prints the same-build noise floor and the
# suggested per-scenario thresholds for this hardware
python -m tests.golden.runner --pod-url ws://POD:1318 \
    --scenario all --repeat 3

python -m tests.golden.runner --pod-url ws://POD:1318 \
    --scenario all --out runs/baseline
# paste the suggested thresholds into refs.json after packing
python -m tests.golden.refs_store pack --runs runs/baseline
python -m tests.golden.refs_store upload          # HF write token
git add tests/golden/refs.json                    # commit the manifest
```

Thresholds come from the probe's measured noise floor (x3 safety
margin), never from guessing: above the same-build noise, well below
audible breakage.

## Comparison policy

The canonical artifact is a **position-aligned region of the song
buffer**: `[anchor + warmup_skip_s, + canonical_s]`, where the anchor is
where slice coverage started (session start, or the swap point). Slice
segmentation is scheduling-dependent, so only position space is
comparable at all; the warm-up skip and the settle margin trim the
highest-variance stretches.

**Do not expect bit-exactness.** Generation is playhead-paced and
windows are re-emitted as they refine through the pipeline depth, so
wire-level audio carries small timing-coupled variance even between
back-to-back runs on one GPU (measured on a 5090: ~5% of samples, max
abs diff ~0.03, perceptually identical). Engine-level seed determinism
is covered separately by `tests/test_stream.py`. Accordingly:

1. **Tier 1 — identity** is a short-circuit when hashes happen to
   match, never a requirement.
2. **Tier 2 — calibrated tolerance** is the actual contract: log-mel
   distance, RMS level shift, and worst per-second spectral cosine
   against thresholds derived from the measured same-build noise floor.
   `runner --repeat N` prints both the observed floor and suggested
   thresholds (floor x3); put those in `refs.json`.

> **Hardware caveat.** The committed `refs.json` carries the canonical
> hashes captured on one card (currently an RTX 5090) and, until a
> variance probe is run, `null` thresholds — so tier 2 falls back to the
> strict `DEFAULT_THRESHOLDS`. On that same card a healthy run is
> bit-exact (tier 1) and passes; on a **different** card/driver/engine
> build a run is legitimately not bit-exact, and the strict fallbacks
> will flag it. That is by design (fail loud rather than silently pass an
> uncalibrated comparison), but it means cross-hardware use needs a
> one-time calibration first: run `runner --repeat N` on the target
> hardware, paste the suggested thresholds into `refs.json`, and commit.
> `test_golden.py` prints this hint on any uncalibrated tier-2 failure.

Actions fire when the **generation frontier** crosses their song
position (not the playhead), so the effect lands inside the compared
region on any machine regardless of its realtime factor.

The 1s windows straddling an action (±1 window around the recorded
trigger position) are **masked from the `win_cos_min` gate**: the
transition is an intentional discontinuity, and exactly where it
catches a pipeline block boundary is stable within a capture session
but drifts between sessions, so spectral identity there is not owed.
The masked windows' worst cosine is still reported
(`win_cos_min_action` in `compare.json`) so a real glitch at the
transition remains visible; everything outside the mask is gated as
usual, and scenarios without actions mask nothing.

When driving a remote pod, set `DEMON_SERVER_GPU` to the pod's card so
the captured env (and the identity gate) reflects the server GPU rather
than the harness box's.

## Latency reports

`test_latency.py` asserts only coarse, env-overridable ceilings (the
kind a chunk-latency architectural regression would trip) and writes the
full percentile detail to `runs/latency-reports/latency-<scenario>.json`
for diffing between builds. Treat the diff, not the pass/fail, as the
interesting output.

Each fired action also carries **knob-to-ear** numbers, derived from the
recorded slices against the runner's simulated 1.0x playhead:
`audible_first_ms` (playhead reaches the first post-action slice ahead
of it — the effect starts ramping in, since in-flight windows refine
their remaining steps with the new params) and `audible_full_ms`
(playhead reaches the action-time generation frontier — windows past it
got every denoise step with the new params). Reference point: the
knob_step baseline on the 5090 at depth 4 measures 234 / 594 ms. The
dominant term is the playback-lead buffer, which is the architectural
quantity worth watching; the ceiling
(`DEMON_LAT_CEILING_ACTION_AUDIBLE_MS`, default 8000) only trips on a
chunk-scale regression.

## Transcripts

Every run records `transcript.jsonl` + `blobs/` — the full wire session
with timestamps. These are the replay inputs for the browser-client
regression tests (driving the real web `RemoteBackend` against recorded
server traffic without a GPU). Keep recording on for baseline captures;
`--no-blobs` exists for quick timing-only runs.

## Adding a scenario

Add one `Scenario` to `scenarios.py`. Keep actions frontier-relative,
leave `deterministic` to the no-actions default, then capture + pack a
reference for it from the baseline build. The drift cost of a scenario
is one entry in `refs.json`; the coverage gain is a whole wire-path.
