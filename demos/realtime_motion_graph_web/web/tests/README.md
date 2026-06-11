# Web-app regression & performance tests

Three tiers, all GPU-free. Tiers B and C consume the recorded golden
session transcripts from the repo-root harness (`tests/golden/`), so the
exact same wire bytes that validated the server also validate the client.

| Tier | Where | Runner | What it covers |
|------|-------|--------|----------------|
| A | `tests/unit/` | vitest (node) | float16 decode (exhaustive vs native Float16Array), slice-epoch stamping (swap-bleed class), AudioPlayer patch/addDelta/swap playhead clamping (the `_spPosition` hard-zero regression), WsReconnector backoff/cancel/give-up, session-store transitions |
| B | `tests/replay/` | vitest (node) | the REAL `RemoteBackend` driven by recorded transcripts through a fake WebSocket: event sequence, send-path wire equality, full buffer reconstruction vs `canonical.f32.raw` (bit-exact), main-thread decode throughput artifact |
| C | `tests/e2e/` | Playwright (chromium) | the whole app (worklet + decode worker + stores + UI) against the Python replay server: start → ready → knob → swap, buffer non-silence, playhead stall detection, browser-side streaming stats artifact, and — after the transcript drains — a **bit-exact hash comparison of the browser-reconstructed canonical buffer region against the reference bundle** (the same bar as Tier B, but through the real worker decode path) |

## Prerequisites

```bash
# from the repo root, via the repo venv — pulls transcripts + canonical
# refs into ~/.cache/demon/test-refs/ (Tier B and C; Tier A needs none).
# venv python: .venv/bin/python (POSIX) or .venv/Scripts/python.exe (Windows)
python -m tests.golden.refs_store fetch

# once, for Tier C
npx playwright install chromium
```

## Running

```bash
npm run test:unit     # Tier A, sub-second
npm run test:replay   # Tier B, ~6 s for all six scenarios
npm test              # A + B
npm run test:e2e      # Tier C, ~40 s (spawns replay server + next dev)
```

Tier B skips (with a fetch hint) any scenario missing from the refs
cache. Tier C manages its two servers itself (`playwright.config.ts`):
`tests/golden/replay_server.py` on :18931 and `next dev` on :3211.

## Comparison policy (Tiers B and C)

Unlike the live golden suite (GPU server, calibrated tolerances), replay
is fully deterministic — the input bytes ARE the recording — so the
reconstructed canonical region must match `canonical.f32.raw`
**bit-exactly**. Any diff is a real client regression: float16 decode,
zstd delta path, slice routing, or epoch handling.

Tier B makes that comparison sample-by-sample in Node (main-thread
decode fallback, full diff tooling). Tier C repeats it in the real
browser via SHA-256 of the player mirror's region (worker decode path,
real AudioPlayer writes) against the bundle's recorded
`canonical_sha256` — same bytes, same layout, hashed in-page to avoid
shipping megabytes out of the browser. On a Tier C hash mismatch, run
Tier B to localize (it reports first-mismatch index and max abs diff).

The canonical artifact is deliberately *buffer position space*, not a
capture of rendered output: playback adds seam/swap crossfades, LUFS
makeup gain and looping on a free-running clock, so a rendered capture
can only be compared with tolerances and alignment search — strictly
weaker than the bit-exact buffer contract the worklet plays from.

## Performance artifacts

Both B and C write JSON reports to `runs/web-replay-reports/`
(gitignored): per-slice decode percentiles and realtime factor (B),
slice lead over the playhead / arrival gaps / stall count / stale-drop
count (C). Like the golden latency reports, the interesting output is
the **diff between builds**, not the pass/fail.

## Refactor portability

These tests import the SDK surface (RemoteBackend, AudioPlayer,
WsReconnector, protocol types/constants) through the same
`@demon/client` alias the app itself uses, plus `@/store/...` for the
app stores. The SDK extraction (`engine/protocol.ts` -> `sdk/`) was
absorbed as a mechanical import-path rename plus a `vitest.config.ts`
alias entry — no test logic changes.

## Browser test hooks

Tier C observes the app through `window.__demonTest`
(`engine/testHooks.ts`, installed at boot beside `__demonDebug`):
session status, playhead, buffer RMS, ws trace, and an opt-in slice
probe (`startProbe()` / `probeStats()`). Also handy for manual
debugging against a live pod.
