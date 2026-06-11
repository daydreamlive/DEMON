# Building a DEMON frontend (agent instructions)

You are building a UI against a DEMON realtime-music pod. Everything you
need is in this directory (`demon-client`) plus two HTTP manifests the
backend serves. Do not reverse-engineer the demo app in `../` — build
from the contract.

## Step 0: discover, don't hardcode

The control surface is backend-owned and self-describing. Fetch both
manifests at boot and build your UI from them:

| Fetch | Returns | Use it for |
|---|---|---|
| `fetchWireContract()` → `GET /api/protocol` | `{version, commands, events, config, handshake}` | Every command you may send, every event you'll receive, the session-init config schema. |
| `fetchKnobManifest()` → `GET /api/knobs` (`?sde=1` for SDE mode) | `{version, knobs}` | The `params` payload: render each knob by `type` (float/int → slider, enum → select, bool → checkbox), group by `group`, bounds/options/defaults from the entry. |

Compile-time: import the generated types from `types/wireContract.gen.ts`
(via the package root). Compare `PROTOCOL_VERSION` and
`KNOB_SCHEMA_VERSION` against the served manifests' `version` fields and
warn on mismatch — that means your build predates a contract change.

Never declare a knob list, a command shape, or an enum option list by
hand. If you find yourself typing `"denoise"` with a hardcoded max, stop
and read it from the manifest instead.

## Step 1: the minimal working session

Follow the Quickstart in [README.md](./README.md) — it is accurate and
complete: build a `SessionConfig` (schema = the contract's `config`
section), `new RemoteBackend(wsUrl, pcm, channels, config)`, `await
remote.connect()`, `new AudioPlayer()`, `init` + `resume`, then drive it
with `sendParams` and the typed `send*` methods. The pod serves HTTP and
WS on one port (default 1318).

**Playing a fixture the pod already has?** Set `use_server_fixture: true`
(+ `fixture_name`) in the config — the server loads the waveform off its
own disk and the binary PCM upload is skipped entirely. Don't download a
fixture in the browser just to re-upload it; that wastes megabytes and
seconds on every session start. Same idea mid-stream:
`sendSwapSourceByName` / `sendSetTimbreFixture` / `sendSetStructureFixture`
instead of their PCM-upload variants whenever the audio is already
server-side.

## Invariants the manifests can't tell you

1. **Send the FULL knob dict every tick.** `sendParams(raw, playbackPos)`
   at UI rate (the demo uses ~125 Hz), `playbackPos` in **seconds**
   (`player.positionSec`). The server merges and clamps server-side; you
   do not need client-side validation to be safe, only to be polite.
   **Seed `raw` from your session's live state, NOT from the manifest's
   static defaults.** Manifest defaults can differ from what your session
   was configured with — e.g. blindly asserting `steps_override`'s
   default of 8 against a session started with `steps: 4` forces a
   pipeline rebuild on tick 1. For knobs your UI doesn't control, either
   omit them or initialize them from the session config you sent /
   the `params_update` echo, then hold those values.
2. **Never skip, reorder, or drop slices on the floor.** Delta-flagged
   slices are zstd deltas against your local buffer mirror; every slice
   must reach `player.patch` / `player.addDelta` or the basis desyncs and
   audio corrupts from then on.
3. **Respect the epoch guard.** In your `slice` listener, drop any slice
   with `slice.epoch !== player.swapCount`. In your `swap_ready`
   listener, call `player.swap(detail.interleaved, detail.channels)` —
   that bumps `swapCount` in lockstep with the epoch the backend just
   bumped. Stale in-flight slices from the old track then drop instead of
   bleeding through.
4. **Serve the worklet.** Copy `assets/audio-worklet.js` into your static
   dir and point `AudioPlayerOptions.workletUrl` at it (default
   `/audio-worklet.js?v=5`). Without it, playback falls back to
   ScriptProcessor (works, but worse).
5. **Reconnect = full re-handshake.** There is no session resume. On an
   unexpected close (`closedByUser === false`), build a new
   `RemoteBackend` from your retained config/PCM, attach ALL listeners
   **before** `connect()`, and call
   `remote.setSliceEpoch(player.swapCount)` so the fresh epoch counter
   matches the surviving player. `wsReconnect.ts` provides the backoff
   loop; the demo's `../hooks/useStartSession.ts` is the reference
   orchestration.

## Echo channels (external control)

Exactly two commands are `origin_sensitive` in the contract: `params`
and `set_prompt_blend`. Sent on your own WS they apply directly; sent
externally (MCP / control bus) they are NOT applied — the server echoes
them back on the command's `echo_event` (`params_echo`,
`prompt_blend_echo`) for YOUR UI to mirror into its own state and
re-send. Every other command applies identically from any origin and
acks via its normal events (`lora_catalog`, `timbre_set`, `swap_ready`,
...). If you want agent-driven control to move your sliders, subscribe
to the echo events (the demo's `useMcpMirror.ts` is the reference).

## State management

The SDK never writes your stores. Subscribe to `RemoteBackend`'s
CustomEvents (`ready`, `slice`, `params`, `params_echo`, `swap_ready`,
`swap_failed`, `stem_assets`, `depth_applied`, `lora_catalog`, `close`,
...) and mirror what you need. Inject app behavior through the option
seams: `RemoteBackendOptions.promptTransform` (e.g. LoRA trigger
prefixing) and `AudioPlayerOptions.loudnessConfig` / `workletUrl`.

## Reference implementations in the demo app

- `../components/Performance/DynamicKnobPanel.tsx` — a complete control
  surface rendered purely from `/api/knobs`, zero hand-declared knobs.
  This is the pattern to copy for any new UI.
- `../hooks/useStartSession.ts` — session + reconnect orchestration.
- `../hooks/useMcpMirror.ts` — echo-channel mirroring.

## If you edit the SDK itself

- No imports from the host app: everything internal is relative; the
  only allowed bare import is `fzstd` (`tests/unit/test_client_sdk.py`
  enforces this).
- After any change to the Python registries (`protocol.py`,
  `acestep/streaming/knobs.py`), regenerate:
  `python demos/realtime_motion_graph_web/scripts/gen_wire_types.py` —
  a stale `types/wireContract.gen.ts` fails CI.
- `assets/audio-worklet.js` and the demo's `public/audio-worklet.js`
  must stay byte-identical (drift-tested). Edit one, copy to the other.
- Binary framing (the `<II` PCM header, the 23-byte slice header) is
  hand-coded by design; it lives in `packPcmFrame` / the slice decoder
  and is documented per-entry in the registry, not typed.

## Agent-driven control without a browser

The MCP server exposes the same contract: `describe_protocol` and
`list_knobs` return the manifests; `set_knob(s)` / `set_prompt` /
`swap_to_fixture` / etc. drive a live session and are validated against
the same registries the wire is.
