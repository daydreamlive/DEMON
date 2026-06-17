# demon-client

Browser SDK for the DEMON realtime music backend. Everything a frontend
needs to drive a live generative-music session: the WebSocket session
client, the binary slice decoder, realtime audio playback, reconnect
orchestration, and typed access to the backend's self-describing
manifests.

Nothing in this directory imports from any host app (drift-guarded by
`tests/unit/test_client_sdk.py`), so it can be copied or packaged
standalone. Two consumption modes ship in-repo:

- **Bundled apps** — the rtmg demo UI in
  `demos/realtime_motion_graph_web/web/` imports the TS source via the
  `@demon/client` tsconfig path alias; its bundler transpiles the SDK
  and bundles the slice-decoder worker automatically.
- **Static no-build pages** — `dist/` holds a committed esbuild bundle
  (`demon-client.js`, `sliceDecoder.worker.js`, `audio-worklet.js`) that
  the demo backend mounts at `/sdk/`. These pages live in external repos
  mounted via the backend's `--demo <path>` flag (e.g. demos from
  [`daydreamlive/demon-example-apps`](https://github.com/daydreamlive/demon-example-apps)),
  never inside this repo's `demos/` tree.
  Pass `RemoteBackendOptions.sliceWorkerUrl` and
  `AudioPlayerOptions.workletUrl` pointing at the mounted siblings.
  Regenerate after any SDK change: `npm install && npm run build` here
  (see `build.mjs`).

Vibecoding a new frontend (or pointing an agent at this SDK)? Start at
[AGENTS.md](./AGENTS.md) — the imperative recipe: discovery-first
workflow, the five invariants the manifests can't express, and the
reference implementations to copy.

## What the backend gives you

The control surface is backend-owned and self-describing. Build against
these, not against hardcoded shapes:

| Surface | What it is |
|---|---|
| `GET /api/protocol` | Full wire contract: every command you may send, every event you'll receive, the session-init config schema, and the upload handshake. `fetchWireContract()`. |
| `GET /api/knobs` | The knob manifest for the `params` channel: name, type, range, options, group, description. `fetchKnobManifest()`. |
| `types/wireContract.gen.ts` | The same contract as compile-time TypeScript types, generated from the Python registry. Typecheck against these. |
| MCP (`describe_protocol`, `list_knobs`) | Same manifests for agent consumers. |

## Quickstart

```ts
import {
  AudioPlayer,
  RemoteBackend,
  SLICE_FLAG_DELTA,
  type AudioSlice,
  type SessionConfig,
} from "@demon/client";

// 1. Source audio: interleaved float32 PCM at 48 kHz, plus channel count.
//    (Decode a file with WebAudio's decodeAudioData, or fetch a fixture
//    the pod already has and set use_server_fixture to skip the upload.)
const config: SessionConfig = { prompt: "minimal techno", depth: 4 };

// 2. Connect. The constructor takes the WS URL, the PCM, and the config;
//    connect() resolves after the ready handshake + initial buffer.
const remote = new RemoteBackend("ws://localhost:1318", interleaved, 2, config);

// 3. Play. AudioPlayer needs the worklet asset served at workletUrl
//    (copy assets/audio-worklet.js into your static dir).
const player = new AudioPlayer();
remote.addEventListener("slice", (e) => {
  const s = (e as CustomEvent<AudioSlice>).detail;
  if (s.epoch !== player.swapCount) return; // drop pre-swap stragglers
  if (s.flags === SLICE_FLAG_DELTA) player.addDelta(s.startSample, s.audio);
  else player.patch(s.startSample, s.audio);
});

await remote.connect();
await player.init(remote.initialBuffer!, remote.channels);
await player.resume();

// 4. Drive it. Send the FULL knob dict every UI tick (the app uses ~125 Hz)
//    with the playhead position in seconds. Knob names/ranges come from
//    /api/knobs; values are clamped server-side.
setInterval(() => {
  remote.sendParams({ denoise: 0.4, feedback: 0.1 }, player.positionSec);
}, 8);

// Discrete controls are typed senders: sendPrompt, sendSetPromptBlend,
// sendEnableLora, sendSwapSourceByName, sendSetTimbreFixture, ...
```

## The protocol state machine (what the manifests can't tell you)

1. **Init handshake.** First frame is the JSON `SessionConfig`. Unless
   `use_server_fixture` is set, one binary frame follows: `<II` header
   (channels, samples) + interleaved float32 PCM. The server replies
   with the `ready` JSON, then a binary float16 initial buffer.
   `RemoteBackend.connect()` runs all of this.
2. **Slice stream.** Audio arrives as binary frames: a 23-byte header
   (see `SLICE_HDR_SIZE` and the worker) + float16 PCM, either raw or
   zstd-compressed as a *delta against your local buffer mirror*.
   `RemoteBackend` decodes off-thread and emits `slice` events;
   `AudioPlayer.patch`/`addDelta` maintain the mirror. You cannot skip
   slices: the delta basis would desync.
3. **Swaps.** `swap_ready` JSON is followed by a full binary replacement
   buffer; `RemoteBackend` pairs them and bumps its slice epoch so
   in-flight slices for the old source are droppable (`slice.epoch` vs
   `AudioPlayer.swapCount`). `stem_assets` similarly heralds one binary
   buffer per listed stem.
4. **Echo channels.** Two commands are `origin_sensitive` in the
   contract — `params` and `set_prompt_blend`. Sent on the session's own
   WS they apply directly; driven externally (MCP / control bus) they are
   *not* applied — the server echoes them back on the command's
   `echo_event` (`params_echo`, `prompt_blend_echo`) for the session's
   own UI to mirror and re-send through its smoothing tween. Every other
   command applies identically from any origin and acks via its normal
   events (`lora_catalog`, `timbre_set`, `swap_ready`, ...).
5. **Upload handshake.** A connection whose first frame is
   `{type: "upload_track"}` + PCM persists a track on the pod without
   starting a stream (`upload_ok` / `upload_failed`), then closes.

## Host integration points

- **Worklet asset**: serve `assets/audio-worklet.js`; override the path
  via `AudioPlayerOptions.workletUrl` if you don't serve it at
  `/audio-worklet.js`.
- **Prompt transform**: `RemoteBackendOptions.promptTransform` is applied
  to tags on every `sendPrompt`. The demo app injects enabled-LoRA
  trigger prefixes there. Omit it and prompts are sent verbatim.
- **Loudness matcher tuning**: `AudioPlayerOptions.loudnessConfig`
  (lazy provider, all fields optional).
- **State**: the SDK never writes app state. Subscribe to the
  `RemoteBackend` CustomEvents (`ready`, `slice`, `swap_ready`,
  `swap_failed`, `stem_assets`, `depth_applied`, `params`,
  `params_echo`, `lora_catalog`, `close`, ...) and mirror what you need.
- **Reconnect**: `wsReconnect.ts` provides the backoff loop; the backend
  has no session resume, so reconnect = re-run the full handshake with
  your own retained state. See the demo's `useStartSession.ts` for a
  reference orchestration.

## Keeping types in sync

`types/wireContract.gen.ts` is generated from the Python registry:

```
python demos/realtime_motion_graph_web/scripts/gen_wire_types.py
```

CI fails if the committed file is stale, if the dispatcher handles a
command the registry doesn't declare, or if this SDK handles an event
the registry doesn't declare. At runtime, compare `PROTOCOL_VERSION`
against the served contract's `version` to detect a stale build.
