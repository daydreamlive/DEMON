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
  the demo backend mounts at `/sdk/`. Repo-local static demos are direct
  children of `demos/` with a manifest and mount automatically; external
  demos can also be mounted via the backend's `--demo <path>` flag (e.g.
  demos from [`daydreamlive/demon-example-apps`](https://github.com/daydreamlive/demon-example-apps)).
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

## Operator defaults (`config.json`)

`config.json` is the **client-side operator-defaults** file — distinct from
the backend-served `/api/knobs` and `/api/protocol` above. Those are
generated from the Python registry and describe what the engine *accepts*;
`config.json` is a hand-authored, per-installation file of *startup values*:
default prompts, knob/param start positions, enabled LoRAs, engine/session
fields, plus per-client UI blocks (the web app's `web` section). The schema
+ pure transforms live in [`config/`](./config) so web / M4L / VST honor one
file with one implementation:

| Export | What it does |
|---|---|
| `loadConfig(baseUrl?)` | Fetch `<baseUrl>/config.json` (no-arg = page-relative `/config.json`) and merge onto `DEFAULT_CONFIG`. |
| `mergeConfig(base, override)` | Layer an override (a file, an import) onto a base. Ignores unknown keys for the typed surface; **preserves** them for the write path. |
| `selectVariant(cfg, scale)` | Resolve the XL (5B) `*_xl` siblings when the active checkpoint is 5B. |
| `resolveLoraCapForSource(durationS, engine)` | The duration-aware LoRA cap (tiers, else the static fallback). |
| `rtmgConfigToSessionConfig(cfg, runtime)` | Map the config's engine block + live runtime bits to the handshake `SessionConfig`. The one place every frontend negotiates an identical session. |
| `applyConfigToState(cfg, scale)` / `captureConfigFromState(snapshot, base)` | Neutral apply/capture adapters — pure shape in, pure shape out. Each client writes a thin wrapper from its own state (zustand, a plugin's param model) to these. |
| `serializeConfig(cfg)` / `getUnknownKeys(cfg)` | Write path: re-emit top-level keys a client read but does not model, untouched (**preserve-unknown on write**). |

This is a deliberate exception to the "build from manifests, never
hand-declare" rule (below): it is a hand-authored, client-side file. Read =
ignore-unknown; write = preserve-unknown; a `version` field is present from
day one so a later loader can branch. Each client keeps its own state
wiring — only the schema + transforms are shared.

## Control copy (`controls/`)

The **user-facing** descriptions and display names a frontend renders on its
knobs/inputs — "how much the model reshapes the source audio…", `denoise` →
`strength`. Like `config.json` this is hand-authored, client-side copy, and
it is a second, distinct layer from the `/api/knobs` manifest's `description`
field: the manifest carries the **terse, agent-facing** one-liner generated
from the Python registry; [`controls/`](./controls) carries the **editorial,
when-to-reach-for-it** prose. Same param ids, different audience.

| Export | What it does |
|---|---|
| `describeControl(param)` | Rich user-facing description for a param id, or `undefined`. LoRA/manual-slot families share one string each (matched by prefix). |
| `displayNameFor(param)` | Friendly label (overrides win; else snake_case → spaced words). |
| `resolveControlDescription(param, manifest?)` | One lookup that prefers the rich copy and falls back to the manifest's terse `description` — so runtime-only knobs (LoRAs, manual slots) still surface *something*. Pass the `/api/knobs` manifest as the second arg. |

Also exported: the raw maps/strings (`CONTROL_DESCRIPTIONS`,
`CONTROL_DISPLAY_NAMES`, `INTERP_PATH_DESCRIPTIONS`, `SOURCE_MODE_HINTS`,
`TIMBRE_REF_DESCRIPTION` / `STRUCTURE_REF_DESCRIPTION`,
`STEM_SECTION_DESCRIPTION`). The tooltip-**rendering** machinery (hover
readouts, DOM wiring) and web-only affordance hints (keyboard chords,
MIDI-learn) stay per-client.

The same module also holds `TERMS` — the canonical product **nouns** for
features the product team renames over time (`tags`, `lora`/`lora_plural` →
flip "LoRA" to "Trained Style" in one place). Keyed by a stable id, plain
serializable data exported from `node.ts` too, so the native (VST) C++ codegen
emits the same labels instead of hand-authored strings. Compose compound
labels by interpolating a term (`` `${TERMS.lora} Library` ``). Headline
labels reference it; deep inline help prose keeps its literals.

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
