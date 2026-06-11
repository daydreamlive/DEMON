// demon-client — browser SDK for the DEMON realtime music backend.
//
// Everything a frontend needs to talk to a DEMON pod lives under this
// entry point; nothing in sdk/ imports from the host app (enforced by
// tests/unit/test_client_sdk.py). See ./README.md for the integration
// walkthrough and the protocol state machine.

// Wire contract: generated command/event payload types + name unions
// (projected from the Python registry by scripts/gen_wire_types.py).
export * from "./types/wireContract.gen";
// Contract envelope shapes served by GET /api/protocol.
export * from "./types/wireContract";
// Knob manifest shapes served by GET /api/knobs.
export * from "./types/knobs";
// Client view-models + binary-framing constants.
export * from "./types/protocol";

// WebSocket session client (binary slice stream, swap/stem state
// machines, typed senders).
export { RemoteBackend, float16ArrayToFloat32 } from "./protocol";
export type { RemoteBackendOptions, WsTrace, WsTracePhase } from "./protocol";

// Realtime audio playback (worklet + ScriptProcessor fallback, loudness
// matcher, stem overlays).
export { AudioPlayer } from "./audio/AudioPlayer";
export type {
  AudioLoudnessConfig,
  AudioPlayerOptions,
} from "./audio/AudioPlayer";
export * from "./audio/lufs";

// Reconnect orchestration (exponential backoff + jitter).
export * from "./wsReconnect";

// Manifest fetchers + retrying fetch helper.
export { fetchWireContract } from "./fetchWireContract";
export { fetchKnobManifest } from "./fetchKnobManifest";
export { fetchWithRetry } from "./fetchWithRetry";
