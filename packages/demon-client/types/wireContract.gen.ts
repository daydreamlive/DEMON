// AUTO-GENERATED — do not edit by hand.
//
// Projected from the Python wire-contract registry
//   demos/realtime_motion_graph_web/protocol.py :: wire_contract()
// by demos/realtime_motion_graph_web/scripts/gen_wire_types.py.
//
// Regenerate after any registry change:
//   python demos/realtime_motion_graph_web/scripts/gen_wire_types.py
// Drift-guarded by tests/unit/test_wire_contract.py
// (test_generated_wire_types_match_contract) — a stale copy fails CI.
//
// The `params` command's `raw` payload (the knob set) is described separately
// by the /api/knobs manifest. Binary framing (PCM uploads, the float16 slice
// stream) is documented per-entry in the source registry, not typed here.

export const PROTOCOL_VERSION = 1;

// Knob-manifest schema version (the `version` field served by GET
// /api/knobs and the MCP list_knobs tool). Compare against the live
// manifest to detect a stale build, exactly like PROTOCOL_VERSION.
export const KNOB_SCHEMA_VERSION = 1;

export type CommandName =
  | "params"
  | "loop_band"
  | "prompt"
  | "set_prompt_blend"
  | "set_interp_method"
  | "set_depth"
  | "enable_lora"
  | "disable_lora"
  | "manual_slot_add"
  | "manual_slot_pop"
  | "set_timbre_strength"
  | "set_timbre_source"
  | "set_timbre_fixture"
  | "clear_timbre_source"
  | "set_structure_source"
  | "set_structure_fixture"
  | "clear_structure_source"
  | "swap_source"
  | "write_audio";

export const COMMAND_NAMES: readonly CommandName[] = [
  "params",
  "loop_band",
  "prompt",
  "set_prompt_blend",
  "set_interp_method",
  "set_depth",
  "enable_lora",
  "disable_lora",
  "manual_slot_add",
  "manual_slot_pop",
  "set_timbre_strength",
  "set_timbre_source",
  "set_timbre_fixture",
  "clear_timbre_source",
  "set_structure_source",
  "set_structure_fixture",
  "clear_structure_source",
  "swap_source",
  "write_audio",
] as const;

export type EventName =
  | "init_ack"
  | "ready"
  | "error"
  | "params_update"
  | "params_echo"
  | "prompt_blend_echo"
  | "prompt_applied"
  | "lora_catalog"
  | "swap_ready"
  | "swap_failed"
  | "stem_assets"
  | "stem_failed"
  | "depth_applied"
  | "manual_slot_count"
  | "timbre_set"
  | "timbre_cleared"
  | "timbre_failed"
  | "structure_set"
  | "structure_cleared"
  | "structure_failed"
  | "audio_written"
  | "audio_write_failed"
  | "command_failed";

export const EVENT_NAMES: readonly EventName[] = [
  "init_ack",
  "ready",
  "error",
  "params_update",
  "params_echo",
  "prompt_blend_echo",
  "prompt_applied",
  "lora_catalog",
  "swap_ready",
  "swap_failed",
  "stem_assets",
  "stem_failed",
  "depth_applied",
  "manual_slot_count",
  "timbre_set",
  "timbre_cleared",
  "timbre_failed",
  "structure_set",
  "structure_cleared",
  "structure_failed",
  "audio_written",
  "audio_write_failed",
  "command_failed",
] as const;

export type HandshakeCommandName =
  | "upload_track";

export type HandshakeEventName =
  | "upload_ok"
  | "upload_failed";

// ── Command payloads (client → server) ──

export interface ParamsCommand {
  type: "params";
  /** Knob name -> value map. The payload schema is the separate /api/knobs manifest; values are clamped/validated server-side. */
  raw: Record<string, unknown>;
  /** Playhead position in SECONDS (not a 0..1 ratio); used for time-keyed curve sampling. */
  playback_pos?: number;
  /** Client monotonic send time in seconds (performance.now()/1000; arbitrary origin). Lets the server estimate how stale a playback_pos report is when messages queue (network congestion, recv backlog) and advance its playhead estimate accordingly. Optional: absent on older clients, which get the uncompensated behavior. */
  client_time?: number | null;
  /** Flow-control ack: cumulative bytes of binary slice frames received on this connection. The server holds back slice emission while its sent-bytes minus this ack exceeds the in-flight window (DEMON_SLICE_WINDOW_BYTES, default 256 KiB) so a bandwidth-limited link receives fresh slices at link rate instead of an ever-staler buffered backlog. Optional; absent on older clients = no flow control. */
  slice_bytes_rx?: number | null;
  /** Worst observed slice landing lead since the previous params message: how far AHEAD of the audible playhead the most-behind audio slice landed when the client applied it (negative = it landed in already-played audio and the raw source was heard). Folded modulo track duration. The server widens its playback lead to keep this positive — covering network transit and client main-thread scheduling (e.g. throttled background tabs). Optional; omitted when no slice arrived since the last report. */
  slice_lead_s?: number | null;
}

export interface LoopBandCommand {
  type: "loop_band";
  /** Loop start in seconds; null/degenerate clears. */
  start_sec?: number | null;
  /** Loop end in seconds; null/degenerate clears. */
  end_sec?: number | null;
}

export interface PromptCommand {
  type: "prompt";
  /** Prompt A (wire text; enabled-LoRA triggers are prepended client-side). */
  tags: string;
  /** Optional prompt B, cached for A/B blend. */
  tags_b?: string;
  /** Musical key, e.g. "C major". */
  key?: string;
  /** Meter numerator, e.g. "3"/"4"/"6". */
  time_signature?: string;
}

export interface SetPromptBlendCommand {
  type: "set_prompt_blend";
  /** 0.0 = A, 1.0 = B. Clamped to [0,1]. */
  value: number;
}

export interface SetInterpMethodCommand {
  type: "set_interp_method";
  /** Which live blend to retune. */
  path: "prompt" | "timbre" | "structure" | "feedback";
  /** Interpolation curve. */
  method: "slerp" | "linear";
}

export interface SetDepthCommand {
  type: "set_depth";
  /** Target ring depth; clamped to [1, max_pipeline_depth]. */
  value: number;
}

export interface EnableLoraCommand {
  type: "enable_lora";
  /** LoRA id/stem (see /api/loras). */
  id: string;
  /** Target strength the refit lands at. */
  strength?: number;
}

export interface DisableLoraCommand {
  type: "disable_lora";
  id: string;
}

export interface ManualSlotAddCommand {
  type: "manual_slot_add";
}

export interface ManualSlotPopCommand {
  type: "manual_slot_pop";
}

export interface SetTimbreStrengthCommand {
  type: "set_timbre_strength";
  /** 1.0 = full reference, 0.0 = silence baseline. Clamped to [0,1]. */
  value: number;
}

export interface SetTimbreSourceCommand {
  type: "set_timbre_source";
  /** Label echoed back in timbre_set. */
  name?: string;
}

export interface SetTimbreFixtureCommand {
  type: "set_timbre_fixture";
  /** Fixture name (see /api/fixtures). */
  name: string;
}

export interface ClearTimbreSourceCommand {
  type: "clear_timbre_source";
}

export interface SetStructureSourceCommand {
  type: "set_structure_source";
  /** Label echoed back in structure_set. */
  name?: string;
}

export interface SetStructureFixtureCommand {
  type: "set_structure_fixture";
  /** Fixture name (see /api/fixtures). */
  name: string;
}

export interface ClearStructureSourceCommand {
  type: "clear_structure_source";
}

export interface SwapSourceCommand {
  type: "swap_source";
  /** Optional new prompt A. */
  tags?: string;
  key?: string;
  time_signature?: string;
  /** Source label; for server-side loads, the fixture/upload name to read off the pod's disk. */
  fixture_name?: string;
  /** For uploads: which model-ripped stem feeds inference. */
  stem_source_mode?: "full" | "vocals" | "instruments";
  /** When true, the server loads the named source off its own disk and NO binary frame is sent. */
  use_server_source?: boolean;
}

export interface WriteAudioCommand {
  type: "write_audio";
  /** Where the buffer's first sample lands on the source, in playback seconds (sample-exact; no frame or grid alignment required). Default 0. Audio past the source end is trimmed, never wrapped. */
  at_s?: number | null;
  /** replace = overwrite the span (declicked against the existing audio at the edges); sum = overdub on top of what's there. */
  mix?: "replace" | "sum";
  /** fill = treat the buffer as ONE period of a loop and lay it across the whole source, phase-anchored at at_s (sample-exact audio-domain tiling; any period length works). Default none = write once. */
  repeat?: "none" | "fill";
  /** The source generation this write targets (from ready/swap_ready, bumped by every swap). A mismatch is rejected with audio_write_failed instead of splicing into the wrong source. Omit to write against whatever is live. */
  source_epoch?: number | null;
  /** Re-encode the self-timbre conditioning against the updated source (~+50 ms). Ignored when a timbre override is active. Default false. */
  refresh_timbre?: boolean;
}

// ── Event payloads (server → client) ──

export interface InitAckEvent {
  type: "init_ack";
  /** Server-minted session id, sent as soon as log context binds so client startup failures correlate with pod logs. */
  session_id?: string;
  /** The config client_id echoed back, or null when the client sent none. */
  client_id?: string | null;
}

export interface ReadyEvent {
  type: "ready";
  duration: number;
  channels: number;
  sample_rate: number;
  lora_catalog?: unknown[];
  lora_dir?: string;
  bpm?: number | null;
  key?: string | null;
  time_signature?: string | null;
  checkpoint?: string;
  checkpoint_scale?: string;
  pipeline_depth?: number;
  max_pipeline_depth?: number;
  /** LoRA ids the server will auto-enable on the first tick (from the session's initial enable set); empty when none. */
  lora_pending_enable?: unknown[];
  /** Server-minted session id, echoed for client/analytics log correlation. */
  session_id?: string;
  /** Source generation counter (0 at create, bumped by every swap). Echo it in write_audio to pin a write to the source it was computed against. */
  source_epoch?: number;
  /** Backend-declared audio geometry: {sample_rate, channels, chunk_rate_hz, duration_s|null}. chunk_rate_hz is the generation cadence (latent fps for diffusion, frame rate for AR models); duration_s null is reserved for endless streams. */
  geometry?: Record<string, unknown>;
  /** Backend capability mask: {capability: bool} over the Capabilities fields (swap, timbre, structure, lora, ...). Client panels and MCP tools gate on it; commands tagged with a matching `requires` fail with command_failed when the bit is false. */
  capabilities?: Record<string, unknown>;
  /** Per-session knob manifest: the same {version, knobs} envelope GET /api/knobs serves, but backend-owned and session-resolved (SDE mode, enabled lora_str_<id> knobs). /api/knobs remains the static pre-session probe. */
  knob_manifest?: Record<string, unknown>;
  /** Active manual steering slot count; drives the client's man_*_<N> row rendering. Updated live via the manual_slot_count event. */
  manual_slot_count?: number;
  /** Server-imposed ceiling on manual steering slots; gates the client's + button. */
  manual_slot_cap?: number;
  /** True when the session's checkpoint has a reachable steering-vector bundle; false hides the steering surface (the steer_*\/man_* knobs are absent from the manifest too). */
  steering_available?: boolean;
}

export interface ErrorEvent {
  type: "error";
  code?: string;
  message?: string;
  build_command?: string;
  /** Present only on the engine_not_built code: the source duration whose TRT profile is missing. */
  duration_s?: number;
}

export interface ParamsUpdateEvent {
  type: "params_update";
  /** Applied params + runtime telemetry (num_gens, tick_ms, dec_ms). */
  params: Record<string, unknown>;
}

export interface ParamsEchoEvent {
  type: "params_echo";
  raw: Record<string, unknown>;
}

export interface PromptBlendEchoEvent {
  type: "prompt_blend_echo";
  value: number;
}

export interface PromptAppliedEvent {
  type: "prompt_applied";
  tags?: string;
}

export interface LoraCatalogEvent {
  type: "lora_catalog";
  catalog: unknown[];
}

export interface SwapReadyEvent {
  type: "swap_ready";
  duration: number;
  sample_rate: number;
  channels: number;
  bpm?: number | null;
  key?: string | null;
  time_signature?: string | null;
  fixture_name?: string | null;
  /** Source generation counter after this swap; write_audio sends targeting the old source are rejected. */
  source_epoch?: number;
}

export interface SwapFailedEvent {
  type: "swap_failed";
  error?: string;
  /** Present only when the swap failed on a missing TRT engine: the command to build the profile for the new source's duration. */
  build_command?: string;
}

export interface StemAssetsEvent {
  type: "stem_assets";
  fixture_name: string;
  sample_rate: number;
  channels: number;
  frames: number;
  /** Ordered subset of ("vocals","instruments"). */
  stems: unknown[];
  source_mode?: "full" | "vocals" | "instruments";
}

export interface StemFailedEvent {
  type: "stem_failed";
  fixture_name?: string;
  error?: string;
}

export interface DepthAppliedEvent {
  type: "depth_applied";
  /** The clamped applied depth. */
  value: number;
}

export interface ManualSlotCountEvent {
  type: "manual_slot_count";
  /** The live manual steering slot count after the command. */
  count: number;
}

export interface TimbreSetEvent {
  type: "timbre_set";
  name: string;
  duration: number;
}

export interface TimbreClearedEvent {
  type: "timbre_cleared";
}

export interface TimbreFailedEvent {
  type: "timbre_failed";
  error?: string;
}

export interface StructureSetEvent {
  type: "structure_set";
  name: string;
  duration: number;
}

export interface StructureClearedEvent {
  type: "structure_cleared";
}

export interface StructureFailedEvent {
  type: "structure_failed";
  error?: string;
}

export interface AudioWrittenEvent {
  type: "audio_written";
  start_s: number;
  end_s: number;
  /** The source generation the write landed on (matches ready/swap_ready). */
  source_epoch: number;
}

export interface AudioWriteFailedEvent {
  type: "audio_write_failed";
  error?: string;
}

export interface CommandFailedEvent {
  type: "command_failed";
  /** The rejected command's wire name. */
  command: string;
  /** The Capabilities field the command needs and the session's backend doesn't declare. */
  requires: string;
  /** Human-readable reason. */
  error?: string;
}

// ── Session-init config (client → server, sent at handshake) ──

export interface SessionConfigPayload {
  sde?: boolean;
  lora?: boolean;
  vae_window?: number;
  crop?: number;
  depth?: number;
  steps?: number;
  prompt?: string;
  prompt_b?: string | null;
  fast_vae?: boolean;
  walk_window?: boolean;
  walk_window_s?: number;
  lead_floor_s?: number;
  lead_ceiling_s?: number;
  lead_release_tau_s?: number;
  fixture_name?: string | null;
  use_server_fixture?: boolean;
  stem_source_mode?: string | null;
  enabled_loras?: unknown[];
  lora_strengths?: Record<string, unknown>;
  lora_paths?: unknown[];
  client_id?: string | null;
  backend?: string;
  // SessionConfig is permissive; extras pass through.
  [k: string]: unknown;
}

// ── Init-phase upload handshake ──

export interface UploadTrackCommand {
  type: "upload_track";
  /** Requested track label; deduped server-side. */
  name?: string;
  /** Optional key override; forces a re-encode instead of the content-dedup fast path. */
  key?: string;
  /** Optional meter override; same effect as key. */
  time_signature?: string;
}

export interface UploadOkEvent {
  type: "upload_ok";
  /** Final persisted track name (may differ from the requested name after dedup/uniquify). */
  name: string;
  bpm?: number;
  key?: string;
  time_signature?: string;
  duration_s?: number;
  samples?: number;
  /** True when the vocal/instrument stem rip is still running on a background thread. The track is immediately swappable (full source); stems land later via a pushed stem_assets frame on the live session (or stem_failed). */
  stems_pending?: boolean;
}

export interface UploadFailedEvent {
  type: "upload_failed";
  error?: string;
}

// ── Discriminated unions ──

export type WireCommand =
  | ParamsCommand
  | LoopBandCommand
  | PromptCommand
  | SetPromptBlendCommand
  | SetInterpMethodCommand
  | SetDepthCommand
  | EnableLoraCommand
  | DisableLoraCommand
  | ManualSlotAddCommand
  | ManualSlotPopCommand
  | SetTimbreStrengthCommand
  | SetTimbreSourceCommand
  | SetTimbreFixtureCommand
  | ClearTimbreSourceCommand
  | SetStructureSourceCommand
  | SetStructureFixtureCommand
  | ClearStructureSourceCommand
  | SwapSourceCommand
  | WriteAudioCommand;

export type WireEvent =
  | InitAckEvent
  | ReadyEvent
  | ErrorEvent
  | ParamsUpdateEvent
  | ParamsEchoEvent
  | PromptBlendEchoEvent
  | PromptAppliedEvent
  | LoraCatalogEvent
  | SwapReadyEvent
  | SwapFailedEvent
  | StemAssetsEvent
  | StemFailedEvent
  | DepthAppliedEvent
  | ManualSlotCountEvent
  | TimbreSetEvent
  | TimbreClearedEvent
  | TimbreFailedEvent
  | StructureSetEvent
  | StructureClearedEvent
  | StructureFailedEvent
  | AudioWrittenEvent
  | AudioWriteFailedEvent
  | CommandFailedEvent;

export type HandshakeCommand =
  | UploadTrackCommand;

export type HandshakeEvent =
  | UploadOkEvent
  | UploadFailedEvent;
