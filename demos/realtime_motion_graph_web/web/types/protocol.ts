// WS wire protocol shapes. Mirrors the Python side
// (DEMON/demos/realtime_motion_graph_web/protocol.py + backend.py).

export interface LoraCatalogEntry {
  id: string;
  name?: string;
  path?: string;
  /** Activation word the LoRA was trained against, sourced from a
   *  `<stem>.trigger.txt` sidecar next to the .safetensors. Always
   *  present in the catalog payload — empty string when no sidecar
   *  exists (no documented trigger for that LoRA, e.g. synthpop).
   *  The engine handles the actual prompt prepending server-side at
   *  encode time; this field is surfaced to the UI for transparency /
   *  tooltips only — do NOT inject it into promptA/promptB. */
  trigger?: string;
  state?: string;
  strength?: number;
  materialized_bytes?: number;
}

/** Sent by the client at session start (config phase). */
export interface SessionConfig {
  sde?: boolean;
  lora?: boolean;
  depth?: number;
  vae_window?: number;
  crop?: number;
  steps?: number;
  fast_vae?: boolean;
  key?: string;
  /** One of "2" | "3" | "4" | "6" — meter numerator that the encoder
   *  bakes into the prompt. Same intentional-omission rule as `key`:
   *  the server resolves it from the fixture sidecar (or default "4")
   *  and echoes the result back in `ready.time_signature`. */
  time_signature?: string;
  enabled_loras?: string[];
  prompt?: string;
  lora_strengths?: Record<string, number>;
  // Allow extras — pyproject's config object is permissive.
  [k: string]: unknown;
}

/** First JSON the server returns once the audio upload is in. */
export interface ReadyMessage {
  type: "ready";
  duration: number;
  channels: number;
  sample_rate: number;
  lora_catalog?: LoraCatalogEntry[];
  lora_dir?: string;
  bpm?: number | null;
  key?: string | null;
  time_signature?: string | null;
}

/** Structured init failure from the server. */
export interface ServerErrorMessage {
  type: "error";
  code?: string;
  message?: string;
  build_command?: string;
}

export interface ParamsUpdateMessage {
  type: "params_update";
  params: Record<string, number>;
}

// Emitted by the server when a `params` message came from the MCP control
// bus rather than the browser's own WS — carries the raw knob values the
// MCP set so the front-end can mirror them in the UI. The browser's own
// param updates do NOT echo (the UI is already in sync with itself).
export interface ParamsEchoMessage {
  type: "params_echo";
  raw: Record<string, number | string | boolean>;
}

// MCP-driven analogue of ParamsEchoMessage for the prompt_blend slider,
// which travels over its own WS message (set_prompt_blend) rather than
// the generic params payload. Same handshake: server skips the apply
// when the message came from the control bus and mirrors the target
// back to the browser so the smoothed slider tween can take it from
// there.
export interface PromptBlendEchoMessage {
  type: "prompt_blend_echo";
  value: number;
}

export interface PromptAppliedMessage {
  type: "prompt_applied";
  tags?: string;
}

export interface LoraCatalogMessage {
  type: "lora_catalog";
  catalog: LoraCatalogEntry[];
}

export interface SwapReadyMessage {
  type: "swap_ready";
  duration: number;
  channels: number;
  key?: string;
  time_signature?: string;
  /** Server echoes the requested source label (fixture name for known
   *  fixtures, upload name for ad-hoc PCM). Populated for swaps driven
   *  by the front-end as well as MCP — useMcpMirror reads this to keep
   *  the fixture dropdown in sync with externally-triggered swaps. */
  fixture_name?: string;
}

export interface SwapFailedMessage {
  type: "swap_failed";
  error?: string;
}

export type ServerJsonMessage =
  | ReadyMessage
  | ServerErrorMessage
  | ParamsUpdateMessage
  | ParamsEchoMessage
  | PromptBlendEchoMessage
  | PromptAppliedMessage
  | LoraCatalogMessage
  | SwapReadyMessage
  | SwapFailedMessage
  | { type: string; [k: string]: unknown };

/** Parsed binary slice from the server. */
export interface AudioSlice {
  flags: number;
  startSample: number;
  numSamples: number;
  channels: number;
  /** Per-generation engine time in ms. */
  tickMs: number;
  /** Decoder latency in ms. */
  decMs: number;
  /** Number of generation calls represented by this slice. */
  numGens: number;
  /** Decoded float32 PCM, interleaved. */
  audio: Float32Array;
  /** Source-buffer epoch this slice was received under. Increments on each
   *  swap_ready. Consumers compare against `AudioPlayer.swapCount` to drop
   *  slices that were generated for a previous track but only finished
   *  decoding (or arrived) after the swap. */
  epoch: number;
}

/** Detail payload for `swap_ready` events on RemoteBackend. */
export interface SwapReadyDetail extends SwapReadyMessage {
  interleaved: Float32Array;
}

export const SAMPLE_RATE = 48000;
/** 60 s of audio at 25 fps latents. */
export const T = 1500;
export const CROSSFADE_SECONDS = 0.025;
export const SLICE_HDR_SIZE = 23; // 1+4+4+2+4+4+4
export const SLICE_FLAG_RAW = 0;
export const SLICE_FLAG_DELTA = 1;
