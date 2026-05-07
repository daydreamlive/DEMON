// WS wire protocol shapes. Mirrors the Python side
// (DEMON/demos/realtime_motion_graph_web/protocol.py + backend.py).

export interface LoraCatalogEntry {
  id: string;
  name?: string;
  path?: string;
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
}

export interface SwapFailedMessage {
  type: "swap_failed";
  error?: string;
}

export type ServerJsonMessage =
  | ReadyMessage
  | ServerErrorMessage
  | ParamsUpdateMessage
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
}

/** Detail payload for `swap_ready` events on RemoteBackend. */
export interface SwapReadyDetail extends SwapReadyMessage {
  interleaved: Float32Array;
}

export const SAMPLE_RATE = 48000;
/** 60 s of audio at 25 fps latents. */
export const T = 1500;
export const CROSSFADE_SECONDS = 0.05;
export const SLICE_HDR_SIZE = 23; // 1+4+4+2+4+4+4
export const SLICE_FLAG_RAW = 0;
export const SLICE_FLAG_DELTA = 1;
