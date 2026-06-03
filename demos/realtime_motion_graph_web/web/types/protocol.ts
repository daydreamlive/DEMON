// WS wire protocol shapes. Mirrors the Python side
// (DEMON/demos/realtime_motion_graph_web/protocol.py + ws_adapter.py).

/** Normalized LoRA metadata mirrored from the Python sidecar loader
 *  (`acestep/lora_metadata.py`). Always shipped — fields are null when
 *  the LoRA has no `<stem>.metadata.json` or `.trigger.txt` sidecar.
 *  `has_metadata` is true iff a real `metadata.json` was loaded (vs a
 *  synthesized fallback record). */
export interface LoraMetadata {
  id: string;
  name: string;
  description: string | null;
  /** The canonical activation token — what we copy to the clipboard
   *  and prepend to the prompt when auto_prepend_lora_triggers is on.
   *  One of the entries in `trigger_words`, or null when the LoRA has
   *  no documented trigger. */
  primary_trigger_word: string | null;
  /** All known activation tokens. May contain multiple aliases. The
   *  runtime only acts on `primary_trigger_word`; the rest are for
   *  documentation / advanced surfaces. */
  trigger_words: string[];
  recommended_strength: number | null;
  recommended_steps: number | null;
  recommended_shift: number | null;
  recommended_guidance: number | null;
  primary_genre: string | null;
  secondary_genres: string[];
  tags: string[];
  moods: string[];
  /** Free-form base-model identifier (e.g. "AceStep v1.5 Turbo"). For
   *  display only; the runtime compares ``base_model_scale``. */
  base_model: string | null;
  /** "2B" or "5B". Compared against the active session's
   *  ``checkpoint_scale`` to hide LoRAs trained for a different
   *  checkpoint. Null when the sidecar doesn't declare it — the UI
   *  treats null as "compatible with everything" so legacy LoRAs
   *  without a scale declaration aren't silently hidden. */
  base_model_scale: string | null;
  has_metadata: boolean;
}

export interface LoraCatalogEntry {
  id: string;
  name?: string;
  path?: string;
  state?: string;
  strength?: number;
  materialized_bytes?: number;
  /** Full normalized metadata record. Always present from servers that
   *  speak the v2 catalog shape; older servers may omit it. */
  metadata?: LoraMetadata;
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
  /** Playback-lead tuning forwarded to the server-side PipelineRunner.
   *  Bound how far the adaptive playhead-lead buffer self-sizes. Omitted
   *  fields fall back to the server default. */
  lead_floor_s?: number;
  lead_ceiling_s?: number;
  lead_release_tau_s?: number;
  key?: string;
  /** One of "2" | "3" | "4" | "6" — meter numerator that the encoder
   *  bakes into the prompt. Same intentional-omission rule as `key`:
   *  the server resolves it from the fixture sidecar (or default "4")
   *  and echoes the result back in `ready.time_signature`. */
  time_signature?: string;
  enabled_loras?: string[];
  prompt?: string;
  prompt_b?: string;
  lora_strengths?: Record<string, number>;
  /** For uploaded tracks, asks the server to model-rip stems and choose
   *  which source should feed inference. Built-in fixtures omit this. */
  stem_source_mode?: "full" | "vocals" | "instruments";
  /** When true, the backend loads a known fixture from its own cache and
   *  the client skips sending the audio frame. Capability-probed first. */
  use_server_fixture?: boolean;
  /** Optional opaque client identifier (e.g. an analytics distinct id).
   *  The pod binds it into loguru's contextvars so every log record on
   *  this connection carries it — useful for joining a browser trace to
   *  a pod-side log line when a user reports a failure. */
  client_id?: string;
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
  checkpoint?: string | null;
  checkpoint_scale?: string | null;
  /** Active StreamPipeline ring-buffer depth (concurrent denoising
   *  slots). Server clamps the requested ``depth`` to
   *  [1, max_pipeline_depth] before building the pipeline; the result
   *  is echoed here so the UI can render the current value without
   *  guessing. */
  pipeline_depth?: number;
  /** Largest depth the loaded backend can serve — TRT engine's
   *  ``hidden_states`` batch_max for TRT decoders, 4 for eager /
   *  compile. The client clamps the depth control to
   *  [1, max_pipeline_depth]. */
  max_pipeline_depth?: number;
}

/** Server ack for a ``set_depth`` request. ``value`` is the actually-
 *  applied depth (clamped to [1, max_pipeline_depth] server-side). */
export interface DepthAppliedMessage {
  type: "depth_applied";
  value: number;
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
  /** Server-resolved BPM for the new source (sidecar value on a hit,
   *  live librosa value otherwise). The client must mirror this into
   *  the perf store so the swap-target's tempo replaces the previous
   *  track's — otherwise the Detected: readout keeps showing the old
   *  BPM after the swap completes. */
  bpm?: number | null;
  key?: string;
  time_signature?: string;
  /** Server echoes the requested source label (fixture name for known
   *  fixtures, upload name for ad-hoc PCM). Populated for swaps driven
   *  by the front-end as well as MCP — useMcpMirror reads this to keep
   *  the fixture dropdown in sync with externally-triggered swaps. */
  fixture_name?: string;
}

export interface StemAssetsMessage {
  type: "stem_assets";
  fixture_name: string;
  sample_rate: number;
  channels: number;
  frames: number;
  stems: ("vocals" | "instruments")[];
  source_mode?: "full" | "vocals" | "instruments";
}

export interface StemFailedMessage {
  type: "stem_failed";
  fixture_name?: string;
  error?: string;
}

export interface SwapFailedMessage {
  type: "swap_failed";
  error?: string;
}

/** Ack for `set_timbre_source` / `set_timbre_fixture`. `duration` is the
 *  applied clip length in seconds, after server-side cap to the playback
 *  source duration. */
export interface TimbreSetMessage {
  type: "timbre_set";
  name: string;
  duration: number;
}

/** Ack for `clear_timbre_source`. */
export interface TimbreClearedMessage {
  type: "timbre_cleared";
}

/** Failure ack for any `set_timbre_*` path. */
export interface TimbreFailedMessage {
  type: "timbre_failed";
  error?: string;
}

/** Ack for `set_structure_source` / `set_structure_fixture`. `duration`
 *  is the applied clip length in seconds before pad/trim to the playback
 *  source's sample count. The server log line also reports the
 *  post-pad/trim target length, but the wire payload does not. */
export interface StructureSetMessage {
  type: "structure_set";
  name: string;
  duration: number;
}

/** Ack for `clear_structure_source`. */
export interface StructureClearedMessage {
  type: "structure_cleared";
}

/** Failure ack for any `set_structure_*` path, AND the server-emitted
 *  notice when a swap drops a previously-set structure override
 *  (`error` will be of the form `"dropped after swap: ..."`). */
export interface StructureFailedMessage {
  type: "structure_failed";
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
  | StemAssetsMessage
  | StemFailedMessage
  | DepthAppliedMessage
  | TimbreSetMessage
  | TimbreClearedMessage
  | TimbreFailedMessage
  | StructureSetMessage
  | StructureClearedMessage
  | StructureFailedMessage
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
