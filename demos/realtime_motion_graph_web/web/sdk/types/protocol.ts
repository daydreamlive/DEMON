// Client-side protocol types. The wire message shapes themselves are
// GENERATED from the Python registry (see ./wireContract.gen.ts); this file
// holds only what the contract can't express: client view-models
// (AudioSlice, SwapReadyDetail), catalog entry shapes that ride inside
// `list`-typed event fields (LoraCatalogEntry), element-type narrowings of
// generated events, and the binary-framing constants.

import type {
  SessionConfigPayload,
  StemAssetsEvent,
  SwapReadyEvent,
} from "./wireContract.gen";

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

/** Element shape of the `lora_catalog` list carried by the `ready` and
 *  `lora_catalog` events (the wire contract types list elements as
 *  `unknown`; this is the client-side refinement). */
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

/** Sent by the client at session start (config phase). Generated from the
 *  SessionConfig dataclass; kept under its historical name. */
export type SessionConfig = SessionConfigPayload;

/** `swap_ready` wire payload. Identical to the generated event shape;
 *  aliased so client code keeps its historical name. */
export type SwapReadyMessage = SwapReadyEvent;

/** `stem_assets` wire payload, with the stem list narrowed to the names
 *  the server actually emits (the registry types list elements as
 *  `unknown`; the allowed values live in its description). */
export interface StemAssetsMessage extends StemAssetsEvent {
  stems: ("vocals" | "instruments")[];
}

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
export interface SwapReadyDetail extends SwapReadyEvent {
  interleaved: Float32Array;
}

/** Backend-declared audio geometry, carried in `ready.geometry` (the
 *  contract types it as a plain dict; this is the client-side
 *  refinement). Servers from before the backend-seam contract surface
 *  omit it — fall back to the legacy flat ready fields / constants. */
export interface ReadyGeometry {
  sample_rate: number;
  channels: number;
  /** Generation cadence in Hz: latent fps for diffusion backends
   *  (ACE = 25), frame rate for AR backends. */
  chunk_rate_hz: number;
  /** Null is reserved for endless streams (v2 `append` song shape);
   *  fixed-duration backends always declare a real duration. */
  duration_s: number | null;
}

/** Backend capability mask, carried in `ready.capabilities`. Keys are
 *  the server's Capabilities field names (swap, timbre, structure,
 *  lora, ...); kept as a record so new capabilities don't need client
 *  type churn. A missing mask (older server / recorded replay) means
 *  "ungated" — treat every capability as available. */
export type CapabilityMask = Record<string, boolean>;

// Legacy constants. Since the Phase-2 contract surface, the declared
// truth for a session's audio shape is the backend-sourced
// `ready.geometry` block (see ReadyGeometry / RemoteBackend.geometry);
// these remain for device-level code (AudioContext creation) and
// pre-geometry servers.
export const SAMPLE_RATE = 48000;
/** 60 s of audio at 25 fps latents. */
export const T = 1500;
export const CROSSFADE_SECONDS = 0.025;
export const SLICE_HDR_SIZE = 23; // 1+4+4+2+4+4+4
export const SLICE_FLAG_RAW = 0;
export const SLICE_FLAG_DELTA = 1;
