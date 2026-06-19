// The `config.json` operator-defaults schema — the portable type surface
// shared by every DEMON frontend (web, M4L, VST). Moved here from the web
// app (`web/lib/config.ts`); the web app re-exports these so its existing
// call sites are unchanged. See ./README context in the SDK README: this is
// the client-side operator-defaults format, distinct from the backend-served
// `/api/knobs` and `/api/protocol` contracts.

import type { LoopGridRes } from "./enums";

// Runtime array is the single source of truth; the union derives from it (same
// pattern as DCW_MODES etc. in ./enums) so the C++ contract emitter walks the
// option values and the drift guard catches any addition/rename.
export const SWAP_SOURCE_MODES = ["full", "vocals", "instruments"] as const;
export type SwapSourceMode = (typeof SWAP_SOURCE_MODES)[number];

/** One entry in `engine.enabled_loras`. Bare string = enable that LoRA
 *  at its sidecar's recommended_strength (or controls.lora_default_strength
 *  as fallback). Object form sets an inline strength override. `name`
 *  may be the filename stem ("deep_house-v1") or the sidecar's display
 *  name ("Deep House"); matching is case-insensitive. */
export type EnabledLoraEntry = string | { name: string; strength?: number };

export interface RtmgConfigEngine {
  sde: boolean;
  lora: boolean;
  depth: number;
  vae_window: number;
  crop: number;
  steps: number;
  fast_vae: boolean;
  /** Route long sources through the walk_window_s (60s) DiT engine by
   * sliding a fixed-T window across the song each tick. Lets a 240s
   * song play through the 60s engine without paying the 240s engine's
   * parameter-update latency. Backend ignores when source ≤ window. */
  walk_window?: boolean;
  walk_window_s?: number;
  /** Playback-lead tuning (server-side PipelineRunner). The lead is the
   *  adaptive audio buffer placed ahead of the live playhead; these bound
   *  how far it self-sizes. Raise the floor for more robustness to GPU
   *  contention (screen capture, a co-resident visualizer, a second
   *  process) at the cost of latency; the ceiling caps how far contention
   *  can inflate it; the release tau sets how fast a contention spike
   *  decays back out. Defaults are the "midway" profile. The server clamps
   *  ``lead_release_tau_s`` up to ``lead_ceiling_s`` if a config sets it
   *  lower (monotonic-decode invariant). Omit any field to use the
   *  server-side default. */
  lead_floor_s?: number;
  lead_ceiling_s?: number;
  lead_release_tau_s?: number;
  key: string;
  /** Default meter numerator the operator dropdown starts on. Mirrors
   * `key` in posture: the server's session-init resolver still wins on
   * sidecar hits, so this is purely a UI seed for the manual "Override"
   * control. Allowed values: "2" | "3" | "4" | "6". */
  time_signature: string;
  /** LoRAs to auto-enable on first catalog load. Empty falls back to
   * the count-rule in useLoraStore (first two from the sorted catalog,
   * with name-fallback per slot). See `EnabledLoraEntry` for the shape
   * of each element. */
  enabled_loras: EnabledLoraEntry[];
  /** When true, enabling a LoRA prepends its primary trigger word to
   *  promptA and promptB so the user sees exactly what the encoder
   *  sees. When false, the trigger never enters the prompt text unless
   *  the user types it themselves — useful for prompt-driven workflows
   *  that want to stay 100% manual. Disabling a LoRA best-effort
   *  removes its trigger from the prompt when it's still at the head.
   *  Defaults to true. */
  auto_prepend_lora_triggers?: boolean;
  /** When true, the LoRA library shows every entry regardless of
   *  whether its trained ``base_model_scale`` matches the active
   *  checkpoint. Useful for inspecting your full collection while
   *  on a specific checkpoint. Default false (auto-hide). LoRAs with
   *  no declared scale are shown either way — we don't hide what we
   *  can't classify. */
  show_incompatible_loras?: boolean;
  /** Maximum number of LoRAs that can be enabled simultaneously.
   *  Null / undefined / non-positive means "no cap" — every LoRA in
   *  the catalog can be enabled at once (the OSS default; preserves
   *  parity for local-DEMON users).
   *
   *  Set this on a hosted deployment that wants a hard ceiling — each
   *  enabled LoRA materializes a refit-state buffer (~1.2 GB on the
   *  current acestep-v15-turbo checkpoint) on top of decoder + VAE
   *  engines, so on a 32 GB card you can OOM cleanly after the third
   *  one when paired with a long-source vae_encode profile.
   *
   *  Used as a constant cap when ``max_concurrent_loras_tiers`` is
   *  absent. With tiers present, this field is the FALLBACK cap used
   *  when no tier matches the current source duration (e.g. before a
   *  source is loaded, or a source longer than every tier threshold).
   *
   *  Enforcement is honoured by ``useLoraStore.enable`` and by the
   *  catalog auto-enable seed (config-driven defaults beyond the cap
   *  are silently clipped). Disabling is never blocked.
   *  ``canEnableMore()`` on the store exposes the predicate so the
   *  UI can render disabled "+" buttons with a "Max N active" hint. */
  max_concurrent_loras?: number | null;
  /** Source-duration-aware cap tiers. The active cap is the ``cap``
   *  field of the FIRST tier whose ``up_to_s`` is ≥ the current source
   *  duration; when no tier matches, falls back to
   *  ``max_concurrent_loras`` (else uncapped).
   *
   *  Why duration-aware: the 240s ``vae_encode`` engine reserves a
   *  ~16 GiB workspace at runtime, which leaves less room for LoRA
   *  materializations than the 60s or 120s engines. A hosted
   *  deployment can keep the cap relaxed (e.g. 3) for short sources
   *  that load the 60s engine and tighten it (e.g. 2) for sources
   *  that trigger the 240s engine.
   *
   *  Example:
   *  ```json
   *  "max_concurrent_loras_tiers": [
   *    { "up_to_s": 60,  "cap": 3 },
   *    { "up_to_s": 120, "cap": 3 },
   *    { "up_to_s": 240, "cap": 2 }
   *  ]
   *  ```
   *  Order doesn't matter — the resolver sorts by ``up_to_s`` ascending.
   *  Recomputed on session start AND on every source swap so the cap
   *  tracks the live engine workspace. */
  max_concurrent_loras_tiers?: Array<{
    up_to_s: number;
    cap: number;
  }> | null;
  /** Hard ceiling on how long a slice of audio the engine will accept
   *  as a source. The upload UI shows an interactive trim dialog
   *  (WaveformTrimDialog) on every upload — the dialog clamps the
   *  selectable window to this value, and only the trimmed slice is
   *  ever sent to the engine.
   *
   *  Default is 120 s: the 60 s and 120 s TRT engines are the stable
   *  pair on current GPUs. The 240 s vae_encode engine reserves
   *  ~16 GiB workspace at runtime which has driven CUDA-OOM crashes
   *  on 32 GiB cards; keeping the cap at 120 s avoids that profile
   *  until the OOM and the related context-creation-returns-None
   *  crash in acestep/nodes/vae_nodes.py are addressed. Operators
   *  with bigger cards (≥48 GiB) who want the 240 s profile can set
   *  this to 240 in their override config. */
  max_source_duration_s?: number;
  /** XL (5B) variant overrides. When the active checkpoint scale is
   *  "5B", these win over their base siblings at applyConfig time.
   *  Absent / undefined falls through to the base field. Selection
   *  happens once at boot in applyConfig() using the scale already
   *  resolved by /api/loras. */
  depth_xl?: number;
  enabled_loras_xl?: EnabledLoraEntry[];
}

export interface RtmgConfigPrompts {
  a: string;
  b: string;
  blend: number;
  /** XL (5B) variant overrides — same selection rule as engine.*_xl. */
  a_xl?: string;
  b_xl?: string;
  blend_xl?: number;
}

export interface RtmgConfigEffects {
  parallax_strength: number;
  bloom_on_kick: number;
  bloom_threshold: number;
  warp_strength: number;
}

export interface RtmgConfigAudio {
  /** Initial state of the loudness matcher at boot. Operator can still
   *  flip it via the LUFS button — this is the seed, not a lock. */
  lufs_enabled: boolean;
  /** Sliding-window length in seconds for the loudness-matching meter
   *  (BS.1770 short-term LUFS). 3 s is the standard. Lowering trades
   *  stability for responsiveness; below ~1.5 s, transients can lock
   *  the high-water mark hot. */
  lufs_window_sec: number;
  /** Loudness metric the matcher uses. "lufs" = ITU-R BS.1770 K-weighted
   *  (broadcast standard, slightly over-reads bright/distorted material).
   *  "dba" = IEC 61672 A-weighted RMS (closer to perceived loudness on
   *  spectrally imbalanced content; tighter step-test gaps in offline
   *  validation). Defaults to "lufs" for backward compatibility. */
  lufs_metric: "lufs" | "dba";
  /** Multiplier applied to the source's true peak when adapting the
   *  matcher's peak ceiling. The default -1 dBTP ceiling (0.891) is
   *  raised to max(0.891, source_peak * lufs_peak_headroom). 4 = +12 dB
   *  of boost-headroom above source peak. Lower values cap how much
   *  the matcher can boost a quieter denoised signal (1.0 = match
   *  source peak; below ~2 the gap to a much quieter denoised stream
   *  cannot be fully closed). Higher values allow more boost at the
   *  cost of harder DAC clipping. */
  lufs_peak_headroom: number;
  /** Disengage threshold in dB. When the chunk at the playhead reads
   *  more than this far below target (or is fully silent), the matcher
   *  ramps gain back to 1.0 instead of computing a makeup gain. Without
   *  this, silence in the model's output (mid-song silence, end of
   *  track, start of loop) gets multiplied by tens to hundreds of
   *  times to "match" source loudness, amplifying low-level artifacts.
   *  30 dB is well outside the range musical content reaches relative
   *  to a gated integrated target; lowering it (e.g. 20 dB) makes the
   *  matcher disengage earlier on quiet passages too. */
  lufs_silence_floor_db: number;
  /** Hysteresis band on the silence floor, in dB. Once the matcher has
   *  disengaged, it re-engages only when the chunk reads back within
   *  (floor - hysteresis) dB of target. Stops chunks hovering at the
   *  threshold from flipping every tick (audible as volume swells).
   *  Set to 0 for a hard threshold; raise to widen the dead band. */
  lufs_silence_floor_hysteresis_db: number;
}

/** controls.* — initial slider values plus the DCW companion controls
 * (enabled / mode / wavelet) and lora_default_strength. Numeric entries
 * seed sliderValues + sliderTargets; the named DCW entries drive the
 * non-numeric DCW state. Unknown keys are ignored. */
export type RtmgConfigControls = Record<string, number | boolean | string>;

/** Per-channel slider range + direction. When present, overrides the
 *  SLIDER_META max for that param and adds a min floor (slider drag,
 *  MIDI knobs, keyboard bumps, and curve writes all clamp to this
 *  range via clampToMeta in usePerformanceStore). `reverse` is a UI
 *  affordance — when true, dragging the slider UP (or turning the
 *  MIDI knob clockwise, or hitting ArrowUp) sends a LOWER engine
 *  value. The stored value still lives in [min, max]; only the
 *  input→value mapping is flipped. Use for channels that "sound
 *  better when turned down" — the operator's instinct to push up
 *  produces the desired result. */
export interface RtmgChannelRange {
  min: number;
  max: number;
  reverse: boolean;
}
export type RtmgConfigChannelRanges = Record<string, RtmgChannelRange>;

/** On session start, snap engine denoise to 0 and play a visual-only
 * display glide from the slider's prior value down to 0 over `glide_ms`.
 * The engine value never moves with the glide; purely a "hear the source
 * first" onboarding cue. Set `enabled: false` to skip the snap entirely;
 * seed `controls.denoise` to whatever starting value you want in that
 * case. The glide is only visible when the slider's value at session-start
 * is non-zero (first session uses controls.denoise; later sessions use
 * wherever the user left it). */
export interface RtmgConfigDenoiseSessionGate {
  enabled: boolean;
  glide_ms: number;
}

/** One control point on a schedule curve. Mirrors the runtime
 *  CurvePoint in store/useCurveStore — duplicated on the wire shape
 *  so the config can be authored and parsed without reaching across
 *  module boundaries. */
export interface RtmgConfigCurvePoint {
  /** 0..1 along the track timeline. Endpoints pinned at 0 and 1. */
  x: number;
  /** 0..1 normalised. Mapped to the param's min/max at apply time. */
  y: number;
  mode: "smooth" | "linear" | "step";
}

export interface RtmgConfigCurve {
  enabled: boolean;
  /** Always ≥ 2 points; first.x === 0 and last.x === 1. */
  points: RtmgConfigCurvePoint[];
}

/** Per-param schedule curves the user (or an operator-supplied config)
 *  draws against the track timeline. Keyed by param name — the fixed
 *  set (denoise, hint_strength, feedback, shift, noise_share) plus
 *  dynamic LoRA strength curves (lora_str_<id>). */
export interface RtmgConfigCurves {
  /** Master enable. When false, no curve drives any param regardless
   *  of per-curve enabled flags. */
  scheduleEnabled: boolean;
  curves: Record<string, RtmgConfigCurve>;
}

/** Playback loop region ("brace") — the same state WaveformScrubBox edits,
 *  lifted into the config so an exported sound carries its loop. */
export interface RtmgConfigLoop {
  /** Loop region in seconds, or null when no region is set. */
  band: { start: number; end: number } | null;
  /** Whether the region is actively looping (vs. armed-but-off). */
  enabled: boolean;
  /** Snap resolution for loop edits. */
  grid: LoopGridRes;
  /** Global full-buffer loop toggle (store `loopOn`) — distinct from the
   *  band loop. Optional so older exports that predate it leave the live
   *  transport setting untouched on import. */
  fullBuffer?: boolean;
}

/** The browser demo's own presentation, playback, and interaction
 *  settings. Deliberately separated from the shared `RtmgConfig` fields
 *  so the config object stays portable across frontends (Ableton M4L,
 *  the VST, MCP / headless): those consume only the shared fields and
 *  keep their own equivalent of this block. On the wire / in config.json
 *  this lives under the top-level `web` key.
 *
 *  Note `channel_ranges` is NOT here — per-channel {min,max,reverse} is
 *  part of the shared *control surface* a native plugin's knob layout
 *  would also honor, so it lives on `RtmgConfig` directly. */
export interface RtmgWebConfig {
  effects: RtmgConfigEffects;
  audio: RtmgConfigAudio;
  reset_seconds: number;
  denoise_session_gate: RtmgConfigDenoiseSessionGate;
  /** Swapping to a new song restarts playback from frame 0. When false,
   * the worklet keeps its current phase across the swap, so a swap at
   * 1:30 into a 4:00 track starts the new track at 1:30. The
   * ScriptProcessor fallback already restarts on swap; this aligns the
   * worklet path with that behavior and makes it operator-tunable. */
  restart_song_on_swap: boolean;
}

export interface RtmgConfig {
  /** Shared-config schema version. Bump on a breaking shape change so
   *  consumers across frontends can detect mismatches. Old config.json
   *  files that predate the field are normalised to 1 on load. */
  version: number;
  engine: RtmgConfigEngine;
  prompts: RtmgConfigPrompts;
  controls: RtmgConfigControls;
  /** Per-channel {min,max,reverse} for the channel-gain controls. Part
   *  of the shared control surface (a VST / M4L knob layout would honor
   *  the same ranges), so it sits alongside `controls` rather than in
   *  the web-only block. */
  channel_ranges: RtmgConfigChannelRanges;
  seed: number;
  /** Default inference source for uploaded-track swaps. Built-in
   *  fixtures keep using their full source unless the track is present
   *  in the custom upload store. Engine-facing (selects the audio sent
   *  for inference), so it stays in the shared config rather than `web`. */
  swap_source_mode: SwapSourceMode;
  /** The browser demo's own presentation/playback/interaction settings.
   *  Other frontends ignore this block. */
  web: RtmgWebConfig;
  /** Per-param schedule curves. Same shape useCurveStore persists to
   *  localStorage today, lifted into the operator-editable config so a
   *  pod's deployed sound can ship its automation alongside its
   *  sliders + prompts. Optional — absent = stock pods fall back to
   *  the store's localStorage hydration / defaultCurveState. */
  curves?: RtmgConfigCurves;
  /** Playback loop region + enabled + snap resolution. Optional — older
   *  exports and stock pods omit it; absent on import leaves the live
   *  loop untouched. */
  loop?: RtmgConfigLoop;
}
