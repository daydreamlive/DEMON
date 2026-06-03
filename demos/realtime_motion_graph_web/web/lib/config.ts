"use client";

import { useEffect, useState } from "react";

import { isLoopGridRes, type LoopGridRes } from "@/lib/loopGrid";
import { useCurveStore } from "@/store/useCurveStore";
import { useLoraStore } from "@/store/useLoraStore";
import { usePerformanceStore } from "@/store/usePerformanceStore";
import { useSessionStore } from "@/store/useSessionStore";
import {
  DCW_MODES,
  DCW_WAVELETS,
  DEFAULT_TIME_SIGNATURE,
  isRcfgMode,
  isTimeSignature,
  type DcwMode,
  type DcwWavelet,
} from "@/types/engine";

export type SwapSourceMode = "full" | "vocals" | "instruments";

// Operator-editable startup config. Mirrors the static app's
// static/config.json (now lost in the React port). Lives at
// web/public/config.json so an installer can edit and refresh without a
// rebuild — the Next.js dev/prod server serves it as-is, no caching.
//
// Boot order:
//   1. RTMGBoot (module load) calls loadConfig() → applyConfig().
//   2. applyConfig() pushes values into the relevant zustand stores
//      (perf, lora) and notifies non-store subscribers (effects renderer,
//      PerformanceShell's useConfig()).
//   3. Stores already initialize with hardcoded defaults that match
//      DEFAULT_CONFIG below, so first paint is correct even if the fetch
//      is still in flight.

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

export interface RtmgConfig {
  engine: RtmgConfigEngine;
  prompts: RtmgConfigPrompts;
  controls: RtmgConfigControls;
  channel_ranges: RtmgConfigChannelRanges;
  seed: number;
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
  /** Default inference source for uploaded-track swaps. Built-in
   *  fixtures keep using their full source unless the track is present
   *  in the custom upload store. */
  swap_source_mode: SwapSourceMode;
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

export const DEFAULT_CONFIG: RtmgConfig = {
  engine: {
    sde: false,
    lora: true,
    depth: 4,
    vae_window: 0.36,
    crop: 0,
    steps: 8,
    fast_vae: false,
    walk_window: false,
    walk_window_s: 60,
    lead_floor_s: 0.25,
    lead_ceiling_s: 1.35,
    lead_release_tau_s: 1.5,
    max_source_duration_s: 120,
    key: "G# minor",
    time_signature: DEFAULT_TIME_SIGNATURE,
    enabled_loras: [],
    auto_prepend_lora_triggers: true,
    show_incompatible_loras: false,
  },
  prompts: {
    a: "heavy dubstep, deathstep, afxdump, growl heavy bass distortion",
    b: "daft punk style, beautiful, four to the floor, angelic",
    blend: 0.4,
  },
  controls: {
    denoise: 0.7,
    hint_strength: 1.0,
    feedback: 0.0,
    feedback_depth: 1,
    shift: 3.5,
    ch_g0: 1.0,
    ch_g1: 1.0,
    ch_g2: 1.0,
    ch_g3: 1.0,
    ch_g4: 1.0,
    ch_g5: 1.0,
    ch_g6: 1.0,
    ch_g7: 1.0,
    ch13: 1.0,
    ch14: 1.0,
    ch19: 1.0,
    ch23: 1.0,
    ch29: 1.0,
    ch56: 1.0,
    dcw_scaler: 0.05,
    dcw_high_scaler: 0.02,
    dcw_enabled: true,
    dcw_mode: "double",
    dcw_wavelet: "haar",
    lora_default_strength: 1.4,
    guidance_scale: 7.0,
    cfg_rescale: 0.0,
    rcfg_mode: "off",
  },
  channel_ranges: {
    ch_g0: { min: 0, max: 2.2, reverse: false },
    ch_g1: { min: 0, max: 2.0, reverse: false },
    ch_g2: { min: 0, max: 2.3, reverse: true },
    ch_g3: { min: 0, max: 2.0, reverse: false },
    ch_g4: { min: 0, max: 2.5, reverse: false },
    ch_g5: { min: 0, max: 2.0, reverse: false },
    ch_g6: { min: 0, max: 2.0, reverse: true },
    ch_g7: { min: 0, max: 2.0, reverse: true },
    ch13: { min: 0, max: 2.0, reverse: true },
    ch14: { min: 0, max: 2.3, reverse: false },
    ch19: { min: 0, max: 2.5, reverse: false },
    ch23: { min: 0, max: 2.45, reverse: false },
    ch29: { min: 0, max: 2.0, reverse: false },
    ch56: { min: 0, max: 2.0, reverse: false },
  },
  seed: 0,
  effects: {
    parallax_strength: 0.4,
    bloom_on_kick: 0.3,
    bloom_threshold: 0.15,
    warp_strength: 0.4,
  },
  audio: {
    lufs_enabled: false,
    lufs_window_sec: 3.0,
    lufs_metric: "lufs",
    lufs_peak_headroom: 4.0,
    lufs_silence_floor_db: 30.0,
    lufs_silence_floor_hysteresis_db: 6.0,
  },
  reset_seconds: 0,
  denoise_session_gate: {
    enabled: true,
    glide_ms: 700,
  },
  restart_song_on_swap: true,
  swap_source_mode: "instruments",
};

let _activeConfig: RtmgConfig = DEFAULT_CONFIG;
let _configApplied = false;
const listeners = new Set<(c: RtmgConfig) => void>();

/** Snapshot of the active config. Read at point-of-use by code paths
 * that don't need re-render reactivity (e.g. useStartSession.buildConfig
 * runs once per Play click). For reactive reads, use useConfig(). */
export function getConfig(): RtmgConfig {
  return _activeConfig;
}

export function isSwapSourceMode(v: unknown): v is SwapSourceMode {
  return v === "full" || v === "vocals" || v === "instruments";
}

export function defaultSwapSourceMode(): SwapSourceMode {
  return isSwapSourceMode(_activeConfig.swap_source_mode)
    ? _activeConfig.swap_source_mode
    : DEFAULT_CONFIG.swap_source_mode;
}

/** Resolve the LoRA cap for a given source duration. Tiers (when
 *  present) take precedence: pick the smallest ``up_to_s`` that's ≥
 *  ``durationS``. When no tier matches (durationS larger than all
 *  thresholds, or tiers absent), fall back to
 *  ``engine.max_concurrent_loras``. ``null`` return = uncapped.
 *
 *  Passing ``durationS = 0`` (no source loaded yet) selects the most
 *  permissive tier — short-source assumptions hold at boot before the
 *  first session config arrives. Callers that want a conservative
 *  boot-time cap can pass the static fallback value directly. */
export function resolveLoraCapForSource(
  durationS: number,
  engine: Pick<
    RtmgConfigEngine,
    "max_concurrent_loras" | "max_concurrent_loras_tiers"
  > = _activeConfig.engine,
): number | null {
  const tiers = engine.max_concurrent_loras_tiers;
  if (tiers && tiers.length > 0) {
    // Sort by threshold ascending; pick the first tier whose ceiling
    // is ≥ durationS. Defensive sort so config-side order doesn't
    // matter to the runtime.
    const sorted = [...tiers]
      .filter((t) => typeof t?.up_to_s === "number" && typeof t?.cap === "number")
      .sort((a, b) => a.up_to_s - b.up_to_s);
    for (const tier of sorted) {
      if (durationS <= tier.up_to_s) return tier.cap;
    }
    // durationS exceeds every tier ceiling — fall through to the
    // static fallback. The fallback is intentionally separate from
    // the last-tier cap so an operator can express "anything past
    // 240s is uncapped" or "anything past 240s is fully blocked"
    // depending on which fallback they set.
  }
  const fallback = engine.max_concurrent_loras;
  return typeof fallback === "number" && fallback >= 0 ? fallback : null;
}

/** Apply a freshly-resolved cap to the LoRA store AND tell the server
 *  about any LoRAs the cap kicks off the enabled list.
 *
 *  ``setMaxEnabled`` alone is purely a client-store mutation — it
 *  clips ``enabled`` down to the new cap (oldest insertion order
 *  wins, newest are dropped). But the SERVER is unaware of the
 *  clip: those dropped LoRAs stay materialized in GPU memory (~1.2
 *  GiB each), invisible to the user, eating the very budget the
 *  smaller cap was trying to free. ``ghost LoRAs.``
 *
 *  This helper composes the two correctly:
 *   1. Snapshot the current enabled set.
 *   2. Diff against the post-clip view to identify the dropped ids.
 *   3. For each dropped id: ``remote.sendDisableLora(id)`` so the
 *      engine actually frees the refit-state buffer.
 *   4. Re-send the prompt so the trigger prefix drops the now-
 *      disabled LoRAs' triggers (useLoraTriggerSync debounce-sends
 *      automatically when ``enabled`` mutates, but we issue an
 *      immediate send here so the prompt and the disables hit the
 *      server in the same logical step).
 *   5. Finally call ``setMaxEnabled`` to clip the store.
 *
 *  When ``remote`` is null (boot path before any session), skips the
 *  WS sends — no server to notify. The store-side clip still applies. */
export function applyLoraCapWithServerSync(cap: number | null): void {
  const lora = useLoraStore.getState();
  const before = lora.enabled;
  const remote = useSessionStore.getState().remote;

  // Match useLoraStore.clipEnabledToCap semantics: drop the
  // most-recently-added entries (everything past index ``cap``).
  if (
    remote &&
    typeof cap === "number" &&
    cap >= 0 &&
    before.size > cap
  ) {
    const ids = Array.from(before);
    const toDrop = ids.slice(cap);
    for (const id of toDrop) {
      remote.sendDisableLora(id);
    }
    const perf = usePerformanceStore.getState();
    remote.sendPrompt(
      perf.promptA,
      perf.activeKey,
      perf.activeTimeSignature,
      perf.promptB,
    );
  }

  lora.setMaxEnabled(cap);
}

/** Whether applyConfig() has been called at least once. Once-per-session
 *  seed paths (useLoraStore.setCatalog → computeSeed) gate on this so a
 *  catalog fetch that beats the config fetch doesn't seed against
 *  DEFAULT_CONFIG. LibraryTile fires its own /api/loras in LOCAL_MODE
 *  at mount, racing RTMGBoot's parallel fetches; without this gate the
 *  loser-wins outcome is non-deterministic. */
export function isConfigApplied(): boolean {
  return _configApplied;
}

/** Subscribe to applyConfig() calls. Returns an unsubscribe. Used by
 * non-store consumers (the effects renderer in useRenderLoop) that need
 * to re-apply settings when the config arrives async after their mount. */
export function subscribeConfig(fn: (c: RtmgConfig) => void): () => void {
  listeners.add(fn);
  return () => {
    listeners.delete(fn);
  };
}

/** React hook variant — subscribes the calling component to config
 * changes so it re-renders when applyConfig() fires. */
export function useConfig(): RtmgConfig {
  const [c, setC] = useState(_activeConfig);
  useEffect(() => subscribeConfig(setC), []);
  return c;
}

export function mergeConfig(
  base: RtmgConfig,
  override: Partial<RtmgConfig> | null | undefined,
): RtmgConfig {
  if (!override) return base;
  return {
    engine: { ...base.engine, ...(override.engine ?? {}) },
    prompts: { ...base.prompts, ...(override.prompts ?? {}) },
    controls: { ...base.controls, ...(override.controls ?? {}) },
    // Per-param shallow merge: an override entry replaces the matching
    // base entry whole (operator-supplied {min,max,reverse} must travel
    // together to be coherent). Unspecified params keep the bundled
    // default range.
    channel_ranges: {
      ...base.channel_ranges,
      ...(override.channel_ranges ?? {}),
    },
    seed: typeof override.seed === "number" ? override.seed : base.seed,
    effects: { ...base.effects, ...(override.effects ?? {}) },
    audio: { ...base.audio, ...(override.audio ?? {}) },
    reset_seconds:
      typeof override.reset_seconds === "number"
        ? override.reset_seconds
        : base.reset_seconds,
    denoise_session_gate: {
      ...base.denoise_session_gate,
      ...(override.denoise_session_gate ?? {}),
    },
    restart_song_on_swap:
      typeof override.restart_song_on_swap === "boolean"
        ? override.restart_song_on_swap
        : base.restart_song_on_swap,
    swap_source_mode: isSwapSourceMode(override.swap_source_mode)
      ? override.swap_source_mode
      : base.swap_source_mode,
    // Curves are operator-authored and only meaningful as a whole bag,
    // so the override entry replaces the base entry whole when present.
    // Absent override keeps whatever the base has (DEFAULT_CONFIG leaves
    // this undefined; stock pods fall through to localStorage hydration).
    ...(override.curves !== undefined ? { curves: override.curves } : (base.curves !== undefined ? { curves: base.curves } : {})),
    // Loop region replaces whole when the import carries one; otherwise
    // keep whatever the base (the live config) holds.
    ...(override.loop !== undefined ? { loop: override.loop } : (base.loop !== undefined ? { loop: base.loop } : {})),
  };
}

/** Lookup the active range for `param`, or null if no override is
 *  configured. Reads from the latest applied config — safe to call
 *  outside React. Consumers that need reactivity should read
 *  `useConfig().channel_ranges` instead. */
export function getChannelRange(param: string): RtmgChannelRange | null {
  return _activeConfig.channel_ranges[param] ?? null;
}

/** Fetch /config.json (no cache). Missing file or parse error → defaults
 * silently — the bundled defaults already match the React port's current
 * behavior, so a deploy without a config.json works unchanged. */
export async function loadConfig(): Promise<RtmgConfig> {
  try {
    const res = await fetch(`/config.json?t=${Date.now()}`, {
      cache: "no-store",
    });
    if (!res.ok) return DEFAULT_CONFIG;
    const json = (await res.json()) as Partial<RtmgConfig>;
    return mergeConfig(DEFAULT_CONFIG, json);
  } catch {
    return DEFAULT_CONFIG;
  }
}

function isDcwMode(v: unknown): v is DcwMode {
  return typeof v === "string" && (DCW_MODES as readonly string[]).includes(v);
}
function isDcwWavelet(v: unknown): v is DcwWavelet {
  return typeof v === "string" && (DCW_WAVELETS as readonly string[]).includes(v);
}

/** Collapse the dual-variant config into a single applied config, picking
 *  the XL (5B) sibling when the active checkpoint scale is "5B" and the
 *  sibling is defined. Any other scale (null, "2B", unknown) keeps the
 *  base values. Unspecified `_xl` siblings always fall through to base,
 *  so existing single-variant config.json files keep working unchanged.
 *
 *  Result keeps the `_xl` fields on the engine/prompts objects so an
 *  Import round-trip (mergeConfig over getConfig() then applyConfig)
 *  doesn't lose them. */
export function selectVariant(cfg: RtmgConfig, scale: string | null): RtmgConfig {
  if (scale !== "5B") return cfg;
  const e = cfg.engine;
  const p = cfg.prompts;
  return {
    ...cfg,
    engine: {
      ...e,
      depth: e.depth_xl ?? e.depth,
      enabled_loras: e.enabled_loras_xl ?? e.enabled_loras,
    },
    prompts: {
      ...p,
      a: p.a_xl ?? p.a,
      b: p.b_xl ?? p.b,
      blend: typeof p.blend_xl === "number" ? p.blend_xl : p.blend,
    },
  };
}

/** Push the supplied config into stores + non-store subscribers. Idempotent;
 * safe to call multiple times. The only mid-session callers today are the
 * boot path; future "Reload config" affordances would call this too.
 *
 * Resolves the XL variant in-place using the current checkpoint scale —
 * RTMGBoot awaits /api/loras before this runs at boot, so the scale is
 * already known the first time we land here. Mid-session re-applies
 * (Import) read whatever scale is currently set in useSessionStore. */
export function applyConfig(c: RtmgConfig): void {
  const scale = useSessionStore.getState().checkpointScale;
  const resolved = selectVariant(c, scale);
  const firstApply = !_configApplied;
  _activeConfig = resolved;
  _configApplied = true;

  // Numeric controls land on sliderValues + sliderTargets so the slider
  // UI and the param-sync tick agree. prompt_blend rides in here too —
  // it lives in the slider system alongside lora_blend.
  const sliderUpdates: Record<string, number> = {};
  for (const [k, v] of Object.entries(resolved.controls)) {
    if (typeof v === "number") sliderUpdates[k] = v;
  }
  sliderUpdates.prompt_blend = resolved.prompts.blend;

  usePerformanceStore.setState((s) => ({
    sliderDefaults: { ...s.sliderDefaults, ...sliderUpdates },
    sliderValues: { ...s.sliderValues, ...sliderUpdates },
    sliderTargets: { ...s.sliderTargets, ...sliderUpdates },
    promptA: resolved.prompts.a,
    promptB: resolved.prompts.b,
    activeKey: resolved.engine.key,
    activeTimeSignature: isTimeSignature(resolved.engine.time_signature)
      ? resolved.engine.time_signature
      : DEFAULT_TIME_SIGNATURE,
    seed: resolved.seed,
    dcwEnabled:
      typeof resolved.controls.dcw_enabled === "boolean"
        ? resolved.controls.dcw_enabled
        : s.dcwEnabled,
    dcwMode: isDcwMode(resolved.controls.dcw_mode) ? resolved.controls.dcw_mode : s.dcwMode,
    dcwWavelet: isDcwWavelet(resolved.controls.dcw_wavelet)
      ? resolved.controls.dcw_wavelet
      : s.dcwWavelet,
    rcfgMode: isRcfgMode(resolved.controls.rcfg_mode) ? resolved.controls.rcfg_mode : s.rcfgMode,
    lufsOn: resolved.audio.lufs_enabled,
  }));

  // Loop region: when the config carries one, push it into the perf
  // store. WaveformScrubBox subscribes, so a mounted editor re-syncs the
  // worklet + server via its own effect; an import before a session just
  // seeds the store and applies once the waveform mounts. Validated
  // because the JSON is operator-editable: a malformed band → no loop.
  if (resolved.loop) {
    const lp = resolved.loop;
    const validBand =
      lp.band &&
      typeof lp.band.start === "number" &&
      typeof lp.band.end === "number" &&
      lp.band.end > lp.band.start
        ? { start: lp.band.start, end: lp.band.end }
        : null;
    const next: Partial<{
      loopBand: { start: number; end: number } | null;
      bandLoopEnabled: boolean;
      loopGridRes: LoopGridRes;
      loopOn: boolean;
    }> = {
      loopBand: validBand,
      bandLoopEnabled: typeof lp.enabled === "boolean" ? lp.enabled : true,
      loopGridRes: isLoopGridRes(lp.grid) ? lp.grid : "beat",
    };
    // Only touch the global full-buffer loop when the export carried it,
    // so an older config (or hand-edited JSON) leaves the live setting be.
    if (typeof lp.fullBuffer === "boolean") next.loopOn = lp.fullBuffer;
    usePerformanceStore.setState(next);
  }

  // Curves: when the config carries them, push the whole bag into
  // useCurveStore via setState (the store has no batch action). Skipped
  // when the field is absent — stock pods fall through to the store's
  // own hydratePersistedCurves localStorage path. Deep-clone the
  // points so later edits in the store don't mutate the active
  // config snapshot.
  if (resolved.curves) {
    useCurveStore.setState({
      scheduleEnabled: resolved.curves.scheduleEnabled,
      curves: Object.fromEntries(
        Object.entries(resolved.curves.curves).map(([param, curve]) => [
          param,
          {
            enabled: curve.enabled,
            points: curve.points.map((p) => ({ x: p.x, y: p.y, mode: p.mode })),
          },
        ]),
      ),
    });
  }

  // LoRA enable/strength state.
  //
  // First applyConfig (boot): if a catalog landed before us (LibraryTile's
  // mount-time /api/loras winning the race against /config.json), the
  // store stashed the catalog but skipped seeding. Re-trigger setCatalog
  // so its once-per-session gate runs against the real enabled_loras.
  //
  // Later applyConfig (an imported config): the store is already seeded,
  // so setCatalog's gate would ignore the new enabled_loras. reset()
  // re-seeds enabled+strengths from the fresh config. The LoRA UI
  // normally sends enable/disable to the engine on click — an import
  // bypasses that path, so push the diff to the engine here and
  // re-encode the prompt so the trigger prefix matches.
  const lora = useLoraStore.getState();
  // Push the boot-time cap. We don't yet know the source duration so
  // resolve against 0 — selects the most-permissive tier. Once a
  // session starts (useStartSession) or a source swap completes
  // (useFixtureSwap), the cap is recomputed against the actual
  // duration via ``resolveLoraCapForSource``. Static
  // ``max_concurrent_loras`` (no tiers) is duration-independent so
  // the boot value persists.
  lora.setMaxEnabled(resolveLoraCapForSource(0, resolved.engine));
  if (firstApply) {
    if (!lora.seeded && lora.catalog.length > 0) {
      lora.setCatalog(lora.catalog);
    }
  } else if (lora.catalog.length > 0) {
    const before = new Set(lora.enabled);
    // setMaxEnabled above already re-clipped any over-cap entries from
    // the prior session; reset() now re-seeds against the new config.
    lora.reset();
    const after = useLoraStore.getState();
    const remote = useSessionStore.getState().remote;
    if (remote) {
      for (const id of before) {
        if (!after.enabled.has(id)) remote.sendDisableLora(id);
      }
      for (const id of after.enabled) {
        if (!before.has(id)) {
          remote.sendEnableLora(id, after.strengths[id] ?? 0);
        }
      }
      remote.sendPrompt(
        resolved.prompts.a,
        resolved.engine.key,
        isTimeSignature(resolved.engine.time_signature)
          ? resolved.engine.time_signature
          : DEFAULT_TIME_SIGNATURE,
        resolved.prompts.b,
      );
    }
  }

  for (const fn of listeners) fn(resolved);
}

/**
 * Snapshot the live stores into an `RtmgConfig` — the inverse of
 * `applyConfig`. Used by the OperatorStrip's Export button and by any
 * caller (demon-public-demo's `captureSessionState`) that wants the
 * DEMON-shaped base of a session without rebuilding the field-mapping
 * logic.
 *
 * Fields the stores don't own (channel_ranges, effects, audio
 * defaults, denoise_session_gate, restart_song_on_swap, swap_source_mode, the
 * non-numeric engine.* config) are pulled from the active config so
 * exports round-trip cleanly through Import.
 */
export function captureRtmgConfig(): RtmgConfig {
  const perf = usePerformanceStore.getState();
  const lora = useLoraStore.getState();
  const curveStore = useCurveStore.getState();
  const active = _activeConfig;

  // Numeric controls land on sliderTargets in the perf store. The DCW
  // non-numeric controls live on dedicated store fields.
  const controls: RtmgConfigControls = { ...perf.sliderTargets };
  controls.dcw_enabled = perf.dcwEnabled;
  controls.dcw_mode = perf.dcwMode;
  controls.dcw_wavelet = perf.dcwWavelet;
  // lora_default_strength isn't tracked live in the perf store; pull
  // from active config so the export reflects the seed value.
  if (typeof active.controls.lora_default_strength !== "undefined") {
    controls.lora_default_strength = active.controls.lora_default_strength;
  }

  return {
    engine: {
      ...active.engine,
      key: perf.activeKey,
      time_signature: perf.activeTimeSignature ?? active.engine.time_signature,
      enabled_loras: Array.from(lora.enabled).map((id) => {
        const strength = lora.strengths[id];
        return typeof strength === "number"
          ? { name: id, strength }
          : id;
      }),
    },
    prompts: {
      a: perf.promptA,
      b: perf.promptB,
      blend: perf.sliderTargets.prompt_blend ?? 0,
    },
    controls,
    channel_ranges: active.channel_ranges,
    seed: perf.seed,
    effects: active.effects,
    audio: { ...active.audio, lufs_enabled: perf.lufsOn },
    reset_seconds: active.reset_seconds,
    denoise_session_gate: active.denoise_session_gate,
    restart_song_on_swap: active.restart_song_on_swap,
    swap_source_mode: active.swap_source_mode,
    loop: {
      band: perf.loopBand,
      enabled: perf.bandLoopEnabled,
      grid: perf.loopGridRes,
      fullBuffer: perf.loopOn,
    },
    curves: {
      scheduleEnabled: curveStore.scheduleEnabled,
      curves: Object.fromEntries(
        Object.entries(curveStore.curves).map(([param, curve]) => [
          param,
          {
            enabled: curve.enabled,
            points: curve.points.map((p) => ({ x: p.x, y: p.y, mode: p.mode })),
          },
        ]),
      ),
    },
  };
}
