"use client";

import { useEffect, useState } from "react";

import { usePerformanceStore } from "@/store/usePerformanceStore";
import {
  DCW_MODES,
  DCW_WAVELETS,
  DEFAULT_TIME_SIGNATURE,
  isTimeSignature,
  type DcwMode,
  type DcwWavelet,
} from "@/types/engine";

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
  key: string;
  /** Default meter numerator the operator dropdown starts on. Mirrors
   * `key` in posture: the server's session-init resolver still wins on
   * sidecar hits, so this is purely a UI seed for the manual "Override"
   * control. Allowed values: "2" | "3" | "4" | "6". */
  time_signature: string;
  /** Filename stems to auto-enable on first catalog load. Empty falls
   * back to the count-rule in useLoraStore (first two from the sorted
   * catalog, with name-fallback per slot). */
  enabled_loras: string[];
}

export interface RtmgConfigPrompts {
  a: string;
  b: string;
  blend: number;
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
}

export const DEFAULT_CONFIG: RtmgConfig = {
  engine: {
    sde: false,
    lora: true,
    depth: 4,
    vae_window: 6,
    crop: 0,
    steps: 8,
    fast_vae: false,
    walk_window: false,
    walk_window_s: 60,
    key: "G# minor",
    time_signature: DEFAULT_TIME_SIGNATURE,
    enabled_loras: [],
  },
  prompts: {
    a: "heavy dubstep, deathstep, afxdump, growl heavy bass distortion",
    b: "daft punk style, beautiful, four to the floor, angelic",
    blend: 0.4,
  },
  controls: {
    denoise: 0.7,
    hint_strength: 1.4,
    feedback: 0.0,
    shift: 0.5,
    noise_share: 0.0,
    ode_noise: 0.0,
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
};

let _activeConfig: RtmgConfig = DEFAULT_CONFIG;
const listeners = new Set<(c: RtmgConfig) => void>();

/** Snapshot of the active config. Read at point-of-use by code paths
 * that don't need re-render reactivity (e.g. useStartSession.buildConfig
 * runs once per Play click). For reactive reads, use useConfig(). */
export function getConfig(): RtmgConfig {
  return _activeConfig;
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

function mergeConfig(
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

/** Push the supplied config into stores + non-store subscribers. Idempotent;
 * safe to call multiple times. The only mid-session callers today are the
 * boot path; future "Reload config" affordances would call this too. */
export function applyConfig(c: RtmgConfig): void {
  _activeConfig = c;

  // Numeric controls land on sliderValues + sliderTargets so the slider
  // UI and the param-sync tick agree.
  const sliderUpdates: Record<string, number> = {};
  for (const [k, v] of Object.entries(c.controls)) {
    if (typeof v === "number") sliderUpdates[k] = v;
  }

  usePerformanceStore.setState((s) => ({
    sliderValues: { ...s.sliderValues, ...sliderUpdates },
    sliderTargets: { ...s.sliderTargets, ...sliderUpdates },
    promptA: c.prompts.a,
    promptB: c.prompts.b,
    blend: c.prompts.blend,
    activeKey: c.engine.key,
    activeTimeSignature: isTimeSignature(c.engine.time_signature)
      ? c.engine.time_signature
      : DEFAULT_TIME_SIGNATURE,
    seed: c.seed,
    dcwEnabled:
      typeof c.controls.dcw_enabled === "boolean"
        ? c.controls.dcw_enabled
        : s.dcwEnabled,
    dcwMode: isDcwMode(c.controls.dcw_mode) ? c.controls.dcw_mode : s.dcwMode,
    dcwWavelet: isDcwWavelet(c.controls.dcw_wavelet)
      ? c.controls.dcw_wavelet
      : s.dcwWavelet,
    lufsOn: c.audio.lufs_enabled,
  }));

  for (const fn of listeners) fn(c);
}
