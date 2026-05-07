"use client";

import { create } from "zustand";

import {
  SLIDER_META,
  type DcwMode,
  type DcwWavelet,
  type DisplayMode,
} from "@/types/engine";

// Top-level performance state. Mirrors app.js's module-level vars
// (sliderValues, seedValue, blendValue, activeKey, prompts, fixture, mode,
// kiosk). LoRA strength sliders live in useLoraStore.

const SHOW_KBD_HINTS_STORAGE_KEY = "demon:showKbdHints";
const SMOOTH_STORAGE_KEY = "demon:smooth";
const SMOOTH_MS_STORAGE_KEY = "demon:smoothMs";

const DEFAULT_SMOOTH_MS = 1000;
const MIN_SMOOTH_MS = 50;
const MAX_SMOOTH_MS = 10000;

function loadBool(key: string, fallback: boolean): boolean {
  if (typeof localStorage === "undefined") return fallback;
  try {
    const v = localStorage.getItem(key);
    if (v === null) return fallback;
    return v === "1";
  } catch {
    return fallback;
  }
}

function saveBool(key: string, b: boolean): void {
  if (typeof localStorage === "undefined") return;
  try {
    localStorage.setItem(key, b ? "1" : "0");
  } catch {}
}

function loadNum(key: string, fallback: number): number {
  if (typeof localStorage === "undefined") return fallback;
  try {
    const v = localStorage.getItem(key);
    if (v === null) return fallback;
    const n = parseInt(v, 10);
    return Number.isFinite(n) ? n : fallback;
  } catch {
    return fallback;
  }
}

function saveNum(key: string, n: number): void {
  if (typeof localStorage === "undefined") return;
  try {
    localStorage.setItem(key, String(n));
  } catch {}
}

// ── Slider tween manager ────────────────────────────────────────────────
// Two-tier model: `sliderTargets` is the user's intent (updated instantly
// on drag / arrow key / MIDI bump); `sliderValues` is what we actually
// send to the engine. When `smooth` is off the two are identical; when on,
// `sliderValues` chases `sliderTargets` along a cubic ease-out tween.
//
// Slider UIs read `sliderTargets` so dragging feels immediate. The
// param-sync tick (hooks/useParamSync.ts, 33 ms) reads `sliderValues`
// so the engine sees the smoothed curve. MIDI knobs hit bumpSlider
// repeatedly; each bump retargets and the tween chases without stutter.

interface Tween {
  start: number;
  startTime: number;
}
const tweens = new Map<string, Tween>();
let rafId: number | null = null;

function tickTweens(): void {
  rafId = null;
  const state = usePerformanceStore.getState();
  if (tweens.size === 0) return;
  const now = performance.now();
  const durationMs = state.smoothMs;
  const updates: Record<string, number> = {};
  for (const [param, t] of tweens) {
    const target = state.sliderTargets[param] ?? 0;
    const elapsed = now - t.startTime;
    if (elapsed >= durationMs) {
      updates[param] = target;
      tweens.delete(param);
      continue;
    }
    const k = elapsed / durationMs;
    // Cubic ease-out: snappy start, settles into target.
    const eased = 1 - Math.pow(1 - k, 3);
    updates[param] = t.start + (target - t.start) * eased;
  }
  if (Object.keys(updates).length > 0) {
    usePerformanceStore.setState((s) => ({
      sliderValues: { ...s.sliderValues, ...updates },
    }));
  }
  if (tweens.size > 0 && typeof requestAnimationFrame !== "undefined") {
    rafId = requestAnimationFrame(tickTweens);
  }
}

/** Start (or restart) a tween for `param`. The tween's "start" is whatever
 *  sliderValues[param] is right now; the "end" is read from sliderTargets
 *  on every tick so retargets mid-flight (MIDI knob spinning) just bend
 *  the curve toward the new target. */
function ensureTween(param: string): void {
  const sliderValues = usePerformanceStore.getState().sliderValues;
  tweens.set(param, {
    start: sliderValues[param] ?? 0,
    startTime: performance.now(),
  });
  if (rafId === null && typeof requestAnimationFrame !== "undefined") {
    rafId = requestAnimationFrame(tickTweens);
  }
}

function cancelTween(param: string): void {
  tweens.delete(param);
}

// ── Manual-override timestamps ─────────────────────────────────────────
// When a user drags a slider OR bumps it via MIDI / arrow key while
// useScheduledCurves is also driving that param, we let the manual
// touch win for a short window. After that window, the curve resumes.
// The timestamps live in a module-level Map (not zustand state) so
// tracking them doesn't trigger re-renders. Read by useScheduledCurves.
const MANUAL_OVERRIDE_WINDOW_MS = 500;
const manualTouchedAt = new Map<string, number>();

function stampManualTouch(param: string): void {
  manualTouchedAt.set(param, performance.now());
}

/** True if `param` was manually touched within MANUAL_OVERRIDE_WINDOW_MS. */
export function isManualOverrideActive(param: string): boolean {
  const ts = manualTouchedAt.get(param);
  if (!ts) return false;
  return performance.now() - ts < MANUAL_OVERRIDE_WINDOW_MS;
}

const DEFAULT_SLIDER_VALUES: Record<string, number> = {
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
};

interface PerformanceState {
  /** What we're actually sending to the engine. When `smooth` is on, this
   *  trails `sliderTargets` along a tween; otherwise it equals it. Read by
   *  param-sync (hooks/useParamSync.ts) and the render loop (the audio-
   *  reactive graph in hooks/useRenderLoop.ts). */
  sliderValues: Record<string, number>;
  /** What the user *intends* — instant target. Slider UIs (SliderGroup,
   *  MobileRemixRail, DesktopEdgeDrag) read this so the visual position
   *  follows the cursor / MIDI knob without the smoothing lag. */
  sliderTargets: Record<string, number>;
  /** Random seed in 0..1; "dice" button reroll. */
  seed: number;
  /** Prompt A/B blend (0 = A, 1 = B). */
  blend: number;
  /** Two prompts. */
  promptA: string;
  promptB: string;
  /** Currently active key (e.g. "G# minor"). May come from auto-detect. */
  activeKey: string;
  /** Selected fixture name (from /api/fixtures). */
  fixture: string;
  /** Detected musical metadata from server's "ready" frame. */
  detectedBpm: number | null;
  detectedKey: string | null;
  /** Display mode toggle. */
  mode: DisplayMode;
  /** Kiosk mode (auto-hide cursor + idle reset). */
  kiosk: boolean;
  /** Pause flag (audio context state). */
  paused: boolean;

  /** DCW (wavelet-domain post-step correction) non-numeric state. The two
   * numeric knobs (dcw_scaler, dcw_high_scaler) live in sliderValues. */
  dcwEnabled: boolean;
  dcwMode: DcwMode;
  dcwWavelet: DcwWavelet;
  /** Show keyboard-shortcut hints under each slider / next to buttons.
   * Default true. Persisted to localStorage. */
  showKbdHints: boolean;
  /** Smooth slider transitions: when enabled, setSlider interpolates
   *  from the current value to the target over `smoothMs` instead of
   *  jumping. bumpSlider stays immediate (small deltas are already
   *  smooth). Persisted to localStorage. */
  smooth: boolean;
  smoothMs: number;

  // ── actions ───────────────────────────────────────────────────────────
  setSlider: (param: string, value: number) => void;
  /** Curve-driven slider write. Skips the smoothing tween (the curve
   *  IS the source of truth, not a target to chase), writes both
   *  sliderTargets and sliderValues synchronously. Does NOT stamp the
   *  manual-override timestamp — useScheduledCurves uses that to
   *  decide when manual drags should briefly win over the curve. */
  setSliderDirect: (param: string, value: number) => void;
  bumpSlider: (param: string, delta: number) => void;
  setSeed: (seed: number) => void;
  randomizeSeed: () => void;
  setBlend: (b: number) => void;
  setPromptA: (s: string) => void;
  setPromptB: (s: string) => void;
  setKey: (k: string) => void;
  setFixture: (name: string) => void;
  setDetected: (bpm: number | null, key: string | null) => void;
  setMode: (m: DisplayMode) => void;
  toggleMode: () => void;
  setKiosk: (k: boolean) => void;
  toggleKiosk: () => void;
  setPaused: (p: boolean) => void;
  togglePause: () => void;
  setDcwEnabled: (b: boolean) => void;
  toggleDcw: () => void;
  setDcwMode: (m: DcwMode) => void;
  setDcwWavelet: (w: DcwWavelet) => void;
  toggleKbdHints: () => void;
  toggleSmooth: () => void;
  setSmoothMs: (ms: number) => void;
  /** Read localStorage-backed prefs (showKbdHints) and
   *  apply them to the store. Called from a client-only useEffect so SSR
   *  always renders with the defaults — without this, hydration mismatches
   *  on OperatorStrip's COMPACT/STANDARD + KBD: ON/OFF buttons. */
  hydratePersistedPrefs: () => void;
  /** Reset every slider + seed + blend to defaults (idle-reset). */
  resetToDefaults: () => void;
}

function clampToMeta(param: string, value: number): number {
  const meta = SLIDER_META[param];
  // LoRA sliders (lora_str_<id>) aren't in SLIDER_META — clamp to [0, 2].
  const max = meta?.max ?? 2.0;
  if (Number.isNaN(value)) return 0;
  return Math.max(0, Math.min(max, value));
}

export const usePerformanceStore = create<PerformanceState>((set) => ({
  sliderValues: { ...DEFAULT_SLIDER_VALUES },
  sliderTargets: { ...DEFAULT_SLIDER_VALUES },
  seed: 0,
  blend: 0.4,
  promptA: "heavy dubstep, deathstep, afxdump, growl heavy bass distortion",
  promptB: "daft punk style, beautiful, four to the floor, angelic",
  activeKey: "G# minor",
  fixture: "",
  detectedBpm: null,
  detectedKey: null,
  mode: "graph",
  kiosk: false,
  paused: false,

  dcwEnabled: true,
  dcwMode: "double",
  dcwWavelet: "haar",

  // Hydrated from localStorage after mount via hydratePersistedPrefs() —
  // do NOT read localStorage here, that breaks SSR hydration.
  showKbdHints: true,
  smooth: false,
  smoothMs: DEFAULT_SMOOTH_MS,

  setSlider: (param, value) => {
    stampManualTouch(param);
    const target = clampToMeta(param, value);
    const state = usePerformanceStore.getState();
    if (!state.smooth || state.smoothMs <= 0) {
      cancelTween(param);
      set((s) => ({
        sliderValues: { ...s.sliderValues, [param]: target },
        sliderTargets: { ...s.sliderTargets, [param]: target },
      }));
      return;
    }
    // Smooth: write the target instantly (so the UI snaps to where the
    // user wants), kick off a tween that pulls sliderValues toward it.
    set((s) => ({
      sliderTargets: { ...s.sliderTargets, [param]: target },
    }));
    ensureTween(param);
  },
  setSliderDirect: (param, value) => {
    // Curve-driven write: bypass smoothing, do NOT stamp a manual-touch
    // timestamp. We want both fields to reflect the curve value
    // synchronously so the param-sync tick AND the slider UI both see
    // the new value on the next read.
    const target = clampToMeta(param, value);
    cancelTween(param);
    set((s) => ({
      sliderValues: { ...s.sliderValues, [param]: target },
      sliderTargets: { ...s.sliderTargets, [param]: target },
    }));
  },
  bumpSlider: (param, delta) => {
    stampManualTouch(param);
    const state = usePerformanceStore.getState();
    const newTarget = clampToMeta(
      param,
      (state.sliderTargets[param] ?? 0) + delta,
    );
    if (!state.smooth || state.smoothMs <= 0) {
      cancelTween(param);
      set((s) => ({
        sliderValues: { ...s.sliderValues, [param]: newTarget },
        sliderTargets: { ...s.sliderTargets, [param]: newTarget },
      }));
      return;
    }
    // Smoothed bump: retarget instantly, let the tween chase. Each MIDI
    // knob message just bends the in-flight curve toward the new target
    // — no stutter, no per-bump 1 s lag.
    set((s) => ({
      sliderTargets: { ...s.sliderTargets, [param]: newTarget },
    }));
    ensureTween(param);
  },
  setSeed: (seed) => set({ seed: Math.max(0, Math.min(1, seed)) }),
  randomizeSeed: () => set({ seed: Math.random() }),
  setBlend: (b) => set({ blend: Math.max(0, Math.min(1, b)) }),
  setPromptA: (s) => set({ promptA: s }),
  setPromptB: (s) => set({ promptB: s }),
  setKey: (k) => set({ activeKey: k }),
  setFixture: (name) => set({ fixture: name }),
  setDetected: (bpm, key) =>
    set((s) => ({
      detectedBpm: bpm,
      detectedKey: key,
      // If user hasn't manually overridden the key (still on default),
      // adopt the detection. Caller can re-set if needed.
      activeKey: key ?? s.activeKey,
    })),
  setMode: (m) => set({ mode: m }),
  toggleMode: () => set((s) => ({ mode: s.mode === "graph" ? "video" : "graph" })),
  setKiosk: (k) => set({ kiosk: k }),
  toggleKiosk: () => set((s) => ({ kiosk: !s.kiosk })),
  setPaused: (p) => set({ paused: p }),
  togglePause: () => set((s) => ({ paused: !s.paused })),
  setDcwEnabled: (b) => set({ dcwEnabled: b }),
  toggleDcw: () => set((s) => ({ dcwEnabled: !s.dcwEnabled })),
  setDcwMode: (m) => set({ dcwMode: m }),
  setDcwWavelet: (w) => set({ dcwWavelet: w }),
  toggleKbdHints: () =>
    set((s) => {
      const next = !s.showKbdHints;
      saveBool(SHOW_KBD_HINTS_STORAGE_KEY, next);
      return { showKbdHints: next };
    }),
  toggleSmooth: () =>
    set((s) => {
      const next = !s.smooth;
      // Smooth is intentionally NOT persisted — the user always starts a
      // fresh session with smoothing OFF, and opts in per-session.
      // Cancelling tweens on disable means in-flight transitions snap to
      // their target (current sliderValues).
      if (!next) tweens.clear();
      return { smooth: next };
    }),
  setSmoothMs: (ms) =>
    set(() => {
      const clamped = Math.max(MIN_SMOOTH_MS, Math.min(MAX_SMOOTH_MS, Math.round(ms)));
      saveNum(SMOOTH_MS_STORAGE_KEY, clamped);
      return { smoothMs: clamped };
    }),
  hydratePersistedPrefs: () =>
    set({
      showKbdHints: loadBool(SHOW_KBD_HINTS_STORAGE_KEY, true),
      // Smooth always defaults to off on page load (operators don't want
      // a stale ON from a previous session silently reshaping their
      // first slider drag). The duration knob still persists.
      smooth: false,
      smoothMs: Math.max(
        MIN_SMOOTH_MS,
        Math.min(MAX_SMOOTH_MS, loadNum(SMOOTH_MS_STORAGE_KEY, DEFAULT_SMOOTH_MS)),
      ),
    }),
  resetToDefaults: () => {
    tweens.clear();
    set(() => ({
      sliderValues: { ...DEFAULT_SLIDER_VALUES },
      sliderTargets: { ...DEFAULT_SLIDER_VALUES },
      seed: 0,
      blend: 0.4,
    }));
  },
}));

/** Compute the prompt string sent to the server given the current blend. */
export function computePromptTags(state: {
  promptA: string;
  promptB: string;
  blend: number;
}): string {
  const { promptA, promptB, blend } = state;
  if (blend <= 0) return promptA;
  if (blend >= 1) return promptB;
  // Server handles A/B blend by receiving both — but the simple wire is one
  // string. Match app.js's pattern: send "<A> | <B>" with the blend
  // parameter delivered separately via params.
  return promptA;
}
