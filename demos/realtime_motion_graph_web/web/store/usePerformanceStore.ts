"use client";

import { create } from "zustand";

import { frameScheduler } from "@/engine/scheduler/FrameScheduler";
import { getChannelRange } from "@/lib/config";
import { type LoopGridRes } from "@/lib/loopGrid";

import {
  DEFAULT_TIME_SIGNATURE,
  LORA_SLIDER_MAX,
  SLIDER_META,
  type DcwMode,
  type DcwWavelet,
  type DisplayMode,
  type RcfgMode,
  type TimeSignature,
} from "@/types/engine";

// Pre-seed defaults for every potential manual steering slot so reset /
// snapback / curve hooks never miss a key when the server allocates a
// new slot at runtime. Must match the prereg ceiling in engine.ts.
const MANUAL_SLOT_PREREG = 16;

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
// Active FrameScheduler registration handle, or null when no tweens are
// running. We register on first tween, unregister when the map empties —
// keeps the master rAF callback from doing dead work while idle.
let tweenUnregister: (() => void) | null = null;

// Tween outputs below this snap to 0 (when the target is 0). The
// cubic ease-out's tail can pass through arbitrarily small positive
// values in the last few percent of the glide, and the engine's
// schedule builder used to do `int(steps / denoise)` on whatever it
// received — a denoise of ~1e-17 lifted from a tween mid-flight made
// it request a 512-PiB linspace and OOM. Below this floor the slider
// is effectively zero anyway, so snap it.
const TWEEN_ZERO_FLOOR = 1e-6;

function tickTweens(now: number): void {
  if (tweens.size === 0) {
    // Defensive: scheduler keeps calling until we unregister, but we
    // already unregister at the end of any frame that drains the map.
    return;
  }
  const state = usePerformanceStore.getState();
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
    let v = t.start + (target - t.start) * eased;
    // Snap subnormal-ish positives to 0 when we're tweening down to 0
    // so the engine never sees a near-zero denoise (see floor comment).
    if (target === 0 && v > 0 && v < TWEEN_ZERO_FLOOR) v = 0;
    updates[param] = v;
  }
  if (Object.keys(updates).length > 0) {
    usePerformanceStore.setState((s) => ({
      sliderValues: { ...s.sliderValues, ...updates },
    }));
  }
  if (tweens.size === 0 && tweenUnregister) {
    tweenUnregister();
    tweenUnregister = null;
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
  if (!tweenUnregister) {
    tweenUnregister = frameScheduler.register("tweens", tickTweens, {
      phase: "compute",
      budgetMs: 1,
    });
  }
}

function cancelTween(param: string): void {
  tweens.delete(param);
  dropTweens.delete(param);
}

// ── Visual-only display tweens ─────────────────────────────────────────
// The "hear source first" gate plays a purely visual demo on each song
// load: the top-edge ribbon glides from its prior position down to 0 as
// a hint that the ribbon is a slider. The engine's denoise value snaps
// to 0 immediately at song load (so the user hears the source from
// frame 1) — this tween is a separate UI-only field that ribbons read
// in preference to sliderTargets. When the tween completes (or the
// user touches the slider), the override key is deleted and ribbons
// fall through to sliderTargets again. Cancelled implicitly by
// setSlider / cancelTween if the user grabs the ribbon mid-demo.
interface DropTween {
  start: number;
  startTime: number;
  target: number;
  durationMs: number;
}
const dropTweens = new Map<string, DropTween>();
let dropTweenUnregister: (() => void) | null = null;

function tickDropTweens(now: number): void {
  if (dropTweens.size === 0) return;
  const updates: Record<string, number> = {};
  const removals: string[] = [];
  for (const [param, t] of dropTweens) {
    const elapsed = now - t.startTime;
    if (elapsed >= t.durationMs) {
      // Demo finished: drop the override entry so the ribbon falls
      // through to sliderTargets[param] (already at the engine value).
      removals.push(param);
      dropTweens.delete(param);
      continue;
    }
    const k = elapsed / t.durationMs;
    // Same cubic ease-out as the regular tween — snappy start, soft
    // landing on zero so the ribbon doesn't slam into the rail edge.
    const eased = 1 - Math.pow(1 - k, 3);
    updates[param] = t.start + (t.target - t.start) * eased;
  }
  if (Object.keys(updates).length > 0 || removals.length > 0) {
    usePerformanceStore.setState((s) => {
      const next = { ...s.sliderDisplayOverride, ...updates };
      for (const p of removals) delete next[p];
      return { sliderDisplayOverride: next };
    });
  }
  if (dropTweens.size === 0 && dropTweenUnregister) {
    dropTweenUnregister();
    dropTweenUnregister = null;
  }
}

function ensureDropTweenRunning(): void {
  if (!dropTweenUnregister) {
    dropTweenUnregister = frameScheduler.register(
      "drop-tweens",
      tickDropTweens,
      { phase: "compute", budgetMs: 1 },
    );
  }
}

// ── Manual-override timestamps ─────────────────────────────────────────
// When a user drags a slider OR bumps it via MIDI / arrow key while
// useScheduledCurves is also driving that param, we let the manual
// touch win for a short window. After that window, the curve resumes.
// The timestamps live in a module-level Map (not zustand state) so
// tracking them doesn't trigger re-renders. Read by useScheduledCurves.
const MANUAL_OVERRIDE_WINDOW_MS = 500;
const manualTouchedAt = new Map<string, number>();

export function stampManualTouch(param: string): void {
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
  hint_strength: 1.0,
  timbre_strength: 1.0,
  lora_blend: 0.5,
  prompt_blend: 0.4,
  feedback: 0.0,
  feedback_depth: 1,
  shift: 3.0,
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
  steer_bright: 0.0,
  steer_warm: 0.0,
  steer_rough: 0.0,
  steer_density: 0.0,
  // Manual steering slot defaults are filled in below the object literal.
  dcw_scaler: 0.05,
  dcw_high_scaler: 0.02,
  dcw_mult_blend: 0.0,
  dcw_mag_phase: 0.0,
  dcw_soft_thresh: 0.0,
  // CFG-path sliders. Only consumed when rcfgMode != "off". The server
  // reads raw.guidance_scale / raw.cfg_rescale and lifts them to
  // uniform [1, T, 1] curves.
  guidance_scale: 7.0,
  cfg_rescale: 0.0,
  steps_override: 8,
};

for (let slot = 1; slot <= MANUAL_SLOT_PREREG; slot++) {
  DEFAULT_SLIDER_VALUES[`man_src_${slot}`] = 0;
  DEFAULT_SLIDER_VALUES[`man_layer_${slot}`] = 9;
  DEFAULT_SLIDER_VALUES[`man_step_${slot}`] = 0;
  DEFAULT_SLIDER_VALUES[`man_alpha_${slot}`] = 0.0;
}

/** A re-applyable record of an active timbre / structure reference.
 *  `timbreName` / `structName` are the server-acked DISPLAY name
 *  (cleared whenever the session leaves "ready"); this is the client's
 *  own record of WHAT it set, kept so a WS reconnect can re-send it to
 *  the fresh server session. `clip` refs re-resolve their PCM from
 *  useCustomTracksStore via loadFixtureAudio, so no buffer is stored. */
export type RefSource =
  | { mode: "fixture"; name: string }
  | { mode: "clip"; name: string };

/** Loop region (the playback "brace") in seconds, or null when none is
 *  set. Owned here rather than in WaveformScrubBox local state so it can
 *  round-trip through config export/import. */
export interface LoopBand {
  start: number;
  end: number;
}

export { type LoopGridRes };

interface PerformanceState {
  /** Per-param defaults used by double-click reset, long-press snap-back,
   *  and idle reset. Seeded from DEFAULT_SLIDER_VALUES and overwritten by
   *  applyConfig() so the reset target matches the operator-configured
   *  startup value, not the hardcoded fallback. */
  sliderDefaults: Record<string, number>;
  /** What we're actually sending to the engine. When `smooth` is on, this
   *  trails `sliderTargets` along a tween; otherwise it equals it. Read by
   *  param-sync (hooks/useParamSync.ts) and the render loop (the audio-
   *  reactive graph in hooks/useRenderLoop.ts). */
  sliderValues: Record<string, number>;
  /** What the user *intends* — instant target. Slider UIs (SliderGroup,
   *  MobileRemixRail, DesktopEdgeDrag) read this so the visual position
   *  follows the cursor / MIDI knob without the smoothing lag. */
  sliderTargets: Record<string, number>;
  /** Sparse visual-only override for slider position. When a key is
   *  present, the top-edge ribbon and mobile rail render this value
   *  instead of `sliderTargets[key]`. The engine still reads
   *  `sliderValues`, untouched by this field. Used by the per-song
   *  "hear source first" demo: the ribbon glides from its prior value
   *  down to 0 while the engine value snaps to 0 immediately. Cleared
   *  per-key when the demo finishes (tickDropTweens) or the user
   *  touches the slider (setSlider / bumpSlider / setSliderDirect). */
  sliderDisplayOverride: Record<string, number>;
  /** Integer seed forwarded straight to torch.manual_seed on the engine.
   *  Range [0, 0xFFFFFFFF] (uint32). The "dice" button rerolls; the
   *  value cell is double-click-editable. Previously a 0..1 float that
   *  pipeline.py multiplied by 1000 — that hidden multiplier capped
   *  entropy at 1001 values; we send the int as-is now. */
  seed: number;
  /** Two prompts. The A/B blend itself lives in
   *  ``sliderValues["prompt_blend"]`` so it rides the Smooth tween
   *  machinery, the graph, and the generic MIDI / keyboard / param
   *  pipelines the same way every other knob does. */
  promptA: string;
  promptB: string;
  /** Currently active key (e.g. "G# minor"). May come from auto-detect. */
  activeKey: string;
  /** Currently active time-signature numerator as a wire string
   *  ("2" | "3" | "4" | "6"). The encoder bakes it into the prompt as
   *  ``- timesignature: <value>``. Source order mirrors activeKey:
   *  sidecar → operator override (AlmostReadyDialog / advanced strip) →
   *  default ("4"). */
  activeTimeSignature: TimeSignature;
  /** Selected fixture name (from /api/fixtures). */
  fixture: string;
  /** Display name of the active uploaded timbre-reference track, or
   *  null when none (server uses self-timbre / playback source). Set
   *  by the server's timbre_set ack and cleared by timbre_cleared.
   *  Per-session — not persisted across reloads. */
  timbreName: string | null;
  /** Display name of the active uploaded structure-reference track, or
   *  null when none (server uses self-structure / playback source's own
   *  semantic hints). Set/cleared by structure_set / structure_cleared
   *  acks. Per-session — not persisted. */
  structName: string | null;
  /** Client-side record of the active timbre / structure references —
   *  enough to re-send them after a WS reconnect (the fresh server
   *  session boots with no overrides, and refs aren't part of the
   *  session config). Null when no override is active. */
  timbreRef: RefSource | null;
  structRef: RefSource | null;
  /** Detected musical metadata from server's "ready" frame. */
  detectedBpm: number | null;
  detectedKey: string | null;
  /** Last time signature the server reported (sidecar value on a hit,
   *  "4" otherwise). Mirrors detectedKey: surfaces in the advanced
   *  strip's "Detected: …" readout even when the operator has
   *  overridden activeTimeSignature. */
  detectedTimeSignature: TimeSignature | null;
  /** When non-null, the next swap_ready handler applies this key
   *  instead of the server-detected one and then clears it. Lets the
   *  upload dialog's "Set manually" mode survive the swap roundtrip
   *  without being clobbered by the CNN's result. */
  pendingKeyOverride: string | null;
  /** Same one-shot semantics as pendingKeyOverride but for time
   *  signature. Set by AlmostReadyDialog when the user picks "Set
   *  manually"; useFixtureSwap consumes and clears it on swap_ready. */
  pendingTimeSignatureOverride: TimeSignature | null;
  /** Display mode toggle. */
  mode: DisplayMode;
  /** Kiosk mode (auto-hide cursor + idle reset). */
  kiosk: boolean;
  /** Pause flag (audio context state). */
  paused: boolean;
  /** Per-session, non-persisted gate: when false, the user has loaded a
   *  song but hasn't yet started the remix. Song-load hooks (start +
   *  fixture swap) reset this to false so each new track plays the
   *  source first; the top edge ribbon's first value-changing drag
   *  flips it true. Drives the "drag to start" affordance and gates
   *  the side-rail tutorial hints. */
  remixStarted: boolean;

  /** One-shot opt-out for useFixtureSwap's perf.fixture subscription.
   *  Set true before writing perf.fixture to suppress the next
   *  user-initiated swap path (loadFixtureAudio → sendSwapSource).
   *  useMcpMirror sets it when adopting an MCP-driven swap whose audio
   *  was already swapped server-side, so the front-end only updates its
   *  display state instead of re-running the swap. Consumed-and-cleared
   *  by useFixtureSwap. */
  skipNextFixtureSwap: boolean;

  /** One-shot opt-out for the "hear source first" denoise gate inside
   *  useFixtureSwap. Set true before writing perf.fixture during a
   *  saved-session resume — the gate would otherwise snap the
   *  restored denoise to 0 and play a glide-to-zero animation,
   *  trampling the saved value. Consumed-and-cleared by
   *  useFixtureSwap inside its swap completion handler. Fresh
   *  starts and song-swaps don't set it, so their existing
   *  snap-and-glide behaviour is unchanged. */
  skipNextDenoiseGate: boolean;

  /** DCW (wavelet-domain post-step correction) non-numeric state. The
   * numeric knobs (dcw_scaler, dcw_high_scaler, dcw_mult_blend,
   * dcw_mag_phase, dcw_soft_thresh) all live in sliderValues. */
  dcwEnabled: boolean;
  dcwMode: DcwMode;
  dcwWavelet: DcwWavelet;
  /** RCFG mode for the engine's APG/CFG path. "off" disables guidance
   *  entirely. "full" runs the standard two-pass CFG (2x cost). "initialize"
   *  runs the uncond pass only at step 0 per slot, caches, reuses (~1.07x).
   *  "self" skips the uncond forward entirely; virtual ``v_uncond ≈ initial
   *  noise`` (~1.06x). See acestep/engine/stream.py. */
  rcfgMode: RcfgMode;
  /** Show keyboard-shortcut hints under each slider / next to buttons.
   * Default true. Persisted to localStorage. */
  showKbdHints: boolean;
  /** Smooth slider transitions: when enabled, setSlider interpolates
   *  from the current value to the target over `smoothMs` instead of
   *  jumping. bumpSlider stays immediate (small deltas are already
   *  smooth). Persisted to localStorage. */
  smooth: boolean;
  smoothMs: number;
  /** Loudness matching. When on, the AudioPlayer continuously meters
   *  short-term LUFS, tracks the running max, and boosts quieter
   *  passages to match the loudest seen (peak-clamped). Initial state
   *  comes from config.json → audio.lufs_enabled (applied at boot).
   *  In-session toggles via the LUFS button are not persisted; next
   *  reload returns to the config value. */
  lufsOn: boolean;
  /** Loop-at-end. When true (default), the AudioPlayer's worklet wraps
   *  the playhead via the seam crossfade. When false, the playhead
   *  freezes at end-of-buffer and the transport auto-pauses. */
  loopOn: boolean;
  /** Loop region ("brace") in seconds, or null when none is set.
   *  WaveformScrubBox is the editor; this is the source of truth so the
   *  region survives config export/import. */
  loopBand: LoopBand | null;
  /** Whether the loop *region* is actively looping. Distinct from
   *  `loopOn` (the global full-buffer loop): this gates only the band.
   *  Decoupled from the region's existence so a region can sit
   *  armed-but-off (the DAW loop-toggle behaviour). */
  bandLoopEnabled: boolean;
  /** Snap resolution for loop-region edits. Beat by default. */
  loopGridRes: LoopGridRes;

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
  setPromptA: (s: string) => void;
  setPromptB: (s: string) => void;
  setKey: (k: string) => void;
  setTimeSignature: (s: TimeSignature) => void;
  setFixture: (name: string) => void;
  setTimbreName: (name: string | null) => void;
  setStructName: (name: string | null) => void;
  setTimbreRef: (ref: RefSource | null) => void;
  setStructRef: (ref: RefSource | null) => void;
  /** Update server-reported metadata. ``timeSignature`` is optional for
   *  call-site convenience (older callers only carry bpm + key); when
   *  omitted, ``detectedTimeSignature`` is left untouched. As with key,
   *  ``activeTimeSignature`` adopts the new value automatically (the
   *  upload dialog / advanced strip path is responsible for layering
   *  operator overrides on top via ``pendingTimeSignatureOverride``). */
  setDetected: (
    bpm: number | null,
    key: string | null,
    timeSignature?: TimeSignature | null,
  ) => void;
  /** One-shot key override consumed by useFixtureSwap when the next
   *  swap_ready arrives. Set by the AlmostReadyDialog when the user
   *  picks "Set manually" before uploading. Cleared after consumption
   *  so subsequent swaps fall back to server-detected key. */
  setPendingKeyOverride: (k: string | null) => void;
  setPendingTimeSignatureOverride: (s: TimeSignature | null) => void;
  setMode: (m: DisplayMode) => void;
  toggleMode: () => void;
  setKiosk: (k: boolean) => void;
  toggleKiosk: () => void;
  setPaused: (p: boolean) => void;
  togglePause: () => void;
  /** Set / clear the loop region (seconds). Null clears it. */
  setLoopBand: (band: LoopBand | null) => void;
  /** Enable / disable band looping without destroying the region. */
  setBandLoopEnabled: (enabled: boolean) => void;
  /** Pick the snap resolution for loop edits. */
  setLoopGridRes: (res: LoopGridRes) => void;
  setRemixStarted: (b: boolean) => void;
  setSkipNextFixtureSwap: (b: boolean) => void;
  setSkipNextDenoiseGate: (b: boolean) => void;
  /** Run a visual-only "slide to zero" demo on `param`. Seeds
   *  `sliderDisplayOverride[param] = fromValue` and tweens it down to 0
   *  over `durationMs` using a cubic ease-out, then deletes the key so
   *  ribbons fall through to `sliderTargets`. The engine value
   *  (`sliderValues[param]`) is untouched — call `setSliderDirect`
   *  beforehand if you want the engine snapped to 0. Used by the
   *  per-song "hear source first" gate. No-ops when fromValue <= 0 or
   *  durationMs <= 0. Cancelled implicitly if the user drags the
   *  ribbon mid-demo. */
  animateSliderDisplayFrom: (
    param: string,
    fromValue: number,
    durationMs: number,
  ) => void;
  setDcwEnabled: (b: boolean) => void;
  toggleDcw: () => void;
  setDcwMode: (m: DcwMode) => void;
  setDcwWavelet: (w: DcwWavelet) => void;
  setRcfgMode: (m: RcfgMode) => void;
  toggleKbdHints: () => void;
  toggleSmooth: () => void;
  setSmoothMs: (ms: number) => void;
  toggleLufs: () => void;
  toggleLoop: () => void;
  /** Read localStorage-backed prefs (showKbdHints) and
   *  apply them to the store. Called from a client-only useEffect so SSR
   *  always renders with the defaults — without this, hydration mismatches
   *  on OperatorStrip's COMPACT/STANDARD + KBD: ON/OFF buttons. */
  hydratePersistedPrefs: () => void;
  /** Reset every slider + seed + blend to defaults (idle-reset). */
  resetToDefaults: () => void;
  /** Reset a single slider to its default. Used for the long-press
   *  "snap back" gesture on mobile sliders / rails. */
  resetSlider: (param: string) => void;
}

function clampToMeta(param: string, value: number): number {
  if (Number.isNaN(value)) return 0;
  // Operator-configured channel range wins: covers the channel-gain
  // params (ch_g* / ch*) whose caps live in public/config.json so they
  // can be tuned per-installation without a rebuild. The reverse flag
  // doesn't affect clamping (the stored value still lives in [min, max])
  // — input-side mapping is where reverse is applied.
  const range = getChannelRange(param);
  if (range) {
    return Math.max(range.min, Math.min(range.max, value));
  }
  const meta = SLIDER_META[param];
  // LoRA sliders (lora_str_<id>) aren't in SLIDER_META — their cap is
  // LORA_SLIDER_MAX (currently 1.8). Same convention as the LibraryTile
  // slider widget, the edge bars, and useScheduledCurves. Without this,
  // MIDI knobs / hardware controllers (which go through bumpSlider /
  // setSlider) would write past the operator-facing 1.8 ceiling up to
  // the generic 2.0 fallback.
  const max = meta?.max
    ?? (param.startsWith("lora_str_") ? LORA_SLIDER_MAX : 2.0);
  const min = meta?.min ?? 0;
  return Math.max(min, Math.min(max, value));
}

/** Returns a partial state that drops `param` from sliderDisplayOverride
 *  if it's present. User-touch setters (setSlider / bumpSlider /
 *  setSliderDirect) spread this so any in-flight visual demo tween for
 *  that param is no longer rendered — the user's drag wins immediately.
 *  Returns an empty object when there's nothing to clear, avoiding a
 *  needless object reference change. */
function clearOverridePatch(
  state: PerformanceState,
  param: string,
): Partial<PerformanceState> {
  if (!(param in state.sliderDisplayOverride)) return {};
  const next = { ...state.sliderDisplayOverride };
  delete next[param];
  return { sliderDisplayOverride: next };
}

export const usePerformanceStore = create<PerformanceState>((set) => ({
  sliderDefaults: { ...DEFAULT_SLIDER_VALUES },
  sliderValues: { ...DEFAULT_SLIDER_VALUES },
  sliderTargets: { ...DEFAULT_SLIDER_VALUES },
  sliderDisplayOverride: {},
  seed: 0,
  promptA: "heavy dubstep, deathstep, afxdump, growl heavy bass distortion",
  promptB: "daft punk style, beautiful, four to the floor, angelic",
  activeKey: "G# minor",
  activeTimeSignature: DEFAULT_TIME_SIGNATURE,
  fixture: "",
  timbreName: null,
  structName: null,
  timbreRef: null,
  structRef: null,
  detectedBpm: null,
  detectedKey: null,
  detectedTimeSignature: null,
  pendingKeyOverride: null,
  pendingTimeSignatureOverride: null,
  mode: "graph",
  kiosk: false,
  paused: false,
  remixStarted: false,
  skipNextFixtureSwap: false,
  skipNextDenoiseGate: false,

  dcwEnabled: true,
  dcwMode: "double",
  dcwWavelet: "haar",
  rcfgMode: "off",

  // Hydrated from localStorage after mount via hydratePersistedPrefs() —
  // do NOT read localStorage here, that breaks SSR hydration.
  showKbdHints: true,
  smooth: false,
  smoothMs: DEFAULT_SMOOTH_MS,
  lufsOn: false,
  loopOn: true,
  loopBand: null,
  bandLoopEnabled: true,
  loopGridRes: "beat",

  setSlider: (param, value) => {
    stampManualTouch(param);
    const target = clampToMeta(param, value);
    // Knob→ear latency marker. `window.__demonLatTrace = true` to enable.
    if (
      typeof window !== "undefined" &&
      (window as unknown as { __demonLatTrace?: boolean }).__demonLatTrace
    ) {
      // eslint-disable-next-line no-console
      console.log(
        `[lat] setSlider ${param}=${target.toFixed(3)} t=${performance.now().toFixed(0)}`,
      );
    }
    const state = usePerformanceStore.getState();
    // Cancel both regular and drop-tween for this param: the user just
    // touched the slider, no in-flight demo or smoothing should keep
    // animating against their input.
    cancelTween(param);
    if (!state.smooth || state.smoothMs <= 0) {
      set((s) => ({
        sliderValues: { ...s.sliderValues, [param]: target },
        sliderTargets: { ...s.sliderTargets, [param]: target },
        ...clearOverridePatch(s, param),
      }));
      return;
    }
    // Smooth: write the target instantly (so the UI snaps to where the
    // user wants), kick off a tween that pulls sliderValues toward it.
    set((s) => ({
      sliderTargets: { ...s.sliderTargets, [param]: target },
      ...clearOverridePatch(s, param),
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
      ...clearOverridePatch(s, param),
    }));
  },
  bumpSlider: (param, delta) => {
    stampManualTouch(param);
    const state = usePerformanceStore.getState();
    const newTarget = clampToMeta(
      param,
      (state.sliderTargets[param] ?? 0) + delta,
    );
    cancelTween(param);
    if (!state.smooth || state.smoothMs <= 0) {
      set((s) => ({
        sliderValues: { ...s.sliderValues, [param]: newTarget },
        sliderTargets: { ...s.sliderTargets, [param]: newTarget },
        ...clearOverridePatch(s, param),
      }));
      return;
    }
    // Smoothed bump: retarget instantly, let the tween chase. Each MIDI
    // knob message just bends the in-flight curve toward the new target
    // — no stutter, no per-bump 1 s lag.
    set((s) => ({
      sliderTargets: { ...s.sliderTargets, [param]: newTarget },
      ...clearOverridePatch(s, param),
    }));
    ensureTween(param);
  },
  setSeed: (seed) =>
    set({ seed: Math.max(0, Math.min(0xffffffff, Math.floor(seed))) }),
  randomizeSeed: () =>
    set({ seed: Math.floor(Math.random() * 0x100000000) }),
  setPromptA: (s) => set({ promptA: s }),
  setPromptB: (s) => set({ promptB: s }),
  setKey: (k) => set({ activeKey: k }),
  setTimeSignature: (s) => set({ activeTimeSignature: s }),
  setFixture: (name) => set({ fixture: name }),
  setTimbreName: (name) => set({ timbreName: name }),
  setStructName: (name) => set({ structName: name }),
  setTimbreRef: (ref) => set({ timbreRef: ref }),
  setStructRef: (ref) => set({ structRef: ref }),
  setDetected: (bpm, key, timeSignature) =>
    set((s) => ({
      detectedBpm: bpm,
      detectedKey: key,
      // If user hasn't manually overridden the key (still on default),
      // adopt the detection. Caller can re-set if needed.
      activeKey: key ?? s.activeKey,
      // Same adopt-on-detection rule for time signature. ``undefined``
      // (the call site doesn't carry the field) leaves both detected
      // and active values untouched; an explicit ``null`` clears the
      // detected readout but still preserves activeTimeSignature so a
      // failed re-detect doesn't silently revert the operator's pick.
      ...(timeSignature !== undefined
        ? {
            detectedTimeSignature: timeSignature,
            activeTimeSignature: timeSignature ?? s.activeTimeSignature,
          }
        : {}),
    })),
  setPendingKeyOverride: (k) => set({ pendingKeyOverride: k }),
  setPendingTimeSignatureOverride: (s) =>
    set({ pendingTimeSignatureOverride: s }),
  setMode: (m) => set({ mode: m }),
  toggleMode: () => set((s) => ({ mode: s.mode === "graph" ? "video" : "graph" })),
  setKiosk: (k) => set({ kiosk: k }),
  toggleKiosk: () => set((s) => ({ kiosk: !s.kiosk })),
  setPaused: (p) => set({ paused: p }),
  togglePause: () => set((s) => ({ paused: !s.paused })),
  setRemixStarted: (b) => set({ remixStarted: b }),
  setSkipNextFixtureSwap: (b) => set({ skipNextFixtureSwap: b }),
  setSkipNextDenoiseGate: (b) => set({ skipNextDenoiseGate: b }),
  animateSliderDisplayFrom: (param, fromValue, durationMs) => {
    // Cancel any in-flight smoothing or prior demo tween for this param.
    cancelTween(param);
    if (fromValue <= 0 || durationMs <= 0) {
      // Nothing to animate — just make sure no stale override lingers
      // so the ribbon falls through to sliderTargets.
      set((s) => clearOverridePatch(s, param));
      return;
    }
    // Seed the override synchronously so the very first frame already
    // shows the ribbon at fromValue (no pop), then let tickDropTweens
    // ease it down to 0 and remove the key when done.
    set((s) => ({
      sliderDisplayOverride: { ...s.sliderDisplayOverride, [param]: fromValue },
    }));
    dropTweens.set(param, {
      start: fromValue,
      startTime: performance.now(),
      target: 0,
      durationMs,
    });
    ensureDropTweenRunning();
  },
  setDcwEnabled: (b) => set({ dcwEnabled: b }),
  toggleDcw: () => set((s) => ({ dcwEnabled: !s.dcwEnabled })),
  setDcwMode: (m) => set({ dcwMode: m }),
  setDcwWavelet: (w) => set({ dcwWavelet: w }),
  setRcfgMode: (m) => set({ rcfgMode: m }),
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
  toggleLufs: () =>
    set((s) => ({ lufsOn: !s.lufsOn })),
  toggleLoop: () =>
    set((s) => ({ loopOn: !s.loopOn })),
  setLoopBand: (band) => set({ loopBand: band }),
  setBandLoopEnabled: (enabled) => set({ bandLoopEnabled: enabled }),
  setLoopGridRes: (res) => set({ loopGridRes: res }),
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
    dropTweens.clear();
    set((s) => ({
      sliderValues: { ...s.sliderDefaults },
      sliderTargets: { ...s.sliderDefaults },
      sliderDisplayOverride: {},
      seed: 0,
      remixStarted: false,
    }));
  },
  resetSlider: (param) => {
    const def = usePerformanceStore.getState().sliderDefaults[param];
    if (typeof def !== "number") return;
    cancelTween(param);
    stampManualTouch(param);
    set((s) => ({
      sliderValues: { ...s.sliderValues, [param]: def },
      sliderTargets: { ...s.sliderTargets, [param]: def },
      ...clearOverridePatch(s, param),
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
