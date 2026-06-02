// Enum-valued MIDI targets — the advanced-drawer dropdowns (DCW mode,
// wavelet, RCFG mode, pipeline depth, key, time-signature). These aren't
// continuous slider params, so they don't ride the `cc` → setSlider path.
// Instead a CC binding (map.ccEnum) quantizes the knob's 0..127 sweep
// across the option list, and a note binding (map.noteEnum) cycles to the
// next option on each press.
//
// Each target is resolved at dispatch time against live store state
// (mirroring useMidi's noteAction switch) so the knob/pad always reads and
// writes the current value, and a change made via the dropdown itself is
// picked up on the next message.

import { useSessionStore } from "@/store/useSessionStore";
import { usePerformanceStore } from "@/store/usePerformanceStore";
import {
  DCW_MODES,
  DCW_WAVELETS,
  RCFG_MODES,
  VALID_KEYSCALES,
  VALID_TIME_SIGNATURES,
  type DcwMode,
  type DcwWavelet,
  type RcfgMode,
  type TimeSignature,
} from "@/types/engine";

export interface EnumTarget {
  /** Current option list. A function so dynamic targets (pipeline depth,
   *  whose ceiling is server-imposed) recompute per call. */
  options: () => readonly string[];
  /** Current selected option value, or null when unavailable (e.g. no
   *  session yet for pipeline depth). */
  get: () => string | null;
  /** Commit a new option value. */
  set: (value: string) => void;
}

// Re-send the prompt after a key / time-signature change — the encoder
// bakes both into the prompt text, so the engine only picks them up on a
// fresh send. Mirrors the OperatorStrip dropdown handlers (minus the
// confirm dialog, which doesn't fit a hardware twist).
function resendPrompt(): void {
  const perf = usePerformanceStore.getState();
  const remote = useSessionStore.getState().remote;
  remote?.sendPrompt(perf.promptA, perf.activeKey, perf.activeTimeSignature);
}

// Pipeline-depth changes rebuild the StreamPipeline (one ring's worth of
// audio glitches per change). A knob sweep crossing several depth bands
// would fire a rebuild storm, so trail the send.
const DEPTH_DEBOUNCE_MS = 150;
let depthTimer: ReturnType<typeof setTimeout> | null = null;
let pendingDepth: number | null = null;

function sendDepthDebounced(depth: number): void {
  pendingDepth = depth;
  if (depthTimer) clearTimeout(depthTimer);
  depthTimer = setTimeout(() => {
    depthTimer = null;
    const v = pendingDepth;
    pendingDepth = null;
    if (v === null) return;
    useSessionStore.getState().remote?.sendSetDepth(v);
  }, DEPTH_DEBOUNCE_MS);
}

const ENUM_TARGETS: Record<string, EnumTarget> = {
  dcw_mode: {
    options: () => DCW_MODES,
    get: () => usePerformanceStore.getState().dcwMode,
    set: (v) => usePerformanceStore.getState().setDcwMode(v as DcwMode),
  },
  dcw_wavelet: {
    options: () => DCW_WAVELETS,
    get: () => usePerformanceStore.getState().dcwWavelet,
    set: (v) => usePerformanceStore.getState().setDcwWavelet(v as DcwWavelet),
  },
  rcfg_mode: {
    options: () => RCFG_MODES,
    get: () => usePerformanceStore.getState().rcfgMode,
    set: (v) => usePerformanceStore.getState().setRcfgMode(v as RcfgMode),
  },
  time_signature: {
    options: () => VALID_TIME_SIGNATURES,
    get: () => usePerformanceStore.getState().activeTimeSignature,
    set: (v) => {
      usePerformanceStore.getState().setTimeSignature(v as TimeSignature);
      resendPrompt();
    },
  },
  // ~70 options — realistically a quantize-by-knob target (a pad cycle
  // would take 70 presses). Mappable all the same.
  key: {
    options: () => VALID_KEYSCALES,
    get: () => usePerformanceStore.getState().activeKey,
    set: (v) => {
      usePerformanceStore.getState().setKey(v);
      resendPrompt();
    },
  },
  // Dynamic: 1..maxPipelineDepth (server ceiling). The store value is
  // updated by the server ack, so set() only sends — it does not write
  // the store optimistically.
  pipeline_depth: {
    options: () => {
      const max = useSessionStore.getState().maxPipelineDepth;
      if (typeof max !== "number" || max < 1) return [];
      return Array.from({ length: max }, (_, i) => String(i + 1));
    },
    get: () => {
      const d = useSessionStore.getState().pipelineDepth;
      return typeof d === "number" ? String(d) : null;
    },
    set: (v) => {
      const n = parseInt(v, 10);
      if (!Number.isFinite(n)) return;
      sendDepthDebounced(n);
    },
  },
};

export function resolveEnum(id: string): EnumTarget | null {
  return ENUM_TARGETS[id] ?? null;
}

export function isEnumTarget(id: string): boolean {
  return id in ENUM_TARGETS;
}

/** All enum target ids, for the ConfigModal mapping table. */
export const ENUM_TARGET_IDS = Object.keys(ENUM_TARGETS);

/** Quantize a raw CC value (0..127) to an option index, with hysteresis
 *  so a knob parked on a boundary doesn't flip-flop. Each option owns a
 *  band of width `127 / (N-1)`; we only leave the current option's band
 *  once the value crosses its centre by more than a quarter-band. Returns
 *  the current index unchanged when N < 2. */
export function quantizeIndex(
  value: number,
  optionCount: number,
  currentIndex: number,
): number {
  const n = optionCount;
  if (n <= 1) return 0;
  const cur = Math.max(0, Math.min(n - 1, currentIndex));
  const band = 127 / (n - 1);
  const hysteresis = band * 0.25;
  const curCenter = cur * band;
  // Stay put inside the dead zone around the current option's centre.
  if (Math.abs(value - curCenter) <= band / 2 + hysteresis) return cur;
  return Math.max(0, Math.min(n - 1, Math.round(value / band)));
}

/** Apply a quantized CC sweep to an enum target. No-op when the target is
 *  unavailable or the index doesn't change (avoids redundant sends /
 *  re-renders / pipeline rebuilds). */
export function applyEnumCC(id: string, value: number): void {
  const target = resolveEnum(id);
  if (!target) return;
  const opts = target.options();
  if (opts.length === 0) return;
  const current = target.get();
  const curIdx = current === null ? 0 : Math.max(0, opts.indexOf(current));
  const nextIdx = quantizeIndex(value, opts.length, curIdx);
  if (nextIdx === curIdx) return;
  target.set(opts[nextIdx]);
}

/** Cycle an enum target to its next option (wraps). No-op when empty. */
export function cycleEnum(id: string): void {
  const target = resolveEnum(id);
  if (!target) return;
  const opts = target.options();
  if (opts.length === 0) return;
  const current = target.get();
  const curIdx = current === null ? -1 : opts.indexOf(current);
  const nextIdx = (curIdx + 1) % opts.length;
  target.set(opts[nextIdx]);
}
