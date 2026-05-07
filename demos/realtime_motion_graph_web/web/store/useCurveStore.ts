"use client";

import { create } from "zustand";

import {
  type CurvePoint,
  type CurveState,
  SCHEDULEABLE_PARAMS,
  defaultCurveState,
} from "@/types/curves";

// Curve-scheduling state. Per-param curves the user draws in the
// ScheduleCurvesOverlay; applied by useScheduledCurves at rAF cadence.
//
// Persistence: the entire state (curves + scheduleEnabled + activeCurve)
// is JSON-stringified to a single localStorage key so user state
// survives reloads. Hydration runs in a client-only useEffect.
//
// Dynamic params: keys can be any string the engine accepts (the fixed
// set in SCHEDULEABLE_PARAMS is just the always-shown tabs; LoRA
// strength curves use lora_str_<id> keys added at runtime when the
// LoRA catalog arrives).

const CURVES_STORAGE_KEY = "demon:curves";
const SCHEDULE_ENABLED_KEY = "demon:scheduleEnabled";

type CurveMap = Record<string, CurveState>;

function freshCurveMap(): CurveMap {
  const out: CurveMap = {};
  for (const p of SCHEDULEABLE_PARAMS) out[p] = defaultCurveState();
  return out;
}

function loadCurves(): CurveMap | null {
  if (typeof localStorage === "undefined") return null;
  try {
    const raw = localStorage.getItem(CURVES_STORAGE_KEY);
    if (!raw) return null;
    const parsed = JSON.parse(raw) as Partial<CurveMap>;
    if (typeof parsed !== "object" || parsed === null) return null;
    // Merge with defaults so a stored snapshot from an earlier version
    // (with fewer params) hydrates cleanly. Persist any extra param
    // keys (LoRA curves) too.
    const fresh: CurveMap = freshCurveMap();
    for (const key of Object.keys(parsed)) {
      const stored = parsed[key];
      if (
        stored &&
        Array.isArray(stored.points) &&
        stored.points.length >= 2 &&
        typeof stored.enabled === "boolean"
      ) {
        fresh[key] = stored as CurveState;
      }
    }
    return fresh;
  } catch {
    return null;
  }
}

function saveCurves(curves: CurveMap): void {
  if (typeof localStorage === "undefined") return;
  try {
    localStorage.setItem(CURVES_STORAGE_KEY, JSON.stringify(curves));
  } catch {}
}

function loadScheduleEnabled(): boolean {
  if (typeof localStorage === "undefined") return true;
  try {
    const v = localStorage.getItem(SCHEDULE_ENABLED_KEY);
    if (v === null) return true;
    return v === "1";
  } catch {
    return true;
  }
}

function saveScheduleEnabled(b: boolean): void {
  if (typeof localStorage === "undefined") return;
  try {
    localStorage.setItem(SCHEDULE_ENABLED_KEY, b ? "1" : "0");
  } catch {}
}

interface CurveStore {
  curves: CurveMap;
  /** Which curve is shown / edited in the tab strip. Stored as a
   *  string so LoRA params (lora_str_<id>) work too. */
  activeCurve: string;
  /** Overlay open / closed. Toggled from OperatorStrip's SCHEDULE
   *  CURVES button (and ESC / × inside the overlay). */
  overlayOpen: boolean;
  /** Master enable. When false, NO curve drives any param, regardless
   *  of per-curve enabled flags. Lets the user "pause" all
   *  automation without losing their drawings. Persisted. */
  scheduleEnabled: boolean;

  setCurvePoints: (param: string, points: CurvePoint[]) => void;
  setCurveEnabled: (param: string, enabled: boolean) => void;
  setActiveCurve: (param: string) => void;
  toggleOverlay: () => void;
  closeOverlay: () => void;
  resetCurve: (param: string) => void;
  toggleScheduleEnabled: () => void;
  setScheduleEnabled: (b: boolean) => void;
  /** Lazily allocate a curve for a param the first time it's
   *  referenced. Useful for dynamic params (LoRA strengths) that
   *  weren't in SCHEDULEABLE_PARAMS at boot. */
  ensureCurve: (param: string) => void;

  /** Read localStorage (client-only) and apply. SSR returns null so
   *  this is a no-op until a useEffect calls it post-mount. */
  hydratePersistedCurves: () => void;
}

export const useCurveStore = create<CurveStore>((set, get) => ({
  curves: freshCurveMap(),
  activeCurve: "denoise",
  overlayOpen: false,
  scheduleEnabled: true,

  setCurvePoints: (param, points) => {
    const sorted = [...points].sort((a, b) => a.x - b.x);
    if (sorted.length >= 2) {
      sorted[0] = { ...sorted[0], x: 0 };
      sorted[sorted.length - 1] = { ...sorted[sorted.length - 1], x: 1 };
    }
    const prev = get().curves[param] ?? defaultCurveState();
    const next: CurveMap = {
      ...get().curves,
      [param]: { enabled: true, points: sorted },
    };
    if (prev.enabled === false && sorted.length === prev.points.length) {
      next[param].enabled = prev.enabled;
    }
    saveCurves(next);
    set({ curves: next });
  },

  setCurveEnabled: (param, enabled) => {
    const prev = get().curves[param] ?? defaultCurveState();
    const next: CurveMap = {
      ...get().curves,
      [param]: { ...prev, enabled },
    };
    saveCurves(next);
    set({ curves: next });
  },

  setActiveCurve: (param) => set({ activeCurve: param }),

  toggleOverlay: () => set((s) => ({ overlayOpen: !s.overlayOpen })),
  closeOverlay: () => set({ overlayOpen: false }),

  resetCurve: (param) => {
    const next: CurveMap = {
      ...get().curves,
      [param]: defaultCurveState(),
    };
    saveCurves(next);
    set({ curves: next });
  },

  toggleScheduleEnabled: () =>
    set((s) => {
      const v = !s.scheduleEnabled;
      saveScheduleEnabled(v);
      return { scheduleEnabled: v };
    }),
  setScheduleEnabled: (b) => {
    saveScheduleEnabled(b);
    set({ scheduleEnabled: b });
  },

  ensureCurve: (param) => {
    if (get().curves[param]) return;
    const next: CurveMap = {
      ...get().curves,
      [param]: defaultCurveState(),
    };
    set({ curves: next });
  },

  hydratePersistedCurves: () => {
    const loaded = loadCurves();
    if (loaded) set({ curves: loaded });
    set({ scheduleEnabled: loadScheduleEnabled() });
  },
}));
