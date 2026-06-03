"use client";

import { create } from "zustand";

import {
  DEFAULT_MIDI_MAP,
  MIDI_STORAGE_KEY,
  type CcMode,
  type MidiMap,
  type NoteAction,
} from "@/engine/midi/types";

/** Deep-clone a (possibly partial / legacy) map into the full current
 *  shape. Every reconstruction in this store routes through here so a new
 *  bucket can never be silently dropped on save — the class of bug that
 *  let `ccActions` disappear through `clearBinding` before this existed. */
function cloneMap(m: Partial<MidiMap> | null | undefined): MidiMap {
  return {
    cc: { ...(m?.cc ?? {}) },
    notes: { ...(m?.notes ?? {}) },
    ccActions: { ...(m?.ccActions ?? {}) },
    ccMode: { ...(m?.ccMode ?? {}) },
    ccEnum: { ...(m?.ccEnum ?? {}) },
    noteEnum: { ...(m?.noteEnum ?? {}) },
  };
}

function loadMap(): MidiMap {
  if (typeof localStorage === "undefined") return cloneMap(DEFAULT_MIDI_MAP);
  try {
    const stored = JSON.parse(localStorage.getItem(MIDI_STORAGE_KEY) || "null");
    if (stored && stored.cc && stored.notes) return cloneMap(stored);
  } catch {}
  return cloneMap(DEFAULT_MIDI_MAP);
}

function saveMap(m: MidiMap): void {
  if (typeof localStorage === "undefined") return;
  try {
    localStorage.setItem(MIDI_STORAGE_KEY, JSON.stringify(m));
  } catch {}
}

// Persist the user's MIDI-in toggle decision so a desktop operator who
// flipped it on once doesn't have to re-enable + re-approve the browser
// permission every page load. Stored separately from the MIDI map so
// the import/export flows can stay map-only.
const MIDI_ENABLED_KEY = "demon:midi:enabled";

function loadEnabled(): boolean {
  if (typeof localStorage === "undefined") return false;
  try {
    return localStorage.getItem(MIDI_ENABLED_KEY) === "1";
  } catch {
    return false;
  }
}

function saveEnabled(b: boolean): void {
  if (typeof localStorage === "undefined") return;
  try {
    localStorage.setItem(MIDI_ENABLED_KEY, b ? "1" : "0");
  } catch {}
}

function clearPhysicalCc(map: MidiMap, key: string): void {
  delete map.cc[key];
  delete map.ccActions![key];
  delete map.ccEnum![key];
  delete map.ccMode![key];
}

function clearPhysicalNote(map: MidiMap, key: string): void {
  delete map.notes[key];
  delete map.noteEnum![key];
}

function clearContinuousTarget(map: MidiMap, target: string): void {
  for (const [key, value] of Object.entries(map.cc)) {
    if (value !== target) continue;
    delete map.cc[key];
    delete map.ccMode![key];
  }
}

function clearActionTarget(map: MidiMap, target: string): void {
  for (const [key, value] of Object.entries(map.notes)) {
    if (value === target) delete map.notes[key];
  }
  for (const [key, value] of Object.entries(map.ccActions ?? {})) {
    if (value !== target) continue;
    delete map.ccActions![key];
    delete map.ccMode![key];
  }
}

function clearEnumTarget(map: MidiMap, target: string): void {
  for (const [key, value] of Object.entries(map.ccEnum ?? {})) {
    if (value !== target) continue;
    delete map.ccEnum![key];
    delete map.ccMode![key];
  }
  for (const [key, value] of Object.entries(map.noteEnum ?? {})) {
    if (value === target) delete map.noteEnum![key];
  }
}

function clearTarget(map: MidiMap, kind: "cc" | "note" | "enum", target: string): void {
  if (kind === "cc") clearContinuousTarget(map, target);
  else if (kind === "note") clearActionTarget(map, target);
  else clearEnumTarget(map, target);
}

interface LearnState {
  /** "cc" = slider param learning a continuous controller.
   *  "note" = action/toggle button learning a pad (or a pad-in-CC-mode).
   *  "enum" = dropdown learning EITHER a knob (→ ccEnum, quantized) OR a
   *           pad (→ noteEnum, cycle), whichever message arrives first. */
  kind: "cc" | "note" | "enum";
  target: string;
  el: HTMLElement | null;
  /** When set (cc learns only), the binding lands with this mode — lets
   *  the right-click menu offer "Learn as Spring" in one gesture. */
  mode?: CcMode;
}

/** Right-click MIDI menu anchor. Opened by the document-level
 *  contextmenu handler in useMidi; rendered by <MidiContextMenu>. */
interface MidiMenuState {
  x: number;
  y: number;
  kind: "cc" | "note" | "enum";
  target: string;
  el: HTMLElement | null;
  extraItems?: MidiMenuItem[];
}

export interface MidiMenuItem {
  label: string;
  onClick: () => void;
  dividerBefore?: boolean;
}

interface MidiState {
  map: MidiMap;
  status: { message: string; tone: "ok" | "warn" | "info" | "off" };
  learn: LearnState | null;
  menu: MidiMenuState | null;
  available: boolean;
  /** User-opt-in gate on the Web MIDI permission prompt. The MIDI-in
   *  jack toggle (HeroMacros' MidiInToggle) flips this; useMidi only
   *  calls navigator.requestMIDIAccess when this is true. Default
   *  false so a fresh page-load never auto-prompts. Persisted in
   *  localStorage so a user who turned it on stays on across loads. */
  enabled: boolean;

  setStatus: (message: string, tone?: MidiState["status"]["tone"]) => void;
  setAvailable: (b: boolean) => void;
  setEnabled: (b: boolean) => void;

  startLearn: (
    kind: LearnState["kind"],
    target: string,
    el: HTMLElement | null,
    mode?: CcMode,
  ) => void;
  cancelLearn: () => void;
  /** Open / close the right-click MIDI menu. */
  openMenu: (menu: MidiMenuState) => void;
  closeMenu: () => void;
  applyLearn: (kind: "cc" | "note", key: string) => boolean;
  clearBinding: (kind: "cc" | "note" | "enum", target: string) => void;
  /** Set the explicit controller type for a bound CC. */
  setCcMode: (cc: string, mode: CcMode) => void;

  resetMap: () => void;
}

export const useMidiStore = create<MidiState>((set, get) => ({
  map: loadMap(),
  status: { message: "MIDI", tone: "off" },
  learn: null,
  menu: null,
  available: false,
  enabled: loadEnabled(),

  setStatus: (message, tone = "info") => set({ status: { message, tone } }),
  setAvailable: (b) => set({ available: b }),
  setEnabled: (b) => {
    saveEnabled(b);
    set({ enabled: b });
  },

  startLearn: (kind, target, el, mode) => {
    const prev = get().learn;
    if (prev?.el) prev.el.classList.remove("midi-learning");
    if (el) el.classList.add("midi-learning");
    set({
      learn: { kind, target, el, mode },
      menu: null,
      status: { message: `Learning ${target} — now move the control`, tone: "warn" },
    });
  },
  cancelLearn: () => {
    const { learn } = get();
    if (learn?.el) learn.el.classList.remove("midi-learning");
    set({ learn: null });
  },
  openMenu: (menu) => set({ menu }),
  closeMenu: () => set({ menu: null }),
  applyLearn: (kind, key) => {
    const { learn, map } = get();
    if (!learn) return false;
    // Pitch bend is a continuous center-resting axis. Do not let it land
    // in enum/action buckets, where runtime dispatch would have no useful
    // press or sweep semantics.
    if (key.startsWith("pb") && learn.kind !== "cc") return false;
    // Strict match: learning a slider param ("cc") only accepts CC.
    // Permissive match: learning an action button ("note") accepts EITHER
    // a NOTE message (the usual pad signal) OR a CC message — some pads
    // send CC even in pad/trigger mode and it's friendlier to bind it
    // than to demand the user reconfigure their hardware. The CC binding
    // for an action goes into `ccActions` so the dispatcher can fire the
    // discrete action when that CC arrives, separately from the
    // continuous-controller `cc` map used by sliders. An enum target
    // accepts either: a CC binds the knob-sweep (ccEnum), a note binds
    // the pad-cycle (noteEnum).
    if (learn.kind === "cc" && kind !== "cc") return false;
    const next = cloneMap(map);

    if (learn.kind === "cc") {
      // Slider param learning a CC.
      clearContinuousTarget(next, learn.target);
      clearPhysicalCc(next, key);
      next.cc[key] = learn.target;
      next.ccMode![key] = learn.mode ?? "absolute";
    } else if (learn.kind === "enum") {
      clearEnumTarget(next, learn.target);
      if (kind === "cc") {
        // Knob → quantized sweep across the enum's options.
        clearPhysicalCc(next, key);
        next.ccEnum![key] = learn.target;
      } else {
        // Pad → cycle to the next option per press.
        clearPhysicalNote(next, key);
        next.noteEnum![key] = learn.target;
      }
    } else if (kind === "note") {
      // Action button learning a NOTE.
      clearActionTarget(next, learn.target);
      clearPhysicalNote(next, key);
      next.notes[key] = learn.target as NoteAction;
    } else {
      // Action button learning a CC (pad in CC mode).
      clearActionTarget(next, learn.target);
      clearPhysicalCc(next, key);
      next.ccActions![key] = learn.target as NoteAction;
    }
    saveMap(next);
    if (learn.el) learn.el.classList.remove("midi-learning");
    set({
      map: next,
      learn: null,
      status: {
        message: `Learned ${learn.target} ← ${key}`,
        tone: "ok",
      },
    });
    return true;
  },
  clearBinding: (kind, target) => {
    const { map } = get();
    const next = cloneMap(map);
    clearTarget(next, kind, target);
    saveMap(next);
    set({
      map: next,
      status: { message: `Cleared ${target}`, tone: "info" },
    });
  },
  setCcMode: (cc, mode) => {
    const next = cloneMap(get().map);
    next.ccMode![cc] = mode;
    saveMap(next);
    set({ map: next });
  },

  resetMap: () => {
    const def = cloneMap(DEFAULT_MIDI_MAP);
    saveMap(def);
    set({ map: def, status: { message: "MIDI reset", tone: "ok" } });
  },
}));
