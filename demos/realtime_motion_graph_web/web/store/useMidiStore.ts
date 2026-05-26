"use client";

import { create } from "zustand";

import { resetKnobDelta } from "@/engine/midi/absoluteDelta";
import {
  DEFAULT_MIDI_MAP,
  MIDI_STORAGE_KEY,
  type MidiMap,
  type NoteAction,
} from "@/engine/midi/types";

function loadMap(): MidiMap {
  if (typeof localStorage === "undefined") {
    return {
      cc: { ...DEFAULT_MIDI_MAP.cc },
      notes: { ...DEFAULT_MIDI_MAP.notes },
      ccActions: { ...(DEFAULT_MIDI_MAP.ccActions ?? {}) },
    };
  }
  try {
    const stored = JSON.parse(localStorage.getItem(MIDI_STORAGE_KEY) || "null");
    if (stored && stored.cc && stored.notes) {
      return {
        cc: { ...stored.cc },
        notes: { ...stored.notes },
        ccActions: { ...(stored.ccActions ?? {}) },
      };
    }
  } catch {}
  return {
    cc: { ...DEFAULT_MIDI_MAP.cc },
    notes: { ...DEFAULT_MIDI_MAP.notes },
    ccActions: { ...(DEFAULT_MIDI_MAP.ccActions ?? {}) },
  };
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

interface LearnState {
  kind: "cc" | "note";
  target: string;
  el: HTMLElement | null;
}

interface MidiState {
  map: MidiMap;
  status: { message: string; tone: "ok" | "warn" | "info" | "off" };
  learn: LearnState | null;
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
  ) => void;
  cancelLearn: () => void;
  applyLearn: (kind: "cc" | "note", num: number) => boolean;
  clearBinding: (kind: "cc" | "note", target: string) => void;

  resetMap: () => void;
}

export const useMidiStore = create<MidiState>((set, get) => ({
  map: loadMap(),
  status: { message: "MIDI", tone: "off" },
  learn: null,
  available: false,
  enabled: loadEnabled(),

  setStatus: (message, tone = "info") => set({ status: { message, tone } }),
  setAvailable: (b) => set({ available: b }),
  setEnabled: (b) => {
    saveEnabled(b);
    set({ enabled: b });
  },

  startLearn: (kind, target, el) => {
    const prev = get().learn;
    if (prev?.el) prev.el.classList.remove("midi-learning");
    if (el) el.classList.add("midi-learning");
    set({
      learn: { kind, target, el },
      status: { message: `Learn: ${target} — twist or press`, tone: "warn" },
    });
  },
  cancelLearn: () => {
    const { learn } = get();
    if (learn?.el) learn.el.classList.remove("midi-learning");
    set({ learn: null });
  },
  applyLearn: (kind, num) => {
    const { learn, map } = get();
    if (!learn) return false;
    // Strict match: learning a slider param ("cc") only accepts CC.
    // Permissive match: learning an action button ("note") accepts EITHER
    // a NOTE message (the usual pad signal) OR a CC message — some pads
    // send CC even in pad/trigger mode and it's friendlier to bind it
    // than to demand the user reconfigure their hardware. The CC binding
    // for an action goes into `ccActions` so the dispatcher can fire the
    // discrete action when that CC arrives, separately from the
    // continuous-controller `cc` map used by sliders.
    if (learn.kind === "cc" && kind !== "cc") return false;
    const next: MidiMap = {
      cc: { ...map.cc },
      notes: { ...map.notes },
      ccActions: { ...(map.ccActions ?? {}) },
    };
    if (learn.kind === "cc") {
      // Slider param learning a CC.
      for (const k of Object.keys(next.cc)) {
        if (next.cc[k] === learn.target) delete next.cc[k];
      }
      next.cc[String(num)] = learn.target;
      // Drop any cached prior value for this CC so the next message
      // re-establishes a baseline. Without this, rebinding CC X from
      // param A to param B would treat the first post-rebind message
      // as a delta from A's last raw value, producing a wrong jump on
      // B's slider.
      resetKnobDelta(num);
    } else if (kind === "note") {
      // Action button learning a NOTE.
      for (const k of Object.keys(next.notes)) {
        if (next.notes[k] === learn.target) delete next.notes[k];
      }
      next.notes[String(num)] = learn.target as NoteAction;
    } else {
      // Action button learning a CC (pad in CC mode).
      const ccActions = next.ccActions ?? {};
      for (const k of Object.keys(ccActions)) {
        if (ccActions[k] === learn.target) delete ccActions[k];
      }
      ccActions[String(num)] = learn.target as NoteAction;
      next.ccActions = ccActions;
    }
    saveMap(next);
    if (learn.el) learn.el.classList.remove("midi-learning");
    set({
      map: next,
      learn: null,
      status: {
        message: `Learned ${learn.target} ← ${kind} ${num}`,
        tone: "ok",
      },
    });
    return true;
  },
  clearBinding: (kind, target) => {
    const { map } = get();
    const next: MidiMap = {
      cc: { ...map.cc },
      notes: { ...map.notes },
    };
    const slot = kind === "cc" ? next.cc : next.notes;
    for (const k of Object.keys(slot)) {
      if (slot[k] === target) delete slot[k];
    }
    saveMap(next);
    set({
      map: next,
      status: { message: `Cleared ${target}`, tone: "info" },
    });
  },

  resetMap: () => {
    const def: MidiMap = {
      cc: { ...DEFAULT_MIDI_MAP.cc },
      notes: { ...DEFAULT_MIDI_MAP.notes },
    };
    saveMap(def);
    // Drop every per-CC delta-tracking baseline. After a full map
    // reset, the next message on any CC should re-establish a
    // baseline, not compute a delta against the previous binding.
    resetKnobDelta();
    set({ map: def, status: { message: "MIDI reset", tone: "ok" } });
  },
}));
