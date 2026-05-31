"use client";

import { create } from "zustand";

// Per-path interpolation method for the four live blends. Mirrors the
// server's StreamingSession state (interp_prompt / interp_timbre /
// interp_structure / interp_feedback); useInterpSync ships changes over
// the set_interp_method WS message and re-pushes the whole set on every
// transition into "ready" so a value picked in a prior session doesn't
// silently disagree with the server (which boots every path at "slerp").
//
// Defaults MUST match the server defaults in acestep/streaming/state.py.

export type InterpMethod = "slerp" | "linear";
export type InterpPath = "prompt" | "timbre" | "structure" | "feedback";

export const INTERP_PATHS: InterpPath[] = [
  "structure",
  "timbre",
  "prompt",
  "feedback",
];

export const INTERP_PATH_LABELS: Record<InterpPath, string> = {
  structure: "Structure",
  timbre: "Timbre",
  prompt: "Prompt",
  feedback: "Feedback",
};

interface InterpState {
  methods: Record<InterpPath, InterpMethod>;
  setMethod: (path: InterpPath, method: InterpMethod) => void;
}

export const useInterpStore = create<InterpState>((set) => ({
  methods: {
    prompt: "slerp",
    timbre: "slerp",
    structure: "slerp",
    feedback: "slerp",
  },
  setMethod: (path, method) =>
    set((s) => ({ methods: { ...s.methods, [path]: method } })),
}));
