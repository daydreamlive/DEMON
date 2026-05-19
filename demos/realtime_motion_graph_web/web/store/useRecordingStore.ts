"use client";

import { create } from "zustand";

// Audio recording state. Discriminated union prevents impossible states
// (preview without blob, recording without start time, etc.). The
// useRecording hook drives transitions; UI components read state and
// dispatch via document CustomEvents (parallel to dd:toggle-drawer).

export type RecState =
  | { kind: "idle" }
  | { kind: "arming" }
  | { kind: "recording"; startedAt: number; bytes: number; pausedMs: number }
  | {
      kind: "paused";
      startedAt: number;
      bytes: number;
      pausedMs: number;
      pausedAt: number;
    }
  | { kind: "finalizing" }
  | {
      kind: "preview";
      blob: Blob;
      url: string;
      durationMs: number;
      mime: string;
      ext: string;
      /** Optional muxed video+audio blob captured alongside the audio
       *  feed (graph canvas via captureStream). Present only when the
       *  browser supports the right MIME and the graph canvas existed
       *  at start time. When absent, the preview falls through to the
       *  audio-only WAV download path. */
      videoBlob?: Blob;
      videoUrl?: string;
      videoMime?: string;
      videoExt?: string;
    }
  | { kind: "error"; reason: string };

interface RecordingStore {
  state: RecState;
  // Single setter — the hook owns the state machine logic, this just stores.
  set: (next: RecState) => void;
  // Live byte tally so the UI can show approximate file size while
  // recording without forcing a state replacement on every chunk.
  bumpBytes: (delta: number) => void;
  /** Transient warning anchored to the record buttons (e.g. "Wait for
   *  connection first" when you click record before the session is
   *  ready). Lives separately from the global StatusBar so it doesn't
   *  overwrite the session's loading/connecting message. */
  warning: string | null;
  setWarning: (msg: string | null) => void;
}

export const useRecordingStore = create<RecordingStore>((set) => ({
  state: { kind: "idle" },
  set: (next) => set({ state: next }),
  bumpBytes: (delta) =>
    set((s) => {
      if (s.state.kind !== "recording" && s.state.kind !== "paused") return s;
      return { state: { ...s.state, bytes: s.state.bytes + delta } };
    }),
  warning: null,
  setWarning: (msg) => set({ warning: msg }),
}));

// Selectors — keep the discriminated union out of components that only
// care "is something happening?".
export function isActive(state: RecState): boolean {
  return state.kind === "recording" || state.kind === "paused";
}

export function elapsedMs(state: RecState, now: number): number {
  if (state.kind === "recording") {
    return now - state.startedAt - state.pausedMs;
  }
  if (state.kind === "paused") {
    return state.pausedAt - state.startedAt - state.pausedMs;
  }
  return 0;
}
