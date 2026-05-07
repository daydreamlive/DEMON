"use client";

import { create } from "zustand";

import type { AudioPlayer } from "@/engine/audio/AudioPlayer";
import type { RemoteBackend } from "@/engine/protocol";

// Live-session lifecycle state. The non-serializable RemoteBackend +
// AudioPlayer instances live here so React components and hooks can react
// to state changes (status, errors) without owning the lifecycle directly.

export type SessionStatus =
  | "idle"
  | "loading-fixture"
  | "connecting"
  | "ready"
  | "error"
  | "closed";

interface SessionState {
  status: SessionStatus;
  message: string;
  remote: RemoteBackend | null;
  player: AudioPlayer | null;
  /** Server-issued WS URL (from /api/queue/join). Null when no queue is in
   *  use — useStartSession falls back to defaultWsUrl(). */
  wsUrl: string | null;

  setStatus: (status: SessionStatus, message?: string) => void;
  setSession: (remote: RemoteBackend | null, player: AudioPlayer | null) => void;
  setWsUrl: (wsUrl: string | null) => void;
  reset: () => void;
}

export const useSessionStore = create<SessionState>((set) => ({
  status: "idle",
  message: "",
  remote: null,
  player: null,
  wsUrl: null,

  setStatus: (status, message = "") => set({ status, message }),
  setSession: (remote, player) => set({ remote, player }),
  setWsUrl: (wsUrl) => set({ wsUrl }),
  reset: () =>
    set({ status: "idle", message: "", remote: null, player: null }),
}));
