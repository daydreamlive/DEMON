"use client";

import { create } from "zustand";

import type { AudioPlayer } from "@/engine/audio/AudioPlayer";
import type { NetworkMonitor } from "@/engine/networkMonitor";
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
  /** Network-quality monitor — owns the slice listener + 500ms evaluator
   *  interval. Lifecycle == session lifecycle so reset() always tears it
   *  down. Mirrors how `remote` and `player` are owned here. */
  monitor: NetworkMonitor | null;
  /** Server-issued WS URL (from /api/queue/join). Null when no queue is in
   *  use — useStartSession falls back to defaultWsUrl(). */
  wsUrl: string | null;

  setStatus: (status: SessionStatus, message?: string) => void;
  setSession: (remote: RemoteBackend | null, player: AudioPlayer | null) => void;
  setMonitor: (monitor: NetworkMonitor | null) => void;
  setWsUrl: (wsUrl: string | null) => void;
  reset: () => void;
}

export const useSessionStore = create<SessionState>((set, get) => ({
  status: "idle",
  message: "",
  remote: null,
  player: null,
  monitor: null,
  wsUrl: null,

  setStatus: (status, message = "") => set({ status, message }),
  setSession: (remote, player) => set({ remote, player }),
  setMonitor: (monitor) => set({ monitor }),
  setWsUrl: (wsUrl) => set({ wsUrl }),
  reset: () => {
    try {
      get().monitor?.stop();
    } catch {}
    set({
      status: "idle",
      message: "",
      remote: null,
      player: null,
      monitor: null,
    });
  },
}));
