"use client";

import { create } from "zustand";

import type { AudioPlayer } from "@demon/client";
import type { NetworkMonitor } from "@/engine/networkMonitor";
import type { CapabilityMask, RemoteBackend, WsTrace } from "@demon/client";
import type { WsReconnector } from "@demon/client";

// Live-session lifecycle state. The non-serializable RemoteBackend +
// AudioPlayer instances live here so React components and hooks can react
// to state changes (status, errors) without owning the lifecycle directly.

export type SessionStatus =
  | "idle"
  | "loading-fixture"
  | "connecting"
  | "ready"
  | "reconnecting"
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
  /** Active reconnect orchestrator. Non-null while the recovery backoff
   *  loop is running after an abnormal WS close. reset() cancels it so a
   *  fresh-session start can't be raced by a stale recovery attempt
   *  silently replacing the new session's remote. */
  reconnector: WsReconnector | null;
  /** Server-issued WS URL (from /api/queue/join). Null when no queue is in
   *  use — useStartSession falls back to defaultWsUrl(). */
  wsUrl: string | null;
  /** Active checkpoint's model-scale label ("2B" | "5B" | null). Set
   *  from the WS ready message and from /api/loras. Null when unknown.
   *  The LoRA library uses this to hide LoRAs whose trained
   *  ``base_model_scale`` doesn't match. */
  checkpointScale: string | null;
  /** Current StreamPipeline ring-buffer depth (concurrent denoising
   *  slots). Mirrors the server's view; updated from ``ready`` and
   *  ``depth_applied`` messages. Null until a session is live. */
  pipelineDepth: number | null;
  /** Server-imposed ceiling on ``pipelineDepth`` — TRT engine batch_max
   *  for TRT decoders, 4 for eager / compile. Null until ready. */
  maxPipelineDepth: number | null;
  /** Backend capability mask from the WS ``ready`` message. The
   *  hand-coded panels (track swap, timbre/structure refs, LoRA
   *  library) gate on it so a backend family that can't honor a
   *  command doesn't render dead controls. Null until ready — and on
   *  pre-Phase-2 servers / recorded replays, which means "ungated"
   *  (see useCapability). */
  capabilities: CapabilityMask | null;
  /** Latest browser-observed WS trace, including orphan remotes that
   *  failed before setSession() could publish them. */
  lastWsTrace: WsTrace | null;
  /** Pod-side session id from optional init_ack telemetry. */
  lastBackendSessionId: string | null;
  /** Client id echoed by init_ack. */
  lastBackendClientId: string | null;
  /** Active manual steering slot count, mirrored from the server.
   *  Null until ready. */
  manualSlotCount: number | null;
  /** Server-imposed cap on manual steering slots. Drives the +/- enable
   *  state in ModTile. Null until ready. */
  manualSlotCap: number | null;
  /** Whether the session's checkpoint has steering vectors. When false,
   *  ModTile hides both steering tiles. Null until ready. */
  steeringAvailable: boolean | null;

  setStatus: (status: SessionStatus, message?: string) => void;
  setSession: (remote: RemoteBackend | null, player: AudioPlayer | null) => void;
  setMonitor: (monitor: NetworkMonitor | null) => void;
  setReconnector: (reconnector: WsReconnector | null) => void;
  setWsUrl: (wsUrl: string | null) => void;
  setCheckpointScale: (scale: string | null) => void;
  setPipelineDepth: (depth: number | null) => void;
  setMaxPipelineDepth: (max: number | null) => void;
  setCapabilities: (capabilities: CapabilityMask | null) => void;
  setLastWsTrace: (trace: WsTrace | null) => void;
  setLastBackendSessionId: (id: string | null) => void;
  setLastBackendClientId: (id: string | null) => void;
  setManualSlotCount: (count: number | null) => void;
  setManualSlotCap: (cap: number | null) => void;
  setSteeringAvailable: (available: boolean | null) => void;
  reset: () => void;
}

export const useSessionStore = create<SessionState>((set, get) => ({
  status: "idle",
  message: "",
  remote: null,
  player: null,
  monitor: null,
  reconnector: null,
  wsUrl: null,
  checkpointScale: null,
  pipelineDepth: null,
  maxPipelineDepth: null,
  capabilities: null,
  lastWsTrace: null,
  lastBackendSessionId: null,
  lastBackendClientId: null,
  manualSlotCount: null,
  manualSlotCap: null,
  steeringAvailable: null,

  setStatus: (status, message = "") => set({ status, message }),
  setSession: (remote, player) => set({ remote, player }),
  setMonitor: (monitor) => set({ monitor }),
  setReconnector: (reconnector) => set({ reconnector }),
  setWsUrl: (wsUrl) => set({ wsUrl }),
  setCheckpointScale: (scale) => set({ checkpointScale: scale }),
  setPipelineDepth: (depth) => set({ pipelineDepth: depth }),
  setMaxPipelineDepth: (max) => set({ maxPipelineDepth: max }),
  setCapabilities: (capabilities) => set({ capabilities }),
  setLastWsTrace: (trace) => set({ lastWsTrace: trace }),
  setLastBackendSessionId: (id) => set({ lastBackendSessionId: id }),
  setLastBackendClientId: (id) => set({ lastBackendClientId: id }),
  setManualSlotCount: (count) => set({ manualSlotCount: count }),
  setManualSlotCap: (cap) => set({ manualSlotCap: cap }),
  setSteeringAvailable: (available) => set({ steeringAvailable: available }),
  reset: () => {
    try {
      get().monitor?.stop();
    } catch {}
    try {
      get().reconnector?.cancel();
    } catch {}
    set({
      status: "idle",
      message: "",
      remote: null,
      player: null,
      monitor: null,
      reconnector: null,
      pipelineDepth: null,
      maxPipelineDepth: null,
      // Capability mask is per-session truth (the next session may pick
      // a different backend family), unlike checkpointScale below.
      capabilities: null,
      lastWsTrace: null,
      lastBackendSessionId: null,
      lastBackendClientId: null,
      manualSlotCount: null,
      manualSlotCap: null,
      steeringAvailable: null,
      // checkpointScale survives reset on purpose: the server's
      // checkpoint doesn't change across sessions, and pre-fetching
      // it from /api/loras lets the library filter render correctly
      // even before the first session starts.
    });
  },
}));
