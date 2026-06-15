"use client";

import { create } from "zustand";

// Connection-quality store. Driven by createNetworkMonitor()
// (engine/networkMonitor.ts), which fuses two signals — a stall
// watchdog (no slices arriving) and a bleed watchdog (slices arriving
// but landing behind the playhead, i.e. raw source audible) — into a
// single `quality` field consumed by NetworkIndicator. The indicator
// is hidden when "healthy"; "unstable" fades in a subtle bottom-center
// pill, agnostic about which leg of the connection is the source of
// the degradation.

export type NetworkQuality = "healthy" | "unstable";

interface NetworkState {
  quality: NetworkQuality;
  /** performance.now() of the most recent slice arrival, or 0 if none yet. */
  lastSliceAt: number;
  /** ms since last slice at the most recent evaluator tick. */
  staleMs: number;
  /** p95 / median of recent inter-arrival deltas. 1 = perfectly steady. */
  jitterRatio: number;

  update: (
    partial: Partial<Omit<NetworkState, "update" | "reset">>,
  ) => void;
  reset: () => void;
}

const INITIAL = {
  quality: "healthy" as NetworkQuality,
  lastSliceAt: 0,
  staleMs: 0,
  jitterRatio: 1,
};

export const useNetworkStore = create<NetworkState>((set) => ({
  ...INITIAL,
  update: (partial) => set(partial),
  reset: () => set(INITIAL),
}));
