"use client";

import { create } from "zustand";

import type { StemOverlayKind } from "@/engine/audio/loadFixture";

interface StemOverlayState {
  enabled: Record<StemOverlayKind, boolean>;
  volumes: Record<StemOverlayKind, number>;
  setEnabled: (kind: StemOverlayKind, enabled: boolean) => void;
  setVolume: (kind: StemOverlayKind, volume: number) => void;
  toggle: (kind: StemOverlayKind) => void;
}

export const useStemOverlayStore = create<StemOverlayState>((set) => ({
  enabled: { vocals: false, instruments: false },
  volumes: { vocals: 0.65, instruments: 0.65 },

  setEnabled: (kind, enabled) =>
    set((s) => ({ enabled: { ...s.enabled, [kind]: enabled } })),

  setVolume: (kind, volume) =>
    set((s) => ({
      volumes: {
        ...s.volumes,
        [kind]: Math.max(0, Math.min(1.5, volume)),
      },
    })),

  toggle: (kind) =>
    set((s) => ({
      enabled: { ...s.enabled, [kind]: !(s.enabled[kind] ?? false) },
    })),
}));
