import { create } from "zustand";

import type { KnobManifest } from "@demon/client";

// Holds the backend knob manifest fetched at boot (GET /api/knobs).
// Seeded empty; RTMGBoot fills it. Any manifest-driven surface (the
// DynamicKnobPanel, or a future re-skin) reads from here. Kept separate
// from usePerformanceStore so it's purely the schema — live values stay
// in the perf store.
interface KnobManifestState {
  knobs: KnobManifest;
  loaded: boolean;
  setManifest: (knobs: KnobManifest) => void;
}

export const useKnobManifestStore = create<KnobManifestState>((set) => ({
  knobs: {},
  loaded: false,
  setManifest: (knobs) => set({ knobs, loaded: true }),
}));
