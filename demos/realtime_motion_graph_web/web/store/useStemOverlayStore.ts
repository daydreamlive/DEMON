"use client";

import { create } from "zustand";

import type { StemOverlayKind } from "@/engine/audio/loadFixture";

// Upper bound for a stem overlay's mix level. Stems can be driven well
// past unity (6.0) so a quiet stem can be pushed loud over the model
// output. Authoritative here — the panner drag, keyboard chords, and the
// MIDI range all clamp to this same ceiling.
export const STEM_OVERLAY_MAX = 6.0;

// Resting level a stem returns to on double-click-to-reset, matching the
// DAW-fader convention (double-click → default). Also the initial volume.
export const STEM_OVERLAY_DEFAULT = 0.65;

interface StemOverlayState {
  enabled: Record<StemOverlayKind, boolean>;
  volumes: Record<StemOverlayKind, number>;
  setEnabled: (kind: StemOverlayKind, enabled: boolean) => void;
  setVolume: (kind: StemOverlayKind, volume: number) => void;
  toggle: (kind: StemOverlayKind) => void;
  reset: (kind: StemOverlayKind) => void;
}

export const useStemOverlayStore = create<StemOverlayState>((set) => ({
  enabled: { vocals: false, instruments: false },
  volumes: {
    vocals: STEM_OVERLAY_DEFAULT,
    instruments: STEM_OVERLAY_DEFAULT,
  },

  setEnabled: (kind, enabled) =>
    set((s) => ({ enabled: { ...s.enabled, [kind]: enabled } })),

  setVolume: (kind, volume) =>
    set((s) => ({
      volumes: {
        ...s.volumes,
        [kind]: Math.max(0, Math.min(STEM_OVERLAY_MAX, volume)),
      },
    })),

  toggle: (kind) =>
    set((s) => ({ enabled: { ...s.enabled, [kind]: !s.enabled[kind] } })),

  // Double-click-to-reset: drop back to the default level and unmute so
  // the reset is audible, mirroring SliderGroup's resetSlider contract.
  reset: (kind) =>
    set((s) => ({
      volumes: { ...s.volumes, [kind]: STEM_OVERLAY_DEFAULT },
      enabled: { ...s.enabled, [kind]: true },
    })),
}));
