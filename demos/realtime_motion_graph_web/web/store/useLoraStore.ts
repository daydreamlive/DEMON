"use client";

import { create } from "zustand";

import {
  LORA_DEFAULT_STRENGTH_FRACTION,
  LORA_SLIDER_MAX,
} from "@/types/engine";
import type { LoraCatalogEntry } from "@/types/protocol";

// Server-driven LoRA catalog + per-id strength + enabled set. The catalog
// arrives via /api/loras (cheap filesystem scan, available before WS) and
// is updated mid-session via the WS "lora_catalog" frame.

// LoRAs flipped on the first time the catalog arrives populated. Matches the
// safetensors filename stems delivered by /api/loras (see scripts/loras.default.txt).
// One-shot: a later WS lora_catalog re-broadcast won't re-enable a LoRA the
// user has explicitly disabled.
const DEFAULT_ENABLED_LORAS = new Set(["deathstep", "synthpop"]);

interface LoraState {
  catalog: LoraCatalogEntry[];
  /** Per-id strength (0..LORA_SLIDER_MAX). */
  strengths: Record<string, number>;
  /** Set of enabled LoRA ids. */
  enabled: Set<string>;
  /** Whether default-on LoRAs have already been seeded for this session. */
  _seeded: boolean;

  setCatalog: (catalog: LoraCatalogEntry[]) => void;
  setStrength: (id: string, value: number) => void;
  enable: (id: string) => void;
  disable: (id: string) => void;
  toggle: (id: string) => void;
  reset: () => void;
}

export const useLoraStore = create<LoraState>((set) => ({
  catalog: [],
  strengths: {},
  enabled: new Set(),
  _seeded: false,

  setCatalog: (catalog) =>
    set((s) => {
      // Seed missing strengths from the server's reported defaults so a
      // freshly-arrived LoRA picks up its on-disk default. If the server
      // omits `strength` (current Python backend behavior), fall back to
      // LORA_DEFAULT_STRENGTH_FRACTION so the slider lands at a useful
      // initial level instead of silently sitting at 0.
      const next: Record<string, number> = { ...s.strengths };
      for (const entry of catalog) {
        if (!(entry.id in next)) {
          next[entry.id] =
            typeof entry.strength === "number"
              ? entry.strength
              : LORA_DEFAULT_STRENGTH_FRACTION * LORA_SLIDER_MAX;
        }
      }
      // First populated catalog: flip on the canonical default LoRAs so the
      // demo plays with its intended sound out of the box. Skipped on later
      // re-broadcasts so disabling a default LoRA sticks.
      let enabled = s.enabled;
      let seeded = s._seeded;
      if (!s._seeded && catalog.length > 0) {
        const nextEnabled = new Set(s.enabled);
        for (const entry of catalog) {
          if (DEFAULT_ENABLED_LORAS.has(entry.id)) nextEnabled.add(entry.id);
        }
        enabled = nextEnabled;
        seeded = true;
      }
      return { catalog, strengths: next, enabled, _seeded: seeded };
    }),
  setStrength: (id, value) =>
    set((s) => ({ strengths: { ...s.strengths, [id]: value } })),
  enable: (id) =>
    set((s) => {
      if (s.enabled.has(id)) return {} as Partial<LoraState>;
      const next = new Set(s.enabled);
      next.add(id);
      return { enabled: next };
    }),
  disable: (id) =>
    set((s) => {
      if (!s.enabled.has(id)) return {} as Partial<LoraState>;
      const next = new Set(s.enabled);
      next.delete(id);
      return { enabled: next };
    }),
  toggle: (id) =>
    set((s) => {
      const next = new Set(s.enabled);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return { enabled: next };
    }),
  reset: () =>
    set({ catalog: [], strengths: {}, enabled: new Set(), _seeded: false }),
}));
