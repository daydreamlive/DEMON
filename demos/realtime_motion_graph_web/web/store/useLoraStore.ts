"use client";

import { create } from "zustand";

import { getConfig } from "@/lib/config";
import {
  LORA_DEFAULT_STRENGTH_FRACTION,
  LORA_SLIDER_MAX,
} from "@/types/engine";
import type { LoraCatalogEntry } from "@/types/protocol";

// Server-driven LoRA catalog + per-id strength + enabled set. The catalog
// arrives via /api/loras (cheap filesystem scan, available before WS) and
// is updated mid-session via the WS "lora_catalog" frame.

// Hardcoded preferred-stems fallback used when the operator's
// config.json leaves engine.enabled_loras empty. If a preferred stem
// isn't in the catalog (different LoRA library locally), the slot
// falls back to the catalog entry at the same index, then to the next
// unclaimed entry — so a fresh page-load always lands with two LoRAs
// hot regardless of which files are on disk. One-shot: a later WS
// lora_catalog re-broadcast won't re-enable a LoRA the user has
// explicitly disabled.
const HARDCODED_PREFERRED_LORAS = ["deathstep", "synthpop"] as const;

/** Build the default strengths + default-enabled set against a given
 *  catalog. Pure: no store reads, no side effects. Used by both the
 *  initial setCatalog seeding path and the reset() path so the two
 *  agree on what "defaults" means. */
function seedFromCatalog(catalog: LoraCatalogEntry[]): {
  strengths: Record<string, number>;
  enabled: Set<string>;
} {
  const cfg = getConfig();
  const cfgStrength = cfg.controls.lora_default_strength;
  const fallbackStrength =
    typeof cfgStrength === "number" && cfgStrength > 0
      ? cfgStrength
      : LORA_DEFAULT_STRENGTH_FRACTION * LORA_SLIDER_MAX;
  const strengths: Record<string, number> = {};
  for (const entry of catalog) {
    strengths[entry.id] =
      typeof entry.strength === "number" && entry.strength > 0
        ? entry.strength
        : fallbackStrength;
  }
  const cfgPreferred = cfg.engine.enabled_loras;
  const preferredList: readonly string[] =
    cfgPreferred.length > 0 ? cfgPreferred : HARDCODED_PREFERRED_LORAS;
  const enabled = new Set<string>();
  const present = new Set(catalog.map((e) => e.id));
  const claimed = new Set<string>();
  for (let i = 0; i < preferredList.length; i++) {
    const preferred = preferredList[i];
    let pick: string | undefined;
    if (present.has(preferred) && !claimed.has(preferred)) {
      pick = preferred;
    } else {
      const slot = catalog[i]?.id;
      pick = slot && !claimed.has(slot)
        ? slot
        : catalog.find((e) => !claimed.has(e.id))?.id;
    }
    if (pick) {
      enabled.add(pick);
      claimed.add(pick);
    }
  }
  return { strengths, enabled };
}

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
      const cfg = getConfig();
      const cfgStrength = cfg.controls.lora_default_strength;
      const fallbackStrength =
        typeof cfgStrength === "number" && cfgStrength > 0
          ? cfgStrength
          : LORA_DEFAULT_STRENGTH_FRACTION * LORA_SLIDER_MAX;
      // Seed missing strengths from the server's reported defaults so a
      // freshly-arrived LoRA picks up its on-disk default. The Python
      // backend currently echoes 0.0 for every entry, which we treat as
      // "unset" and fall back to controls.lora_default_strength (from
      // config.json) so the slider lands at a useful initial level. A
      // genuine non-zero server default still wins.
      const next: Record<string, number> = { ...s.strengths };
      for (const entry of catalog) {
        if (!(entry.id in next)) {
          next[entry.id] =
            typeof entry.strength === "number" && entry.strength > 0
              ? entry.strength
              : fallbackStrength;
        }
      }
      // First populated catalog: flip on the preferred default LoRAs so the
      // demo plays with its intended sound out of the box. Preference
      // order is config.json's engine.enabled_loras, falling back to
      // HARDCODED_PREFERRED_LORAS when the config leaves it empty. If a
      // preferred stem isn't in the catalog, fall back to the catalog
      // entry at that slot index (with dedup so two missing defaults
      // don't both claim the first entry). Skipped on later re-broadcasts
      // so disabling a seeded LoRA sticks.
      let enabled = s.enabled;
      let seeded = s._seeded;
      if (!s._seeded && catalog.length > 0) {
        const cfgPreferred = cfg.engine.enabled_loras;
        const preferredList: readonly string[] =
          cfgPreferred.length > 0 ? cfgPreferred : HARDCODED_PREFERRED_LORAS;
        const nextEnabled = new Set(s.enabled);
        const present = new Set(catalog.map((e) => e.id));
        const claimed = new Set<string>();
        for (let i = 0; i < preferredList.length; i++) {
          const preferred = preferredList[i];
          let pick: string | undefined;
          if (present.has(preferred) && !claimed.has(preferred)) {
            pick = preferred;
          } else {
            const slot = catalog[i]?.id;
            pick = slot && !claimed.has(slot)
              ? slot
              : catalog.find((e) => !claimed.has(e.id))?.id;
          }
          if (pick) {
            nextEnabled.add(pick);
            claimed.add(pick);
          }
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
    set((s) => {
      // Keep the catalog — it's server-driven, not user state. Clearing
      // it would flip LibraryTile to its "no LoRAs found" empty state
      // until the next session start. Re-seed strengths + default-on
      // enabled set against the existing catalog, matching the initial
      // setCatalog seeding behaviour so "reset" actually means "back to
      // defaults", not "lose the catalog".
      const { strengths, enabled } = seedFromCatalog(s.catalog);
      return { strengths, enabled };
    }),
}));
