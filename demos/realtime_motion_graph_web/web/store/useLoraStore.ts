"use client";

import { create } from "zustand";

import { getConfig } from "@/lib/config";
import {
  containsTrigger,
  getLoraTrigger,
  withTrigger,
  withoutTrigger,
} from "@/lib/loraTriggers";
import {
  LORA_DEFAULT_STRENGTH_FRACTION,
  LORA_SLIDER_MAX,
} from "@/types/engine";
import type { LoraCatalogEntry } from "@/types/protocol";

import { usePerformanceStore } from "./usePerformanceStore";

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
  /** Trigger words this store has auto-prepended to promptA/B as a side
   *  effect of LoRA toggles. Used to decide whether to strip a trigger
   *  on toggle-off — if the user typed the trigger manually, it's NOT
   *  in this set, so disabling the LoRA leaves the user's text alone.
   *  Lower-case for case-insensitive set semantics. */
  _autoTriggers: Set<string>;

  setCatalog: (catalog: LoraCatalogEntry[]) => void;
  setStrength: (id: string, value: number) => void;
  enable: (id: string) => void;
  disable: (id: string) => void;
  toggle: (id: string) => void;
  reset: () => void;
}

/** Side-effect helper — auto-prepend a LoRA's trigger to both prompts
 *  if it's registered in lib/loraTriggers and not already present. Idem-
 *  potent: re-calling for the same LoRA is a no-op. Updates the local
 *  `_autoTriggers` set so we know which triggers we own and can later
 *  strip cleanly. Returns the next set to fold into the LoRA store
 *  state (the perf store mutation is fire-and-forget). */
function applyTriggerOnEnable(
  id: string,
  prevAuto: Set<string>,
): Set<string> {
  const t = getLoraTrigger(id);
  if (!t) return prevAuto;
  const trigger = t.trigger.toLowerCase();
  const perf = usePerformanceStore.getState();
  let touched = false;
  if (!containsTrigger(perf.promptA, t.trigger)) {
    perf.setPromptA(withTrigger(perf.promptA, t.trigger));
    touched = true;
  }
  if (!containsTrigger(perf.promptB, t.trigger)) {
    perf.setPromptB(withTrigger(perf.promptB, t.trigger));
    touched = true;
  }
  if (!touched && prevAuto.has(trigger)) return prevAuto;
  const next = new Set(prevAuto);
  next.add(trigger);
  return next;
}

/** Side-effect helper — strip a LoRA's auto-injected trigger from both
 *  prompts when we own the injection. If the trigger isn't in
 *  `_autoTriggers` (e.g. the user typed it manually) we leave both
 *  prompts untouched. */
function stripTriggerOnDisable(
  id: string,
  prevAuto: Set<string>,
): Set<string> {
  const t = getLoraTrigger(id);
  if (!t) return prevAuto;
  const trigger = t.trigger.toLowerCase();
  if (!prevAuto.has(trigger)) return prevAuto;
  const perf = usePerformanceStore.getState();
  perf.setPromptA(withoutTrigger(perf.promptA, t.trigger));
  perf.setPromptB(withoutTrigger(perf.promptB, t.trigger));
  const next = new Set(prevAuto);
  next.delete(trigger);
  return next;
}

export const useLoraStore = create<LoraState>((set) => ({
  catalog: [],
  strengths: {},
  enabled: new Set(),
  _seeded: false,
  _autoTriggers: new Set<string>(),

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
      // For every LoRA enabled by the initial seed, auto-prepend its
      // trigger to both prompts so the default-on demo actually
      // activates the style. Idempotent against the user's default
      // promptA/B (e.g. `afxdump` is already hard-coded in the
      // default promptA → withTrigger no-ops; the deathstep slot is
      // covered without doubling). Skipped for LoRAs absent from
      // loraTriggers.ts (e.g. synthpop).
      let autoTriggers = s._autoTriggers;
      if (seeded && !s._seeded) {
        for (const id of enabled) {
          autoTriggers = applyTriggerOnEnable(id, autoTriggers);
        }
      }
      return {
        catalog,
        strengths: next,
        enabled,
        _seeded: seeded,
        _autoTriggers: autoTriggers,
      };
    }),
  setStrength: (id, value) =>
    set((s) => ({ strengths: { ...s.strengths, [id]: value } })),
  enable: (id) =>
    set((s) => {
      if (s.enabled.has(id)) return {} as Partial<LoraState>;
      const next = new Set(s.enabled);
      next.add(id);
      const autoTriggers = applyTriggerOnEnable(id, s._autoTriggers);
      return { enabled: next, _autoTriggers: autoTriggers };
    }),
  disable: (id) =>
    set((s) => {
      if (!s.enabled.has(id)) return {} as Partial<LoraState>;
      const next = new Set(s.enabled);
      next.delete(id);
      const autoTriggers = stripTriggerOnDisable(id, s._autoTriggers);
      return { enabled: next, _autoTriggers: autoTriggers };
    }),
  toggle: (id) =>
    set((s) => {
      const next = new Set(s.enabled);
      let autoTriggers: Set<string>;
      if (next.has(id)) {
        next.delete(id);
        autoTriggers = stripTriggerOnDisable(id, s._autoTriggers);
      } else {
        next.add(id);
        autoTriggers = applyTriggerOnEnable(id, s._autoTriggers);
      }
      return { enabled: next, _autoTriggers: autoTriggers };
    }),
  reset: () =>
    set((s) => {
      // Keep the catalog — it's server-driven, not user state. Clearing
      // it would flip LibraryTile to its "no LoRAs found" empty state
      // until the next session start. Re-seed strengths + default-on
      // enabled set against the existing catalog, matching the initial
      // setCatalog seeding behaviour so "reset" actually means "back to
      // defaults", not "lose the catalog".
      //
      // Strip every auto-injected trigger from the prompts before
      // re-seeding so a fresh "reset" doesn't carry stale trigger
      // residue from the previous enabled-set; the seed pass below
      // re-applies whichever triggers belong to the default-on LoRAs.
      let autoTriggers = s._autoTriggers;
      for (const id of s.enabled) {
        autoTriggers = stripTriggerOnDisable(id, autoTriggers);
      }
      const { strengths, enabled } = seedFromCatalog(s.catalog);
      for (const id of enabled) {
        autoTriggers = applyTriggerOnEnable(id, autoTriggers);
      }
      return { strengths, enabled, _autoTriggers: autoTriggers };
    }),
}));
