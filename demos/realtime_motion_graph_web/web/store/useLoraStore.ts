"use client";

import { create } from "zustand";

import { getConfig } from "@/lib/config";
import { useSessionStore } from "@/store/useSessionStore";
import {
  LORA_DEFAULT_STRENGTH_FRACTION,
  LORA_SLIDER_MAX,
} from "@/types/engine";
import type { LoraCatalogEntry } from "@/types/protocol";

/** True iff a LoRA's trained ``base_model_scale`` is compatible with
 *  the active session's checkpoint scale. Unknown values on EITHER
 *  side return true so we don't accidentally hide LoRAs we can't
 *  classify against an undocumented checkpoint, or vice versa.
 *  Exported so the LibraryTile can reuse the same predicate for the
 *  visual filter. */
export function isLoraCompatibleWithScale(
  entry: LoraCatalogEntry,
  sessionScale: string | null,
): boolean {
  if (sessionScale === null || sessionScale === undefined) return true;
  const loraScale = entry.metadata?.base_model_scale;
  if (!loraScale) return true;
  return loraScale === sessionScale;
}

/** Stable sort that floats sidecar-backed LoRAs to the top and sinks
 *  bare entries (no metadata.json, no .trigger.txt) to the bottom.
 *  Tiebreaker is the LoRA id so the order is deterministic across
 *  catalog refreshes. Pure: doesn't mutate the input array. */
function sortCatalogForDisplay(
  catalog: LoraCatalogEntry[],
): LoraCatalogEntry[] {
  return [...catalog].sort((a, b) => {
    const ha = a.metadata?.has_metadata ?? false;
    const hb = b.metadata?.has_metadata ?? false;
    if (ha !== hb) return ha ? -1 : 1;
    return a.id.localeCompare(b.id);
  });
}

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

/** Resolve the initial strength for a catalog entry.
 *
 *  Priority (highest first):
 *    1. Server-reported live strength (`entry.strength > 0`) — set when
 *       the LoRA is already enabled mid-session.
 *    2. Per-LoRA override in `engine.lora_strength_overrides[id]` — the
 *       operator's escape hatch for sidecar values that don't suit a
 *       given checkpoint.
 *    3. Sidecar `recommended_strength` from the metadata loader.
 *    4. `controls.lora_default_strength` from the config.
 *    5. Hardcoded `LORA_DEFAULT_STRENGTH_FRACTION * LORA_SLIDER_MAX`.
 */
function resolveDefaultStrength(
  entry: LoraCatalogEntry,
  fallbackStrength: number,
  overrides: Record<string, number>,
): number {
  if (typeof entry.strength === "number" && entry.strength > 0) {
    return entry.strength;
  }
  const override = overrides[entry.id];
  if (typeof override === "number" && override > 0) {
    return override;
  }
  const recommended = entry.metadata?.recommended_strength;
  if (typeof recommended === "number" && recommended > 0) {
    return recommended;
  }
  return fallbackStrength;
}

/** Build the default strengths + default-enabled set against a given
 *  catalog. Pure: no side effects on the store. Reads getConfig() and
 *  the current session scale, both of which are external invariants by
 *  the time setCatalog/reset run.
 *
 *  Used by both the initial setCatalog seeding path and the reset()
 *  path so the two agree on what "defaults" means — the only
 *  difference is that setCatalog *merges* this into the existing user
 *  state while reset() *replaces*. */
function computeSeed(catalog: LoraCatalogEntry[]): {
  strengths: Record<string, number>;
  enabled: Set<string>;
} {
  const cfg = getConfig();
  const cfgStrength = cfg.controls.lora_default_strength;
  const fallbackStrength =
    typeof cfgStrength === "number" && cfgStrength > 0
      ? cfgStrength
      : LORA_DEFAULT_STRENGTH_FRACTION * LORA_SLIDER_MAX;
  const overrides = cfg.engine.lora_strength_overrides ?? {};
  const strengths: Record<string, number> = {};
  for (const entry of catalog) {
    strengths[entry.id] = resolveDefaultStrength(
      entry,
      fallbackStrength,
      overrides,
    );
  }
  // Pick the auto-enable set from the COMPATIBLE subset of the catalog
  // so a 5B LoRA listed in enabled_loras can't auto-enable on a 2B
  // session (would never apply at the engine level, and the visible-
  // prepend would inject a useless trigger into the prompt). The
  // session scale comes from useSessionStore.checkpointScale, which
  // listLoras() populates from /api/loras and the WS ready frame
  // refreshes. Unknown scale = no filtering.
  const sessionScale = useSessionStore.getState().checkpointScale;
  const compatibleCatalog = catalog.filter((e) =>
    isLoraCompatibleWithScale(e, sessionScale),
  );
  const cfgPreferred = cfg.engine.enabled_loras;
  const preferredList: readonly string[] =
    cfgPreferred.length > 0 ? cfgPreferred : HARDCODED_PREFERRED_LORAS;
  const enabled = new Set<string>();
  const present = new Set(compatibleCatalog.map((e) => e.id));
  const claimed = new Set<string>();
  for (let i = 0; i < preferredList.length; i++) {
    const preferred = preferredList[i];
    let pick: string | undefined;
    if (present.has(preferred) && !claimed.has(preferred)) {
      pick = preferred;
    } else {
      const slot = compatibleCatalog[i]?.id;
      pick = slot && !claimed.has(slot)
        ? slot
        : compatibleCatalog.find((e) => !claimed.has(e.id))?.id;
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
  seeded: boolean;

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
  seeded: false,

  setCatalog: (incomingCatalog) =>
    set((s) => {
      // Float metadata-backed LoRAs to the top so the operator's eye
      // lands on the documented ones; bare-stem LoRAs sink to the
      // bottom. Stable, deterministic across re-broadcasts.
      const catalog = sortCatalogForDisplay(incomingCatalog);
      const fresh = computeSeed(catalog);
      // Strength merge: fresh defaults first, then existing values
      // overwrite — so a user-edited strength survives a mid-session
      // catalog re-broadcast, and newly-appearing entries pick up
      // their default from the priority chain.
      const strengths = { ...fresh.strengths, ...s.strengths };
      // Auto-enable only once per session. The first populated catalog
      // flips on the preferred default LoRAs (config.engine.enabled_loras
      // → HARDCODED_PREFERRED_LORAS fallback, with index-slot dedup);
      // later re-broadcasts must not resurrect a user-disabled LoRA.
      const shouldSeed = !s.seeded && catalog.length > 0;
      const enabled = shouldSeed
        ? new Set<string>([...s.enabled, ...fresh.enabled])
        : s.enabled;
      return {
        catalog,
        strengths,
        enabled,
        seeded: s.seeded || shouldSeed,
      };
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
      const catalog = sortCatalogForDisplay(s.catalog);
      const { strengths, enabled } = computeSeed(catalog);
      return { catalog, strengths, enabled };
    }),
}));
