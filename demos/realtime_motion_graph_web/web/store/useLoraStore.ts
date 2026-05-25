"use client";

import { create } from "zustand";

import { getConfig, isConfigApplied } from "@/lib/config";
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

/** Build a lowercased alias→id lookup from the catalog. Each entry
 *  contributes its filename stem (`entry.id`) and, when a metadata
 *  sidecar is present, its display name (`entry.metadata.name`). Used
 *  to resolve config references that may be written as either form
 *  (`"deep_house-v1"` or `"Deep House"`). Lookup is case-insensitive;
 *  the returned value is always the canonical id. */
function buildLoraAliasMap(catalog: LoraCatalogEntry[]): Map<string, string> {
  const aliases = new Map<string, string>();
  for (const entry of catalog) {
    aliases.set(entry.id.toLowerCase(), entry.id);
    const name = entry.metadata?.name;
    if (name) aliases.set(name.toLowerCase(), entry.id);
  }
  return aliases;
}

/** Resolve the initial strength for a catalog entry.
 *
 *  Priority (highest first):
 *    1. Server-reported live strength (`entry.strength > 0`) — set when
 *       the LoRA is already enabled mid-session.
 *    2. Inline strength on the `enabled_loras` entry (object form
 *       `{ name, strength }`) — the operator's escape hatch for sidecar
 *       values that don't suit a given checkpoint.
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

  // Alias resolution: `enabled_loras` entries reference LoRAs by either
  // filename stem or sidecar display name, case-insensitively. Build
  // from `compatibleCatalog` (not the full catalog) so a display name
  // shared across scale variants — e.g. both `ambient-v1` (2B) and
  // `ambient-xl-v1` (5B) carry `metadata.name = "Ambient"` — resolves
  // to the entry matching the active checkpoint. The full-catalog map
  // suffered last-write-wins: xl entries register after turbo, so
  // "Ambient" would always resolve to `ambient-xl-v1`, and slot
  // fallback would fire on 2B sessions.
  const aliases = buildLoraAliasMap(compatibleCatalog);

  // Parse `enabled_loras` entries into (a) a list of names to resolve
  // in the auto-enable loop and (b) a per-id strength override map for
  // any object-form entries that specify a `strength`. Bare strings
  // contribute only to (a) and fall through to sidecar
  // recommended_strength.
  const preferredNames: string[] = [];
  const overrides: Record<string, number> = {};
  for (const entry of cfg.engine.enabled_loras) {
    if (typeof entry === "string") {
      preferredNames.push(entry);
    } else {
      preferredNames.push(entry.name);
      if (typeof entry.strength === "number") {
        const id = aliases.get(entry.name.toLowerCase());
        if (id) overrides[id] = entry.strength;
      }
    }
  }

  const strengths: Record<string, number> = {};
  for (const entry of catalog) {
    strengths[entry.id] = resolveDefaultStrength(
      entry,
      fallbackStrength,
      overrides,
    );
  }
  const preferredList: readonly string[] =
    preferredNames.length > 0 ? preferredNames : HARDCODED_PREFERRED_LORAS;
  const enabled = new Set<string>();
  const present = new Set(compatibleCatalog.map((e) => e.id));
  const claimed = new Set<string>();
  for (let i = 0; i < preferredList.length; i++) {
    const preferred = preferredList[i];
    // Resolve display name or stem to canonical id. Falls through to
    // the original string on miss so the slot-fallback path below
    // still fires (matches pre-alias behavior).
    const resolved = aliases.get(preferred.toLowerCase()) ?? preferred;
    let pick: string | undefined;
    if (present.has(resolved) && !claimed.has(resolved)) {
      pick = resolved;
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

/** Time (ms) the user-driven LoRA strength slider stays locked after a
 *  commit gesture. Picked from the empirical refit cost on a fleet pod:
 *  every observed ``Refitted 192 weights`` log entry sat at 265-295 ms;
 *  400 ms gives ~33 % headroom over the worst observed case without
 *  feeling unresponsive on a fast refit. A faster GPU / smaller LoRA
 *  resolves quicker than this and the slider just sits locked for the
 *  spare time — acceptable trade for never-misleading UX.
 *
 *  Ryan flagged in design discussion: "the refit time is variable, and
 *  rapid updates will not behave as users expect — it is variable but
 *  certain to be not real time. the current UX is misleading." This
 *  lock is the user-facing honest-time contract. */
const REFIT_LOCK_MS = 400;

interface LoraState {
  catalog: LoraCatalogEntry[];
  /** Per-id strength (0..LORA_SLIDER_MAX). */
  strengths: Record<string, number>;
  /** Set of enabled LoRA ids. */
  enabled: Set<string>;
  /** Whether default-on LoRAs have already been seeded for this session. */
  seeded: boolean;
  /** LoRA ids whose strength slider is currently locked while the
   *  server-side refit settles after a user commit gesture. Populated
   *  by ``markPendingRefit`` (typically from the slider's pointerup
   *  handler) and auto-cleared after ``REFIT_LOCK_MS`` via an internal
   *  timer. UI reads this to disable input on the locked rows so users
   *  can't queue multiple rapid commits that the server's refit
   *  pipeline can't service in real time. */
  pendingRefit: Set<string>;

  setCatalog: (catalog: LoraCatalogEntry[]) => void;
  setStrength: (id: string, value: number) => void;
  enable: (id: string) => void;
  disable: (id: string) => void;
  toggle: (id: string) => void;
  reset: () => void;
  /** Lock a LoRA's slider for ``REFIT_LOCK_MS`` after a user-direct
   *  commit gesture. Calling again while still pending resets the
   *  timer (the new commit gets its own full lock window). Idempotent
   *  for already-pending ids — only the timer is bumped. */
  markPendingRefit: (id: string) => void;
  /** Force-clear the lock for an id. The timer auto-clears too — this
   *  is for callers that want to release early (e.g. the LoRA was
   *  disabled while pending, or a session reset). */
  clearPendingRefit: (id: string) => void;
}

// Module-scoped timer map. Keeping the timer IDs OUT of zustand state
// avoids triggering subscribers on every mark/clear pair, and lets us
// reset a pending timer without touching state for "still pending"
// commits. The map is cleared by ``clearPendingRefit`` AND by the
// timer's own expiry handler — whichever fires first.
const _refitTimers = new Map<string, ReturnType<typeof setTimeout>>();

export const useLoraStore = create<LoraState>((set, get) => ({
  catalog: [],
  strengths: {},
  enabled: new Set(),
  seeded: false,
  pendingRefit: new Set(),

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
      //
      // Also gate on isConfigApplied(): in LOCAL_MODE, LibraryTile fires
      // its own /api/loras at mount, racing RTMGBoot's parallel fetches.
      // If the catalog lands before applyConfig(), seeding here would
      // read DEFAULT_CONFIG.engine.enabled_loras = [], fall through to
      // the count-rule, and pick whichever LoRAs sort first. Stash the
      // catalog and let applyConfig() retrigger this method once the
      // config is in place.
      const shouldSeed =
        !s.seeded && catalog.length > 0 && isConfigApplied();
      const enabled = shouldSeed
        ? new Set<string>([...s.enabled, ...fresh.enabled])
        : s.enabled;
      // Note: seeding a LoRA does NOT mutate promptA/promptB. Trigger
      // words are injected onto the WS `prompt` message at send-time by
      // RemoteBackend.sendPrompt (via enabledLoraTriggerPrefix) — the
      // Tags A/B textareas stay the operator's clean prompt text.
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
      // The LoRA's trigger word is NOT written into promptA/promptB.
      // It's injected onto the WS `prompt` message at send-time by
      // RemoteBackend.sendPrompt (via enabledLoraTriggerPrefix), so
      // the Tags A/B textareas stay the operator's clean prompt text.
      return { enabled: next };
    }),
  disable: (id) =>
    set((s) => {
      if (!s.enabled.has(id)) return {} as Partial<LoraState>;
      const next = new Set(s.enabled);
      next.delete(id);
      // Symmetric with enable(): no prompt mutation. Dropping the LoRA
      // from `enabled` is enough — the next sendPrompt rebuilds the
      // wire-side trigger prefix without it.
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
      // Cancel any in-flight refit locks — the value the user was
      // committing is being replaced. Leaving them set would block
      // the freshly-seeded sliders for no reason.
      for (const t of _refitTimers.values()) clearTimeout(t);
      _refitTimers.clear();
      return { catalog, strengths, enabled, pendingRefit: new Set() };
    }),
  markPendingRefit: (id) => {
    // Reset any in-flight timer for this id BEFORE we issue the new
    // one — a back-to-back commit gets its own full lock window
    // rather than expiring on the prior commit's clock.
    const existing = _refitTimers.get(id);
    if (existing !== undefined) clearTimeout(existing);
    const handle = setTimeout(() => {
      _refitTimers.delete(id);
      // Read get() at fire time so the timer survives store re-creates
      // in HMR; idempotent if the id was already cleared.
      get().clearPendingRefit(id);
    }, REFIT_LOCK_MS);
    _refitTimers.set(id, handle);
    set((s) => {
      if (s.pendingRefit.has(id)) {
        // Already locked → only the timer needed bumping; don't churn
        // subscribers with a no-op set.
        return {} as Partial<LoraState>;
      }
      const next = new Set(s.pendingRefit);
      next.add(id);
      return { pendingRefit: next };
    });
  },
  clearPendingRefit: (id) => {
    const existing = _refitTimers.get(id);
    if (existing !== undefined) {
      clearTimeout(existing);
      _refitTimers.delete(id);
    }
    set((s) => {
      if (!s.pendingRefit.has(id)) return {} as Partial<LoraState>;
      const next = new Set(s.pendingRefit);
      next.delete(id);
      return { pendingRefit: next };
    });
  },
}));
