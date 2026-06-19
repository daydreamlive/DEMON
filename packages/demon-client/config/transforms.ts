// Pure transforms over the operator-defaults config — no DOM, no store, no
// fetch. These are the portable core every frontend shares: merge an
// override onto a base, lift the legacy flat shape, resolve the XL (5B)
// variant, resolve the duration-aware LoRA cap, and the preserve-unknown
// serialization seam (read drops nothing it doesn't model; write re-emits
// it untouched).

import { SWAP_SOURCE_MODES } from "./types";
import type {
  RtmgConfig,
  RtmgConfigEngine,
  RtmgWebConfig,
  SwapSourceMode,
} from "./types";

export function isSwapSourceMode(v: unknown): v is SwapSourceMode {
  return (SWAP_SOURCE_MODES as readonly string[]).includes(v as string);
}

// ── Preserve-unknown-on-write ──────────────────────────────────────────
//
// Forward-compat rule (see plan 01): a client reads a config, then writes
// it back, and must re-emit any TOP-LEVEL keys it read but does not model —
// untouched. Without this, the first non-web client to gain an export
// feature would silently drop `effects` / `curves` / `web` etc. authored by
// another frontend.
//
// Mechanism: `mergeConfig` stashes unknown top-level keys onto the result
// under a non-enumerable symbol, so they never leak through `{...cfg}`,
// `JSON.stringify(cfg)`, or a typed field — the re-emit is explicit, via
// `serializeConfig`. Web routes its Export through `serializeConfig` too
// (OperatorStrip), so an unknown key authored by another frontend survives a
// web load → export round-trip; web layers its own `inputs` / `tracks`
// extensions onto the serialized object out of band. A config exported with
// no unknown keys present is byte-identical to the legacy schema-only shape.

/** The schema's own top-level keys. Anything else a loaded config carries
 *  is "unknown" and travels through untouched on a write. */
export const KNOWN_TOP_LEVEL_KEYS: ReadonlySet<string> = new Set([
  "version",
  "engine",
  "prompts",
  "controls",
  "channel_ranges",
  "seed",
  "swap_source_mode",
  "web",
  "curves",
  "loop",
]);

const UNKNOWN = Symbol("rtmgConfigUnknownKeys");
type WithUnknown = { [UNKNOWN]?: Record<string, unknown> };

/** The unknown top-level keys a config carried in from its source, or an
 *  empty object. Read by `serializeConfig` and by anyone writing their own
 *  exporter (the neutral capture adapter re-emits these). */
export function getUnknownKeys(cfg: RtmgConfig): Record<string, unknown> {
  return (cfg as WithUnknown)[UNKNOWN] ?? {};
}

/** Attach a preserved-unknown bag to a config (non-enumerable, so it never
 *  rides through a spread or JSON.stringify). No-op for an empty bag. */
export function withUnknownKeys<T extends RtmgConfig>(
  cfg: T,
  unknown: Record<string, unknown>,
): T {
  if (Object.keys(unknown).length === 0) return cfg;
  Object.defineProperty(cfg, UNKNOWN, {
    value: unknown,
    enumerable: false,
    configurable: true,
    writable: true,
  });
  return cfg;
}

/** Render a config to a plain JSON-ready object, re-emitting any preserved
 *  unknown top-level keys alongside the schema fields. This is the
 *  "preserve-unknown on write" exit: a client that loaded a config it
 *  doesn't fully model serializes it back without losing keys.
 *
 *  Known fields win on a (by-construction impossible) name collision —
 *  unknown keys are, by definition, outside `KNOWN_TOP_LEVEL_KEYS`. */
export function serializeConfig(cfg: RtmgConfig): Record<string, unknown> {
  return { ...getUnknownKeys(cfg), ...cfg };
}

// ── Legacy flat-shape lift (pre-`web` configs) ─────────────────────────

/** Fields that lived at the top level before the `web` split and now
 *  belong under `web`. `channel_ranges` is intentionally excluded — it
 *  was top-level in the old flat shape and stays top-level (shared) in
 *  the new one, so it needs no lifting. */
const LEGACY_WEB_KEYS = [
  "effects",
  "audio",
  "reset_seconds",
  "denoise_session_gate",
  "restart_song_on_swap",
] as const;

/** Lift a pre-`web` flat config (effects/audio/... at the top level)
 *  into the current nested shape so older config.json files and exported
 *  sounds keep loading. Only fills `web` sub-fields the override doesn't
 *  already set under `web`; a config that already nests `web` is returned
 *  untouched. Operates on a shallow copy — never mutates input. */
export function normalizeConfigShape(
  raw: Partial<RtmgConfig> & Record<string, unknown>,
): Partial<RtmgConfig> {
  const hasLegacyTopLevel = LEGACY_WEB_KEYS.some((k) => k in raw);
  if (!hasLegacyTopLevel) return raw;
  const liftedWeb: Record<string, unknown> = { ...(raw.web ?? {}) };
  for (const k of LEGACY_WEB_KEYS) {
    if (k in raw && !(k in liftedWeb))
      liftedWeb[k] = (raw as Record<string, unknown>)[k];
  }
  const out: Record<string, unknown> = { ...raw, web: liftedWeb };
  for (const k of LEGACY_WEB_KEYS) delete out[k];
  return out as Partial<RtmgConfig>;
}

export function mergeConfig(
  base: RtmgConfig,
  rawOverride: Partial<RtmgConfig> | null | undefined,
): RtmgConfig {
  if (!rawOverride) return base;
  const override = normalizeConfigShape(
    rawOverride as Partial<RtmgConfig> & Record<string, unknown>,
  );
  const web: Partial<RtmgWebConfig> = override.web ?? {};
  const merged: RtmgConfig = {
    version:
      typeof override.version === "number" ? override.version : base.version,
    engine: { ...base.engine, ...(override.engine ?? {}) },
    prompts: { ...base.prompts, ...(override.prompts ?? {}) },
    controls: { ...base.controls, ...(override.controls ?? {}) },
    // Per-param shallow merge: an override entry replaces the matching
    // base entry whole (operator-supplied {min,max,reverse} must travel
    // together to be coherent). Unspecified params keep the bundled
    // default range.
    channel_ranges: {
      ...base.channel_ranges,
      ...(override.channel_ranges ?? {}),
    },
    seed: typeof override.seed === "number" ? override.seed : base.seed,
    swap_source_mode: isSwapSourceMode(override.swap_source_mode)
      ? override.swap_source_mode
      : base.swap_source_mode,
    web: {
      effects: { ...base.web.effects, ...(web.effects ?? {}) },
      audio: { ...base.web.audio, ...(web.audio ?? {}) },
      reset_seconds:
        typeof web.reset_seconds === "number"
          ? web.reset_seconds
          : base.web.reset_seconds,
      denoise_session_gate: {
        ...base.web.denoise_session_gate,
        ...(web.denoise_session_gate ?? {}),
      },
      restart_song_on_swap:
        typeof web.restart_song_on_swap === "boolean"
          ? web.restart_song_on_swap
          : base.web.restart_song_on_swap,
    },
    // Curves are operator-authored and only meaningful as a whole bag,
    // so the override entry replaces the base entry whole when present.
    // Absent override keeps whatever the base has (DEFAULT_CONFIG leaves
    // this undefined; stock pods fall through to localStorage hydration).
    ...(override.curves !== undefined
      ? { curves: override.curves }
      : base.curves !== undefined
        ? { curves: base.curves }
        : {}),
    // Loop region replaces whole when the import carries one; otherwise
    // keep whatever the base (the live config) holds.
    ...(override.loop !== undefined
      ? { loop: override.loop }
      : base.loop !== undefined
        ? { loop: base.loop }
        : {}),
  };

  // Preserve-unknown: union the base's already-preserved unknowns with any
  // unknown top-level keys this override carried. Scanning the *normalized*
  // override means lifted legacy keys (effects/audio/...) are already folded
  // into `web` and never mistaken for unknowns.
  const unknown: Record<string, unknown> = { ...getUnknownKeys(base) };
  for (const k of Object.keys(override)) {
    if (!KNOWN_TOP_LEVEL_KEYS.has(k)) {
      unknown[k] = (override as Record<string, unknown>)[k];
    }
  }
  return withUnknownKeys(merged, unknown);
}

/** Resolve the LoRA cap for a given source duration. Tiers (when
 *  present) take precedence: pick the smallest ``up_to_s`` that's ≥
 *  ``durationS``. When no tier matches (durationS larger than all
 *  thresholds, or tiers absent), fall back to
 *  ``engine.max_concurrent_loras``. ``null`` return = uncapped.
 *
 *  Passing ``durationS = 0`` (no source loaded yet) selects the most
 *  permissive tier — short-source assumptions hold at boot before the
 *  first session config arrives. Callers that want a conservative
 *  boot-time cap can pass the static fallback value directly. */
export function resolveLoraCapForSource(
  durationS: number,
  engine: Pick<
    RtmgConfigEngine,
    "max_concurrent_loras" | "max_concurrent_loras_tiers"
  >,
): number | null {
  const tiers = engine.max_concurrent_loras_tiers;
  if (tiers && tiers.length > 0) {
    // Sort by threshold ascending; pick the first tier whose ceiling
    // is ≥ durationS. Defensive sort so config-side order doesn't
    // matter to the runtime.
    const sorted = [...tiers]
      .filter(
        (t) => typeof t?.up_to_s === "number" && typeof t?.cap === "number",
      )
      .sort((a, b) => a.up_to_s - b.up_to_s);
    for (const tier of sorted) {
      if (durationS <= tier.up_to_s) return tier.cap;
    }
    // durationS exceeds every tier ceiling — fall through to the
    // static fallback. The fallback is intentionally separate from
    // the last-tier cap so an operator can express "anything past
    // 240s is uncapped" without changing the final explicit tier.
  }
  const fallback = engine.max_concurrent_loras;
  return typeof fallback === "number" && fallback > 0 ? fallback : null;
}

/** Collapse the dual-variant config into a single applied config, picking
 *  the XL (5B) sibling when the active checkpoint scale is "5B" and the
 *  sibling is defined. Any other scale (null, "2B", unknown) keeps the
 *  base values. Unspecified `_xl` siblings always fall through to base,
 *  so existing single-variant config.json files keep working unchanged.
 *
 *  Result keeps the `_xl` fields on the engine/prompts objects so an
 *  Import round-trip (mergeConfig over getConfig() then applyConfig)
 *  doesn't lose them. */
export function selectVariant(cfg: RtmgConfig, scale: string | null): RtmgConfig {
  if (scale !== "5B") return cfg;
  const e = cfg.engine;
  const p = cfg.prompts;
  // Spread drops the non-enumerable preserved-unknown carrier, so re-attach
  // it onto the resolved config (a 5B import round-trips losslessly too).
  return withUnknownKeys(
    {
      ...cfg,
      engine: {
        ...e,
        depth: e.depth_xl ?? e.depth,
        enabled_loras: e.enabled_loras_xl ?? e.enabled_loras,
      },
      prompts: {
        ...p,
        a: p.a_xl ?? p.a,
        b: p.b_xl ?? p.b,
        blend: typeof p.blend_xl === "number" ? p.blend_xl : p.blend,
      },
    },
    getUnknownKeys(cfg),
  );
}
