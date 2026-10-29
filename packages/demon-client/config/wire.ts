// The seam between the operator-defaults config and the live engine: the
// RtmgConfig → wire `SessionConfig` mapping (the highest-value extraction —
// it is what makes every frontend drive the engine identically) plus the
// neutral state adapters. These are PURE — they take plain snapshots and
// return plain patches/configs. Each frontend keeps its own thin wiring from
// its internal state (zustand for web, the param model for a native plugin)
// to these neutral shapes.

import type { SessionConfig } from "../types/protocol";
import {
  isDcwMode,
  isDcwWavelet,
  isLoopGridRes,
  isRcfgMode,
  isTimeSignature,
  DEFAULT_TIME_SIGNATURE,
  type DcwMode,
  type DcwWavelet,
  type LoopGridRes,
  type RcfgMode,
  type TimeSignature,
} from "./enums";
import { getUnknownKeys, selectVariant, withUnknownKeys } from "./transforms";
import type {
  EnabledLoraEntry,
  RtmgConfig,
  RtmgConfigControls,
  RtmgConfigCurves,
  RtmgConfigLoop,
} from "./types";

// ── RtmgConfig → wire SessionConfig ────────────────────────────────────

/** The per-Play runtime bits the config itself doesn't carry: which LoRAs
 *  are live (and at what strength), the current prompt text, the resolved
 *  fixture + stem mode, and the optional client id. The host gathers these
 *  from its live state at session-start. */
export interface SessionConfigRuntime {
  enabledLoras: string[];
  loraStrengths: Record<string, number>;
  promptA: string;
  promptB: string;
  fixtureName: string;
  /** stem_source_mode — omitted from the wire when null/undefined. */
  stemSourceMode?: string | null;
  useServerFixture: boolean;
  /** Opaque per-browser identifier — omitted when null/undefined. */
  clientId?: string | null;
  /** Checkpoint to serve this session (a `--checkpoint`-style name from
   *  server-info `checkpoints`). Omitted when null/undefined/empty, so the
   *  pod serves its default and the payload is byte-identical to before. */
  checkpoint?: string | null;
}

/** Handshake config plus the optional per-session `checkpoint` field.
 *  LOCAL TYPING: the server half adds `checkpoint` to the SessionConfig
 *  registry; once the regenerated wireContract.gen.ts carries it, this
 *  intersection collapses to plain `SessionConfig` and should be dropped. */
export type SessionConfigWithCheckpoint = SessionConfig & {
  checkpoint?: string | null;
};

/** Build the handshake `SessionConfig` from the operator config's engine
 *  block plus the live runtime bits. The single place the engine fields are
 *  mapped onto the wire, so every frontend negotiates an identical session.
 *
 *  `key` is intentionally not sent: the server's session-init resolver
 *  ignores config.key and uses sidecar.key for known fixtures (or
 *  CNN-detects on a miss); the result echoes back in `ready.key`. Sending
 *  the dropdown's stale value would re-introduce the override-wins regression. */
export function rtmgConfigToSessionConfig(
  cfg: RtmgConfig,
  runtime: SessionConfigRuntime,
): SessionConfigWithCheckpoint {
  const e = cfg.engine;
  return {
    telemetry_version: 1,
    sde: e.sde,
    lora: e.lora,
    depth: e.depth,
    vae_window: e.vae_window,
    crop: e.crop,
    steps: e.steps,
    fast_vae: e.fast_vae,
    walk_window: e.walk_window ?? false,
    walk_window_s: e.walk_window_s ?? 60,
    lead_floor_s: e.lead_floor_s,
    lead_ceiling_s: e.lead_ceiling_s,
    lead_release_tau_s: e.lead_release_tau_s,
    enabled_loras: runtime.enabledLoras,
    prompt: runtime.promptA,
    prompt_b: runtime.promptB,
    lora_strengths: runtime.loraStrengths,
    fixture_name: runtime.fixtureName,
    ...(runtime.stemSourceMode
      ? { stem_source_mode: runtime.stemSourceMode }
      : {}),
    use_server_fixture: runtime.useServerFixture,
    ...(runtime.clientId ? { client_id: runtime.clientId } : {}),
    ...(runtime.checkpoint ? { checkpoint: runtime.checkpoint } : {}),
  };
}

// ── Neutral apply adapter ──────────────────────────────────────────────

/** The deterministic state writes an applied config implies, computed
 *  purely (no store, no current-state reads). The host spreads these into
 *  its own state container. Fields that the config may leave invalid
 *  (`dcwEnabled` / `dcwMode` / `dcwWavelet` / `rcfgMode`) are OMITTED when
 *  invalid, so a host that merges the patch keeps its prior value — the same
 *  "fall back to current" semantics the web store had inline. */
export interface ConfigApplyPatch {
  /** The config after XL (5B) variant resolution. The host stores this as
   *  its active config (non-store fields + preserved-unknown carrier). */
  resolved: RtmgConfig;
  /** Numeric controls + `prompt_blend`, to seed slider value/target/default
   *  maps. */
  sliderUpdates: Record<string, number>;
  promptA: string;
  promptB: string;
  activeKey: string;
  activeTimeSignature: TimeSignature;
  seed: number;
  lufsOn: boolean;
  dcwEnabled?: boolean;
  dcwMode?: DcwMode;
  dcwWavelet?: DcwWavelet;
  rcfgMode?: RcfgMode;
  /** Validated loop patch — present only when the config carried a loop.
   *  `loopOn` is present only when the export carried `fullBuffer`. */
  loop?: {
    loopBand: { start: number; end: number } | null;
    bandLoopEnabled: boolean;
    loopGridRes: LoopGridRes;
    loopOn?: boolean;
  };
  /** Curve bag (points deep-cloned) — present only when the config carried
   *  curves. */
  curves?: RtmgConfigCurves;
}

/** Resolve a config against the active checkpoint scale and compute the
 *  deterministic state patch. The host applies the patch to its own state
 *  and owns the stateful side effects (LoRA seed/diff, engine sends,
 *  subscriber notifications) that can't be expressed as a pure shape. */
export function applyConfigToState(
  cfg: RtmgConfig,
  scale: string | null,
): ConfigApplyPatch {
  const resolved = selectVariant(cfg, scale);

  const sliderUpdates: Record<string, number> = {};
  for (const [k, v] of Object.entries(resolved.controls)) {
    if (typeof v === "number") sliderUpdates[k] = v;
  }
  sliderUpdates.prompt_blend = resolved.prompts.blend;

  const patch: ConfigApplyPatch = {
    resolved,
    sliderUpdates,
    promptA: resolved.prompts.a,
    promptB: resolved.prompts.b,
    activeKey: resolved.engine.key,
    activeTimeSignature: isTimeSignature(resolved.engine.time_signature)
      ? resolved.engine.time_signature
      : DEFAULT_TIME_SIGNATURE,
    seed: resolved.seed,
    lufsOn: resolved.web.audio.lufs_enabled,
  };
  if (typeof resolved.controls.dcw_enabled === "boolean") {
    patch.dcwEnabled = resolved.controls.dcw_enabled;
  }
  if (isDcwMode(resolved.controls.dcw_mode)) {
    patch.dcwMode = resolved.controls.dcw_mode;
  }
  if (isDcwWavelet(resolved.controls.dcw_wavelet)) {
    patch.dcwWavelet = resolved.controls.dcw_wavelet;
  }
  if (isRcfgMode(resolved.controls.rcfg_mode)) {
    patch.rcfgMode = resolved.controls.rcfg_mode;
  }

  // Loop region: validate the operator-editable band (a malformed band → no
  // loop). Only carry the global full-buffer toggle when the export had it.
  if (resolved.loop) {
    const lp = resolved.loop;
    const validBand =
      lp.band &&
      typeof lp.band.start === "number" &&
      typeof lp.band.end === "number" &&
      lp.band.end > lp.band.start
        ? { start: lp.band.start, end: lp.band.end }
        : null;
    patch.loop = {
      loopBand: validBand,
      bandLoopEnabled: typeof lp.enabled === "boolean" ? lp.enabled : true,
      loopGridRes: isLoopGridRes(lp.grid) ? lp.grid : "beat",
    };
    if (typeof lp.fullBuffer === "boolean") patch.loop.loopOn = lp.fullBuffer;
  }

  // Curves: deep-clone the points so later host edits don't mutate the
  // active config snapshot.
  if (resolved.curves) {
    patch.curves = {
      scheduleEnabled: resolved.curves.scheduleEnabled,
      curves: Object.fromEntries(
        Object.entries(resolved.curves.curves).map(([param, curve]) => [
          param,
          {
            enabled: curve.enabled,
            points: curve.points.map((p) => ({ x: p.x, y: p.y, mode: p.mode })),
          },
        ]),
      ),
    };
  }

  return patch;
}

// ── Neutral capture adapter ────────────────────────────────────────────

/** A neutral snapshot of the live, user-editable state a frontend captures
 *  into an `RtmgConfig`. The host reads these from wherever it keeps them
 *  (the web store, a plugin's param model); the non-store fields
 *  (channel_ranges, swap_source_mode, the non-numeric engine.* config, the
 *  whole `web` block) are sourced from the active config passed as `base`. */
export interface ConfigStateSnapshot {
  /** Numeric slider targets plus the non-numeric DCW controls and
   *  lora_default_strength — the full `controls` bag to emit. */
  controls: RtmgConfigControls;
  promptA: string;
  promptB: string;
  promptBlend: number;
  key: string;
  timeSignature: string;
  seed: number;
  enabledLoras: EnabledLoraEntry[];
  lufsOn: boolean;
  loop: RtmgConfigLoop;
  curves: RtmgConfigCurves;
}

/** Assemble an `RtmgConfig` from a live state snapshot and the active config
 *  (`base`) — the inverse of `applyConfigToState`. Re-emits `base`'s
 *  preserved-unknown top-level keys so a config loaded with keys this client
 *  doesn't model round-trips untouched (preserve-unknown on write). */
export function captureConfigFromState(
  snapshot: ConfigStateSnapshot,
  base: RtmgConfig,
): RtmgConfig {
  return withUnknownKeys(
    {
      version: base.version,
      engine: {
        ...base.engine,
        key: snapshot.key,
        time_signature: snapshot.timeSignature,
        enabled_loras: snapshot.enabledLoras,
      },
      prompts: {
        a: snapshot.promptA,
        b: snapshot.promptB,
        blend: snapshot.promptBlend,
      },
      controls: snapshot.controls,
      channel_ranges: base.channel_ranges,
      seed: snapshot.seed,
      swap_source_mode: base.swap_source_mode,
      web: {
        effects: base.web.effects,
        audio: { ...base.web.audio, lufs_enabled: snapshot.lufsOn },
        reset_seconds: base.web.reset_seconds,
        denoise_session_gate: base.web.denoise_session_gate,
        restart_song_on_swap: base.web.restart_song_on_swap,
      },
      loop: snapshot.loop,
      curves: snapshot.curves,
    },
    getUnknownKeys(base),
  );
}
