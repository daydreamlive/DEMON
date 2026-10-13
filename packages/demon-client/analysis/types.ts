// Wire shapes of the analysis seam: the `analysis_tap` command and the
// `analysis_state` / `analysis_update` events. The payloads are the
// generated wire types (types/wireContract.gen.ts, projected from
// protocol.py, the source of truth); this module names them for the
// analysis surface and adds the per-analyzer option typing the generic
// `options` record cannot carry.
//
// Times are canvas seconds, the same frame as `params.playback_pos`.

import type {
  AnalysisItem,
  AnalysisItemByKind,
  AnalysisStateEvent,
  AnalysisTapCommand,
  AnalysisUpdateEvent,
  NoteAnalysisItem,
} from "../types/wireContract.gen";

export type {
  AnalysisItem,
  AnalysisItemByKind,
  AnalysisStateEvent,
  AnalysisTapCommand,
  AnalysisUpdateEvent,
  NoteAnalysisItem,
} from "../types/wireContract.gen";

/** Wire `type` of the client -> server command. */
export const ANALYSIS_TAP_COMMAND = "analysis_tap" as const;

/** Server -> client event types this module owns. RemoteBackend
 *  re-dispatches each as a CustomEvent of the same name. */
export const ANALYSIS_EVENT_TYPES = ["analysis_state", "analysis_update"] as const;
export type AnalysisEventType = (typeof ANALYSIS_EVENT_TYPES)[number];

/** Options of the `notes` analyzer (MuScriptor). Every field is optional;
 *  the server fills defaults and rejects unknown keys through
 *  `analysis_state.error`. */
export interface NotesTapOptions {
  /** Model size, one of the analyzer's sizes (`small` by default). */
  size?: string;
  /** MuScriptor instrument group names (`acoustic_piano`, `drums`, ...). */
  instruments?: string[];
  /** Seconds ahead of the playhead that ring canvases are analysed, 4..30. */
  horizon_s?: number;
}

/** Per-analyzer options map. A new analyzer adds one entry. */
export interface AnalysisTapOptionsByTap {
  notes: NotesTapOptions;
}

/** Analyzer name. `notes` is the only one today; the server validates
 *  against its registry, so any string reaches the wire. */
export type AnalysisTapName = keyof AnalysisTapOptionsByTap | (string & {});

/** Options payload for `tap`: typed for known analyzers, open otherwise. */
export type AnalysisTapOptions<T extends string = string> =
  T extends keyof AnalysisTapOptionsByTap
    ? AnalysisTapOptionsByTap[T]
    : Record<string, unknown>;

export type AnalysisTapCommandPayload = AnalysisTapCommand;

/** Scheduling mode a tap reports. */
export type AnalysisMode = AnalysisStateEvent["mode"];

/** `provisional`: the region may still be rewritten. `settled`: append-only
 *  behind the write frontier, or a ring region the playhead passed whose
 *  audio has not changed since it was analysed. */
export type AnalysisBasis = AnalysisUpdateEvent["basis"];

/** Sent on enable, disable, mode change and error. */
export type AnalysisStateEventPayload = AnalysisStateEvent;

/** One transcribed note (item kind `note`). */
export type NoteItem = NoteAnalysisItem;

export type AnalysisItemKind = keyof AnalysisItemByKind;

/** Range replace: the receiver drops every item of `tap` whose `start_s`
 *  lies in [range_start_s, range_start_s + span_s) and inserts `items`.
 *  `clear` drops everything for the tap first (sent on enable and on swap
 *  with the new `source_epoch`). `playhead_s` is null when the server has
 *  no playhead report. */
export type AnalysisUpdateEventPayload = AnalysisUpdateEvent;

export type AnalysisEventPayload =
  | AnalysisStateEventPayload
  | AnalysisUpdateEventPayload;

/** Loop band in canvas seconds, `start_s < end_s`. */
export interface LoopBand {
  start_s: number;
  end_s: number;
}

/** Analyzer names the pod offers, from `ready.capabilities.analysis_taps`.
 *  Read defensively (older servers and replays omit it, and the generated
 *  ready event types the mask as a plain record). Empty when absent. */
export function analysisTapsAvailable(
  capabilities: Record<string, unknown> | null | undefined,
): string[] {
  const raw = capabilities?.analysis_taps;
  if (!Array.isArray(raw)) return [];
  return raw.filter((name): name is string => typeof name === "string");
}
