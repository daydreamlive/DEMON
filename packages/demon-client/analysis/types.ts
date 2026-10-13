// Wire shapes of the analysis seam: the `analysis_tap` command and the
// `analysis_state` / `analysis_update` events. Written by hand from the
// pinned wire contract (protocol.py is the source of truth); they mirror
// the generated AnalysisTapCommand / AnalysisStateEvent /
// AnalysisUpdateEvent field for field so either name can be used.
//
// Times are canvas seconds, the same frame as `params.playback_pos`.

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

export interface AnalysisTapCommandPayload {
  type: typeof ANALYSIS_TAP_COMMAND;
  tap: string;
  enabled: boolean;
  options?: Record<string, unknown>;
}

/** Scheduling mode a tap reports. */
export type AnalysisMode =
  | "lookahead"
  | "confirm"
  | "no_playhead"
  | "backoff"
  | "off";

/** `provisional`: the region may still be rewritten. `settled`: append-only
 *  behind the write frontier, or a ring region the playhead passed whose
 *  audio has not changed since it was analysed. */
export type AnalysisBasis = "provisional" | "settled";

/** Sent on enable, disable, mode change and error. */
export interface AnalysisStateEventPayload {
  type: "analysis_state";
  tap: string;
  enabled: boolean;
  engine: string;
  size: string;
  /** Licence terms of the analyzer's weights; clients show it. */
  licence: string;
  mode: AnalysisMode;
  /** User-facing failure, or null. */
  error: string | null;
  /** Cumulative live requests withdrawn or dropped by this tap. */
  dropped: number;
}

/** One transcribed note (item kind `note`). */
export interface NoteItem {
  start_s: number;
  end_s: number;
  /** MIDI pitch 0..127. */
  pitch: number;
  /** MuScriptor instrument group name (`drums` for percussion). */
  instrument: string;
}

/** Item payload by `kind`. A new kind is one entry here and one in
 *  protocol.py. */
export interface AnalysisItemByKind {
  note: NoteItem;
}
export type AnalysisItemKind = keyof AnalysisItemByKind;
export type AnalysisItem = AnalysisItemByKind[AnalysisItemKind];

/** Range replace: the receiver drops every item of `tap` whose `start_s`
 *  lies in [range_start_s, range_start_s + span_s) and inserts `items`.
 *  `clear` drops everything for the tap first (sent on enable and on swap
 *  with the new `source_epoch`). */
export interface AnalysisUpdateEventPayload<
  K extends AnalysisItemKind = AnalysisItemKind,
> {
  type: "analysis_update";
  tap: string;
  /** Monotonic per session. */
  seq: number;
  source_epoch: number;
  range_start_s: number;
  span_s: number;
  basis: AnalysisBasis;
  /** The playhead had already passed `range_start_s` in play order. */
  late: boolean;
  clear: boolean;
  kind: K;
  items: AnalysisItemByKind[K][];
  /** Server playhead when the update was published. */
  playhead_s: number;
  /** Inference wall time of the analysis, seconds. */
  wall_s: number;
}

export type AnalysisEventPayload =
  | AnalysisStateEventPayload
  | AnalysisUpdateEventPayload;

/** Loop band in canvas seconds, `start_s < end_s`. */
export interface LoopBand {
  start_s: number;
  end_s: number;
}

/** True when `msg` is one of the analysis events (by `type` only; field
 *  shapes are the server's contract). */
export function isAnalysisEvent(msg: unknown): msg is AnalysisEventPayload {
  if (typeof msg !== "object" || msg === null) return false;
  const t = (msg as { type?: unknown }).type;
  return t === "analysis_state" || t === "analysis_update";
}

/** Analyzer names the pod offers, from `ready.capabilities.analysis_taps`.
 *  The capability mask is typed as booleans for the generator bits; this
 *  one is a list, so it is read defensively. Empty when absent. */
export function analysisTapsAvailable(
  capabilities: Record<string, unknown> | null | undefined,
): string[] {
  const raw = capabilities?.analysis_taps;
  if (!Array.isArray(raw)) return [];
  return raw.filter((name): name is string => typeof name === "string");
}
