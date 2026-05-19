// Curve-based parameter scheduling types. A "curve" is a sequence of
// control points along the track timeline that drives a slider param's
// value as the audio plays. Drawn in ScheduleCurvesOverlay, applied by
// useScheduledCurves at rAF cadence.

/** Per-point interpolation mode. The curve renderer chooses how to
 *  bridge from THIS point to the NEXT one based on this point's mode:
 *  - smooth: Catmull-Rom spline (default, organic feel)
 *  - linear: straight line
 *  - step:   value held flat until the next point */
export type CurveInterpMode = "smooth" | "linear" | "step";

export interface CurvePoint {
  /** 0..1 along the track duration. Endpoints pinned at 0 and 1. */
  x: number;
  /** 0..1 normalised. Mapped to param min/max (always 0..meta.max)
   *  via `SLIDER_META[param]` at apply time. */
  y: number;
  mode: CurveInterpMode;
}

export interface CurveState {
  /** When false, the curve is drawn but doesn't drive the param —
   *  manual sliders / MIDI behave as if no curve existed. */
  enabled: boolean;
  /** Always ≥ 2 points; first.x === 0 and last.x === 1. */
  points: CurvePoint[];
}

/** The curated FIXED param set the schedule-curves overlay always
 *  exposes as tabs. LoRA strengths are dynamic (loaded from the
 *  engine's catalog at runtime) and added on top via
 *  loraCurveParam(id) — see ScheduleCurvesOverlay's tab build. */
export const SCHEDULEABLE_PARAMS = [
  "denoise",
  "hint_strength",
  "feedback",
  "shift",
] as const;

export type ScheduleableParam = (typeof SCHEDULEABLE_PARAMS)[number];

/** Build the curve-store key for a LoRA's strength param. Mirrors the
 *  `lora_str_<id>` convention used by SLIDER_META at runtime. */
export function loraCurveParam(id: string): string {
  return `lora_str_${id}`;
}

/** Max number of LoRA tabs in the schedule overlay. Capped because
 *  the tab strip is finite real estate and most users automate at
 *  most a couple of LoRAs. */
export const MAX_LORA_CURVES = 2;

/** Display label shown in the tab strip. CSS uppercases via
 *  text-transform; we store natural case so screen readers and tooltips
 *  read sensibly. Mirrors the labels in MainTile / EngineTile / HUD so
 *  users see the same names everywhere ("Remix strength", not the
 *  internal `denoise` key). */
export const SCHEDULEABLE_PARAM_LABEL: Record<ScheduleableParam, string> = {
  denoise: "Remix strength",
  hint_strength: "Structure strength",
  feedback: "Feedback",
  shift: "Shift",
};

/** Default curve: a flat line at midrange. Two points, both `smooth`. */
export function defaultCurveState(): CurveState {
  return {
    enabled: false,
    points: [
      { x: 0, y: 0.5, mode: "smooth" },
      { x: 1, y: 0.5, mode: "smooth" },
    ],
  };
}
