// Snap resolutions for the playback loop region ("brace"). Declared once
// here and shared by the editor (WaveformScrubBox), the performance store,
// and config import/export so the set never drifts out of sync between the
// three. Coarse→fine.

export type LoopGridRes = "bar" | "half" | "beat" | "eighth";

export const LOOP_GRID_ORDER: LoopGridRes[] = ["bar", "half", "beat", "eighth"];

export const LOOP_GRID_LABEL: Record<LoopGridRes, string> = {
  bar: "Bar",
  half: "½ Bar",
  beat: "Beat",
  eighth: "⅛",
};

/** Runtime guard — config JSON is operator-editable, so an imported grid
 *  value has to be validated before it's trusted as a `LoopGridRes`. */
export function isLoopGridRes(v: unknown): v is LoopGridRes {
  return typeof v === "string" && (LOOP_GRID_ORDER as string[]).includes(v);
}
