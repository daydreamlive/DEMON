import type { ServedCheckpoint } from "@/store/useSessionStore";

/** Parse server-info `checkpoints` defensively. Old pods don't ship the
 *  key, so anything malformed yields an empty list (picker hidden). */
export function parseServedCheckpoints(info: unknown): ServedCheckpoint[] {
  const raw = (info as { checkpoints?: unknown } | null)?.checkpoints;
  if (!Array.isArray(raw)) return [];
  const out: ServedCheckpoint[] = [];
  for (const e of raw) {
    if (!e || typeof e !== "object") continue;
    const r = e as Record<string, unknown>;
    if (typeof r.name !== "string" || r.name === "") continue;
    out.push({
      name: r.name,
      family: typeof r.family === "string" ? r.family : "",
      model_id: typeof r.model_id === "string" ? r.model_id : r.name,
      scale: typeof r.scale === "string" ? r.scale : null,
      default: r.default === true,
    });
  }
  return out;
}

/** The Model picker renders only when the pod serves a real choice, so
 *  every single-model pod looks exactly as before. */
export function showCheckpointPicker(list: readonly ServedCheckpoint[]): boolean {
  return list.length >= 2;
}
