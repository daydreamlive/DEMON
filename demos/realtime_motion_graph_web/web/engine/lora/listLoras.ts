import { podHttp } from "@/engine/podUrl";
import { useSessionStore } from "@/store/useSessionStore";
import type { LoraCatalogEntry } from "@/types/protocol";

/** Fetch /api/loras and return the LoRA catalog.
 *
 *  Side effect: writes the server-reported ``checkpoint_scale`` into
 *  ``useSessionStore`` so the LoRA library can hide LoRAs trained for
 *  a different checkpoint even before the WS ready frame arrives.
 *  Older servers that don't ship the field leave it ``null``, which
 *  the UI treats as "don't filter".
 */
export async function listLoras(): Promise<LoraCatalogEntry[]> {
  const res = await fetch(podHttp("/api/loras"));
  if (!res.ok) throw new Error(`/api/loras failed: ${res.status}`);
  const json = (await res.json()) as {
    dir: string;
    loras: LoraCatalogEntry[];
    checkpoint_scale?: string | null;
  };
  useSessionStore
    .getState()
    .setCheckpointScale(json.checkpoint_scale ?? null);
  return json.loras ?? [];
}
