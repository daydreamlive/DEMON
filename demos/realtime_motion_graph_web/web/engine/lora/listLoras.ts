import { podHttp } from "@/engine/podUrl";
import type { LoraCatalogEntry } from "@/types/protocol";

export async function listLoras(): Promise<LoraCatalogEntry[]> {
  const res = await fetch(podHttp("/api/loras"));
  if (!res.ok) throw new Error(`/api/loras failed: ${res.status}`);
  const json = (await res.json()) as { dir: string; loras: LoraCatalogEntry[] };
  return json.loras ?? [];
}
