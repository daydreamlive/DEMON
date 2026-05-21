import { fetchWithRetry } from "@/engine/fetchWithRetry";
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
 *
 *  Uses fetchWithRetry so a backend that's still booting (502 from the
 *  Next dev proxy) is transparently waited on instead of leaving the
 *  catalog empty until the operator refreshes.
 */
export async function listLoras(): Promise<LoraCatalogEntry[]> {
  const res = await fetchWithRetry(podHttp("/api/loras"));
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

/** Fetch the set of LoRA ids an admin has hidden from the Library.
 *
 *  Unlike listLoras() this hits the **app origin** (`/api/loras/hidden`,
 *  a Vercel route backed by the orchestrator) — NOT the pod — because
 *  the hidden list is global, not per-pod. The Library tile uses it to
 *  drop hidden LoRAs from the catalog it renders.
 *
 *  Fail-open: any error (route missing, orchestrator down, malformed
 *  body) yields an empty set, i.e. "nothing hidden" → the Library shows
 *  everything. A broken visibility service must never blank the tile.
 */
export async function listHiddenLoras(): Promise<Set<string>> {
  try {
    const res = await fetch("/api/loras/hidden", { cache: "no-store" });
    if (!res.ok) return new Set();
    const json = (await res.json()) as { hidden?: unknown };
    return new Set(
      Array.isArray(json.hidden)
        ? json.hidden.filter((x): x is string => typeof x === "string")
        : [],
    );
  } catch {
    return new Set();
  }
}
