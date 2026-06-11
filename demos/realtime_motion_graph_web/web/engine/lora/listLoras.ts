import { fetchWithRetry } from "@demon/client";
import { podHttp } from "@/engine/podUrl";
import { useSessionStore } from "@/store/useSessionStore";
import type { LoraCatalogEntry } from "@demon/client";

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

/** Fetch the set of admin-hidden LoRA ids.
 *
 *  Unlike listLoras() this hits the APP ORIGIN (`/api/loras/hidden`),
 *  not the pod — admin visibility is webapp/orchestrator state, not
 *  engine state. The route proxies the orchestrator and is fail-open.
 *
 *  Fail-open here too: any error (route missing in a standalone DEMON
 *  dev server, orchestrator down, bad JSON) returns an empty set so a
 *  broken visibility backend never blanks the Library — worst case is
 *  "nothing hidden", never "nothing shown".
 */
export async function listHiddenLoras(): Promise<Set<string>> {
  try {
    const res = await fetch("/api/loras/hidden", { cache: "no-store" });
    if (!res.ok) return new Set();
    const json = (await res.json()) as { hidden?: unknown };
    if (!Array.isArray(json.hidden)) return new Set();
    return new Set(
      json.hidden.filter((x): x is string => typeof x === "string"),
    );
  } catch {
    return new Set();
  }
}
