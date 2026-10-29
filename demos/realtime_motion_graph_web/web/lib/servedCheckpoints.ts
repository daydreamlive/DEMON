import { useSessionStore, type ServedCheckpoint } from "@/store/useSessionStore";

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

/** The pod's `/api/server-info` URL, derived from its WS URL (the pod
 *  serves HTTP and WS on one origin; query string dropped). */
export function serverInfoUrl(wsUrl: string): string {
  const u = new URL(wsUrl);
  u.protocol = u.protocol === "wss:" ? "https:" : "http:";
  const basePath = u.pathname.replace(/\/+$/, "");
  u.pathname = `${basePath}/api/server-info`;
  u.search = "";
  return u.toString();
}

/** Fetch server-info (4 s timeout) and write its `checkpoints` to the
 *  session store. Returns the parsed body, or null on any failure / old
 *  backend (the store is then left as it was). Never throws. */
export async function fetchServerInfo(wsUrl: string): Promise<unknown | null> {
  try {
    const ctrl = new AbortController();
    const timer = setTimeout(() => ctrl.abort(), 4000);
    let res: Response;
    try {
      res = await fetch(serverInfoUrl(wsUrl), { signal: ctrl.signal });
    } finally {
      clearTimeout(timer);
    }
    if (!res.ok) return null;
    const info = (await res.json()) as unknown;
    useSessionStore.getState().setServedCheckpoints(parseServedCheckpoints(info));
    return info;
  } catch {
    return null;
  }
}

/** Boot-time gate, the same one LibraryTile uses for its pre-Start
 *  /api/loras fetch: a queue-issued wsUrl, or local mode (no queue). */
export function shouldFetchServerInfoAtBoot(
  wsUrl: string | null,
  localMode: boolean,
): boolean {
  return !!wsUrl || localMode;
}
