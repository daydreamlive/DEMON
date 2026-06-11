import { fetchWithRetry } from "./fetchWithRetry";
import type { WireContract } from "./types/wireContract";

/** Fetch the backend WebSocket wire contract from `GET /api/protocol`.
 *
 *  The self-describing command/event vocabulary (see types/wireContract.ts).
 *  Session-independent — unlike the knob manifest, it doesn't vary with SDE
 *  mode or the enabled LoRAs. Uses fetchWithRetry so a still-booting backend
 *  (502 from a dev proxy) is waited on rather than yielding null.
 *
 *  `toUrl` maps the API path to a fetchable URL. Default is identity
 *  (same-origin / proxied paths); the shipped app passes its pod URL
 *  builder so the manifest can be fetched cross-origin before the WS
 *  handshake.
 */
export async function fetchWireContract(
  toUrl: (path: string) => string = (p) => p,
): Promise<WireContract> {
  const res = await fetchWithRetry(toUrl("/api/protocol"));
  if (!res.ok) throw new Error(`/api/protocol failed: ${res.status}`);
  return (await res.json()) as WireContract;
}
