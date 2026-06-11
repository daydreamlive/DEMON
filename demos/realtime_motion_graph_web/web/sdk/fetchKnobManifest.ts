import { fetchWithRetry } from "./fetchWithRetry";
import type { KnobManifest, KnobManifestResponse } from "./types/knobs";

/** Fetch the backend knob manifest from `GET /api/knobs`.
 *
 *  Returns the full `{version, knobs}` envelope so callers can compare
 *  `version` against the generated `KNOB_SCHEMA_VERSION` to detect a stale
 *  build (mirroring the wire-contract `PROTOCOL_VERSION` check). The knobs
 *  map is the static schema (per-session `lora_str_<id>` knobs and live
 *  values arrive separately over the WS ready frame). `sde` selects the
 *  SDE-mode core set. Uses fetchWithRetry so a still-booting backend (502
 *  from a dev proxy) is waited on rather than yielding an empty manifest.
 *
 *  `toUrl` maps the API path to a fetchable URL. Default is identity
 *  (same-origin / proxied paths); the shipped app passes its pod URL
 *  builder.
 */
export async function fetchKnobManifest(
  sde = false,
  toUrl: (path: string) => string = (p) => p,
): Promise<KnobManifestResponse> {
  const res = await fetchWithRetry(toUrl(`/api/knobs${sde ? "?sde=1" : ""}`));
  if (!res.ok) throw new Error(`/api/knobs failed: ${res.status}`);
  const json = (await res.json()) as {
    version?: number;
    knobs?: KnobManifest;
  };
  return { version: json.version, knobs: json.knobs ?? {} };
}
