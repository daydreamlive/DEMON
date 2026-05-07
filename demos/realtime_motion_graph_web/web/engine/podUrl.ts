// URL helpers — back-compat re-export shim.

export {
  podHttp,
  setPodSessionId,
  getPodSessionId,
} from "./rtmgConfig";

/** WS URL fallback. Returns `?ws=<override>` from the URL or
 *  `NEXT_PUBLIC_POD_BASE_URL` rewritten as `ws://`. */
export function defaultWsUrl(): string {
  if (typeof window !== "undefined") {
    const params = new URLSearchParams(window.location.search);
    const override = params.get("ws");
    if (override) return override;
  }
  const base = process.env.NEXT_PUBLIC_POD_BASE_URL ?? "";
  return base.replace(/\/$/, "").replace(/^http/, "ws") + "/";
}

export function podBaseUrl(): string {
  return (process.env.NEXT_PUBLIC_POD_BASE_URL ?? "").replace(/\/$/, "");
}
