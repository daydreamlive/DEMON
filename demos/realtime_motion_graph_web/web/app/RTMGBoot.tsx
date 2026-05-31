"use client";

import { installDemonDebug } from "@/engine/debugReconnect";
import { listLoras } from "@/engine/lora/listLoras";
import { setEngineUrlBuilder } from "@/engine/rtmgConfig";
import { applyConfig, loadConfig } from "@/lib/config";
import { useLoraStore } from "@/store/useLoraStore";
import type { LoraCatalogEntry } from "@/types/protocol";

// Engine URL builder. Builds ABSOLUTE URLs straight to the backend
// (NEXT_PUBLIC_POD_BASE_URL) for /api/*, /fixtures/*, /loras/*, /videos/* —
// the same host the WebSocket connects to. The backend sends
// `Access-Control-Allow-Origin: *`, so the cross-origin fetch is allowed.
//
// We deliberately do NOT rely on next.config.ts rewrites to proxy these:
// the dev bundler doesn't reliably forward them, which surfaces as 404s on
// /api/* even though the engine serves them fine over curl. Going direct to
// the backend makes the remote client/server case "just work" once
// NEXT_PUBLIC_POD_BASE_URL points at the server (see run.py --client-host).
//
// Falls back to a same-origin relative path when the base URL is unset (the
// old rewrite path). Configured at module load (top-level, not in useEffect)
// so it's ready before any child component's mount-time fetch fires.
const _engineBase = (process.env.NEXT_PUBLIC_POD_BASE_URL ?? "").replace(/\/$/, "");
setEngineUrlBuilder((path) => {
  const p = path.startsWith("/") ? path : `/${path}`;
  return _engineBase ? `${_engineBase}${p}` : p;
});

// Fire the config + LoRA catalog fetches in parallel and await both
// before applyConfig(). listLoras' side effect writes the server's
// checkpoint_scale into useSessionStore, which applyConfig reads to
// pick between base (turbo / 2B) and `_xl` variant fields.
//
// After applyConfig we also push the catalog we fetched into the lora
// store. LibraryTile's mount-time listLoras (LOCAL_MODE) races this
// path, and either side could win the /api/loras response. The
// isConfigApplied gate inside setCatalog + the retroactive setCatalog
// trigger inside applyConfig together make the seeding deterministic
// regardless of order — this push just ensures the common case (boot
// wins) does the work directly rather than via the retro path.
if (typeof window !== "undefined") {
  // Expose the WS reconnect test harness on window.__demonDebug. Cheap
  // (no listeners or intervals), so no need to gate on NODE_ENV — having
  // it available in prod also lets us hot-test the reconnect path
  // against a live pod without needing a separate build.
  installDemonDebug();
  void (async () => {
    const [cfg, catalog] = await Promise.all([
      loadConfig(),
      listLoras().catch(() => [] as LoraCatalogEntry[]),
    ]);
    applyConfig(cfg);
    if (catalog.length > 0) {
      useLoraStore.getState().setCatalog(catalog);
    }
  })();
}

export function RTMGBoot() {
  return null;
}
