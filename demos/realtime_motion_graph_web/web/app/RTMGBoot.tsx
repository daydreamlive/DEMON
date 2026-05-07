"use client";

import { setEngineUrlBuilder } from "@/engine/rtmgConfig";

// Same-origin URL builder. The engine's HTTP routes (/api/*, /fixtures/*,
// /loras/*, /videos/*) are proxied to the Python backend at :8765 by the
// Next.js rewrites in next.config.ts. The WebSocket URL goes through
// `defaultWsUrl()` which reads NEXT_PUBLIC_POD_BASE_URL — set in .env.local.
//
// Configured at module load (top-level, not in useEffect) so it's ready
// before any child component's mount-time fetch fires.
setEngineUrlBuilder((path) => (path.startsWith("/") ? path : `/${path}`));

export function RTMGBoot() {
  return null;
}
