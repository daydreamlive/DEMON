"use client";

import { useSessionStore } from "@/store/useSessionStore";

/** Gate a hand-coded panel on the backend capability mask from the WS
 *  `ready` message (backend-seam plan §3.1).
 *
 *  Returns false ONLY when the live session's mask explicitly declares
 *  the capability false. A null mask (no session yet, a pre-Phase-2
 *  server, or a recorded replay transcript) and a mask missing the key
 *  both gate OPEN — old behavior is "everything available", and hiding
 *  controls on missing data would regress every existing session. The
 *  server still rejects unsupported commands loudly (`command_failed`),
 *  so an open gate on stale data fails safe.
 */
export function useCapability(name: string): boolean {
  return useSessionStore((s) => s.capabilities?.[name] !== false);
}
