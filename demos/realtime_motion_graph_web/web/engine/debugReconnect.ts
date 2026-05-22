// Dev / test harness for the WS reconnect path.
//
// In production the dominant cause of 1006 (abnormal closure) is the
// pod's tunnel layer (RunPod / vast.ai / Cloudflare) dropping the TCP
// connection without a WebSocket close frame. We can't easily reproduce
// that locally, so this module exposes a small `window.__demonDebug`
// API that synthesizes the same client-side effect: dispatch a close
// event the reconnect handler treats as 1006.
//
// Two paths, both ending in the same place:
//
//   __demonDebug.simulate1006() — synthesize the close event entirely
//     in the client (RemoteBackend.simulateClose). Routes through the
//     real listener chain so the reconnect orchestrator behaves
//     exactly as it would on a real drop. Best for verifying recovery
//     UX without touching the server.
//
//   __demonDebug.killWs() — call the underlying WebSocket's .close()
//     directly. The browser dispatches close code 1005 ("No Status
//     Received") rather than 1006, but the reconnect path treats both
//     identically (any !closedByUser close kicks the loop). Useful for
//     a quick console smoke-test from inside a real session.

import { useSessionStore } from "@/store/useSessionStore";

interface DemonDebug {
  simulate1006: (reason?: string) => boolean;
  killWs: () => boolean;
  status: () => string;
}

declare global {
  interface Window {
    __demonDebug?: DemonDebug;
  }
}

function api(): DemonDebug {
  return {
    simulate1006(reason = "simulated 1006") {
      const remote = useSessionStore.getState().remote;
      if (!remote) return false;
      remote.simulateClose(1006, reason);
      return true;
    },
    killWs() {
      const remote = useSessionStore.getState().remote;
      const ws = remote?.ws;
      if (!ws) return false;
      // Don't set closedByUser — we want the close event to flow
      // through the same path as a real network drop.
      try {
        ws.close();
      } catch {}
      return true;
    },
    status() {
      const s = useSessionStore.getState();
      return JSON.stringify({
        status: s.status,
        message: s.message,
        hasRemote: !!s.remote,
        hasPlayer: !!s.player,
        reconnecting: !!s.reconnector,
        wsReadyState: s.remote?.ws?.readyState,
      });
    },
  };
}

export function installDemonDebug(): void {
  if (typeof window === "undefined") return;
  window.__demonDebug = api();
}
