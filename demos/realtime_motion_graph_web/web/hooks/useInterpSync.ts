"use client";

import { useEffect } from "react";

import {
  INTERP_PATHS,
  useInterpStore,
  type InterpPath,
} from "@/store/useInterpStore";
import { useSessionStore } from "@/store/useSessionStore";

// Mirrors the per-path interpolation methods to the server via the
// dedicated set_interp_method WS message. These are discrete settings
// (not smoothed sliders), so each change ships immediately with no
// throttle/echo. On every transition into "ready" the full set is
// re-pushed so the server matches the dropdowns even after a reconnect
// or restart.

export function useInterpSync() {
  useEffect(() => {
    const sendAll = () => {
      const session = useSessionStore.getState();
      if (session.status !== "ready" || !session.remote) return;
      const { methods } = useInterpStore.getState();
      for (const path of INTERP_PATHS) {
        session.remote.sendSetInterpMethod(path, methods[path]);
      }
    };

    const unsubInterp = useInterpStore.subscribe((s, prev) => {
      const session = useSessionStore.getState();
      if (session.status !== "ready" || !session.remote) return;
      for (const path of INTERP_PATHS) {
        const p = path as InterpPath;
        if (s.methods[p] !== prev.methods[p]) {
          session.remote.sendSetInterpMethod(p, s.methods[p]);
        }
      }
    });

    const unsubSession = useSessionStore.subscribe((s, prev) => {
      if (s.status === "ready" && prev.status !== "ready") sendAll();
    });

    return () => {
      unsubInterp();
      unsubSession();
    };
  }, []);
}
