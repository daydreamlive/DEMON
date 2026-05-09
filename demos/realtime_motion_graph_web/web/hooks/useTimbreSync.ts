"use client";

import { useEffect } from "react";

import { usePerformanceStore } from "@/store/usePerformanceStore";
import { useSessionStore } from "@/store/useSessionStore";

// Mirrors timbre_strength to the server via its dedicated WS message
// (set_timbre_strength). The server keeps a cached (cond_silence,
// cond_self) pair and lerp-blends them by alpha — cheap enough that we
// don't need per-tick streaming through the params path. We send only
// on actual change, rAF-throttled so a fast drag collapses to one send
// per frame.

export function useTimbreSync() {
  useEffect(() => {
    let lastSent = -1;
    let rafId = 0;
    let pending: number | null = null;

    const flush = () => {
      rafId = 0;
      if (pending === null) return;
      const v = pending;
      pending = null;
      const session = useSessionStore.getState();
      if (session.status !== "ready" || !session.remote) return;
      if (Math.abs(v - lastSent) < 1e-3) return;
      lastSent = v;
      session.remote.sendSetTimbreStrength(v);
    };

    const unsubPerf = usePerformanceStore.subscribe((s, prev) => {
      const v = s.sliderValues.timbre_strength ?? 1.0;
      const pv = prev.sliderValues.timbre_strength ?? 1.0;
      if (v === pv) return;
      pending = v;
      if (rafId === 0) rafId = requestAnimationFrame(flush);
    });

    // On every transition into "ready" (initial connect or restart),
    // re-sync the server to whatever the slider currently reads.
    // Without this a non-default slider value carried over from a prior
    // session would silently disagree with the server, which always
    // boots at strength=1.0.
    const unsubSession = useSessionStore.subscribe((s, prev) => {
      if (s.status === "ready" && prev.status !== "ready") {
        lastSent = -1;
        pending = usePerformanceStore.getState().sliderValues.timbre_strength
          ?? 1.0;
        if (rafId === 0) rafId = requestAnimationFrame(flush);
      }
    });

    return () => {
      if (rafId !== 0) cancelAnimationFrame(rafId);
      unsubPerf();
      unsubSession();
    };
  }, []);
}
