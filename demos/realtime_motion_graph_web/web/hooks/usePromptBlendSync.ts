"use client";

import { useEffect } from "react";

import { usePerformanceStore } from "@/store/usePerformanceStore";
import { useSessionStore } from "@/store/useSessionStore";

// Mirrors prompt-blend (A/B slider) to the server via its dedicated WS
// message (set_prompt_blend). Server caches cond pairs for promptA and
// promptB (refreshed only when Send Prompt is fired) and lerps between
// them per tick — same shape as useTimbreSync. Sends only on actual
// change, rAF-throttled so a fast drag collapses to one send per frame.

export function usePromptBlendSync() {
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
      session.remote.sendSetPromptBlend(v);
    };

    const unsubPerf = usePerformanceStore.subscribe((s, prev) => {
      if (s.blend === prev.blend) return;
      pending = s.blend;
      if (rafId === 0) rafId = requestAnimationFrame(flush);
    });

    // Re-sync on every transition into "ready" so a non-default slider
    // value carried over from a prior session doesn't silently disagree
    // with the server (which always boots at blend=0).
    const unsubSession = useSessionStore.subscribe((s, prev) => {
      if (s.status === "ready" && prev.status !== "ready") {
        lastSent = -1;
        pending = usePerformanceStore.getState().blend;
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
