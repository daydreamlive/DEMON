"use client";

import { useEffect } from "react";

import { usePerformanceStore } from "@/store/usePerformanceStore";
import { useSessionStore } from "@/store/useSessionStore";

// Mirrors prompt-blend (A/B slider) to the server via its dedicated WS
// message (set_prompt_blend). Server caches cond pairs for promptA and
// promptB (refreshed only when Send Prompt is fired) and lerps between
// them per tick — same shape as useTimbreSync.
//
// Reads ``sliderValues.prompt_blend`` (the SMOOTHED value), not
// ``sliderTargets``. With the Smooth toggle on, this is what gives the
// server a tweened crossfade over ``smoothMs`` instead of an instant
// jump. With Smooth off, sliderValues and sliderTargets are identical.
//
// Min-interval throttle (NOT debounce): at most one send per
// THROTTLE_MS, with a trailing flush so the final value still lands.
// Sized to ~one diffusion tick — sending at rAF rate (60+ Hz) causes
// stream.conditioning to mutate multiple times per engine tick, which
// produces audible clicks because cond_pair_a and cond_pair_b are
// encoded from very different prompt texts (unlike timbre's
// silence/full pair, where adjacent ticks barely differ). Throttling
// to ~10 Hz means each engine tick sees a stable conditioning value.

const THROTTLE_MS = 100;

export function usePromptBlendSync() {
  useEffect(() => {
    let lastSent = -1;
    let lastSentAt = 0;
    let pending: number | null = null;
    let timerId = 0;

    const flush = () => {
      timerId = 0;
      if (pending === null) return;
      const v = pending;
      pending = null;
      const session = useSessionStore.getState();
      if (session.status !== "ready" || !session.remote) return;
      if (Math.abs(v - lastSent) < 1e-3) return;
      lastSent = v;
      lastSentAt = performance.now();
      session.remote.sendSetPromptBlend(v);
    };

    const schedule = () => {
      if (timerId !== 0) return;
      const elapsed = performance.now() - lastSentAt;
      const delay = Math.max(0, THROTTLE_MS - elapsed);
      timerId = window.setTimeout(flush, delay);
    };

    const unsubPerf = usePerformanceStore.subscribe((s, prev) => {
      const v = s.sliderValues.prompt_blend ?? 0;
      const pv = prev.sliderValues.prompt_blend ?? 0;
      if (v === pv) return;
      pending = v;
      schedule();
    });

    // Re-sync on every transition into "ready" so a non-default slider
    // value carried over from a prior session doesn't silently disagree
    // with the server (which always boots at blend=0).
    const unsubSession = useSessionStore.subscribe((s, prev) => {
      if (s.status === "ready" && prev.status !== "ready") {
        lastSent = -1;
        lastSentAt = 0;
        pending =
          usePerformanceStore.getState().sliderValues.prompt_blend ?? 0;
        schedule();
      }
    });

    return () => {
      if (timerId !== 0) window.clearTimeout(timerId);
      unsubPerf();
      unsubSession();
    };
  }, []);
}
