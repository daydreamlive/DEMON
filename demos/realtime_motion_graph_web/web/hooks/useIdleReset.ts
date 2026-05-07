"use client";

import { useEffect } from "react";

import { useLoraStore } from "@/store/useLoraStore";
import { usePerformanceStore } from "@/store/usePerformanceStore";

// After N seconds with no user interaction (pointer move, key, MIDI), revert
// every slider + seed + blend back to defaults. The DEMON original gates
// this by config.controls.reset_seconds; 0 disables. Phase 10 keeps the
// gating: hook accepts an optional duration (seconds); 0 = disabled.
//
// Also drives `body.cursor-idle` (existing CSS toggle) for the auto-hide
// cursor in kiosk mode.

const CURSOR_IDLE_MS = 2000;

export function useIdleReset(seconds: number) {
  useEffect(() => {
    if (typeof document === "undefined") return;
    let resetTimer: number | null = null;
    let idleTimer: number | null = null;
    const intervalMs = seconds > 0 ? seconds * 1000 : 0;

    function bump() {
      document.body.classList.remove("cursor-idle");
      if (idleTimer !== null) clearTimeout(idleTimer);
      idleTimer = window.setTimeout(
        () => document.body.classList.add("cursor-idle"),
        CURSOR_IDLE_MS,
      );
      if (intervalMs > 0) {
        if (resetTimer !== null) clearTimeout(resetTimer);
        resetTimer = window.setTimeout(() => {
          usePerformanceStore.getState().resetToDefaults();
          useLoraStore.getState().reset();
        }, intervalMs);
      }
    }

    bump();
    document.addEventListener("mousemove", bump, { passive: true });
    document.addEventListener("pointerdown", bump, { passive: true });
    document.addEventListener("keydown", bump);
    // MIDI: route via the same listener pattern. The MIDI hook fires
    // store updates; we listen on those updates to reset the timers.
    const unsubPerf = usePerformanceStore.subscribe(bump);

    return () => {
      document.removeEventListener("mousemove", bump);
      document.removeEventListener("pointerdown", bump);
      document.removeEventListener("keydown", bump);
      unsubPerf();
      if (resetTimer !== null) clearTimeout(resetTimer);
      if (idleTimer !== null) clearTimeout(idleTimer);
      document.body.classList.remove("cursor-idle");
    };
  }, [seconds]);
}
