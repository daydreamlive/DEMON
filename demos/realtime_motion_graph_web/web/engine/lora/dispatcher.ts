"use client";

import {
  stampManualTouch,
  usePerformanceStore,
} from "@/store/usePerformanceStore";
import { useLoraStore } from "@/store/useLoraStore";
import { LORA_SLIDER_MAX } from "@/types/engine";

// Trailing-edge debounce for LoRA strength updates. LoRA strength
// changes trigger a weight refit on the server, which is far more
// expensive than a decode tick. Streaming the smoothed value at the
// param-sync cadence (8 ms) costs one refit per decode window during
// a drag — typically a stall per refit because the refit blocks the
// next tick. Instead we debounce: only the value the slider settles
// on (or pauses on for DEBOUNCE_MS) reaches the engine, so a long
// drag commits exactly one refit.
//
// We bypass the smoothing tween entirely for LoRA params: smoothing
// would just stretch the refit-storm across the smoothMs window
// instead of preventing it.

const DEBOUNCE_MS = 300;

interface PendingEntry {
  timer: number;
  value: number;
}

const pending = new Map<string, PendingEntry>();

function clamp(value: number): number {
  if (!Number.isFinite(value)) return 0;
  return Math.max(0, Math.min(LORA_SLIDER_MAX, value));
}

export const loraStrengthDispatcher = {
  /** Drag/MIDI/edge-bar entry point. Pushes the value into the UI
   *  stores synchronously so the slider/fill/ribbon tracks the cursor,
   *  then arms (or resets) a trailing-edge timer that finally commits
   *  to sliderValues — where useParamSync picks it up and ships it. */
  set(id: string, rawValue: number): void {
    const value = clamp(rawValue);
    const param = `lora_str_${id}`;

    useLoraStore.getState().setStrength(id, value);
    stampManualTouch(param);
    usePerformanceStore.setState((s) => ({
      sliderTargets: { ...s.sliderTargets, [param]: value },
    }));

    const existing = pending.get(id);
    if (existing) clearTimeout(existing.timer);
    const timer = window.setTimeout(() => {
      pending.delete(id);
      usePerformanceStore.getState().setSliderDirect(param, value);
    }, DEBOUNCE_MS);
    pending.set(id, { timer, value });
  },
};
