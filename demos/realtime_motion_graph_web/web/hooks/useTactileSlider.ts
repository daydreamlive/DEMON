"use client";

import { useEffect, useRef } from "react";

import { valueToT, type SliderMapping } from "@/lib/sliderMapping";
import { usePerformanceStore } from "@/store/usePerformanceStore";

// Lightweight per-slider tactile augmentation: feature-detected haptics on
// landmark crossings (0, 0.5, 1.0 of the THUMB position) and a long-press
// reset gesture.
//
// Touch the rail/slider for >= LONG_PRESS_MS without moving more than
// LONG_PRESS_MOVE_PX and we treat it as a "snap back" gesture: the param
// is reset to its default. iOS Safari ignores `navigator.vibrate` so the
// haptic is a no-op there but the long-press still works.
//
// We do NOT take over the pointer pipeline — callers keep their own drag
// handlers. We just attach a few extra listeners to the same element and
// react to the same events.

const LONG_PRESS_MS = 500;
const LONG_PRESS_MOVE_PX = 8;
const HAPTIC_CROSSINGS = [0, 0.5, 1.0] as const;
const HAPTIC_TOL = 0.04;

function vibrate(ms: number): void {
  if (typeof navigator === "undefined") return;
  const v = (navigator as Navigator & { vibrate?: (n: number) => boolean })
    .vibrate;
  if (typeof v === "function") {
    try {
      v.call(navigator, ms);
    } catch {}
  }
}

interface Options {
  /** sliderTargets key (e.g. "denoise", "lora_blend"). */
  param: string;
  /** Same mapping bundle SliderGroup uses (min/max/unity/reverse). We
   *  fire haptics on the thumb's position crossings (0 / 0.5 / 1 of the
   *  rail), not on engine-value crossings, so reverse + unity-anchored
   *  channels still feel landmarks at the bottom, middle, and top of
   *  the rail. Bypasses any asymmetry between value and thumb position
   *  introduced by the unity-anchored piecewise mapping. */
  mapping: SliderMapping;
  /** Element to attach long-press detection on. */
  ref: React.MutableRefObject<HTMLElement | null>;
}

export function useTactileSlider({ param, mapping, ref }: Options): void {
  // Track previous fraction so we only fire haptic at the moment a crossing
  // happens, not on every redraw at that position.
  const prevFrac = useRef<number | null>(null);

  // Reactively re-bind to mapping changes via the primitive fields, not
  // the object identity (which gets rebuilt every parent render).
  const { min, max, unity, reverse } = mapping;

  // Subscribe directly to the store: cheaper than re-rendering the parent
  // for every value change, and lets us fire vibrate() outside React's
  // commit phase.
  useEffect(() => {
    const m: SliderMapping = { min, max, unity, reverse };
    const fire = () => {
      const v = usePerformanceStore.getState().sliderTargets[param] ?? 0;
      // Thumb position fraction (0 at bottom, 1 at top). For unity-
      // anchored bands, value=unity always lands at frac=0.5 — so the
      // mid-rail haptic fires when the operator drags through unity
      // regardless of where unity sits in the channel's [min, max].
      const frac = valueToT(v, m);
      const prev = prevFrac.current;
      prevFrac.current = frac;
      if (prev === null) return;
      for (const cross of HAPTIC_CROSSINGS) {
        const wasNear = Math.abs(prev - cross) <= HAPTIC_TOL;
        const isNear = Math.abs(frac - cross) <= HAPTIC_TOL;
        if (!wasNear && isNear) {
          vibrate(8);
          break;
        }
      }
    };
    return usePerformanceStore.subscribe(fire);
  }, [param, min, max, unity, reverse]);

  // Long-press to reset.
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    let timer: ReturnType<typeof setTimeout> | null = null;
    let startX = 0;
    let startY = 0;

    const cancel = () => {
      if (timer) {
        clearTimeout(timer);
        timer = null;
      }
    };

    const onDown = (e: PointerEvent) => {
      startX = e.clientX;
      startY = e.clientY;
      cancel();
      timer = setTimeout(() => {
        usePerformanceStore.getState().resetSlider(param);
        // Stronger feedback for the destructive-feeling reset.
        vibrate(20);
      }, LONG_PRESS_MS);
    };
    const onMove = (e: PointerEvent) => {
      if (!timer) return;
      const dx = e.clientX - startX;
      const dy = e.clientY - startY;
      if (dx * dx + dy * dy > LONG_PRESS_MOVE_PX * LONG_PRESS_MOVE_PX) {
        cancel();
      }
    };
    const onUp = () => {
      cancel();
    };

    el.addEventListener("pointerdown", onDown);
    el.addEventListener("pointermove", onMove);
    el.addEventListener("pointerup", onUp);
    el.addEventListener("pointercancel", onUp);
    return () => {
      cancel();
      el.removeEventListener("pointerdown", onDown);
      el.removeEventListener("pointermove", onMove);
      el.removeEventListener("pointerup", onUp);
      el.removeEventListener("pointercancel", onUp);
    };
  }, [param, ref]);
}
