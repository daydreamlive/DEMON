"use client";

import type { CSSProperties } from "react";
import { useEffect, useRef } from "react";

import { useTactileSlider } from "@/hooks/useTactileSlider";
import { tToValue, valueToT } from "@/lib/sliderMapping";
import { usePerformanceStore } from "@/store/usePerformanceStore";
import { SLIDER_META } from "@/types/engine";

import { tooltipFor } from "./SliderTile";

// Vertical slider matching DEMON's CSS layout (.slider-track + .slider-fill
// + .slider-thumb). Click + drag on the track to set; the value reads from
// usePerformanceStore. LoRA sliders (param starting with `lora_str_`)
// supply their own max via prop because they aren't in SLIDER_META.

interface Props {
  param: string;
  label: string;
  /** Override max + step for ad-hoc sliders (LoRA strength). */
  max?: number;
  /** Floor for the value range. Defaults to 0. Used by per-channel
   *  ranges loaded from config (see lib/config.ts → channel_ranges). */
  min?: number;
  /** If true, dragging the thumb UP drives the engine value DOWN. The
   *  thumb still moves in the natural direction (so the visual fill
   *  grows with the drag); only the value-of-thumb mapping is flipped.
   *  Use for channels that "sound better when turned down" so the
   *  operator's instinct to push up produces the desired result. */
  reverse?: boolean;
  /** When set, pins this value to the midpoint of the rail and uses
   *  piecewise-linear mapping above/below the anchor. Lets a bank of
   *  sliders with different per-channel [min, max] caps display the
   *  same default at the same visual rail height (see
   *  lib/sliderMapping.ts). Typical value: 1.0 ("unity gain") for the
   *  channel-gain bank. */
  unity?: number;
  kbd?: string;
}

// Palette stops mirror the .slider-fill gradient
// (linear-gradient(to top, --dd-4 0%, --dd-3 35%, --dd-2 70%, --dd-1 100%)).
// `t` is the slider fraction 0→1 measured from the BOTTOM of the track,
// so t=0 maps to --dd-4 (coral), t=1 to --dd-1 (teal). Sampling here lets
// the label / rail / thumb-border share the same color the fill gradient
// would show at the current value.
const TINT_STOPS: ReadonlyArray<readonly [number, readonly [number, number, number]]> = [
  [0.0, [232, 79, 61]],
  [0.3, [240, 138, 72]],
  [0.65, [199, 181, 102]],
  [1.0, [61, 182, 190]],
];

function tintAt(t: number): string {
  const clamped = Math.max(0, Math.min(1, t));
  for (let i = 1; i < TINT_STOPS.length; i++) {
    const [p1, c1] = TINT_STOPS[i - 1];
    const [p2, c2] = TINT_STOPS[i];
    if (clamped <= p2) {
      const k = p2 === p1 ? 0 : (clamped - p1) / (p2 - p1);
      const r = Math.round(c1[0] + (c2[0] - c1[0]) * k);
      const g = Math.round(c1[1] + (c2[1] - c1[1]) * k);
      const b = Math.round(c1[2] + (c2[2] - c1[2]) * k);
      return `rgb(${r} ${g} ${b})`;
    }
  }
  const [, last] = TINT_STOPS[TINT_STOPS.length - 1];
  return `rgb(${last[0]} ${last[1]} ${last[2]})`;
}

export function SliderGroup({
  param,
  label,
  max,
  min,
  reverse,
  unity,
  kbd,
}: Props) {
  const meta = SLIDER_META[param];
  const effectiveMax = max ?? meta?.max ?? 1.0;
  const effectiveMin = min ?? 0;
  // Mapping bundle, passed to lib/sliderMapping helpers (and the
  // tactile-slider hook). When `unity` is set, the rail uses piecewise
  // mapping anchored at the midpoint; otherwise linear from min..max.
  const mapping = {
    min: effectiveMin,
    max: effectiveMax,
    unity,
    reverse: !!reverse,
  };
  // Read the user's target (instant), not the smoothed sent value, so
  // dragging tracks the cursor without smoothing lag.
  const value = usePerformanceStore(
    (s) => s.sliderTargets[param] ?? 0,
  );
  const setSlider = usePerformanceStore((s) => s.setSlider);
  const trackRef = useRef<HTMLDivElement | null>(null);

  // Haptics on landmark crossings (0, 0.5, 1.0 of slider position) +
  // long-press reset. Pass the full mapping so the haptic fires on the
  // thumb's position, not the engine value — on reverse + unity-anchored
  // channels those move opposite each other (and at different slopes).
  useTactileSlider({
    param,
    mapping,
    ref: trackRef,
  });

  // Slider thumb fraction t ∈ [0, 1] (0 = bottom, 1 = top). For
  // unity-anchored bands, the unity value lands at t=0.5 regardless of
  // the channel's actual [min, max] — which is the whole point of the
  // visual rail [0, 2] convention.
  const t = valueToT(value, mapping);
  const pct = t * 100;

  useEffect(() => {
    const el = trackRef.current;
    if (!el) return;

    // Cache the track rect at pointerdown and reuse for the lifetime of
    // the drag. Without this, every pointermove called
    // getBoundingClientRect(), which forces a synchronous layout flush
    // and evicts paint caches — the dominant source of cursor jank
    // during slider drags. The track does not resize during a drag.
    let dragging = false;
    let cachedRect: DOMRect | null = null;
    let pendingClientY = 0;
    let rafId = 0;
    // Touch-only: defer the initial commit by ENGAGE_MS so a brief brush
    // against the slider doesn't yank the value. Movement before the
    // timeout promotes us to engaged immediately. Mouse/pen pointers
    // engage instantly (desktop expectation).
    let engaged = false;
    let engageTimer: ReturnType<typeof setTimeout> | null = null;
    const ENGAGE_MS = 50;
    // DAW-style double-click-to-reset. Tracking pointerdown timestamps
    // ourselves (instead of leaning on the synthetic `dblclick` event) is
    // reliable in the presence of pointer capture and cross-browser quirks.
    let lastDownAt = 0;
    const DBLCLICK_MS = 350;

    const commit = () => {
      if (!cachedRect) return;
      const tFrac = 1 - (pendingClientY - cachedRect.top) / cachedRect.height;
      setSlider(param, tToValue(tFrac, mapping));
    };

    const flush = () => {
      rafId = 0;
      if (!dragging || !engaged) return;
      commit();
    };

    const clearEngageTimer = () => {
      if (engageTimer) {
        clearTimeout(engageTimer);
        engageTimer = null;
      }
    };

    const onPointerDown = (e: PointerEvent) => {
      // Right-click reserved for MIDI-learn.
      if (e.button !== 0) return;
      const now = performance.now();
      if (now - lastDownAt < DBLCLICK_MS) {
        // Second click within the dblclick window: reset and bail before
        // any drag state is set. Coexists with the long-press reset on
        // mobile (useTactileSlider) — operators can use whichever feels
        // natural.
        usePerformanceStore.getState().resetSlider(param);
        lastDownAt = 0;
        return;
      }
      lastDownAt = now;
      dragging = true;
      engaged = false;
      cachedRect = el.getBoundingClientRect();
      el.setPointerCapture(e.pointerId);
      pendingClientY = e.clientY;
      if (e.pointerType !== "touch") {
        engaged = true;
        commit();
        return;
      }
      engageTimer = setTimeout(() => {
        engageTimer = null;
        engaged = true;
        commit();
      }, ENGAGE_MS);
    };
    const onPointerMove = (e: PointerEvent) => {
      if (!dragging) return;
      pendingClientY = e.clientY;
      if (!engaged) {
        clearEngageTimer();
        engaged = true;
        commit();
        return;
      }
      if (rafId === 0) rafId = requestAnimationFrame(flush);
    };
    const onPointerUp = (e: PointerEvent) => {
      if (!dragging) return;
      dragging = false;
      clearEngageTimer();
      if (rafId !== 0) {
        cancelAnimationFrame(rafId);
        rafId = 0;
      }
      cachedRect = null;
      el.releasePointerCapture(e.pointerId);
    };

    el.addEventListener("pointerdown", onPointerDown);
    el.addEventListener("pointermove", onPointerMove);
    el.addEventListener("pointerup", onPointerUp);
    el.addEventListener("pointercancel", onPointerUp);
    return () => {
      if (rafId !== 0) cancelAnimationFrame(rafId);
      el.removeEventListener("pointerdown", onPointerDown);
      el.removeEventListener("pointermove", onPointerMove);
      el.removeEventListener("pointerup", onPointerUp);
      el.removeEventListener("pointercancel", onPointerUp);
    };
    // `mapping` is rebuilt every render; depending on its identity here
    // would re-bind listeners every render. We pull out the primitive
    // fields it's built from instead.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [param, effectiveMin, effectiveMax, unity, reverse, setSlider]);

  // Tint follows the thumb's position so the color sweep matches the
  // drag, not the value — keeps the gradient coherent for reverse and
  // unity-anchored channels (operator sees teal at the top regardless
  // of direction or anchor).
  const tintStyle = { "--slider-tint": tintAt(t) } as CSSProperties;

  // Descriptive tooltip on the LABEL only — hovering the track during
  // drag would otherwise trigger the tooltip mid-interaction. Label is
  // the natural "what is this?" hover target. data-dd-tooltip-wide
  // wraps the longer copy across multiple lines (vs the default nowrap
  // chrome tooltips).
  const tooltip = tooltipFor(param);

  return (
    <div className="slider-group" data-param={param} style={tintStyle}>
      <div
        className="slider-label"
        {...(tooltip
          ? { "data-dd-tooltip": tooltip, "data-dd-tooltip-wide": "" }
          : {})}
      >
        {label}
      </div>
      <div className="slider-track" ref={trackRef}>
        <div className="slider-fill" style={{ height: `${pct}%` }} />
        <div className="slider-thumb" style={{ bottom: `${pct}%` }} />
      </div>
      <div className="slider-value">{value.toFixed(2)}</div>
      {kbd && <kbd className="desktop-only">{kbd}</kbd>}
    </div>
  );
}
