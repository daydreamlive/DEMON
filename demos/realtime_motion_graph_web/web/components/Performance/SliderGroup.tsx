"use client";

import type { CSSProperties } from "react";
import { useEffect, useRef } from "react";

import { usePerformanceStore } from "@/store/usePerformanceStore";
import { SLIDER_META } from "@/types/engine";

// Vertical slider matching DEMON's CSS layout (.slider-track + .slider-fill
// + .slider-thumb). Click + drag on the track to set; the value reads from
// usePerformanceStore. LoRA sliders (param starting with `lora_str_`)
// supply their own max via prop because they aren't in SLIDER_META.

interface Props {
  param: string;
  label: string;
  /** Override max + step for ad-hoc sliders (LoRA strength). */
  max?: number;
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

export function SliderGroup({ param, label, max, kbd }: Props) {
  const meta = SLIDER_META[param];
  const effectiveMax = max ?? meta?.max ?? 1.0;
  // Read the user's target (instant), not the smoothed sent value, so
  // dragging tracks the cursor without smoothing lag.
  const value = usePerformanceStore(
    (s) => s.sliderTargets[param] ?? 0,
  );
  const setSlider = usePerformanceStore((s) => s.setSlider);
  const trackRef = useRef<HTMLDivElement | null>(null);

  const fraction = effectiveMax > 0 ? value / effectiveMax : 0;
  const pct = Math.max(0, Math.min(1, fraction)) * 100;

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

    const commit = () => {
      if (!cachedRect) return;
      const t = 1 - (pendingClientY - cachedRect.top) / cachedRect.height;
      setSlider(param, Math.max(0, Math.min(1, t)) * effectiveMax);
    };

    // RAF-coalesce store writes during drag: pointermove fires 60–120
    // times/sec, but the render loop can only show ~60 frames/sec.
    // Coalescing also collapses the cascade of React re-renders triggered
    // by setSlider() into one per frame.
    const flush = () => {
      rafId = 0;
      if (!dragging) return;
      commit();
    };

    const onPointerDown = (e: PointerEvent) => {
      // Right-click reserved for MIDI-learn (Phase 9).
      if (e.button !== 0) return;
      dragging = true;
      cachedRect = el.getBoundingClientRect();
      el.setPointerCapture(e.pointerId);
      pendingClientY = e.clientY;
      // Commit immediately so a click-without-drag still moves the value.
      commit();
    };
    const onPointerMove = (e: PointerEvent) => {
      if (!dragging) return;
      pendingClientY = e.clientY;
      if (rafId === 0) rafId = requestAnimationFrame(flush);
    };
    const onPointerUp = (e: PointerEvent) => {
      if (!dragging) return;
      dragging = false;
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
  }, [param, effectiveMax, setSlider]);

  const tintStyle = { "--slider-tint": tintAt(fraction) } as CSSProperties;

  return (
    <div className="slider-group" data-param={param} style={tintStyle}>
      <div className="slider-label">{label}</div>
      <div className="slider-track" ref={trackRef}>
        <div className="slider-fill" style={{ height: `${pct}%` }} />
        <div className="slider-thumb" style={{ bottom: `${pct}%` }} />
      </div>
      <div className="slider-value">{value.toFixed(2)}</div>
      {kbd && <kbd className="desktop-only">{kbd}</kbd>}
    </div>
  );
}
