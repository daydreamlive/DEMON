"use client";

import type { CSSProperties } from "react";
import { useEffect, useMemo, useRef, useState } from "react";

import { useTactileSlider } from "@/hooks/useTactileSlider";
import { tToValue, valueToT } from "@/lib/sliderMapping";
import { usePerformanceStore } from "@/store/usePerformanceStore";
import { SLIDER_META } from "@/types/engine";

import { KnobCap } from "./KnobCap";
import { tooltipFor } from "./SliderTile";

// Rotary control matching the visual vocabulary of inShaper / GrainDust:
// 270° sweep from 7:30 to 4:30 (default DAW convention), an arc fill
// behind the body that shows the Daydream gradient at the current
// value, and a single indicator notch on the body rim.
//
// Interaction mirrors SliderGroup so the muscle memory carries over:
//  - Vertical drag (up = increase). PIXELS_PER_RANGE pixels = full sweep.
//  - Shift = fine (×5 sensitivity).
//  - Double-click = reset to default.
//  - Mouse wheel = ±SCROLL_STEP.
//  - Same store contract: reads `sliderTargets[param]`, writes via
//    `setSlider(param, …)`, resets via `resetSlider(param)`. Same
//    `useTactileSlider` hook for landmark haptics.
//
// Used for CORE + MOD knobs (continuous "tweak with one hand" params).
// VOICE keeps vertical faders because those channels are mix-shaped
// (Sound Particles uses faders for their MIXER too).

interface Props {
  param: string;
  label: string;
  /** Override max for ad-hoc params not in SLIDER_META. */
  max?: number;
  min?: number;
  reverse?: boolean;
  /** Pins this value to the rail midpoint for piecewise-linear mapping
   * (matches SliderGroup's unity behavior). */
  unity?: number;
  kbd?: string;
}

// Arc geometry. 270° sweep starting from 7:30 (-135°) going clockwise
// to 4:30 (+135°). Standard DAW convention.
const ARC_START_DEG = -135;
const ARC_END_DEG = 135;
const ARC_RANGE_DEG = ARC_END_DEG - ARC_START_DEG;

// Drag sensitivity. 280px of vertical motion = full sweep — matches
// Ableton's default ratio for a ~44px knob cap. Shift = fine (×5).
const PIXELS_PER_RANGE = 280;
const FINE_DIVISOR = 5;
const SCROLL_STEP = 0.01;
const DBLCLICK_MS = 350;

// Palette stops mirror the .slider-fill gradient. Same array as
// SliderGroup so a knob's arc color matches the corresponding fader
// fill at the same value.
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

// SVG arc helper. Given a center, radius, and start/end angles in
// degrees (0 = right, 90 = down, -90 = up — SVG convention), build a
// path `d` string. Handles both small and large arcs.
function arcPath(
  cx: number,
  cy: number,
  r: number,
  startDeg: number,
  endDeg: number,
): string {
  // Rotate by -90 so 0° lands at the top in our visual coordinates.
  const startRad = ((startDeg - 90) * Math.PI) / 180;
  const endRad = ((endDeg - 90) * Math.PI) / 180;
  const x1 = cx + r * Math.cos(startRad);
  const y1 = cy + r * Math.sin(startRad);
  const x2 = cx + r * Math.cos(endRad);
  const y2 = cy + r * Math.sin(endRad);
  const largeArc = Math.abs(endDeg - startDeg) > 180 ? 1 : 0;
  return `M ${x1} ${y1} A ${r} ${r} 0 ${largeArc} 1 ${x2} ${y2}`;
}

export function Knob({ param, label, max, min, reverse, unity, kbd }: Props) {
  const meta = SLIDER_META[param];
  const effectiveMax = max ?? meta?.max ?? 1.0;
  const effectiveMin = min ?? meta?.min ?? 0;
  const integerDisplay = (meta?.step ?? 0) >= 1;
  const formatValue = (v: number) =>
    integerDisplay ? String(Math.round(v)) : v.toFixed(2);

  // Stable reference — without useMemo this object identity changed on
  // every render, which made the drag useEffect (deps: [..., mapping])
  // tear down and rebuild mid-drag. The cleanup removed the
  // document-level pointermove/pointerup listeners while the user was
  // still holding the pointer, killing the drag silently.
  const mapping = useMemo(
    () => ({
      min: effectiveMin,
      max: effectiveMax,
      unity,
      reverse: !!reverse,
    }),
    [effectiveMin, effectiveMax, unity, reverse],
  );

  const value = usePerformanceStore((s) => s.sliderTargets[param] ?? 0);
  const setSlider = usePerformanceStore((s) => s.setSlider);
  const bodyRef = useRef<HTMLDivElement | null>(null);

  // Double-click on the value cell swaps it for a text input. Same
  // contract as SliderGroup.
  const [editing, setEditing] = useState(false);
  const [editText, setEditText] = useState("");

  const startEdit = () => {
    setEditText(formatValue(value));
    setEditing(true);
  };
  const commitEdit = () => {
    const parsed = parseFloat(editText);
    if (!Number.isNaN(parsed)) setSlider(param, parsed);
    setEditing(false);
  };
  const cancelEdit = () => setEditing(false);

  useTactileSlider({ param, mapping });

  const t = valueToT(value, mapping);
  // Indicator angle in our local coord system (0° = top, +cw).
  const indicatorDeg = ARC_START_DEG + t * ARC_RANGE_DEG;
  const fillTint = tintAt(t);

  useEffect(() => {
    const el = bodyRef.current;
    if (!el) return;

    let dragging = false;
    let startClientY = 0;
    let startT = 0;
    let fine = false;
    let pendingClientY = 0;
    let rafId = 0;
    let lastDownAt = 0;
    let prevBodyCursor = "";

    const commit = (clientY: number) => {
      const dy = startClientY - clientY;
      const divisor = fine ? PIXELS_PER_RANGE * FINE_DIVISOR : PIXELS_PER_RANGE;
      const tFrac = Math.max(0, Math.min(1, startT + dy / divisor));
      setSlider(param, tToValue(tFrac, mapping));
    };

    const flush = () => {
      rafId = 0;
      if (!dragging) return;
      commit(pendingClientY);
    };

    const onDocPointerMove = (e: PointerEvent) => {
      if (!dragging) return;
      pendingClientY = e.clientY;
      fine = e.shiftKey;
      if (!rafId) rafId = requestAnimationFrame(flush);
    };

    const onDocPointerUp = () => {
      if (!dragging) return;
      // Cancel any rAF in flight, then do a final synchronous commit so
      // the last frame of motion before release isn't dropped. Without
      // this the value could land ~1 frame stale relative to where the
      // cursor actually was at release time — looks like the knob
      // "jumps" when the user lets go.
      if (rafId) {
        cancelAnimationFrame(rafId);
        rafId = 0;
      }
      commit(pendingClientY);
      dragging = false;
      document.body.style.cursor = prevBodyCursor;
      document.removeEventListener("pointermove", onDocPointerMove);
      document.removeEventListener("pointerup", onDocPointerUp);
      document.removeEventListener("pointercancel", onDocPointerUp);
    };

    const onPointerDown = (e: PointerEvent) => {
      // Right-click reserved for MIDI-learn (matches SliderGroup).
      if (e.button !== 0) return;
      const now = performance.now();
      if (now - lastDownAt < DBLCLICK_MS) {
        usePerformanceStore.getState().resetSlider(param);
        lastDownAt = 0;
        return;
      }
      lastDownAt = now;
      dragging = true;
      startClientY = e.clientY;
      // Seed pendingClientY so a release without any pointermove (a
      // quick tap) commits the same value the drag started with —
      // otherwise the final commit on pointerup would read the stale
      // 0 initializer and snap the slider to its bottom.
      pendingClientY = e.clientY;
      startT = valueToT(
        usePerformanceStore.getState().sliderTargets[param] ?? 0,
        mapping,
      );
      fine = e.shiftKey;
      // Keep the cursor in ns-resize the whole drag so it doesn't
      // visually wander off the knob — without this the cursor reverts
      // to the underlying element's cursor (often default) whenever it
      // crosses panel chrome, which felt like a jump on release.
      prevBodyCursor = document.body.style.cursor;
      document.body.style.cursor = "ns-resize";
      // Document-level listeners ride along until pointerup. Avoids
      // setPointerCapture which silently fails on a few combinations
      // (transformed ancestors + element pickup mid-drag), which was
      // the root cause of the user's "knobs won't drag" reports across
      // Wave 9 / Wave 10.
      document.addEventListener("pointermove", onDocPointerMove);
      document.addEventListener("pointerup", onDocPointerUp);
      document.addEventListener("pointercancel", onDocPointerUp);
      e.preventDefault();
    };

    // Scroll-wheel adjustment. Same pattern as SliderGroup's wheel
    // handler — small step per tick, Shift = fine.
    const onWheel = (e: WheelEvent) => {
      e.preventDefault();
      const dir = e.deltaY > 0 ? -1 : 1;
      const step = e.shiftKey ? SCROLL_STEP / FINE_DIVISOR : SCROLL_STEP;
      const current = valueToT(
        usePerformanceStore.getState().sliderTargets[param] ?? 0,
        mapping,
      );
      const next = Math.max(0, Math.min(1, current + dir * step));
      setSlider(param, tToValue(next, mapping));
    };

    el.addEventListener("pointerdown", onPointerDown);
    el.addEventListener("wheel", onWheel, { passive: false });
    return () => {
      el.removeEventListener("pointerdown", onPointerDown);
      el.removeEventListener("wheel", onWheel);
      document.removeEventListener("pointermove", onDocPointerMove);
      document.removeEventListener("pointerup", onDocPointerUp);
      document.removeEventListener("pointercancel", onDocPointerUp);
      if (rafId) cancelAnimationFrame(rafId);
      // If we tore down mid-drag (HMR, unmount), restore the cursor.
      if (dragging) document.body.style.cursor = prevBodyCursor;
    };
  }, [param, mapping, setSlider]);

  const tooltip = tooltipFor(param);
  const style = { "--knob-tint": fillTint } as CSSProperties;
  // Indicator endpoints (rim → inward). Pre-computed so the JSX stays
  // tidy and the angle math doesn't repeat.
  const indRad = ((indicatorDeg - 90) * Math.PI) / 180;
  const indCos = Math.cos(indRad);
  const indSin = Math.sin(indRad);
  // Slight inset so the notch reads as an etched groove, not a stick
  // poking off the rim — starts at r=14 (just inside the body edge)
  // and goes to r=8 (about halfway to center).
  const indX1 = 24 + 14 * indCos;
  const indY1 = 24 + 14 * indSin;
  const indX2 = 24 + 8 * indCos;
  const indY2 = 24 + 8 * indSin;

  return (
    <div
      className="knob-group"
      style={style}
      data-dd-tooltip={tooltip}
      data-dd-tooltip-wide={tooltip ? "" : undefined}
      data-dd-tooltip-title={label}
    >
      <div className="knob-label" title={tooltip}>
        {label}
      </div>
      <div
        ref={bodyRef}
        className="knob-body"
        role="slider"
        aria-label={label}
        aria-valuemin={effectiveMin}
        aria-valuemax={effectiveMax}
        aria-valuenow={value}
        tabIndex={0}
      >
        <svg
          className="knob-svg"
          viewBox="0 0 48 48"
          width="48"
          height="48"
          aria-hidden="true"
        >
          {/* Background arc (full sweep, dim) — sits in the inset
              channel around the cap, like the engraved scale on a
              hardware knob. Drawn before the cap so it reads as
              engraved INTO the panel underneath. */}
          <path
            d={arcPath(24, 24, 21, ARC_START_DEG, ARC_END_DEG)}
            className="knob-arc-bg"
            fill="none"
          />
          {/* Value arc — from 0 to current t. Color from the per-value
              gradient stop, matching the corresponding fader fill. */}
          <path
            d={arcPath(24, 24, 21, ARC_START_DEG, indicatorDeg)}
            className="knob-arc-fill"
            fill="none"
            stroke="var(--knob-tint)"
          />
          {/* Shared 3D cap chrome: shadow + body + beveled rim. */}
          <KnobCap />
          {/* Indicator notch — etched groove from rim toward center at
              the indicator angle. Drawn LAST so it sits on top of the
              cap. Two strokes: a dark recessed line, then a 1px tint
              line on top so the indicator catches the per-value color
              and reads as the active "pointing finger" of the knob. */}
          <line
            x1={indX1}
            y1={indY1}
            x2={indX2}
            y2={indY2}
            className="knob-indicator-shadow"
          />
          <line
            x1={indX1}
            y1={indY1}
            x2={indX2}
            y2={indY2}
            className="knob-indicator"
            stroke="var(--knob-tint)"
          />
        </svg>
      </div>
      <div className="knob-value" onDoubleClick={startEdit}>
        {editing ? (
          <input
            type="text"
            className="knob-value-input"
            value={editText}
            autoFocus
            onChange={(e) => setEditText(e.target.value)}
            onBlur={commitEdit}
            onKeyDown={(e) => {
              if (e.key === "Enter") commitEdit();
              else if (e.key === "Escape") cancelEdit();
            }}
          />
        ) : (
          formatValue(value)
        )}
      </div>
      {kbd && <kbd className="knob-kbd">{kbd}</kbd>}
    </div>
  );
}
