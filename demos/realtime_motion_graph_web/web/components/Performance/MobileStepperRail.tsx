"use client";

import { useCallback, useEffect, useRef } from "react";

import { useLoraStore } from "@/store/useLoraStore";
import { usePerformanceStore } from "@/store/usePerformanceStore";
import { useSessionStore } from "@/store/useSessionStore";

// Vertical fader rail for the screen edges on phones in landscape mode.
// Wave 13.2 replaces the prior up/down stepper buttons with a single
// drag-to-value track + cap so the rail reads as a proper DAW fader.
// Tap anywhere on the track jumps the cap to that position; drag is
// continuous. Sublabels (top/bottom) stay for the blend rail's
// per-LoRA naming.
//
// `side="left"` mounts on the left edge; `side="right"` mirrors. The
// install-edge-{side} writhe canvas underneath keeps reading --fill so
// the perimeter ribbon animates in lockstep with the cap.

interface Props {
  side: "left" | "right";
  /** sliderTargets key. */
  param: string;
  /** Slider max — used to clamp + compute --fill on the edge writhe. */
  max: number;
  /** Header label (e.g. "Denoise"). */
  label: string;
  /** Optional per-end labels — rendered just inside the screen edge.
   *  For LoRA blend: top = the LoRA at fader-top (value=0 when
   *  invert=true), bottom = the LoRA at fader-bottom. */
  sublabelTop?: string;
  sublabelBottom?: string;
  /** Inverted means dragging UP decreases value. Use for the blend rail
   *  where "top of rail = LoRA A" = blend value 0. */
  invert?: boolean;
  /** Visual writhe-fill orientation: when true, --fill = 1 - frac (top
   *  of rail bright when value is 0). Match the rail's `invert`. */
  invertFill?: boolean;
  /** When true, pulses the top of the track as a "do this" affordance.
   *  Used by the per-song "drag to start" gate on the denoise rail. */
  pulseUp?: boolean;
}

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

export function MobileStepperRail({
  side,
  param,
  max,
  label,
  sublabelTop,
  sublabelBottom,
  invert = false,
  invertFill = false,
  pulseUp = false,
}: Props) {
  const value = usePerformanceStore(
    (s) => s.sliderDisplayOverride[param] ?? s.sliderTargets[param] ?? 0,
  );
  const setSlider = usePerformanceStore((s) => s.setSlider);

  // Mirror value into the edge's --fill so the perimeter ribbon
  // animates in lockstep with the fader cap.
  useEffect(() => {
    const edge = document.querySelector<HTMLElement>(`.install-edge-${side}`);
    if (!edge) return;
    const frac = max > 0 ? Math.max(0, Math.min(1, value / max)) : 0;
    edge.style.setProperty("--fill", (invertFill ? 1 - frac : frac).toString());
  }, [value, max, side, invertFill]);

  const trackRef = useRef<HTMLDivElement | null>(null);
  const draggingRef = useRef(false);

  const commitFromClient = useCallback(
    (clientY: number) => {
      const el = trackRef.current;
      if (!el) return;
      const rect = el.getBoundingClientRect();
      const innerH = Math.max(1, rect.height);
      // Pointer Y inside the track → fraction (1 = top).
      const fromTop = Math.max(0, Math.min(innerH, clientY - rect.top));
      let frac = 1 - fromTop / innerH;
      if (invert) frac = 1 - frac;
      const next = Math.max(0, Math.min(max, frac * max));
      const current =
        usePerformanceStore.getState().sliderTargets[param] ?? 0;
      if (Math.abs(next - current) < (max > 1 ? 0.01 : 0.001)) return;
      setSlider(param, next);
    },
    [invert, max, param, setSlider],
  );

  useEffect(() => {
    const el = trackRef.current;
    if (!el) return;
    const onDocMove = (e: PointerEvent) => {
      if (!draggingRef.current) return;
      commitFromClient(e.clientY);
    };
    const onDocUp = () => {
      draggingRef.current = false;
      document.removeEventListener("pointermove", onDocMove);
      document.removeEventListener("pointerup", onDocUp);
      document.removeEventListener("pointercancel", onDocUp);
    };
    const onDown = (e: PointerEvent) => {
      if (e.button !== 0 && e.pointerType === "mouse") return;
      draggingRef.current = true;
      commitFromClient(e.clientY);
      vibrate(6);
      document.addEventListener("pointermove", onDocMove);
      document.addEventListener("pointerup", onDocUp);
      document.addEventListener("pointercancel", onDocUp);
      e.preventDefault();
    };
    el.addEventListener("pointerdown", onDown);
    return () => {
      el.removeEventListener("pointerdown", onDown);
      document.removeEventListener("pointermove", onDocMove);
      document.removeEventListener("pointerup", onDocUp);
      document.removeEventListener("pointercancel", onDocUp);
    };
  }, [commitFromClient]);

  // Cap position as a 0..1 fraction from the BOTTOM of the track. When
  // invert=true, flip so the cap reads "more A = top" / "more B = bottom".
  const frac = max > 0 ? Math.max(0, Math.min(1, value / max)) : 0;
  const capFromBottom = invert ? 1 - frac : frac;
  const capBottomPct = `${capFromBottom * 100}%`;
  const fillBottomPct = `${capFromBottom * 100}%`;

  return (
    <div
      className={`fader-rail fader-rail--${side}`}
      data-param={param}
      role="slider"
      aria-label={label}
      aria-valuemin={0}
      aria-valuemax={max}
      aria-valuenow={value}
      aria-valuetext={`${label} ${value.toFixed(2)}`}
      data-gate={pulseUp ? "up" : undefined}
    >
      {sublabelTop && (
        <div className="fader-rail-sublabel fader-rail-sublabel--top">
          {sublabelTop}
        </div>
      )}
      <div className="fader-rail-label">{label}</div>
      <div ref={trackRef} className="fader-rail-track">
        <div className="fader-rail-fill" style={{ height: fillBottomPct }} />
        <div className="fader-rail-cap" style={{ bottom: capBottomPct }} />
      </div>
      {sublabelBottom && (
        <div className="fader-rail-sublabel fader-rail-sublabel--bottom">
          {sublabelBottom}
        </div>
      )}
    </div>
  );
}

// Thin wrappers — left rail drives Denoise; right rail drives the LoRA
// blend and resolves the paired LoRA names as sublabels.

export function MobileRemixStepper() {
  const remixStarted = usePerformanceStore((s) => s.remixStarted);
  const denoise = usePerformanceStore((s) => s.sliderTargets["denoise"] ?? 0);
  const sessionReady = useSessionStore((s) => s.status === "ready");
  useEffect(() => {
    if (!remixStarted && denoise > 0) {
      usePerformanceStore.getState().setRemixStarted(true);
    }
  }, [remixStarted, denoise]);
  return (
    <MobileStepperRail
      side="left"
      param="denoise"
      max={1.0}
      label="Strength"
      pulseUp={sessionReady && !remixStarted}
    />
  );
}

export function MobileLoraBlendStepper() {
  const enabled = useLoraStore((s) => s.enabled);
  const catalog = useLoraStore((s) => s.catalog);
  const ids = Array.from(enabled);
  while (ids.length < 2) {
    const next = catalog.find((c) => !ids.includes(c.id));
    if (!next) break;
    ids.push(next.id);
  }
  const nameOf = (id: string | undefined) =>
    id ? (catalog.find((c) => c.id === id)?.name ?? id) : null;
  const a = nameOf(ids[0]);
  const b = nameOf(ids[1]);

  return (
    <MobileStepperRail
      side="right"
      param="lora_blend"
      max={1.0}
      label="LoRA Blend"
      sublabelTop={a ?? undefined}
      sublabelBottom={b ?? undefined}
      invert
      invertFill
    />
  );
}
