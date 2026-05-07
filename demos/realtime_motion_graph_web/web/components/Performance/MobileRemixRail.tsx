"use client";

import { useCallback, useEffect, useRef } from "react";

import { usePerformanceStore } from "@/store/usePerformanceStore";

const PARAM = "denoise";
const MAX = 1.0;

// Always-visible vertical Remix Strength rail on the left edge. The visual
// is the existing writhing-ribbon system rendered inside .install-edge-left
// (same look as the desktop top edge); this component just provides the
// rotated label, the value readout, and a wide pointer-capturing drag zone
// stacked over it.
export function MobileRemixRail() {
  const value = usePerformanceStore(
    (s) => s.sliderTargets[PARAM] ?? 0,
  );
  const setSlider = usePerformanceStore((s) => s.setSlider);
  const trackRef = useRef<HTMLDivElement | null>(null);

  // Mirror denoise into --fill on .install-edge-left so the existing
  // ribbon system writhes with the slider value (same wiring the desktop
  // top edge uses, just on the side bar instead).
  useEffect(() => {
    const edge = document.querySelector<HTMLElement>(".install-edge-left");
    if (!edge) return;
    const frac = Math.max(0, Math.min(1, MAX > 0 ? value / MAX : 0));
    edge.style.setProperty("--fill", frac.toString());
  }, [value]);

  const setFromClientY = useCallback(
    (clientY: number) => {
      const el = trackRef.current;
      if (!el) return;
      const rect = el.getBoundingClientRect();
      const t = 1 - (clientY - rect.top) / rect.height;
      setSlider(PARAM, Math.max(0, Math.min(1, t)) * MAX);
    },
    [setSlider],
  );

  useEffect(() => {
    const el = trackRef.current;
    if (!el) return;
    let dragging = false;

    const onPointerDown = (e: PointerEvent) => {
      if (e.button !== 0 && e.pointerType === "mouse") return;
      dragging = true;
      el.setPointerCapture(e.pointerId);
      setFromClientY(e.clientY);
    };
    const onPointerMove = (e: PointerEvent) => {
      if (!dragging) return;
      setFromClientY(e.clientY);
    };
    const onPointerUp = (e: PointerEvent) => {
      if (!dragging) return;
      dragging = false;
      el.releasePointerCapture(e.pointerId);
    };

    el.addEventListener("pointerdown", onPointerDown);
    el.addEventListener("pointermove", onPointerMove);
    el.addEventListener("pointerup", onPointerUp);
    el.addEventListener("pointercancel", onPointerUp);
    return () => {
      el.removeEventListener("pointerdown", onPointerDown);
      el.removeEventListener("pointermove", onPointerMove);
      el.removeEventListener("pointerup", onPointerUp);
      el.removeEventListener("pointercancel", onPointerUp);
    };
  }, [setFromClientY]);

  return (
    <div
      className="remix-rail"
      data-param={PARAM}
      role="slider"
      aria-label="Remix strength"
      aria-valuemin={0}
      aria-valuemax={MAX}
      aria-valuenow={value}
    >
      <span className="remix-rail-label">Remix Strength</span>
      <div ref={trackRef} className="remix-rail-drag" />
      <span className="remix-rail-value">{value.toFixed(2)}</span>
    </div>
  );
}
