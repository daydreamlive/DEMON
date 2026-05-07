"use client";

import { useCallback, useEffect, useRef, useState } from "react";

import { useLoraStore } from "@/store/useLoraStore";
import { usePerformanceStore } from "@/store/usePerformanceStore";
import {
  LORA_DEFAULT_STRENGTH_FRACTION,
  LORA_SIDE_VISIBLE_FLOOR,
  LORA_SLIDER_MAX,
} from "@/types/engine";

import { RemixHint } from "./RemixHint";

type Side = "top" | "left" | "right";

interface Props {
  side: Side;
}

const TOP_PARAM = "denoise";
const TOP_MAX = 1.0;

// localStorage key for the one-time "drag the strings" affordance hint
// shown beneath the Remix Strength slider AND on each non-empty LoRA
// strength bar. Once the user successfully drags any of them, the hint
// is dismissed forever across all three sliders. Key kept as legacy
// (`remix-strength`) to avoid resetting users who already learned.
const HINT_DISMISSED_KEY = "remix-strength:dismissed";

// Custom event used to keep the three <DesktopEdgeDrag /> instances in
// sync: when one dismisses the hint, the others need to hear about it
// so they can hide their hint *immediately* without waiting for a reload.
const HINT_DISMISSED_EVENT = "dd:drag-hint-dismissed";

// Transparent pointer overlay sitting on top of an .install-edge-* HUD bar
// so the perimeter ribbons act as draggable sliders on desktop.
//
//   side="top"   →  horizontal drag, controls denoise (remix strength).
//   side="left"  →  vertical drag, controls the FIRST enabled LoRA strength.
//   side="right" →  vertical drag, controls the SECOND enabled LoRA strength.
//
// The LoRA-side variants read the bound id from `.install-edge-{side}`'s
// data-bar attribute (set by useEdgeLoraBinding). When no LoRA is bound,
// data-empty="true" switches the cursor back to default and the
// pointerdown handler early-returns, so a drag on an empty slot is a
// no-op. The overlay itself stays mounted and accepts hover events so
// the RemixHint keeps rendering through "connecting" / catalog updates.
//
// Each side also renders <RemixHint /> as a child (until the user's first
// successful drag dismisses it) — the orientation prop tells the hint
// whether to track X or Y, and the side prop controls which side of the
// bar the hint sits on.
export function DesktopEdgeDrag({ side }: Props) {
  const trackRef = useRef<HTMLDivElement | null>(null);
  // Re-render when LoRA enabled/strengths change so aria values + the
  // data-empty attribute stay accurate. Drag handlers themselves read
  // the latest store via getState().
  const enabled = useLoraStore((s) => s.enabled);
  const strengths = useLoraStore((s) => s.strengths);
  // Reads the target (intent) so the visual responds immediately to the
  // drag, even when smooth-mode is on (sliderValues lags via tween).
  const denoise = usePerformanceStore((s) => s.sliderTargets[TOP_PARAM] ?? 0);

  const isTop = side === "top";
  const orientation: "horizontal" | "vertical" = isTop
    ? "horizontal"
    : "vertical";

  // Hint state. `dismissed` reads from localStorage on mount; `hover` /
  // `dragging` drive the live label. Position is driven by the slider's
  // own value (see valueFraction below) so the hint always sits at the
  // head — no separate cursor state needed.
  const [hintDismissed, setHintDismissed] = useState(true); // SSR-safe default
  const [hover, setHover] = useState(false);
  const [dragging, setDragging] = useState(false);

  useEffect(() => {
    if (typeof localStorage === "undefined") {
      setHintDismissed(false);
      return;
    }
    try {
      setHintDismissed(localStorage.getItem(HINT_DISMISSED_KEY) === "1");
    } catch {
      setHintDismissed(false);
    }
  }, []);

  // Cross-instance dismissal sync: when one slider's drag dismisses the
  // hint, the other two should hide their hint without waiting for the
  // page to reload.
  useEffect(() => {
    const onDismiss = () => setHintDismissed(true);
    document.addEventListener(HINT_DISMISSED_EVENT, onDismiss);
    return () =>
      document.removeEventListener(HINT_DISMISSED_EVENT, onDismiss);
  }, []);

  const slotIndex = side === "left" ? 0 : side === "right" ? 1 : -1;
  const loraId =
    slotIndex >= 0 ? Array.from(enabled)[slotIndex] ?? null : null;
  const isEmpty = side !== "top" && loraId === null;

  const value = (() => {
    if (side === "top") return denoise;
    // Empty side bar (no LoRA bound yet) — return the same default the
    // canvas paints via --fill (see useEdgeLoraBinding), so the hint
    // head sits where the ribbon visually ends and ARIA aria-valuenow
    // matches what the user sees.
    if (loraId === null) return LORA_DEFAULT_STRENGTH_FRACTION * LORA_SLIDER_MAX;
    return strengths[loraId] ?? 0;
  })();
  const max = side === "top" ? TOP_MAX : LORA_SLIDER_MAX;
  // Drives RemixHint position so the hint always sits at the head (the
  // slider's current value), not at a stale cursor position. Side bars
  // get the same visibility floor the canvas uses so the hint stays
  // attached to the ribbon's visible end when strength is below the
  // floor (otherwise the hint floats below the ribbon).
  const rawFraction = max > 0 ? Math.max(0, Math.min(1, value / max)) : 0;
  const valueFraction = isTop
    ? rawFraction
    : Math.max(rawFraction, LORA_SIDE_VISIBLE_FLOOR);

  // Capture the slider's value at pointerdown so we can detect whether
  // the drag actually moved the value (the dismissal contract requires
  // a value-changing drag — a stationary tap-and-release shouldn't
  // count as "the user learned to drag").
  const readCurrentValue = useCallback((): number => {
    if (isTop) {
      return usePerformanceStore.getState().sliderTargets[TOP_PARAM] ?? 0;
    }
    const id = Array.from(useLoraStore.getState().enabled)[slotIndex];
    return id ? useLoraStore.getState().strengths[id] ?? 0 : 0;
  }, [isTop, slotIndex]);

  useEffect(() => {
    const el = trackRef.current;
    if (!el) return;

    // Cache the host rect at pointerdown and reuse for the lifetime of
    // the drag. Without this, every pointermove called
    // getBoundingClientRect(), forcing a synchronous layout flush per
    // event. RAF-coalesce the store write so a 60–120 evt/sec pointer
    // stream commits at most once per frame.
    let isDragging = false;
    let dragInitialValue = 0;
    let cachedRect: DOMRect | null = null;
    let pendingClientX = 0;
    let pendingClientY = 0;
    let rafId = 0;

    const commitValue = () => {
      if (!cachedRect) return;
      if (isTop) {
        const t = (pendingClientX - cachedRect.left) / cachedRect.width;
        usePerformanceStore
          .getState()
          .setSlider(TOP_PARAM, Math.max(0, Math.min(1, t)) * TOP_MAX);
        return;
      }
      const t = 1 - (pendingClientY - cachedRect.top) / cachedRect.height;
      const ids = Array.from(useLoraStore.getState().enabled);
      const id = ids[slotIndex];
      if (!id) return;
      useLoraStore
        .getState()
        .setStrength(id, Math.max(0, Math.min(1, t)) * LORA_SLIDER_MAX);
    };

    const flush = () => {
      rafId = 0;
      if (!cachedRect) return;
      if (isDragging) commitValue();
    };

    const schedule = () => {
      if (rafId === 0) rafId = requestAnimationFrame(flush);
    };

    const onPointerDown = (e: PointerEvent) => {
      if (e.button !== 0 && e.pointerType === "mouse") return;
      // Defensive: if no LoRA is bound (or a fast LoRA disable race lands
      // `data-empty` mid-press), don't capture — drag is a no-op.
      if (el.dataset.empty === "true") return;
      isDragging = true;
      setDragging(true);
      dragInitialValue = readCurrentValue();
      cachedRect = el.getBoundingClientRect();
      pendingClientX = e.clientX;
      pendingClientY = e.clientY;
      el.setPointerCapture(e.pointerId);
      // Commit immediately so a click-without-drag still moves the value.
      commitValue();
    };
    const onPointerMove = (e: PointerEvent) => {
      if (!isDragging) return;
      pendingClientX = e.clientX;
      pendingClientY = e.clientY;
      schedule();
    };
    const onPointerUp = (e: PointerEvent) => {
      if (!isDragging) return;
      isDragging = false;
      if (rafId !== 0) {
        cancelAnimationFrame(rafId);
        rafId = 0;
      }
      el.releasePointerCapture(e.pointerId);
      setDragging(false);
      // First successful (value-changing) drag dismisses the hint forever.
      // Broadcast to sibling instances so all three hide together.
      const finalValue = readCurrentValue();
      if (finalValue !== dragInitialValue) {
        try {
          if (typeof localStorage !== "undefined") {
            localStorage.setItem(HINT_DISMISSED_KEY, "1");
          }
        } catch {}
        setHintDismissed(true);
        document.dispatchEvent(new CustomEvent(HINT_DISMISSED_EVENT));
      }
      cachedRect = null;
    };

    const onPointerEnter = () => {
      if (el.dataset.empty === "true") return;
      setHover(true);
    };
    const onPointerLeave = () => {
      setHover(false);
    };

    el.addEventListener("pointerdown", onPointerDown);
    el.addEventListener("pointermove", onPointerMove);
    el.addEventListener("pointerup", onPointerUp);
    el.addEventListener("pointercancel", onPointerUp);
    el.addEventListener("pointerenter", onPointerEnter);
    el.addEventListener("pointerleave", onPointerLeave);
    return () => {
      if (rafId !== 0) cancelAnimationFrame(rafId);
      el.removeEventListener("pointerdown", onPointerDown);
      el.removeEventListener("pointermove", onPointerMove);
      el.removeEventListener("pointerup", onPointerUp);
      el.removeEventListener("pointercancel", onPointerUp);
      el.removeEventListener("pointerenter", onPointerEnter);
      el.removeEventListener("pointerleave", onPointerLeave);
    };
  }, [readCurrentValue, isTop, slotIndex]);

  // Always show the hint until dismissed — even for empty LoRA slots, so
  // the user has a stable visual anchor through "connecting" / catalog
  // updates. Drag on an empty slot is a no-op (data-empty guards
  // pointerdown), but the hint stays visible so nothing "disappears".
  const showHint = !hintDismissed;

  return (
    <div
      ref={trackRef}
      className="desktop-edge-drag"
      data-side={side}
      data-empty={isEmpty ? "true" : "false"}
      role="slider"
      aria-label={
        side === "top"
          ? "Remix strength"
          : side === "left"
            ? "LoRA 1 strength"
            : "LoRA 2 strength"
      }
      aria-valuemin={0}
      aria-valuemax={max}
      aria-valuenow={value}
    >
      {showHint && (
        <RemixHint
          hover={hover}
          dragging={dragging}
          valueFraction={valueFraction}
          orientation={orientation}
          side={side === "right" ? "right" : "left"}
          draggingLabel={
            side === "left"
              ? "— mosh —"
              : side === "right"
                ? "— vibe —"
                : "— rave —"
          }
        />
      )}
    </div>
  );
}
