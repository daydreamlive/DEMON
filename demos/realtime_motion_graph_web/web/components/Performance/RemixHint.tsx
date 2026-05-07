"use client";

import type { CSSProperties } from "react";
import { useEffect, useState } from "react";

// First-time-user affordance for the three "string" sliders (Remix
// Strength on top, LoRA-1/2 on the sides). All three are non-traditional
// — colored ribbon strings with no thumb / fill / readout — so new users
// have no signal that the strings are draggable. This monospace hint
// sits at the slider's *head* (its current value) along the drag axis,
// morphs its copy by interaction state, and disappears forever after the
// user's first successful drag (persisted in localStorage).
//
// Position contract (don't change without re-reading this comment):
//   - The hint is `position: absolute` inside the slider's hit-area
//     wrapper (.desktop-edge-drag[data-side]). It sits *inside* that
//     wrapper (not floating beside it) so accidental clicks on the
//     hint still hit the drag overlay's pointerdown handler — pointer-
//     events: none on the hint lets the click pass to the parent.
//   - Position is driven by `valueFraction` (0..1, 0 = min, 1 = max) so
//     it always points at where the slider's value currently is — not
//     the cursor. During drag, the value updates from cursor input, so
//     the hint follows naturally without separate cursor tracking.
//   - Horizontal slider: `left: <pct>%` where pct = valueFraction*100.
//     `transform: translate(-50%, …)` handles centring + the idle bob.
//     Keeping the % in `left` (not `transform`) is critical — % inside
//     `translate(...)` resolves against the element itself, not the
//     wrapper, and the math breaks.
//   - Vertical slider: `top: <pct>%` where pct = (1 - valueFraction)*100,
//     because the slider fills bottom→top (value 0 = bottom, max = top,
//     mirroring the `bottom: pct%` convention used by .slider-thumb).
//     Anchored at the inward edge of the drag overlay so the text reads
//     toward the screen centre while still inside the click area.
//     The arrow points *at* the bar by default (← for left bar, → for
//     right bar) so the hint visually labels the ribbon as the thing to
//     grab. While dragging, the arrow flips to ↕ to indicate the live
//     drag axis. Bob is on X, toward the bar.

type Orientation = "horizontal" | "vertical";

interface Props {
  /** Pointer is inside the slider's hit area. */
  hover: boolean;
  /** Pointer is captured (mouse-down → up). */
  dragging: boolean;
  /** 0..1 normalized slider value (0 = min, 1 = max). The hint sits at
   * this position along the drag axis so it always points at the head. */
  valueFraction: number;
  /** Drag axis. Vertical sliders need a different position formula and
   * a different arrow glyph. */
  orientation: Orientation;
  /** For vertical sliders: which side of the screen the bar lives on,
   * so the hint can sit on the *inside* (toward the centre of the
   * viewport) rather than off-screen. */
  side?: "left" | "right";
  /** Copy shown while the user is actively dragging. Differs per slider
   * so each "string" has its own personality (rave / mosh / vibe). */
  draggingLabel?: string;
}

export function RemixHint({
  hover,
  dragging,
  valueFraction,
  orientation,
  side = "left",
  draggingLabel = "— rave —",
}: Props) {
  const [pulse, setPulse] = useState(0);

  // Idle pulse: ~4.4s sine wave (period = 2π · 700 ms) on opacity + a tiny
  // bob. Cancelled the moment the user hovers or drags, restarted when
  // both go false.
  useEffect(() => {
    if (hover || dragging) {
      setPulse(0);
      return;
    }
    let raf = 0;
    const start = performance.now();
    const tick = (t: number) => {
      const p = (Math.sin((t - start) / 700) + 1) / 2;
      setPulse(p);
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [hover, dragging]);

  // Idle/hover: the arrow points *at* the bar (← for left, → for right
  // or top horizontal) so the hint visually labels the ribbon as the
  // thing to grab. Dragging on a vertical bar swaps to ↕ to show the
  // live drag axis; horizontal top keeps the personality label as-is.
  const isVerticalLeft = orientation === "vertical" && side === "left";
  const idleArrow = isVerticalLeft ? "←" : "→";
  // Arrow on the left of the verb only for the left bar — that's the
  // side that needs the arrow facing back at the bar.
  const arrowBefore = isVerticalLeft;
  const compose = (verb: string, arrow: string) =>
    arrowBefore ? `${arrow} ${verb}` : `${verb} ${arrow}`;
  let label: string;
  let opacity: number;
  let bobAmt: number;
  if (dragging) {
    if (orientation === "vertical") {
      // Live drag indicator on the side facing the bar.
      label = compose(draggingLabel, "↕");
    } else {
      // Top horizontal: keep the personality label by itself.
      label = draggingLabel;
    }
    opacity = 0.9;
    bobAmt = 0;
  } else if (hover) {
    label = compose("pull", idleArrow);
    opacity = 0.9;
    bobAmt = 0;
  } else {
    label = compose("drag", idleArrow);
    opacity = 0.4 + pulse * 0.35;
    bobAmt = pulse * 6;
  }

  // Horizontal: hint below the bar at value's X, bobs down.
  // Vertical sides: anchored INSIDE the drag overlay at the inward edge
  //   (right edge for left bar, left edge for right bar) so clicks on
  //   the hint area still register on the drag overlay. Bob toward the
  //   bar — the hint nudges visually back at the slider it points at.
  // Vertical Y is inverted (value 0 = bottom = top:100%, max = top: 0%)
  //   so it lines up with the bottom-up fill direction of the LoRA bars.
  const horizontalPct = valueFraction * 100;
  const verticalPct = (1 - valueFraction) * 100;
  let style: CSSProperties;
  if (orientation === "horizontal") {
    style = {
      top: "calc(100% + 6px)",
      left: `${horizontalPct}%`,
      transform: `translate(-50%, ${bobAmt}px)`,
      opacity,
    };
  } else if (side === "left") {
    style = {
      left: "auto",
      right: "4px",
      top: `${verticalPct}%`,
      transform: `translate(${-bobAmt}px, -50%)`,
      opacity,
    };
  } else {
    style = {
      right: "auto",
      left: "4px",
      top: `${verticalPct}%`,
      transform: `translate(${bobAmt}px, -50%)`,
      opacity,
    };
  }

  return (
    <div className="remix-hint" style={style}>
      {label}
    </div>
  );
}
