"use client";

import { useEffect, useState } from "react";

// Portal-positioned tooltip for hero-bay controls (knobs + style
// faders). The CSS `::after` pseudo can't render a two-tone title +
// body the way DrawerHelpBar / HudHelpReadout chips do — `::after::
// first-line` chaining breaks Turbopack's CSS parser, and there's no
// other CSS-only way to style part of a pseudo's content. So for the
// hero bay we render a real React component with the same
// title-then-body structure those chips use.
//
// Hero-bay is the only surface that uses this — the in-panel tooltips
// + drawer help bar handle the rest. Scoped to `.hero-macros` via
// closest() check. The existing CSS `::after` is suppressed for
// hero-bay elements (see `.hero-macros [data-dd-tooltip-wide]::after
// { display: none }`).

interface HoverState {
  /** Center-x of the hovered element in viewport coords. */
  cx: number;
  /** Top edge of the hovered element in viewport coords. */
  top: number;
  title: string;
  text: string;
}

const HOVER_DELAY_MS = 200;

export function HeroMacrosTooltip() {
  const [state, setState] = useState<HoverState | null>(null);

  useEffect(() => {
    let pending: number | null = null;
    let lastEl: Element | null = null;

    const clearPending = () => {
      if (pending !== null) {
        window.clearTimeout(pending);
        pending = null;
      }
    };

    const hide = () => {
      clearPending();
      lastEl = null;
      setState(null);
    };

    const onMove = (e: PointerEvent) => {
      const target = e.target as Element | null;
      if (!target) {
        hide();
        return;
      }
      // Only act inside `.hero-macros`. Anything else (canvas, drawer,
      // chrome) leaves this tooltip alone.
      if (!target.closest(".hero-macros")) {
        hide();
        return;
      }
      const el = target.closest<HTMLElement>(
        "[data-dd-tooltip-wide][data-dd-tooltip-title]",
      );
      if (!el) {
        hide();
        return;
      }
      const text = el.getAttribute("data-dd-tooltip");
      const title = el.getAttribute("data-dd-tooltip-title");
      if (!text || !title) {
        hide();
        return;
      }
      // Already showing for this element — leave it alone, otherwise
      // the rect would jitter as the pointer moves inside it.
      if (lastEl === el && state) return;
      clearPending();
      pending = window.setTimeout(() => {
        const rect = el.getBoundingClientRect();
        setState({
          cx: rect.left + rect.width / 2,
          top: rect.top,
          title,
          text,
        });
        lastEl = el;
        pending = null;
      }, HOVER_DELAY_MS);
    };

    const onLeave = (e: PointerEvent) => {
      const next = e.relatedTarget as Element | null;
      if (next && next.closest(".hero-macros")) return;
      hide();
    };

    window.addEventListener("pointermove", onMove);
    document.addEventListener("pointerleave", onLeave);
    return () => {
      window.removeEventListener("pointermove", onMove);
      document.removeEventListener("pointerleave", onLeave);
      clearPending();
    };
    // `state` intentionally omitted from deps — handler captures it
    // via closure but we don't need the effect to re-bind on every
    // state change, only on mount/unmount.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  if (!state) return null;
  return (
    <div
      className="hero-macros-tooltip"
      role="tooltip"
      style={{
        // Center horizontally on the hovered control, sit just above
        // its top edge (8px gap, same as the prior CSS pseudo).
        left: state.cx,
        top: state.top - 8,
        transform: "translate(-50%, -100%)",
      }}
    >
      <div className="hero-macros-tooltip-title">{state.title}</div>
      <p className="hero-macros-tooltip-text">{state.text}</p>
    </div>
  );
}
