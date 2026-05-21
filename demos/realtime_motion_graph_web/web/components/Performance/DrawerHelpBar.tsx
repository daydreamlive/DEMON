"use client";

import { useCallback, useRef } from "react";

import { useTooltipHover } from "@/hooks/useTooltipHover";

// Persistent help readout pinned to the bottom of the Full Controls
// panel. Mirrors whatever control the user is hovering — reads its
// `data-dd-tooltip` attribute (which all knob/slider/button labels in
// the panel already carry) and surfaces the long-form description so
// the user doesn't need to hold a hover long enough for the floating
// tooltip to appear. Acts as the panel's "info strip" — same role as
// the readout band on the bottom of a Sound Particles plugin.
//
// In spread (all-controls) mode the static info strip would eat the
// grid height the layout exists to maximise. So in that mode the bar
// renders only while actively hovering something, with overlay
// positioning (see .drawer-help-bar--overlay) so it floats over the
// bottom of the panel instead of pushing the grid up.

interface DrawerHelpBarProps {
  /** When true, render as a floating overlay that only appears while
   *  the user is actively hovering a tooltipped control. Used by the
   *  drawer's spread/all-controls layout. */
  spread?: boolean;
}

export function DrawerHelpBar({ spread = false }: DrawerHelpBarProps) {
  const barRef = useRef<HTMLDivElement | null>(null);
  const getRoot = useCallback(() => document.getElementById("install-sheet"), []);
  const { title, text } = useTooltipHover({ getRoot, selfRef: barRef });

  if (spread && !text) return null;

  return (
    <div
      ref={barRef}
      className={`drawer-help-bar${text ? " drawer-help-bar--active" : ""}${spread ? " drawer-help-bar--overlay" : ""}`}
      role="status"
      aria-live="polite"
    >
      {text ? (
        <>
          {title && <div className="drawer-help-bar-title">{title}</div>}
          <p className="drawer-help-bar-text">{text}</p>
        </>
      ) : (
        <p className="drawer-help-bar-hint">
          Hover any control to read about it.
        </p>
      )}
    </div>
  );
}
