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

export function DrawerHelpBar() {
  const barRef = useRef<HTMLDivElement | null>(null);
  const getRoot = useCallback(() => document.getElementById("install-sheet"), []);
  const { title, text } = useTooltipHover({ getRoot, selfRef: barRef });

  return (
    <div
      ref={barRef}
      className={`drawer-help-bar${text ? " drawer-help-bar--active" : ""}`}
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
