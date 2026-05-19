"use client";

import { useEffect, useRef, useState } from "react";

// Persistent help readout pinned to the bottom of the Full Controls
// panel. Mirrors whatever control the user is hovering — reads its
// `data-dd-tooltip` attribute (which all knob/slider/button labels in
// the panel already carry) and surfaces the long-form description so
// the user doesn't need to hold a hover long enough for the floating
// tooltip to appear. Acts as the panel's "info strip" — same role as
// the readout band on the bottom of a Sound Particles plugin.

export function DrawerHelpBar() {
  const [text, setText] = useState<string | null>(null);
  const [title, setTitle] = useState<string | null>(null);
  const barRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    const root = document.getElementById("install-sheet");
    if (!root) return;

    const onMove = (e: PointerEvent) => {
      const target = e.target as Element | null;
      if (!target) return;
      // Ignore hover over the help bar itself — otherwise hovering the
      // text would clear / overwrite the very text being read.
      if (barRef.current && barRef.current.contains(target)) return;
      const el = target.closest<HTMLElement>("[data-dd-tooltip]");
      if (!el) {
        setText(null);
        setTitle(null);
        return;
      }
      const t = el.getAttribute("data-dd-tooltip");
      if (!t) {
        setText(null);
        setTitle(null);
        return;
      }
      setText(t);
      // Title source: prefer the element's own visible text (e.g. a
      // knob label "denoise"), fall back to aria-label.
      const visible = el.textContent?.trim().split(/\s+/).slice(0, 4).join(" ");
      const aria = el.getAttribute("aria-label");
      setTitle(visible || aria || null);
    };
    const onLeave = (e: PointerEvent) => {
      const next = (e.relatedTarget as Element | null) ?? null;
      if (next && root.contains(next)) return;
      setText(null);
      setTitle(null);
    };

    root.addEventListener("pointermove", onMove);
    root.addEventListener("pointerleave", onLeave);
    return () => {
      root.removeEventListener("pointermove", onMove);
      root.removeEventListener("pointerleave", onLeave);
    };
  }, []);

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
