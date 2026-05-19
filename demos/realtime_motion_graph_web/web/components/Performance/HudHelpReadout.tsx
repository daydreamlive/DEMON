"use client";

import { useEffect, useState } from "react";

// Ambient help readout that surfaces `data-dd-tooltip` copy when the
// Full Controls panel is closed. Mirrors the DrawerHelpBar inside the
// panel, but listens to window-level pointer events so hovering knobs
// in the hero macros bay, the audio-source dock, etc. populates the
// readout. Hidden whenever the panel is open (`body.drawer-open`) so
// the panel's own help bar is the canonical readout there.
//
// Only renders text while actively hovering something — no persistent
// "hover any control to read about it" hint, since this floats in the
// canvas and would otherwise be permanent visual chrome.

export function HudHelpReadout() {
  const [text, setText] = useState<string | null>(null);
  const [title, setTitle] = useState<string | null>(null);

  useEffect(() => {
    const onMove = (e: PointerEvent) => {
      const target = e.target as Element | null;
      if (!target) return;
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
      const visible = el.textContent?.trim().split(/\s+/).slice(0, 4).join(" ");
      const aria = el.getAttribute("aria-label");
      setTitle(visible || aria || null);
    };
    const onLeave = (e: PointerEvent) => {
      if (e.relatedTarget) return;
      setText(null);
      setTitle(null);
    };
    window.addEventListener("pointermove", onMove);
    document.addEventListener("pointerleave", onLeave);
    return () => {
      window.removeEventListener("pointermove", onMove);
      document.removeEventListener("pointerleave", onLeave);
    };
  }, []);

  if (!text) return null;

  return (
    <div className="hud-help-readout" role="status" aria-live="polite">
      {title && <div className="hud-help-readout-title">{title}</div>}
      <p className="hud-help-readout-text">{text}</p>
    </div>
  );
}
