"use client";

import { useTooltipHover } from "@/hooks/useTooltipHover";

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
//
// Hero-macros bay is excluded — those knobs already surface their
// tooltip + kbd hint directly above the control via the CSS ::after
// pseudo, and mirroring the same copy at the top of the viewport
// just doubles the noise.

export function HudHelpReadout() {
  const { title, text } = useTooltipHover({
    excludeSelector: ".hero-macros",
  });
  if (!text) return null;

  return (
    <div className="hud-help-readout" role="status" aria-live="polite">
      {title && <div className="hud-help-readout-title">{title}</div>}
      <p className="hud-help-readout-text">{text}</p>
    </div>
  );
}
