"use client";

import { EdgeFader } from "./EdgeFader";

// Three of the four edges (top, left, right). The bottom edge is the
// Advanced drawer handle. ribbons.js (Phase 8) mounts SVG paths inside
// each .install-edge-bar to drive the writhe animation; on left/right
// we layer a traditional vertical fader on top so the LoRA strength
// reads as a proper channel-strip fader instead of a ribbon.

interface EdgeProps {
  side: "top" | "left" | "right";
  label?: string;
  bar?: string;
}

function Edge({ side, label, bar }: EdgeProps) {
  return (
    <div
      className={`install-edge install-edge-${side}`}
      data-bar={bar}
    >
      <span className="install-edge-label">{label ?? ""}</span>
      <div className="install-edge-bar" />
      {(side === "left" || side === "right") && <EdgeFader side={side} />}
    </div>
  );
}

export function HUDFrame() {
  return (
    <>
      {/* The top "Remix Strength" ribbon is gone — the DENOISE knob in
          the always-visible hero macros row at the bottom now serves
          the same drag-affordance role. Top edge stays empty so the
          show frames itself with negative space, not chrome. */}
      {/* Left/right bars track the first/second currently enabled LoRA.
          Their data-bar and label are populated at runtime from the
          server's catalog. */}
      <Edge side="left" />
      <Edge side="right" />
    </>
  );
}
