"use client";

// Three of the four edges (top, left, right). The bottom edge is the
// Advanced drawer handle. ribbons.js (Phase 8) will mount SVG paths inside
// each .install-edge-bar to drive the writhe animation; until then the bars
// stay as plain DIVs so the layout reserves the right space.

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
    </div>
  );
}

export function HUDFrame() {
  return (
    <>
      <Edge side="top" label="Remix Strength" bar="denoise" />
      {/* Left/right bars track the first/second currently enabled LoRA.
          Their data-bar and label are populated at runtime from the
          server's catalog (Phase 11). */}
      <Edge side="left" />
      <Edge side="right" />
    </>
  );
}
