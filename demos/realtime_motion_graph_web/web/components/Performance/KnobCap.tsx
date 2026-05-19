"use client";

import { useId } from "react";

// Shared 3D-cap chrome for the Knob family — Knob (continuous param)
// and SeedKnob (randomize button). Renders inside a parent `<svg>` and
// owns its own per-instance gradient ids so a single SVG can host
// many caps without their <defs> colliding. Modeled on inShaper's gain
// knobs: radial-gradient body with a soft top-left highlight, beveled
// metallic rim, and a hairline shadow underneath.
//
// Stacking: parents draw arcs/etc BEFORE this fragment so they sit
// behind the cap, and overlays (indicator notch / dice glyph) AFTER
// so they sit on top.

export function KnobCap() {
  // Per-instance ids so multiple caps in one SVG don't collide.
  // Without this, all caps would reference the same gradient and only
  // one would render correctly under SSR + out-of-order hydration.
  const uid = useId().replace(/:/g, "_");
  const capId = `knob-cap-${uid}`;
  const rimLightId = `knob-rim-${uid}`;

  return (
    <>
      <defs>
        {/* Cap body — radial gradient with a soft top-left highlight.
            Models a 3D rounded knob cap without going photorealistic. */}
        <radialGradient
          id={capId}
          cx="0.35"
          cy="0.28"
          r="0.85"
          fx="0.32"
          fy="0.22"
        >
          <stop offset="0%" stopColor="rgb(78, 84, 96)" />
          <stop offset="45%" stopColor="rgb(36, 40, 48)" />
          <stop offset="100%" stopColor="rgb(8, 10, 14)" />
        </radialGradient>
        {/* Outer rim — thin metallic gradient ring that gives the edge
            of the cap a beveled look. */}
        <linearGradient id={rimLightId} x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor="rgba(255, 255, 255, 0.22)" />
          <stop offset="50%" stopColor="rgba(255, 255, 255, 0.04)" />
          <stop offset="100%" stopColor="rgba(0, 0, 0, 0.35)" />
        </linearGradient>
      </defs>
      {/* Hairline shadow under the cap. */}
      <circle cx="24" cy="25" r="15.5" className="knob-shadow" />
      {/* Cap body. */}
      <circle cx="24" cy="24" r="15" fill={`url(#${capId})`} />
      {/* Beveled rim — 1px stroke with a top-to-bottom gradient giving
          the cap edge a metallic catch-light at the top and a darker
          bottom edge. */}
      <circle
        cx="24"
        cy="24"
        r="15"
        fill="none"
        stroke={`url(#${rimLightId})`}
        strokeWidth="1"
      />
    </>
  );
}
