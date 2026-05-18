"use client";

// Small chrome badge on the graph that says "the kick / music drives
// this." Pulses with the same --bloom-amount the rest of the scene
// reads, so when chorus bursts fire across every lane the user has
// one anchor element labelled BEAT to attribute the energy to.
//
// All visual reactivity is CSS — reading `var(--bloom-amount)` set on
// #performance by useRenderLoop. No JS rAF, no per-frame state. The
// dot scales + brightens; the wordmark stays static so the label is
// always legible.

export function BeatIndicator() {
  return (
    <div className="beat-indicator" aria-hidden="true">
      <span className="beat-indicator-dot" />
      <span className="beat-indicator-label">Beat</span>
    </div>
  );
}
