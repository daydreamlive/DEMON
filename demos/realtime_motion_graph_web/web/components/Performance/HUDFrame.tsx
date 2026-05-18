"use client";

// HUD edge containers — kept in the DOM as invisible mount points for
// the things that still query them (useEdgeLoraBinding writes
// --fill / data-bar / labels; MobileStepperRail mounts its rail
// surface on top; the ribbons canvas binds its writhe animation here).
//
// The TOP edge is gone — the DENOISE knob in HeroMacros covers that
// drag affordance. The LEFT/RIGHT edges stay in the DOM but their
// visual content is hidden via CSS (`.install-edge-bar`, ribbon canvas
// opacity:0) so the consolidated right-side <MasterPanel/> reads as
// the only LoRA surface on desktop. On mobile, MobileStepperRail
// continues to render against these hidden containers — no behavioral
// regression for the touch flow.

interface EdgeProps {
  side: "left" | "right";
  bar?: string;
}

function Edge({ side, bar }: EdgeProps) {
  return (
    <div
      className={`install-edge install-edge-${side}`}
      data-bar={bar}
    >
      <span className="install-edge-label" />
      <div className="install-edge-bar" />
    </div>
  );
}

export function HUDFrame() {
  return (
    <>
      <Edge side="left" />
      <Edge side="right" />
    </>
  );
}
