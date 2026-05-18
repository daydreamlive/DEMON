"use client";

import { useEffect, useState } from "react";

import { useCurveStore } from "@/store/useCurveStore";
import { usePerformanceStore } from "@/store/usePerformanceStore";
import { useSessionStore } from "@/store/useSessionStore";

import { Knob } from "./Knob";
import { defaultLabelFor, kbdHintFor } from "./SliderTile";

// Permanent 3-knob row at the bottom-center of the canvas, with a
// stacked control column on the right (Seed → Curve Editor → Full
// Controls). The "performance palette" — the three macros every
// musician reaches for, always visible, plus quick access to the two
// most-used tool surfaces and the drawer toggle.
//
// Visibility:
//  - Hidden while the session is idle.
//  - Knobs hide when the drawer is open (CORE tab covers them);
//    the tool column stays so Simple Controls / Curve Editor / Seed
//    stay reachable. Curve overlay does the same.
//  - Hidden below 768 px viewport (mobile gets LiteControls).

const HERO_PARAMS = ["denoise", "hint_strength", "feedback"] as const;

function DiceIcon() {
  return (
    <svg
      viewBox="0 0 16 16"
      width={12}
      height={12}
      fill="none"
      stroke="currentColor"
      strokeWidth={1.4}
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      <rect x="2.5" y="2.5" width="11" height="11" rx="2" />
      <circle cx="5.5" cy="5.5" r="0.9" fill="currentColor" stroke="none" />
      <circle cx="10.5" cy="5.5" r="0.9" fill="currentColor" stroke="none" />
      <circle cx="5.5" cy="10.5" r="0.9" fill="currentColor" stroke="none" />
      <circle cx="10.5" cy="10.5" r="0.9" fill="currentColor" stroke="none" />
    </svg>
  );
}

function CurveIcon() {
  return (
    <svg
      viewBox="0 0 16 16"
      width={12}
      height={12}
      fill="none"
      stroke="currentColor"
      strokeWidth={1.4}
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      <path d="M2 12 C 5 12, 5 4, 8 4 S 11 12, 14 12" />
    </svg>
  );
}

export function HeroMacros() {
  const status = useSessionStore((s) => s.status);
  const started = status !== "idle";
  const curveOpen = useCurveStore((s) => s.overlayOpen);
  const toggleCurve = useCurveStore((s) => s.toggleOverlay);
  const randomizeSeed = usePerformanceStore((s) => s.randomizeSeed);
  const [drawerOpen, setDrawerOpen] = useState(false);

  // Mirror body.drawer-open so the toggle label/caret flip with the
  // drawer state. The drawer is the source of truth.
  useEffect(() => {
    const sync = () => {
      setDrawerOpen(document.body.classList.contains("drawer-open"));
    };
    const obs = new MutationObserver(sync);
    obs.observe(document.body, { attributes: true, attributeFilter: ["class"] });
    sync();
    return () => obs.disconnect();
  }, []);

  if (!started) return null;
  return (
    <div
      className={`hero-macros${drawerOpen ? " hero-macros--drawer-open" : ""}${curveOpen ? " hero-macros--curve-open" : ""}`}
      data-hero-macros
    >
      <div className="hero-macros-knobs">
        {HERO_PARAMS.map((p) => (
          <Knob
            key={p}
            param={p}
            label={defaultLabelFor(p)}
            kbd={kbdHintFor(p)}
          />
        ))}
      </div>
      <div className="hero-macros-divider" aria-hidden="true" />
      <div className="hero-macros-tools">
        <button
          type="button"
          className="hero-macros-tool"
          onClick={() => randomizeSeed()}
          aria-label="Randomize seed"
        >
          <DiceIcon />
          <span className="hero-macros-tool-label">Seed</span>
        </button>
        <button
          type="button"
          className={`hero-macros-tool${curveOpen ? " hero-macros-tool--active" : ""}`}
          onClick={() => toggleCurve()}
          aria-pressed={curveOpen}
          aria-label="Toggle curve editor"
        >
          <CurveIcon />
          <span className="hero-macros-tool-label">Curve Editor</span>
        </button>
        <button
          type="button"
          className="hero-macros-toggle"
          onClick={() => document.dispatchEvent(new Event("dd:toggle-drawer"))}
          aria-label={drawerOpen ? "Close Full Controls" : "Open Full Controls"}
          aria-expanded={drawerOpen}
        >
          <span className="hero-macros-toggle-label">
            {drawerOpen ? "Simple Controls" : "Full Controls"}
          </span>
          <span className="hero-macros-toggle-caret" aria-hidden="true">
            {drawerOpen ? "◂" : "▸"}
          </span>
        </button>
      </div>
    </div>
  );
}
