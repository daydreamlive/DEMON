"use client";

import { useEffect, useState } from "react";

import { usePerformanceStore } from "@/store/usePerformanceStore";

// First-time-visitor onboarding affordance pointing AT the Strength
// knob (the leftmost knob in HeroMacros). Caption ("Turn this first")
// sits above the knob with a hand-drawn SVG arrow swooping down to
// the knob's cap.
//
// Desktop-only — mobile already surfaces the Strength fader
// prominently on the left rail + lite bay; no need for a tutorial
// overlay there.
//
// Dismissal — any of:
//   1. User moves the Strength knob (denoise > 0). The interaction
//      IS the dismissal — once the user has obviously understood,
//      we don't want a dangling hint.
//   2. The hint has been dismissed in a prior session (localStorage
//      flag below). One-shot for the lifetime of the device.

const STORAGE_KEY = "demon:hint:strength-onboarding-v1";

function hintDismissed(): boolean {
  if (typeof localStorage === "undefined") return false;
  try {
    return localStorage.getItem(STORAGE_KEY) === "1";
  } catch {
    return false;
  }
}

function persistDismissed(): void {
  if (typeof localStorage === "undefined") return;
  try {
    localStorage.setItem(STORAGE_KEY, "1");
  } catch {
    // localStorage disabled (private window, quota) — hint will
    // re-show on next visit. Acceptable degradation.
  }
}

export function StrengthOnboardingHint() {
  const denoise = usePerformanceStore((s) => s.sliderTargets["denoise"] ?? 0);
  // SSR-safe: localStorage isn't readable on the server, so we start
  // closed and flip to "should render" on mount. Avoids hydration
  // mismatch.
  const [show, setShow] = useState(false);

  useEffect(() => {
    if (!hintDismissed() && denoise === 0) setShow(true);
  }, []);

  // First touch on the knob ⇒ persist + hide. Reading denoise from the
  // store catches drag, MIDI, AND keyboard — any path that moves
  // the value counts as "user understood the affordance".
  useEffect(() => {
    if (denoise > 0 && show) {
      persistDismissed();
      setShow(false);
    }
  }, [denoise, show]);

  if (!show) return null;

  return (
    <div className="strength-onboarding-hint" aria-hidden="true">
      <span className="strength-onboarding-hint-text">Turn this first</span>
      <svg
        className="strength-onboarding-hint-arrow"
        viewBox="0 0 60 70"
        fill="none"
        stroke="currentColor"
        strokeWidth={2.2}
        strokeLinecap="round"
        strokeLinejoin="round"
      >
        {/* Hand-drawn squiggle from upper-right, swooping down and
         *  to the left so the tip lands at the bottom-center of the
         *  SVG box — which is positioned over the knob's cap. Three
         *  small bumps in the curve read as a sketch rather than a
         *  perfect bezier. */}
        <path d="M 50 6 Q 38 4 28 16 Q 18 26 32 36 Q 46 44 24 52 Q 14 56 22 64" />
        {/* Arrowhead pointing down at the knob. */}
        <path d="M 16 58 L 22 66 L 30 60" />
      </svg>
    </div>
  );
}
