"use client";

import { useEffect, useRef, useState } from "react";

import { usePerformanceStore } from "@/store/usePerformanceStore";

// First-time-visitor onboarding affordance pointing AT the Strength
// knob from its LEFT side. A hand-drawn arrow gif points right + a
// white "Turn this first" caption sits to its left.
//
// Desktop-only — mobile already surfaces the Strength fader
// prominently on the left rail + lite bay; no need for a tutorial
// overlay there.
//
// Lifecycle:
//   1. Component mounts when HeroMacros mounts (== session started).
//   2. The session-start gate glides denoise from its prior value
//      (config default 0.7) down to 0. While that glide is active we
//      stay hidden — surfacing the hint mid-glide would tell the
//      user to do something they're already watching the engine do.
//   3. Once denoise has settled at ~0, we show the hint and freeze
//      the baseline. From this moment any rise in denoise counts as
//      user interaction → dismiss + persist.
//   4. Returning visit: localStorage flag short-circuits the whole
//      flow (component never shows).

const STORAGE_KEY = "demon:hint:strength-onboarding-v1";
// Threshold for "denoise has settled" / "user moved it". Tracks one
// step of the value-readout precision (sliders quantize to 0.01),
// with a small float-jitter margin.
const SETTLED_THRESHOLD = 0.01;

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
    // localStorage disabled — hint re-shows next session. OK.
  }
}

export function StrengthOnboardingHint() {
  const denoise = usePerformanceStore((s) => s.sliderTargets["denoise"] ?? 0);
  const [show, setShow] = useState(false);
  /** Latched once denoise has crossed below SETTLED_THRESHOLD —
   *  i.e. the session-start gate has finished its glide and the
   *  knob is at rest. From this point a rise back above the
   *  threshold means the user moved the knob. */
  const settled = useRef(false);

  useEffect(() => {
    if (hintDismissed()) return;
    if (!settled.current) {
      // Phase 1: wait for the session-start gate to glide denoise
      // down to ~0. Don't show yet — the gate's own animation tells
      // the user the engine is starting; layering a hint on top
      // would create competing signals.
      if (denoise <= SETTLED_THRESHOLD) {
        settled.current = true;
        setShow(true);
      }
      return;
    }
    // Phase 2: shown. Any meaningful rise means the user has
    // interacted with the Strength knob → dismiss + persist.
    if (show && denoise > SETTLED_THRESHOLD) {
      persistDismissed();
      setShow(false);
    }
  }, [denoise, show]);

  if (!show) return null;

  return (
    <div className="strength-onboarding-hint" aria-hidden="true">
      <span className="strength-onboarding-hint-text">Turn this first</span>
      <img
        className="strength-onboarding-hint-arrow"
        src="/strength-onboarding-arrow.gif"
        alt=""
        width={68}
        height={68}
      />
    </div>
  );
}
