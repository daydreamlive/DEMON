"use client";

import { useEffect, useRef, useState } from "react";

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
// Show/dismiss:
//   - Mounts visible (unless previously dismissed via localStorage).
//     We do NOT gate the initial show on ``denoise === 0`` because
//     ``controls.denoise`` defaults to 0.7 in DEMON's DEFAULT_CONFIG,
//     so the slider's target value is non-zero from session start —
//     a value-based gate would never fire.
//   - Initial denoise value is captured AFTER a brief grace period
//     (the session-start gate plays a visual glide from prior value
//     down to 0 over ~1 s and we don't want that animation to count
//     as "user interacted").
//   - Once the captured "stable" value drifts (== user moved the
//     knob via drag / MIDI / keyboard), we persist dismiss and
//     unmount.

const STORAGE_KEY = "demon:hint:strength-onboarding-v1";
const GRACE_MS = 1500;

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
  // hidden and flip on after a mount-time check.
  const [show, setShow] = useState(false);
  /** Set after GRACE_MS so the session-start glide doesn't count as
   *  user interaction. While null, denoise changes are ignored. */
  const baseline = useRef<number | null>(null);

  useEffect(() => {
    if (hintDismissed()) return;
    setShow(true);
    // After the session-start animation settles, freeze the current
    // value as the "untouched" baseline. Subsequent drift = user
    // interaction.
    const t = window.setTimeout(() => {
      baseline.current = usePerformanceStore.getState().sliderTargets["denoise"] ?? 0;
    }, GRACE_MS);
    return () => window.clearTimeout(t);
  }, []);

  useEffect(() => {
    if (!show) return;
    if (baseline.current === null) return;
    // Tiny epsilon so floating-point jitter (e.g. param-smoothing
    // tween over-/undershoots) doesn't false-fire.
    if (Math.abs(denoise - baseline.current) > 0.005) {
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
