"use client";

import { useCallback, useEffect, useRef, useState } from "react";

import { useCustomTracksStore } from "@/store/useCustomTracksStore";
import { usePerformanceStore } from "@/store/usePerformanceStore";

// Sequel to the StrengthOnboardingHint — points at the bottom-right
// Upload button in the AudioSourceCrate dock. Issue #156: users
// reported "Upload" is hard to spot tucked into the corner.
//
// We chain it AFTER the Strength hint dismisses (== denoise rose
// above the settled threshold) so the user only sees one piece of
// onboarding noise at a time. The Strength hint teaches "what
// strength does"; this hint then teaches "you can bring your own
// track."
//
// Trigger conditions (all must hold to reveal):
//   - localStorage flag is unset (first-time visitor for this hint),
//   - the user has no custom tracks yet (returning visitors who've
//     already uploaded once don't need the nudge),
//   - the session-start gate has settled denoise to ~0 (we've
//     observed it cross BELOW the threshold), AND THEN denoise has
//     risen back above the threshold. The two-phase latch is the
//     same one StrengthOnboardingHint uses: the gate's slow glide
//     from 0.7 → 0 shouldn't be misread as "user moved Strength."
//
// Dismissal:
//   - customCount rises above its at-reveal baseline (== upload
//     completed), OR
//   - dismiss() is called explicitly (e.g. the user clicks the
//     Upload button — they've clearly found it, the hint's job is
//     done even if they cancel the picker).
//
// Both dismissal paths persist the localStorage flag so the hint
// never re-appears.

const STORAGE_KEY = "demon:hint:upload-onboarding-v1";
const SETTLED_THRESHOLD = 0.01;
// Delay between Strength-hint dismissal and Upload-hint reveal so the
// two transitions don't visually overlap. The Strength hint has no
// fade-out animation, but a brief beat reads as deliberate sequencing
// rather than a swap.
const REVEAL_DELAY_MS = 700;

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

export interface UploadOnboardingHint {
  visible: boolean;
  dismiss: () => void;
}

export function useUploadOnboardingHint(): UploadOnboardingHint {
  const denoise = usePerformanceStore((s) => s.sliderTargets["denoise"] ?? 0);
  const customCount = useCustomTracksStore((s) => s.names.length);
  const [visible, setVisible] = useState(false);
  // Custom-track count latched at reveal time. Any later rise dismisses
  // the hint (== they successfully completed an upload).
  const baselineCustomCount = useRef<number | null>(null);
  const revealTimer = useRef<number | null>(null);
  // Latched to make the dismiss/persist decision idempotent — once we
  // dismiss we don't want subsequent effect runs to second-guess.
  const dismissed = useRef(false);
  // Mirrors StrengthOnboardingHint's settled-latch: only after we've
  // observed denoise cross BELOW the threshold (== the session-start
  // glide finished) does a rise back above it count as user input.
  // Without this, the gate's 0.7 → 0 glide would prematurely trigger
  // the reveal timer because denoise > THRESHOLD is true for most of
  // the glide.
  const settled = useRef(false);

  const dismiss = useCallback(() => {
    if (dismissed.current) return;
    dismissed.current = true;
    persistDismissed();
    setVisible(false);
    if (revealTimer.current !== null) {
      window.clearTimeout(revealTimer.current);
      revealTimer.current = null;
    }
  }, []);

  // Reveal gate: phase 1 waits for the session-start glide to settle
  // denoise to ~0, phase 2 watches for the user to lift it back above
  // the threshold, then schedules a one-shot reveal.
  useEffect(() => {
    if (dismissed.current) return;
    if (visible) return;
    if (hintDismissed()) {
      dismissed.current = true;
      return;
    }
    if (customCount > 0) {
      // Returning visitor (or first-session user who somehow uploaded
      // before this hint mounted). They've found Upload — suppress
      // forever.
      dismiss();
      return;
    }
    if (!settled.current) {
      if (denoise <= SETTLED_THRESHOLD) settled.current = true;
      return;
    }
    if (denoise <= SETTLED_THRESHOLD) return;
    if (revealTimer.current !== null) return;
    revealTimer.current = window.setTimeout(() => {
      revealTimer.current = null;
      if (dismissed.current) return;
      if (hintDismissed()) return;
      if (useCustomTracksStore.getState().names.length > 0) {
        // User uploaded during the delay — they don't need the hint.
        dismiss();
        return;
      }
      baselineCustomCount.current =
        useCustomTracksStore.getState().names.length;
      setVisible(true);
    }, REVEAL_DELAY_MS);
  }, [denoise, customCount, visible, dismiss]);

  // Cleanup any pending reveal timer on unmount.
  useEffect(() => {
    return () => {
      if (revealTimer.current !== null) {
        window.clearTimeout(revealTimer.current);
        revealTimer.current = null;
      }
    };
  }, []);

  // Dismiss on successful upload.
  useEffect(() => {
    if (!visible) return;
    if (baselineCustomCount.current === null) return;
    if (customCount > baselineCustomCount.current) {
      dismiss();
    }
  }, [customCount, visible, dismiss]);

  return { visible, dismiss };
}
