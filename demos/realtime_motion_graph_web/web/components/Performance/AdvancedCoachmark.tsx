"use client";

import { useEffect } from "react";

// ─────────────────────────────────────────────────────────────────────
// HINT SEQUENCE — canonical doc. KEEP THIS COMMENT IN SYNC when adding
// new coaching steps.
// ─────────────────────────────────────────────────────────────────────
//
// First-time users get a cascade of UI hints. Each stage waits on the
// previous one (or on an explicit engagement signal) so we never fire
// two attention-grabbers simultaneously — that's the bug this sequence
// exists to prevent.
//
//   Stage A — Pre-session
//     Trigger: initial page load (no session yet)
//     Shows:   <StartOverlay/> "click to begin"
//     Exits:   user clicks → session boots
//
//   Stage B — Session ready / new song
//     Trigger: useSessionStore.status === "ready"
//              AND usePerformanceStore.remixStarted === false
//              (the latter resets per fixture swap, so every song gets
//              its own gated entry)
//     Shows:   Top-ribbon <RemixHint prominent/> — "drag to start"
//              The gated next-action; the user MUST drag this ribbon
//              to engage the remix engine. Owned by <DesktopEdgeDrag/>.
//     Exits:   user drags the top ribbon → remixStarted := true
//
//   Stage C — Remix gate cleared
//     Trigger: remixStarted := true (first top-ribbon drag this song)
//     Shows:   Side-ribbon <RemixHint/> on left + right LoRA ribbons
//              (idle pulse, contextual — not prominent)
//     Exits:   each side hint dismisses on first drag of that slider
//
//   Stage D — User engaged
//     Trigger: Stage C completed at least once this session
//              AND ~12s have passed since remixStarted first flipped
//              AND the advanced drawer is still closed
//     Shows:   <AdvancedCoachmark/> — "Click for more controls"
//     Dismiss: pointerdown anywhere, Esc, drawer open, or 8s auto-hide
//     Persist: dd:advanced-coachmark-dismissed = "1" in localStorage
//     Owned by: <AdvancedDrawer/> (this component is just the view)
//
// Adding a new coaching step? Slot it into the sequence above, pick a
// trigger that waits on a real engagement signal (not just session-
// ready), and update this comment. Audio-source / upload do NOT need
// coachmarks — their copy + always-visible icon are the affordances.
// ─────────────────────────────────────────────────────────────────────
//
// This file: the Stage-D view. The visibility gate, dismissal
// persistence, and remix-gate-clear timing all live in
// <AdvancedDrawer/>. This component just renders the callout and
// wires the document-level dismiss listeners.

export const advancedCoachmarkStorageKey = "dd:advanced-coachmark-dismissed";
const AUTO_HIDE_MS = 8000;

interface Props {
  visible: boolean;
  onDismiss: () => void;
}

export function AdvancedCoachmark({ visible, onDismiss }: Props) {
  // Auto-hide after AUTO_HIDE_MS even without explicit dismissal so we
  // don't nag once the user has had a chance to notice it.
  useEffect(() => {
    if (!visible) return;
    const t = window.setTimeout(onDismiss, AUTO_HIDE_MS);
    return () => window.clearTimeout(t);
  }, [visible, onDismiss]);

  // Pointer / Esc dismissal. We listen at the document level so any
  // interaction at all counts as acknowledgement — clicks on the
  // coachmark itself fall through (pointer-events: none in CSS) and
  // hit whatever's underneath, so the user can click the handle to
  // both dismiss + open in one motion.
  useEffect(() => {
    if (!visible) return;
    const onPointer = () => onDismiss();
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onDismiss();
    };
    document.addEventListener("pointerdown", onPointer, { once: true });
    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("pointerdown", onPointer);
      document.removeEventListener("keydown", onKey);
    };
  }, [visible, onDismiss]);

  if (!visible) return null;

  return (
    <div className="advanced-coachmark" role="status" aria-live="polite">
      <span className="advanced-coachmark-text">
        Click for more controls — Press <kbd>O</kbd> to toggle
      </span>
      <span className="advanced-coachmark-arrow" aria-hidden="true">
        ▾
      </span>
    </div>
  );
}
