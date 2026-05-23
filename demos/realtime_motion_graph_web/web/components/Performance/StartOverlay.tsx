"use client";

import { useState } from "react";

import { START_MARK_PALETTE } from "@/engine/render/ribbons";
import { useIsMobile } from "@/hooks/useIsMobile";

interface Props {
  onPlay: () => void;
  hidden?: boolean;
}

// Total length of the click-to-launch CSS animation. Kept in sync with
// the @keyframes durations in globals.css so the actual onPlay() fires
// only after the halo has finished morphing into the DEMON brand mark.
const LAUNCH_DURATION_MS = 1300;

// The brand mark IS the play button. The icon halo + the "click/tap to
// begin" whisper sit inside a single <button> so anywhere on either
// element triggers onPlay — testers were instinctively trying to click
// the copy itself. On click the writhing halo shrinks + translates to
// where the O of DEMON lives in the top-left brand mark, while the
// surrounding letters slide in around it (the entrance animation in
// DemonBrandMark fires off the session-status change). The icon + the
// whisper fade out independently so they don't follow the halo on its
// trip to the corner.
export function StartOverlay({ onPlay, hidden }: Props) {
  const isMobile = useIsMobile();
  const [launching, setLaunching] = useState(false);

  function handleClick() {
    if (launching) return;
    // FLIP morph: measure the halo's current screen rect and the brand
    // mark O's final rect, then write the delta translate + scale onto
    // CSS vars consumed by the start-ribbons-morph keyframes. The brand
    // mark's letters have no transform in their resting state (the
    // entrance animation supplies the offset transform via @keyframes
    // only), so measuring the O letter directly gives its true final
    // position — no need to subtract out a slide-in offset.
    const haloEl = document.querySelector(
      ".start-mark-ribbons",
    ) as SVGElement | null;
    const targetEl = document.querySelector(
      '.demon-brand-mark .demon-letter[data-letter="O"]',
    ) as HTMLElement | null;
    if (haloEl && targetEl) {
      const h = haloEl.getBoundingClientRect();
      const t = targetEl.getBoundingClientRect();
      const haloCx = h.left + h.width / 2;
      const haloCy = h.top + h.height / 2;
      const targetCx = t.left + t.width / 2;
      const targetCy = t.top + t.height / 2;
      const scale = h.width > 0 ? t.width / h.width : 0.12;
      haloEl.style.setProperty("--halo-tx", `${(targetCx - haloCx).toFixed(2)}px`);
      haloEl.style.setProperty("--halo-ty", `${(targetCy - haloCy).toFixed(2)}px`);
      haloEl.style.setProperty("--halo-scale", scale.toFixed(4));
    }
    setLaunching(true);
    // Fire onPlay() immediately so the queue.start() network round-trip
    // overlaps with the launch animation rather than starting after it.
    // It also flips session.status off "idle", which is the trigger
    // DemonBrandMark watches to run its letter entrance animation in
    // parallel with the halo morph.
    onPlay();
    window.setTimeout(() => {
      // Happy-path: parent re-renders us with hidden=true (queue admits
      // OR session starts) and this state never matters. Sad-path: gate,
      // error, or paywall keeps us mounted and we'd otherwise be stuck
      // post-launch animation showing nothing — reset so the user can
      // see + interact with the title screen again.
      setLaunching(false);
    }, LAUNCH_DURATION_MS);
  }

  const whisper = isMobile ? "tap to begin" : "click to begin";

  // useStartSession calls setStatus("loading-fixture") synchronously the
  // moment we invoke onPlay(), which flips PerformanceShell's `started`
  // and bounces back `hidden={true}` in the very next render — the same
  // render that adds our `start-cta--launching` class. `.hidden` is
  // `display: none !important`, so without this guard the overlay
  // would be removed from the layout tree before the launch animation
  // could paint a single frame (the halo would appear to instantly
  // vanish). While `launching` is true we ignore the prop; the
  // setTimeout above resets it after LAUNCH_DURATION_MS, at which
  // point the parent's hidden value takes over and the overlay
  // disappears for real.
  const visuallyHidden = hidden && !launching;

  return (
    <div
      id="start-overlay"
      className={`${visuallyHidden ? "hidden" : ""}${launching ? " start-overlay--launching" : ""}`.trim()}
    >
      <button
        type="button"
        className={`start-cta${launching ? " start-cta--launching" : ""}`}
        onClick={handleClick}
        aria-label={isMobile ? "Tap to begin" : "Click to begin"}
        disabled={launching}
      >
        <span className="start-cta-halo" aria-hidden="true">
          {/* Writhing ribbon halo around the logo — populated by
              useRenderLoop's tickStartMarkRibbon. */}
          <svg
            className="start-mark-ribbons"
            viewBox="0 0 100 100"
            preserveAspectRatio="xMidYMid meet"
            aria-hidden="true"
          >
            {START_MARK_PALETTE.map((color) => (
              <path
                key={color}
                stroke={color}
                fill="none"
                strokeLinecap="round"
                strokeLinejoin="round"
              />
            ))}
          </svg>
          <img
            className="start-mark"
            src="/daydream-icon-clean.png"
            alt=""
          />
        </span>
        <span className="start-whisper">{whisper}</span>
      </button>
    </div>
  );
}
