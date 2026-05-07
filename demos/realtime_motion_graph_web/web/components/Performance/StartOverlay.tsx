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
// only after the ribbons have finished blowing up.
const LAUNCH_DURATION_MS = 700;

// The brand mark IS the play button. The icon halo + the "click/tap to
// begin" whisper sit inside a single <button> so anywhere on either
// element triggers onPlay — testers were instinctively trying to click
// the copy itself. On click, the ribbons spin + explode outward while
// the icon zooms forward; only then does onPlay fire and the overlay
// give way to the app.
export function StartOverlay({ onPlay, hidden }: Props) {
  const isMobile = useIsMobile();
  const [launching, setLaunching] = useState(false);

  function handleClick() {
    if (launching) return;
    setLaunching(true);
    // Fire onPlay() immediately so the queue.start() network round-trip
    // overlaps with the 700ms animation rather than starting after it.
    // If we deferred onPlay() to the timeout, the launching class would
    // come off at t=700ms while queue.status is still 'idle'/'joining',
    // briefly snapping the CTA back to its base "click to begin" state
    // until the join response landed.
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

  return (
    <div id="start-overlay" className={hidden ? "hidden" : ""}>
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
