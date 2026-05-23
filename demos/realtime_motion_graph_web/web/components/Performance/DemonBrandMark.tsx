"use client";

import { useEffect, useState } from "react";

import { DEMON_LETTER_PALETTE } from "@/engine/render/demonLetters";
import { useSessionStore } from "@/store/useSessionStore";

// DEMON brand wordmark — 5 SVG letters whose `d` attributes are ticked
// each frame by useRenderLoop → tickDemonLetters with the live kick
// value, so the writhe amplitude breathes with the audio in the same
// language as the halo / perimeter ribbons.
//
// Entrance choreography: idle state hides D/E/M/N at opacity 0; the O
// is held separately at visibility:hidden so the halo→O hand-off
// involves zero opacity transitions of any kind. When the session
// leaves "idle" (the user clicks Play in StartOverlay), a one-shot
// --entered modifier kicks in. The CSS runs the D/E/M/N letters in
// from the left with staggered delays, while the O stays
// visibility:hidden through the halo morph and then instantly snaps
// to visibility:visible right before the overlay vanishes — an
// instant swap from writhing halo (which lands at the O slot via FLIP
// translate + scale in StartOverlay.handleClick) to the fixed letter.
const DEMON_LETTERS: ReadonlyArray<string> = ["D", "E", "M", "O", "N"];

export function DemonBrandMark() {
  const status = useSessionStore((s) => s.status);
  // Once we've ever left idle, keep --entered on for the page lifetime —
  // the animation is `forwards` so the final state sticks. If the
  // session later resets back to idle (useIdleReset), the brand mark
  // also drops back to hidden so the next launch can play the entrance
  // again from a clean slate.
  const [entered, setEntered] = useState(false);
  useEffect(() => {
    if (status === "idle") {
      setEntered(false);
    } else {
      setEntered(true);
    }
  }, [status]);

  return (
    <div
      className={`demon-brand-mark demon-letters${entered ? " demon-brand-mark--entered" : ""}`}
      aria-hidden="true"
    >
      {DEMON_LETTERS.map((letter, i) => (
        <svg
          key={letter}
          className="demon-letter"
          data-letter={letter}
          data-style="wave"
          data-phase={i}
          viewBox="0 0 100 100"
          preserveAspectRatio="xMidYMid meet"
          aria-hidden="true"
        >
          {DEMON_LETTER_PALETTE.map((color) => (
            <path
              key={color}
              stroke={color}
              fill="none"
              strokeLinecap="round"
              strokeLinejoin="round"
            />
          ))}
        </svg>
      ))}
    </div>
  );
}
