"use client";

import { useEffect, useRef } from "react";

import { LiteTrackCarousel } from "./LiteTrackCarousel";
import { RecordToggle } from "./RecordToggle";
import { SliderGroup } from "./SliderGroup";

interface Props {
  onOpenAllControls: () => void;
  /** Render a small pulsing dot next to "All controls" — typically
   *  toggled by the host when there are unsaved session tweaks. DEMON
   *  doesn't ship session save (auth + /api/sessions live in
   *  demon-public-demo), so the prop stays optional. */
  unsavedDot?: boolean;
}

// Mobile mixer bay. Everything in one panel — no MIX/TRACK tab switch.
// The bay grows to fill the bottom half of the viewport so the operator
// sees the full performance surface at once: track picker on top, the
// four main faders mid-body, REC + "All controls" on the action row.
//
// The previous tabbed layout hid either the track or the faders behind
// a pill toggle, which on a one-thumb mobile session meant the operator
// kept paging back and forth. Surfacing both rows is the cheap win;
// progressive disclosure still hides Mod/Channels/Styles behind the
// "All controls" sheet.
//
// Side effect: a ResizeObserver writes the live bay height to
// ``--lite-controls-h`` on ``#performance`` (the scene root) so the
// waveform-scrub strip and the graph wrap can anchor their bottoms
// to the actual top of the bay instead of guessing 50vh. Without
// this the bay was hiding the bottom of both on tall phones /
// unusual aspect ratios.
//
// We deliberately scope this var to ``#performance`` (not
// ``documentElement``): per PERFORMANCE.md, writing a CSS var on
// documentElement forces a whole-document style recalc + paint
// cascade that on Android Chromium briefly blanks consumers with
// backdrop-filter / mask-image (this bay has both, and so does
// #graph). Scoping the write to #performance keeps the recalc inside
// the scene subtree.
export function LiteControls({ onOpenAllControls, unsavedDot }: Props) {
  const rootRef = useRef<HTMLDivElement | null>(null);
  useEffect(() => {
    const el = rootRef.current;
    if (!el) return;
    const host =
      (document.getElementById("performance") as HTMLElement | null) ??
      document.documentElement;
    const apply = () => {
      host.style.setProperty("--lite-controls-h", `${el.offsetHeight}px`);
    };
    apply();
    const ro = new ResizeObserver(apply);
    ro.observe(el);
    return () => {
      ro.disconnect();
      // Clear the var so consumers fall back to their CSS default
      // when the bay unmounts (e.g. operator rotates to landscape
      // and the lite controls flip off).
      host.style.removeProperty("--lite-controls-h");
    };
  }, []);
  return (
    <div ref={rootRef} className="lite-controls">
      <div className="lite-row lite-row--track">
        <LiteTrackCarousel />
      </div>
      <div className="lite-row lite-row--main">
        <SliderGroup param="denoise" label="strength" />
        <SliderGroup param="hint_strength" label="structure" />
        <SliderGroup param="timbre_strength" label="timbre" />
        <SliderGroup param="shift" label="shift" />
      </div>
      <div className="lite-row lite-row--actions">
        <RecordToggle />
        <button
          type="button"
          className="lite-all-controls"
          onClick={onOpenAllControls}
          aria-label="All controls"
          data-dd-tooltip="All controls"
          data-dd-tooltip-pos="above"
        >
          {unsavedDot && (
            <span
              className="lite-all-controls-dot"
              aria-label="Unsaved changes"
            />
          )}
          <span className="lite-all-controls-label">More</span>
          <span className="lite-all-controls-arrow" aria-hidden="true">
            →
          </span>
        </button>
      </div>
    </div>
  );
}
