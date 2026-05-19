"use client";

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

// Mobile bottom palette. Holds two mid-performance sliders
// (STRUCTURE / FEEDBACK), the track carousel, record toggle, and the
// "All controls" gateway. DENOISE lives on the left fader rail
// (MobileRemixStepper) and is intentionally omitted here so the bar
// doesn't duplicate the most prominent control.
//
// Renders as fixed bottom chrome on mobile (the AdvancedDrawer mobile
// branch mounts this directly, not inside a slide-up drawer). Click
// "All controls" → opens <MobileFullSheet/>, the full 7-tab equivalent.
export function LiteControls({ onOpenAllControls, unsavedDot }: Props) {
  return (
    <div className="lite-controls">
      <div className="lite-row lite-row--main">
        <SliderGroup param="hint_strength" label="structure" />
        <SliderGroup param="feedback" label="feedback" />
      </div>
      <LiteTrackCarousel />
      <div className="lite-row lite-row--actions">
        <RecordToggle />
        <button
          type="button"
          className="lite-all-controls"
          onClick={onOpenAllControls}
        >
          All controls
          {unsavedDot && (
            <span
              className="lite-all-controls-dot"
              aria-label="Unsaved changes"
            />
          )}
          <span className="lite-all-controls-arrow" aria-hidden="true">
            →
          </span>
        </button>
      </div>
    </div>
  );
}
