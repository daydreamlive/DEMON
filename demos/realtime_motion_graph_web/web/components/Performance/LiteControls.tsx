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
export function LiteControls({ onOpenAllControls, unsavedDot }: Props) {
  return (
    <div className="lite-controls">
      <div className="lite-row lite-row--track">
        <LiteTrackCarousel />
      </div>
      <div className="lite-row lite-row--main">
        <SliderGroup param="denoise" label="denoise" />
        <SliderGroup param="hint_strength" label="structure" />
        <SliderGroup param="feedback" label="feedback" />
        <SliderGroup param="lora_blend" label="blend" />
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
