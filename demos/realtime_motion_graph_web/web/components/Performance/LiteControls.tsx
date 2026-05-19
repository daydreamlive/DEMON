"use client";

import { useState } from "react";

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

type LiteTab = "mix" | "track";

// Wave 13.4: tabbed mobile mixer. Two contexts:
//   MIX   — 4 faders (denoise / structure / feedback / lora_blend) +
//           "All controls" gateway to the full 7-tab sheet.
//   TRACK — record toggle + track carousel (upload + mic chips
//           handled by the carousel's track-picker).
//
// One bay, two contexts — keeps the bottom strip narrow enough that
// the graph + waveform get max canvas headroom regardless of which
// tab's active.
export function LiteControls({ onOpenAllControls, unsavedDot }: Props) {
  const [tab, setTab] = useState<LiteTab>("mix");
  return (
    <div className="lite-controls">
      <div
        className="lite-tabs"
        role="tablist"
        aria-label="Mobile mixer"
      >
        <button
          type="button"
          role="tab"
          aria-selected={tab === "mix"}
          className={`lite-tab${tab === "mix" ? " lite-tab--active" : ""}`}
          onClick={() => setTab("mix")}
        >
          Mix
          {unsavedDot && tab !== "mix" && (
            <span
              className="lite-tab-dot"
              aria-label="Unsaved changes"
            />
          )}
        </button>
        <button
          type="button"
          role="tab"
          aria-selected={tab === "track"}
          className={`lite-tab${tab === "track" ? " lite-tab--active" : ""}`}
          onClick={() => setTab("track")}
        >
          Track
        </button>
      </div>
      <div className="lite-tab-body">
        {tab === "mix" ? (
          <>
            <div className="lite-row lite-row--main">
              <SliderGroup param="denoise" label="denoise" />
              <SliderGroup param="hint_strength" label="structure" />
              <SliderGroup param="feedback" label="feedback" />
              <SliderGroup param="lora_blend" label="lora blend" />
            </div>
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
          </>
        ) : (
          <>
            <LiteTrackCarousel />
            <RecordToggle />
          </>
        )}
      </div>
    </div>
  );
}
