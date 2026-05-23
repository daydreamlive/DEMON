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

const TABS: LiteTab[] = ["mix", "track"];
const TAB_LABELS: Record<LiteTab, string> = {
  mix: "Mix",
  track: "Track",
};

// Mobile mixer bay. A simple two-pill tab strip switches between two
// content sections — Mix and Track — that render conditionally (only
// the active one is mounted). The previous scroll-snap carousel
// implementation overlapped both sections on iOS Safari and let the
// inner track-carousel swipe leak out to the outer scroller; replacing
// it with conditional render eliminates both classes of bug at the
// cost of losing the swipe-to-switch gesture.
//
//   MIX   — 4 faders (denoise / structure / feedback / lora_blend) +
//           "All controls" gateway to the full 7-tab sheet.
//   TRACK — track carousel (upload + mic inside the picker) + REC.
export function LiteControls({ onOpenAllControls, unsavedDot }: Props) {
  const [tab, setTab] = useState<LiteTab>("mix");

  return (
    <div className="lite-controls">
      <div
        className="lite-tabs"
        role="tablist"
        aria-label="Mobile mixer"
      >
        {TABS.map((t) => (
          <button
            key={t}
            type="button"
            role="tab"
            aria-selected={tab === t}
            className={`lite-tab${tab === t ? " lite-tab--active" : ""}`}
            onClick={() => setTab(t)}
          >
            {TAB_LABELS[t]}
            {/* Unsaved dot surfaces on whichever pill isn't currently
                active — so the cue is visible regardless of context. */}
            {unsavedDot && tab !== t && t === "mix" && (
              <span
                className="lite-tab-dot"
                aria-label="Unsaved changes"
              />
            )}
          </button>
        ))}
      </div>
      <div className="lite-tab-body" data-active-tab={tab}>
        {tab === "mix" ? (
          <section data-tab="mix" className="lite-tab-section">
            <div className="lite-row lite-row--main">
              <SliderGroup param="denoise" label="denoise" />
              <SliderGroup param="hint_strength" label="structure" />
              <SliderGroup param="feedback" label="feedback" />
              <SliderGroup param="lora_blend" label="blend" />
            </div>
            <button
              type="button"
              className="lite-all-controls"
              onClick={onOpenAllControls}
              aria-label="All controls"
              data-dd-tooltip="All controls"
              data-dd-tooltip-pos="below"
            >
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
          </section>
        ) : (
          <section data-tab="track" className="lite-tab-section">
            <LiteTrackCarousel />
            <RecordToggle />
          </section>
        )}
      </div>
    </div>
  );
}
