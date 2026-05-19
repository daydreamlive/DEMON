"use client";

import { Knob } from "./Knob";
import { RefControl } from "./RefControl";
import { SeedKnob } from "./SeedKnob";
import { defaultLabelFor, kbdHintFor } from "./SliderTile";
import { TrackPicker } from "./TrackPicker";

// CORE tab — dial-it-and-go macros every musician knows, drawn as
// rotary knobs (matches the inShaper / GrainDust visual vocabulary for
// continuous "tweak with one hand" params). Plus the two reference-
// track pickers that pair with TIMBRE + TRACK.
//
// Feedback lives on the MOD tab next to feedback_depth — they're a
// pair (depth controls how far back the feedback knob reaches), and
// keeping them adjacent reads better than splitting them across tabs.
// The bottom HeroMacros bay still surfaces feedback as a quick-access
// knob when the drawer is closed.
//
// All labels route through defaultLabelFor() so DISPLAY_NAMES in
// SliderTile.tsx stays the single source of truth for graph-lane
// pills, MIDI map UI, and knob/fader labels.
export function CoreTile() {
  return (
    <div className="knob-tile" data-tile="core">
      <div className="knob-rack" id="sliders">
        <Knob
          param="denoise"
          label={defaultLabelFor("denoise")}
          kbd={kbdHintFor("denoise")}
        />
        <Knob
          param="hint_strength"
          label={defaultLabelFor("hint_strength")}
          kbd={kbdHintFor("hint_strength")}
        />
        <Knob
          param="timbre_strength"
          label={defaultLabelFor("timbre_strength")}
          kbd={kbdHintFor("timbre_strength")}
        />
        <SeedKnob />
      </div>
      <div className="knob-ref-row">
        <TrackPicker />
        <RefControl kind="timbre" />
        <RefControl kind="structure" />
      </div>
    </div>
  );
}
