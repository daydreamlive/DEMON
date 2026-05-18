"use client";

import { RefControl } from "./RefControl";
import { SliderGroup } from "./SliderGroup";
import { defaultLabelFor, kbdHintFor } from "./SliderTile";

// The CORE tab body — the dial-it-and-go macros every musician knows.
// Six sliders in audio-traditional vocabulary (MIX / TRACK / TIMBRE /
// FEEDBACK / BASS / TREBLE) plus the two reference-track pickers that
// pair with TIMBRE + TRACK. Replaces the prior MainTile + the slider
// half of DcwTile inside the new tabbed drawer. The DCW on/off + mode
// + wavelet controls live in <ModTile/>.
//
// All labels route through defaultLabelFor() so DISPLAY_NAMES in
// SliderTile.tsx stays the single source of truth for graph-lane
// pills, MIDI map UI, and tile labels.
export function CoreTile() {
  return (
    <div className="mixer-tile" data-tile="core">
      <div className="mixer-tile-label">Core</div>
      <div className="mixer-channels" id="sliders">
        <SliderGroup
          param="denoise"
          label={defaultLabelFor("denoise")}
          kbd={kbdHintFor("denoise")}
        />
        <SliderGroup
          param="hint_strength"
          label={defaultLabelFor("hint_strength")}
          kbd={kbdHintFor("hint_strength")}
        />
        <SliderGroup
          param="timbre_strength"
          label={defaultLabelFor("timbre_strength")}
          kbd={kbdHintFor("timbre_strength")}
        />
        <SliderGroup
          param="feedback"
          label={defaultLabelFor("feedback")}
          kbd={kbdHintFor("feedback")}
        />
        <SliderGroup
          param="dcw_scaler"
          label={defaultLabelFor("dcw_scaler")}
          kbd={kbdHintFor("dcw_scaler")}
        />
        <SliderGroup
          param="dcw_high_scaler"
          label={defaultLabelFor("dcw_high_scaler")}
          kbd={kbdHintFor("dcw_high_scaler")}
        />
      </div>
      <RefControl kind="timbre" />
      <RefControl kind="structure" />
    </div>
  );
}
