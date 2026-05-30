"use client";

import { useCustomTracksStore } from "@/store/useCustomTracksStore";
import { usePerformanceStore } from "@/store/usePerformanceStore";

import { Knob } from "./Knob";
import { RefControl } from "./RefControl";
import { SeedKnob } from "./SeedKnob";
import { defaultLabelFor, kbdHintFor } from "./SliderTile";
import { SourceModeSwitch } from "./SourceModeSwitch";
import { StemPanner } from "./StemPanner";
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
      <CoreStems />
    </div>
  );
}

// Stem layers under the file pickers — the same panners the HeroMacros
// bay carries, surfaced in the drawer so they stay reachable while the
// drawer is open (the bay's copy hides on drawer-open). Only shown for
// uploaded tracks (sourceMode set); built-in fixtures have no stems.
function CoreStems() {
  const fixture = usePerformanceStore((s) => s.fixture);
  const sourceMode = useCustomTracksStore((s) =>
    fixture ? s.tracks.get(fixture)?.sourceMode : undefined,
  );
  const stemStatus = useCustomTracksStore((s) =>
    fixture ? s.tracks.get(fixture)?.stemStatus : undefined,
  );
  const stemError = useCustomTracksStore((s) =>
    fixture ? s.tracks.get(fixture)?.stemError : undefined,
  );
  const stemsReady = useCustomTracksStore((s) =>
    Boolean(fixture && s.tracks.get(fixture)?.stems),
  );
  if (!fixture || !sourceMode) return null;
  // The source switch carries its own readout, so the status line only
  // narrates the in-flight / failed / pre-play states.
  const processing = stemStatus === "processing";
  const summary =
    processing
      ? "Swapping inference source…"
      : stemStatus === "failed"
        ? stemError || "Stem rip failed"
        : stemsReady
          ? "Inference source"
          : "Stems will load on play";
  return (
    <div className="core-stems">
      <div className="core-stems-label">Stem Layers</div>
      <div className="core-stems-status" title={stemError || undefined}>
        {summary}
      </div>
      <SourceModeSwitch fixture={fixture} current={sourceMode} busy={processing} />
      <div className="hero-stem-panners">
        <StemPanner kind="vocals" />
        <StemPanner kind="instruments" />
      </div>
    </div>
  );
}
