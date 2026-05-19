"use client";

import { RefControl } from "./RefControl";
import { SliderGroup } from "./SliderGroup";

export function MainTile() {
  return (
    <div className="mixer-tile" data-tile="main">
      <div className="mixer-tile-label">Main</div>
      <div className="mixer-channels" id="sliders">
        <SliderGroup param="denoise" label="remix strength" kbd="A + ▲▼" />
        <SliderGroup
          param="hint_strength"
          label="structure strength"
          kbd="G + ▲▼"
        />
        <SliderGroup
          param="timbre_strength"
          label="timbre strength"
          kbd="C + ▲▼"
        />
      </div>
      <RefControl kind="timbre" />
      <RefControl kind="structure" />
    </div>
  );
}
