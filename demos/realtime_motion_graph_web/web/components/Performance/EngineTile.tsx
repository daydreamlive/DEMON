"use client";

import { defaultLabelFor, SliderTile } from "./SliderTile";

const PARAMS = [
  "feedback",
  "shift",
  "noise_share",
  "ode_noise",
];

export function EngineTile() {
  return (
    <SliderTile
      label="Engine"
      params={PARAMS.map((p) => ({ param: p, label: defaultLabelFor(p) }))}
    />
  );
}
