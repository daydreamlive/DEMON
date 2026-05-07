"use client";

import { defaultLabelFor, SliderTile } from "./SliderTile";

const PARAMS = ["ch_g0", "ch_g1", "ch_g2", "ch_g3", "ch_g4", "ch_g5", "ch_g6", "ch_g7"];

export function ChannelGainsTile() {
  return (
    <SliderTile
      label="Channel Gains"
      params={PARAMS.map((p) => ({ param: p, label: defaultLabelFor(p) }))}
    />
  );
}
