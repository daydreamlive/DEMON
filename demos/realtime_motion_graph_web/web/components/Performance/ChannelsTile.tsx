"use client";

import { defaultLabelFor, SliderTile } from "./SliderTile";

const PARAMS = ["ch13", "ch14", "ch19", "ch23", "ch29", "ch56"];

export function ChannelsTile() {
  return (
    <SliderTile
      label="Channels"
      params={PARAMS.map((p) => ({ param: p, label: defaultLabelFor(p) }))}
    />
  );
}
