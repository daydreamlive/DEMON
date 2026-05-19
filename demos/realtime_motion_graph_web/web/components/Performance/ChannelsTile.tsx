"use client";

import { useConfig } from "@/lib/config";

import { defaultLabelFor, SliderTile } from "./SliderTile";

const PARAMS = ["ch13", "ch14", "ch19", "ch23", "ch29", "ch56"];

export function ChannelsTile() {
  const ranges = useConfig().channel_ranges;
  return (
    <SliderTile
      label="Channels (experimental)"
      params={PARAMS.map((p) => {
        const r = ranges[p];
        return {
          param: p,
          label: defaultLabelFor(p),
          // Real per-channel range from config; unity=1.0 anchors the
          // default to the rail midpoint so the whole bank lines up.
          // See ChannelGainsTile + lib/sliderMapping.ts.
          min: r?.min,
          max: r?.max,
          reverse: r?.reverse,
          unity: 1.0,
        };
      })}
    />
  );
}
