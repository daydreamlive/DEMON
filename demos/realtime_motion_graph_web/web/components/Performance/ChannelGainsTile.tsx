"use client";

import { useConfig } from "@/lib/config";

import { defaultLabelFor, SliderTile } from "./SliderTile";

const PARAMS = ["ch_g0", "ch_g1", "ch_g2", "ch_g3", "ch_g4", "ch_g5", "ch_g6", "ch_g7"];

export function ChannelGainsTile() {
  const ranges = useConfig().channel_ranges;
  return (
    <SliderTile
      label="Channel Gains (experimental)"
      params={PARAMS.map((p) => {
        const r = ranges[p];
        return {
          param: p,
          label: defaultLabelFor(p),
          // Real per-channel range from config (e.g. ch_g4 → [0, 2.5],
          // ch_g0 → [0, 2.2]); the slider widget translates between
          // its visual rail [0, 2] and these actual bounds via
          // lib/sliderMapping with unity=1.0 anchoring at the rail's
          // midpoint. Defaults of 1.0 for every channel therefore land
          // at the same height across the bank.
          min: r?.min,
          max: r?.max,
          reverse: r?.reverse,
          unity: 1.0,
        };
      })}
    />
  );
}
