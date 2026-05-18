"use client";

import { useEffect, useRef, useState } from "react";

import {
  PLAYHEAD_INSET_PX_FRAC,
  getActiveGraphRenderer,
  type LaneState,
} from "@/engine/render/GraphRenderer";

import { defaultLabelFor } from "./SliderTile";

// DOM overlay that paints DAW-style name pills at each line's playhead
// intersection. Sits as a sibling of the graph canvas, positioned
// absolutely over it, so the text rendering is crisp (canvas text is
// fuzzy at small sizes) and the canvas hot draw loop stays untouched.
//
// Tick cadence is 50 ms (matches the graph sample rate), driven by a
// plain setInterval. Per PERFORMANCE.md this is NOT a hot loop —
// `getLaneStates()` returns a fresh array per call, but at 20 Hz the
// allocation is negligible. The canvas-side hot path doesn't change.

const TICK_MS = 50;

interface DisplayLane {
  param: string;
  display: string;
  y: number;
  color: string;
}

export function GraphLaneLabels() {
  const [lanes, setLanes] = useState<DisplayLane[]>([]);
  const [size, setSize] = useState({ w: 0, h: 0 });
  const stableLanesRef = useRef<string>(""); // signature for cheap equality

  useEffect(() => {
    const id = window.setInterval(() => {
      const g = getActiveGraphRenderer();
      if (!g) return;
      const states = g.getLaneStates();
      const w = g.cssWidth;
      const h = g.cssHeight;
      // Build the next display lane list. Skip lines whose value is
      // currently zero — the user hasn't touched these knobs (or they're
      // pinned at zero), and showing 14 pills all at the same y is
      // visual noise.
      const next: DisplayLane[] = [];
      for (const s of states) {
        if (s.value < 0.005) continue;
        next.push({
          param: s.name,
          display: defaultLabelFor(s.name),
          y: s.y,
          color: `rgb(${s.color[0]},${s.color[1]},${s.color[2]})`,
        });
      }
      // Cheap dirty-check: signature of (param + rounded y) for each
      // lane. Avoids React state churn when nothing visible moved.
      let sig = `${w}x${h}|`;
      for (const l of next) sig += `${l.param}:${Math.round(l.y)}|`;
      if (sig === stableLanesRef.current) return;
      stableLanesRef.current = sig;
      setLanes(next);
      setSize({ w, h });
    }, TICK_MS);
    return () => window.clearInterval(id);
  }, []);

  if (lanes.length === 0 || size.w === 0) return null;
  const playheadX = size.w * (1 - PLAYHEAD_INSET_PX_FRAC);

  return (
    <div
      className="graph-lane-labels"
      style={{ width: size.w, height: size.h }}
      aria-hidden="true"
    >
      {lanes.map((l) => (
        <span
          key={l.param}
          className="graph-lane-label"
          style={{
            left: playheadX,
            top: l.y,
            color: l.color,
            borderColor: l.color,
          }}
        >
          {l.display}
        </span>
      ))}
    </div>
  );
}
