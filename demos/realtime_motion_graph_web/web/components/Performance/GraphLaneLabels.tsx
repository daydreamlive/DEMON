"use client";

import { useEffect, useRef, useState } from "react";

import {
  PLAYHEAD_INSET_PX_FRAC,
  getActiveGraphRenderer,
  type LaneBand,
  type LaneState,
} from "@/engine/render/GraphRenderer";

import { defaultLabelFor } from "./SliderTile";

// DOM overlay that paints DAW-style name pills next to each lane / line.
// Sits as a sibling of the graph canvas, positioned absolutely over it,
// so the text rendering is crisp (canvas text is fuzzy at small sizes)
// and the canvas hot draw loop stays untouched.
//
// Two layouts based on the renderer's mode:
//
//   "polylines"  — historical behavior. One pill per signal, anchored
//                  at the polyline's y at the playhead. Pills follow
//                  the line as it tweens.
//
//   "lanes"      — Concept A. One pill per active lane, anchored at the
//                  LEFT edge of the canvas at the lane's vertical
//                  center. Pills stay put (lane bands are stable across
//                  frames); they don't chase a moving polyline.
//
// Tick cadence is 50 ms (matches the graph sample rate), driven by a
// plain setInterval. Per PERFORMANCE.md this is NOT a hot loop — the
// allocation is small (≤ MAX_LANES entries) at 20 Hz, and the canvas-
// side hot path doesn't change.

const TICK_MS = 50;

type DisplayMode = "polylines" | "lanes";

interface DisplayPill {
  param: string;
  display: string;
  // Pixel offsets within the overlay (which mirrors the canvas bounds).
  x: number;
  y: number;
  color: string;
  idle: boolean;
  // Special "+N more" overflow chip. Rendered with a neutral CSS
  // variant (.graph-lane-label--more) — color/borderColor inline are
  // ignored in favor of CSS tokens.
  isMore?: boolean;
}

// Sentinel param name for the overflow chip so React's key stays stable
// across ticks (the count changes but the chip is the same DOM node).
const MORE_PILL_PARAM = "__hidden_more__";

// Display label for a param when rendering a lane pill in the gutter.
// The gutter is ~168px wide; long lora_str_<id> names overflow even
// after the v2 widening. The colored track tag inside the gutter
// already conveys "this is a LoRA, here's its family color", so the
// "LORA STR" prefix on the label is redundant noise. Strip it and let
// the LoRA's own name carry the pill.
//
// Sliders elsewhere in the mixer keep the full defaultLabelFor name
// (they have more horizontal room) — this override is graph-only.
function laneDisplayFor(param: string): string {
  if (param.startsWith("lora_str_")) {
    return param.slice("lora_str_".length).replace(/[_-]/g, " ");
  }
  return defaultLabelFor(param);
}

export function GraphLaneLabels() {
  const [pills, setPills] = useState<DisplayPill[]>([]);
  const [mode, setMode] = useState<DisplayMode>("polylines");
  const [size, setSize] = useState({ w: 0, h: 0 });
  const stableSigRef = useRef<string>(""); // signature for cheap equality

  useEffect(() => {
    // Mirror the renderer's layoutMode onto a `data-graph-layout`
    // attribute on `#graph-wrap`. CSS uses this to switch the
    // canvas-edge mask off in lanes mode — the historical
    // `-webkit-mask-image` horizontal feather on `#graph` was designed
    // for polylines (old samples drift left and fade out gracefully).
    // In lanes mode it fades the gutter labels and the playhead-anchor
    // dots; CSS overrides `--graph-feather-x: 0` when this attribute
    // is "lanes".
    const wrap =
      typeof document !== "undefined"
        ? (document.querySelector("#graph-wrap") as HTMLElement | null)
        : null;
    const id = window.setInterval(() => {
      const g = getActiveGraphRenderer();
      if (!g) return;
      const w = g.cssWidth;
      const h = g.cssHeight;
      const renderMode: DisplayMode = g.layoutMode;
      // Update the wrap attribute when it changes. Cheap string
      // compare avoids touching the DOM every tick.
      if (wrap && wrap.getAttribute("data-graph-layout") !== renderMode) {
        wrap.setAttribute("data-graph-layout", renderMode);
      }

      const next: DisplayPill[] = [];

      if (renderMode === "lanes") {
        // Read the renderer's published lane bands. Pill anchors at
        // bandTop + bandHeight/2 (vertical center), positioned at the
        // canvas left edge — Ableton's clip-view labels live to the
        // LEFT of the lane, not chasing a polyline at the playhead.
        const bands = g.getLaneBands();
        const count = g.getLaneBandCount();
        for (let i = 0; i < count; i++) {
          const b: LaneBand = bands[i];
          if (!b || !b.name) continue;
          next.push({
            param: b.name,
            display: laneDisplayFor(b.name),
            x: 0,
            y: b.bandTop + b.bandHeight / 2,
            color: `rgb(${b.color[0]},${b.color[1]},${b.color[2]})`,
            idle: b.idle,
          });
        }
        // Overflow chip — when more lanes qualify than MAX_LANES, surface
        // the hidden count as a neutral "+N more" pill anchored below
        // the last visible lane in the gutter. Centers a few px below
        // the stack so it reads as a continuation marker, not a lane.
        const hidden = g.getHiddenLaneCount();
        if (hidden > 0 && count > 0) {
          const last = bands[count - 1];
          if (last && last.bandTop > 0) {
            const chipY = last.bandTop + last.bandHeight + 14;
            next.push({
              param: MORE_PILL_PARAM,
              display: `+${hidden} more`,
              x: 0,
              y: chipY,
              color: "", // ignored; CSS variant carries the neutral color
              idle: false,
              isMore: true,
            });
          }
        }
      } else {
        // Polylines mode — pill anchored at the polyline's y at the
        // playhead, same as before. Skip lines whose value is currently
        // zero (untouched LoRAs / pinned params) so 14 pills don't pile
        // up on top of each other at y = laneBottom.
        const states = g.getLaneStates();
        const playheadX = w * (1 - PLAYHEAD_INSET_PX_FRAC);
        for (const s of states as LaneState[]) {
          if (s.value < 0.005) continue;
          next.push({
            param: s.name,
            display: defaultLabelFor(s.name),
            x: playheadX,
            y: s.y,
            color: `rgb(${s.color[0]},${s.color[1]},${s.color[2]})`,
            idle: false,
          });
        }
      }

      // Cheap dirty-check: mode + signature of (param + rounded y +
      // idle flag) for each pill. Avoids React state churn when nothing
      // visible moved between ticks.
      let sig = `${renderMode}|${w}x${h}|`;
      for (const l of next) {
        sig += `${l.param}:${Math.round(l.y)}:${l.idle ? "1" : "0"}|`;
      }
      if (sig === stableSigRef.current) return;
      stableSigRef.current = sig;
      setPills(next);
      setSize({ w, h });
      if (renderMode !== mode) setMode(renderMode);
    }, TICK_MS);
    return () => {
      window.clearInterval(id);
      // Clean up the data attribute so the mask reverts to its default
      // (polyline-mode) behavior if the labels component unmounts.
      if (wrap) wrap.removeAttribute("data-graph-layout");
    };
  }, [mode]);

  if (pills.length === 0 || size.w === 0) return null;

  return (
    <div
      className={`graph-lane-labels graph-lane-labels--${mode}`}
      style={{ width: size.w, height: size.h }}
      aria-hidden="true"
    >
      {pills.map((p) => {
        const classes = ["graph-lane-label"];
        if (p.idle) classes.push("graph-lane-label--idle");
        if (p.isMore) classes.push("graph-lane-label--more");
        // For the overflow chip, omit inline color/borderColor so the
        // CSS variant's neutral palette takes over.
        const style: React.CSSProperties = p.isMore
          ? { left: p.x, top: p.y }
          : { left: p.x, top: p.y, color: p.color, borderColor: p.color };
        return (
          <span key={p.param} className={classes.join(" ")} style={style}>
            {p.display}
          </span>
        );
      })}
    </div>
  );
}
