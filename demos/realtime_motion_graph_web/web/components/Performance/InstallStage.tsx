"use client";

import { forwardRef } from "react";

import { GraphLaneLabels } from "./GraphLaneLabels";
import { GraphPauseOverlay } from "./GraphPauseOverlay";
import { ScheduleCurvesOverlay } from "./ScheduleCurvesOverlay";

// Stage shell: ambient video back-layer, focal video pair (A/B crossfade),
// effects canvas overlay, focal-side blur strips, plus the legacy hud + graph
// canvases (kept for app.js feature parity; CSS hides them in install mode).
//
// All canvases / videos are exposed via refs so subsequent phases (HUD,
// Graph, EffectsRenderer, VideoLayer) can attach without JSX-level coupling.

export interface InstallStageRefs {
  ambient: HTMLVideoElement | null;
  videoA: HTMLVideoElement | null;
  videoB: HTMLVideoElement | null;
  effectsCanvas: HTMLCanvasElement | null;
  hudCanvas: HTMLCanvasElement | null;
  graphCanvas: HTMLCanvasElement | null;
}

interface Props {
  refs: {
    ambient: React.RefObject<HTMLVideoElement | null>;
    videoA: React.RefObject<HTMLVideoElement | null>;
    videoB: React.RefObject<HTMLVideoElement | null>;
    effectsCanvas: React.RefObject<HTMLCanvasElement | null>;
    hudCanvas: React.RefObject<HTMLCanvasElement | null>;
    graphCanvas: React.RefObject<HTMLCanvasElement | null>;
  };
}

export const InstallStage = forwardRef<HTMLDivElement, Props>(
  function InstallStage({ refs }, ref) {
    return (
      <div id="install-stage" ref={ref}>
        <video ref={refs.ambient} id="install-ambient" muted loop playsInline />
        <div id="install-video-area">
          <div id="video-wrap">
            <video ref={refs.videoA} id="video-a" muted playsInline />
            <video ref={refs.videoB} id="video-b" muted playsInline />
            <canvas
              ref={refs.effectsCanvas}
              id="effects-canvas"
              aria-hidden="true"
            />
            <div
              className="focal-side-blur focal-side-blur-left"
              aria-hidden="true"
            />
            <div
              className="focal-side-blur focal-side-blur-right"
              aria-hidden="true"
            />
          </div>
          <canvas ref={refs.hudCanvas} id="hud" />
          <div id="graph-wrap">
            <canvas ref={refs.graphCanvas} id="graph" />
            <GraphLaneLabels />
            <GraphPauseOverlay />
            <ScheduleCurvesOverlay />
          </div>
        </div>
      </div>
    );
  },
);
