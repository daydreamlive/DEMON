// Composites the waveform scrub strip (top band) + the motion graph
// (main body) into a single offscreen canvas at recording resolution,
// then exposes a captureStream() the MediaRecorder can consume.
//
// Why an offscreen compositor instead of capturing the graph canvas
// directly: the user wants the recorded clip to look like the
// performance — graph + waveform together, not just the graph in
// isolation. The waveform canvases live in a separate DOM container
// (.waveform-scrub-bg / .waveform-scrub-fg, owned by WaveformScrubBox)
// and can't be merged into the graph canvas at the source. So we
// blit both into a private offscreen canvas each frame and capture
// THAT.
//
// Tick uses its own rAF loop (not the shared frameScheduler) because
// it only runs during an active recording — registering a permanent
// frame budget for a feature that's off 99% of the time would be
// wasteful. The source canvases are already being drawn each frame
// by their own loops; we're just reading them.

import { getActiveGraphRenderer } from "@/engine/render/GraphRenderer";

// Recording target. 1280x720 is a standard download resolution that
// reads cleanly on a phone and a laptop; bumping to 1920x1080 doubles
// the bitrate cost with no visible win for the line-art graph.
const WIDTH = 1280;
const HEIGHT = 720;

// Vertical split — waveform strip takes the top band, graph fills the
// rest. WAVEFORM_FRAC is intentionally generous (~14%) so the peaks
// don't get squished into a hairline when scaled down from the
// full-width DOM strip.
const WAVEFORM_FRAC = 0.14;
const GAP_PX = 6;

export class RecordingCompositor {
  readonly canvas: HTMLCanvasElement;
  private ctx: CanvasRenderingContext2D;
  private rafId = 0;
  private running = false;

  constructor() {
    this.canvas = document.createElement("canvas");
    this.canvas.width = WIDTH;
    this.canvas.height = HEIGHT;
    const ctx = this.canvas.getContext("2d", { desynchronized: true });
    if (!ctx) {
      throw new Error("RecordingCompositor: 2d context unavailable");
    }
    this.ctx = ctx;
  }

  /** Begin rAF compositing and return a MediaStream tracking the
   *  offscreen canvas at 30 fps. Idempotent — calling twice returns
   *  a fresh stream off the same canvas. */
  start(): MediaStream {
    this.running = true;
    const tick = () => {
      if (!this.running) return;
      this.drawFrame();
      this.rafId = requestAnimationFrame(tick);
    };
    this.rafId = requestAnimationFrame(tick);
    return this.canvas.captureStream(30);
  }

  /** Stop the rAF loop. The captureStream tracks end on their own once
   *  the consumer (MediaRecorder) closes them — no track-stop needed
   *  here. */
  stop() {
    this.running = false;
    if (this.rafId) {
      cancelAnimationFrame(this.rafId);
      this.rafId = 0;
    }
  }

  private drawFrame() {
    const { ctx } = this;
    const w = WIDTH;
    const h = HEIGHT;

    // Solid backdrop matches the install-stage HUD color so the recorded
    // clip reads as a slice of the app, not a transparent overlay.
    ctx.fillStyle = "rgb(8, 8, 14)";
    ctx.fillRect(0, 0, w, h);

    const waveH = Math.round(h * WAVEFORM_FRAC);
    const graphY = waveH + GAP_PX;
    const graphH = h - graphY;

    // Waveform strip — two canvases (bg = peaks, fg = playhead + band)
    // stacked. Queried by class because they're owned by an unrelated
    // component (WaveformScrubBox); no prop drilling. Skips silently
    // on mobile / pre-buffer where the scrub box isn't mounted.
    const wfBg = document.querySelector<HTMLCanvasElement>(
      ".waveform-scrub-bg",
    );
    const wfFg = document.querySelector<HTMLCanvasElement>(
      ".waveform-scrub-fg",
    );
    if (wfBg && wfBg.width > 0 && wfBg.height > 0) {
      ctx.drawImage(wfBg, 0, 0, w, waveH);
    }
    if (wfFg && wfFg.width > 0 && wfFg.height > 0) {
      ctx.drawImage(wfFg, 0, 0, w, waveH);
    }

    // Graph fills the rest. The renderer canvas is sized to the DOM
    // container (variable aspect); we drawImage it with an explicit
    // dest rect so the recording's aspect ratio stays fixed regardless
    // of the user's viewport size.
    const graph = getActiveGraphRenderer()?.canvas;
    if (graph && graph.width > 0 && graph.height > 0) {
      ctx.drawImage(graph, 0, graphY, w, graphH);
    }
  }
}
