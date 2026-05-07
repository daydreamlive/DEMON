// Parameter-history graph display. Maintains a rolling buffer per signal
// and renders glowing polylines + a playhead. Direct port of static/graph.js.
//
// Phase 4 perf note: ctx.filter:blur is software-rendered in many browsers;
// reduce blur intensity slightly vs. the original (4px → 3px baseline) to
// keep frame budget healthy. Visually nearly identical at typical pulse
// values.

import { SLIDER_META, type SliderMeta } from "@/types/engine";

type RGB = [number, number, number];

const GRAPH_COLORS: Record<string, RGB> = {
  denoise: [61, 182, 190],
  feedback: [240, 138, 72],
  shift: [232, 79, 61],
  hint_strength: [199, 181, 102],
  noise_share: [61, 182, 190],
  ode_noise: [199, 181, 102],
  seed: [240, 138, 72],
  ch_g0: [255, 80, 80],
  ch_g1: [255, 160, 60],
  ch_g2: [255, 220, 40],
  ch_g3: [180, 255, 60],
  ch_g4: [60, 255, 140],
  ch_g5: [40, 220, 255],
  ch_g6: [100, 140, 255],
  ch_g7: [200, 120, 255],
  ch13: [255, 100, 100],
  ch14: [255, 180, 80],
  ch19: [220, 255, 80],
  ch23: [80, 255, 180],
  ch29: [80, 180, 255],
  ch56: [180, 80, 255],
};

const _LORA_HUE_PALETTE: RGB[] = [
  [255, 50, 200],
  [200, 50, 255],
  [50, 200, 255],
  [255, 150, 50],
  [120, 255, 80],
  [255, 80, 120],
  [180, 255, 200],
  [255, 200, 100],
];

function _loraColor(id: string): RGB {
  let h = 0;
  for (let i = 0; i < id.length; i++) h = (h * 31 + id.charCodeAt(i)) | 0;
  return _LORA_HUE_PALETTE[Math.abs(h) % _LORA_HUE_PALETTE.length];
}

function _colorFor(name: string): RGB {
  if (name in GRAPH_COLORS) return GRAPH_COLORS[name];
  if (name.startsWith("lora_str_"))
    return _loraColor(name.slice("lora_str_".length));
  return [255, 255, 255];
}

const HISTORY_LEN = 600;
// How many of the most-recent samples fill the canvas width. Sampling
// runs at SAMPLE_INTERVAL_MS = 50 ms (see useRenderLoop), so 120 samples =
// 6 s of visible history. Newest sample sits at the right edge; samples
// older than this drift past the left edge and are clipped (not recycled
// back onto the right). The playhead is offset rightward of center by
// PLAYHEAD_LEAD_SEC so a fresh slider change drifts from the right edge
// to the playhead in ~engine-latency seconds — i.e. the visual marker
// reaches the playhead just as the audio change becomes audible.
const VISIBLE_SAMPLES = 120;
const VISIBLE_SEC = 6; // VISIBLE_SAMPLES * SAMPLE_INTERVAL_MS / 1000
// Lead time between the right edge ("just sampled") and the playhead
// ("audible now"). Tuned to the engine's round-trip latency. If the graph
// reads ahead of the audio, raise; if behind, lower.
const PLAYHEAD_LEAD_SEC = 1.0;
// Vertical breathing room so polylines at v=0 / v=1 (e.g. side-LoRA
// strengths pulled all the way) aren't clipped against the canvas edge.
// Sized for the max stroke width (5 px) + shadow blur (~3 px) + a little
// air.
const Y_PAD = 12;

interface History {
  buf: Float32Array;
  head: number;
  filled: number;
}

export class GraphRenderer {
  readonly canvas: HTMLCanvasElement;
  private readonly ctx: CanvasRenderingContext2D;
  private readonly histories: Map<string, History> = new Map();
  private readonly _resizeObs: ResizeObserver;
  private w = 1;
  private h = 1;

  constructor(canvas: HTMLCanvasElement) {
    this.canvas = canvas;
    const ctx = canvas.getContext("2d");
    if (!ctx) throw new Error("GraphRenderer: 2D context unavailable");
    this.ctx = ctx;
    this._resizeObs = new ResizeObserver(() => this._resize());
    this._resizeObs.observe(canvas);
    this._resize();
  }

  private _resize(): void {
    const dpr = window.devicePixelRatio || 1;
    const r = this.canvas.getBoundingClientRect();
    this.canvas.width = Math.max(1, Math.floor(r.width * dpr));
    this.canvas.height = Math.max(1, Math.floor(r.height * dpr));
    this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    this.w = r.width;
    this.h = r.height;
  }

  /** Append a new sample point per signal. `defs` supplies max for normalization. */
  sample(
    values: Record<string, number>,
    defs: Record<string, SliderMeta> = SLIDER_META,
  ): void {
    for (const name of Object.keys(values)) {
      const v = values[name];
      const max = defs[name]?.max ?? 1;
      let hist = this.histories.get(name);
      if (!hist) {
        hist = { buf: new Float32Array(HISTORY_LEN), head: 0, filled: 0 };
        this.histories.set(name, hist);
      }
      hist.buf[hist.head] = Math.max(0, Math.min(1, v / max));
      hist.head = (hist.head + 1) % HISTORY_LEN;
      if (hist.filled < HISTORY_LEN) hist.filled += 1;
    }
  }

  draw(pulse = 0): void {
    // ResizeObserver in the constructor already keeps {w, h} in sync,
    // including the display:none → block transition. The legacy
    // getBoundingClientRect() self-heal that used to live here forced a
    // synchronous full-document layout flush every frame, clearing the
    // browser's paint-region caches and tanking cursor box-shadow paint.
    const ctx = this.ctx;
    const { w, h } = this;
    pulse = Math.max(0, Math.min(1, pulse));

    ctx.fillStyle = "#000";
    ctx.fillRect(0, 0, w, h);

    // Playhead offset rightward so the newest sample at the right edge takes
    // PLAYHEAD_LEAD_SEC to drift to the playhead. That delay matches the
    // engine round-trip, so a slider change reaches the playhead just as the
    // audio change becomes audible.
    const playheadX = w * (1 - PLAYHEAD_LEAD_SEC / VISIBLE_SEC);

    if (pulse > 0.02) {
      const grad = ctx.createRadialGradient(
        playheadX,
        h / 2,
        0,
        playheadX,
        h / 2,
        h * 0.8,
      );
      grad.addColorStop(0, `rgba(150, 180, 220, ${0.18 * pulse})`);
      grad.addColorStop(1, "rgba(150, 180, 220, 0)");
      ctx.fillStyle = grad;
      ctx.fillRect(0, 0, w, h);
    }

    // Glow + crisp line per signal. Build the path once, stroke twice:
    // first wide with shadowBlur (GPU-accelerated on Chromium/Safari's
    // accelerated canvas), then thin and sharp on top. This replaces
    // `ctx.filter = blur(...)`, which falls back to software in Skia
    // and was eating 5–15 ms / frame during active music — exactly the
    // "whole-page lag during beats" shape.
    for (const [name, hist] of this.histories) {
      const n = Math.min(hist.filled, VISIBLE_SAMPLES);
      if (n < 2) continue;
      const [r, g, b] = _colorFor(name);

      const pxPerSample = w / (VISIBLE_SAMPLES - 1);
      const xStart = w - (n - 1) * pxPerSample;
      ctx.beginPath();
      for (let i = 0; i < n; i++) {
        // Walk the ring backward from the newest sample (head - 1) so we
        // always plot the freshest n entries in chronological order.
        const bufIdx = (hist.head - n + i + HISTORY_LEN) % HISTORY_LEN;
        const v = hist.buf[bufIdx];
        const x = xStart + i * pxPerSample;
        const y = (h - Y_PAD) - v * (h - 2 * Y_PAD);
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }

      if (pulse > 0.05) {
        ctx.save();
        ctx.globalCompositeOperation = "lighter";
        ctx.shadowColor = `rgba(${r},${g},${b},${0.8 + 0.2 * pulse})`;
        ctx.shadowBlur = 3 + 2 * pulse;
        ctx.shadowOffsetX = 0;
        ctx.shadowOffsetY = 0;
        ctx.strokeStyle = `rgba(${r},${g},${b},${0.6 + 0.4 * pulse})`;
        ctx.lineWidth = 3 + 2 * pulse;
        ctx.stroke();
        ctx.restore();
      }

      ctx.shadowBlur = 0;
      ctx.strokeStyle = `rgba(${r},${g},${b},1)`;
      ctx.lineWidth = 1;
      ctx.stroke();
    }

    // Playhead — same shadowBlur trick. Glow halo + crisp 1px line.
    if (pulse > 0.05) {
      ctx.save();
      ctx.globalCompositeOperation = "lighter";
      ctx.shadowColor = `rgba(150, 180, 220, ${0.9 * pulse})`;
      ctx.shadowBlur = 4 * pulse;
      ctx.shadowOffsetX = 0;
      ctx.shadowOffsetY = 0;
      ctx.strokeStyle = `rgba(150, 180, 220, ${0.9 * pulse})`;
      ctx.lineWidth = 2 + 5 * pulse;
      ctx.beginPath();
      ctx.moveTo(playheadX + 0.5, 0);
      ctx.lineTo(playheadX + 0.5, h);
      ctx.stroke();
      ctx.restore();
    }

    ctx.shadowBlur = 0;
    ctx.strokeStyle = `rgba(255, 255, 255, ${0.6 + 0.4 * pulse})`;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(playheadX + 0.5, 0);
    ctx.lineTo(playheadX + 0.5, h);
    ctx.stroke();
  }

  destroy(): void {
    this._resizeObs.disconnect();
  }
}
