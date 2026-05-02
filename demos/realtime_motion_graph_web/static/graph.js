// Parameter-history graph display, mirroring the python GUI's hud.py trails.
// Maintains a rolling buffer per signal and renders glowing polylines.

const GRAPH_COLORS = {
  denoise:       [0, 255, 0],
  feedback:      [0, 200, 255],
  shift:         [180, 0, 255],
  hint_strength: [255, 200, 50],
  noise_share:   [120, 120, 255],
  ode_noise:     [200, 200, 200],
  seed:          [255, 180, 0],
  ch_g0: [255, 80, 80],  ch_g1: [255, 160, 60], ch_g2: [255, 220, 40], ch_g3: [180, 255, 60],
  ch_g4: [60, 255, 140], ch_g5: [40, 220, 255], ch_g6: [100, 140, 255], ch_g7: [200, 120, 255],
  ch13: [255, 100, 100], ch14: [255, 180, 80],  ch19: [220, 255, 80],
  ch23: [80, 255, 180],  ch29: [80, 180, 255],  ch56: [180, 80, 255],
};

// Hash a LoRA id to a stable hue so per-LoRA history lines stay visually
// distinct on the graph without requiring config-driven colors.
const _LORA_HUE_PALETTE = [
  [255, 50, 200], [200, 50, 255], [50, 200, 255], [255, 150, 50],
  [120, 255, 80], [255, 80, 120], [180, 255, 200], [255, 200, 100],
];
function _loraColor(id) {
  let h = 0;
  for (let i = 0; i < id.length; i++) h = (h * 31 + id.charCodeAt(i)) | 0;
  return _LORA_HUE_PALETTE[Math.abs(h) % _LORA_HUE_PALETTE.length];
}
function _colorFor(name) {
  if (name in GRAPH_COLORS) return GRAPH_COLORS[name];
  if (name.startsWith("lora_str_")) return _loraColor(name.slice("lora_str_".length));
  return [255, 255, 255];
}

const HISTORY_LEN = 600;

export class GraphRenderer {
  constructor(canvas) {
    this.canvas = canvas;
    this.ctx = canvas.getContext("2d");
    this.histories = new Map();
    this.w = 1;
    this.h = 1;
    this._resize();
    this._resizeObs = new ResizeObserver(() => this._resize());
    this._resizeObs.observe(canvas);
  }

  _resize() {
    const dpr = window.devicePixelRatio || 1;
    const r = this.canvas.getBoundingClientRect();
    this.canvas.width = Math.max(1, Math.floor(r.width * dpr));
    this.canvas.height = Math.max(1, Math.floor(r.height * dpr));
    this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    this.w = r.width;
    this.h = r.height;
  }

  sample(values, defs) {
    for (const [name, v] of Object.entries(values)) {
      const max = defs[name]?.max || 1;
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

  draw(frac = 1, pulse = 0) {
    // Self-heal: if the canvas was hidden when we first measured, re-measure now.
    const r = this.canvas.getBoundingClientRect();
    if (Math.abs(r.width - this.w) > 0.5 || Math.abs(r.height - this.h) > 0.5) {
      this._resize();
    }
    const ctx = this.ctx;
    const { w, h } = this;
    pulse = Math.max(0, Math.min(1, pulse));

    ctx.fillStyle = "#000";
    ctx.fillRect(0, 0, w, h);

    const playheadX = Math.max(0, Math.min(w - 1, frac * w));

    // Audio-reactive background haze behind the playhead.
    if (pulse > 0.02) {
      const grad = ctx.createRadialGradient(playheadX, h / 2, 0, playheadX, h / 2, h * 0.8);
      grad.addColorStop(0, `rgba(150, 180, 220, ${0.18 * pulse})`);
      grad.addColorStop(1, "rgba(150, 180, 220, 0)");
      ctx.fillStyle = grad;
      ctx.fillRect(0, 0, w, h);
    }

    // Glow pass: blurred additive polylines, intensity modulated by RMS.
    ctx.save();
    ctx.globalCompositeOperation = "lighter";
    ctx.filter = `blur(${4 + 3 * pulse}px)`;
    ctx.lineWidth = 3 + 2 * pulse;
    const glowAlpha = 0.8 + 0.2 * pulse;
    for (const [name, hist] of this.histories) {
      this._polyline(ctx, hist, playheadX, _colorFor(name), glowAlpha);
    }
    ctx.restore();

    // Sharp 1px lines on top.
    ctx.save();
    ctx.lineWidth = 1;
    for (const [name, hist] of this.histories) {
      this._polyline(ctx, hist, playheadX, _colorFor(name), 1);
    }
    ctx.restore();

    // Playhead: thick soft glow scaled by RMS, then a crisp 1px top.
    ctx.save();
    ctx.globalCompositeOperation = "lighter";
    ctx.filter = `blur(${6 * pulse}px)`;
    ctx.strokeStyle = `rgba(150, 180, 220, ${0.9 * pulse})`;
    ctx.lineWidth = 2 + 5 * pulse;
    ctx.beginPath();
    ctx.moveTo(playheadX + 0.5, 0);
    ctx.lineTo(playheadX + 0.5, h);
    ctx.stroke();
    ctx.restore();

    ctx.strokeStyle = `rgba(255, 255, 255, ${0.6 + 0.4 * pulse})`;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(playheadX + 0.5, 0);
    ctx.lineTo(playheadX + 0.5, h);
    ctx.stroke();
  }

  _polyline(ctx, hist, playheadX, [r, g, b], alpha) {
    const n = Math.min(hist.filled, Math.floor(playheadX));
    if (n < 2) return;
    ctx.strokeStyle = `rgba(${r},${g},${b},${alpha})`;
    ctx.beginPath();
    for (let i = 0; i < n; i++) {
      const bufIdx = (hist.head - n + i + HISTORY_LEN) % HISTORY_LEN;
      const v = hist.buf[bufIdx];
      const x = playheadX - (n - 1) + i;
      const y = this.h - v * this.h;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();
  }

  destroy() {
    this._resizeObs.disconnect();
  }
}
