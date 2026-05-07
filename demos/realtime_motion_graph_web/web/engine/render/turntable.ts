// Turntable grooves — concentric polar polylines whose radii are modulated
// by the live audio mirror buffer, so the grooves literally render the
// music being played. Renders to a 2D canvas (was SVG paths until the
// per-frame setAttribute('d', …) thrash showed up in profiles).
//
// The canvas inherits the disc's CSS filter (drop-shadow keyed off
// --bloom-amount) so the warm coral glow follows the audio kick exactly
// like the SVG did.

const SEGMENTS = 72;
const VIEWBOX = 64;
const CENTER = VIEWBOX / 2;
const INNER_R = 13;
const OUTER_R = 28;
const AMPLITUDE = 1.6; // px deflection at sample = 1.0 (in viewBox units)
const N_GROOVES = 6;

// Stroke colors mirror the original SVG CSS exactly (alpha baked in).
const STROKE_EVEN = "rgba(232, 79, 61, 0.28)";
const STROKE_ODD = "rgba(240, 138, 72, 0.22)";

export interface Turntable {
  el: HTMLElement;
  canvas: HTMLCanvasElement;
  ctx: CanvasRenderingContext2D;
  resizeObs: ResizeObserver;
  w: number;
  h: number;
}

export function initTurntable(host: HTMLElement): Turntable | null {
  const canvas = host.querySelector(
    ".turntable-grooves",
  ) as HTMLCanvasElement | null;
  if (!canvas || canvas.tagName !== "CANVAS") return null;
  const ctx = canvas.getContext("2d");
  if (!ctx) return null;

  const tt: Turntable = {
    el: host,
    canvas,
    ctx,
    resizeObs: null as unknown as ResizeObserver,
    w: 1,
    h: 1,
  };
  const resize = () => {
    const dpr = window.devicePixelRatio || 1;
    const r = canvas.getBoundingClientRect();
    canvas.width = Math.max(1, Math.floor(r.width * dpr));
    canvas.height = Math.max(1, Math.floor(r.height * dpr));
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    tt.w = r.width;
    tt.h = r.height;
  };
  tt.resizeObs = new ResizeObserver(resize);
  tt.resizeObs.observe(canvas);
  resize();
  return tt;
}

/**
 * Repaint each groove. `mirror` is the interleaved Float32Array exposed by
 * AudioPlayer.getMirror(); each groove samples a different region of it so
 * the rings don't overlap visually. `time` advances the sample window so
 * the waveform appears to scroll, mimicking a record being played. `bloom`
 * is the same binned value the render loop writes into `--bloom-amount`
 * (passed in directly so we don't pay for a `getComputedStyle` flush per
 * frame).
 */
export function tickTurntable(
  tt: Turntable,
  mirror: Float32Array | null,
  channels: number,
  time: number,
  bloom = 0,
): void {
  const N = mirror ? Math.floor(mirror.length / Math.max(1, channels)) : 0;
  const phase = time * 0.35;
  const ctx = tt.ctx;
  const w = tt.w;
  const h = tt.h;
  if (w <= 0 || h <= 0) return;

  ctx.clearRect(0, 0, w, h);

  // Map viewBox (0..64) to actual canvas size; the host preserves a square
  // aspect, so width-based scaling is sufficient.
  const scale = w / VIEWBOX;

  const recording = tt.el.classList.contains("turntable--recording");

  ctx.save();
  ctx.lineWidth = (recording ? 1 + bloom * 0.6 : 0.85) * scale;
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
  ctx.globalAlpha = recording ? 0.5 : 1;

  for (let g = 0; g < N_GROOVES; g++) {
    const t = N_GROOVES > 1 ? g / (N_GROOVES - 1) : 0;
    const r0 = INNER_R + t * (OUTER_R - INNER_R);
    const grooveOffset = g * 0.137;

    ctx.strokeStyle = g % 2 === 0 ? STROKE_EVEN : STROKE_ODD;
    ctx.beginPath();
    for (let i = 0; i <= SEGMENTS; i++) {
      const theta = (i / SEGMENTS) * Math.PI * 2;
      let amp = 0;
      if (mirror && N > 0) {
        let pos = theta / (Math.PI * 2) + phase + grooveOffset;
        pos = pos - Math.floor(pos);
        const idx = Math.min(N - 1, Math.floor(pos * N));
        let s = 0;
        for (let c = 0; c < channels; c++) {
          s += mirror[idx * channels + c] || 0;
        }
        amp = (s / channels) * AMPLITUDE;
      }
      const r = r0 + amp;
      const x = (CENTER + r * Math.cos(theta)) * scale;
      const y = (CENTER + r * Math.sin(theta)) * scale;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.closePath();
    ctx.stroke();
  }
  ctx.restore();
}

export function destroyTurntable(tt: Turntable): void {
  try {
    tt.resizeObs.disconnect();
  } catch {}
}
