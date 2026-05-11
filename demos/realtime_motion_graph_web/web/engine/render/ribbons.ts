// Multi-color organic ribbons painted along the three slider edges + the
// halo badge perimeter. Edge bars and the halo render to 2D canvases (was
// SVG paths until per-frame setAttribute('d', …) thrash showed up in
// profiles); the start-mark on the title screen stays SVG because it
// participates in CSS transform animations during the launch sequence.
//
// Visual contract: same trig-noise writhe math as before, same stroke
// colors, same per-frame stroke width / opacity scaling against
// --bloom-amount. Drop-shadow glow filters live in CSS on the canvas
// elements (CSS filter applies to canvases just like SVG).

import { LORA_SIDE_VISIBLE_FLOOR, REMIX_VISIBLE_FLOOR } from "@/types/engine";

const PALETTE = [
  "#3db6be", // teal
  "#c7b566", // mustard
  "#f08a48", // orange
  "#e84f3d", // coral
];

const ALONG = 1000;
const ACROSS = 100;
const SEGMENTS = 24; // perf: 36 -> 24
const RIBBON_SPACING = 3;
const NOISE_AMP_BASE = 6;
const NOISE_AMP_KICK = 8;
const INWARD_DISTANCE = 8;
// Along-axis margin (in canvas CSS pixels) reserved for stroke half-width
// + bloom drop-shadow halo, so the writhe path's start and end don't sit
// flush against the canvas bitmap edge and get sliced. Stroke peaks at
// ~6 px wide and the drop-shadow blur reaches ~11 px past the stroke at
// max --bloom-amount, so 16 px leaves clear headroom on both ends.
const ALONG_END_INSET_PX = 16;

// "Filled" colored ribbons converge onto a shared meeting point in the
// last HEAD_CONVERGE_START fraction of their writhe, then each ribbon's
// path continues into a polar wrap around the halo center — so the four
// ribbons LITERALLY become the halo at the slider's value position.
// No separate halo draw call; the wrap is the halo.
const HEAD_CONVERGE_START = 0.85;
const HEAD_HALO_BASE_R_PX = 9;
const HEAD_HALO_KICK_R_PX = 4;
const HEAD_HALO_RADIAL_SPREAD_PX = 1.1;
const HEAD_HALO_NOISE_AMP_BASE_PX = 0.9;
const HEAD_HALO_NOISE_AMP_KICK_PX = 1.6;
const HEAD_HALO_WRAP_STEPS = 28;
// Wrap covers ~340° so the ribbon ends just shy of closing back on
// itself — implies a coil at the thumb position without a visible seam.
const HEAD_HALO_WRAP_SPAN = Math.PI * 2 * 0.95;
const HEAD_HALO_FAN_FRACTION = 0.25; // radial offset ramps from 0 over first 25% of wrap

// Gray "track" ribbons drawn full-length-minus-fill behind the colored
// fill, so the slider reads as a traditional fill-up-to-value control
// — but the track is itself moving ribbons, so it's cohesive with the
// fill rather than a different visual language.
const TRACK_COLOR = "#5a5a60";
// Multiplier on the canvas-wide alpha for the track pass — keeps the
// track subordinate while still letting it breathe with --bloom-amount.
const TRACK_ALPHA_MUL = 0.42;
// How long the gray track fans out from the convergence point as the
// along axis moves away from the halo. Mirrors HEAD_CONVERGE_START so
// the track's start lines up with where the fill's wrap began.
const TRACK_FAN_FRACTION = 0.15;

// Floors for ribbon length, defined in types/engine. The side floor
// is also consumed by DesktopEdgeDrag for the hint head position so
// the hint stays attached to the ribbon's visible end. The top floor
// is render-only — denoise=0 still passes through to the engine
// untouched; the sliver only ensures the user can find the slider
// after dragging it all the way left.

interface BarConfig {
  sel: string;
  horizontal: boolean;
  flipAlong: boolean;
  innerSign: 1 | -1;
  /** Which side of the canvas (in CSS layout terms) has the inward bleed
   * — i.e. extra canvas pixels past the host's content area into the
   * central gutter, so writhing curls aren't clipped. The CSS rules in
   * globals.css extend the canvas in the corresponding direction. */
  bleedSide: "bottom" | "left" | "right";
}

const BAR_CONFIG: BarConfig[] = [
  { sel: ".install-edge-top", horizontal: true, flipAlong: false, innerSign: 1, bleedSide: "bottom" },
  { sel: ".install-edge-left", horizontal: false, flipAlong: true, innerSign: 1, bleedSide: "right" },
  { sel: ".install-edge-right", horizontal: false, flipAlong: true, innerSign: -1, bleedSide: "left" },
];

export interface RibbonBar {
  edge: HTMLElement;
  canvas: HTMLCanvasElement;
  ctx: CanvasRenderingContext2D;
  resizeObs: ResizeObserver;
  horizontal: boolean;
  flipAlong: boolean;
  innerSign: 1 | -1;
  bleedSide: "bottom" | "left" | "right";
  w: number; // CSS pixels (canvas, including bleed)
  h: number;
  /** Cached --ribbon-bleed (CSS custom prop). Refreshed on resize so we
   * don't pay for getComputedStyle on every frame. */
  bleedPx: number;
}

function makeRibbonCanvas(): HTMLCanvasElement {
  const c = document.createElement("canvas");
  c.className = "install-ribbons";
  c.setAttribute("aria-hidden", "true");
  return c;
}

function attachResize(
  canvas: HTMLCanvasElement,
  ctx: CanvasRenderingContext2D,
  setSize: (w: number, h: number) => void,
  onResized?: () => void,
): ResizeObserver {
  const resize = () => {
    const dpr = window.devicePixelRatio || 1;
    const r = canvas.getBoundingClientRect();
    canvas.width = Math.max(1, Math.floor(r.width * dpr));
    canvas.height = Math.max(1, Math.floor(r.height * dpr));
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    setSize(r.width, r.height);
    onResized?.();
  };
  const obs = new ResizeObserver(resize);
  obs.observe(canvas);
  resize();
  return obs;
}

function readBleed(canvas: HTMLCanvasElement): number {
  // Bleed = how much the canvas overflows its host bar in the inward axis.
  // Computed from actual layout dimensions because CSS custom-property values
  // do not auto-resolve calc()/clamp() expressions in the computed-value
  // string returned by getComputedStyle().getPropertyValue() — without
  // CSS.registerProperty({syntax: "<length>"}), `--ribbon-bleed` reads back
  // as the literal "calc(...)" token and parseFloat returns NaN. Reading the
  // already-resolved bounding boxes side-steps that entirely.
  const host = canvas.parentElement;
  if (!host) return 0;
  const c = canvas.getBoundingClientRect();
  const h = host.getBoundingClientRect();
  return Math.max(c.width - h.width, c.height - h.height, 0);
}

export function initRibbons(): RibbonBar[] {
  const bars: RibbonBar[] = [];
  for (const cfg of BAR_CONFIG) {
    const edge = document.querySelector(cfg.sel) as HTMLElement | null;
    if (!edge) continue;

    // Drop the legacy 2 px bar; the canvas owns the meter now.
    const oldBar = edge.querySelector(".install-edge-bar");
    if (oldBar) oldBar.remove();
    // Drop any leftover SVG from a hot-reloaded prior shape.
    const oldSvg = edge.querySelector("svg.install-ribbons");
    if (oldSvg) oldSvg.remove();

    const canvas = makeRibbonCanvas();
    const ctx = canvas.getContext("2d");
    if (!ctx) continue;
    edge.appendChild(canvas);

    const bar: RibbonBar = {
      edge,
      canvas,
      ctx,
      resizeObs: null as unknown as ResizeObserver,
      horizontal: cfg.horizontal,
      flipAlong: cfg.flipAlong,
      innerSign: cfg.innerSign,
      bleedSide: cfg.bleedSide,
      w: 1,
      h: 1,
      bleedPx: 0,
    };
    bar.resizeObs = attachResize(
      canvas,
      ctx,
      (w, h) => {
        bar.w = w;
        bar.h = h;
      },
      () => {
        bar.bleedPx = readBleed(canvas);
      },
    );
    bars.push(bar);
  }
  return bars;
}

export function destroyRibbons(bars: RibbonBar[]): void {
  for (const bar of bars) {
    try {
      bar.resizeObs.disconnect();
    } catch {}
    try {
      bar.canvas.remove();
    } catch {}
  }
}

/** Canvas-pixel transform for a bar's viewBox. Shared by drawFillRibbon
 *  and drawTrackRibbon so both ribbons land in identical canvas space. */
function barTransform(bar: RibbonBar, bleedPx: number) {
  const alongSize = bar.horizontal ? bar.w : bar.h;
  const acrossSize = bar.horizontal ? bar.h : bar.w;
  const hostAcross = Math.max(1, acrossSize - bleedPx);
  const acrossPerUnit = hostAcross / ACROSS;
  const alongPerUnit = Math.max(1, alongSize - 2 * ALONG_END_INSET_PX) / ALONG;
  // The canvas is bleedPx larger than the host on its bleedSide; only
  // the "left" bleed (right-edge bar) needs an offset on the across
  // axis. See drawFillRibbon's preserved comment in earlier revisions
  // for the long version of why.
  const acrossOffset = bar.bleedSide === "left" ? bleedPx : 0;
  const alongOffset = ALONG_END_INSET_PX;
  return {
    sx: bar.horizontal ? alongPerUnit : acrossPerUnit,
    sy: bar.horizontal ? acrossPerUnit : alongPerUnit,
    acrossOffset,
    alongOffset,
  };
}

/** Map a (along, across) viewBox point to canvas-CSS-pixel coords. */
function viewBoxToCanvas(
  bar: RibbonBar,
  t: ReturnType<typeof barTransform>,
  along: number,
  across: number,
): { x: number; y: number } {
  if (bar.horizontal) {
    return { x: t.alongOffset + along * t.sx, y: across * t.sy };
  }
  const y = bar.flipAlong ? ALONG - along : along;
  return { x: t.acrossOffset + across * t.sx, y: t.alongOffset + y * t.sy };
}

/** Unit vector (in canvas-pixel space) the ribbon travels as `along`
 *  increases. The "outboard" direction the halo center sits relative
 *  to the convergence point. */
function travelDir(bar: RibbonBar): { dx: number; dy: number } {
  if (bar.horizontal) {
    return bar.flipAlong ? { dx: -1, dy: 0 } : { dx: 1, dy: 0 };
  }
  return bar.flipAlong ? { dx: 0, dy: -1 } : { dx: 0, dy: 1 };
}

/** Colored "fill" ribbon — 0..drawLen along the bar, converging
 *  laterally at the head, then continuing as a polar wrap around the
 *  halo center so the ribbon literally becomes the halo (no abrupt end,
 *  no separate halo-draw call). One ribbon = one stroked path. */
function drawFillRibbon(
  ctx: CanvasRenderingContext2D,
  progress: number,
  ribbonIdx: number,
  time: number,
  kick: number,
  bar: RibbonBar,
  bleedPx: number,
): void {
  // Both axes get a visibility floor so the ribbon never disappears at
  // strength=0 — otherwise the user has no cue the slider still exists.
  // The top (Remix) floor is smaller than the side floor because the
  // top bar is much wider; same proportional readability either way.
  const drawProgress = bar.horizontal
    ? Math.max(progress, REMIX_VISIBLE_FLOOR)
    : Math.max(progress, LORA_SIDE_VISIBLE_FLOOR);
  const drawLen = drawProgress * ALONG;
  const lateral = (ribbonIdx - (PALETTE.length - 1) / 2) * RIBBON_SPACING;
  const phase = ribbonIdx * 0.8;
  const writheAmp = NOISE_AMP_BASE + kick * NOISE_AMP_KICK;
  const center =
    bar.innerSign > 0 ? ACROSS - INWARD_DISTANCE : INWARD_DISTANCE;
  const tform = barTransform(bar, bleedPx);

  ctx.beginPath();

  // PHASE 1 — Writhe from along=0 → drawLen. Over the last
  // (1 - HEAD_CONVERGE_START) of the writhe, both `lateral` and the
  // writhe noise taper to 0, so all four ribbons land at exactly
  // (drawLen, center) — a single shared convergence point.
  for (let i = 0; i <= SEGMENTS; i++) {
    const t = i / SEGMENTS;
    const along = t * drawLen;
    const noise =
      Math.sin(along * 0.012 + time * 1.3 + phase) * 0.7 +
      Math.sin(along * 0.025 - time * 0.9 + phase * 1.4) * 0.3;
    const convergeT = t < HEAD_CONVERGE_START
      ? 0
      : (t - HEAD_CONVERGE_START) / (1 - HEAD_CONVERGE_START);
    const lateralFactor = 1 - convergeT;
    const writheFactor = 1 - convergeT; // → 0 at convergence: clean meeting point
    const across =
      center + lateral * lateralFactor + noise * writheAmp * writheFactor;
    const { x, y } = viewBoxToCanvas(bar, tform, along, across);
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  }

  // PHASE 2 — Polar wrap around the halo center. The halo sits
  // HEAD_HALO radius units outboard of the convergence point, so the
  // convergence lies on the halo's back perimeter; the ribbon
  // continues from there, fanning OUT to its concentric radial slot
  // over the first HEAD_HALO_FAN_FRACTION of the wrap. No moveTo —
  // same path as phase 1, so the visual transition has no seam.
  if (drawLen > 8) {
    const conv = viewBoxToCanvas(bar, tform, drawLen, center);
    const dir = travelDir(bar);
    const haloR = HEAD_HALO_BASE_R_PX + kick * HEAD_HALO_KICK_R_PX;
    const haloCx = conv.x + dir.dx * haloR;
    const haloCy = conv.y + dir.dy * haloR;
    // Angle from halo center pointing BACK at the convergence point.
    const entryAngle = Math.atan2(-dir.dy, -dir.dx);
    const radialSlot =
      (ribbonIdx - (PALETTE.length - 1) / 2) * HEAD_HALO_RADIAL_SPREAD_PX;
    const wrapWritheAmp =
      HEAD_HALO_NOISE_AMP_BASE_PX + kick * HEAD_HALO_NOISE_AMP_KICK_PX;
    const wrapPhase = ribbonIdx * 0.7;
    const tw = time * 1.3;
    for (let j = 1; j <= HEAD_HALO_WRAP_STEPS; j++) {
      const wrapT = j / HEAD_HALO_WRAP_STEPS;
      const theta = entryAngle + wrapT * HEAD_HALO_WRAP_SPAN;
      const fanT = Math.min(1, wrapT / HEAD_HALO_FAN_FRACTION);
      const noise =
        Math.sin(theta * 3 + tw + wrapPhase) * 0.7 +
        Math.sin(theta * 7 - tw * 0.7 + wrapPhase * 1.4) * 0.3;
      const r = haloR + radialSlot * fanT + noise * wrapWritheAmp * fanT;
      const x = haloCx + r * Math.cos(theta);
      const y = haloCy + r * Math.sin(theta);
      ctx.lineTo(x, y);
    }
  }
  ctx.stroke();
}

/** Gray "track" ribbon — drawLen..ALONG along the bar, mirroring the
 *  fill ribbon's writhe math so the unfilled portion looks like the
 *  same four ribbons, just dim. Starts at the convergence point
 *  (where the halo lives) and fans BACK OUT to the normal lateral
 *  spread over the first TRACK_FAN_FRACTION of the track. */
function drawTrackRibbon(
  ctx: CanvasRenderingContext2D,
  progress: number,
  ribbonIdx: number,
  time: number,
  kick: number,
  bar: RibbonBar,
  bleedPx: number,
): void {
  const drawProgress = bar.horizontal
    ? Math.max(progress, REMIX_VISIBLE_FLOOR)
    : Math.max(progress, LORA_SIDE_VISIBLE_FLOOR);
  const drawLen = drawProgress * ALONG;
  const trackLen = ALONG - drawLen;
  if (trackLen <= 1) return; // slider essentially maxed — no track to draw
  const lateral = (ribbonIdx - (PALETTE.length - 1) / 2) * RIBBON_SPACING;
  const phase = ribbonIdx * 0.8;
  const writheAmp = NOISE_AMP_BASE + kick * NOISE_AMP_KICK;
  const center =
    bar.innerSign > 0 ? ACROSS - INWARD_DISTANCE : INWARD_DISTANCE;
  const tform = barTransform(bar, bleedPx);
  // Density: one segment per ~4% of bar length, floored at 8 so even
  // a near-maxed slider still has enough segments for the fan-out
  // arc to look smooth.
  const segs = Math.max(8, Math.round(SEGMENTS * (trackLen / ALONG)));

  ctx.beginPath();
  for (let i = 0; i <= segs; i++) {
    const trackT = i / segs;
    const along = drawLen + trackT * trackLen;
    const noise =
      Math.sin(along * 0.012 + time * 1.3 + phase) * 0.7 +
      Math.sin(along * 0.025 - time * 0.9 + phase * 1.4) * 0.3;
    const fanT = trackT < TRACK_FAN_FRACTION
      ? trackT / TRACK_FAN_FRACTION
      : 1;
    const across = center + lateral * fanT + noise * writheAmp * fanT;
    const { x, y } = viewBoxToCanvas(bar, tform, along, across);
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  }
  ctx.stroke();
}

export function tickRibbons(
  bars: RibbonBar[],
  time: number,
  kick: number,
  bloom = 0,
): void {
  // CSS contract from .install-ribbons: stroke-width = 2px + bloom*4px,
  // opacity = 0.6 + bloom*0.45. Mirror exactly so canvas matches the SVG.
  // `bloom` is passed in (the binned kick the render loop also writes
  // into --bloom-amount) so we don't pay for a getComputedStyle flush.
  const lineWidthPx = 2 + bloom * 4;
  const alpha = Math.min(1, 0.6 + bloom * 0.45);

  for (const bar of bars) {
    if (bar.w <= 0 || bar.h <= 0) continue;
    const fill = parseFloat(bar.edge.style.getPropertyValue("--fill")) || 0;
    const ctx = bar.ctx;
    ctx.clearRect(0, 0, bar.w, bar.h);
    ctx.save();
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
    ctx.lineWidth = lineWidthPx;

    // PASS 1 — Gray track ribbons (unfilled portion of the slider).
    // Painted FIRST so the colored fill + halo wrap render on top.
    // Same four-ribbon writhe language as the fill, just dim; reads
    // as "this is the rail of the slider," cohesive with the active
    // portion.
    ctx.globalAlpha = alpha * TRACK_ALPHA_MUL;
    ctx.strokeStyle = TRACK_COLOR;
    for (let i = 0; i < PALETTE.length; i++) {
      drawTrackRibbon(ctx, fill, i, time, kick, bar, bar.bleedPx);
    }

    // PASS 2 — Colored fill ribbons (0..value), each ending in a
    // polar wrap around the halo center. The four wraps overlay each
    // other into a concentric multi-color ring — the slider's thumb
    // — but it's literally part of each ribbon's path, so the
    // morph from writhe → halo has no seam.
    ctx.globalAlpha = alpha;
    for (let i = 0; i < PALETTE.length; i++) {
      ctx.strokeStyle = PALETTE[i];
      drawFillRibbon(ctx, fill, i, time, kick, bar, bar.bleedPx);
    }
    ctx.restore();
  }
}

// ---------------------------------------------------------------------------
// Halo badge ribbons — same trig-noise writhe language as the linear bars
// but in polar coordinates, so the four ribbons trace the badge's circular
// border. Renders to a 2D canvas inside <HaloBadge />.
// ---------------------------------------------------------------------------

const HALO_SEGMENTS = 56;
const HALO_BASE_R = 46;
const HALO_RADIAL_SPREAD = 0.9;
const HALO_NOISE_AMP_BASE = 1.6;
const HALO_NOISE_AMP_KICK = 3.4;
const HALO_TIME_SCALE = 1.3;
const HALO_VIEWBOX = 100;

/** Stroke colors for the halo ribbons in the order paths are rendered.
 * Exported so HaloBadge / queue scenes can reuse without redefining. */
export const HALO_PALETTE = PALETTE;

export interface HaloRibbon {
  el: HTMLElement;
  canvas: HTMLCanvasElement;
  ctx: CanvasRenderingContext2D;
  resizeObs: ResizeObserver;
  w: number;
  h: number;
}

export function initHaloRibbon(host: HTMLElement): HaloRibbon | null {
  const canvas = host.querySelector(
    "canvas.halo-ribbons",
  ) as HTMLCanvasElement | null;
  if (!canvas) return null;
  const ctx = canvas.getContext("2d");
  if (!ctx) return null;
  const ring: HaloRibbon = {
    el: host,
    canvas,
    ctx,
    resizeObs: null as unknown as ResizeObserver,
    w: 1,
    h: 1,
  };
  ring.resizeObs = attachResize(canvas, ctx, (w, h) => {
    ring.w = w;
    ring.h = h;
  });
  return ring;
}

export function destroyHaloRibbon(ring: HaloRibbon): void {
  try {
    ring.resizeObs.disconnect();
  } catch {}
}

/**
 * Path-d builder shared by the canvas-driven HaloBadge tick and the
 * SVG-driven QueueScene tick. Returns a Path "d" string in halo viewBox
 * space (0..100), centered around (50, 50).
 */
function haloRingPathD(ribbonIdx: number, time: number, kick: number): string {
  const cx = 50;
  const cy = 50;
  const phase = ribbonIdx * 0.7;
  const radialOffset =
    (ribbonIdx - (PALETTE.length - 1) / 2) * HALO_RADIAL_SPREAD;
  const writheAmp = HALO_NOISE_AMP_BASE + kick * HALO_NOISE_AMP_KICK;
  const t = time * HALO_TIME_SCALE;
  let d = "";
  for (let i = 0; i <= HALO_SEGMENTS; i++) {
    const theta = (i / HALO_SEGMENTS) * Math.PI * 2;
    const noise =
      Math.sin(theta * 3 + t * 1.2 + phase) * 0.7 +
      Math.sin(theta * 7 - t * 0.9 + phase * 1.4) * 0.3;
    const r = HALO_BASE_R + radialOffset + noise * writheAmp;
    const x = cx + r * Math.cos(theta);
    const y = cy + r * Math.sin(theta);
    d += (i === 0 ? "M" : "L") + x.toFixed(2) + " " + y.toFixed(2) + " ";
  }
  d += "Z";
  return d;
}

/**
 * SVG-path variant — used by QueueScene which composes its own halo SVG
 * (rather than a dedicated `<canvas class="halo-ribbons">`). Cheaper-than-
 * canvas-migration: the queue scene is a low-traffic warmup screen, so
 * the per-frame setAttribute cost is acceptable. Each path is written
 * only when its "d" actually changes.
 */
export function tickHaloRibbonPaths(
  paths: SVGPathElement[],
  time: number,
  kick: number,
  lastD?: string[],
): void {
  for (let i = 0; i < paths.length; i++) {
    const d = haloRingPathD(i, time, kick);
    if (!lastD || lastD[i] !== d) {
      paths[i].setAttribute("d", d);
      if (lastD) lastD[i] = d;
    }
  }
}

export function tickHaloRibbon(
  ring: HaloRibbon,
  time: number,
  kick: number,
  bloom = 0,
): void {
  const w = ring.w;
  const h = ring.h;
  if (w <= 0 || h <= 0) return;
  const ctx = ring.ctx;
  ctx.clearRect(0, 0, w, h);

  // Halo viewBox is 100x100 with preserveAspectRatio xMidYMid meet — i.e.
  // uniform scale to fit, centered. Match that.
  const scale = Math.min(w, h) / HALO_VIEWBOX;
  const offsetX = (w - HALO_VIEWBOX * scale) / 2;
  const offsetY = (h - HALO_VIEWBOX * scale) / 2;

  // CSS contract: stroke-width = 1px + bloom*1.2px, opacity = 0.6 + bloom*0.3.
  // `bloom` comes from the render loop (same binned kick).
  const lineWidthPx = 1 + bloom * 1.2;
  const alpha = Math.min(1, 0.6 + bloom * 0.3);

  ctx.save();
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
  ctx.lineWidth = lineWidthPx;
  ctx.globalAlpha = alpha;

  const cx = HALO_VIEWBOX / 2;
  const cy = HALO_VIEWBOX / 2;
  const tScaled = time * HALO_TIME_SCALE;
  const writheAmp = HALO_NOISE_AMP_BASE + kick * HALO_NOISE_AMP_KICK;

  for (let r = 0; r < PALETTE.length; r++) {
    const phase = r * 0.7;
    const radialOffset =
      (r - (PALETTE.length - 1) / 2) * HALO_RADIAL_SPREAD;
    ctx.strokeStyle = PALETTE[r];
    ctx.beginPath();
    for (let i = 0; i <= HALO_SEGMENTS; i++) {
      const theta = (i / HALO_SEGMENTS) * Math.PI * 2;
      const noise =
        Math.sin(theta * 3 + tScaled * 1.2 + phase) * 0.7 +
        Math.sin(theta * 7 - tScaled * 0.9 + phase * 1.4) * 0.3;
      const radius = HALO_BASE_R + radialOffset + noise * writheAmp;
      const x = offsetX + (cx + radius * Math.cos(theta)) * scale;
      const y = offsetY + (cy + radius * Math.sin(theta)) * scale;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.closePath();
    ctx.stroke();
  }
  ctx.restore();
}

// ---------------------------------------------------------------------------
// Start-mark ribbons — the title-screen logo's writhing halo. Stays SVG
// because the launch sequence applies CSS transforms (rotate + scale) and
// SVG strokes don't widen with the transform thanks to non-scaling-stroke,
// which has no canvas equivalent without per-frame compensation.
// ---------------------------------------------------------------------------

const START_MARK_SEGMENTS = 72;
const START_MARK_BASE_R = 40;
const START_MARK_RADIAL_SPREAD = 2.4;
const START_MARK_NOISE_AMP = 5.5;
const START_MARK_TIME_SCALE = 0.55;

export interface StartMarkRibbon {
  el: HTMLElement;
  paths: SVGPathElement[];
  // Last-written `d` per path so we can skip redundant setAttribute calls
  // (still cheaper than full string rebuild but avoids SVG repaint).
  lastD: string[];
}

export function initStartMarkRibbon(host: HTMLElement): StartMarkRibbon | null {
  const svg = host.querySelector(".start-mark-ribbons");
  if (!svg) return null;
  const paths = Array.from(svg.querySelectorAll<SVGPathElement>("path"));
  if (paths.length === 0) return null;
  return { el: host, paths, lastD: paths.map(() => "") };
}

function startMarkRingPathD(ribbonIdx: number, time: number): string {
  const cx = 50;
  const cy = 50;
  const phase = ribbonIdx * 0.9;
  const radialOffset =
    (ribbonIdx - (PALETTE.length - 1) / 2) * START_MARK_RADIAL_SPREAD;
  const t = time * START_MARK_TIME_SCALE;

  let d = "";
  for (let i = 0; i <= START_MARK_SEGMENTS; i++) {
    const theta = (i / START_MARK_SEGMENTS) * Math.PI * 2;
    const noise =
      Math.sin(theta * 2 + t + phase) * 0.65 +
      Math.sin(theta * 5 - t * 1.3 + phase * 1.5) * 0.35;
    const r = START_MARK_BASE_R + radialOffset + noise * START_MARK_NOISE_AMP;
    const x = cx + r * Math.cos(theta);
    const y = cy + r * Math.sin(theta);
    d += (i === 0 ? "M" : "L") + x.toFixed(2) + " " + y.toFixed(2) + " ";
  }
  d += "Z";
  return d;
}

export function tickStartMarkRibbon(
  ring: StartMarkRibbon,
  time: number,
): void {
  // Skip work entirely when the host has been removed from the DOM
  // (start-cta unmounts after the user clicks play). Detached SVGs
  // wouldn't paint anyway, but the math still costs cycles.
  if (!ring.el.isConnected) return;
  for (let i = 0; i < ring.paths.length; i++) {
    const d = startMarkRingPathD(i, time);
    if (d !== ring.lastD[i]) {
      ring.paths[i].setAttribute("d", d);
      ring.lastD[i] = d;
    }
  }
}

/** Same color order as halo + bar ribbons; exported for StartOverlay JSX. */
export const START_MARK_PALETTE = PALETTE;
