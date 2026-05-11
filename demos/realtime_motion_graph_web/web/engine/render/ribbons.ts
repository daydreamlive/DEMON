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
// rgb triplets parsed once at module load — used to build per-frame
// linear-gradient strokeStyles that fade each ribbon's leading end to
// alpha 0. Kept in sync with PALETTE by index.
const PALETTE_RGB: ReadonlyArray<readonly [number, number, number]> = [
  [0x3d, 0xb6, 0xbe],
  [0xc7, 0xb5, 0x66],
  [0xf0, 0x8a, 0x48],
  [0xe8, 0x4f, 0x3d],
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

// Half-revolution curl at each ribbon's leading edge — same geometry
// as the pre-2026-05 version, restored after the per-ribbon coil and
// halo-wrap variants were tried and rejected.
const HEAD_CURL_STEPS = 8;
const HEAD_CURL_BASE_R = 7;
const HEAD_CURL_KICK_R = 3;

// Leading-end fade — mirrors the trailing-end's CSS mask fade but
// tracks the value position. Implemented via a createLinearGradient()
// strokeStyle so it's a single stroke per ribbon (no sub-stroke alpha
// juggling). Fade region: from FADE_START_T × drawLen (still full
// color) → drawLen + FADE_END_EXTRA_PX (transparent), which makes
// the visible ribbon look ~10% shorter than its drawLen + the curl
// dissolves cleanly past the fade.
const LEAD_FADE_START_T = 0.82;
const LEAD_FADE_END_EXTRA_PX = 18;

// Gray "track" ribbons drawn drawLen..ALONG behind the colored fill so
// the slider reads as a traditional fill-up-to-value control — but the
// track is itself moving ribbons, cohesive with the fill rather than a
// different visual language.
const TRACK_COLOR = "#5a5a60";
// Multiplier on the canvas-wide alpha for the track pass — keeps the
// track subordinate while still letting it breathe with --bloom-amount.
const TRACK_ALPHA_MUL = 0.42;

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

/** Colored "fill" ribbon — 0..drawLen along the bar, terminating in
 *  the original half-revolution curl. Stroked with a per-frame linear
 *  gradient strokeStyle so the leading end fades to alpha 0 — mirrors
 *  the trailing-end CSS mask fade but tracks the value position.
 *  Single beginPath / stroke per ribbon. */
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

  // ── Leading-end fade gradient ─────────────────────────────────────
  // Build a linear gradient along the bar's along axis. Points where
  // the path falls BEFORE the gradient's start get the first stop
  // (full color); points AFTER the end get the last stop (transparent).
  // So the writhe is full color from along=0 to ~82% of drawLen, then
  // fades to 0 over the next ~18% + the curl extent.
  const [cr, cg, cb] = PALETTE_RGB[ribbonIdx];
  let g0x = 0, g0y = 0, g1x = 0, g1y = 0;
  if (bar.horizontal) {
    g0x = tform.alongOffset + drawLen * LEAD_FADE_START_T * tform.sx;
    g1x = tform.alongOffset + drawLen * tform.sx + LEAD_FADE_END_EXTRA_PX;
  } else if (bar.flipAlong) {
    // Vertical bar, value end at TOP of canvas (small y).
    g0y = tform.alongOffset + (ALONG - drawLen * LEAD_FADE_START_T) * tform.sy;
    g1y = tform.alongOffset + (ALONG - drawLen) * tform.sy - LEAD_FADE_END_EXTRA_PX;
  } else {
    g0y = tform.alongOffset + drawLen * LEAD_FADE_START_T * tform.sy;
    g1y = tform.alongOffset + drawLen * tform.sy + LEAD_FADE_END_EXTRA_PX;
  }
  const grad = ctx.createLinearGradient(g0x, g0y, g1x, g1y);
  grad.addColorStop(0, `rgba(${cr},${cg},${cb},1)`);
  grad.addColorStop(1, `rgba(${cr},${cg},${cb},0)`);
  ctx.strokeStyle = grad;

  // ── Writhe (full lateral spread, no convergence — pre-2026-05 look) ──
  ctx.beginPath();
  let prevVbX = 0,
    prevVbY = 0,
    lastVbX = 0,
    lastVbY = 0;
  for (let i = 0; i <= SEGMENTS; i++) {
    const t = i / SEGMENTS;
    const along = t * drawLen;
    const noise =
      Math.sin(along * 0.012 + time * 1.3 + phase) * 0.7 +
      Math.sin(along * 0.025 - time * 0.9 + phase * 1.4) * 0.3;
    const across = center + lateral + noise * writheAmp;

    let vbX: number, vbY: number;
    if (bar.horizontal) {
      vbX = along;
      vbY = across;
    } else {
      vbX = across;
      vbY = bar.flipAlong ? ALONG - along : along;
    }
    const px = bar.horizontal
      ? tform.alongOffset + vbX * tform.sx
      : tform.acrossOffset + vbX * tform.sx;
    const py = bar.horizontal
      ? vbY * tform.sy
      : tform.alongOffset + vbY * tform.sy;
    if (i > 0) {
      prevVbX = lastVbX;
      prevVbY = lastVbY;
    }
    if (i === 0) ctx.moveTo(px, py);
    else ctx.lineTo(px, py);
    lastVbX = vbX;
    lastVbY = vbY;
  }

  // ── Half-revolution curl at the leading edge (original) ──
  // Tangent from the writhe's last segment defines the curl plane;
  // altSign / innerSign alternation gives adjacent ribbons opposite
  // curl handedness. The gradient strokeStyle fades the curl out as
  // it extends past drawLen, since the canvas position falls past
  // the gradient's last stop.
  if (drawLen > 8) {
    const dx = lastVbX - prevVbX;
    const dy = lastVbY - prevVbY;
    const segLen = Math.hypot(dx, dy) || 1;
    const ux = dx / segLen;
    const uy = dy / segLen;
    const altSign = ribbonIdx % 2 === 0 ? 1 : -1;
    const sign = altSign * (bar.innerSign > 0 ? 1 : -1);
    const pvx = -uy * sign;
    const pvy = ux * sign;
    const rCurl = HEAD_CURL_BASE_R + kick * HEAD_CURL_KICK_R;
    const cx = lastVbX + pvx * rCurl;
    const cy = lastVbY + pvy * rCurl;
    for (let j = 1; j <= HEAD_CURL_STEPS; j++) {
      const a = (j / HEAD_CURL_STEPS) * Math.PI;
      const sa = Math.sin(a);
      const ca = Math.cos(a);
      const rx = -pvx * ca + ux * sa;
      const ry = -pvy * ca + uy * sa;
      const tipAlong = bar.horizontal ? cx + rx * rCurl : cy + ry * rCurl;
      const tipAcross = bar.horizontal ? cy + ry * rCurl : cx + rx * rCurl;
      const tipX = bar.horizontal
        ? tform.alongOffset + tipAlong * tform.sx
        : tform.acrossOffset + tipAcross * tform.sx;
      const tipY = bar.horizontal
        ? tipAcross * tform.sy
        : tform.alongOffset + tipAlong * tform.sy;
      ctx.lineTo(tipX, tipY);
    }
  }
  ctx.stroke();
}

/** Gray "track" ribbon — drawLen..ALONG along the bar. Uses the
 *  identical writhe math (same noise function, same lateral spread,
 *  same phase) as drawFillRibbon, so where fill ends and track begins
 *  the two share the SAME point — no visual seam at the value
 *  boundary, no special fan-in/fan-out treatment needed. */
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
  // Density: roughly proportional to track length, floored at 8 so
  // a near-maxed slider still has enough segments for the writhe to
  // read smoothly past the curl region.
  const segs = Math.max(8, Math.round(SEGMENTS * (trackLen / ALONG)));

  ctx.beginPath();
  for (let i = 0; i <= segs; i++) {
    const trackT = i / segs;
    const along = drawLen + trackT * trackLen;
    const noise =
      Math.sin(along * 0.012 + time * 1.3 + phase) * 0.7 +
      Math.sin(along * 0.025 - time * 0.9 + phase * 1.4) * 0.3;
    const across = center + lateral + noise * writheAmp;
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

    // PASS 2 — Colored fill ribbons (0..value), each terminating in
    // the original half-revolution curl. Per-ribbon gradient
    // strokeStyle (set inside drawFillRibbon) fades the leading end
    // so the curl dissolves rather than ending abruptly.
    ctx.globalAlpha = alpha;
    for (let i = 0; i < PALETTE.length; i++) {
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
