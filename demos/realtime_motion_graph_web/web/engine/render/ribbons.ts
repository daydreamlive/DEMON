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

import { LORA_SIDE_VISIBLE_FLOOR } from "@/types/engine";

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

// Floor for the side-bar (LoRA) ribbon length. Defined in types/engine
// so DesktopEdgeDrag can floor the hint head against the same value —
// otherwise drags below this threshold leave the hint floating below
// the ribbon's visible end. Top bar (Remix Strength) is excluded —
// denoise=0 there is a meaningful "off" that should collapse the ribbon
// fully so the user can grade all the way to zero.

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

function drawRibbon(
  ctx: CanvasRenderingContext2D,
  progress: number,
  ribbonIdx: number,
  time: number,
  kick: number,
  bar: RibbonBar,
  bleedPx: number,
): void {
  // Side bars (LoRA strengths) get a visibility floor so the ribbon
  // never disappears even at strength=0. Top bar (Remix) is excluded —
  // see LORA_SIDE_VISIBLE_FLOOR comment.
  const drawProgress = bar.horizontal
    ? progress
    : Math.max(progress, LORA_SIDE_VISIBLE_FLOOR);
  const drawLen = drawProgress * ALONG;
  const lateral = (ribbonIdx - (PALETTE.length - 1) / 2) * RIBBON_SPACING;
  const phase = ribbonIdx * 0.8;
  const writheAmp = NOISE_AMP_BASE + kick * NOISE_AMP_KICK;
  const center =
    bar.innerSign > 0 ? ACROSS - INWARD_DISTANCE : INWARD_DISTANCE;

  // The canvas is bleedPx larger than the host on its bleedSide. Map the
  // viewBox so the ACROSS axis fills only the host content area; values
  // past 0..ACROSS land in the bleed pixels (which is exactly where the
  // writhing curls want to go).
  const alongSize = bar.horizontal ? bar.w : bar.h;
  const acrossSize = bar.horizontal ? bar.h : bar.w;
  const hostAcross = Math.max(1, acrossSize - bleedPx);
  const acrossPerUnit = hostAcross / ACROSS;
  // Where viewBox across=0 maps to in canvas pixels. For "top" / "left" /
  // "right", bleed lives on the inward side of the host, so across=0
  // (the outer side) is at the canvas edge — except the right bar has
  // its bleed on the left, so across=0 (bar's left = host's left edge,
  // which is in the central gutter, beyond the inner side's bleed) needs
  // to start at canvas_x = bleedPx, then increase toward the screen edge
  // (across=100 = bar's right edge).
  // ── wait: the right bar has innerSign=-1 (inner side at across=8), and
  //    we want across=0 at the host's left edge (which IS the inner side
  //    visually, in the central gutter). The CSS for .install-edge-right
  //    pins the host with right:0, so the bar's right edge in CSS is the
  //    screen's right edge (across=100 in viewBox), and the bar's left
  //    edge (across=0 in viewBox) is hud-thickness inward. The canvas
  //    extends LEFT of the bar by bleedPx — so canvas_x=0 is bleedPx
  //    leftward of the bar's left edge (deeper into the central gutter).
  //    Therefore across=0 → canvas_x = bleedPx. across=100 → canvas_x =
  //    bleedPx + hostAcross = bar.w. across=-something (writhe curling
  //    further inward) → canvas_x < bleedPx (into the bleed zone). ✓
  const acrossOffset = bar.bleedSide === "left" ? bleedPx : 0;
  // Same idea as acrossOffset but on the along axis: the writhe path's
  // first and last points (along=0 and along=ALONG) would sit flush with
  // the canvas bitmap edge, so the stroke half-width + drop-shadow halo
  // get sliced. Inset both ends by ALONG_END_INSET_PX to give them room.
  const hostAlong = Math.max(1, alongSize - 2 * ALONG_END_INSET_PX);
  const alongPerUnit = hostAlong / ALONG;
  const alongOffset = ALONG_END_INSET_PX;
  const sx = bar.horizontal ? alongPerUnit : acrossPerUnit;
  const sy = bar.horizontal ? acrossPerUnit : alongPerUnit;

  ctx.beginPath();
  let prevX = 0,
    prevY = 0,
    lastX = 0,
    lastY = 0;
  for (let i = 0; i <= SEGMENTS; i++) {
    const t = i / SEGMENTS;
    const along = t * drawLen;
    const noise =
      Math.sin(along * 0.012 + time * 1.3 + phase) * 0.7 +
      Math.sin(along * 0.025 - time * 0.9 + phase * 1.4) * 0.3;
    const across = center + lateral + noise * writheAmp;

    let x: number, y: number;
    if (bar.horizontal) {
      x = along;
      y = across;
    } else {
      x = across;
      y = bar.flipAlong ? ALONG - along : along;
    }
    // For horizontal bars, the across axis is y; for vertical bars, it's
    // x. Apply the bleed offset to whichever one is the across axis (only
    // matters when bleedSide === "left" → right-edge bar). The along
    // offset packs the writhe inside ALONG_END_INSET_PX of margin on
    // both ends so strokes don't get clipped at the canvas bitmap edge.
    const px = bar.horizontal ? alongOffset + x * sx : acrossOffset + x * sx;
    const py = bar.horizontal ? y * sy : alongOffset + y * sy;
    if (i > 0) {
      prevX = lastX;
      prevY = lastY;
    }
    if (i === 0) ctx.moveTo(px, py);
    else ctx.lineTo(px, py);
    lastX = x;
    lastY = y;
  }

  // Curling arrowhead at the leading edge (in viewBox space, then scaled).
  if (drawLen > 8) {
    const dx = lastX - prevX;
    const dy = lastY - prevY;
    const segLen = Math.hypot(dx, dy) || 1;
    const ux = dx / segLen;
    const uy = dy / segLen;
    const altSign = ribbonIdx % 2 === 0 ? 1 : -1;
    const sign = altSign * (bar.innerSign > 0 ? 1 : -1);
    const px = -uy * sign;
    const py = ux * sign;
    const r = 7 + kick * 3;
    const cx = lastX + px * r;
    const cy = lastY + py * r;
    const curlSteps = 8;
    for (let j = 1; j <= curlSteps; j++) {
      const a = (j / curlSteps) * Math.PI;
      const sa = Math.sin(a);
      const ca = Math.cos(a);
      const rx = -px * ca + ux * sa;
      const ry = -py * ca + uy * sa;
      // The arrowhead tip lives in viewBox space as (cx + rx*r, cy + ry*r).
      // For a horizontal bar this is (along, across); for vertical bars it's
      // (across, along). Match the same axis-aware offset as above.
      const tipAlong = bar.horizontal ? cx + rx * r : cy + ry * r;
      const tipAcross = bar.horizontal ? cy + ry * r : cx + rx * r;
      const tipX = bar.horizontal
        ? alongOffset + tipAlong * sx
        : acrossOffset + tipAcross * sx;
      const tipY = bar.horizontal
        ? tipAcross * sy
        : alongOffset + tipAlong * sy;
      ctx.lineTo(tipX, tipY);
    }
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
    ctx.globalAlpha = alpha;
    for (let i = 0; i < PALETTE.length; i++) {
      ctx.strokeStyle = PALETTE[i];
      drawRibbon(ctx, fill, i, time, kick, bar, bar.bleedPx);
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
