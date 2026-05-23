// DEMON title letters — same writhing-ribbon language as the start-mark
// halo (engine/render/ribbons.ts → tickStartMarkRibbon), but instead of
// tracing a closed circle each letter is a single contiguous parametric
// path. Four ribbons per letter, parallel-offset perpendicular to the
// letter's tangent, with the same two-sine writhe noise (same coefficients,
// same time scale, same palette).
//
// Letter geometry is a polyline of anchor points. Closed letters (D, O)
// wrap back to anchor 0; open letters (E variants, M, N) terminate at
// the last anchor. The constraint is "one continuous stroke per letter",
// so E is offered in three shapes: a backtrack-through-the-mid-bar
// version, a Greek epsilon (one sweep, no overlap), and a sigma zigzag.

const PALETTE = [
  "#3db6be",
  "#c7b566",
  "#f08a48",
  "#e84f3d",
];

const LETTER_SEGMENTS = 80;
// Match start-mark halo (engine/render/ribbons.ts → START_MARK_*) so the
// ribbon spacing + writhe amplitude on the letters reads as the same
// language as the play-circle, not a tamer cousin of it.
const LETTER_STROKE_SPREAD = 2.4;
const LETTER_NOISE_AMP = 5.5;
const LETTER_TIME_SCALE = 0.55;
// Audio-reactive writhe boost. The renderer feeds in the same per-frame
// kick value the halo / perimeter ribbons get (see HALO_NOISE_AMP_KICK
// in engine/render/ribbons.ts), and we add it to the baseline noise
// amplitude so transients punch the letters out of their resting writhe.
const LETTER_NOISE_AMP_KICK = 3.5;

// Per-letter style overrides. Read from `data-style` on each .demon-letter
// SVG so individual letters in a row can opt out of the writhe (solidly
// stacked), flatten to a single stroke, or shift onto a different axis
// of the noise space (pulse / chromatic). Anything not listed here uses
// the LETTER_* defaults above.
//
// Cross-cutting knobs that don't deserve their own style enum entry are
// passed in from the JSX as data-attributes:
//   data-phase  — extra time offset (seconds) added before the noise
//                 evaluation, used to phase-shift adjacent letters into
//                 a wave that ripples across the wordmark.
type LetterStyle = "default" | "static" | "pulse" | "wave";

interface StyleSpec {
  spread: number;
  noise: number;
  timeScale: number;
  /** When > 0, the noise amplitude is modulated by sin(time * pulseRate)
   * so the writhe breathes between min and max instead of running at a
   * constant amp. The whole word (or whichever letters share this
   * style) inhales and exhales in lockstep. */
  pulseRate: number;
  /** When > 0, the noise amplitude is modulated like pulseRate, but
   * the modulation is phased per-letter via phaseOffset so a peak of
   * writhe intensity travels through the row. Distinct axis from
   * pulse: pulse is synchronous, wave is sequential. */
  waveRate: number;
}

const STYLE_PARAMS: Record<LetterStyle, StyleSpec> = {
  default: {
    spread: LETTER_STROKE_SPREAD,
    noise: LETTER_NOISE_AMP,
    timeScale: LETTER_TIME_SCALE,
    pulseRate: 0,
    waveRate: 0,
  },
  // Same 4-ribbon perpendicular spread as default, but noise=0 freezes
  // the writhe so the colored layers are stationary.
  static: {
    spread: LETTER_STROKE_SPREAD,
    noise: 0,
    timeScale: LETTER_TIME_SCALE,
    pulseRate: 0,
    waveRate: 0,
  },
  // Pulse: same writhe geometry as default, but the noise amplitude
  // breathes between ~20% and ~100% on a ~4s cycle (the whole word
  // inhales and exhales together).
  pulse: {
    spread: LETTER_STROKE_SPREAD,
    noise: LETTER_NOISE_AMP,
    timeScale: LETTER_TIME_SCALE,
    pulseRate: 1.5,
    waveRate: 0,
  },
  // Wave: noise amplitude is a gaussian envelope around a peak position
  // that ping-pongs between the two ends of the row at waveRate. Each
  // letter's phaseOffset is its position-index in the row; the renderer
  // computes |index - peakPos| and falls off via exp(-(d/width)²). One
  // localized wave-packet of writhe bounces back and forth across DEMON.
  wave: {
    spread: LETTER_STROKE_SPREAD,
    noise: LETTER_NOISE_AMP,
    timeScale: LETTER_TIME_SCALE,
    pulseRate: 0,
    waveRate: 0.7,
  },
};

const STATIC_STYLES: ReadonlySet<LetterStyle> = new Set(["static"]);

// Hard-coded for the current 5-letter wordmark so the wave style knows
// where its bounce endpoints are. If we ever render a different number
// of letters in a wave row, plumb the actual count in.
const WAVE_ROW_LAST_INDEX = 4;
// Gaussian width (in letter-index units) of the writhe peak. Smaller =
// tighter focus (more letters fully still); larger = broader hump.
const WAVE_PEAK_WIDTH = 1.5;

type Pt = readonly [number, number];

interface LetterDef {
  pts: readonly Pt[];
  closed: boolean;
}

// All letters share viewBox 0..100 × 0..100. Anchor coords are in viewBox
// space; final SVG paths render in the same space.

const D_DEF: LetterDef = {
  // Down the spine, across the bottom flat, around the bow (5 anchors so
  // the curve reads as a curve), across the top flat, close to start.
  // Bbox is x ∈ [12.5, 87.5] so the glyph is centered at viewBox x=50,
  // matching E/M/O/N — without this offset the D's bow extending to x=90
  // would pull the whole wordmark visually right by ~0.7px.
  pts: [
    [12.5, 10],
    [12.5, 90],
    [47.5, 90],
    [75.8, 78.3],
    [87.5, 50],
    [75.8, 21.7],
    [47.5, 10],
  ],
  closed: true,
};

const E_DEF: LetterDef = {
  // Sigma E — backwards-Z drawn as one sweep: top bar, diagonal down
  // to mid-right, diagonal down to bottom-left, bottom bar. No
  // retrace, no overlap. (Other E variants — backtrack, epsilon —
  // existed earlier as comparison rows but were ruled out.)
  pts: [
    [85, 10],
    [15, 10],
    [70, 50],
    [15, 90],
    [85, 90],
  ],
  closed: false,
};

const M_DEF: LetterDef = {
  pts: [
    [10, 90],
    [10, 10],
    [50, 55],
    [90, 10],
    [90, 90],
  ],
  closed: false,
};

const O_DEF: LetterDef = (() => {
  const pts: Pt[] = [];
  const cx = 50;
  const cy = 50;
  const rx = 38;
  const ry = 42;
  const N = 24;
  for (let i = 0; i < N; i++) {
    const a = (i / N) * Math.PI * 2 - Math.PI / 2;
    pts.push([cx + rx * Math.cos(a), cy + ry * Math.sin(a)]);
  }
  return { pts, closed: true };
})();

const N_DEF: LetterDef = {
  pts: [
    [10, 90],
    [10, 10],
    [90, 90],
    [90, 10],
  ],
  closed: false,
};

const LETTER_DEFS: Record<string, LetterDef> = {
  D: D_DEF,
  E: E_DEF,
  M: M_DEF,
  O: O_DEF,
  N: N_DEF,
};

function letterPathD(
  letter: LetterDef,
  ribbonIdx: number,
  time: number,
  style: LetterStyle,
  ribbonCount: number,
  phaseOffset: number,
  kick: number,
): string {
  const {
    spread,
    noise: baseNoise,
    timeScale,
    pulseRate,
    waveRate,
  } = STYLE_PARAMS[style];
  // Pulse + wave compose multiplicatively on the noise amplitude.
  //   pulseRate > 0 → whole word breathes in lockstep, [0.2 .. 1.0] of base.
  //   waveRate  > 0 → wave-packet of writhe ping-pongs across DEMON: the
  //                   peak position bounces between index 0 and the last
  //                   letter sinusoidally, and each letter's amp is a
  //                   gaussian envelope around its distance from that
  //                   peak. The trough letters sit at ~0.1 (nearly still
  //                   colored offsets) and the peak letter writhes at
  //                   ~1.4× base.
  let ampMul = 1;
  if (pulseRate > 0) {
    ampMul *= 0.6 + 0.4 * Math.sin(time * pulseRate);
  }
  if (waveRate > 0) {
    const peakPos =
      (WAVE_ROW_LAST_INDEX / 2) * (1 + Math.sin(time * waveRate));
    const dist = phaseOffset - peakPos;
    const env = Math.exp(-(dist * dist) / (WAVE_PEAK_WIDTH * WAVE_PEAK_WIDTH));
    ampMul *= 0.1 + 1.3 * env;
  }
  // Kick raises the resting writhe amplitude before the pulse / wave
  // multiplier scales it, so audio transients punch through even when
  // a letter is sitting in the trough of a pulse/wave envelope.
  const noiseAmp = (baseNoise + kick * LETTER_NOISE_AMP_KICK) * ampMul;
  // Writhe-shape phase still tracks phaseOffset for the wave style so
  // adjacent letters don't show identical noise patterns even when their
  // amps happen to match. For non-wave styles phaseOffset is 0, so
  // localTime collapses to plain time.
  const localTime = time + phaseOffset;
  const pts = letter.pts;
  const closed = letter.closed;
  const n = pts.length;
  const segmentCount = closed ? n : n - 1;
  if (segmentCount < 1) return "";

  // Cumulative arc length per segment.
  const lens: number[] = new Array(segmentCount);
  let total = 0;
  for (let i = 0; i < segmentCount; i++) {
    const j = (i + 1) % n;
    const dx = pts[j][0] - pts[i][0];
    const dy = pts[j][1] - pts[i][1];
    const len = Math.hypot(dx, dy);
    lens[i] = len;
    total += len;
  }
  if (total < 1e-6) return "";

  const sampleCount = LETTER_SEGMENTS + 1;
  const sx = new Array<number>(sampleCount);
  const sy = new Array<number>(sampleCount);
  for (let s = 0; s < sampleCount; s++) {
    const u = s / LETTER_SEGMENTS;
    const target = u * total;
    let walked = 0;
    let i = 0;
    while (i < segmentCount - 1 && target > walked + lens[i]) {
      walked += lens[i];
      i++;
    }
    const localT = (target - walked) / Math.max(1e-9, lens[i]);
    const j = (i + 1) % n;
    sx[s] = pts[i][0] + (pts[j][0] - pts[i][0]) * localT;
    sy[s] = pts[i][1] + (pts[j][1] - pts[i][1]) * localT;
  }

  // Tangent via centered finite difference; falls back to one-sided
  // diff when centered collapses to ~0 (i.e. a 180° retrace tip such
  // as the mid-bar return in the backtrack E). Without that fallback
  // the perpendicular at the tip would be undefined and the four
  // ribbons would visibly pinch together.
  const tx = new Array<number>(sampleCount);
  const ty = new Array<number>(sampleCount);
  for (let s = 0; s < sampleCount; s++) {
    let prev: number;
    let next: number;
    if (closed) {
      prev = (s - 1 + LETTER_SEGMENTS) % LETTER_SEGMENTS;
      next = (s + 1) % LETTER_SEGMENTS;
    } else {
      prev = Math.max(0, s - 1);
      next = Math.min(sampleCount - 1, s + 1);
    }
    const dx = sx[next] - sx[prev];
    const dy = sy[next] - sy[prev];
    const mag = Math.hypot(dx, dy);
    if (mag > 1e-3) {
      tx[s] = dx / mag;
      ty[s] = dy / mag;
    } else {
      const inboundIdx = closed
        ? (s - 1 + LETTER_SEGMENTS) % LETTER_SEGMENTS
        : Math.max(0, s - 1);
      const ix = sx[s] - sx[inboundIdx];
      const iy = sy[s] - sy[inboundIdx];
      const m2 = Math.hypot(ix, iy) || 1;
      tx[s] = ix / m2;
      ty[s] = iy / m2;
    }
  }

  // ribbonCount comes from the actual number of <path> elements rendered
  // in the JSX (not PALETTE.length), so single-stroke variants like mono
  // and accent center their one path on the polyline instead of being
  // pushed off-axis by half a default-spread.
  const lateralOffset =
    (ribbonIdx - (ribbonCount - 1) / 2) * spread;
  const phase = ribbonIdx * 0.9;
  const t = localTime * timeScale;

  let d = "";
  for (let s = 0; s < sampleCount; s++) {
    const u = s / LETTER_SEGMENTS;
    const phaseParam = u * Math.PI * 2;
    const noise =
      Math.sin(phaseParam * 2 + t + phase) * 0.65 +
      Math.sin(phaseParam * 5 - t * 1.3 + phase * 1.5) * 0.35;
    const offset = lateralOffset + noise * noiseAmp;
    const nx = -ty[s];
    const ny = tx[s];
    const px = sx[s] + nx * offset;
    const py = sy[s] + ny * offset;
    d += (s === 0 ? "M" : "L") + px.toFixed(2) + " " + py.toFixed(2) + " ";
  }
  if (closed) d += "Z";
  return d;
}

export const DEMON_LETTER_PALETTE = PALETTE;

export interface DemonLetters {
  el: HTMLElement;
  groups: Array<{
    def: LetterDef;
    paths: SVGPathElement[];
    lastD: string[];
    style: LetterStyle;
    /** Extra time offset, in seconds, used as both the writhe-phase
     * shift on the noise and (for the wave style) the per-letter phase
     * of the amp modulation. Set per letter from phasePerLetter. */
    phaseOffset: number;
    /** Set once for static/mono letters (their path is constant) so we
     * don't recompute per frame on top of the lastD setAttribute skip. */
    rendered: boolean;
  }>;
}

function styleFromDataset(value: string | undefined): LetterStyle {
  switch (value) {
    case "static":
    case "pulse":
    case "wave":
      return value;
    default:
      return "default";
  }
}

export function initDemonLetters(host: HTMLElement): DemonLetters | null {
  const letterEls = Array.from(
    host.querySelectorAll<SVGElement>(".demon-letter"),
  );
  if (letterEls.length === 0) return null;
  const groups: DemonLetters["groups"] = [];
  for (const svg of letterEls) {
    const key = svg.dataset.letter || "";
    const def = LETTER_DEFS[key];
    if (!def) continue;
    const paths = Array.from(svg.querySelectorAll<SVGPathElement>("path"));
    if (paths.length === 0) continue;
    const style = styleFromDataset(svg.dataset.style);
    const phaseOffset = parseFloat(svg.dataset.phase || "0") || 0;
    groups.push({
      def,
      paths,
      lastD: paths.map(() => ""),
      style,
      phaseOffset,
      rendered: false,
    });
  }
  if (groups.length === 0) return null;
  return { el: host, groups };
}

export function tickDemonLetters(
  letters: DemonLetters,
  time: number,
  kick = 0,
): void {
  if (!letters.el.isConnected) return;
  // Static letters re-render whenever the kick has moved them off the
  // resting path (otherwise they'd freeze the first frame and ignore
  // audio). Once kick returns to ~0 we let them lock to the cached path
  // again, same as before the audio plumbing landed.
  const audioActive = kick > 1e-3;
  for (const g of letters.groups) {
    const staticLocked =
      STATIC_STYLES.has(g.style) && g.rendered && !audioActive;
    if (staticLocked) continue;
    for (let i = 0; i < g.paths.length; i++) {
      const d = letterPathD(
        g.def,
        i,
        time,
        g.style,
        g.paths.length,
        g.phaseOffset,
        kick,
      );
      if (d !== g.lastD[i]) {
        g.paths[i].setAttribute("d", d);
        g.lastD[i] = d;
      }
    }
    if (STATIC_STYLES.has(g.style) && !audioActive) g.rendered = true;
  }
}
