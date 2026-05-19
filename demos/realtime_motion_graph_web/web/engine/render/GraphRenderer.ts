// Parameter-history graph display. Maintains a rolling buffer per signal
// and renders glowing polylines + a playhead.
//
// Beat energy reads through three channels: line width pulses with
// kick, the playhead does an additive "lighter" glow stroke on strong
// kicks, and the spark bursts (below) trail behind the playhead. The
// pre-2026 implementation used per-line ctx.shadowBlur strokes scaled
// by pulse — that defeats Skia's blur cache and was the dominant
// per-frame cost during music. Don't add shadowBlur back without
// reading PERFORMANCE.md first.
//
// Independently of pulse, each signal renders a small orbital dot at its
// playhead intersection (a colored disc + a white satellite on a slow
// orbit driven by `now`). Echoes the cursor's 4-particle constellation
// so the graph never reads as frozen between samples.

import { SLIDER_META, type SliderMeta } from "@/types/engine";

type RGB = [number, number, number];

const GRAPH_COLORS: Record<string, RGB> = {
  denoise: [61, 182, 190],
  feedback: [240, 138, 72],
  shift: [232, 79, 61],
  hint_strength: [199, 181, 102],
  seed: [240, 138, 72],
  prompt_blend: [220, 110, 220],
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
// Sampling runs at SAMPLE_INTERVAL_MS = 50 ms (see useRenderLoop), so 120
// samples = 6 s of history. The newest sample is plotted AT the playhead;
// the line extends leftward into the past and clips at x = 0. The area to
// the right of the playhead is intentionally empty — that's "future" the
// engine hasn't generated yet. Samples are drawn at the same horizontal
// density as before, so the visual scroll rate hasn't changed; only the
// anchor point moved from the right edge to the playhead.
const VISIBLE_SAMPLES = 120;
// Distance from the right edge to the playhead. The playhead is the
// anchor for new samples / dots / sparks — keeping it inset from the edge
// gives sparks somewhere to fly into and reads as "now is here, not at
// the boundary." Lower it (toward 0) to use more of the canvas for
// history; raise it for more breathing room on the right.
const PLAYHEAD_INSET_PX_FRAC = 1 / 6;
// Vertical breathing room so polylines at v=0 / v=1 (e.g. side-LoRA
// strengths pulled all the way) aren't clipped against the canvas edge.
// Sized for the max stroke width (5 px) + shadow blur (~3 px) + a little
// air.
const Y_PAD = 12;
// How far the line extends past the playhead before fading to alpha 0.
// Sized as a fraction of the empty "future" space to the right of the
// playhead (which itself is `w * PLAYHEAD_INSET_PX_FRAC` wide). Filling
// ~70% of that gap gives a trail substantial enough to read as
// continuation rather than a token nudge, while leaving 30% breathing
// margin so the fade lands before the canvas edge. Scales with viewport
// width — same visual ratio on ultrawide vs phone. Floored so the
// trail stays visible on very narrow canvases. Drawn before dots +
// playhead marker, so those still sit crisply on top.
const OVERSHOOT_FUTURE_FRAC = 0.7;
const OVERSHOOT_MIN_PX = 24;

interface History {
  buf: Float32Array;
  head: number;
  filled: number;
}

// Confetti sparks (cursor.ts vocabulary). Two firing layers:
//
// 1. Baseline — every BASELINE_INTERVAL_MS, ONE randomly-picked line
//    fires a small comet trail. Single trail in flight at a time;
//    different y each burst. Reads as a wandering motion across the
//    graph, with negative space between bursts so the eye can track
//    each one. Independent of audio.
//
// 2. Chorus — when an audio kick's peak strength exceeds CHORUS_THRESH
//    (a higher bar than just "kick is happening"), every line fires a
//    bigger burst simultaneously. Punctuates the music: most kicks
//    pass quietly, but the big ones light up the whole graph.
//
// All sparks fly leftward (toward the past, away from the playhead's
// "now"). Reads as a chromatic streak behind the playhead — sparks
// trail across the rendered line history in their line's color,
// reinforcing the "time is flowing past you" cue.
//
// Storage: struct-of-arrays in pre-allocated TypedArrays. Allocation
// (_allocSpark) prefers dead slots and protects pre-birth staggered
// chorus sparks. Zero per-frame allocation, zero array shift/splice,
// no per-spark fillStyle string. The pre-2026-perf-pass implementation
// used a Spark[] with shift()/splice()/push() and built an `rgba(...)`
// string per spark per frame; on chorus kicks (240 sparks alive) that
// was the dominant per-frame allocator and cause of beat-correlated
// jank. See PERFORMANCE.md for the full incident write-up.

// Spark physics. Disc size matches cursor.ts confetti (2px); trails
// are tuned long + flat so they extend visibly along the rendered
// line history rather than arcing down quickly.
const SPARK_GRAVITY = 0.06; // was 0.10 (cursor 0.16); even flatter for trails along the line
const SPARK_RADIUS = 2.5; // bumped from 2 for more visible streaks
const SPARK_MIN_SPEED = 4.5;
const SPARK_MAX_SPEED = 8.5;
const SPARK_LIFE_MS = 1800; // longer so trails reach further into the history and bouncy sparks have time to hop down through several lines
const SPARK_CONE_RAD = Math.PI / 5; // ~36° spread around the leftward axis

// Per-line cascade stagger on chorus moments. When chorus fires, each
// line's sparks get a small `birthAt` offset so they don't all spawn
// on the same frame — reads as a brief sweep across the cluster
// instead of one big synchronized cloud. Hash-based per line so the
// cascade order is stable per name.
const CHORUS_STAGGER_MAX_MS = 120;
const LEFT_ANGLE = Math.PI; // 180° — pure leftward, toward the past

// Baseline trigger — fires on the falling edge of small/medium kicks
// (peak in [BEAT_THRESH, CHORUS_THRESH)). Picks one random line per
// fire so the eye sees a wandering trail rather than constant rain.
// Rate-limited to BASELINE_MIN_INTERVAL_MS between fires; if music is
// silent for longer than BASELINE_MAX_INTERVAL_MS, fires anyway so
// the graph never goes fully still.
const BEAT_THRESH = 0.3;
const BASELINE_MIN_INTERVAL_MS = 250; // was 400; allows ~every-beat firing at 120 BPM
const BASELINE_MAX_INTERVAL_MS = 1200; // was 1500; silence fallback fires a little sooner
const BASELINE_BURST_SPARKS = 7; // was 4; denser single trail reads more clearly

// Chorus — when a kick's peak strength exceeds CHORUS_THRESH, every
// line fires a bigger burst simultaneously. Probabilistic so not
// every strong kick lights up the whole graph: most do, but enough
// don't that the chorus moment retains its surprise. A failed chorus
// roll falls through to a regular baseline fire (subject to its own
// rate-limit), so the kick still reads — it just gets a wandering
// single-line trail instead of the full-cluster blast.
const CHORUS_THRESH = 0.5;
// Probability that a strong-kick disarm fires the full multi-line burst
// (vs. falling through to a single-line baseline trail). Lower means
// chorus moments stay rarer / more special; higher means more crowded
// graph during dense passages. Tuned by feel.
const CHORUS_FIRE_PROB = 0.35;
const CHORUS_BURST_BASE = 6;
const CHORUS_BURST_PEAK = 6; // up to +6 more sparks per line scaled by peakPulse

// Bouncy spark trait. Each spawned spark has BOUNCE_PROB chance of
// being tagged bouncy. Two-phase bounce model:
//
//  Phase 1 — own line: the spark bounces TWICE on its spawn line.
//  Each bounce reflects velocity around the line's local normal at
//  the bounce point and damps the result by BOUNCE_DAMPING. Inclined
//  lines kick the spark off at the right angle.
//
//  Phase 2 — fall-through: once the spark has used its 2 own-line
//  bounces, it falls through and lands on each LOWER line below in
//  turn — exactly one bounce per line, tracked via a bitmask. Reads
//  as a stone skipping down a flight of stairs.
//
// Velocity is queried from the line's CURRENT geometry (not a captured
// spawn-time y), so a slider movement that re-shapes the line shows
// up correctly in the bounce direction. Lines with colorIdx >= 32
// are not tracked in the multi-line phase (would need a wider mask);
// that case only triggers if a single session uses >32 distinct line
// names, which the default fixture set doesn't approach.
const BOUNCE_PROB = 0.4;
// Initial upward velocity for bouncy sparks. Tuned so the round trip
// (rise + fall) fits comfortably in SPARK_LIFE_MS — at vy=-1.5 with
// gravity 0.06, peak is ~19 px above the line and round trip is
// ~830 ms, leaving ~400 ms after the first bounce for one or two
// smaller hops before the spark fades out.
const BOUNCE_INIT_VY = -1.5;
const BOUNCE_DAMPING = 0.7;
const BOUNCE_MIN_SPEED = 0.4;
// Max bounces on a spark's spawn line before it switches to fall-
// through mode. User-facing tuning knob: 2 reads as "skipping stone
// hops twice on a stair, then continues down."
const OWN_LINE_BOUNCE_LIMIT = 2;

// Pool size. Sized so a full chorus burst (~20 lines × up to 12 sparks
// = 240) plus baseline trails (~7 / 250 ms = ~36 alive over a 1.3 s
// lifetime) plus a second chorus event arriving inside the first one's
// lifetime all fit without forcing eviction of pre-birth sparks. The
// chorus stagger pushes some sparks' birthAt up to 120 ms into the
// future; if those slots get overwritten before they're born, the user
// never sees them — which is exactly the "no big burst on strong kick"
// regression we hit in perf pass #4. Pool sizing + a smarter allocator
// (see _allocSpark below) make that case impossible.
const MAX_SPARKS = 384;

// Per-line vertical "dodge" so signals with similar values at the
// playhead don't squish into a single visual blob during chorus. Hash
// of the line name picks a stable offset, deterministic per line and
// stable across frames. Dots dodge by a small amount (still close
// enough to read as "on the line"); spark origins dodge by a larger
// amount so trails fan out into distinct y-bands instead of stacking.
const DOT_DODGE_PX = 2;
const SPARK_DODGE_PX = 5;

export class GraphRenderer {
  readonly canvas: HTMLCanvasElement;
  private readonly ctx: CanvasRenderingContext2D;
  private readonly histories: Map<string, History> = new Map();
  private readonly _resizeObs: ResizeObserver;
  // Spark pool — SoA TypedArrays. Slot is alive when _spAlive[i] === 1.
  // Allocation goes through _allocSpark(now), which prefers free slots
  // and never evicts pre-birth (staggered chorus) sparks. See that
  // method for why this is non-trivial. The pre-2026-perf-pass
  // implementation used a Spark[] with shift()/splice()/push() — see
  // PERFORMANCE.md for the full incident write-up.
  private readonly _spX = new Float32Array(MAX_SPARKS);
  private readonly _spY = new Float32Array(MAX_SPARKS);
  private readonly _spVX = new Float32Array(MAX_SPARKS);
  private readonly _spVY = new Float32Array(MAX_SPARKS);
  private readonly _spAge = new Float32Array(MAX_SPARKS);
  private readonly _spLife = new Float32Array(MAX_SPARKS);
  private readonly _spBirth = new Float32Array(MAX_SPARKS);
  private readonly _spColor = new Uint16Array(MAX_SPARKS);
  private readonly _spAlive = new Uint8Array(MAX_SPARKS);
  // Bouncy flag — when set, the spark participates in the two-phase
  // bounce model (see BOUNCE_PROB notes). Set probabilistically at
  // spawn. Cleared when the spark exhausts its bouncing (either it's
  // hit BOUNCE_MIN_SPEED on rebound, or it's bounced once on every
  // line in the otherBouncedMask, or — pragmatically — its velocity
  // has decayed enough that further crossings produce no perceptible
  // hop).
  private readonly _spBouncy = new Uint8Array(MAX_SPARKS);
  // Phase-1 own-line bounce counter. Goes 0 → 1 → 2; at 2 the spark
  // graduates to phase-2 fall-through mode where bouncedMask governs.
  private readonly _spOwnBounces = new Uint8Array(MAX_SPARKS);
  // Phase-2 mask: one bit per colorIdx of "other" lines already
  // bounced on. Bit set ⇒ pass through next time we cross. Limited
  // to 32 colors per spark; lines with colorIdx >= 32 always pass
  // through (acceptable: see BOUNCE_PROB doc).
  private readonly _spOtherBouncedMask = new Uint32Array(MAX_SPARKS);
  // Hint for the allocator's free-slot search. Always start scanning
  // from here; advance past whatever we hand out. When the pool is
  // mostly empty, this gives O(1) allocation; when full, the scan
  // degrades gracefully (see _allocSpark).
  private _spAllocHint = 0;
  // Per-line color cache: name → index into _colorTable. The table holds
  // pre-built `rgb(r,g,b)` strings so the render loop sets fillStyle to a
  // string we already own, never allocating a new one per spark.
  private readonly _colorIdxByName: Map<string, number> = new Map();
  private readonly _colorTable: string[] = [];
  // Wall-clock millis at which the most recent baseline burst fired,
  // and the line picked to fire it. `_baselineLine` is consumed (set
  // to null) inside the per-line loop once that line actually fires,
  // so a single bucket only fires once even if a frame is missed.
  private _lastBaselineFireAt = 0;
  private _baselineLine: string | null = null;
  // Beat arming + peak tracking. Falling-edge dispatch decides whether
  // the just-ended kick was big enough for chorus or only triggers
  // baseline (or neither, if too soon since the last baseline).
  private _aboveBeat = false;
  private _peakPulse = 0;
  private _lastNow = 0;
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
    // Cap DPR at 2 — matches HUD + EffectsRenderer. On phones with DPR=3+
    // the extra pixels are imperceptible on this kind of plot but cost
    // ~2.25x in fragment work per frame.
    const dpr = Math.min(2, window.devicePixelRatio || 1);
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

  draw(pulse = 0, now: number = performance.now()): void {
    // Defense in depth: clamp must come BEFORE the Math.max/Math.min
    // because those don't catch NaN. A single non-finite pulse value
    // would otherwise propagate into baseAlpha and addColorStop, which
    // throws `SyntaxError: rgba(...,NaN)` and kills the render loop.
    if (!Number.isFinite(pulse)) pulse = 0;
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

    // Playhead is inset from the right edge by PLAYHEAD_INSET_PX_FRAC. New
    // samples spawn at this x; line history extends leftward.
    const playheadX = w * (1 - PLAYHEAD_INSET_PX_FRAC);
    // Trail length scales with the right-side gap so the visual ratio
    // holds across viewport widths. See OVERSHOOT_FUTURE_FRAC above.
    const overshootPx = Math.max(
      OVERSHOOT_MIN_PX,
      w * PLAYHEAD_INSET_PX_FRAC * OVERSHOOT_FUTURE_FRAC,
    );

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

    // One stroke per signal. The pre-2026-perf-pass implementation did a
    // second wide stroke with `shadowBlur = 1 + 1.5 * pulse` for a glow
    // halo whenever pulse > 0.1 — but per-stroke shadowBlur with a
    // beat-driven radius defeats Skia's compositor cache (every frame is
    // a fresh blur kernel) and was firing for every line on every frame
    // of music. Beat energy still reads through the playhead glow below
    // and the spark bursts; the per-line halo wasn't pulling its weight.
    for (const [name, hist] of this.histories) {
      const n = Math.min(hist.filled, VISIBLE_SAMPLES);
      if (n < 2) continue;
      const [r, g, b] = _colorFor(name);

      const pxPerSample = w / (VISIBLE_SAMPLES - 1);
      // Anchor newest sample at the playhead, not at the right edge. With
      // playheadX inset by ~w/6, the oldest few samples can land at x<0
      // and clip naturally against the canvas — same horizontal density,
      // just shifted left.
      const xStart = playheadX - (n - 1) * pxPerSample;
      ctx.beginPath();
      let lastY = 0;
      for (let i = 0; i < n; i++) {
        // Walk the ring backward from the newest sample (head - 1) so we
        // always plot the freshest n entries in chronological order.
        const bufIdx = (hist.head - n + i + HISTORY_LEN) % HISTORY_LEN;
        const v = hist.buf[bufIdx];
        const x = xStart + i * pxPerSample;
        const y = (h - Y_PAD) - v * (h - 2 * Y_PAD);
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
        lastY = y;
      }

      // Line widens slightly with pulse so beats still register on the
      // line itself, but no shadowBlur — pure crisp stroke.
      const baseAlpha = 0.85 + 0.15 * pulse;
      const lineWidth = 1 + 0.5 * pulse;
      ctx.strokeStyle = `rgba(${r},${g},${b},${baseAlpha})`;
      ctx.lineWidth = lineWidth;
      ctx.stroke();

      // Overshoot fade past the playhead. The polyline's newest point
      // sits at (playheadX, lastY); we extend overshootPx further right
      // at the same y, with a horizontal alpha gradient that ends at 0.
      // Same per-frame gradient pattern as the kick wash above. Drawn
      // before the per-line dots and the playhead marker so they stay
      // crisp on top.
      const grad = ctx.createLinearGradient(
        playheadX,
        0,
        playheadX + overshootPx,
        0,
      );
      grad.addColorStop(0, `rgba(${r},${g},${b},${baseAlpha})`);
      grad.addColorStop(1, `rgba(${r},${g},${b},0)`);
      ctx.strokeStyle = grad;
      ctx.beginPath();
      ctx.moveTo(playheadX, lastY);
      ctx.lineTo(playheadX + overshootPx, lastY);
      ctx.stroke();
    }

    // Per-line dot at the playhead + two-layer leftward confetti
    // trails shed from the dot. Sparks live in the SoA pool above
    // (_spX/_spY/...), capped at MAX_SPARKS, allocated via _allocSpark.
    //
    // Layer 1 (baseline): on the falling edge of small/medium kicks
    // (peakPulse in [BEAT_THRESH, CHORUS_THRESH)), pick ONE random
    // line and fire a small comet trail from it. Rate-limited so it
    // can fire at most every BASELINE_MIN_INTERVAL_MS — at the higher
    // end of the previous range, a deliberate wandering rather than
    // frantic. Falls back to a time-only fire every
    // BASELINE_MAX_INTERVAL_MS during silence so the graph never
    // freezes. Disabled when the curve editor overlay is open so
    // users editing curves aren't distracted.
    //
    // Layer 2 (chorus): on the falling edge of strong kicks (peak ≥
    // CHORUS_THRESH), every line fires a bigger burst at once. This
    // layer is NOT gated by the curve editor — big musical moments
    // still register even while editing.
    {
      const dt = this._lastNow ? Math.min(50, now - this._lastNow) : 16;
      this._lastNow = now;
      const dtScale = dt / 16;

      // Falling-edge peak detection over BEAT_THRESH. peakPulse on the
      // disarm frame tells us which layer (if any) to fire.
      let chorusFire = false;
      let chorusPeakStrength = 0;
      let baselineFire = false;
      if (pulse > BEAT_THRESH) {
        this._aboveBeat = true;
        if (pulse > this._peakPulse) this._peakPulse = pulse;
      } else if (this._aboveBeat) {
        const peak = this._peakPulse;
        // Chorus is probabilistic — even on strong kicks it only
        // fires CHORUS_FIRE_PROB of the time. A denied chorus roll
        // falls through to baseline so the kick still registers as
        // a wandering trail rather than disappearing entirely.
        if (peak >= CHORUS_THRESH && Math.random() < CHORUS_FIRE_PROB) {
          chorusFire = true;
          chorusPeakStrength = peak;
        } else if (
          now - this._lastBaselineFireAt >= BASELINE_MIN_INTERVAL_MS
        ) {
          baselineFire = true;
        }
        this._aboveBeat = false;
        this._peakPulse = 0;
      }
      // Silence fallback: if no beats have fired baseline for too
      // long, fire one anyway so the graph never goes fully still.
      if (
        !chorusFire &&
        !baselineFire &&
        now - this._lastBaselineFireAt >= BASELINE_MAX_INTERVAL_MS
      ) {
        baselineFire = true;
      }

      // Pick the baseline line at fire-time so the user sees a fresh
      // random pick on every burst.
      if (baselineFire && this.histories.size > 0) {
        const names = Array.from(this.histories.keys());
        this._baselineLine = names[Math.floor(Math.random() * names.length)];
        this._lastBaselineFireAt = now;
      }

      const chorusBurstCount = chorusFire
        ? CHORUS_BURST_BASE +
          Math.round(CHORUS_BURST_PEAK * chorusPeakStrength)
        : 0;

      ctx.save();
      ctx.globalCompositeOperation = "source-over";
      ctx.shadowBlur = 0;

      for (const [name, hist] of this.histories) {
        const n = Math.min(hist.filled, VISIBLE_SAMPLES);
        if (n < 2) continue;
        // Newest sample lives at the playhead, so the dot/spark anchor
        // value is just the most-recent entry in the ring buffer.
        const headIdx = (hist.head - 1 + HISTORY_LEN) % HISTORY_LEN;
        const v = hist.buf[headIdx];
        const yAtHead = h - Y_PAD - v * (h - 2 * Y_PAD);
        const [r, g, b] = _colorFor(name);

        // Hash → stable [-0.5, 0.5) per-line dodge factor. Reused for
        // both the dot and the spark spawn origin so a line's burst
        // always trails from a position related to where its dot sits.
        let hash = 0;
        for (let i = 0; i < name.length; i++) {
          hash = (hash * 31 + name.charCodeAt(i)) | 0;
        }
        const dodgeT = ((Math.abs(hash >> 7) % 1000) / 1000) - 0.5;
        const dotY = yAtHead + dodgeT * 2 * DOT_DODGE_PX;
        const sparkY = yAtHead + dodgeT * 2 * SPARK_DODGE_PX;

        // Disc anchored on the line at the playhead.
        ctx.fillStyle = `rgb(${r},${g},${b})`;
        ctx.beginPath();
        ctx.arc(playheadX, dotY, 3, 0, Math.PI * 2);
        ctx.fill();

        // Decide this line's burst size for this frame. Chorus fires
        // every line; baseline fires only the chosen line. Mutually
        // exclusive — chorus already fires the chosen line, so baseline
        // is suppressed during chorus.
        let burstCount = 0;
        let burstBirthAt = now;
        if (chorusFire) {
          burstCount = chorusBurstCount;
          // Hash → [0, CHORUS_STAGGER_MAX_MS) per-line cascade offset.
          // Reuse the same hash already computed for the y dodge;
          // different bit window so stagger and dodge aren't correlated
          // (otherwise the line that dodges most would also fire last,
          // which reads as a single tilted sweep).
          const staggerMs =
            (Math.abs(hash >> 17) % 1000) / 1000 * CHORUS_STAGGER_MAX_MS;
          burstBirthAt = now + staggerMs;
        } else if (name === this._baselineLine) {
          burstCount = BASELINE_BURST_SPARKS;
          this._baselineLine = null; // consumed
        }

        if (burstCount > 0) {
          // Resolve / cache this line's color-table index once outside
          // the inner spawn loop. _colorTable holds pre-built `rgb(...)`
          // strings; the render pass reuses them as fillStyle without
          // ever building a per-spark string.
          let colorIdx = this._colorIdxByName.get(name);
          if (colorIdx === undefined) {
            colorIdx = this._colorTable.length;
            this._colorTable.push(`rgb(${r},${g},${b})`);
            this._colorIdxByName.set(name, colorIdx);
          }
          for (let i = 0; i < burstCount; i++) {
            const sa =
              LEFT_ANGLE + (Math.random() - 0.5) * 2 * SPARK_CONE_RAD;
            const sp =
              SPARK_MIN_SPEED +
              Math.random() * (SPARK_MAX_SPEED - SPARK_MIN_SPEED);
            const slot = this._allocSpark(now);
            const bouncy = Math.random() < BOUNCE_PROB ? 1 : 0;
            this._spX[slot] = playheadX;
            this._spY[slot] = sparkY;
            this._spVX[slot] = Math.cos(sa) * sp;
            // Bouncy sparks need a controlled, modest upward velocity
            // at spawn so the rise + fall round trip fits inside
            // SPARK_LIFE_MS. Free-running sa would put half of them
            // moving DOWN at spawn (instant bounce, no visible skip)
            // and the upward half might fly too high to fall back
            // within the spark's lifetime. BOUNCE_INIT_VY = -1.5 puts
            // the peak ~19 px above the line at ~415 ms, the first
            // bounce ~830 ms in, leaving room for 1–2 smaller hops.
            this._spVY[slot] = bouncy
              ? BOUNCE_INIT_VY
              : Math.sin(sa) * sp;
            this._spAge[slot] = 0;
            this._spLife[slot] = SPARK_LIFE_MS - 150 + Math.random() * 300;
            this._spBirth[slot] = burstBirthAt;
            this._spColor[slot] = colorIdx;
            this._spAlive[slot] = 1;
            // Bouncy trait — see BOUNCE_PROB for the model. Two-phase
            // bookkeeping reset to zero so each spawn starts fresh.
            this._spBouncy[slot] = bouncy;
            this._spOwnBounces[slot] = 0;
            this._spOtherBouncedMask[slot] = 0;
          }
        }
      }

      // Sparks — physics + render in a single pool walk. Alpha is
      // applied via globalAlpha (one numeric assignment) instead of a
      // per-spark `rgba(...)` string allocation; fillStyle changes only
      // when the next alive spark belongs to a different line. Disc
      // shape uses arc()+fill() so dots read as round at any size; the
      // per-spark cost is negligible (~3x of fillRect, still sub-ms for
      // ~MAX_SPARKS sparks per frame on M-class hardware).
      const TAU = Math.PI * 2;
      // Hoisted out of the per-spark loop; reused per crossing scan.
      const pxPerSampleSpark = w / (VISIBLE_SAMPLES - 1);
      const yPad2 = h - 2 * Y_PAD;
      const hMinusYPad = h - Y_PAD;
      let lastColorIdx = -1;
      for (let i = 0; i < MAX_SPARKS; i++) {
        if (!this._spAlive[i]) continue;
        if (now < this._spBirth[i]) continue;
        const age = this._spAge[i] + dt;
        const life = this._spLife[i];
        if (age >= life) {
          this._spAlive[i] = 0;
          continue;
        }
        this._spAge[i] = age;
        let newVX = this._spVX[i];
        let newVY = this._spVY[i] + SPARK_GRAVITY * dtScale;
        const prevY = this._spY[i];
        let x = this._spX[i] + newVX * dtScale;
        let y = prevY + newVY * dtScale;

        // Two-phase bounce. While newVY > 0 (moving down) and the
        // spark is bouncy, scan all line histories for the FIRST line
        // crossing in (prevY, y]. Eligibility:
        //   - own line and ownBounces < OWN_LINE_BOUNCE_LIMIT, OR
        //   - other line and ownBounces == OWN_LINE_BOUNCE_LIMIT and
        //     this color's bit isn't set in otherBouncedMask.
        // On a successful bounce, reflect (newVX, newVY) around the
        // local line normal, damp by BOUNCE_DAMPING, clamp y to the
        // line's y, and update the bookkeeping. Lines further down
        // the canvas in the same frame are NOT bounced on — after the
        // reflection vy is upward, so the spark physically moves away
        // from them this frame.
        if (this._spBouncy[i] && newVY > 0) {
          const samplesFromHead = Math.round(
            (playheadX - x) / pxPerSampleSpark,
          );
          if (samplesFromHead >= 0) {
            const ownColor = this._spColor[i];
            const ownBounces = this._spOwnBounces[i];
            const otherMask = this._spOtherBouncedMask[i];
            let bestColorIdx = -1;
            let bestLineY = Infinity;
            let bestSlope = 0;

            for (const [name, hist] of this.histories) {
              if (samplesFromHead >= hist.filled) continue;
              const c = this._colorIdxByName.get(name);
              if (c === undefined) continue;

              const isOwn = c === ownColor;
              let eligible: boolean;
              if (isOwn) {
                eligible = ownBounces < OWN_LINE_BOUNCE_LIMIT;
              } else if (ownBounces >= OWN_LINE_BOUNCE_LIMIT && c < 32) {
                eligible = (otherMask & (1 << c)) === 0;
              } else {
                eligible = false;
              }
              if (!eligible) continue;

              // Line y at the spark's current x.
              const bufIdx =
                (hist.head - 1 - samplesFromHead + HISTORY_LEN) % HISTORY_LEN;
              const v = hist.buf[bufIdx];
              const lineY = hMinusYPad - v * yPad2;
              if (lineY <= prevY || lineY > y) continue;
              if (lineY >= bestLineY) continue;

              // Local slope: central difference between neighboring
              // samples. dy/dx = (y_next - y_prev) / (x_next - x_prev).
              // Newer sample = right (smaller samplesFromHead).
              // Clamp to ends of the buffer so edges don't blow up.
              const sNewer = Math.max(0, samplesFromHead - 1);
              const sOlder = Math.min(hist.filled - 1, samplesFromHead + 1);
              let slope = 0;
              if (sNewer !== sOlder) {
                const bufNewer =
                  (hist.head - 1 - sNewer + HISTORY_LEN) % HISTORY_LEN;
                const bufOlder =
                  (hist.head - 1 - sOlder + HISTORY_LEN) % HISTORY_LEN;
                const yNewer = hMinusYPad - hist.buf[bufNewer] * yPad2;
                const yOlder = hMinusYPad - hist.buf[bufOlder] * yPad2;
                // x_newer > x_older (newer sits closer to playhead, larger x).
                slope = (yNewer - yOlder) / ((sOlder - sNewer) * pxPerSampleSpark);
              }

              bestColorIdx = c;
              bestLineY = lineY;
              bestSlope = slope;
            }

            if (bestColorIdx >= 0) {
              // Reflect velocity around line normal. Tangent = (1, slope)
              // unit-normalised; normal = (-slope, 1) / sqrt(1 + slope²).
              // Reflected v = v - 2 (v · n) n; energy lost via uniform
              // damping factor on both components.
              const slope = bestSlope;
              const invNormMag = 1 / Math.sqrt(1 + slope * slope);
              const nx = -slope * invNormMag;
              const ny = invNormMag;
              const vDotN = newVX * nx + newVY * ny;
              newVX = (newVX - 2 * vDotN * nx) * BOUNCE_DAMPING;
              newVY = (newVY - 2 * vDotN * ny) * BOUNCE_DAMPING;
              y = bestLineY;

              // If the reflection kicked the spark rightward (toward
              // the playhead), cap its remaining lifetime so it fades
              // out before it reaches the playhead-as-wall. Avoids the
              // visual hiccup where a rightward-moving spark hits the
              // wall and stops dead. Using 85% of the time-to-impact
              // as the new lifetime gives the alpha decay enough room
              // to take the spark to ~zero before the wall would.
              if (newVX > 0 && x < playheadX) {
                const framesToWall = (playheadX - x) / newVX;
                const msToWall = framesToWall * 16; // approximate frame ms
                const cappedRemaining = msToWall * 0.85;
                if (cappedRemaining < life - age) {
                  this._spLife[i] = age + cappedRemaining;
                }
              }

              if (bestColorIdx === ownColor) {
                this._spOwnBounces[i] = ownBounces + 1;
              } else if (bestColorIdx < 32) {
                this._spOtherBouncedMask[i] = otherMask | (1 << bestColorIdx);
              }
              if (Math.abs(newVY) < BOUNCE_MIN_SPEED) {
                this._spBouncy[i] = 0;
              }

              // Dev instrumentation: count bounces under window in
              // local-test mode so a quick `window.__bounceCount` poll
              // verifies the bounce path is firing. Cheap (a single
              // typeof + numeric increment) and never set in prod
              // since __localTestPlayer is the local-test sentinel.
              if (
                typeof window !== "undefined" &&
                (window as { __localTestPlayer?: unknown })
                  .__localTestPlayer
              ) {
                const w = window as { __bounceCount?: number };
                w.__bounceCount = (w.__bounceCount ?? 0) + 1;
              }
            }
          }
        }

        // Playhead clamp — no spark may pass to the right of the
        // playhead. Treat it as a vertical wall: clamp x and force
        // vx leftward if the bounce reflection nudged it rightward.
        if (x > playheadX) {
          x = playheadX;
          if (newVX > 0) newVX = -newVX;
        }

        this._spVX[i] = newVX;
        this._spVY[i] = newVY;
        this._spX[i] = x;
        this._spY[i] = y;
        const f = age / life;
        const radius = SPARK_RADIUS * (1 - f * 0.7);
        if (radius <= 0.1) continue;
        const colorIdx = this._spColor[i];
        if (colorIdx !== lastColorIdx) {
          ctx.fillStyle = this._colorTable[colorIdx];
          lastColorIdx = colorIdx;
        }
        ctx.globalAlpha = 1 - f;
        ctx.beginPath();
        ctx.arc(x, y, radius, 0, TAU);
        ctx.fill();
      }
      ctx.globalAlpha = 1;

      ctx.restore();
    }

    // Playhead glow. The pre-2026-perf-pass version used
    // `shadowBlur = 4 * pulse` and fired whenever pulse > 0.05 —
    // i.e. essentially every frame of any non-silent music, with a
    // continuously varying blur radius (worst case for Skia's blur
    // cache). Replaced with a wider semi-transparent stroke under
    // "lighter" composite — visually similar at speed, no shadowBlur.
    // Gated at pulse > 0.2 so it only fires on meaningful kicks.
    if (pulse > 0.2) {
      ctx.save();
      ctx.globalCompositeOperation = "lighter";
      ctx.strokeStyle = `rgba(150, 180, 220, ${0.45 * pulse})`;
      ctx.lineWidth = 4 + 8 * pulse;
      ctx.beginPath();
      ctx.moveTo(playheadX + 0.5, 0);
      ctx.lineTo(playheadX + 0.5, h);
      ctx.stroke();
      ctx.restore();
    }


    ctx.strokeStyle = `rgba(255, 255, 255, ${0.6 + 0.4 * pulse})`;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(playheadX + 0.5, 0);
    ctx.lineTo(playheadX + 0.5, h);
    ctx.stroke();
  }

  /**
   * Allocate a slot for a new spark. Strategy:
   *   1. Scan forward from `_spAllocHint` for a dead slot — O(1) when
   *      the pool has any free space, which is the common case.
   *   2. If the pool is fully alive, evict the slot whose age/life
   *      ratio is highest (the spark closest to dying naturally).
   *      Skip pre-birth slots (now < birthAt) — those are staggered
   *      chorus sparks the user hasn't seen yet; overwriting them is
   *      the worst outcome.
   *   3. If even step 2 finds nothing (every slot is pre-birth — only
   *      possible at saturating spawn rates), fall back to the alloc
   *      hint and accept the visual loss.
   *
   * This replaced a naive ring-pointer allocator from the initial
   * pool rewrite that overwrote whatever was at the next slot — which
   * during a chorus burst meant the burst's own staggered late-firing
   * sparks were getting overwritten by subsequent baseline trails
   * before they could be born. Symptom: chorus bursts looked sparse
   * or missing entirely on strong kicks.
   */
  private _allocSpark(now: number): number {
    // Step 1: linear probe for a dead slot starting at the hint.
    for (let attempt = 0; attempt < MAX_SPARKS; attempt++) {
      const idx = this._spAllocHint;
      this._spAllocHint =
        this._spAllocHint + 1 >= MAX_SPARKS ? 0 : this._spAllocHint + 1;
      if (!this._spAlive[idx]) return idx;
    }
    // Step 2: pool fully alive. Pick the spark closest to natural death,
    // skipping pre-birth slots (they're "promised" to the user).
    let bestIdx = -1;
    let bestF = -1;
    for (let i = 0; i < MAX_SPARKS; i++) {
      if (now < this._spBirth[i]) continue;
      const f = this._spAge[i] / this._spLife[i];
      if (f > bestF) {
        bestF = f;
        bestIdx = i;
      }
    }
    if (bestIdx >= 0) return bestIdx;
    // Step 3: every slot is pre-birth. Vanishingly rare — would require
    // 384 staggered sparks all in the future, more than a chorus event
    // ever spawns. Accept the loss and overwrite at the hint.
    const idx = this._spAllocHint;
    this._spAllocHint =
      this._spAllocHint + 1 >= MAX_SPARKS ? 0 : this._spAllocHint + 1;
    return idx;
  }

  destroy(): void {
    this._resizeObs.disconnect();
  }
}
