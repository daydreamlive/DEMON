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

// ---------------------------------------------------------------
// Layout mode + fireworks flags (Concept A prototype — May 2026).
//
// The center viz is being rethought to read like a DAW (Ableton's
// clip-view automation lanes). Two flags below switch between the
// historical polyline graph and the new lanes layout, and gate the
// spark/burst "fireworks" system independent of which layout is
// active. Flip them here to A/B without removing code.
//
// LAYOUT_MODE = "lanes"     → one horizontal lane per active param,
//                             filled envelope from baseline → value,
//                             brand-gradient fill, polyline contour.
// LAYOUT_MODE = "polylines" → original overlapping polylines + dots.
//
// FIREWORKS_ENABLED = false → no spark allocations at all (automation
//                             queue is still populated but never
//                             consumed for spawn; chorus path skipped).
// ---------------------------------------------------------------
const LAYOUT_MODE: "lanes" | "polylines" = "lanes";
const FIREWORKS_ENABLED = false;

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
// Exported so the DOM lane-labels overlay (GraphLaneLabels) can place
// its pills at the same x without duplicating the constant.
export const PLAYHEAD_INSET_PX_FRAC = 1 / 6;
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

// Beat arming threshold — used solely to peak-detect the falling edge
// of a kick (chorus dispatch below). The prior "baseline" layer that
// picked one random line per small/medium kick was removed when sparks
// became per-lane (automation-driven): the user now reads sparks as
// "this knob is being moved right now," and random-line fires
// contradicted that signal.
const BEAT_THRESH = 0.3;

// Per-lane automation trigger. When sample() sees a value change at
// least AUTOMATION_DELTA_THRESH on a given line, that line earns a
// small spark burst at its playhead intersection — read as "this
// lane is being automated right now." Rate-limited to
// AUTOMATION_MIN_INTERVAL_MS per line so a fast drag doesn't drown
// the pool. Sized small (3 sparks) because they fire frequently
// during play; chorus moments still produce the big multi-line
// blast via the kick-driven path.
const AUTOMATION_DELTA_THRESH = 0.005;
const AUTOMATION_MIN_INTERVAL_MS = 80;
const AUTOMATION_BURST_SPARKS = 3;

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

// ---------------------------------------------------------------
// Lanes mode constants (Concept A prototype).
//
// Lane layout maths: usable height = canvas height minus top/bottom
// padding (top reserves room for the "NOW" pill above the playhead).
// Each lane is sized to fit the active lane count, clamped between
// LANE_MIN_H and LANE_MAX_H. Inter-lane padding gives lanes their
// "channel" feel without becoming a tight rack of bars.
// ---------------------------------------------------------------
// MAX_LANES is the HARD CEILING used to pre-allocate the scratch
// buffers (_laneNameBuf, _laneLastFireBuf). It is INTENTIONALLY
// generous (32) so the per-frame `effectiveMaxLanes` calculation —
// not this buffer — is what governs how many lanes show on screen.
// On a 1440p+ monitor with the drawer closed, the canvas can fit
// well over 20 lanes; capping the buffer at 20 would silently stop
// the resize-reactive lane count from scaling up. 32 slots × (one
// string + one float) ≈ 280 bytes; negligible. Bumping this further
// only matters if a single session ever surfaces > 32 simultaneously-
// touched parameters, which the engine doesn't approach today.
const MAX_LANES = 32;
// Target height per lane when computing how many fit on this canvas.
// `effectiveMaxLanes` is recomputed EVERY frame from the live canvas
// height (`this.h`, which the ResizeObserver in `_resize` keeps in
// sync with the canvas's CSS box). That means lane count adapts on:
//   - window resize / drawer open-close (canvas box changes)
//   - moving the window between monitors (CSS box re-flows)
//   - browser zoom changes
//   - mobile portrait↔landscape rotation
// No additional listeners needed — the per-frame `_drawLanes` already
// reads the freshest `this.h`. Lanes individually clamp to
// [LANE_MIN_H, LANE_MAX_H], so on tall canvases with few active params
// the stack is vertically centered rather than stretched.
const LANE_TARGET_H = 30;
const LANE_HARD_MIN_COUNT = 8;
// Reserve at the bottom of the canvas to keep the lane stack clear of
// the <HeroMacros/> floating panel. Hero macros are `position: fixed`
// at the bottom of the viewport with z-index 5 — they paint OVER the
// graph canvas (which is z-index 2) without affecting graph-wrap's
// inset, so the lane stack would otherwise extend into their region.
//
// Sizing: we want the visible bottom gap (stack bottom → macros panel
// top) to equal the visible top gap (canvas top → stack top, which
// equals LANE_TOP_PAD when uncentered). The macros panel sticks UP
// into the canvas by ≈ `delta_macros` pixels, where:
//
//     delta_macros = macros_height + 24 − hud_thickness − ribbon_bleed
//
// For typical desktop (macros_height ≈ 100 px, hud_thickness ≈ 56–90 px
// via `clamp(56px, 6vmin, 90px)`, ribbon_bleed ≈ 2 px), `delta_macros`
// lands in the 30–70 px range, ≈ 55 px on a mid-size viewport.
//
// To make the visible top and bottom gaps symmetric, this reserve
// needs to be (LANE_TOP_PAD − LANE_BOTTOM_PAD + delta_macros) ≈ 64 px.
// A fixed constant is "near right" across viewport sizes — the
// remaining 5–15 px asymmetry on extreme viewports is acceptable for
// the prototype. If precise symmetry matters, switch to measuring
// `.hero-macros` getBoundingClientRect on resize.
//
// Hero macros are hidden under 768 px viewport width (mobile gets
// <LiteControls/> instead, and graph-wrap's media-query inset already
// accounts for the LiteControls bay). So this reserve only applies on
// desktop — read via matchMedia in _drawLanes.
const LANE_HERO_MACROS_RESERVE = 64;
const LANE_TOP_PAD = 18;
const LANE_BOTTOM_PAD = 10;
const LANE_INTER_PAD = 3;
const LANE_MIN_H = 18;
const LANE_MAX_H = 38;
// A lane is "idle" if its value hasn't changed in this many ms. Idle
// lanes draw at LANE_IDLE_ALPHA (their channel still reads, but doesn't
// compete with the lane the user is touching).
const LANE_IDLE_MS = 2500;
const LANE_IDLE_ALPHA = 0.4;
// Left "track header" gutter — Ableton-style strip on the canvas left
// that holds the lane labels. Dark glass backing makes each row read as
// a discrete TRACK with a header, not just a polyline floating across
// the screen. Width sized so the longest current label ("FEEDBACK
// DEPTH" at 14 mono-caps chars + tracking + pill padding) fits inside
// the gutter without overflowing into the envelope body. Pills inside
// the gutter come from <GraphLaneLabels/>.
const LANE_GUTTER_W = 168;
// Pre-built fillStyle constants for lane backgrounds. Single unified
// tint per lane (no alternating rows) — the gutter on the left and
// the labels above it already carry the "row" affordance; tinting
// every other lane just added rainbow stripes to compete with the
// gradient fill. Idle still dims so the lane the user is touching
// "lights up" relative to the rest.
const LANE_BG_ACTIVE = "rgba(255, 255, 255, 0.042)";
const LANE_BG_IDLE = "rgba(255, 255, 255, 0.018)";
// Inter-lane hairline. Soft enough to not duel with the envelope
// contour but visible enough to mark each row's edge. Used both at
// each lane's top and as the closing rule below the stack — same
// rhythm top-to-bottom so the stack reads as a single ruled rack.
const LANE_RULE = "rgba(255, 255, 255, 0.075)";
// Gutter fill + base divider. Dark enough to occlude the underlying
// lane envelope (so labels sit on a clean opaque backing), warm-tinted
// to match the HUD chrome (--frame / --frame-line in globals.css).
// LANE_GUTTER_RULE is the BASE divider — drawn full-height behind the
// per-lane colored "track tags" so the divider still exists for any
// rows whose tag color isn't cached yet on the first draw.
const LANE_GUTTER_FILL = "rgba(10, 10, 18, 0.92)";
const LANE_GUTTER_RULE = "rgba(255, 255, 255, 0.10)";
// Per-lane "track tag" — a colored stripe straddling the gutter
// divider in each param's family color, the DAW convention for
// identifying a track at a glance. Width chosen so the tag reads as
// an intentional marker (not a hairline) without crowding the lane
// label inside the gutter.
const LANE_TAG_W = 3;

export interface LaneBand {
  /** Engine parameter name (matches LaneState.name). */
  name: string;
  /** Line color, same as LaneState.color — used for the lane label
   * pill border + text. */
  color: RGB;
  /** Current normalized [0, 1] value at the playhead. */
  value: number;
  /** Top edge of the lane band in CSS pixels (canvas coordinate space). */
  bandTop: number;
  /** Height of the lane band in CSS pixels. */
  bandHeight: number;
  /** True when the lane hasn't moved in > LANE_IDLE_MS — the labels
   * overlay dims its pill to match the renderer's idle dimming. */
  idle: boolean;
}

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
  // Per-lane "automation pending" queue. sample() pushes lane names
  // here when their value changes more than AUTOMATION_DELTA_THRESH;
  // draw() reads + consumes them on the next frame, firing a small
  // burst at each one's playhead intersection. Cleared per draw() so
  // a single sample tick produces exactly one burst even if multiple
  // draws happen before the next sample.
  private readonly _automationPending: Set<string> = new Set();
  // Per-line wall-clock millis of last automation fire (rate-limit).
  // Plain Map (not Float32Array) because line names appear/disappear
  // dynamically as LoRAs come online + the map stays small (≤ ~20).
  private readonly _lastAutomationFireAtByName: Map<string, number> = new Map();
  // Per-line previous sample value for delta detection. Mirrors the
  // ring buffer's most-recent entry from the prior sample() call.
  private readonly _lastSampleValueByName: Map<string, number> = new Map();
  // Beat arming + peak tracking. Falling-edge dispatch decides whether
  // the just-ended kick was big enough for chorus or only triggers
  // baseline (or neither, if too soon since the last baseline).
  private _aboveBeat = false;
  private _peakPulse = 0;
  private _lastNow = 0;
  private w = 1;
  private h = 1;

  // ---------- Lanes-mode state -----------------------------------
  // Cached brand-gradient fill for a lane's envelope. Created lazily on
  // first lanes-mode draw and re-created whenever the lane height
  // changes. createLinearGradient is anchored to canvas coordinates,
  // not the current transform, so the gradient is built at y = [0,
  // laneHeight] and we ctx.translate() before each lane fill so the
  // gradient lines up with the lane band. One CanvasGradient ⇒ ~zero
  // per-frame cost.
  private _laneGradient: CanvasGradient | null = null;
  private _laneGradientHeight = 0;
  // Reusable lane band buffer. Filled in place each frame so the labels
  // overlay can read a stable snapshot. The labels overlay polls at
  // 20 Hz — well under draw cadence — so a quick race that reads a
  // mid-fill buffer is harmless (it self-corrects on the next tick).
  private _laneBands: LaneBand[] = [];
  private _laneBandCount = 0;
  // Number of lanes that qualified for display but didn't fit inside
  // MAX_LANES this frame. Read by <GraphLaneLabels/> to render a
  // "+N more" pill below the stack. Zero when everything fits.
  private _hiddenLaneCount = 0;
  // Per-lane scratch arrays for _drawLanes. Pre-allocated so the
  // hot draw path never .push()es above MAX_LANES.
  private readonly _laneNameBuf: (string | null)[] = new Array(MAX_LANES).fill(
    null,
  );
  private readonly _laneLastFireBuf = new Float32Array(MAX_LANES);
  // ---------------------------------------------------------------

  constructor(canvas: HTMLCanvasElement) {
    this.canvas = canvas;
    const ctx = canvas.getContext("2d");
    if (!ctx) throw new Error("GraphRenderer: 2D context unavailable");
    this.ctx = ctx;
    this._resizeObs = new ResizeObserver(() => this._resize());
    this._resizeObs.observe(canvas);
    this._resize();
    _activeGraph = this;
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

  /** Append a new sample point per signal. `defs` supplies max for
   * normalization. Also detects automation (per-lane value change
   * above AUTOMATION_DELTA_THRESH) and queues a spark burst for the
   * next draw(). The burst is read as "this knob is being moved right
   * now" — the per-lane semantic the user feedback called for. */
  sample(
    values: Record<string, number>,
    defs: Record<string, SliderMeta> = SLIDER_META,
  ): void {
    const now = performance.now();
    for (const name of Object.keys(values)) {
      const v = values[name];
      const max = defs[name]?.max ?? 1;
      let hist = this.histories.get(name);
      if (!hist) {
        hist = { buf: new Float32Array(HISTORY_LEN), head: 0, filled: 0 };
        this.histories.set(name, hist);
      }
      const normalized = Math.max(0, Math.min(1, v / max));
      hist.buf[hist.head] = normalized;
      hist.head = (hist.head + 1) % HISTORY_LEN;
      if (hist.filled < HISTORY_LEN) hist.filled += 1;

      // Automation detection. First sample of a new lane gets a free
      // pass (no delta to compare against) but doesn't fire — wait
      // until the user actually moves it.
      const prev = this._lastSampleValueByName.get(name);
      this._lastSampleValueByName.set(name, normalized);
      if (prev === undefined) continue;
      if (Math.abs(normalized - prev) < AUTOMATION_DELTA_THRESH) continue;
      const lastFire = this._lastAutomationFireAtByName.get(name) ?? 0;
      if (now - lastFire < AUTOMATION_MIN_INTERVAL_MS) continue;
      this._lastAutomationFireAtByName.set(name, now);
      this._automationPending.add(name);
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

    // Playhead inset. Polylines mode keeps the historical 1/6-of-width
    // inset (the overshoot trail fills that future space). Lanes mode
    // hugs the right edge — the gutter on the left is CHROME (track
    // headers), the envelope IS the graph, so the graph should reach
    // edge-to-edge. ~20 px is enough breathing room for the playhead
    // glow stroke (up to ~12 px wide at full pulse) and the per-lane
    // anchor dots without clipping. Symmetric "mirror the gutter on
    // the right" looked balanced mathematically but read as misaligned
    // because the dark gutter has more visual weight than empty space.
    const playheadX =
      LAYOUT_MODE === "lanes"
        ? Math.max(w * 0.5, w - 20)
        : w * (1 - PLAYHEAD_INSET_PX_FRAC);
    // Trail length scales with the right-side gap so the visual ratio
    // holds across viewport widths. See OVERSHOOT_FUTURE_FRAC above.
    const overshootPx = Math.max(
      OVERSHOOT_MIN_PX,
      w * PLAYHEAD_INSET_PX_FRAC * OVERSHOOT_FUTURE_FRAC,
    );

    // Pulse-driven radial wash centered at the playhead. Polylines mode
    // only — in lanes mode the playhead now hugs the right edge, so this
    // glow concentrates as a bluish fade extending left from the edge
    // (the asymmetric "fade on that side" the user noticed). DAW
    // automation views don't have a playhead glow either; the kick
    // already reads through the contour line-width modulation and the
    // playhead's own alpha pulse.
    if (LAYOUT_MODE === "polylines" && pulse > 0.02) {
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
    if (LAYOUT_MODE === "polylines") {
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
    } else {
      // Lanes mode — one Ableton-style automation lane per active param,
      // filled envelope (brand gradient) under the polyline contour.
      this._drawLanes(ctx, w, h, playheadX, pulse, now);
    }

    // Per-line dot at the playhead + two-layer leftward confetti
    // trails shed from the dot. Sparks live in the SoA pool above
    // (_spX/_spY/...), capped at MAX_SPARKS, allocated via _allocSpark.
    //
    // Layer 1 (automation): when a lane's value changed in the latest
    // sample() call (> AUTOMATION_DELTA_THRESH, rate-limited per lane),
    // that lane fires a small burst. Reads as "this knob is being
    // moved right now" — the per-lane semantic that addresses the
    // user feedback about not knowing what each line represents.
    //
    // Layer 2 (chorus): on the falling edge of strong kicks (peak ≥
    // CHORUS_THRESH), every line fires a bigger burst at once. This
    // is "the music did that" — distinct from a user automating one
    // knob. Probabilistic so big moments stay rare.
    {
      const dt = this._lastNow ? Math.min(50, now - this._lastNow) : 16;
      this._lastNow = now;
      const dtScale = dt / 16;

      // Falling-edge peak detection over BEAT_THRESH. peakPulse on the
      // disarm frame tells us whether this kick rises to chorus.
      let chorusFire = false;
      let chorusPeakStrength = 0;
      if (pulse > BEAT_THRESH) {
        this._aboveBeat = true;
        if (pulse > this._peakPulse) this._peakPulse = pulse;
      } else if (this._aboveBeat) {
        const peak = this._peakPulse;
        // Chorus is probabilistic — even on strong kicks it only
        // fires CHORUS_FIRE_PROB of the time, so big moments retain
        // their surprise.
        if (
          FIREWORKS_ENABLED &&
          peak >= CHORUS_THRESH &&
          Math.random() < CHORUS_FIRE_PROB
        ) {
          chorusFire = true;
          chorusPeakStrength = peak;
        }
        this._aboveBeat = false;
        this._peakPulse = 0;
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

        // Disc anchored on the line at the playhead. Polylines mode only
        // — lanes mode shows the value via the envelope contour at the
        // playhead, so an extra disc would be redundant.
        if (LAYOUT_MODE === "polylines") {
          ctx.fillStyle = `rgb(${r},${g},${b})`;
          ctx.beginPath();
          ctx.arc(playheadX, dotY, 3, 0, Math.PI * 2);
          ctx.fill();
        }

        // Decide this line's burst size for this frame. Chorus fires
        // every line; automation fires only the lanes whose values
        // changed in the latest sample() call. Chorus takes priority
        // — its multi-line burst already covers every lane, so an
        // automation queue entry would be redundant on the same frame.
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
        } else if (FIREWORKS_ENABLED && this._automationPending.has(name)) {
          burstCount = AUTOMATION_BURST_SPARKS;
          // Single-frame consumption: the queue is the only signal
          // that this lane just moved. Don't clear here — the outer
          // loop is iterating histories, not the pending set; we
          // clear the whole set after the per-line loop instead.
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
      // Consume the automation queue for this frame. Set.clear() reuses
      // the existing bucket array (no realloc), so this is allocation-
      // free at runtime.
      if (this._automationPending.size > 0) {
        this._automationPending.clear();
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
    // Polylines mode only — same reasoning as the radial wash above.
    if (LAYOUT_MODE === "polylines" && pulse > 0.2) {
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
   * Lanes-mode draw path (Concept A — May 2026). One Ableton-style
   * automation lane per actively-used parameter:
   *
   * 1. Filter histories to lanes worth showing: any history with at
   *    least one non-zero sample OR that has fired automation
   *    (last fire timestamp > 0).
   * 2. Cap to MAX_LANES, prioritizing the most-recently changed.
   * 3. Sort the surviving lanes alphabetically for stable on-screen
   *    order — lanes shouldn't shuffle as the user plays.
   * 4. For each lane: faint band background, filled brand-gradient
   *    envelope under the polyline, polyline contour in the param's
   *    family color, anchor dot at the playhead. Idle lanes (no
   *    movement for LANE_IDLE_MS) dim to LANE_IDLE_ALPHA.
   * 5. Publish lane band geometry on this._laneBands so the labels
   *    overlay can pin its name pills at each lane's center.
   *
   * Perf: no per-frame allocations in the hot loop — strings come from
   * _colorTable (cached), the lane gradient is reused across lanes via
   * ctx.translate, lane scratch buffers are pre-allocated. Per
   * PERFORMANCE.md.
   */
  private _drawLanes(
    ctx: CanvasRenderingContext2D,
    w: number,
    h: number,
    playheadX: number,
    pulse: number,
    now: number,
  ): void {
    // Per-frame effective cap: derive how many lanes fit comfortably on
    // THIS canvas height. Targets LANE_TARGET_H per lane, clamps in
    // [LANE_HARD_MIN_COUNT, MAX_LANES]. Tall canvases show more
    // lanes (up to the buffer ceiling); short canvases still keep a
    // usable rack.
    //
    // `macrosReserve` keeps the lane stack clear of the HeroMacros
    // floating panel (desktop only — the panel matches the same
    // 768 px breakpoint). On mobile, graph-wrap's media-query inset
    // already accounts for LiteControls, so no reserve here.
    const macrosReserve =
      typeof window !== "undefined" &&
      typeof window.matchMedia === "function" &&
      window.matchMedia("(min-width: 768px)").matches
        ? LANE_HERO_MACROS_RESERVE
        : 0;
    const availableHForCount = Math.max(
      0,
      h - LANE_TOP_PAD - LANE_BOTTOM_PAD - macrosReserve,
    );
    const computedMax = Math.floor(
      (availableHForCount + LANE_INTER_PAD) /
        (LANE_TARGET_H + LANE_INTER_PAD),
    );
    const effectiveMaxLanes = Math.max(
      LANE_HARD_MIN_COUNT,
      Math.min(MAX_LANES, computedMax),
    );

    // Step 1+2: collect active lanes into the pre-allocated scratch
    // buffer. When the pool is full, evict the lane with the oldest
    // lastFire if this candidate is more recent (linear scan, n ≤ effectiveMaxLanes).
    // Also count totalEligible so we know how many qualifying lanes
    // didn't fit — surfaces as a "+N more" pill via the labels overlay.
    let activeCount = 0;
    let totalEligible = 0;
    for (const [name, hist] of this.histories) {
      if (hist.filled === 0) continue;
      const headIdx = (hist.head - 1 + HISTORY_LEN) % HISTORY_LEN;
      const headV = hist.buf[headIdx];
      const lastFire = this._lastAutomationFireAtByName.get(name) ?? 0;
      // Skip lanes that have never moved AND are currently at zero —
      // dormant LoRAs / unused params, would only add noise.
      if (headV < 0.005 && lastFire === 0) continue;
      totalEligible++;
      if (activeCount < effectiveMaxLanes) {
        this._laneNameBuf[activeCount] = name;
        this._laneLastFireBuf[activeCount] = lastFire;
        activeCount++;
      } else {
        let oldestIdx = 0;
        let oldestVal = this._laneLastFireBuf[0];
        for (let i = 1; i < effectiveMaxLanes; i++) {
          if (this._laneLastFireBuf[i] < oldestVal) {
            oldestVal = this._laneLastFireBuf[i];
            oldestIdx = i;
          }
        }
        if (lastFire > oldestVal) {
          this._laneNameBuf[oldestIdx] = name;
          this._laneLastFireBuf[oldestIdx] = lastFire;
        }
      }
    }
    this._hiddenLaneCount = Math.max(0, totalEligible - activeCount);
    if (activeCount === 0) {
      this._laneBandCount = 0;
      return;
    }

    // Step 3: stable alphabetical order via insertion sort (n ≤ 10;
    // no allocations).
    for (let i = 1; i < activeCount; i++) {
      const ni = this._laneNameBuf[i]!;
      const fi = this._laneLastFireBuf[i];
      let j = i - 1;
      while (j >= 0 && this._laneNameBuf[j]! > ni) {
        this._laneNameBuf[j + 1] = this._laneNameBuf[j];
        this._laneLastFireBuf[j + 1] = this._laneLastFireBuf[j];
        j--;
      }
      this._laneNameBuf[j + 1] = ni;
      this._laneLastFireBuf[j + 1] = fi;
    }

    // Step 4: lane geometry + gradient cache.
    // Both lane-height computation and vertical centering use the
    // SAME macros-reserved available area as the count cap above —
    // otherwise the stack would still extend into the macros zone
    // when active lanes < cap.
    const usableH = Math.max(
      0,
      h - LANE_TOP_PAD - LANE_BOTTOM_PAD - macrosReserve,
    );
    const idealLaneH =
      (usableH - (activeCount - 1) * LANE_INTER_PAD) / activeCount;
    const laneH = Math.max(LANE_MIN_H, Math.min(LANE_MAX_H, idealLaneH));
    // Vertical centering within the lane area (above any macros zone).
    // Lane heights clamp at LANE_MAX_H so a tall canvas with few active
    // lanes leaves dead space — center the stack vertically in the
    // available area (LANE_TOP_PAD → h − LANE_BOTTOM_PAD − macrosReserve)
    // so it feels balanced and never bleeds into the macros region.
    const stackH = activeCount * laneH + (activeCount - 1) * LANE_INTER_PAD;
    const availableBottom = h - LANE_BOTTOM_PAD - macrosReserve;
    const stackTop = Math.max(
      LANE_TOP_PAD,
      (LANE_TOP_PAD + availableBottom - stackH) / 2,
    );
    if (this._laneGradient === null || this._laneGradientHeight !== laneH) {
      const g = ctx.createLinearGradient(0, 0, 0, laneH);
      // Subtle warm wash — accent at the top (where high values live),
      // fading to near-transparent at the baseline. Replaces the v1
      // four-stop rainbow which read as "every lane is a colorful slab"
      // and competed with the envelope contours + labels for attention.
      // The polyline contour still carries each param's family color
      // (orange for feedback, teal for denoise, etc.) so identity
      // doesn't get lost.
      g.addColorStop(0.0, "rgba(240, 138, 72, 0.16)"); // --accent at top
      g.addColorStop(1.0, "rgba(240, 138, 72, 0.02)"); // ~zero at baseline
      this._laneGradient = g;
      this._laneGradientHeight = laneH;
    }

    const pxPerSample = w / (VISIBLE_SAMPLES - 1);

    // Grow the published-bands array lazily. Only on first runs / when
    // a new high-water-mark of active lanes appears; reused thereafter.
    while (this._laneBands.length < activeCount) {
      this._laneBands.push({
        name: "",
        color: [255, 255, 255] as RGB,
        value: 0,
        bandTop: 0,
        bandHeight: 0,
        idle: false,
      });
    }

    for (let li = 0; li < activeCount; li++) {
      const name = this._laneNameBuf[li]!;
      const hist = this.histories.get(name);
      if (!hist) continue;
      const n = Math.min(hist.filled, VISIBLE_SAMPLES);
      const headIdx = (hist.head - 1 + HISTORY_LEN) % HISTORY_LEN;
      const headV = hist.buf[headIdx];
      const lastFire = this._laneLastFireBuf[li];
      const sinceChange = now - lastFire;
      const idle = lastFire === 0 || sinceChange > LANE_IDLE_MS;
      const dimFactor = idle ? LANE_IDLE_ALPHA : 1;
      const laneTop = stackTop + li * (laneH + LANE_INTER_PAD);
      const color = _colorFor(name);

      // Lane background — single unified tint. The gutter on the left
      // + labels carry the "row" affordance; alternating row tints
      // from v1 added rainbow stripes that competed with the gradient.
      // Idle lanes dim so the active row "lights up" relative to the
      // rest of the stack.
      ctx.fillStyle = idle ? LANE_BG_IDLE : LANE_BG_ACTIVE;
      ctx.fillRect(0, laneTop, w, laneH);
      // Subtle top hairline. Marks each row's edge clearly without
      // dueling the envelope contour. Same color is reused below the
      // stack (LANE_RULE) so top + bottom share rhythm.
      ctx.fillStyle = LANE_RULE;
      ctx.fillRect(0, laneTop, w, 1);

      if (n >= 2) {
        const xStart = playheadX - (n - 1) * pxPerSample;
        ctx.save();
        ctx.translate(0, laneTop);

        // Envelope polygon. y inside the lane: 0 value = lane bottom,
        // 1.0 = lane top. Closed at the baseline so a fill renders
        // the area under the polyline.
        ctx.beginPath();
        ctx.moveTo(xStart, laneH);
        let lastY = laneH;
        for (let i = 0; i < n; i++) {
          const bufIdx = (hist.head - n + i + HISTORY_LEN) % HISTORY_LEN;
          const v = hist.buf[bufIdx];
          const x = xStart + i * pxPerSample;
          const y = laneH - v * laneH;
          ctx.lineTo(x, y);
          lastY = y;
        }
        ctx.lineTo(playheadX, laneH);
        ctx.closePath();
        ctx.globalAlpha = dimFactor;
        ctx.fillStyle = this._laneGradient!;
        ctx.fill();

        // Polyline contour — pre-built rgb string via _colorTable so
        // no per-frame string allocation. The param's family color is
        // preserved (orange for feedback, teal for denoise, etc.) so
        // existing color identity holds.
        let colorIdx = this._colorIdxByName.get(name);
        if (colorIdx === undefined) {
          colorIdx = this._colorTable.length;
          this._colorTable.push(`rgb(${color[0]},${color[1]},${color[2]})`);
          this._colorIdxByName.set(name, colorIdx);
        }
        ctx.strokeStyle = this._colorTable[colorIdx];
        ctx.globalAlpha = dimFactor * (0.85 + 0.15 * pulse);
        ctx.lineWidth = 1 + 0.3 * pulse;
        ctx.beginPath();
        for (let i = 0; i < n; i++) {
          const bufIdx = (hist.head - n + i + HISTORY_LEN) % HISTORY_LEN;
          const v = hist.buf[bufIdx];
          const x = xStart + i * pxPerSample;
          const y = laneH - v * laneH;
          if (i === 0) ctx.moveTo(x, y);
          else ctx.lineTo(x, y);
        }
        ctx.stroke();

        // Anchor dot at the playhead. Always full-alpha — even an idle
        // lane's current value should be locatable at a glance.
        ctx.globalAlpha = 1;
        ctx.fillStyle = this._colorTable[colorIdx];
        ctx.beginPath();
        ctx.arc(playheadX, lastY, 2.5, 0, Math.PI * 2);
        ctx.fill();

        ctx.restore();
      }

      // Publish band geometry for the labels overlay. Mutate the
      // existing slot rather than allocating a fresh object per frame.
      const band = this._laneBands[li];
      band.name = name;
      band.color[0] = color[0];
      band.color[1] = color[1];
      band.color[2] = color[2];
      band.value = headV;
      band.bandTop = laneTop;
      band.bandHeight = laneH;
      band.idle = idle;
    }
    // Closing rule below the final lane — same hairline color as the
    // per-lane top rules so the stack reads as a single ruled rack
    // bracketed top-and-bottom.
    const stackBottom = stackTop + stackH;
    ctx.fillStyle = LANE_RULE;
    ctx.fillRect(0, stackBottom, w, 1);
    // Left "track header" gutter — drawn LAST so it occludes the
    // leftmost slice of the envelope fill, giving the lane labels a
    // clean opaque backing instead of floating over the gradient.
    ctx.fillStyle = LANE_GUTTER_FILL;
    ctx.fillRect(0, stackTop, LANE_GUTTER_W, stackBottom - stackTop);
    // Base gutter divider — drawn full-height. The per-lane colored
    // track tags below overpaint this with each param's family color,
    // but the base rule is present so any not-yet-cached lane still
    // has a divider on the first frame after it becomes active.
    ctx.fillStyle = LANE_GUTTER_RULE;
    ctx.fillRect(LANE_GUTTER_W, stackTop, 1, stackBottom - stackTop);
    // Per-lane "track tag" — a 3px colored stripe straddling the
    // gutter divider, in each param's family color. Ableton's classic
    // track-color marker: gives every lane a per-row identity cue at
    // the gutter without saturating the lane body. Drawn AFTER the
    // gutter fill so it sits on top, AFTER the gutter divider so it
    // replaces the white rule for active rows. globalAlpha = 1 even
    // on idle lanes — the tag is the lane's identity, never dimmed.
    for (let li = 0; li < activeCount; li++) {
      const name = this._laneNameBuf[li]!;
      const colorIdx = this._colorIdxByName.get(name);
      if (colorIdx === undefined) continue;
      const laneTop = stackTop + li * (laneH + LANE_INTER_PAD);
      ctx.fillStyle = this._colorTable[colorIdx];
      ctx.fillRect(LANE_GUTTER_W - 1, laneTop, LANE_TAG_W, laneH);
    }
    this._laneBandCount = activeCount;
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
    if (_activeGraph === this) _activeGraph = null;
  }

  /**
   * Read-only snapshot of every line's current state at the playhead.
   * Consumed by the DOM-side <GraphLaneLabels/> overlay so it can
   * position name pills at each line's current value without
   * round-tripping through canvas text (which renders fuzzy at small
   * sizes and bloats the hot draw loop). Called from a 50-ms tick in
   * the labels component — NOT in any per-frame path — so a small
   * allocation per call (one array of LaneState records) is acceptable.
   */
  getLaneStates(): LaneState[] {
    const out: LaneState[] = [];
    for (const [name, hist] of this.histories) {
      if (hist.filled === 0) continue;
      const headIdx = (hist.head - 1 + HISTORY_LEN) % HISTORY_LEN;
      const v = hist.buf[headIdx];
      const y = this.h - Y_PAD - v * (this.h - 2 * Y_PAD);
      const color = _colorFor(name);
      out.push({ name, y, color, value: v });
    }
    return out;
  }

  /** Canvas width in CSS pixels — matches the DOM bounding box that
   * the labels overlay positions against. */
  get cssWidth(): number {
    return this.w;
  }
  /** Canvas height in CSS pixels. */
  get cssHeight(): number {
    return this.h;
  }

  /**
   * Snapshot of currently-rendered lane bands. Returns a reference to
   * the internal buffer (NOT a copy); callers must not mutate. Pair
   * with getLaneBandCount() — the underlying array can be longer than
   * the populated active count across frames.
   */
  getLaneBands(): LaneBand[] {
    return this._laneBands;
  }

  /** Populated entries in getLaneBands(). */
  getLaneBandCount(): number {
    return this._laneBandCount;
  }

  /** Count of qualifying lanes that didn't fit inside MAX_LANES this
   * frame. Surfaced by <GraphLaneLabels/> as a "+N more" pill below
   * the stack so the user knows there's more activity than they're
   * seeing. Zero when everything fits. */
  getHiddenLaneCount(): number {
    return this._hiddenLaneCount;
  }

  /** Current layout mode — read by the labels overlay to decide
   * whether to position pills per-line (at the polyline y at the
   * playhead) or per-lane (at each lane's vertical center). */
  get layoutMode(): "lanes" | "polylines" {
    return LAYOUT_MODE;
  }
}

export interface LaneState {
  /** Engine parameter name (e.g. `denoise`, `ch_g0`). The labels
   * overlay maps this to a display name via DISPLAY_NAMES. */
  name: string;
  /** Current y position in CSS pixels (canvas coordinate system). */
  y: number;
  /** Line color, matching the polyline + playhead dot. */
  color: RGB;
  /** Normalized value [0, 1] used for any "is this lane active /
   * non-default" filtering the overlay wants to do. */
  value: number;
}

// Module-level handle to the currently mounted renderer. The labels
// overlay reads this on its own 50-ms poll; no per-frame coupling.
// Lifecycle is owned by GraphRenderer (constructor sets, destroy()
// clears). Single-instance is true today and the only way it'd change
// is mid-session canvas teardown + re-mount, where the constructor on
// the new instance replaces the slot before any consumer races.
let _activeGraph: GraphRenderer | null = null;
export function getActiveGraphRenderer(): GraphRenderer | null {
  return _activeGraph;
}
