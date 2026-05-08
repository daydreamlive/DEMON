// Parameter-history graph display. Maintains a rolling buffer per signal
// and renders glowing polylines + a playhead.
//
// Glow uses the GPU-accelerated shadowBlur path (not ctx.filter:blur,
// which falls back to software in Skia). The glow pass is intentionally
// dialed back from earlier iterations — at full pulse the bloom should
// read as a gentle in-color smear, not an additive white flash. Tuning
// knobs live in the per-line loop below; bump them in lockstep if the
// pulse needs to read more aggressively again.
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
interface Spark {
  x: number;
  y: number;
  vx: number;
  vy: number;
  age: number;
  life: number;
  r: number;
  g: number;
  b: number;
}

// Spark physics. Disc size matches cursor.ts confetti (2px); trails
// are tuned long + flat so they extend visibly along the rendered
// line history rather than arcing down quickly.
const SPARK_GRAVITY = 0.06; // was 0.10 (cursor 0.16); even flatter for trails along the line
const SPARK_RADIUS = 2; // matches SPARK_RADIUS in cursor.ts (4px diameter)
const SPARK_MIN_SPEED = 4.5;
const SPARK_MAX_SPEED = 8.5;
const SPARK_LIFE_MS = 1200; // was 900; longer so trails reach further into the history
const SPARK_CONE_RAD = Math.PI / 5; // ~36° spread around the leftward axis
const LEFT_ANGLE = Math.PI; // 180° — pure leftward, toward the past

// Baseline trigger — fires on the falling edge of small/medium kicks
// (peak in [BEAT_THRESH, CHORUS_THRESH)). Picks one random line per
// fire so the eye sees a wandering trail rather than constant rain.
// Rate-limited to BASELINE_MIN_INTERVAL_MS between fires; if music is
// silent for longer than BASELINE_MAX_INTERVAL_MS, fires anyway so
// the graph never goes fully still.
const BEAT_THRESH = 0.3;
const BASELINE_MIN_INTERVAL_MS = 400;
const BASELINE_MAX_INTERVAL_MS = 1500;
const BASELINE_BURST_SPARKS = 4;

// Chorus — when a kick's peak strength exceeds CHORUS_THRESH, every
// line fires a bigger burst simultaneously. Big kicks light the whole
// graph; smaller ones get the wandering single-line baseline instead.
// Chorus also fires through the curve-editor gate (see
// `curveEditorOpen` arg on draw()) — the curve scheduler dims the
// baseline so users editing curves aren't distracted, but big musical
// moments still register.
const CHORUS_THRESH = 0.6;
const CHORUS_BURST_BASE = 4;
const CHORUS_BURST_PEAK = 6; // up to +6 more sparks per line scaled by peakPulse

const MAX_SPARKS = 240;

export class GraphRenderer {
  readonly canvas: HTMLCanvasElement;
  private readonly ctx: CanvasRenderingContext2D;
  private readonly histories: Map<string, History> = new Map();
  private readonly _resizeObs: ResizeObserver;
  private readonly _sparks: Spark[] = [];
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

  draw(
    pulse = 0,
    now: number = performance.now(),
    curveEditorOpen = false,
  ): void {
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

      if (pulse > 0.1) {
        ctx.save();
        ctx.globalCompositeOperation = "lighter";
        ctx.shadowColor = `rgba(${r},${g},${b},${0.25 + 0.25 * pulse})`;
        ctx.shadowBlur = 1 + 1.5 * pulse;
        ctx.shadowOffsetX = 0;
        ctx.shadowOffsetY = 0;
        ctx.strokeStyle = `rgba(${r},${g},${b},${0.3 + 0.4 * pulse})`;
        ctx.lineWidth = 1.5 + 1.5 * pulse;
        ctx.stroke();
        ctx.restore();
      }

      ctx.shadowBlur = 0;
      ctx.strokeStyle = `rgba(${r},${g},${b},1)`;
      ctx.lineWidth = 1;
      ctx.stroke();
    }

    // Per-line dot at the playhead + two-layer leftward confetti
    // trails shed from the dot. Sparks live in `this._sparks`, capped
    // at MAX_SPARKS.
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
        if (peak >= CHORUS_THRESH) {
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
      // long, fire one anyway. Skipped when the curve editor is open.
      if (
        !curveEditorOpen &&
        !chorusFire &&
        !baselineFire &&
        now - this._lastBaselineFireAt >= BASELINE_MAX_INTERVAL_MS
      ) {
        baselineFire = true;
      }
      // Curve-editor gate: drop baseline entirely; chorus still fires.
      if (curveEditorOpen) baselineFire = false;

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

      const pxPerSample = w / (VISIBLE_SAMPLES - 1);
      const samplesFromHead = Math.round((w - playheadX) / pxPerSample);

      ctx.save();
      ctx.globalCompositeOperation = "source-over";
      ctx.shadowBlur = 0;

      for (const [name, hist] of this.histories) {
        const n = Math.min(hist.filled, VISIBLE_SAMPLES);
        if (n < 2 || samplesFromHead >= n) continue;
        const headIdx =
          (hist.head - 1 - samplesFromHead + HISTORY_LEN) % HISTORY_LEN;
        const v = hist.buf[headIdx];
        const yAtHead = h - Y_PAD - v * (h - 2 * Y_PAD);
        const [r, g, b] = _colorFor(name);

        // Disc anchored on the line at the playhead.
        ctx.fillStyle = `rgb(${r},${g},${b})`;
        ctx.beginPath();
        ctx.arc(playheadX, yAtHead, 3, 0, Math.PI * 2);
        ctx.fill();

        // Decide this line's burst size for this frame. Chorus fires
        // every line; baseline fires only the chosen line. Mutually
        // exclusive — chorus already fires the chosen line, so baseline
        // is suppressed during chorus.
        let burstCount = 0;
        if (chorusFire) {
          burstCount = chorusBurstCount;
        } else if (name === this._baselineLine) {
          burstCount = BASELINE_BURST_SPARKS;
          this._baselineLine = null; // consumed
        }

        for (let i = 0; i < burstCount; i++) {
          if (this._sparks.length >= MAX_SPARKS) this._sparks.shift();
          const sa = LEFT_ANGLE + (Math.random() - 0.5) * 2 * SPARK_CONE_RAD;
          const sp =
            SPARK_MIN_SPEED +
            Math.random() * (SPARK_MAX_SPEED - SPARK_MIN_SPEED);
          this._sparks.push({
            x: playheadX,
            y: yAtHead,
            vx: Math.cos(sa) * sp,
            vy: Math.sin(sa) * sp,
            age: 0,
            life: SPARK_LIFE_MS - 150 + Math.random() * 300,
            r,
            g,
            b,
          });
        }
      }

      // Sparks — physics + render + cull. Walking backwards so splice
      // doesn't shift indices we still need to visit.
      for (let i = this._sparks.length - 1; i >= 0; i--) {
        const s = this._sparks[i];
        s.age += dt;
        if (s.age >= s.life) {
          this._sparks.splice(i, 1);
          continue;
        }
        s.vy += SPARK_GRAVITY * dtScale;
        s.x += s.vx * dtScale;
        s.y += s.vy * dtScale;
        const f = s.age / s.life;
        const alpha = 1 - f;
        const radius = SPARK_RADIUS * (1 - f * 0.7);
        if (radius <= 0.1) continue;
        ctx.fillStyle = `rgba(${s.r},${s.g},${s.b},${alpha.toFixed(3)})`;
        ctx.beginPath();
        ctx.arc(s.x, s.y, radius, 0, Math.PI * 2);
        ctx.fill();
      }

      ctx.restore();
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
