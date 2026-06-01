"use client";

import { useCallback, useEffect, useRef, useState } from "react";

import { computePeaks, drawPeaks } from "@/engine/curves/waveformPeaks";
import { frameScheduler } from "@/engine/scheduler/FrameScheduler";
import { useOneShotTooltip } from "@/hooks/useOneShotTooltip";
import {
  LOOP_GRID_LABEL,
  LOOP_GRID_ORDER,
  type LoopGridRes,
} from "@/lib/loopGrid";
import { useCurveStore } from "@/store/useCurveStore";
import { usePerformanceStore } from "@/store/usePerformanceStore";
import { useSessionStore } from "@/store/useSessionStore";

// Bottom-center scrub strip. Shows the source-track waveform, overlays
// a playhead at the current AudioPlayer.positionSec, lets the operator
// click/drag anywhere to call player.seek(t).
//
// The audio buffer here IS the looped source track — the engine streams
// generated `audio_slice` messages that `patch` into this same buffer.
// Seek is therefore client-only and instant: jumping to t=X plays
// whichever lives there now (generated audio if that region has been
// touched in a prior lap; otherwise the original source). No engine
// protocol message is involved — see audio-worklet.js:106 and
// AudioPlayer.seek():339.
//
// Loop bands (v3 — DAW-style):
//   • LOOP button       → with no region, arms "draw mode"; a plain drag
//                          then draws a band. With a region, toggles the
//                          loop ENABLED on/off WITHOUT destroying the
//                          region (DAWs keep the brace, just stop looping).
//   • Shift + drag      → draw a band without arming (power-user shortcut)
//   • Drag band body    → move the whole band (length preserved)
//   • Drag band edge    → resize that edge
//   • Right-click / ✕   → remove the region entirely
//   • Grid selector     → snap resolution (Bar / ½ / Beat / ⅛); Beat default,
//                          always available (even before the first draw)
//   • ÷2 / ×2           → halve / double the loop length (anchored at start)
//   • ◀ / ▶            → nudge the region by one grid step
//   • Length readout    → bars (+ seconds) of the current region
//   • Keyboard          → L toggle, ←/→ nudge, Shift+←/→ move-by-length
//                          (dispatched from useKeyboardShortcuts via the
//                          dd:loop-toggle / dd:loop-nudge custom events)
// Band edges snap to the selected musical grid (derived from the detected
// BPM + time signature); hold Alt while dragging for a free, un-snapped
// adjustment. The beat/bar grid is drawn faintly on the strip.
//
// Playback is client-side via AudioPlayer.setLoopBand/clearLoopBand, which
// the AudioWorklet honours by wrapping end→start (now with a seam
// crossfade) on each pass. React owns the region + enabled flag as the
// single source of truth: while enabled the band is mirrored to the
// worklet AND to the server (remote.sendLoopBand) so the pipeline wraps
// its predictive decode target inside the band; while disabled both are
// cleared, so the worklet resumes full-buffer looping and the backend
// stops chasing the band. No worklet or server change is needed for the
// enable/disable split — clearing the band is the existing "no loop" path.

const WAVEFORM_BUCKETS = 640;
const EDGE_PADDING_PX = 4;
const BAND_EDGE_HIT_PX = 7; // hit-zone radius around each band edge
const MIN_BAND_SEC = 0.05;  // 50 ms — below this the band is meaningless
const FALLBACK_STEP_SEC = 0.5; // nudge step when BPM is unknown

function ensureCanvasSize(canvas: HTMLCanvasElement): {
  w: number;
  h: number;
  dpr: number;
} {
  const rect = canvas.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  const w = Math.max(1, Math.floor(rect.width));
  const h = Math.max(1, Math.floor(rect.height));
  const targetW = Math.floor(w * dpr);
  const targetH = Math.floor(h * dpr);
  if (canvas.width !== targetW || canvas.height !== targetH) {
    canvas.width = targetW;
    canvas.height = targetH;
  }
  return { w, h, dpr };
}

/** Length (seconds) of one snap unit at the given resolution. Returns 0
 *  when the beat length is unknown (BPM not detected) → snapping is a
 *  no-op and edits stay free. A bar is `beatsPerBar` beats, a half-bar is
 *  half that, a beat is 1, an eighth is ½ a beat. */
function snapUnitSec(res: LoopGridRes, beatSec: number, beatsPerBar: number): number {
  if (beatSec <= 0) return 0;
  switch (res) {
    case "bar":
      return beatSec * beatsPerBar;
    case "half":
      return (beatSec * beatsPerBar) / 2;
    case "beat":
      return beatSec;
    case "eighth":
      return beatSec / 2;
  }
}

/** Snap a time to the nearest multiple of `unit`. `unit <= 0` → no-op. */
function snapToUnit(t: number, unit: number): number {
  if (unit <= 0) return t;
  return Math.round(t / unit) * unit;
}

type Band = { start: number; end: number };
type Grid = { beatSec: number; beatsPerBar: number };
type DragMode =
  | "seek"
  | "draw-band"
  | "move-band"
  | "resize-band-start"
  | "resize-band-end";

interface DragState {
  mode: DragMode;
  /** Anchor time at the moment of pointerdown — interpretation depends
   *  on mode. seek/draw-band: t at pointerdown. move-band: original
   *  pointerdown t. resize-band-*: original t of the edge being grabbed. */
  anchorT: number;
  /** For move-band: original band at pointerdown so we can recompute
   *  start/end from delta on every move without drift. */
  startBand?: Band;
}

/** Move a whole band by `delta` seconds, clamped into [0, duration] with
 *  the length preserved (slides against the buffer ends rather than
 *  resizing). Shared by drag-move and keyboard nudge. */
function moveBandBy(b: Band, delta: number, duration: number): Band {
  const len = b.end - b.start;
  let start = b.start + delta;
  if (start < 0) start = 0;
  if (start + len > duration) start = Math.max(0, duration - len);
  return { start, end: start + len };
}

/** Scale a band's length about its start by `factor`, clamped to the
 *  buffer and re-snapped to the grid. Used by the ÷2 / ×2 buttons. */
function scaleBand(
  b: Band,
  factor: number,
  duration: number,
  unit: number,
): Band {
  const len = Math.max(MIN_BAND_SEC, (b.end - b.start) * factor);
  let end = Math.min(duration, b.start + len);
  if (unit > 0) {
    end = Math.min(
      duration,
      Math.max(b.start + MIN_BAND_SEC, snapToUnit(end, unit)),
    );
  }
  return { start: b.start, end };
}

export function WaveformScrubBox() {
  const player = useSessionStore((s) => s.player);
  const curvesOpen = useCurveStore((s) => s.overlayOpen);
  // Musical grid inputs — drive both the drawn grid and edge snapping.
  const detectedBpm = usePerformanceStore((s) => s.detectedBpm);
  const timeSignature = usePerformanceStore((s) => s.activeTimeSignature);
  const boxRef = useRef<HTMLDivElement>(null);
  const bgCanvasRef = useRef<HTMLCanvasElement>(null);
  const fgCanvasRef = useRef<HTMLCanvasElement>(null);

  // The loop region (seconds), its enabled flag, and the snap resolution
  // all live in the performance store so they survive config export /
  // import — WaveformScrubBox is just the editor. Mirrored to the worklet
  // + server only when `bandLoopEnabled` is also true. Refs mirror the
  // store values so the rAF foreground tick + pointer/keyboard handlers can
  // read them without re-subscribing on every change.
  const bandState = usePerformanceStore((s) => s.loopBand);
  const setBandState = usePerformanceStore((s) => s.setLoopBand);
  const bandRef = useRef<Band | null>(null);
  bandRef.current = bandState;

  // Whether band looping is active. Decoupled from the region's existence:
  // a region can sit armed-but-off (drawn dim) and be re-enabled without
  // redrawing — exactly how a DAW loop toggle behaves. Defaults on so a
  // freshly drawn region loops immediately.
  const bandLoopEnabled = usePerformanceStore((s) => s.bandLoopEnabled);
  const setBandLoopEnabledStore = usePerformanceStore(
    (s) => s.setBandLoopEnabled,
  );
  const bandLoopEnabledRef = useRef(true);
  bandLoopEnabledRef.current = bandLoopEnabled;

  // Draw-arm: when true (and no region yet) a plain drag draws a band.
  // Transient UI (not exported), so it stays local. Replaces the old
  // "loopMode" — once a region exists the LOOP button toggles
  // `bandLoopEnabled` instead.
  const [armDraw, setArmDraw] = useState(false);
  const armDrawRef = useRef(false);
  armDrawRef.current = armDraw;

  // Snap resolution. Beat by default — fine enough to grab sub-bar loops
  // straight away (matches the pre-grid-selector behaviour); cycle to
  // Bar / ½ / ⅛ as needed. Hold Alt while dragging for fully free edits.
  const gridRes = usePerformanceStore((s) => s.loopGridRes);
  const setGridResStore = usePerformanceStore((s) => s.setLoopGridRes);
  const gridResRef = useRef<LoopGridRes>("beat");
  gridResRef.current = gridRes;

  // Beat grid, recomputed each render and shared with the rAF tick +
  // pointer handlers via a ref.
  const beatsPerBar = Math.max(1, parseInt(String(timeSignature), 10) || 4);
  const beatSec = detectedBpm && detectedBpm > 0 ? 60 / detectedBpm : 0;
  const gridRef = useRef<Grid>({ beatSec: 0, beatsPerBar: 4 });
  gridRef.current = { beatSec, beatsPerBar };

  const [hasPeaks, setHasPeaks] = useState(false);
  const hasPlayer = player !== null;

  // ── Background canvas: waveform ───────────────────────────────────
  useEffect(() => {
    if (!hasPlayer) {
      setHasPeaks(false);
      return;
    }
    const canvas = bgCanvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    let peaks: Float32Array | null = null;

    const redraw = () => {
      const { w, h, dpr } = ensureCanvasSize(canvas);
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, w, h);
      if (!peaks) return;
      ctx.fillStyle = "rgba(240, 138, 72, 0.28)";
      drawPeaks(ctx, peaks, w, h);
      ctx.fillStyle = "rgba(255, 255, 255, 0.08)";
      drawPeaks(ctx, peaks, w, h);
    };

    const recompute = () => {
      const p = useSessionStore.getState().player;
      if (!p) {
        peaks = null;
        setHasPeaks(false);
      } else {
        const mirror = p.getMirror();
        peaks = computePeaks(mirror, p.channels, WAVEFORM_BUCKETS);
        setHasPeaks(true);
      }
      redraw();
    };

    recompute();

    const ro = new ResizeObserver(redraw);
    ro.observe(canvas);

    const unsubMirror = player.onMirrorChange?.(() => recompute()) ?? (() => {});
    const unsubSession = useSessionStore.subscribe((s, prev) => {
      if (s.player !== prev.player) recompute();
    });

    return () => {
      ro.disconnect();
      unsubMirror();
      unsubSession();
    };
  }, [hasPlayer, player]);

  // ── Foreground canvas: grid + playhead + active band ──────────────
  useEffect(() => {
    if (!hasPlayer) return;
    const canvas = fgCanvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const tick = () => {
      const p = useSessionStore.getState().player;
      if (!p) return;
      const duration = p.duration;
      if (duration <= 0) return;
      const { w, h, dpr } = ensureCanvasSize(canvas);
      const innerW = Math.max(1, w - 2 * EDGE_PADDING_PX);
      const tToX = (t: number) =>
        EDGE_PADDING_PX + (Math.min(1, Math.max(0, t / duration))) * innerW;

      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, w, h);

      // Beat grid — faint beat lines, brighter bar lines. Lets the
      // operator see where bars start (and where edges will snap).
      const grid = gridRef.current;
      if (grid.beatSec > 0) {
        const beatPath = new Path2D();
        const barPath = new Path2D();
        let i = 0;
        for (let tb = 0; tb <= duration + 1e-6; tb += grid.beatSec, i++) {
          const gx = tToX(tb);
          const path = i % grid.beatsPerBar === 0 ? barPath : beatPath;
          path.moveTo(gx, 0);
          path.lineTo(gx, h);
        }
        ctx.lineWidth = 1;
        ctx.strokeStyle = "rgba(255, 255, 255, 0.05)";
        ctx.stroke(beatPath);
        ctx.strokeStyle = "rgba(255, 222, 196, 0.20)";
        ctx.stroke(barPath);
      }

      // Active band — translucent orange rect. Drawn dim when the region
      // exists but looping is disabled, so "armed-but-off" reads at a
      // glance (the region is still there, just not looping).
      const band = bandRef.current;
      if (band) {
        const enabled = bandLoopEnabledRef.current;
        const x0 = tToX(band.start);
        const x1 = tToX(band.end);
        ctx.fillStyle = enabled
          ? "rgba(240, 138, 72, 0.18)"
          : "rgba(240, 138, 72, 0.06)";
        ctx.fillRect(Math.min(x0, x1), 0, Math.abs(x1 - x0), h);
        // Edge markers — slightly brighter so the resize hit-zones are
        // findable by eye; dimmer when the loop is off.
        ctx.fillStyle = enabled
          ? "rgba(255, 222, 196, 0.55)"
          : "rgba(255, 222, 196, 0.25)";
        ctx.fillRect(x0 - 0.5, 0, 1.5, h);
        ctx.fillRect(x1 - 0.5, 0, 1.5, h);
      }

      // Playhead — orange line with halo.
      const x = tToX(p.positionSec);
      ctx.strokeStyle = "rgba(240, 138, 72, 0.28)";
      ctx.lineWidth = 3;
      ctx.beginPath();
      ctx.moveTo(x, 2);
      ctx.lineTo(x, h - 2);
      ctx.stroke();
      ctx.strokeStyle = "rgba(255, 222, 196, 0.95)";
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(x, 2);
      ctx.lineTo(x, h - 2);
      ctx.stroke();
    };

    const unregister = frameScheduler.register("waveform-scrub", tick, {
      phase: "compute",
      budgetMs: 0.2,
    });
    return () => unregister();
  }, [hasPlayer]);

  // Apply / clear band on the worklet + server whenever the region or the
  // enabled flag changes. Active = a valid region AND looping enabled;
  // anything else clears both, which is the existing "no loop" path (so a
  // disabled-but-kept region needs no new worklet/server message). Guarded
  // with `typeof === "function"` because a session that started before
  // these methods shipped has an AudioPlayer whose prototype predates them.
  useEffect(() => {
    if (!hasPlayer || !player) return;
    const setBand = (player as unknown as {
      setLoopBand?: (s: number, e: number) => void;
    }).setLoopBand;
    const clearBand = (player as unknown as {
      clearLoopBand?: () => void;
    }).clearLoopBand;
    const remote = useSessionStore.getState().remote;
    const active =
      bandState !== null &&
      bandLoopEnabled &&
      bandState.end - bandState.start >= MIN_BAND_SEC &&
      bandState.start >= 0;
    if (active) {
      if (typeof setBand === "function") {
        setBand.call(player, bandState.start, bandState.end);
      }
      remote?.sendLoopBand(bandState.start, bandState.end);
    } else {
      if (typeof clearBand === "function") clearBand.call(player);
      remote?.sendLoopBand(null, null);
    }
  }, [bandState, bandLoopEnabled, hasPlayer, player]);

  // ── Shared loop actions (used by buttons + keyboard) ──────────────
  // All read live state from refs / the session store, so the stable
  // ([]) identities never go stale.

  // LOOP toggle: with a region, flip band looping (region persists). With
  // no region, arm/disarm draw mode.
  const toggleLoop = useCallback(() => {
    if (bandRef.current) {
      setBandLoopEnabledStore(!bandLoopEnabledRef.current);
    } else {
      setArmDraw((v) => !v);
    }
  }, [setBandLoopEnabledStore]);

  // Remove the region entirely. Leaves band-loop enabled so the next drawn
  // region loops immediately.
  const clearRegion = useCallback(() => {
    setBandState(null);
    setArmDraw(false);
    setBandLoopEnabledStore(true);
  }, [setBandState, setBandLoopEnabledStore]);

  // Live edit context (region + buffer length + snap unit) shared by nudge
  // and scale. Null when there's nothing editable (no region, or duration
  // not known yet). Reads only refs / the session store, so it's stable.
  const editCtx = useCallback(() => {
    const b = bandRef.current;
    if (!b) return null;
    const duration = useSessionStore.getState().player?.duration ?? 0;
    if (duration <= 0) return null;
    const unit = snapUnitSec(
      gridResRef.current,
      gridRef.current.beatSec,
      gridRef.current.beatsPerBar,
    );
    return { b, duration, unit };
  }, []);

  // Nudge / move the region. `byLength` moves by the region's own length
  // (phrase jump); otherwise by one grid step (falls back to a fixed step
  // when BPM is unknown).
  const nudgeRegion = useCallback(
    (dir: 1 | -1, byLength: boolean) => {
      const ctx = editCtx();
      if (!ctx) return;
      const { b, duration, unit } = ctx;
      const step = byLength
        ? b.end - b.start
        : unit > 0
          ? unit
          : FALLBACK_STEP_SEC;
      setBandState(moveBandBy(b, dir * step, duration));
    },
    [editCtx, setBandState],
  );

  // Halve / double the region length, anchored at its start.
  const scaleRegion = useCallback(
    (factor: number) => {
      const ctx = editCtx();
      if (!ctx) return;
      setBandState(scaleBand(ctx.b, factor, ctx.duration, ctx.unit));
    },
    [editCtx, setBandState],
  );

  const cycleGrid = useCallback(() => {
    const cur = gridResRef.current;
    setGridResStore(
      LOOP_GRID_ORDER[(LOOP_GRID_ORDER.indexOf(cur) + 1) % LOOP_GRID_ORDER.length],
    );
  }, [setGridResStore]);

  // ── Keyboard bridge ───────────────────────────────────────────────
  // useKeyboardShortcuts dispatches dd:loop-toggle / dd:loop-nudge so the
  // loop keys live in the one global shortcut hub; the region state lives
  // here, so we listen for those events.
  useEffect(() => {
    if (!hasPlayer) return;
    const onToggle = () => toggleLoop();
    const onNudge = (e: Event) => {
      const detail = (e as CustomEvent).detail as
        | { dir: 1 | -1; byLength: boolean }
        | undefined;
      if (!detail) return;
      nudgeRegion(detail.dir, detail.byLength);
    };
    document.addEventListener("dd:loop-toggle", onToggle);
    document.addEventListener("dd:loop-nudge", onNudge);
    return () => {
      document.removeEventListener("dd:loop-toggle", onToggle);
      document.removeEventListener("dd:loop-nudge", onNudge);
    };
  }, [hasPlayer, toggleLoop, nudgeRegion]);

  // ── Pointer state machine ─────────────────────────────────────────
  useEffect(() => {
    if (!hasPlayer) return;
    const box = boxRef.current;
    if (!box) return;

    let drag: DragState | null = null;

    const tFromEvent = (e: PointerEvent): number => {
      const p = useSessionStore.getState().player;
      if (!p) return 0;
      const duration = p.duration;
      if (duration <= 0) return 0;
      const rect = box.getBoundingClientRect();
      const innerW = Math.max(1, rect.width - 2 * EDGE_PADDING_PX);
      const x = e.clientX - rect.left - EDGE_PADDING_PX;
      return Math.min(duration, Math.max(0, (x / innerW) * duration));
    };

    /** Pixels per second at the current canvas width. Used to convert
     *  the band-edge hit-zone (defined in pixels) to a tolerance in
     *  seconds at pointerdown. */
    const secPerPx = (): number => {
      const p = useSessionStore.getState().player;
      if (!p || p.duration <= 0) return 0;
      const rect = box.getBoundingClientRect();
      const innerW = Math.max(1, rect.width - 2 * EDGE_PADDING_PX);
      return p.duration / innerW;
    };

    /** Snap a time to the active grid resolution unless Alt is held. */
    const snapT = (t: number, e: PointerEvent): number =>
      e.altKey
        ? t
        : snapToUnit(
            t,
            snapUnitSec(
              gridResRef.current,
              gridRef.current.beatSec,
              gridRef.current.beatsPerBar,
            ),
          );

    const onDown = (e: PointerEvent) => {
      // Right-click → clear region (handled in contextmenu). Ignore here.
      if (e.button !== 0) return;
      // A press on any loop control (children of this box) must not also
      // scrub: the native event still bubbles to this listener even
      // though the buttons call React's synthetic stopPropagation.
      if (
        (e.target as HTMLElement | null)?.closest?.(".waveform-loop-controls")
      ) {
        return;
      }

      const t = tFromEvent(e);
      const band = bandRef.current;
      const tol = secPerPx() * BAND_EDGE_HIT_PX;
      // Loop editing is active when the user armed draw mode OR is
      // holding Shift (the power-user shortcut).
      const loopEditing = e.shiftKey || armDrawRef.current;

      let mode: DragMode = "seek";
      let anchorT = t;
      let startBand: Band | undefined;

      if (band && Math.abs(t - band.start) <= tol) {
        mode = "resize-band-start";
      } else if (band && Math.abs(t - band.end) <= tol) {
        mode = "resize-band-end";
      } else if (band && t >= band.start && t <= band.end) {
        mode = "move-band";
        anchorT = t;
        startBand = { ...band };
      } else if (loopEditing) {
        // Draw a brand-new band from this point (snapped to the grid).
        // A fresh draw re-enables looping so it's audible immediately.
        mode = "draw-band";
        anchorT = snapT(t, e);
        setBandLoopEnabledStore(true);
        setBandState({ start: anchorT, end: anchorT });
      } else {
        // Plain click outside any band → regular seek.
        mode = "seek";
      }

      drag = { mode, anchorT, startBand };
      box.setPointerCapture(e.pointerId);

      if (mode === "seek") {
        const p = useSessionStore.getState().player;
        p?.seek(t);
      }
    };

    const onMove = (e: PointerEvent) => {
      if (!drag) return;
      const t = tFromEvent(e);
      const p = useSessionStore.getState().player;
      const duration = p?.duration ?? 0;

      switch (drag.mode) {
        case "seek":
          p?.seek(t);
          return;
        case "draw-band": {
          const a = drag.anchorT;
          const ts = snapT(t, e);
          const start = Math.max(0, Math.min(a, ts));
          const end = Math.min(duration, Math.max(a, ts));
          setBandState({ start, end });
          return;
        }
        case "move-band": {
          const sb = drag.startBand;
          if (!sb) return;
          const delta = t - drag.anchorT;
          // Snap the moved start to the grid, then preserve length and
          // clamp against the buffer ends.
          const snappedStart = snapT(sb.start + delta, e);
          setBandState(
            moveBandBy(sb, snappedStart - sb.start, duration),
          );
          return;
        }
        case "resize-band-start": {
          const b = bandRef.current;
          if (!b) return;
          const newStart = Math.min(
            b.end - MIN_BAND_SEC,
            Math.max(0, snapT(t, e)),
          );
          setBandState({ start: newStart, end: b.end });
          return;
        }
        case "resize-band-end": {
          const b = bandRef.current;
          if (!b) return;
          const newEnd = Math.max(
            b.start + MIN_BAND_SEC,
            Math.min(duration, snapT(t, e)),
          );
          setBandState({ start: b.start, end: newEnd });
          return;
        }
      }
    };

    const onUp = (e: PointerEvent) => {
      if (!drag) return;
      // Draw mode finalises on release: if the user just tap-clicked
      // (no real drag), kill the band so we don't lock playback to a
      // zero-width sliver. A successful draw disarms draw mode.
      if (drag.mode === "draw-band") {
        const b = bandRef.current;
        if (!b || b.end - b.start < MIN_BAND_SEC) {
          setBandState(null);
        } else {
          setArmDraw(false);
        }
      }
      drag = null;
      try {
        box.releasePointerCapture(e.pointerId);
      } catch {}
    };

    const onContextMenu = (e: MouseEvent) => {
      // Right-click removes the region entirely. Operators familiar with
      // DAWs expect "right-click loop marker = remove" so we mirror that.
      if (bandRef.current) {
        e.preventDefault();
        clearRegion();
      }
    };

    box.addEventListener("pointerdown", onDown);
    box.addEventListener("pointermove", onMove);
    box.addEventListener("pointerup", onUp);
    box.addEventListener("pointercancel", onUp);
    box.addEventListener("contextmenu", onContextMenu);
    return () => {
      box.removeEventListener("pointerdown", onDown);
      box.removeEventListener("pointermove", onMove);
      box.removeEventListener("pointerup", onUp);
      box.removeEventListener("pointercancel", onUp);
      box.removeEventListener("contextmenu", onContextMenu);
    };
  }, [hasPlayer, clearRegion]);

  // One-shot tooltip on first hover.
  const tipProps = useOneShotTooltip(
    "waveform-scrub",
    "Click to scrub · LOOP to set a loop region",
  );

  // Button visual state: with a region, "active" tracks bandLoopEnabled;
  // with no region, it tracks the draw-arm.
  const hasBand = bandState !== null;
  const loopActive = hasBand ? bandLoopEnabled : armDraw;
  const loopTitle = hasBand
    ? bandLoopEnabled
      ? "Disable loop (keeps the region)"
      : "Enable loop"
    : armDraw
      ? "Cancel — disarm loop drawing"
      : "Arm loop, then drag across the waveform to set a region";

  // Length readout in bars (when BPM known) + seconds.
  const lengthLabel = (() => {
    if (!bandState) return "";
    const len = bandState.end - bandState.start;
    const secs = `${len.toFixed(2)}s`;
    if (beatSec > 0) {
      const bars = len / (beatSec * beatsPerBar);
      const barStr =
        Math.abs(bars - Math.round(bars)) < 1e-3
          ? String(Math.round(bars))
          : bars.toFixed(2);
      return `${barStr} bars · ${secs}`;
    }
    return secs;
  })();

  // Render the DOM as soon as we have a player so the peak-compute
  // effect can find the canvases in the DOM. The strip stays visually
  // hidden (opacity 0) until the first peak-pass lands.
  if (!hasPlayer) return null;

  return (
    <div
      ref={boxRef}
      className="waveform-scrub-box"
      data-curves-open={curvesOpen ? "true" : undefined}
      data-ready={hasPeaks ? "true" : undefined}
      data-has-band={hasBand ? "true" : undefined}
      data-loop-mode={armDraw ? "true" : undefined}
      data-dd-tooltip-pos="below"
      role="slider"
      aria-label="Scrub playhead"
      {...tipProps}
    >
      <canvas ref={bgCanvasRef} className="waveform-scrub-bg" aria-hidden="true" />
      <canvas ref={fgCanvasRef} className="waveform-scrub-fg" aria-hidden="true" />
      <div className="waveform-loop-controls">
        <button
          type="button"
          className="waveform-loop-toggle"
          data-active={loopActive ? "true" : undefined}
          aria-pressed={loopActive}
          title={loopTitle}
          onClick={(e) => {
            e.stopPropagation();
            toggleLoop();
          }}
        >
          LOOP
        </button>
        {beatSec > 0 && (
          <button
            type="button"
            className="waveform-loop-btn"
            title={`Loop snap: ${LOOP_GRID_LABEL[gridRes]} (click to cycle · hold Alt while dragging for free)`}
            onClick={(e) => {
              e.stopPropagation();
              cycleGrid();
            }}
          >
            {LOOP_GRID_LABEL[gridRes]}
          </button>
        )}
        {hasBand && (
          <>
            <button
              type="button"
              className="waveform-loop-btn"
              title="Halve the loop length"
              onClick={(e) => {
                e.stopPropagation();
                scaleRegion(0.5);
              }}
            >
              ÷2
            </button>
            <button
              type="button"
              className="waveform-loop-btn"
              title="Double the loop length"
              onClick={(e) => {
                e.stopPropagation();
                scaleRegion(2);
              }}
            >
              ×2
            </button>
            <button
              type="button"
              className="waveform-loop-btn"
              title="Nudge loop earlier (Shift-click moves by its length)"
              onClick={(e) => {
                e.stopPropagation();
                nudgeRegion(-1, e.shiftKey);
              }}
            >
              ◀
            </button>
            <button
              type="button"
              className="waveform-loop-btn"
              title="Nudge loop later (Shift-click moves by its length)"
              onClick={(e) => {
                e.stopPropagation();
                nudgeRegion(1, e.shiftKey);
              }}
            >
              ▶
            </button>
            <span className="waveform-loop-len" aria-hidden="true">
              {lengthLabel}
            </span>
            <button
              type="button"
              className="waveform-loop-btn waveform-loop-clear"
              title="Remove the loop region"
              onClick={(e) => {
                e.stopPropagation();
                clearRegion();
              }}
            >
              ✕
            </button>
          </>
        )}
      </div>
    </div>
  );
}
