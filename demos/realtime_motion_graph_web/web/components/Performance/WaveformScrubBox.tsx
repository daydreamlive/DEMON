"use client";

import { useEffect, useRef, useState } from "react";

import { computePeaks, drawPeaks } from "@/engine/curves/waveformPeaks";
import { frameScheduler } from "@/engine/scheduler/FrameScheduler";
import { useOneShotTooltip } from "@/hooks/useOneShotTooltip";
import { useConfig } from "@/lib/config";
import { useCurveStore } from "@/store/useCurveStore";
import { usePerformanceStore } from "@/store/usePerformanceStore";
import { useSessionStore } from "@/store/useSessionStore";
import type { LoopBandSuggestionMessage } from "@/types/protocol";

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
// Loop bands (v2):
//   • LOOP button       → arm "loop mode"; while armed a plain drag on
//                          the waveform draws a band (no Shift needed).
//                          Click again clears the loop entirely.
//   • Shift + drag      → draw a band without arming (power-user shortcut)
//   • Drag band body    → move the whole band
//   • Drag band edge    → resize that edge
//   • Right-click band  → clear
// Band edges snap to the musical grid (bars + beats, from the detected
// BPM + time signature); hold Alt while dragging for a free, un-snapped
// adjustment. The grid itself is drawn faintly on the strip so bar
// boundaries are visible at a glance.
//
// Playback is client-side via AudioPlayer.setLoopBand/clearLoopBand, which
// the AudioWorklet honours by wrapping end→start on each pass. The band is
// also mirrored to the server (remote.sendLoopBand) so the pipeline wraps
// its predictive decode target inside the band — without that the backend
// chases the raw playhead, decodes past the band end, and leaves one stale
// window of pre-change audio at the loop start on every restart.

const WAVEFORM_BUCKETS = 640;
const EDGE_PADDING_PX = 4;
const BAND_EDGE_HIT_PX = 7; // hit-zone radius around each band edge
const MIN_BAND_SEC = 0.05;  // 50 ms — below this the band is meaningless

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

/** Snap a time to the nearest beat. `beatSec <= 0` (BPM unknown) → no-op. */
function snapToBeat(t: number, beatSec: number): number {
  if (beatSec <= 0) return t;
  return Math.round(t / beatSec) * beatSec;
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

export function WaveformScrubBox() {
  const config = useConfig();
  const player = useSessionStore((s) => s.player);
  const curvesOpen = useCurveStore((s) => s.overlayOpen);
  // Musical grid inputs — drive both the drawn grid and edge snapping.
  const detectedBpm = usePerformanceStore((s) => s.detectedBpm);
  const timeSignature = usePerformanceStore((s) => s.activeTimeSignature);
  const boxRef = useRef<HTMLDivElement>(null);
  const bgCanvasRef = useRef<HTMLCanvasElement>(null);
  const fgCanvasRef = useRef<HTMLCanvasElement>(null);

  // Active band (seconds). Mirrored to the worklet via player.setLoopBand
  // any time it changes to a complete band. Kept in a ref so the rAF
  // foreground tick can read it without making React state changes
  // invalidate the tick's frameScheduler subscription.
  const [bandState, setBandState] = useState<Band | null>(null);
  const bandRef = useRef<Band | null>(null);
  bandRef.current = bandState;
  const smartAnchorRef = useRef<Band | null>(null);
  const smartRequestIdRef = useRef(0);
  const smartPendingRef = useRef<{ id: number; sentAt: number } | null>(null);

  const setUserBand = (band: Band | null) => {
    smartAnchorRef.current = band ? { ...band } : null;
    smartPendingRef.current = null;
    setBandState(band);
  };

  // Loop mode: when armed, a plain drag on the waveform draws/edits the
  // band (Shift no longer required). The pointer handlers read the ref.
  const [loopMode, setLoopMode] = useState(false);
  const loopModeRef = useRef(false);
  loopModeRef.current = loopMode;

  // Beat grid, recomputed each render and shared with the rAF tick +
  // pointer handlers via a ref.
  const beatsPerBar = Math.max(1, parseInt(String(timeSignature), 10) || 4);
  const beatSec = detectedBpm && detectedBpm > 0 ? 60 / detectedBpm : 0;
  const gridRef = useRef<Grid>({ beatSec: 0, beatsPerBar: 4 });
  gridRef.current = { beatSec, beatsPerBar };

  const [hasPeaks, setHasPeaks] = useState(false);
  const hasPlayer = player !== null;
  const audioConfig = config.audio;
  const smartLoopEnabled = audioConfig.smart_loop_enabled;

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

      // Active band — translucent orange rect.
      const band = bandRef.current;
      if (band) {
        const x0 = tToX(band.start);
        const x1 = tToX(band.end);
        ctx.fillStyle = "rgba(240, 138, 72, 0.18)";
        ctx.fillRect(Math.min(x0, x1), 0, Math.abs(x1 - x0), h);
        // Edge markers — slightly brighter so the resize hit-zones are
        // findable by eye.
        ctx.fillStyle = "rgba(255, 222, 196, 0.55)";
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

  // Apply / clear band on the worklet whenever the React band state
  // settles to a valid range (or null). Guarded with `typeof === "function"`
  // because a session that started before this code shipped has an
  // AudioPlayer instance whose prototype predates these methods —
  // calling them blind would crash the render and tear the session.
  useEffect(() => {
    if (!hasPlayer || !player) return;
    const setBand = (player as unknown as {
      setLoopBand?: (s: number, e: number) => void;
    }).setLoopBand;
    const clearBand = (player as unknown as {
      clearLoopBand?: () => void;
    }).clearLoopBand;
    // Also mirror the band to the server so the pipeline wraps its decode
    // target inside it (kills the one-window snap-back to old audio at each
    // loop restart). Worklet stays the source of truth for playback; this
    // is purely a generation hint, so a missing/old remote degrades to the
    // prior client-only behaviour.
    const remote = useSessionStore.getState().remote;
    if (
      bandState &&
      bandState.end - bandState.start >= MIN_BAND_SEC &&
      bandState.start >= 0
    ) {
      if (typeof setBand === "function") {
        setBand.call(player, bandState.start, bandState.end);
      }
      remote?.sendLoopBand(bandState.start, bandState.end);
    } else if (bandState === null) {
      if (typeof clearBand === "function") clearBand.call(player);
      remote?.sendLoopBand(null, null);
    }
  }, [bandState, hasPlayer, player]);

  // Smart loop: ask the backend to run PyMusicLooper against its live
  // denoised buffer. The browser only owns the transport and applies the
  // suggested band; loop-point analysis stays in the Python library the
  // feature is based on.
  useEffect(() => {
    const remote = useSessionStore.getState().remote;
    if (!hasPlayer || !player || !remote || !smartLoopEnabled) return;

    const request = () => {
      const anchor = smartAnchorRef.current;
      const current = bandRef.current;
      if (!anchor || !current) return;
      const anchorDuration = anchor.end - anchor.start;
      const minDuration = Math.max(
        MIN_BAND_SEC,
        audioConfig.smart_loop_min_duration_sec,
      );
      if (anchorDuration < minDuration) return;
      const pending = smartPendingRef.current;
      if (pending && performance.now() - pending.sentAt < 8000) return;
      const requestId = ++smartRequestIdRef.current;
      smartPendingRef.current = { id: requestId, sentAt: performance.now() };
      // Search stays anchored to the user's original band. Chaining each
      // analysis from the previous suggestion lets one bad candidate
      // recursively collapse the loop.
      remote.sendSmartLoop(
        requestId,
        anchor.start,
        anchor.end,
        anchorDuration,
        {
          durationFlexPct: audioConfig.smart_loop_duration_flex_pct,
          maxEdgeShiftSec: audioConfig.smart_loop_max_edge_shift_sec,
          minLoopDurationSec: minDuration,
          disablePruning: audioConfig.smart_loop_disable_pruning,
        },
      );
    };

    const onSuggestion = (event: Event) => {
      const msg = (event as CustomEvent<LoopBandSuggestionMessage>).detail;
      if (!msg || msg.request_id !== smartRequestIdRef.current) return;
      smartPendingRef.current = null;
      const current = bandRef.current;
      if (!current) return;
      const anchor = smartAnchorRef.current;
      if (!anchor) return;
      const anchorDuration = anchor.end - anchor.start;
      const nextDuration = msg.end_sec - msg.start_sec;
      const flex = Math.max(0, audioConfig.smart_loop_duration_flex_pct);
      const maxShift = Math.max(0, audioConfig.smart_loop_max_edge_shift_sec);
      if (nextDuration < audioConfig.smart_loop_min_duration_sec) return;
      if (Math.abs(nextDuration - anchorDuration) > anchorDuration * flex) return;
      if (Math.abs(msg.start_sec - anchor.start) > maxShift) return;
      if (Math.abs(msg.end_sec - anchor.end) > maxShift) return;
      const moved =
        Math.abs(msg.start_sec - current.start) > 0.015 ||
        Math.abs(msg.end_sec - current.end) > 0.015;
      if (moved) {
        setBandState({ start: msg.start_sec, end: msg.end_sec });
      }
    };

    remote.addEventListener("loop_band_suggestion", onSuggestion);
    const intervalMs = Math.max(1000, audioConfig.smart_loop_interval_ms);
    const intervalId = window.setInterval(request, intervalMs);
    request();

    return () => {
      remote.removeEventListener("loop_band_suggestion", onSuggestion);
      window.clearInterval(intervalId);
    };
  }, [audioConfig, hasPlayer, player, smartLoopEnabled]);

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

    /** Snap a time to the beat grid unless Alt is held (free adjust). */
    const snapT = (t: number, e: PointerEvent): number =>
      e.altKey ? t : snapToBeat(t, gridRef.current.beatSec);

    const onDown = (e: PointerEvent) => {
      // Right-click → clear band (handled in contextmenu). Ignore here.
      if (e.button !== 0) return;
      // A press on the LOOP toggle (a child of this box) must not also
      // scrub: the native event still bubbles to this listener even
      // though the button calls React's synthetic stopPropagation.
      if (
        (e.target as HTMLElement | null)?.closest?.(".waveform-loop-toggle")
      ) {
        return;
      }

      const t = tFromEvent(e);
      const band = bandRef.current;
      const tol = secPerPx() * BAND_EDGE_HIT_PX;
      // Loop editing is active when the user armed loop mode OR is
      // holding Shift (the power-user shortcut).
      const loopEditing = e.shiftKey || loopModeRef.current;

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
        mode = "draw-band";
        anchorT = snapT(t, e);
        setUserBand({ start: anchorT, end: anchorT });
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
          setUserBand({ start, end });
          return;
        }
        case "move-band": {
          const sb = drag.startBand;
          if (!sb) return;
          const len = sb.end - sb.start;
          const delta = t - drag.anchorT;
          let start = snapT(sb.start + delta, e);
          let end = start + len;
          // Clamp to buffer ends without resizing.
          if (start < 0) {
            start = 0;
            end = len;
          }
          if (end > duration) {
            end = duration;
            start = end - len;
          }
          setUserBand({ start, end });
          return;
        }
        case "resize-band-start": {
          const b = bandRef.current;
          if (!b) return;
          const newStart = Math.min(
            b.end - MIN_BAND_SEC,
            Math.max(0, snapT(t, e)),
          );
          setUserBand({ start: newStart, end: b.end });
          return;
        }
        case "resize-band-end": {
          const b = bandRef.current;
          if (!b) return;
          const newEnd = Math.max(
            b.start + MIN_BAND_SEC,
            Math.min(duration, snapT(t, e)),
          );
          setUserBand({ start: b.start, end: newEnd });
          return;
        }
      }
    };

    const onUp = (e: PointerEvent) => {
      if (!drag) return;
      // Draw mode finalises on release: if the user just tap-clicked
      // (no real drag), kill the band so we don't lock playback to a
      // zero-width sliver.
      if (drag.mode === "draw-band") {
        const b = bandRef.current;
        if (!b || b.end - b.start < MIN_BAND_SEC) {
          setUserBand(null);
        }
      }
      drag = null;
      try {
        box.releasePointerCapture(e.pointerId);
      } catch {}
    };

    const onContextMenu = (e: MouseEvent) => {
      // Right-click clears the band entirely. Operators familiar with
      // DAWs expect "right-click loop marker = remove" so we mirror that
      // without spinning up a real context menu in v1.
      if (bandRef.current) {
        e.preventDefault();
        setUserBand(null);
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
  }, [hasPlayer]);

  // One-shot tooltip on first hover.
  const tipProps = useOneShotTooltip(
    "waveform-scrub",
    "Click to scrub · LOOP to set a loop region",
  );

  // LOOP button: with no loop set, arms loop mode (a plain drag then
  // draws the band). With a loop active, clears it. Mirrors the single
  // on/off mental model operators expect from a loop toggle.
  const loopActive = loopMode || bandState !== null;
  const toggleLoop = () => {
    if (loopActive) {
      setLoopMode(false);
      setUserBand(null);
    } else {
      setLoopMode(true);
    }
  };

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
      data-has-band={bandState ? "true" : undefined}
      data-loop-mode={loopMode ? "true" : undefined}
      data-dd-tooltip-pos="below"
      role="slider"
      aria-label="Scrub playhead"
      {...tipProps}
    >
      <canvas ref={bgCanvasRef} className="waveform-scrub-bg" aria-hidden="true" />
      <canvas ref={fgCanvasRef} className="waveform-scrub-fg" aria-hidden="true" />
      <button
        type="button"
        className="waveform-loop-toggle"
        data-active={loopActive ? "true" : undefined}
        aria-pressed={loopActive}
        title={
          loopActive
            ? "Clear the loop"
            : "Arm loop mode, then drag across the waveform to set a loop"
        }
        onClick={(e) => {
          e.stopPropagation();
          toggleLoop();
        }}
      >
        LOOP
      </button>
    </div>
  );
}
