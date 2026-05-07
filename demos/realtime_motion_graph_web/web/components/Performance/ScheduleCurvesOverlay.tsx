"use client";

import { useEffect, useRef, useState } from "react";

import {
  buildPreset,
  PRESET_LABEL,
  type CurvePresetId,
} from "@/engine/curves/presets";
import { tessellate } from "@/engine/curves/interp";
import {
  computePeaks,
  drawPeaks,
} from "@/engine/curves/waveformPeaks";
import { useCurveStore } from "@/store/useCurveStore";
import { useLoraStore } from "@/store/useLoraStore";
import { usePerformanceStore } from "@/store/usePerformanceStore";
import { useSessionStore } from "@/store/useSessionStore";
import {
  type CurveInterpMode,
  type CurvePoint,
  loraCurveParam,
  MAX_LORA_CURVES,
  SCHEDULEABLE_PARAMS,
  SCHEDULEABLE_PARAM_LABEL,
} from "@/types/curves";

// ScheduleCurvesOverlay — full curve scheduler. Mounts inside
// #graph-wrap, gates on useCurveStore.overlayOpen.
//
// Three layered canvases (in DOM source order, painted bottom-up):
//   1. .schedule-curves-bg     — waveform underlay (computed once
//      per fixture-swap, redrawn on resize).
//   2. .schedule-curves-canvas — curve, control points, playhead.
//      Redrawn on every rAF tick (cheap; we tessellate to ~256
//      samples and stroke a polyline) so the playhead animates
//      smoothly. Also handles all pointer interaction.
//
// Interaction model (rtmg-vst lineage):
//   - Click empty space → insert smooth point at click position.
//   - Drag a point → move it; first/last points constrained to x
//     endpoints (y still free).
//   - Right-click point → cycle smooth → linear → step → smooth.
//   - Delete / Backspace on hovered point → delete (endpoints exempt).
//   - Right-click a tab → context menu: enable/disable, reset, presets.

const WAVEFORM_BUCKETS_DESIRED = 720;
const POINT_HIT_RADIUS_PX = 12;
const CURVE_TESSELLATION = 256;

function ensureCanvasSize(
  canvas: HTMLCanvasElement,
): { w: number; h: number; dpr: number } {
  const rect = canvas.getBoundingClientRect();
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const w = Math.max(1, Math.round(rect.width));
  const h = Math.max(1, Math.round(rect.height));
  if (canvas.width !== w * dpr || canvas.height !== h * dpr) {
    canvas.width = w * dpr;
    canvas.height = h * dpr;
  }
  return { w, h, dpr };
}

function nextInterpMode(m: CurveInterpMode): CurveInterpMode {
  if (m === "smooth") return "linear";
  if (m === "linear") return "step";
  return "smooth";
}

interface PresetMenuState {
  param: string;
  /** Distance from overlay's left edge to the menu's left edge. */
  left: number;
  /** Distance from overlay's bottom edge to the menu's bottom edge. The
   *  menu grows UPWARD from this anchor (above the tab strip), so it
   *  always fits within the overlay no matter where the user
   *  right-clicks. */
  bottom: number;
}

export function ScheduleCurvesOverlay() {
  const open = useCurveStore((s) => s.overlayOpen);
  const closeOverlay = useCurveStore((s) => s.closeOverlay);
  const activeCurve = useCurveStore((s) => s.activeCurve);
  const setActiveCurve = useCurveStore((s) => s.setActiveCurve);
  const curves = useCurveStore((s) => s.curves);
  const setCurvePoints = useCurveStore((s) => s.setCurvePoints);
  const setCurveEnabled = useCurveStore((s) => s.setCurveEnabled);
  const resetCurve = useCurveStore((s) => s.resetCurve);
  const ensureCurve = useCurveStore((s) => s.ensureCurve);
  const scheduleEnabled = useCurveStore((s) => s.scheduleEnabled);
  const toggleScheduleEnabled = useCurveStore(
    (s) => s.toggleScheduleEnabled,
  );
  const detectedBpm = usePerformanceStore((s) => s.detectedBpm);

  // First N enabled LoRAs become dynamic tabs in the strip. Set
  // preserves insertion order, so [...enabled].slice(0, MAX) gives the
  // user's most-recently-enabled LoRAs as their curve targets. The
  // catalog provides the human-readable name for each id.
  const loraEnabled = useLoraStore((s) => s.enabled);
  const loraCatalog = useLoraStore((s) => s.catalog);
  const loraTabs = (() => {
    const ids = Array.from(loraEnabled).slice(0, MAX_LORA_CURVES);
    return ids.map((id) => ({
      id,
      param: loraCurveParam(id),
      label:
        loraCatalog.find((e) => e.id === id)?.name?.toUpperCase() ??
        id.toUpperCase(),
    }));
  })();

  // Helper to look up a tab's display label whether it's a fixed
  // schedulable param OR a LoRA. Used for tab buttons + preset menu
  // header.
  const labelFor = (param: string): string => {
    if (param in SCHEDULEABLE_PARAM_LABEL) {
      return SCHEDULEABLE_PARAM_LABEL[
        param as keyof typeof SCHEDULEABLE_PARAM_LABEL
      ];
    }
    const tab = loraTabs.find((t) => t.param === param);
    return tab?.label ?? param;
  };

  const overlayRef = useRef<HTMLDivElement | null>(null);
  const bgCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const fgCanvasRef = useRef<HTMLCanvasElement | null>(null);

  const [presetMenu, setPresetMenu] = useState<PresetMenuState | null>(null);

  // Per-render snapshot of the active curve's points. We keep a ref
  // alongside the store value so canvas event handlers see the latest
  // without stale-closure pitfalls. Dynamic params (LoRA curves) may
  // not exist yet; ensureCurve allocates a fresh default if missing.
  const pointsRef = useRef<CurvePoint[]>(
    curves[activeCurve]?.points ?? [
      { x: 0, y: 0.5, mode: "smooth" },
      { x: 1, y: 0.5, mode: "smooth" },
    ],
  );
  useEffect(() => {
    ensureCurve(activeCurve);
    const c = useCurveStore.getState().curves[activeCurve];
    if (c) pointsRef.current = c.points;
  }, [activeCurve, curves, ensureCurve]);

  // ESC closes the overlay (or the preset menu, if open).
  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key !== "Escape") return;
      if (presetMenu) setPresetMenu(null);
      else closeOverlay();
    };
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, [open, closeOverlay, presetMenu]);

  // Close the preset menu on any click outside it.
  useEffect(() => {
    if (!presetMenu) return;
    const onClick = (e: MouseEvent) => {
      const target = e.target as HTMLElement;
      if (!target.closest(".schedule-curves-preset-menu")) setPresetMenu(null);
    };
    document.addEventListener("mousedown", onClick);
    return () => document.removeEventListener("mousedown", onClick);
  }, [presetMenu]);

  // Waveform underlay (bg canvas).
  useEffect(() => {
    if (!open) return;
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
      // Very low opacity — the moving-lines graph behind needs to stay
      // visible. The waveform is a reference layer, not the focal point.
      ctx.fillStyle = "rgba(240, 138, 72, 0.08)";
      drawPeaks(ctx, peaks, w, h);
      ctx.fillStyle = "rgba(255, 255, 255, 0.05)";
      drawPeaks(ctx, peaks, w, h);
    };

    const recompute = () => {
      const player = useSessionStore.getState().player;
      if (!player) {
        peaks = null;
      } else {
        const mirror = player.getMirror();
        peaks = computePeaks(mirror, player.channels, WAVEFORM_BUCKETS_DESIRED);
      }
      redraw();
    };

    recompute();

    const ro = new ResizeObserver(redraw);
    ro.observe(canvas);

    const player = useSessionStore.getState().player;
    const unsubMirror = player?.onMirrorChange?.(() => recompute()) ?? (() => {});
    const unsubSession = useSessionStore.subscribe((s, prev) => {
      if (s.player !== prev.player) recompute();
    });

    return () => {
      ro.disconnect();
      unsubMirror();
      unsubSession();
    };
  }, [open]);

  // Foreground canvas: curve, control points, playhead. Redrawn every
  // rAF tick so the playhead animates smoothly. Pointer interaction is
  // also wired here.
  useEffect(() => {
    if (!open) return;
    const canvas = fgCanvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    let raf = 0;
    let cancelled = false;

    // Pointer state — kept in closure so it survives across rAF ticks.
    let dragIndex: number | null = null;
    let hoverIndex: number | null = null;

    const sizeMeta = () => ensureCanvasSize(canvas);

    /** Convert a pointer event's clientX/Y to canvas-local pixels. */
    const eventToLocal = (e: PointerEvent | MouseEvent) => {
      const rect = canvas.getBoundingClientRect();
      return {
        x: e.clientX - rect.left,
        y: e.clientY - rect.top,
      };
    };

    /** Convert pixel coords → curve-space (x ∈ [0,1], y ∈ [0,1]).
     *  Y is inverted (top of canvas = y=1, bottom = y=0). */
    const localToCurve = (px: number, py: number, w: number, h: number) => ({
      x: Math.min(1, Math.max(0, px / w)),
      y: Math.min(1, Math.max(0, 1 - py / h)),
    });

    /** Hit-test a click against existing points. Returns the point's
     *  index in `pointsRef.current`, or null if no point under the
     *  cursor within POINT_HIT_RADIUS_PX. */
    const hitTestPoint = (
      px: number,
      py: number,
      w: number,
      h: number,
    ): number | null => {
      const points = pointsRef.current;
      let bestIdx: number | null = null;
      let bestDist = POINT_HIT_RADIUS_PX * POINT_HIT_RADIUS_PX;
      for (let i = 0; i < points.length; i++) {
        const cx = points[i].x * w;
        const cy = (1 - points[i].y) * h;
        const dx = cx - px;
        const dy = cy - py;
        const d2 = dx * dx + dy * dy;
        if (d2 < bestDist) {
          bestDist = d2;
          bestIdx = i;
        }
      }
      return bestIdx;
    };

    // ── Render frame ─────────────────────────────────────────────────
    const draw = () => {
      if (cancelled) return;
      const { w, h, dpr } = sizeMeta();
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, w, h);

      const points = pointsRef.current;

      // Grid: horizontal thirds + dashed midline. Helps users eyeball
      // where 0.5 / 0.33 / 0.67 sit on the y axis.
      ctx.strokeStyle = "rgba(255, 255, 255, 0.06)";
      ctx.lineWidth = 1;
      for (const yFrac of [0.25, 0.5, 0.75]) {
        const y = h - yFrac * h;
        ctx.beginPath();
        ctx.moveTo(0, y);
        ctx.lineTo(w, y);
        ctx.stroke();
      }

      // Curve — tessellate to N samples and stroke a polyline.
      const samples = tessellate(points, CURVE_TESSELLATION);
      ctx.strokeStyle = "rgba(240, 138, 72, 0.95)";
      ctx.lineWidth = 2;
      ctx.shadowColor = "rgba(240, 138, 72, 0.55)";
      ctx.shadowBlur = 8;
      ctx.beginPath();
      for (let i = 0; i < samples.length; i++) {
        const x = (i / (samples.length - 1)) * w;
        const y = (1 - samples[i]) * h;
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.stroke();
      ctx.shadowBlur = 0;

      // Control points — circle (smooth), diamond (linear), square (step).
      for (let i = 0; i < points.length; i++) {
        const p = points[i];
        const cx = p.x * w;
        const cy = (1 - p.y) * h;
        const isHover = hoverIndex === i;
        const isDrag = dragIndex === i;
        const r = isHover || isDrag ? 7 : 5;

        ctx.fillStyle = isDrag
          ? "rgba(255, 255, 255, 1)"
          : isHover
            ? "rgba(255, 230, 200, 0.95)"
            : "rgba(240, 138, 72, 1)";
        ctx.strokeStyle = "rgba(0, 0, 0, 0.6)";
        ctx.lineWidth = 1.5;

        ctx.beginPath();
        if (p.mode === "smooth") {
          ctx.arc(cx, cy, r, 0, Math.PI * 2);
        } else if (p.mode === "linear") {
          // diamond
          ctx.moveTo(cx, cy - r);
          ctx.lineTo(cx + r, cy);
          ctx.lineTo(cx, cy + r);
          ctx.lineTo(cx - r, cy);
          ctx.closePath();
        } else {
          // square
          ctx.rect(cx - r, cy - r, r * 2, r * 2);
        }
        ctx.fill();
        ctx.stroke();
      }

      // Playhead — vertical orange line at currentPositionSec / duration.
      const session = useSessionStore.getState();
      const player = session.player;
      const remote = session.remote;
      if (player && remote && remote.duration > 0) {
        const t = Math.min(1, Math.max(0, player.positionSec / remote.duration));
        const xp = t * w;
        ctx.strokeStyle = "rgba(240, 138, 72, 0.85)";
        ctx.lineWidth = 1.5;
        ctx.shadowColor = "rgba(240, 138, 72, 0.6)";
        ctx.shadowBlur = 6;
        ctx.beginPath();
        ctx.moveTo(xp, 0);
        ctx.lineTo(xp, h);
        ctx.stroke();
        ctx.shadowBlur = 0;
      }

      raf = requestAnimationFrame(draw);
    };
    raf = requestAnimationFrame(draw);

    // ── Pointer events ───────────────────────────────────────────────
    const onPointerDown = (e: PointerEvent) => {
      // Block right-click here; handled in oncontextmenu below.
      if (e.button === 2) return;
      const { w, h } = sizeMeta();
      const { x: px, y: py } = eventToLocal(e);
      const hit = hitTestPoint(px, py, w, h);
      if (hit !== null) {
        dragIndex = hit;
        canvas.setPointerCapture(e.pointerId);
        return;
      }
      // Click on empty area → insert a new smooth point at that x.
      const { x, y } = localToCurve(px, py, w, h);
      const points = pointsRef.current.slice();
      // Don't insert at the very edges — those are pinned endpoints.
      if (x <= 0.001 || x >= 0.999) return;
      const newPoint: CurvePoint = { x, y, mode: "smooth" };
      points.push(newPoint);
      points.sort((a, b) => a.x - b.x);
      pointsRef.current = points;
      setCurvePoints(activeCurve, points);
    };

    const onPointerMove = (e: PointerEvent) => {
      const { w, h } = sizeMeta();
      const { x: px, y: py } = eventToLocal(e);
      if (dragIndex !== null) {
        const points = pointsRef.current.slice();
        const i = dragIndex;
        const { x, y } = localToCurve(px, py, w, h);
        // Endpoint x is pinned; midpoint x must stay strictly between
        // its neighbours so the curve stays well-formed.
        const isFirst = i === 0;
        const isLast = i === points.length - 1;
        const newX = isFirst ? 0 : isLast ? 1 : Math.min(
          Math.max(x, points[i - 1].x + 0.001),
          points[i + 1].x - 0.001,
        );
        points[i] = { ...points[i], x: newX, y };
        pointsRef.current = points;
        // Throttle store writes by RAF cadence — pointermove fires
        // ~mouse-event-rate; the canvas re-renders from pointsRef
        // each frame anyway. Storing every move keeps undo / persist
        // honest at the cost of one set per move; cheap.
        setCurvePoints(activeCurve, points);
        return;
      }
      // Hover hit-test for visual feedback only.
      const hit = hitTestPoint(px, py, w, h);
      if (hit !== hoverIndex) hoverIndex = hit;
    };

    const onPointerUp = (e: PointerEvent) => {
      if (dragIndex !== null) {
        canvas.releasePointerCapture(e.pointerId);
        dragIndex = null;
      }
    };

    const onContextMenu = (e: MouseEvent) => {
      e.preventDefault();
      const { w, h } = sizeMeta();
      const { x: px, y: py } = eventToLocal(e);
      const hit = hitTestPoint(px, py, w, h);
      if (hit === null) return;
      const points = pointsRef.current.slice();
      points[hit] = {
        ...points[hit],
        mode: nextInterpMode(points[hit].mode),
      };
      pointsRef.current = points;
      setCurvePoints(activeCurve, points);
    };

    const onKeyDown = (e: KeyboardEvent) => {
      if (e.key !== "Delete" && e.key !== "Backspace") return;
      if (hoverIndex === null) return;
      const points = pointsRef.current;
      // Endpoints can't be deleted; they're pinned at x=0 and x=1.
      if (hoverIndex === 0 || hoverIndex === points.length - 1) return;
      const next = points.filter((_, i) => i !== hoverIndex);
      pointsRef.current = next;
      hoverIndex = null;
      setCurvePoints(activeCurve, next);
    };

    canvas.addEventListener("pointerdown", onPointerDown);
    canvas.addEventListener("pointermove", onPointerMove);
    canvas.addEventListener("pointerup", onPointerUp);
    canvas.addEventListener("pointercancel", onPointerUp);
    canvas.addEventListener("contextmenu", onContextMenu);
    document.addEventListener("keydown", onKeyDown);

    return () => {
      cancelled = true;
      cancelAnimationFrame(raf);
      canvas.removeEventListener("pointerdown", onPointerDown);
      canvas.removeEventListener("pointermove", onPointerMove);
      canvas.removeEventListener("pointerup", onPointerUp);
      canvas.removeEventListener("pointercancel", onPointerUp);
      canvas.removeEventListener("contextmenu", onContextMenu);
      document.removeEventListener("keydown", onKeyDown);
    };
  }, [open, activeCurve, setCurvePoints]);

  if (!open) return null;

  // Tab right-click → preset menu. Anchored to the tab button itself
  // and grows UPWARD from above the tab strip; the overlay clips at
  // its bounds, so positioning at the click point would clip the menu
  // off the bottom edge.
  const onTabContext = (
    e: React.MouseEvent<HTMLButtonElement>,
    param: string,
  ) => {
    e.preventDefault();
    const overlay = overlayRef.current;
    if (!overlay) return;
    const orect = overlay.getBoundingClientRect();
    const trect = e.currentTarget.getBoundingClientRect();
    setPresetMenu({
      param,
      // Align left edge with the tab's left edge.
      left: trect.left - orect.left,
      // Anchor the menu's bottom 6px above the tab's top edge so it
      // grows upward inside the overlay.
      bottom: orect.bottom - trect.top + 6,
    });
  };

  const applyPresetFromMenu = (preset: CurvePresetId) => {
    if (!presetMenu) return;
    const session = useSessionStore.getState();
    const duration = session.remote?.duration ?? 60;
    const points = buildPreset(preset, {
      durationSec: duration,
      bpm: detectedBpm,
    });
    setCurvePoints(presetMenu.param, points);
    setActiveCurve(presetMenu.param);
    setPresetMenu(null);
  };

  return (
    <div ref={overlayRef} className="schedule-curves-overlay" role="dialog">
      <canvas
        ref={bgCanvasRef}
        className="schedule-curves-bg"
        aria-hidden="true"
      />
      <canvas ref={fgCanvasRef} className="schedule-curves-canvas" />
      <div className="schedule-curves-tabs">
        {SCHEDULEABLE_PARAMS.map((param) => {
          const isActive = param === activeCurve;
          const isEnabled = curves[param]?.enabled ?? false;
          return (
            <button
              key={param}
              type="button"
              className={
                "schedule-curves-tab" +
                (isActive ? " schedule-curves-tab--active" : "") +
                (isEnabled ? " schedule-curves-tab--enabled" : "")
              }
              onClick={() => setActiveCurve(param)}
              onContextMenu={(e) => onTabContext(e, param)}
              title={`${SCHEDULEABLE_PARAM_LABEL[param]} — ${
                isEnabled ? "active" : "drawn but not driving"
              } (right-click for presets)`}
            >
              {SCHEDULEABLE_PARAM_LABEL[param]}
            </button>
          );
        })}
        {loraTabs.map((tab) => {
          const isActive = tab.param === activeCurve;
          const isEnabled = curves[tab.param]?.enabled ?? false;
          return (
            <button
              key={tab.param}
              type="button"
              className={
                "schedule-curves-tab schedule-curves-tab--lora" +
                (isActive ? " schedule-curves-tab--active" : "") +
                (isEnabled ? " schedule-curves-tab--enabled" : "")
              }
              onClick={() => {
                ensureCurve(tab.param);
                setActiveCurve(tab.param);
              }}
              onContextMenu={(e) => {
                ensureCurve(tab.param);
                onTabContext(e, tab.param);
              }}
              title={`LoRA ${tab.label} — ${
                isEnabled ? "active" : "drawn but not driving"
              } (right-click for presets)`}
            >
              {tab.label}
            </button>
          );
        })}
        {/* Master enable / kill switch — flips ALL curves off without
            losing per-curve drawings. The application loop checks this
            first; when off, no slider gets driven. */}
        <button
          type="button"
          className={
            "schedule-curves-master" +
            (scheduleEnabled ? " schedule-curves-master--on" : "")
          }
          onClick={toggleScheduleEnabled}
          title={
            scheduleEnabled
              ? "Curves are driving sliders. Click to pause all automation."
              : "All curves paused. Click to resume."
          }
        >
          {scheduleEnabled ? "ON" : "OFF"}
        </button>
        <button
          type="button"
          className="schedule-curves-close"
          onClick={closeOverlay}
          aria-label="Close schedule curves"
          title="Close (Esc)"
        >
          ×
        </button>
      </div>

      {presetMenu && (
        <div
          className="schedule-curves-preset-menu"
          style={{ left: presetMenu.left, bottom: presetMenu.bottom }}
          onMouseDown={(e) => e.stopPropagation()}
        >
          <div className="schedule-curves-preset-header">
            {labelFor(presetMenu.param)}
          </div>
          <button
            type="button"
            className="schedule-curves-preset-item"
            onClick={() => {
              const enabled = curves[presetMenu.param]?.enabled ?? false;
              setCurveEnabled(presetMenu.param, !enabled);
              setPresetMenu(null);
            }}
          >
            {curves[presetMenu.param]?.enabled ? "Disable" : "Enable"}
          </button>
          <button
            type="button"
            className="schedule-curves-preset-item"
            onClick={() => {
              resetCurve(presetMenu.param);
              setPresetMenu(null);
            }}
          >
            Reset
          </button>
          <div className="schedule-curves-preset-divider" />
          {(Object.keys(PRESET_LABEL) as CurvePresetId[]).map((preset) => (
            <button
              key={preset}
              type="button"
              className="schedule-curves-preset-item"
              onClick={() => applyPresetFromMenu(preset)}
            >
              {PRESET_LABEL[preset]}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}
