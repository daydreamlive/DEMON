// Curve interpolation. Given an array of CurvePoints (sorted by x,
// endpoints pinned at x=0 and x=1, all y values in [0,1]), evaluate
// the curve at arbitrary x ∈ [0,1] OR tessellate to N evenly-spaced
// samples for rendering. Per-point `mode` chooses how that point
// connects to the NEXT point: smooth (Catmull-Rom), linear, step.

import type { CurvePoint } from "@/types/curves";

/** Catmull-Rom uniform spline. Given four 1-D control values
 *  (p0, p1, p2, p3) and a parameter `t ∈ [0,1]`, return the
 *  interpolated value between p1 and p2. p0 and p3 shape the slope
 *  at the endpoints — for the first/last segments we duplicate the
 *  edge points to keep the curve smooth without a tail wiggle.
 *  Lifted from rtmg-vst-webapp's CurveEditor.tsx; ~5 lines, well-tested. */
export function catmull(
  p0: number,
  p1: number,
  p2: number,
  p3: number,
  t: number,
): number {
  const t2 = t * t;
  const t3 = t2 * t;
  return (
    0.5 *
    (2 * p1 +
      (-p0 + p2) * t +
      (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2 +
      (-p0 + 3 * p1 - 3 * p2 + p3) * t3)
  );
}

/** Evaluate a curve at x ∈ [0,1]. Returns y ∈ [0,1] (clamped).
 *  Caller should pass at least 2 points with endpoints pinned. */
export function evaluateCurve(points: CurvePoint[], x: number): number {
  if (points.length === 0) return 0;
  if (points.length === 1) return clamp01(points[0].y);
  // Clamp x to the curve's range.
  const xc = Math.min(1, Math.max(0, x));
  if (xc <= points[0].x) return clamp01(points[0].y);
  if (xc >= points[points.length - 1].x)
    return clamp01(points[points.length - 1].y);

  // Find the segment [i, i+1] containing xc. Linear scan is fine —
  // curves are tiny (≤30 points typical).
  let i = 0;
  for (let k = 0; k < points.length - 1; k++) {
    if (xc >= points[k].x && xc <= points[k + 1].x) {
      i = k;
      break;
    }
  }
  const p1 = points[i];
  const p2 = points[i + 1];
  const span = p2.x - p1.x;
  const u = span === 0 ? 0 : (xc - p1.x) / span;

  switch (p1.mode) {
    case "step":
      // Hold p1.y until we reach p2.x, then jump.
      return clamp01(p1.y);
    case "linear":
      return clamp01(p1.y + (p2.y - p1.y) * u);
    case "smooth":
    default: {
      // Catmull-Rom needs 4 control points. Duplicate the edge ones
      // when at the boundary so the curve doesn't overshoot off the
      // ends; that's the standard Catmull behaviour for endpoints.
      const p0 = i > 0 ? points[i - 1] : p1;
      const p3 = i + 2 < points.length ? points[i + 2] : p2;
      return clamp01(catmull(p0.y, p1.y, p2.y, p3.y, u));
    }
  }
}

/** Tessellate the curve to N evenly-spaced samples in [0,1]. Used by
 *  the canvas renderer — sample once, then stroke a polyline through
 *  the samples (fast, no per-pixel evaluate). 256 is plenty for
 *  pixel-smooth display at typical viewport widths. */
export function tessellate(points: CurvePoint[], samples = 256): Float32Array {
  const out = new Float32Array(samples);
  for (let i = 0; i < samples; i++) {
    const x = samples === 1 ? 0 : i / (samples - 1);
    out[i] = evaluateCurve(points, x);
  }
  return out;
}

function clamp01(v: number): number {
  if (v < 0) return 0;
  if (v > 1) return 1;
  return v;
}
