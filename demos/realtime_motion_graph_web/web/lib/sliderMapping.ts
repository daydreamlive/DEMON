// Visual↔value translation for vertical sliders. Two modes:
//
//  1. Linear (default — `unity` omitted). The slider's internal
//     fraction `t` ∈ [0, 1] (0 = bottom of rail, 1 = top) maps linearly
//     onto [min, max]. With `reverse`, top of rail = min, bottom = max.
//
//  2. Unity-anchored (`unity` provided). Treats the rail as visual
//     [0, 2] with the midpoint pinned to `unity` (typically 1.0). The
//     lower half of the rail [t ∈ 0..0.5] covers values [min, unity];
//     the upper half [t ∈ 0.5..1] covers [unity, max]. Each half is
//     linear within itself, so the slopes above and below the anchor
//     can differ when the configured range isn't symmetric around
//     unity (e.g., [0, 2.5] with unity=1.0: lower half slope = 1.0/0.5
//     = 2.0, upper half = 1.5/0.5 = 3.0). With `reverse`, the rail
//     flips: top of rail = min, midpoint = unity, bottom = max — so
//     "drag UP" still moves the engine value DOWN.
//
//  Why unity-anchored mode exists: lets a bank of sliders with
//  different per-channel [min, max] caps display the same default
//  value (the unity point) at the same visual rail height, so the bank
//  reads at a glance, while each channel still uses its full
//  configured range above and below unity.

export interface SliderMapping {
  min: number;
  max: number;
  /** When set, anchors this value to the midpoint of the rail. The
   *  half above unity covers [unity, max]; the half below covers
   *  [min, unity]. Falls back to linear mapping when omitted, or when
   *  unity isn't strictly between min and max. */
  unity?: number;
  reverse?: boolean;
}

function isUnityAnchored(
  m: SliderMapping,
): m is SliderMapping & { unity: number } {
  // Both half-ranges must be non-degenerate; otherwise fall back to
  // linear so we don't divide by zero on a misconfigured channel.
  return (
    typeof m.unity === "number" &&
    m.max > m.unity &&
    m.unity > m.min
  );
}

const clamp01 = (x: number) => Math.max(0, Math.min(1, x));

/** Engine value → slider thumb fraction t ∈ [0, 1]. */
export function valueToT(value: number, m: SliderMapping): number {
  const reverse = !!m.reverse;
  if (isUnityAnchored(m)) {
    let t: number;
    if (!reverse) {
      t =
        value <= m.unity
          ? (value - m.min) / (m.unity - m.min) / 2
          : 0.5 + (value - m.unity) / (m.max - m.unity) / 2;
    } else {
      t =
        value >= m.unity
          ? (m.max - value) / (m.max - m.unity) / 2
          : 0.5 + (m.unity - value) / (m.unity - m.min) / 2;
    }
    return clamp01(t);
  }
  const span = Math.max(0, m.max - m.min);
  const frac = span > 0 ? (value - m.min) / span : 0;
  return clamp01(reverse ? 1 - frac : frac);
}

/** Slider thumb fraction t ∈ [0, 1] → engine value. */
export function tToValue(t: number, m: SliderMapping): number {
  const reverse = !!m.reverse;
  const c = clamp01(t);
  if (isUnityAnchored(m)) {
    if (!reverse) {
      return c <= 0.5
        ? m.min + 2 * c * (m.unity - m.min)
        : m.unity + (2 * c - 1) * (m.max - m.unity);
    }
    return c <= 0.5
      ? m.max - 2 * c * (m.max - m.unity)
      : m.unity - (2 * c - 1) * (m.unity - m.min);
  }
  const span = Math.max(0, m.max - m.min);
  const fwd = reverse ? 1 - c : c;
  return m.min + fwd * span;
}
