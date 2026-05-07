// Absolute-knob → delta conversion to fix the classic MIDI takeover snap.
//
// The naive mapping `slider = (value/127) * max` snaps the slider to the
// knob's physical position the instant a CC arrives, so a knob parked at
// 127 + a slider currently at 0 produces a min→max jump on the very first
// twist. We instead emit deltas relative to the previous raw value: each
// MIDI message moves the slider by (Δvalue / 127) * max. The knob always
// "tracks" the slider's current position, no matter where it physically is.
//
// Trade-off: absolute correspondence is lost — the knob's physical
// position no longer maps 1:1 to the slider value. Acceptable for a live
// performance UI (FL Studio + most VST hosts default to encoder-like
// behavior), and it sidesteps the secondary bug where decodeKnob's
// auto-detect misclassifies a barely-turned absolute knob as relative
// when its history sits in the 0-4 / 123-127 zone.

const lastValue = new Map<number, number>();

// Cap delta at this many raw ticks per message. Fast spins on a real
// hardware knob can produce 30-60 ticks between two CC frames; anything
// bigger is a state-mismatch (e.g. operator manually repositioned the
// knob while we weren't listening). We CLAMP rather than DROP so a fast
// sweep still moves the slider — the prior "drop on |delta|>cap" rule
// produced dead zones near the knob extremes when the user spun fast.
const MAX_DELTA_TICKS = 64;

/** Knob position considered "at the extreme" — useMidi snaps the slider
 *  hard to min or max in this band so a fast sweep that ends at 0 or
 *  127 always reaches the bound, even if the prior delta-tracked value
 *  was off. */
export const EXTREME_LOW_THRESHOLD = 1;
export const EXTREME_HIGH_THRESHOLD = 126;

export interface KnobReading {
  /** When non-null, caller should setSlider(value/127 * max) — the knob
   *  is parked at an extreme and the slider should snap there. */
  absolute: number | null;
  /** Otherwise, caller bumpSlider(delta/127 * max). null on the first
   *  CC frame for this knob (no prior value to delta against). */
  delta: number | null;
}

/** Decide what the slider should do for this MIDI value. */
export function readKnob(cc: number, value: number): KnobReading {
  const last = lastValue.get(cc);
  lastValue.set(cc, value);

  // Snap to extremes regardless of prior history. Without this, a fast
  // sweep from mid-range to 0 might leave the slider at 0.07 because
  // delta clamping limited each step.
  if (value <= EXTREME_LOW_THRESHOLD) {
    return { absolute: 0, delta: null };
  }
  if (value >= EXTREME_HIGH_THRESHOLD) {
    return { absolute: 127, delta: null };
  }

  if (last === undefined) return { absolute: null, delta: null };
  const raw = value - last;
  // Clamp instead of drop. A fast sweep still produces motion in the
  // right direction; the next message refines.
  const clamped = Math.max(-MAX_DELTA_TICKS, Math.min(MAX_DELTA_TICKS, raw));
  return { absolute: null, delta: clamped };
}

/** Legacy compat — useMidi previously called `knobDelta` and applied
 *  delta math only. Kept for callers that don't yet handle the
 *  absolute-snap path. */
export function knobDelta(cc: number, value: number): number | null {
  const r = readKnob(cc, value);
  return r.delta;
}

/** Drop the cached previous value for one CC (or all). Use when the
 *  operator rebinds a CC to a different slider — otherwise the first
 *  post-rebind message would be interpreted as a delta from the previous
 *  slider's last value. */
export function resetKnobDelta(cc?: number): void {
  if (cc === undefined) lastValue.clear();
  else lastValue.delete(cc);
}
