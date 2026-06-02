// Relative-encoder decoding for MIDI CC.
//
// Endless encoders don't send an absolute position — they send a signed
// delta per detent in 2's-complement / binary-offset form. Absolute
// knobs/faders are handled directly in useMidi (value/127 → range); there
// is no auto-detect and no per-CC delta tracking, so a knob's physical
// position always corresponds 1:1 to the parameter.

/** One MIDI-input tick (relative-encoder detent, or one scroll-wheel
 *  notch) expressed as a fraction of the slider rail (T-space). MIDI
 *  relative mode and the SliderGroup / Knob wheel handlers all step by
 *  this so a knob detent and a scroll notch move a parameter by exactly
 *  the same visible amount, through whatever mapping the control uses. */
export const MIDI_TICK_T = 0.01;

/** Cap on relative ticks honoured per message. A corrupt / runaway frame
 *  can't sweep the rail end-to-end in one step. */
const MAX_RELATIVE_TICKS = 32;

/** Decode a relative-encoder CC value into a signed tick count.
 *  2's-complement / binary-offset: 1..63 = +1..+63 (CW), 65..127 =
 *  -63..-1 (CCW), 0 and 64 = no motion. Pure — no per-CC state. */
export function decodeRelativeDelta(value: number): number {
  let delta = 0;
  if (value > 0 && value < 64) delta = value;
  else if (value > 64 && value <= 127) delta = value - 128;
  return Math.max(-MAX_RELATIVE_TICKS, Math.min(MAX_RELATIVE_TICKS, delta));
}

