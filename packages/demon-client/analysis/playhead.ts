// Mapping the player's read position to the audible playhead, the time an
// analysis consumer should draw or schedule against.

import type { LoopBand } from "./types";

/**
 * Audible playhead in canvas seconds.
 *
 * `positionSec` is `AudioPlayer.positionSec`, the worklet's read position;
 * what the listener hears lags it by the output latency
 * (`AudioContext.outputLatency`, plus `baseLatency` if the caller wants it).
 * Subtracting the latency can step back across a loop seam, so the result
 * is wrapped into `loopBand` when the read position lies inside it, else
 * into [0, durationSec) when a canvas duration is given (the player loops
 * the whole canvas), else clamped at 0.
 */
export function playheadForNotes(
  positionSec: number,
  outputLatencySec: number,
  loopBand?: LoopBand | null,
  durationSec?: number,
): number {
  const latency =
    Number.isFinite(outputLatencySec) && outputLatencySec > 0
      ? outputLatencySec
      : 0;
  const t = positionSec - latency;
  if (
    loopBand &&
    loopBand.end_s > loopBand.start_s &&
    positionSec >= loopBand.start_s &&
    positionSec < loopBand.end_s
  ) {
    const period = loopBand.end_s - loopBand.start_s;
    return loopBand.start_s + mod(t - loopBand.start_s, period);
  }
  if (durationSec !== undefined && durationSec > 0) return mod(t, durationSec);
  return Math.max(0, t);
}

function mod(x: number, m: number): number {
  return ((x % m) + m) % m;
}
