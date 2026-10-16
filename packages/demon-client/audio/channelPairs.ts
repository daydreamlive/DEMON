// Channel-pair helpers for multichannel buffers.
//
// A buffer with more than two channels is read as stereo pairs side by
// side: channels [2k, 2k+1] = pair k. The player mixes the pairs down to
// stereo with one gain per pair (missing entries = 1). The audio worklet
// carries its own copy of this sum (it cannot import modules); these
// helpers serve the ScriptProcessor fallback and the loudness matcher.

/** Gain for pair `k`: the caller's value, or unity when absent. */
export function pairGain(gains: ArrayLike<number>, k: number): number {
  return k < gains.length ? gains[k] : 1;
}

/** Sanitize a caller's gain list: non-finite entries become 1, negatives 0. */
export function sanitizeChannelGains(gains: ArrayLike<number>): number[] {
  const out: number[] = [];
  for (let k = 0; k < gains.length; k++) {
    const g = Number(gains[k]);
    out.push(Number.isFinite(g) ? Math.max(0, g) : 1);
  }
  return out;
}

/**
 * Mix an interleaved N-channel buffer (N > 2, pairs side by side) to an
 * interleaved stereo buffer: out[f, c] = sum_k gain_k * in[f, 2k + c].
 * A trailing odd channel is ignored.
 */
export function mixChannelPairs(
  interleaved: Float32Array,
  channels: number,
  gains: ArrayLike<number>,
): Float32Array {
  const frames = (interleaved.length / channels) | 0;
  const nPairs = channels >> 1;
  const out = new Float32Array(frames * 2);
  for (let k = 0; k < nPairs; k++) {
    const g = pairGain(gains, k);
    if (g === 0) continue;
    const off = 2 * k;
    for (let f = 0; f < frames; f++) {
      const src = f * channels + off;
      out[2 * f] += g * interleaved[src];
      out[2 * f + 1] += g * interleaved[src + 1];
    }
  }
  return out;
}

/**
 * One step of the single-pole gain smoother shared by the worklet and
 * the ScriptProcessor fallback: each live gain moves `alpha` of the way
 * to its target (missing targets = 1). Mutates and returns `live`.
 */
export function smoothPairGains(
  live: Float32Array,
  targets: ArrayLike<number>,
  alpha: number,
): Float32Array {
  for (let k = 0; k < live.length; k++) {
    live[k] += (pairGain(targets, k) - live[k]) * alpha;
  }
  return live;
}
