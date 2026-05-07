// Downsample an interleaved Float32 PCM buffer to a peaks array suitable
// for drawing a static waveform silhouette. Output is alternating
// (min, max) per pixel column — N output buckets × 2 floats each.
//
// Usage:
//   const mirror = audioPlayer.getMirror();
//   const peaks  = computePeaks(mirror, 2 /* channels */, 800 /* px wide */);
//   drawPeaks(ctx, peaks, 800, height);
//
// The bg canvas of ScheduleCurvesOverlay calls this once per fixture-
// swap (mirror change) — not per-frame — and caches the result.

/** Returns a Float32Array of length `buckets * 2` with alternating
 *  min, max per bucket. Channels are averaged before bucketing. */
export function computePeaks(
  interleaved: Float32Array | null,
  channels: number,
  buckets: number,
): Float32Array {
  const out = new Float32Array(buckets * 2);
  if (!interleaved || channels <= 0 || buckets <= 0) return out;
  const frameCount = interleaved.length / channels;
  if (frameCount <= 0) return out;
  const framesPerBucket = Math.max(1, Math.floor(frameCount / buckets));

  for (let b = 0; b < buckets; b++) {
    const startFrame = b * framesPerBucket;
    let endFrame = startFrame + framesPerBucket;
    if (b === buckets - 1) endFrame = frameCount; // last bucket eats remainder
    let mn = Infinity;
    let mx = -Infinity;
    for (let f = startFrame; f < endFrame; f++) {
      // Average across channels.
      let sum = 0;
      const offset = f * channels;
      for (let c = 0; c < channels; c++) sum += interleaved[offset + c];
      const v = sum / channels;
      if (v < mn) mn = v;
      if (v > mx) mx = v;
    }
    if (!Number.isFinite(mn)) mn = 0;
    if (!Number.isFinite(mx)) mx = 0;
    out[b * 2] = mn;
    out[b * 2 + 1] = mx;
  }
  return out;
}

/** Stroke the peaks array as a centered silhouette spanning the canvas
 *  width. Caller sets fillStyle / strokeStyle on the context. */
export function drawPeaks(
  ctx: CanvasRenderingContext2D,
  peaks: Float32Array,
  w: number,
  h: number,
): void {
  const buckets = peaks.length / 2;
  if (buckets <= 0 || w <= 0 || h <= 0) return;
  const mid = h / 2;
  const colW = w / buckets;
  ctx.beginPath();
  // Top edge — max envelope.
  for (let b = 0; b < buckets; b++) {
    const x = b * colW;
    const mx = peaks[b * 2 + 1];
    const y = mid - mx * mid;
    if (b === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  }
  // Bottom edge — min envelope, traversed right-to-left.
  for (let b = buckets - 1; b >= 0; b--) {
    const x = b * colW;
    const mn = peaks[b * 2];
    const y = mid - mn * mid;
    ctx.lineTo(x, y);
  }
  ctx.closePath();
  ctx.fill();
}
