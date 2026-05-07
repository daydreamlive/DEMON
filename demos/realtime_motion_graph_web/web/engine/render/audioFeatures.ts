// Cheap kick / RMS extraction from an interleaved float32 buffer at a given
// playhead position. Used to drive the bloom amount, graph pulse, and
// effects shader uniform.

const KICK_WINDOW_FRAMES = 480; // ~10 ms at 48 kHz

export function kickRms(
  interleaved: Float32Array | null,
  positionSec: number,
  durationSec: number,
  channels: number,
): number {
  if (!interleaved || interleaved.length === 0 || durationSec <= 0) return 0;
  const totalFrames = interleaved.length / channels;
  const playFrame = Math.floor(
    (positionSec / durationSec) * totalFrames,
  );
  const start = Math.max(0, playFrame - KICK_WINDOW_FRAMES);
  const end = Math.min(totalFrames, start + KICK_WINDOW_FRAMES);
  if (end <= start) return 0;
  let acc = 0;
  let count = 0;
  for (let i = start; i < end; i++) {
    let s = 0;
    for (let c = 0; c < channels; c++) s += interleaved[i * channels + c];
    s /= channels;
    acc += s * s;
    count++;
  }
  if (count === 0) return 0;
  // Soft-clip RMS so bright passages don't blow out the bloom.
  const rms = Math.sqrt(acc / count);
  return Math.max(0, Math.min(1, rms * 1.6));
}
