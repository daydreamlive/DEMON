import type { DecodedFixture } from "@/engine/audio/loadFixture";

// Server-side latent pool size, mirroring loadFixture.ts and acestep.streaming.session.
// Trim boundaries MUST be multiples of this so the sliced buffer still
// satisfies the VAE-encode constraint.
const SAMPLE_POOL = 9600;

/** Slice a DecodedFixture to a [startS, endS] window, aligned to the
 *  server's sample-pool boundary. Returns a fresh Float32Array — the
 *  original buffer is not retained. Used by WaveformTrimDialog after
 *  the user confirms the trim window.
 *
 *  Pool alignment matches the rule in loadFixture.decodeArrayBuffer:
 *  start floor-aligns to the previous pool boundary, end floor-aligns
 *  to a pool boundary at-or-before endS. The resulting length is at
 *  least one pool (0.2 s at 48 kHz) — caller is responsible for not
 *  passing a window narrower than that. */
export function trimAudioBuffer(
  decoded: DecodedFixture,
  startS: number,
  endS: number,
): DecodedFixture {
  const { interleaved, channels, sampleRate, frames } = decoded;
  const startFrameRaw = Math.max(0, Math.round(startS * sampleRate));
  const endFrameRaw = Math.min(frames, Math.round(endS * sampleRate));
  const startFrame = Math.floor(startFrameRaw / SAMPLE_POOL) * SAMPLE_POOL;
  const endFrameAligned =
    Math.floor(endFrameRaw / SAMPLE_POOL) * SAMPLE_POOL;
  const newFrames = Math.max(SAMPLE_POOL, endFrameAligned - startFrame);
  const out = new Float32Array(newFrames * channels);
  out.set(
    interleaved.subarray(
      startFrame * channels,
      (startFrame + newFrames) * channels,
    ),
  );
  return {
    interleaved: out,
    channels,
    frames: newFrames,
    sampleRate,
  };
}
