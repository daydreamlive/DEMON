// Loudness measurement utilities. Two metrics:
//
//   - LUFS  (BS.1770-4 / EBU R128 K-weighted, with absolute + relative
//           gating). Standard for broadcast loudness; over-reads bright
//           material because of the K-filter's high-shelf prefilter.
//   - dBA   (IEC 61672 A-weighted RMS). Tracks perceived loudness at
//           low-to-moderate SPL more closely than K-weighting on
//           spectrally imbalanced content.
//
// Plus the source-buffer scanner used by AudioPlayer's loudness matcher
// at session start: scans the whole source for the loudest short-term
// window, which becomes the high-water-mark floor so the matcher has a
// stable target even when mirror overwrites obscure source content
// during live operation.
//
// All filter coefficients are pinned to 48 kHz (the only sample rate
// the worklet runs at). Functions return null if a non-48k buffer
// reaches them, rather than reporting a wrong reading.

export type LoudnessMetric = "lufs" | "dba";

interface LufsMeasurement {
  integratedLufs: number | null;
  peak: number;
}

interface DbaMeasurement {
  integratedDba: number | null;
  peak: number;
}

// -- BS.1770-4 K-weighting (pre-filter + RLB high-pass). -------------
const PRE_B = [1.53512485958697, -2.69169618940638, 1.19839281085285];
const PRE_A = [-1.69065929318241, 0.73248077421585];
const RLB_B = [1.0, -2.0, 1.0];
const RLB_A = [-1.99004745483398, 0.99007225036621];

// -- IEC 61672 A-weighting at 48 kHz, bilinear-transformed, expressed
//    as three cascaded biquads (a0 normalized to 1.0). Derived from
//    the analog A-weighting filter; verified against the IEC reference
//    curve to within ~0.1 dB up to 8 kHz. ---------------------------
const A_SECTIONS: ReadonlyArray<readonly [number, number, number, number, number]> = [
  [0.2343017922995135, 0.4686035845990268, 0.2343017922995133, -0.2245584580597783, 0.0126066252715464],
  [1.0000000000000000, -1.9999999663681969, 0.9999999830617619, -1.8938704944148377, 0.8951597688151856],
  [1.0000000000000000, -2.0000000336318098, 1.0000000169382441, -1.9946144563012589, 0.9946217073246171],
];

const BLOCK_S = 0.4;
const HOP_S = 0.1;
const ABSOLUTE_GATE_LKFS = -70;
const RELATIVE_GATE_OFFSET = 10;

function biquadInPlace(
  x: Float32Array,
  bn: readonly number[],
  an: readonly number[],
): void {
  const [b0, b1, b2] = bn;
  const [a1, a2] = an;
  let z1 = 0;
  let z2 = 0;
  for (let i = 0; i < x.length; i++) {
    const v = x[i];
    const y = b0 * v + z1;
    z1 = b1 * v - a1 * y + z2;
    z2 = b2 * v - a2 * y;
    x[i] = y;
  }
}

function kWeight(x: Float32Array): void {
  biquadInPlace(x, PRE_B, PRE_A);
  biquadInPlace(x, RLB_B, RLB_A);
}

function aWeight(x: Float32Array): void {
  for (const sec of A_SECTIONS) {
    const [b0, b1, b2, a1, a2] = sec;
    biquadInPlace(x, [b0, b1, b2], [a1, a2]);
  }
}

function deinterleave(
  interleaved: Float32Array,
  channels: number,
  frames: number,
): Float32Array[] {
  const out: Float32Array[] = [];
  for (let c = 0; c < channels; c++) {
    const ch = new Float32Array(frames);
    for (let i = 0; i < frames; i++) ch[i] = interleaved[i * channels + c];
    out.push(ch);
  }
  return out;
}

function samplePeak(interleaved: Float32Array): number {
  let peak = 0;
  for (let i = 0; i < interleaved.length; i++) {
    const a = Math.abs(interleaved[i]);
    if (a > peak) peak = a;
  }
  return peak;
}

function blockMeanSquares(
  perChannel: Float32Array[],
  frames: number,
  blockSize: number,
  hopSize: number,
): Float64Array {
  const numBlocks = Math.max(0, Math.floor((frames - blockSize) / hopSize) + 1);
  const out = new Float64Array(numBlocks);
  for (let b = 0; b < numBlocks; b++) {
    const start = b * hopSize;
    let weighted = 0;
    for (let c = 0; c < perChannel.length; c++) {
      const ch = perChannel[c];
      let ms = 0;
      for (let i = start; i < start + blockSize; i++) ms += ch[i] * ch[i];
      weighted += ms / blockSize;
    }
    out[b] = weighted;
  }
  return out;
}

// BS.1770 two-stage gating: -70 LKFS absolute, then -10 LU relative
// to the absolute-gated mean. Returns gated mean MS or null.
function gatedMean(
  blocks: Float64Array,
  startIdx: number,
  endIdx: number,
  toLkfs: (ms: number) => number,
): number | null {
  if (endIdx <= startIdx) return null;

  let absSum = 0;
  let absCount = 0;
  for (let i = startIdx; i < endIdx; i++) {
    const ms = blocks[i];
    if (ms > 0 && toLkfs(ms) >= ABSOLUTE_GATE_LKFS) {
      absSum += ms;
      absCount++;
    }
  }
  if (absCount === 0) return null;
  const meanAbs = absSum / absCount;
  const relThreshold = toLkfs(meanAbs) - RELATIVE_GATE_OFFSET;

  let relSum = 0;
  let relCount = 0;
  for (let i = startIdx; i < endIdx; i++) {
    const ms = blocks[i];
    if (ms > 0 && toLkfs(ms) >= ABSOLUTE_GATE_LKFS && toLkfs(ms) >= relThreshold) {
      relSum += ms;
      relCount++;
    }
  }
  if (relCount === 0) return null;
  return relSum / relCount;
}

// -- LUFS: -0.691 + 10 log10(MS) -------------------------------------
function lkfsLufs(meanSquare: number): number {
  return -0.691 + 10 * Math.log10(meanSquare);
}

// -- dBA:  10 log10(MS) ----------------------------------------------
//    No -0.691 offset (that's a BS.1770 broadcast normalization). The
//    gating procedure still uses dB units; the absolute and relative
//    gates are interpreted in dBA the same way they're interpreted in
//    LUFS (dB SPL above some implicit reference).
function lkfsDba(meanSquare: number): number {
  return 10 * Math.log10(meanSquare);
}

export function measureIntegratedLufs(
  interleaved: Float32Array,
  channels: number,
  sampleRate: number,
): LufsMeasurement {
  if (sampleRate !== 48000) return { integratedLufs: null, peak: 0 };
  const peak = samplePeak(interleaved);
  const frames = (interleaved.length / channels) | 0;
  const blockSize = Math.round(BLOCK_S * sampleRate);
  const hopSize = Math.round(HOP_S * sampleRate);
  if (frames < blockSize) return { integratedLufs: null, peak };

  const perChannel = deinterleave(interleaved, channels, frames);
  for (const ch of perChannel) kWeight(ch);

  const blocks = blockMeanSquares(perChannel, frames, blockSize, hopSize);
  const meanRel = gatedMean(blocks, 0, blocks.length, lkfsLufs);
  if (meanRel === null) return { integratedLufs: null, peak };
  return { integratedLufs: lkfsLufs(meanRel), peak };
}

export function measureIntegratedDba(
  interleaved: Float32Array,
  channels: number,
  sampleRate: number,
): DbaMeasurement {
  if (sampleRate !== 48000) return { integratedDba: null, peak: 0 };
  const peak = samplePeak(interleaved);
  const frames = (interleaved.length / channels) | 0;
  const blockSize = Math.round(BLOCK_S * sampleRate);
  const hopSize = Math.round(HOP_S * sampleRate);
  if (frames < blockSize) return { integratedDba: null, peak };

  const perChannel = deinterleave(interleaved, channels, frames);
  for (const ch of perChannel) aWeight(ch);

  const blocks = blockMeanSquares(perChannel, frames, blockSize, hopSize);
  const meanRel = gatedMean(blocks, 0, blocks.length, lkfsDba);
  if (meanRel === null) return { integratedDba: null, peak };
  return { integratedDba: lkfsDba(meanRel), peak };
}

// Generic dispatch -- callers in AudioPlayer are metric-agnostic.
export function measureLoudness(
  interleaved: Float32Array,
  channels: number,
  sampleRate: number,
  metric: LoudnessMetric,
): { value: number | null; peak: number } {
  if (metric === "dba") {
    const m = measureIntegratedDba(interleaved, channels, sampleRate);
    return { value: m.integratedDba, peak: m.peak };
  }
  const m = measureIntegratedLufs(interleaved, channels, sampleRate);
  return { value: m.integratedLufs, peak: m.peak };
}

/**
 * Scan a buffer for the loudest short-term window. The matcher uses
 * this at init() / swap() to seed the high-water-mark floor with the
 * loudest passage the listener will eventually hear from the source,
 * so the gain target is stable even when mirror overwrites obscure
 * source content during the live meter loop.
 *
 * Returns null if the buffer is shorter than one window or has no
 * audible content.
 *
 * Implementation: K-weight (or A-weight) the whole buffer once, then
 * slide a window of `windowSec` over the per-block mean-squares with a
 * 250 ms hop, gating each window like the live meter does.
 */
export function findLoudestShortTermLoudness(
  interleaved: Float32Array,
  channels: number,
  sampleRate: number,
  windowSec: number,
  metric: LoudnessMetric,
): number | null {
  if (sampleRate !== 48000) return null;
  const frames = (interleaved.length / channels) | 0;
  const blockSize = Math.round(BLOCK_S * sampleRate);
  const hopSize = Math.round(HOP_S * sampleRate);
  const winFrames = Math.round(windowSec * sampleRate);
  if (frames < winFrames) return null;

  const perChannel = deinterleave(interleaved, channels, frames);
  if (metric === "dba") {
    for (const ch of perChannel) aWeight(ch);
  } else {
    for (const ch of perChannel) kWeight(ch);
  }

  const blocks = blockMeanSquares(perChannel, frames, blockSize, hopSize);
  const blocksPerWindow = Math.max(
    1,
    Math.floor((winFrames - blockSize) / hopSize) + 1,
  );
  const winHopBlocks = Math.max(1, Math.round(0.25 / HOP_S));
  const toLkfs = metric === "dba" ? lkfsDba : lkfsLufs;

  let best: number | null = null;
  for (let i = 0; i + blocksPerWindow <= blocks.length; i += winHopBlocks) {
    const meanRel = gatedMean(blocks, i, i + blocksPerWindow, toLkfs);
    if (meanRel === null) continue;
    const v = toLkfs(meanRel);
    if (best === null || v > best) best = v;
  }
  return best;
}

export function lufsMakeupGain(
  integratedLufs: number,
  targetLufs: number,
  peak: number,
  peakCeiling: number,
): number {
  if (peak <= 0) return 1.0;
  const desired = Math.pow(10, (targetLufs - integratedLufs) / 20);
  const peakClamp = peakCeiling / peak;
  return Math.min(desired, peakClamp);
}

/**
 * Single-block loudness reading: weight, mean-square, log. No gating
 * (caller is measuring a known-active block, not an integrated track).
 * Used by AudioPlayer to maintain a per-chunk loudness map of the
 * mirror so the matcher can look up "what's at this playhead position"
 * in O(1) instead of waiting for a sliding meter window to fill.
 *
 * Returns loudness in the same dB scale as `measureIntegrated*`:
 *   - LUFS: -0.691 + 10 log10(MS) (BS.1770 channel-summed K-weighted)
 *   - dBA:  10 log10(MS)
 *
 * Block lengths shorter than ~100 ms include some K/A-filter
 * transient, but for matching-purposes the relative loudness is
 * accurate enough since the same filter runs on every block.
 */
export function measureBlock(
  interleaved: Float32Array,
  channels: number,
  metric: LoudnessMetric,
): { loudness: number; peak: number } {
  if (interleaved.length === 0) {
    return { loudness: Number.NEGATIVE_INFINITY, peak: 0 };
  }
  const frames = (interleaved.length / channels) | 0;
  let peak = 0;
  for (let i = 0; i < interleaved.length; i++) {
    const a = Math.abs(interleaved[i]);
    if (a > peak) peak = a;
  }
  const perChannel: Float32Array[] = [];
  for (let c = 0; c < channels; c++) {
    const ch = new Float32Array(frames);
    for (let i = 0; i < frames; i++) ch[i] = interleaved[i * channels + c];
    perChannel.push(ch);
  }
  if (metric === "dba") {
    for (const ch of perChannel) aWeight(ch);
  } else {
    for (const ch of perChannel) kWeight(ch);
  }
  let weightedMs = 0;
  for (const ch of perChannel) {
    let ms = 0;
    for (let i = 0; i < frames; i++) ms += ch[i] * ch[i];
    weightedMs += frames > 0 ? ms / frames : 0;
  }
  if (weightedMs <= 0) {
    return { loudness: Number.NEGATIVE_INFINITY, peak };
  }
  const offset = metric === "dba" ? 0 : -0.691;
  return { loudness: offset + 10 * Math.log10(weightedMs), peak };
}
