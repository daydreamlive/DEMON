// PcmFramer: turns arbitrary-length mono blocks from an audio graph into
// fixed 48 kHz frames for the `audio_in` stream.
//
// Part of the demon-client SDK: no imports from the host app, no DOM.
// Pure so the framing (carry-over between blocks, seq/startS numbering,
// resampling from a device rate the AudioContext would not change) is unit
// testable without any audio API.

/** The canvas rate every PCM command uses. */
export const AUDIO_IN_SAMPLE_RATE = 48000;
/** Default frame length of the live stream. */
export const AUDIO_IN_FRAME_MS = 100;

/** One frame ready for `RemoteBackend.sendAudioIn`. */
export interface PcmFrame {
  /** Mono float32 at 48 kHz, exactly `frameSamples` long. */
  pcm: Float32Array;
  /** 0, 1, 2, ... since the last reset. */
  seq: number;
  /** Position of `pcm[0]` on the capture clock, seconds since the reset. */
  startS: number;
  /** Root-mean-square level of the frame (0..1 for in-range audio). */
  rms: number;
  /** Absolute peak of the frame. */
  peak: number;
}

export interface PcmFramerOptions {
  /** Sample rate of the blocks passed to `push` (the AudioContext's). */
  inputRate: number;
  /** Frame length in milliseconds (default 100). */
  frameMs?: number;
}

export class PcmFramer {
  readonly inputRate: number;
  readonly frameSamples: number;
  private readonly step: number;
  private frame: Float32Array;
  private filled = 0;
  private seq = 0;
  // Resampler state: unconsumed input (starting at the sample the next
  // output interpolates from) and the fractional read position into it.
  private carry: Float32Array = new Float32Array(0);
  private pos = 0;

  constructor(opts: PcmFramerOptions) {
    if (!(opts.inputRate > 0)) throw new Error("inputRate must be > 0");
    this.inputRate = opts.inputRate;
    const ms = opts.frameMs ?? AUDIO_IN_FRAME_MS;
    this.frameSamples = Math.max(1, Math.round((AUDIO_IN_SAMPLE_RATE * ms) / 1000));
    this.step = this.inputRate / AUDIO_IN_SAMPLE_RATE;
    this.frame = new Float32Array(this.frameSamples);
  }

  /** Feed one block (mono, `inputRate`); returns the frames it completed. */
  push(block: Float32Array): PcmFrame[] {
    const out: PcmFrame[] = [];
    if (this.step === 1) {
      for (let i = 0; i < block.length; i++) this.put(block[i], out);
      return out;
    }
    // Linear interpolation across the block boundary: the carry holds the
    // input from the last unconsumed sample onward.
    const buf = new Float32Array(this.carry.length + block.length);
    buf.set(this.carry, 0);
    buf.set(block, this.carry.length);
    let t = this.pos;
    while (t + 1 < buf.length) {
      const i = Math.floor(t);
      const f = t - i;
      this.put(buf[i] * (1 - f) + buf[i + 1] * f, out);
      t += this.step;
    }
    const keep = Math.min(Math.floor(t), buf.length);
    this.carry = buf.slice(keep);
    this.pos = t - keep;
    return out;
  }

  /** Drop any partial frame and restart numbering at seq 0 (a new take on
   *  the server, which reads a seq that did not advance as a restart). */
  reset(): void {
    this.frame = new Float32Array(this.frameSamples);
    this.filled = 0;
    this.seq = 0;
    this.carry = new Float32Array(0);
    this.pos = 0;
  }

  private put(sample: number, out: PcmFrame[]): void {
    this.frame[this.filled++] = sample;
    if (this.filled < this.frameSamples) return;
    const pcm = this.frame;
    let sum = 0;
    let peak = 0;
    for (let i = 0; i < pcm.length; i++) {
      const v = pcm[i];
      sum += v * v;
      const a = v < 0 ? -v : v;
      if (a > peak) peak = a;
    }
    out.push({
      pcm,
      seq: this.seq,
      startS: (this.seq * this.frameSamples) / AUDIO_IN_SAMPLE_RATE,
      rms: Math.sqrt(sum / pcm.length),
      peak,
    });
    this.seq += 1;
    this.frame = new Float32Array(this.frameSamples);
    this.filled = 0;
  }
}
