// MicCapture: microphone -> 48 kHz mono float32 frames for `audio_in`.
//
// Part of the demon-client SDK: no imports from the host app. Opens the
// microphone with getUserMedia (no echo cancellation, noise suppression or
// gain control: those rewrite the signal the analysers listen to), taps it
// with an AudioWorklet (ScriptProcessor where AudioWorklet is missing),
// downmixes to mono and frames it through PcmFramer (100 ms frames by
// default; resampled when the context does not run at 48 kHz). Nothing is
// routed to the speakers.
//
//   const mic = new MicCapture({
//     onFrame: (f) => remote.sendAudioIn(f.pcm, f.seq, { startS: f.startS }),
//     onLevel: ({ rms, peak }) => meter.draw(rms, peak),
//   });
//   await mic.start();   // must follow a user gesture
//   ...
//   await mic.stop();    // the next start() restarts seq at 0: a new take

import { AUDIO_IN_SAMPLE_RATE, PcmFramer, type PcmFrame } from "./framer";

export interface MicLevel {
  rms: number;
  peak: number;
}

export interface MicCaptureOptions {
  /** Called with every completed frame, in order. */
  onFrame: (frame: PcmFrame) => void;
  /** Called once per frame with its level, for an input meter. */
  onLevel?: (level: MicLevel) => void;
  /** Frame length in milliseconds (default 100). */
  frameMs?: number;
  /** Input device to open (default: the system default microphone). */
  deviceId?: string;
  /** Injected for tests / non-standard hosts; defaults to
   *  `navigator.mediaDevices`. */
  mediaDevices?: Pick<MediaDevices, "getUserMedia">;
  /** Injected for tests; defaults to `new AudioContext(opts)`. */
  audioContextFactory?: (opts: AudioContextOptions) => AudioContext;
  /** Set false to force the ScriptProcessor path. Default true. */
  useWorklet?: boolean;
}

const WORKLET_NAME = "demon-mic-tap";
// Posts mono blocks of 1024 samples to the main thread (eight render
// quanta per message instead of one).
const WORKLET_SOURCE = `
class DemonMicTap extends AudioWorkletProcessor {
  constructor() { super(); this.buf = new Float32Array(1024); this.n = 0; }
  process(inputs) {
    const input = inputs[0];
    if (input && input.length) {
      const ch = input.length;
      const len = input[0].length;
      for (let i = 0; i < len; i++) {
        let s = 0;
        for (let c = 0; c < ch; c++) s += input[c][i];
        this.buf[this.n++] = s / ch;
        if (this.n === this.buf.length) {
          this.port.postMessage(this.buf, [this.buf.buffer]);
          this.buf = new Float32Array(1024);
          this.n = 0;
        }
      }
    }
    return true;
  }
}
registerProcessor("${WORKLET_NAME}", DemonMicTap);
`;

const SCRIPT_PROCESSOR_BLOCK = 2048;

export class MicCapture {
  private readonly opts: MicCaptureOptions;
  private ctx: AudioContext | null = null;
  private stream: MediaStream | null = null;
  private source: MediaStreamAudioSourceNode | null = null;
  private tap: AudioNode | null = null;
  private sink: GainNode | null = null;
  private framer: PcmFramer | null = null;
  private _running = false;

  /** Whether the AudioWorklet path is in use (false = ScriptProcessor). */
  usingWorklet = false;

  constructor(opts: MicCaptureOptions) {
    this.opts = opts;
  }

  get running(): boolean {
    return this._running;
  }

  /** The AudioContext rate in use (48000 unless the browser refused it);
   *  null while stopped. Frames are always 48 kHz. */
  get contextSampleRate(): number | null {
    return this.ctx ? this.ctx.sampleRate : null;
  }

  /** Open the microphone and start framing. Rejects (and leaves nothing
   *  open) when permission is denied or no input exists. */
  async start(): Promise<void> {
    if (this._running) return;
    const media =
      this.opts.mediaDevices ??
      (typeof navigator !== "undefined" ? navigator.mediaDevices : undefined);
    if (!media?.getUserMedia) {
      throw new Error("this browser cannot open a microphone (no getUserMedia)");
    }
    const audio: MediaTrackConstraints = {
      channelCount: 1,
      echoCancellation: false,
      noiseSuppression: false,
      autoGainControl: false,
    };
    if (this.opts.deviceId) audio.deviceId = { exact: this.opts.deviceId };
    const stream = await media.getUserMedia({ audio });
    try {
      const make =
        this.opts.audioContextFactory ??
        ((o: AudioContextOptions) => new AudioContext(o));
      const ctx = make({ sampleRate: AUDIO_IN_SAMPLE_RATE, latencyHint: "interactive" });
      this.ctx = ctx;
      this.stream = stream;
      this.framer = new PcmFramer({ inputRate: ctx.sampleRate, frameMs: this.opts.frameMs });
      this.source = ctx.createMediaStreamSource(stream);
      // A silent sink keeps the tap pulled by the graph without making a
      // sound (a ScriptProcessor only runs while connected to the output).
      const sink = ctx.createGain();
      sink.gain.value = 0;
      sink.connect(ctx.destination);
      this.sink = sink;
      this.tap = await this.makeTap(ctx);
      this.source.connect(this.tap);
      this.tap.connect(sink);
      if (ctx.state === "suspended") await ctx.resume();
      this._running = true;
    } catch (e) {
      await this.teardown();
      for (const t of stream.getTracks()) t.stop();
      throw e;
    }
  }

  /** Stop and release the microphone. The next `start` numbers frames
   *  from seq 0 again, which the server reads as a new take. A partial
   *  frame at stop is dropped. */
  async stop(): Promise<void> {
    if (!this._running && !this.ctx) return;
    this._running = false;
    await this.teardown();
  }

  private async makeTap(ctx: AudioContext): Promise<AudioNode> {
    const worklet = (ctx as AudioContext & { audioWorklet?: AudioWorklet }).audioWorklet;
    const Node =
      typeof AudioWorkletNode !== "undefined" ? AudioWorkletNode : undefined;
    if (this.opts.useWorklet !== false && worklet && Node) {
      const url = URL.createObjectURL(
        new Blob([WORKLET_SOURCE], { type: "application/javascript" }),
      );
      try {
        await worklet.addModule(url);
      } finally {
        URL.revokeObjectURL(url);
      }
      const node = new Node(ctx, WORKLET_NAME, {
        numberOfInputs: 1,
        numberOfOutputs: 1,
        outputChannelCount: [1],
      });
      node.port.onmessage = (ev: MessageEvent<Float32Array>) => this.feed(ev.data);
      this.usingWorklet = true;
      return node;
    }
    const proc = ctx.createScriptProcessor(SCRIPT_PROCESSOR_BLOCK, 2, 1);
    proc.onaudioprocess = (ev: AudioProcessingEvent) => {
      const input = ev.inputBuffer;
      const ch = input.numberOfChannels;
      const first = input.getChannelData(0);
      if (ch <= 1) {
        this.feed(new Float32Array(first));
        return;
      }
      const mono = new Float32Array(first.length);
      for (let c = 0; c < ch; c++) {
        const d = input.getChannelData(c);
        for (let i = 0; i < mono.length; i++) mono[i] += d[i] / ch;
      }
      this.feed(mono);
    };
    this.usingWorklet = false;
    return proc;
  }

  private feed(block: Float32Array): void {
    if (!this._running || !this.framer) return;
    for (const frame of this.framer.push(block)) {
      this.opts.onLevel?.({ rms: frame.rms, peak: frame.peak });
      this.opts.onFrame(frame);
    }
  }

  private async teardown(): Promise<void> {
    const tap = this.tap;
    if (tap && "port" in tap) (tap as AudioWorkletNode).port.onmessage = null;
    if (tap && "onaudioprocess" in tap) (tap as ScriptProcessorNode).onaudioprocess = null;
    for (const node of [this.source, tap, this.sink]) {
      try {
        node?.disconnect();
      } catch {}
    }
    this.source = null;
    this.tap = null;
    this.sink = null;
    if (this.stream) for (const t of this.stream.getTracks()) t.stop();
    this.stream = null;
    const ctx = this.ctx;
    this.ctx = null;
    this.framer = null;
    if (ctx) {
      try {
        await ctx.close();
      } catch {}
    }
  }
}
