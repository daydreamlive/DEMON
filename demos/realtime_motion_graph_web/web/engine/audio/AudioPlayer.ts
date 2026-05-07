// Main-thread wrapper around the realtime-buffer AudioWorklet.
// Falls back to ScriptProcessorNode when AudioWorklet is unavailable
// (non-secure contexts like plain HTTP to a remote IP).
//
// Direct TS port of DEMON/demos/realtime_motion_graph_web/static/audio.js.

import { SAMPLE_RATE } from "@/engine/protocol";

type MirrorListener = () => void;

interface AudioWorkletNodeWithPort extends AudioNode {
  port: MessagePort;
}

export class AudioPlayer {
  ctx: AudioContext | null = null;
  node: AudioWorkletNode | ScriptProcessorNode | null = null;
  positionSec = 0;
  swapCount = 0;
  channels = 2;
  frameCount = 0;

  private _listeners: Set<MirrorListener> = new Set();
  private _mirror: Float32Array | null = null;
  private _useWorklet = false;
  private _spBuffer: Float32Array | null = null;
  private _spPosition = 0;
  private _recordDest: MediaStreamAudioDestinationNode | null = null;

  get duration(): number {
    return this.frameCount / SAMPLE_RATE;
  }

  async init(
    initialBufferInterleaved: Float32Array,
    channels: number,
  ): Promise<void> {
    this.ctx = new AudioContext({
      sampleRate: SAMPLE_RATE,
      latencyHint: "interactive",
    });

    this.channels = channels;
    this.frameCount = initialBufferInterleaved.length / channels;
    this._mirror = initialBufferInterleaved.slice();

    this._useWorklet = !!this.ctx.audioWorklet;

    if (this._useWorklet) {
      // Stable URL — worklet ships from public/ so AudioContext can resolve it.
      await this.ctx.audioWorklet.addModule("/audio-worklet.js");

      const node = new AudioWorkletNode(this.ctx, "realtime-buffer", {
        numberOfInputs: 0,
        numberOfOutputs: 1,
        outputChannelCount: [channels],
      });
      this.node = node;

      node.port.onmessage = (e: MessageEvent) => {
        const msg = e.data as { type: string; positionSec?: number; swapCount?: number };
        if (msg.type === "position") {
          this.positionSec = msg.positionSec ?? 0;
          this.swapCount = msg.swapCount ?? this.swapCount;
        }
      };

      const send = initialBufferInterleaved.slice();
      node.port.postMessage(
        { type: "init", buffer: send, channels },
        [send.buffer],
      );
    } else {
      // ScriptProcessorNode fallback for non-secure contexts.
      console.warn(
        "[AudioPlayer] AudioWorklet unavailable (non-secure context). Using ScriptProcessor fallback.",
      );
      this._spBuffer = initialBufferInterleaved.slice();
      this._spPosition = 0;
      const BUFFER_SIZE = 4096;
      const sp = this.ctx.createScriptProcessor(BUFFER_SIZE, 0, channels);
      this.node = sp;
      sp.onaudioprocess = (e: AudioProcessingEvent) => this._spProcess(e);
    }

    this.node.connect(this.ctx.destination);
  }

  /** Overwrite a region of the worklet's buffer. */
  patch(startFrame: number, audioInterleaved: Float32Array): void {
    this._writeMirror(startFrame, audioInterleaved, false);
    if (this._useWorklet && this.node) {
      const send = audioInterleaved.slice();
      (this.node as AudioWorkletNode).port.postMessage(
        { type: "patch", start: startFrame, audio: send },
        [send.buffer],
      );
    } else {
      this._writeSPBuffer(startFrame, audioInterleaved, false);
    }
  }

  /**
   * Replace the entire loop buffer. The worklet crossfades old → new over
   * CROSSFADE_SECONDS (50 ms); ScriptProcessor fallback does an instant
   * swap (the seam-fade still hides the wrap).
   */
  swap(interleavedBuffer: Float32Array, channels?: number): void {
    this.channels = channels || this.channels;
    this.frameCount = interleavedBuffer.length / this.channels;
    this._mirror = interleavedBuffer.slice();
    this.swapCount++;
    for (const fn of this._listeners) fn();
    if (this._useWorklet && this.node) {
      const send = interleavedBuffer.slice();
      (this.node as AudioWorkletNode).port.postMessage(
        { type: "swap", buffer: send, channels: this.channels },
        [send.buffer],
      );
    } else {
      this._spBuffer = interleavedBuffer.slice();
      this._spPosition = 0;
    }
  }

  /** Delta-add into a region of the worklet's buffer. */
  addDelta(startFrame: number, deltaInterleaved: Float32Array): void {
    this._writeMirror(startFrame, deltaInterleaved, true);
    if (this._useWorklet && this.node) {
      const send = deltaInterleaved.slice();
      (this.node as AudioWorkletNode).port.postMessage(
        { type: "add", start: startFrame, audio: send },
        [send.buffer],
      );
    } else {
      this._writeSPBuffer(startFrame, deltaInterleaved, true);
    }
  }

  /** Read-only view of the current buffer (for waveform rendering). */
  getMirror(): Float32Array | null {
    return this._mirror;
  }

  onMirrorChange(fn: MirrorListener): () => void {
    this._listeners.add(fn);
    return () => {
      this._listeners.delete(fn);
    };
  }

  async resume(): Promise<void> {
    if (this.ctx?.state === "suspended") await this.ctx.resume();
  }

  /**
   * Lazily create a MediaStream tee'd off the worklet output for recording.
   * Same node graph as the live destination — bit-identical to what the
   * user hears. Stays alive for the rest of the session once created.
   */
  getRecordingStream(): MediaStream | null {
    if (!this.ctx || !this.node) return null;
    if (!this._recordDest) {
      this._recordDest = this.ctx.createMediaStreamDestination();
      this.node.connect(this._recordDest);
    }
    return this._recordDest.stream;
  }

  async close(): Promise<void> {
    try {
      this.node?.disconnect();
    } catch {}
    this._recordDest = null;
    try {
      await this.ctx?.close();
    } catch {}
  }

  // ── internals ────────────────────────────────────────────────────────

  private _writeSPBuffer(
    startFrame: number,
    audioInterleaved: Float32Array,
    add: boolean,
  ): void {
    if (!this._spBuffer) return;
    const ch = this.channels;
    const base = startFrame * ch;
    const n = Math.min(audioInterleaved.length, this._spBuffer.length - base);
    if (n <= 0) return;
    if (add) {
      for (let i = 0; i < n; i++) this._spBuffer[base + i] += audioInterleaved[i];
    } else {
      for (let i = 0; i < n; i++) this._spBuffer[base + i] = audioInterleaved[i];
    }
  }

  private _writeMirror(
    startFrame: number,
    audioInterleaved: Float32Array,
    add: boolean,
  ): void {
    if (!this._mirror) return;
    const ch = this.channels;
    const base = startFrame * ch;
    const n = Math.min(audioInterleaved.length, this._mirror.length - base);
    if (n <= 0) return;
    if (add) {
      for (let i = 0; i < n; i++) this._mirror[base + i] += audioInterleaved[i];
    } else {
      for (let i = 0; i < n; i++) this._mirror[base + i] = audioInterleaved[i];
    }
    this.swapCount++;
    for (const fn of this._listeners) fn();
  }

  private _spProcess(e: AudioProcessingEvent): void {
    const output = e.outputBuffer;
    const frames = output.length;
    const ch = this.channels;
    const buf = this._spBuffer;
    if (!buf || this.frameCount === 0 || !this.ctx) {
      for (let c = 0; c < output.numberOfChannels; c++) {
        output.getChannelData(c).fill(0);
      }
      return;
    }
    const nFrames = this.frameCount;
    // Mirror the worklet's loop-seam crossfade so non-secure-context playback
    // (ScriptProcessor fallback) gets the same smooth wrap.
    const seamFadeLen = Math.max(1, Math.floor(this.ctx.sampleRate * 0.05));
    const seam = Math.min(seamFadeLen, Math.floor(nFrames / 4));
    const outChs: Float32Array[] = [];
    for (let c = 0; c < output.numberOfChannels; c++) {
      outChs.push(output.getChannelData(c));
    }
    let pos = this._spPosition;
    for (let i = 0; i < frames; i++) {
      if (seam > 0 && nFrames - pos <= seam) {
        const distFromEnd = nFrames - pos;
        const t = (seam - distFromEnd) / seam;
        const headPos = seam - distFromEnd;
        for (let c = 0; c < outChs.length; c++) {
          const cc = Math.min(c, ch - 1);
          const sTail = buf[pos * ch + cc];
          const sHead = buf[headPos * ch + cc];
          outChs[c][i] = sTail * (1 - t) + sHead * t;
        }
      } else {
        for (let c = 0; c < outChs.length; c++) {
          const cc = Math.min(c, ch - 1);
          outChs[c][i] = buf[pos * ch + cc];
        }
      }
      pos++;
      if (pos >= nFrames) pos = seam;
    }
    this._spPosition = pos;
    this.positionSec = this._spPosition / SAMPLE_RATE;
  }
}
