// Main-thread wrapper around the realtime-buffer AudioWorklet.
// Falls back to ScriptProcessorNode when AudioWorklet is unavailable
// (non-secure contexts like plain HTTP to a remote IP).

import { SAMPLE_RATE } from "./protocol.js";

export class AudioPlayer {
  constructor() {
    this.ctx = null;
    this.node = null;
    this.positionSec = 0;
    this.swapCount = 0;
    this.channels = 2;
    this.frameCount = 0;
    this._listeners = new Set();
    this._mirror = null;  // Float32Array, interleaved, kept in sync with worklet
    this._useWorklet = false;
    // ScriptProcessor fallback state
    this._spBuffer = null;
    this._spPosition = 0;
  }

  get duration() { return this.frameCount / SAMPLE_RATE; }

  async init(initialBufferInterleaved, channels) {
    this.ctx = new AudioContext({ sampleRate: SAMPLE_RATE, latencyHint: "interactive" });

    this.channels = channels;
    this.frameCount = initialBufferInterleaved.length / channels;
    this._mirror = initialBufferInterleaved.slice();

    this._useWorklet = !!(this.ctx.audioWorklet);

    if (this._useWorklet) {
      await this.ctx.audioWorklet.addModule("audio-worklet.js");

      this.node = new AudioWorkletNode(this.ctx, "realtime-buffer", {
        numberOfInputs: 0,
        numberOfOutputs: 1,
        outputChannelCount: [channels],
      });

      this.node.port.onmessage = (e) => {
        const msg = e.data;
        if (msg.type === "position") {
          this.positionSec = msg.positionSec;
          this.swapCount = msg.swapCount;
        }
      };

      const send = initialBufferInterleaved.slice();
      this.node.port.postMessage(
        { type: "init", buffer: send, channels },
        [send.buffer],
      );
    } else {
      // ScriptProcessorNode fallback for non-secure contexts
      console.warn("AudioWorklet unavailable (non-secure context). Using ScriptProcessor fallback.");
      this._spBuffer = initialBufferInterleaved.slice();
      this._spPosition = 0;
      const BUFFER_SIZE = 4096;
      this.node = this.ctx.createScriptProcessor(BUFFER_SIZE, 0, channels);
      this.node.onaudioprocess = (e) => {
        const output = e.outputBuffer;
        const frames = output.length;
        const ch = this.channels;
        const buf = this._spBuffer;
        if (!buf || this.frameCount === 0) {
          for (let c = 0; c < output.numberOfChannels; c++) output.getChannelData(c).fill(0);
          return;
        }
        const nFrames = this.frameCount;
        // Mirror the worklet's loop-seam crossfade so non-secure-context
        // playback (ScriptProcessor fallback) gets the same smooth wrap.
        const seamFadeLen = Math.max(1, Math.floor(this.ctx.sampleRate * 0.05));
        const seam = Math.min(seamFadeLen, Math.floor(nFrames / 4));
        const outChs = [];
        for (let c = 0; c < output.numberOfChannels; c++) outChs.push(output.getChannelData(c));
        let pos = this._spPosition;
        for (let i = 0; i < frames; i++) {
          if (seam > 0 && (nFrames - pos) <= seam) {
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
      };
    }

    this.node.connect(this.ctx.destination);
  }

  // Overwrite a region of the worklet's buffer.
  patch(startFrame, audioInterleaved) {
    this._writeMirror(startFrame, audioInterleaved, /*add=*/false);
    if (this._useWorklet) {
      const send = audioInterleaved.slice();
      this.node.port.postMessage(
        { type: "patch", start: startFrame, audio: send },
        [send.buffer],
      );
    } else {
      this._writeSPBuffer(startFrame, audioInterleaved, false);
    }
  }

  // Replace the entire loop buffer with a new source. The worklet
  // crossfades the old and new buffers over CROSSFADE_SECONDS (50 ms by
  // default), so callers don't hear a click. ScriptProcessor fallback
  // does an instant swap (the seam-fade still hides the wrap).
  swap(interleavedBuffer, channels) {
    this.channels = channels || this.channels;
    this.frameCount = interleavedBuffer.length / this.channels;
    this._mirror = interleavedBuffer.slice();
    this.swapCount++;
    for (const fn of this._listeners) fn();
    if (this._useWorklet) {
      const send = interleavedBuffer.slice();
      this.node.port.postMessage(
        { type: "swap", buffer: send, channels: this.channels },
        [send.buffer],
      );
    } else {
      this._spBuffer = interleavedBuffer.slice();
      this._spPosition = 0;
    }
  }

  // Delta-add into a region of the worklet's buffer.
  addDelta(startFrame, deltaInterleaved) {
    this._writeMirror(startFrame, deltaInterleaved, /*add=*/true);
    if (this._useWorklet) {
      const send = deltaInterleaved.slice();
      this.node.port.postMessage(
        { type: "add", start: startFrame, audio: send },
        [send.buffer],
      );
    } else {
      this._writeSPBuffer(startFrame, deltaInterleaved, true);
    }
  }

  _writeSPBuffer(startFrame, audioInterleaved, add) {
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

  _writeMirror(startFrame, audioInterleaved, add) {
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

  // Read-only view of the current buffer (for waveform rendering).
  getMirror() { return this._mirror; }

  onMirrorChange(fn) {
    this._listeners.add(fn);
    return () => this._listeners.delete(fn);
  }

  async resume() { if (this.ctx?.state === "suspended") await this.ctx.resume(); }

  async close() {
    try { this.node?.disconnect(); } catch {}
    try { await this.ctx?.close(); } catch {}
  }
}
