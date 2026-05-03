// AudioWorklet that plays a looping PCM buffer the main thread can
// patch in place. Mirrors demos/realtime_motion_graph_web/audio_engine.py
// (swap / patch / crossfade behaviour).
//
// Messages from main thread:
//   {type:'init',     buffer:Float32Array (interleaved), channels:int}
//   {type:'patch',    start:int (frame),  audio:Float32Array (interleaved)}
//   {type:'add',      start:int (frame),  audio:Float32Array (interleaved)}  // delta add
//   {type:'swap',     buffer:Float32Array, channels:int}
//
// Messages to main thread:
//   {type:'position', positionSec:float, swapCount:int}

const CROSSFADE_SECONDS = 0.05;
const SEAM_FADE_SECONDS = 0.05;  // loop-seam crossfade at end-of-buffer
const REPORT_EVERY = 1024;  // frames between position reports

class RealtimeBufferProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.current = null;          // Float32Array, interleaved
    this.channels = 2;
    this.frameCount = 0;          // frames in current buffer
    this.position = 0;            // playhead in frames
    this.swapCount = 0;

    this.oldBuffer = null;
    this.oldFrameCount = 0;
    this.fading = false;
    this.fadePos = 0;
    this.crossfadeLen = Math.max(1, Math.floor(sampleRate * CROSSFADE_SECONDS));
    this.seamFadeLen = Math.max(1, Math.floor(sampleRate * SEAM_FADE_SECONDS));

    this._framesSinceReport = 0;

    this.port.onmessage = (e) => this._onmessage(e.data);
  }

  _onmessage(msg) {
    const { type } = msg;
    if (type === "init") {
      this.current = msg.buffer;
      this.channels = msg.channels || 2;
      this.frameCount = this.current.length / this.channels;
      this.position = 0;
      this.swapCount = 0;
      return;
    }
    if (type === "swap") {
      this.oldBuffer = this.current;
      this.oldFrameCount = this.frameCount;
      this.current = msg.buffer;
      this.channels = msg.channels || this.channels;
      this.frameCount = this.current.length / this.channels;
      this.swapCount++;
      this.fading = true;
      this.fadePos = 0;
      return;
    }
    if (type === "patch" || type === "add") {
      if (!this.current) return;
      const start = msg.start | 0;
      const slice = msg.audio;            // Float32Array, interleaved
      const ch = this.channels;
      const sliceFrames = (slice.length / ch) | 0;
      if (sliceFrames <= 0) return;

      const end = Math.min(start + sliceFrames, this.frameCount);
      const actual = end - start;
      if (actual <= 0) return;

      const base = start * ch;
      const count = actual * ch;
      if (type === "add") {
        // Delta add: matches ``current[s:e] += data`` in server/client path.
        for (let i = 0; i < count; i++) this.current[base + i] += slice[i];
      } else {
        // Raw patch: matches AudioEngine.patch.  Server only sends raw
        // patches for the initial buffer, so seam xfades are unnecessary.
        for (let i = 0; i < count; i++) this.current[base + i] = slice[i];
      }
      return;
    }
  }

  process(_inputs, outputs) {
    const output = outputs[0];
    const frames = output[0].length;
    const outChannels = output.length;

    if (!this.current || this.frameCount === 0) {
      for (let c = 0; c < outChannels; c++) output[c].fill(0);
      return true;
    }

    const ch = this.channels;
    const nCur = this.frameCount;
    // Loop-seam crossfade: blend the last `seam` frames with the first
    // `seam` frames of the buffer, then wrap the playhead to `seam` so
    // the head we just mixed isn't replayed. Hides the click that would
    // otherwise pop on every wrap, and softens the structural mismatch
    // when the source song doesn't loop musically.
    const seam = Math.min(this.seamFadeLen, Math.floor(nCur / 4));

    for (let i = 0; i < frames; i++) {
      const pos = this.position;

      if (this.fading && this.oldBuffer) {
        const fadeT = Math.min(1, this.fadePos / this.crossfadeLen);
        const oldPos = this.position % this.oldFrameCount;
        for (let c = 0; c < outChannels; c++) {
          const cc = Math.min(c, ch - 1);
          const sNew = this.current[pos * ch + cc];
          const sOld = this.oldBuffer[oldPos * ch + cc];
          output[c][i] = sOld * (1 - fadeT) + sNew * fadeT;
        }
        this.fadePos++;
        if (this.fadePos >= this.crossfadeLen) {
          this.fading = false;
          this.oldBuffer = null;
        }
      } else if (seam > 0 && (nCur - pos) <= seam) {
        const distFromEnd = nCur - pos;
        const t = (seam - distFromEnd) / seam;
        const headPos = seam - distFromEnd;
        for (let c = 0; c < outChannels; c++) {
          const cc = Math.min(c, ch - 1);
          const sTail = this.current[pos * ch + cc];
          const sHead = this.current[headPos * ch + cc];
          output[c][i] = sTail * (1 - t) + sHead * t;
        }
      } else {
        for (let c = 0; c < outChannels; c++) {
          const cc = Math.min(c, ch - 1);
          output[c][i] = this.current[pos * ch + cc];
        }
      }

      this.position++;
      if (this.position >= nCur) this.position = seam;
    }

    this._framesSinceReport += frames;
    if (this._framesSinceReport >= REPORT_EVERY) {
      this._framesSinceReport = 0;
      this.port.postMessage({
        type: "position",
        positionSec: this.position / sampleRate,
        swapCount: this.swapCount,
      });
    }

    return true;
  }
}

registerProcessor("realtime-buffer", RealtimeBufferProcessor);
