// The wire side of one mixer layer: session config, knob values, the
// prompt A/B + blend commands and the playhead reports. No DOM and no
// audio here, so the page (sfx.js) and the headless driver
// (scripts/sa3/sfx_demo_wire_smoke.mjs) send exactly the same sequences.

export const SAMPLE_RATE = 48000;
export const CHANNELS = 2;
export const PARAMS_TICK_MS = 80;

// Ambience: a 20 s canvas under the server's long song label, read as a
// crop of a longer file, so the bed never composes an ending and loops
// seamlessly. One-shot: a 4 s canvas labelled with its own length
// (sa3_song_seconds = 0), the upstream whole-file semantics, so the
// event decays to silence inside the clip.
export const MODES = {
  ambience: { label: "Ambience", duration: 20, songSeconds: null },
  oneshot: { label: "One-shot", duration: 4, songSeconds: 0 },
};

export const MAX_RECORD_S = 4;

export function sessionConfig({ mode, promptA, promptB, depth }) {
  const m = MODES[mode];
  const config = {
    telemetry_version: 1,
    backend: "sa3",
    prompt: promptA,
    sa3_duration_s: m.duration,
  };
  if (promptB && promptB !== promptA) config.prompt_b = promptB;
  if (m.songSeconds != null) config.sa3_song_seconds = m.songSeconds;
  if (depth != null) config.depth = depth;
  return config;
}

// "Evolve" is sa3_denoise. With a recorded source it is the plain
// audio-to-audio strength: low keeps the recording, 1.0 ignores it. A
// text layer has only a silent anchor, so below 1.0 it re-noises its
// own earlier output instead (feedback 1), which makes the sound drift
// slowly rather than fade toward silence.
export function knobValues({ evolve, seed, hasSource }) {
  const values = { sa3_denoise: evolve, seed };
  values.feedback = !hasSource && evolve < 1 ? 1 : 0;
  return values;
}

export function silentSource(mode) {
  const frames = Math.round(MODES[mode].duration * SAMPLE_RATE);
  return new Float32Array(frames * CHANNELS);
}

export class LayerWire {
  constructor(RemoteBackend, { wsUrl, sliceWorkerUrl, onSlice, onClose } = {}) {
    this.RemoteBackend = RemoteBackend;
    this.wsUrl = wsUrl;
    this.sliceWorkerUrl = sliceWorkerUrl;
    this.onSlice = onSlice;
    this.onClose = onClose;
    this.remote = null;
    this.duration = 0;
    this.values = {};
    this.blend = 0;
    this.promptA = "";
    this.promptB = "";
  }

  // source: null (text layer) or { interleaved, channels } at 48 kHz,
  // the recorded/uploaded sound for audio-to-audio.
  async open({ mode, promptA, promptB, blend = 0, evolve, seed, source = null, depth }) {
    const config = sessionConfig({ mode, promptA, promptB, depth });
    const pcm = source ? source.interleaved : silentSource(mode);
    const channels = source ? source.channels : CHANNELS;
    const opts = this.sliceWorkerUrl ? { sliceWorkerUrl: this.sliceWorkerUrl } : {};
    const remote = new this.RemoteBackend(this.wsUrl, pcm, channels, config, opts);
    this.remote = remote;
    remote.addEventListener("slice", (event) => this.onSlice?.(event.detail));
    remote.addEventListener("close", () => {
      if (!remote.closedByUser) this.onClose?.();
    });
    await remote.connect();
    if (!remote.initialBuffer) throw new Error("server sent no initial buffer");
    this.duration = remote.duration ?? MODES[mode].duration;
    this.promptA = promptA;
    this.promptB = promptB && promptB !== promptA ? promptB : "";
    this.values = knobValues({ evolve, seed, hasSource: !!source });
    this.hasSource = !!source;
    if (blend > 0) this.setBlend(blend);
    return remote;
  }

  setPrompts(promptA, promptB) {
    this.promptA = promptA;
    this.promptB = promptB && promptB !== promptA ? promptB : "";
    this.remote?.sendPrompt(promptA, undefined, undefined, this.promptB || undefined);
    // The backend keeps its blend across a prompt swap; re-send it so the
    // server and the slider can never disagree.
    this.remote?.sendSetPromptBlend(this.blend);
  }

  setBlend(value) {
    this.blend = Math.max(0, Math.min(1, value));
    this.remote?.sendSetPromptBlend(this.blend);
  }

  setEvolve(evolve) {
    this.values = { ...this.values, ...knobValues({ evolve, seed: this.values.seed, hasSource: this.hasSource }) };
  }

  setSeed(seed) {
    this.values = { ...this.values, seed };
  }

  // Playhead report, every PARAMS_TICK_MS. The ring renders just ahead
  // of this position, so a one-shot layer reports a free-running virtual
  // playhead (VirtualPlayhead) and the whole clip keeps refreshing even
  // while the audible playhead is parked between triggers.
  report(positionSec) {
    return this.remote?.sendParams(this.values, positionSec) ?? false;
  }

  close() {
    try { this.remote?.close(); } catch {}
    this.remote = null;
  }
}

// Free-running loop position for the server, restarted at each trigger
// so the freshest slices land just ahead of what is about to play.
export class VirtualPlayhead {
  constructor(duration, now = () => performance.now() / 1000) {
    this.duration = duration;
    this.now = now;
    this.t0 = now();
  }
  restart() { this.t0 = this.now(); }
  position() { return (this.now() - this.t0) % this.duration; }
}

export function randomSeed() {
  return Math.floor(Math.random() * 100000);
}

// 16-bit PCM WAV from interleaved float samples.
export function encodeWav(interleaved, channels, rate) {
  const frames = Math.floor(interleaved.length / channels);
  const buf = new ArrayBuffer(44 + frames * channels * 2);
  const dv = new DataView(buf);
  const str = (o, s) => [...s].forEach((c, i) => dv.setUint8(o + i, c.charCodeAt(0)));
  str(0, "RIFF"); dv.setUint32(4, 36 + frames * channels * 2, true); str(8, "WAVE");
  str(12, "fmt "); dv.setUint32(16, 16, true); dv.setUint16(20, 1, true);
  dv.setUint16(22, channels, true); dv.setUint32(24, rate, true);
  dv.setUint32(28, rate * channels * 2, true); dv.setUint16(32, channels * 2, true);
  dv.setUint16(34, 16, true); str(36, "data"); dv.setUint32(40, frames * channels * 2, true);
  for (let i = 0; i < frames * channels; i++) {
    dv.setInt16(44 + i * 2, Math.max(-1, Math.min(1, interleaved[i])) * 32767, true);
  }
  return buf;
}

// Sum layers into one stereo mix of `seconds`: ambience loops tile, a
// one-shot plays once from the start. Each entry: { mirror, channels,
// gain, loop }.
export function mixLayers(layers, seconds, rate = SAMPLE_RATE) {
  const frames = Math.round(seconds * rate);
  const out = new Float32Array(frames * CHANNELS);
  for (const { mirror, channels, gain, loop } of layers) {
    if (!mirror || !gain) continue;
    const n = Math.floor(mirror.length / channels);
    if (n === 0) continue;
    for (let f = 0; f < frames; f++) {
      if (!loop && f >= n) break;
      const src = (f % n) * channels;
      out[f * 2] += gain * mirror[src];
      out[f * 2 + 1] += gain * mirror[src + (channels > 1 ? 1 : 0)];
    }
  }
  return out;
}
