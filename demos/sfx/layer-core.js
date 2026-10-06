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
// event decays to silence inside the clip. `duration` is also the
// longest input recording a layer in that mode takes.
export const MODES = {
  ambience: { label: "Ambience", duration: 20, songSeconds: null },
  oneshot: { label: "One-shot", duration: 4, songSeconds: 0 },
};

// A one-shot canvas never goes below this, however short the input.
export const MIN_ONESHOT_S = 1;
// Fade-out applied to the end of a one-shot input (see fitInput).
export const ONESHOT_FADE_S = 0.5;
// Longest input kept at decode time: the longest canvas of any mode, so
// switching a layer between modes never needs the file again.
export const MAX_INPUT_S = Math.max(...Object.values(MODES).map((m) => m.duration));
// Denoise range and default for a layer with input audio. Without input
// a layer is pure text-to-audio at denoise 1.0 and has no denoise knob.
export const DENOISE_MIN = 0.1;
export const DENOISE_DEFAULT = 0.5;

export function sessionConfig({ mode, promptA, promptB, depth, duration }) {
  const m = MODES[mode];
  const config = {
    telemetry_version: 1,
    backend: "sa3",
    prompt: promptA,
    sa3_duration_s: duration ?? m.duration,
  };
  if (promptB && promptB !== promptA) config.prompt_b = promptB;
  if (m.songSeconds != null) config.sa3_song_seconds = m.songSeconds;
  if (depth != null) config.depth = depth;
  return config;
}

// The knobs a layer sends. `sa3_denoise` is SA3's audio-to-audio
// strength: every regeneration renoises the session's FIXED source
// latent (the input recording) by that much. `feedback` is pinned to 0:
// above 0 the backend blends the layer's own previous output into that
// source, and at 1 it replaces it, so every generation re-renders the
// last one and codec/sampler error compounds into a feedback loop (the
// v2 "evolve" knob did exactly that on text layers). Without input
// audio the source is silence, so denoise is pinned to 1.0 (pure noise
// at every slot; the source never enters).
export function knobValues({ denoise, seed, hasInput }) {
  return { sa3_denoise: hasInput ? denoise : 1, seed, feedback: 0 };
}

export function silentSource(mode) {
  const frames = Math.round(MODES[mode].duration * SAMPLE_RATE);
  return new Float32Array(frames * CHANNELS);
}

// Fit an input recording ({ interleaved, channels: 2, seconds } at
// 48 kHz) to a layer's canvas. One-shot: the canvas follows the
// recording rounded up to whole seconds (at least MIN_ONESHOT_S, at most
// the mode's 4 s; the rest is zero-padded) and its end fades to silence.
// Ambience: the 20 s canvas is
// tiled with the recording. Longer than the canvas: trimmed. Returns
// the PCM to upload, the canvas length and a short note for the UI.
export function fitInput(input, mode) {
  const max = MODES[mode].duration;
  const inFrames = Math.floor(input.interleaved.length / CHANNELS);
  const inSec = inFrames / SAMPLE_RATE;
  let duration;
  // Whole seconds: SA3 small-sfx only ends a clip in silence when its
  // seconds_total label is an integer. At 2.4 / 2.5 / 2.9 / 3.2 / 3.5 s
  // the event never decays (also in upstream's own generate()), at
  // 2 / 3 / 4 s it fades to below -75 dB by the end.
  if (mode === "oneshot") duration = Math.min(max, Math.max(MIN_ONESHOT_S, Math.ceil(inSec - 0.01)));
  else duration = max;
  const frames = Math.round(duration * SAMPLE_RATE);
  const pcm = new Float32Array(frames * CHANNELS);
  const tile = mode === "ambience" && inFrames > 0 && inFrames < frames;
  if (tile) {
    for (let o = 0; o < frames; o += inFrames) {
      const n = Math.min(inFrames, frames - o);
      pcm.set(input.interleaved.subarray(0, n * CHANNELS), o * CHANNELS);
    }
  } else {
    pcm.set(input.interleaved.subarray(0, Math.min(inFrames, frames) * CHANNELS));
  }
  // A one-shot must end in silence, and at low denoise the output follows
  // the recording, so a take that is still sounding at the cut (a hum,
  // a room tone) would never decay. Fade its last ONESHOT_FADE_S (at most
  // a quarter of the canvas) to silence; the true-length label then lets
  // the model close the event inside the clip.
  if (mode === "oneshot") {
    const fade = Math.min(Math.round(ONESHOT_FADE_S * SAMPLE_RATE), Math.floor(frames / 4));
    for (let i = 0; i < fade; i++) {
      const g = 0.5 * (1 + Math.cos((Math.PI * (i + 1)) / fade));
      const f = frames - fade + i;
      pcm[f * CHANNELS] *= g;
      pcm[f * CHANNELS + 1] *= g;
    }
  }
  const trimmed = inFrames > frames;
  let note;
  if (trimmed) note = `trimmed ${inSec.toFixed(1)} s to ${duration.toFixed(1)} s`;
  else if (tile) note = `${inSec.toFixed(1)} s, looped to ${duration.toFixed(0)} s`;
  else if (inSec < duration) note = `${inSec.toFixed(1)} s, padded to ${duration.toFixed(1)} s`;
  else note = `${inSec.toFixed(1)} s`;
  return { pcm, duration, trimmed, tiled: tile, note };
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

  // input: null (text layer) or { interleaved, channels: 2, seconds } at
  // 48 kHz, the layer's recording. It is fitted to the canvas (fitInput)
  // and uploaded once as the session source; every regeneration renoises
  // that fixed latent, never the previous output.
  async open({ mode, promptA, promptB, blend = 0, denoise = DENOISE_DEFAULT, seed, input = null, depth }) {
    const fit = input ? fitInput(input, mode) : null;
    this.fit = fit;
    const config = sessionConfig({ mode, promptA, promptB, depth, duration: fit?.duration });
    const pcm = fit ? fit.pcm : silentSource(mode);
    const channels = CHANNELS;
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
    this.hasInput = !!fit;
    this.values = knobValues({ denoise, seed, hasInput: this.hasInput });
    if (blend > 0) this.setBlend(blend);
    return remote;
  }

  // Send prompts A and B as one prompt swap. A no-op (returns false)
  // when A is empty or nothing changed, so Enter, blur and card clicks
  // can all call it freely without re-capturing the same conditioning.
  setPrompts(promptA, promptB) {
    const a = (promptA ?? "").trim();
    const bRaw = (promptB ?? "").trim();
    const b = bRaw && bRaw !== a ? bRaw : "";
    if (!a || !this.remote || (a === this.promptA && b === this.promptB)) return false;
    this.promptA = a;
    this.promptB = b;
    this.remote.sendPrompt(a, undefined, undefined, b || undefined);
    // The backend keeps its blend across a prompt swap; re-send it so the
    // server and the slider can never disagree.
    this.remote.sendSetPromptBlend(this.blend);
    return true;
  }

  // One slot ("a" or "b") changes, the other keeps its live value: what
  // an example card click does, sent at once with no Apply step.
  setPromptSlot(slot, text) {
    return slot === "b" ? this.setPrompts(this.promptA, text) : this.setPrompts(text, this.promptB);
  }

  setBlend(value) {
    this.blend = Math.max(0, Math.min(1, value));
    this.remote?.sendSetPromptBlend(this.blend);
  }

  setDenoise(denoise) {
    this.values = { ...this.values, ...knobValues({ denoise, seed: this.values.seed, hasInput: this.hasInput }) };
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
