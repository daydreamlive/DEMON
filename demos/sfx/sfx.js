import { AudioPlayer, RemoteBackend, SLICE_FLAG_DELTA } from "/sdk/demon-client.js";
import { LIBRARY } from "./library.js";
import {
  CHANNELS, DENOISE_DEFAULT, MAX_INPUT_S, MODES, PARAMS_TICK_MS, SAMPLE_RATE,
  LayerWire, VirtualPlayhead, encodeWav, fitInput, mixLayers, randomSeed,
} from "./layer-core.js";

// Layers the RTX 5090 sustains in real time with one backend serving them
// all (measured headlessly; see notes/sa3_variants/results_sfx.md).
const MAX_LAYERS = 4;
// Trigger-to-sound: the first analyser frame above this RMS (about -50 dBFS).
const ONSET_RMS = 0.003;
// A typed prompt is sent this long after its field loses focus (Enter
// sends at once), so tabbing A -> B -> blend does not fire two swaps.
const BLUR_SEND_MS = 250;

const $ = (sel, root = document) => root.querySelector(sel);
const els = {
  play: $("#play"), add: $("#add"), saveMix: $("#save-mix"), status: $("#status"),
  layers: $("#layers"), library: $("#library"),
};

const app = { playing: false, layers: [], nextId: 1, focused: null };

function wsUrl() {
  const override = new URLSearchParams(window.location.search).get("ws");
  if (override) return override;
  const url = new URL(window.location.href);
  url.protocol = url.protocol === "https:" ? "wss:" : "ws:";
  url.pathname = "/";
  url.search = "";
  url.hash = "";
  return url.toString();
}

const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name);

// --- one mixer layer -------------------------------------------------------

class Layer {
  constructor({ mode = "ambience", promptA = "", promptB = "" } = {}) {
    this.id = app.nextId++;
    this.mode = mode;
    // The layer's input recording ({ interleaved, channels, seconds },
    // 48 kHz stereo, up to MAX_INPUT_S) or null for pure text-to-audio.
    // It is fitted to the canvas per mode (fitInput) at session start and
    // every regeneration renoises that fixed recording.
    this.input = null;
    this.inputLabel = "";
    this.denoise = DENOISE_DEFAULT;
    this.recorder = null;
    this.blurTimer = null;
    this.seed = randomSeed();
    this.gain = 0.8;
    this.muted = false;
    this.soloed = false;
    this.status = "idle";
    this.message = "";
    this.tickMs = null;
    this.player = null;
    this.analyser = null;
    this.timer = null;
    this.playhead = null;
    this.onset = null; // { t0 } while waiting for a trigger to sound
    this.latencyMs = null;
    this.gen = 0;
    this.wire = new LayerWire(RemoteBackend, {
      wsUrl: wsUrl(),
      sliceWorkerUrl: "/sdk/sliceDecoder.worker.js",
      onSlice: (d) => this.applySlice(d),
      onClose: () => this.setStatus("error", "connection lost"),
    });
    this.buildDom(promptA, promptB);
  }

  get running() { return this.status === "ready" || this.status === "connecting"; }
  get ready() { return this.status === "ready"; }
  get index() { return app.layers.indexOf(this); }

  buildDom(promptA, promptB) {
    const root = document.createElement("article");
    root.className = "layer";
    root.innerHTML = `
      <div class="layer-head">
        <span class="dot"></span>
        <b class="layer-name"></b>
        <div class="seg" role="group" aria-label="Mode">
          <button type="button" data-mode="ambience">Ambience</button>
          <button type="button" data-mode="oneshot">One-shot</button>
        </div>
        <span class="chip src"></span>
        <span class="spacer"></span>
        <button type="button" class="mute" title="Mute">M</button>
        <button type="button" class="solo" title="Solo">S</button>
        <input class="gain" type="range" min="0" max="1.5" step="0.01" aria-label="Gain" title="Gain">
        <button type="button" class="reroll" title="New seed">Seed <em></em></button>
        <button type="button" class="save" title="Save this layer as WAV">WAV</button>
        <button type="button" class="power">Start</button>
        <button type="button" class="remove" title="Remove layer">&times;</button>
      </div>
      <label class="prompt"><span>A</span><input class="pa" type="text" spellcheck="false" placeholder="describe a sound (Enter sends)"></label>
      <div class="blend">
        <span class="end">A</span>
        <input class="blend-range" type="range" min="0" max="1" step="0.005" value="0" aria-label="Blend A to B">
        <span class="end">B</span>
        <output class="blend-value">0.00</output>
      </div>
      <label class="prompt"><span>B</span><input class="pb" type="text" spellcheck="false" placeholder="optional: a second sound to blend toward"></label>
      <div class="input-row">
        <span class="input-label">Input</span>
        <button type="button" class="rec">Record</button>
        <label class="load">Load file<input class="file" type="file" accept="audio/*"></label>
        <button type="button" class="clear">Clear</button>
        <canvas class="input-scope" height="36" aria-label="Input waveform"></canvas>
        <span class="input-note"></span>
      </div>
      <div class="denoise-row">
        <label class="denoise">denoise <input class="denoise-range" type="range" min="0.1" max="1" step="0.01"><em></em></label>
        <span class="hint">low = keeps the recording, high = only the prompt</span>
      </div>
      <div class="layer-foot">
        <span class="oneshot-ctl">
          <button type="button" class="trigger">Trigger <kbd></kbd></button>
          <span class="latency"></span>
        </span>
        <span class="tick"></span>
        <span class="msg"></span>
      </div>
      <canvas class="scope" height="150" aria-label="Layer waveform and spectrum"></canvas>`;
    this.el = root;
    this.q = (s) => $(s, root);
    this.q(".pa").value = promptA;
    this.q(".pb").value = promptB;
    this.q(".gain").value = String(this.gain);
    this.q(".denoise-range").value = String(this.denoise);

    for (const btn of root.querySelectorAll(".seg button")) {
      btn.addEventListener("click", () => this.setMode(btn.dataset.mode));
    }
    for (const sel of [".pa", ".pb"]) {
      const input = this.q(sel);
      input.addEventListener("focus", () => { app.focused = input; renderFocus(); });
      input.addEventListener("input", () => this.markDirty());
      input.addEventListener("keydown", (ev) => { if (ev.key === "Enter") this.sendPrompts(); });
      input.addEventListener("blur", () => {
        window.clearTimeout(this.blurTimer);
        this.blurTimer = window.setTimeout(() => this.sendPrompts(), BLUR_SEND_MS);
      });
    }
    this.q(".blend-range").addEventListener("input", (ev) => {
      const v = Number(ev.target.value);
      this.q(".blend-value").textContent = v.toFixed(2);
      this.wire.setBlend(v);
    });
    this.q(".denoise-range").addEventListener("input", (ev) => {
      this.denoise = Number(ev.target.value);
      this.wire.setDenoise(this.denoise);
      this.render();
    });
    this.q(".rec").addEventListener("click", () => void this.record());
    this.q(".file").addEventListener("change", (ev) => void this.loadFile(ev.target));
    this.q(".clear").addEventListener("click", () => this.setInput(null));
    this.q(".gain").addEventListener("input", (ev) => { this.gain = Number(ev.target.value); applyGains(); });
    this.q(".mute").addEventListener("click", () => { this.muted = !this.muted; applyGains(); });
    this.q(".solo").addEventListener("click", () => { this.soloed = !this.soloed; applyGains(); });
    this.q(".reroll").addEventListener("click", () => this.reroll());
    this.q(".save").addEventListener("click", () => this.saveWav());
    this.q(".power").addEventListener("click", () => (this.running ? void this.stop() : void this.start()));
    this.q(".remove").addEventListener("click", () => removeLayer(this));
    this.q(".trigger").addEventListener("click", () => this.trigger());
  }

  setStatus(status, message = "") {
    this.status = status;
    this.message = message;
    this.render();
    renderGlobal();
  }

  // A prompt field whose text has not reached the server yet.
  markDirty() {
    const sent = { ".pa": this.wire.promptA, ".pb": this.wire.promptB };
    for (const [sel, value] of Object.entries(sent)) {
      const input = this.q(sel);
      input.classList.toggle("dirty", this.ready && input.value.trim() !== (value || ""));
    }
  }

  render() {
    const i = this.index;
    this.q(".layer-name").textContent = `Layer ${i + 1}`;
    this.q(".dot").className = `dot dot-${this.status}`;
    for (const btn of this.el.querySelectorAll(".seg button")) {
      btn.classList.toggle("on", btn.dataset.mode === this.mode);
    }
    const src = this.q(".src");
    src.textContent = this.input ? "text + input" : "text";
    src.classList.toggle("mic", !!this.input);
    this.q(".mute").classList.toggle("on", this.muted);
    this.q(".solo").classList.toggle("on", this.soloed);
    $("em", this.q(".reroll")).textContent = String(this.seed);
    const power = this.q(".power");
    power.textContent = this.running ? "Stop" : "Start";
    power.classList.toggle("on", this.running);
    this.q(".save").disabled = !this.ready;
    this.q(".blend-range").disabled = !this.ready;
    this.el.classList.toggle("has-input", !!this.input);
    $("em", this.q(".denoise")).textContent = this.denoise.toFixed(2);
    const rec = this.q(".rec");
    rec.textContent = this.recorder ? "Stop" : "Record";
    rec.classList.toggle("on", !!this.recorder);
    this.q(".clear").disabled = !this.input;
    if (!this.recorder) {
      this.q(".input-note").textContent = this.input
        ? `${this.inputLabel}: ${fitInput(this.input, this.mode).note}`
        : `none (record or load up to ${MODES[this.mode].duration} s)`;
    }
    this.el.classList.toggle("is-oneshot", this.mode === "oneshot");
    $("kbd", this.q(".trigger")).textContent = String(i + 1);
    this.q(".trigger").disabled = !this.ready;
    this.q(".latency").textContent = this.latencyMs == null
      ? "" : `trigger to sound ${Math.round(this.latencyMs)} ms`;
    this.q(".tick").textContent = this.tickMs == null ? "" : `tick ${this.tickMs.toFixed(1)} ms`;
    this.q(".msg").textContent = this.status === "ready" ? "" : (this.message || this.status);
  }

  setMode(mode) {
    if (mode === this.mode) return;
    this.mode = mode;
    this.latencyMs = null;
    this.render();
    this.drawInput();
    if (this.running) void this.start(); // the canvas length is a session property
  }

  async start() {
    await this.stop();
    const gen = this.gen;
    this.setStatus("connecting", "starting session...");
    try {
      await this.wire.open({
        mode: this.mode,
        promptA: this.q(".pa").value.trim() || "Steady heavy rain on a tin roof",
        promptB: this.q(".pb").value.trim(),
        blend: Number(this.q(".blend-range").value),
        denoise: this.denoise,
        seed: this.seed,
        input: this.input,
      });
      if (gen !== this.gen) return;
      const remote = this.wire.remote;
      const player = new AudioPlayer({ workletUrl: "/sdk/audio-worklet.js?v=5" });
      this.player = player;
      await player.init(remote.initialBuffer, remote.channels);
      await player.resume();
      const analyser = player.ctx.createAnalyser();
      analyser.fftSize = 2048;
      analyser.smoothingTimeConstant = 0.55;
      player.node.connect(analyser);
      this.analyser = analyser;
      this.freq = new Uint8Array(analyser.frequencyBinCount);
      this.time = new Float32Array(analyser.fftSize);
      this.playhead = new VirtualPlayhead(this.wire.duration);
      if (this.mode === "oneshot") {
        // Parked until a trigger; the server still sweeps the whole clip.
        player.setLoop(false);
        player.seek(player.duration);
      }
      this.timer = window.setInterval(() => this.report(), PARAMS_TICK_MS);
      this.setStatus("ready");
      this.markDirty();
      applyGains();
    } catch (err) {
      await this.stop();
      this.setStatus("error", err instanceof Error ? err.message : "start failed");
    }
  }

  async stop() {
    this.gen += 1;
    if (this.timer != null) window.clearInterval(this.timer);
    this.timer = null;
    this.wire.close();
    const player = this.player;
    this.player = null;
    this.analyser = null;
    this.tickMs = null;
    this.onset = null;
    try { await player?.close(); } catch {}
    if (this.status !== "idle") this.setStatus("idle");
  }

  report() {
    if (!this.ready || !this.player) return;
    const pos = this.mode === "oneshot" ? this.playhead.position() : this.player.positionSec;
    this.wire.report(pos);
    this.render();
  }

  applySlice(d) {
    const player = this.player;
    if (!player || d.epoch !== player.swapCount) return;
    if (typeof d.tickMs === "number" && Number.isFinite(d.tickMs)) this.tickMs = d.tickMs;
    const startFrame = Math.floor(d.startSample);
    if (d.flags === SLICE_FLAG_DELTA) player.addDelta(startFrame, d.audio);
    else player.patch(startFrame, d.audio);
  }

  // Send whatever A and B hold now. Enter, blur (debounced) and example
  // cards all land here; the wire skips a send when nothing changed.
  sendPrompts() {
    window.clearTimeout(this.blurTimer);
    if (this.ready) this.wire.setPrompts(this.q(".pa").value, this.q(".pb").value);
    this.markDirty();
  }

  reroll() {
    this.seed = randomSeed();
    this.wire.setSeed(this.seed);
    this.render();
  }

  trigger() {
    if (!this.ready || this.mode !== "oneshot") return;
    this.playhead.restart();
    this.player.seek(0);
    this.onset = { t0: performance.now() };
  }

  // --- input audio ---------------------------------------------------------

  // A new input (or null to clear) is a new session source, so a running
  // layer restarts on it; prompt, blend, seed and denoise carry over.
  setInput(input, label = "") {
    this.input = input;
    this.inputLabel = label;
    this.render();
    this.drawInput();
    if (this.running) void this.start();
  }

  async record() {
    if (this.recorder) {
      this.recorder.stop();
      return;
    }
    let stream;
    try {
      stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    } catch {
      this.q(".input-note").textContent = "no microphone access; load a file instead";
      return;
    }
    const rec = new MediaRecorder(stream);
    this.recorder = rec;
    const chunks = [];
    const maxS = MODES[this.mode].duration;
    const timeout = setTimeout(() => rec.state === "recording" && rec.stop(), maxS * 1000);
    rec.ondataavailable = (ev) => chunks.push(ev.data);
    rec.onstop = async () => {
      clearTimeout(timeout);
      stream.getTracks().forEach((t) => t.stop());
      this.recorder = null;
      try {
        this.setInput(await decodeInput(await new Blob(chunks).arrayBuffer()), "recording");
      } catch {
        this.render();
        this.q(".input-note").textContent = "could not decode the recording";
      }
    };
    rec.start();
    this.render();
    this.q(".input-note").textContent = `recording (up to ${maxS} s), click Stop to finish`;
  }

  async loadFile(picker) {
    const file = picker.files?.[0];
    picker.value = "";
    if (!file) return;
    try {
      this.setInput(await decodeInput(await file.arrayBuffer()), file.name);
    } catch {
      this.q(".input-note").textContent = "could not decode that file";
    }
  }

  // The input as the session gets it: fitted to this mode's canvas.
  drawInput() {
    const c = this.q(".input-scope");
    const ctx = c.getContext("2d");
    const w = (c.width = Math.max(1, Math.round(c.clientWidth * devicePixelRatio)));
    const h = (c.height = Math.max(1, Math.round(c.clientHeight * devicePixelRatio)));
    ctx.fillStyle = css("--scope-bg");
    ctx.fillRect(0, 0, w, h);
    if (!this.input) return;
    const { pcm } = fitInput(this.input, this.mode);
    const frames = pcm.length / CHANNELS;
    ctx.fillStyle = css("--mic");
    for (let x = 0; x < w; x++) {
      const f0 = Math.floor((x / w) * frames);
      const f1 = Math.floor(((x + 1) / w) * frames);
      let peak = 0;
      for (let f = f0; f < f1; f += 2) peak = Math.max(peak, Math.abs(pcm[f * CHANNELS]));
      const bar = Math.max(1, Math.min(1, peak) * h);
      ctx.fillRect(x, (h - bar) / 2, 1, bar);
    }
  }

  // Called every animation frame: trigger onset detection + drawing.
  frame() {
    if (this.analyser && this.onset) {
      this.analyser.getFloatTimeDomainData(this.time);
      let sum = 0;
      for (let i = 0; i < this.time.length; i++) sum += this.time[i] * this.time[i];
      const rms = Math.sqrt(sum / this.time.length);
      const waited = performance.now() - this.onset.t0;
      if (rms > ONSET_RMS) {
        const ctx = this.player.ctx;
        const outMs = 1000 * ((ctx.baseLatency || 0) + (ctx.outputLatency || 0));
        this.latencyMs = waited + outMs;
        this.onset = null;
        this.render();
      } else if (waited > this.wire.duration * 1000) {
        this.onset = null; // nothing audible in the clip yet
      }
    }
    this.draw();
  }

  draw() {
    const c = this.q(".scope");
    const ctx = c.getContext("2d");
    const w = Math.max(1, Math.round(c.clientWidth * devicePixelRatio));
    const h = Math.max(1, Math.round(c.clientHeight * devicePixelRatio));
    if (c.width !== w) c.width = w;
    if (c.height !== h) c.height = h;
    ctx.fillStyle = css("--scope-bg");
    ctx.fillRect(0, 0, w, h);
    const waveH = h * 0.62;
    const mirror = this.player?.getMirror();
    if (mirror && mirror.length) {
      const ch = this.wire.remote?.channels ?? CHANNELS;
      const frames = Math.floor(mirror.length / ch);
      const per = Math.max(1, Math.floor(frames / w));
      ctx.fillStyle = css("--wave");
      for (let x = 0; x < w; x++) {
        const f0 = Math.floor((x / w) * frames);
        let peak = 0;
        for (let f = f0; f < f0 + per && f < frames; f += 3) {
          const v = Math.abs(mirror[f * ch]);
          if (v > peak) peak = v;
        }
        const bar = Math.max(1, Math.min(1, peak) * waveH * 0.92);
        ctx.fillRect(x, (waveH - bar) / 2, 1, bar);
      }
      const dur = this.player.duration || 1;
      const px = (Math.min(this.player.positionSec, dur) / dur) * w;
      ctx.fillStyle = css("--accent");
      ctx.fillRect(px - devicePixelRatio, 0, 2 * devicePixelRatio, waveH);
    }
    if (this.analyser) {
      this.analyser.getByteFrequencyData(this.freq);
      const bins = 112;
      const bw = w / bins;
      const specH = h - waveH;
      ctx.fillStyle = css("--spec");
      for (let i = 0; i < bins; i++) {
        // log-spaced bins so the low end is more than one bar
        const idx = Math.min(this.freq.length - 1, Math.floor(Math.pow(this.freq.length, i / bins)));
        const v = this.freq[idx] / 255;
        ctx.fillRect(i * bw, h - v * specH, Math.max(1, bw - 1), v * specH);
      }
    }
  }

  saveWav() {
    const mirror = this.player?.getMirror();
    if (!mirror) return;
    const ch = this.wire.remote?.channels ?? CHANNELS;
    const slug = (this.q(".pa").value || "layer").toLowerCase().replace(/[^a-z0-9]+/g, "_").slice(0, 40);
    download(encodeWav(mirror, ch, SAMPLE_RATE), `sfx_layer${this.index + 1}_${slug}_seed${this.seed}.wav`);
  }
}

// --- the board -------------------------------------------------------------

function effectiveGain(layer) {
  const anySolo = app.layers.some((l) => l.soloed);
  if (layer.muted || (anySolo && !layer.soloed)) return 0;
  return layer.gain;
}

function applyGains() {
  for (const layer of app.layers) {
    // AudioPlayer's output gain node: the last stage before the speakers.
    const out = layer.player?._masterOut;
    if (out) out.gain.setTargetAtTime(effectiveGain(layer), layer.player.ctx.currentTime, 0.015);
    layer.render();
  }
}

function addLayer(opts) {
  if (app.layers.length >= MAX_LAYERS) return null;
  const layer = new Layer(opts);
  app.layers.push(layer);
  els.layers.append(layer.el);
  app.layers.forEach((l) => l.render());
  layer.drawInput();
  renderGlobal();
  if (app.playing) void layer.start();
  return layer;
}

function removeLayer(layer) {
  void layer.stop();
  app.layers = app.layers.filter((l) => l !== layer);
  layer.el.remove();
  if (app.focused && layer.el.contains(app.focused)) app.focused = null;
  app.layers.forEach((l) => l.render());
  renderGlobal();
}

function renderGlobal() {
  els.play.textContent = app.playing ? "Stop all" : "Play";
  els.play.classList.toggle("on", app.playing);
  els.add.disabled = app.layers.length >= MAX_LAYERS;
  els.add.textContent = `+ Layer (${app.layers.length}/${MAX_LAYERS})`;
  els.saveMix.disabled = !app.layers.some((l) => l.ready);
  const live = app.layers.filter((l) => l.ready).length;
  const errors = app.layers.filter((l) => l.status === "error").length;
  els.status.textContent = app.playing
    ? `${live} of ${app.layers.length} layers live${errors ? `, ${errors} failed` : ""}`
    : "idle";
}

function renderFocus() {
  for (const input of document.querySelectorAll("input[type=text]")) {
    input.classList.toggle("focused", input === app.focused);
  }
}

function saveMix() {
  const live = app.layers.filter((l) => l.ready && l.player?.getMirror());
  if (!live.length) return;
  const seconds = Math.max(...live.map((l) => l.wire.duration));
  const mix = mixLayers(live.map((l) => ({
    mirror: l.player.getMirror(),
    channels: l.wire.remote?.channels ?? CHANNELS,
    gain: effectiveGain(l),
    loop: l.mode === "ambience",
  })), seconds);
  download(encodeWav(mix, CHANNELS, SAMPLE_RATE), `sfx_mix_${live.length}layers.wav`);
}

function download(buffer, name) {
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([buffer], { type: "audio/wav" }));
  a.download = name;
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 1000);
}

// --- example library -------------------------------------------------------

function renderLibrary() {
  els.library.replaceChildren(...LIBRARY.map(({ category, items }) => {
    const group = document.createElement("div");
    group.className = "lib-group";
    const h = document.createElement("h3");
    h.textContent = category;
    const cards = document.createElement("div");
    cards.className = "cards";
    for (const { text } of items) {
      const card = document.createElement("button");
      card.type = "button";
      card.className = "card";
      card.textContent = text;
      card.addEventListener("click", () => fillPrompt(text));
      cards.append(card);
    }
    group.append(h, cards);
    return group;
  }));
}

// A card goes into the focused prompt slot (A or B of the selected layer;
// layer 1's A by default) and is sent to the backend at once.
function fillPrompt(text) {
  let target = app.focused && document.body.contains(app.focused) ? app.focused : null;
  if (!target) {
    const first = app.layers[0] ?? addLayer({});
    if (!first) return;
    target = $(".pa", first.el);
  }
  target.value = text;
  app.focused = target;
  renderFocus();
  app.layers.find((l) => l.el.contains(target))?.sendPrompts();
}

// --- input audio -----------------------------------------------------------

// Decode a recorded or loaded sound to 48 kHz stereo interleaved PCM,
// kept up to MAX_INPUT_S (the longest canvas); fitInput fits it to the
// layer's mode at session start.
async function decodeInput(arrayBuffer) {
  const ctx = new OfflineAudioContext(CHANNELS, SAMPLE_RATE, SAMPLE_RATE);
  const decoded = await ctx.decodeAudioData(arrayBuffer);
  const frames = Math.min(decoded.length, Math.round(MAX_INPUT_S * SAMPLE_RATE));
  const l = decoded.getChannelData(0);
  const r = decoded.numberOfChannels > 1 ? decoded.getChannelData(1) : l;
  const interleaved = new Float32Array(frames * CHANNELS);
  for (let i = 0; i < frames; i++) {
    interleaved[2 * i] = l[i];
    interleaved[2 * i + 1] = r[i];
  }
  return { interleaved, channels: CHANNELS, seconds: frames / SAMPLE_RATE };
}

// --- wiring ----------------------------------------------------------------

function setPlaying(on) {
  app.playing = on;
  for (const layer of app.layers) {
    if (on && !layer.running) void layer.start();
    if (!on) void layer.stop();
  }
  renderGlobal();
}

els.play.addEventListener("click", () => setPlaying(!app.playing));
els.add.addEventListener("click", () => {
  const layer = addLayer({ mode: "oneshot", promptA: "Heavy metal door slam, single impact, reverberant warehouse" });
  if (layer) { app.focused = $(".pa", layer.el); renderFocus(); }
});
els.saveMix.addEventListener("click", saveMix);
window.addEventListener("keydown", (ev) => {
  const t = ev.target;
  if (t instanceof HTMLInputElement || t instanceof HTMLSelectElement || ev.repeat) return;
  const n = "1234".indexOf(ev.key);
  if (n >= 0 && app.layers[n]) app.layers[n].trigger();
});
window.addEventListener("beforeunload", () => app.layers.forEach((l) => l.wire.close()));

function loop() {
  for (const layer of app.layers) layer.frame();
  requestAnimationFrame(loop);
}

// Default scene: one ambience layer, so Play makes sound within seconds.
// Rain blending into a storm is the first thing to try.
renderLibrary();
addLayer({
  mode: "ambience",
  promptA: "Steady heavy rain on a tin roof",
  promptB: "Thunderstorm with rolling thunder and heavy rain",
});
app.focused = $(".pa", app.layers[0].el);
renderFocus();
loop();
