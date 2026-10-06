import { AudioPlayer, RemoteBackend, SLICE_FLAG_DELTA } from "/sdk/demon-client.js";

// Stable Audio 3 small-sfx on the streaming ring. The ring re-renders one
// short looping canvas continuously, so a pad is just a prompt change: the
// loop turns into the new sound within a few generations. Banks pick the
// canvas length (a session property), so changing bank restarts the session.
const BANKS = {
  oneshots: {
    label: "One-shots (4 s)",
    duration: 4,
    pads: [
      "Heavy metal door slam, single impact, reverberant warehouse",
      "Glass bottle shattering on concrete",
      "Deep cinematic boom impact with long sub tail",
      "Futuristic laser blast, sharp energy pulse, arcade style",
      "Short UI confirm chime, clean bright notification",
      "Soft UI click, minimal interface tap",
      "Whoosh swipe, fast air movement past the microphone",
      "Rising tension riser, building synth swell",
      "Single gunshot outdoors with echo",
      "Wooden door creak, slow opening",
      "Footsteps on gravel, a few steps",
      "Magical sparkle shimmer, fantasy spell burst",
    ],
  },
  ambience: {
    label: "Ambience loops (20 s)",
    duration: 20,
    pads: [
      "Steady heavy rain on a tin roof",
      "Busy city street traffic ambience, distant horns",
      "Forest at dawn, birdsong and light wind",
      "Crowd murmur in a large indoor hall",
      "Idling diesel engine, constant rumble",
      "Ocean waves on a pebble beach",
      "Crackling campfire at night, crickets",
      "Spaceship interior hum, low drone and beeps",
      "Thunderstorm with rolling thunder and rain",
      "Cafe ambience, cups clinking, chatter",
      "Wind howling through a canyon",
      "Factory machinery, rhythmic industrial clanking",
    ],
  },
};
const PARAMS_TICK_MS = 80;
const STUB_RATE = 48000;
const STUB_CHANNELS = 2;

const $ = (sel) => document.querySelector(sel);
const els = {
  bank: $("#bank"), duration: $("#duration"), transport: $("#transport"),
  tick: $("#tick"), dot: $("#status-dot"), status: $("#status-text"),
  scope: $("#scope"), pads: $("#pads"), prompt: $("#prompt"),
  reroll: $("#reroll"), save: $("#save"), morph: $("#morph"),
  morphTarget: $("#morph-target"), morphValue: $("#morph-value"), knobs: $("#knobs"),
};

const state = {
  bank: "oneshots", prompts: {}, active: 0, target: null,
  remote: null, player: null, analyser: null, freq: null,
  status: "idle", message: "", tickMs: null,
  knobs: [], values: {}, paramsTimer: null, gen: 0,
};
for (const [key, bank] of Object.entries(BANKS)) state.prompts[key] = [...bank.pads];

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

const running = () => state.status === "ready" || state.status === "connecting";
const ready = () => state.status === "ready";
const prompts = () => state.prompts[state.bank];

function setStatus(status, message = "") {
  state.status = status;
  state.message = message;
  render();
}

function render() {
  els.transport.textContent = running() ? "Stop" : "Start";
  els.transport.classList.toggle("on", running());
  els.bank.disabled = running();
  els.duration.disabled = running();
  els.reroll.disabled = !ready();
  els.save.disabled = !ready();
  els.morph.disabled = !ready() || state.target == null;
  els.tick.textContent = state.tickMs == null ? "--.-" : Number(state.tickMs).toFixed(1);
  els.dot.className = `dot dot-${state.status}`;
  els.status.textContent = state.message || state.status;
  els.morphTarget.textContent =
    state.target == null ? "(shift-click a pad)" : `pad ${state.target + 1}`;
  [...els.pads.children].forEach((pad, i) => {
    pad.classList.toggle("active", i === state.active);
    pad.classList.toggle("target", i === state.target);
  });
}

function renderBanks() {
  els.bank.replaceChildren(...Object.entries(BANKS).map(([key, bank]) => {
    const opt = document.createElement("option");
    opt.value = key;
    opt.textContent = bank.label;
    return opt;
  }));
  els.bank.value = state.bank;
  els.duration.value = String(BANKS[state.bank].duration);
}

function renderPads() {
  els.pads.replaceChildren(...prompts().map((text, i) => {
    const pad = document.createElement("button");
    pad.type = "button";
    pad.className = "pad";
    const num = document.createElement("b");
    num.textContent = String(i + 1);
    const label = document.createElement("span");
    label.textContent = text;
    pad.append(num, label);
    pad.title = "click: play this sound. shift-click: set as morph target";
    pad.addEventListener("click", (ev) => (ev.shiftKey ? setTarget(i) : selectPad(i)));
    return pad;
  }));
  els.prompt.value = prompts()[state.active];
  render();
}

// --- session -------------------------------------------------------------

function sendPrompt() {
  if (!ready()) return;
  const a = prompts()[state.active];
  const b = state.target != null && state.target !== state.active
    ? prompts()[state.target] : undefined;
  state.remote.sendPrompt(a, undefined, undefined, b);
  els.morph.value = "0";
  els.morphValue.textContent = "0.00";
}

function selectPad(i) {
  state.active = i;
  if (state.target === i) state.target = null;
  els.prompt.value = prompts()[i];
  render();
  if (ready()) sendPrompt();
  else if (!running()) void start();
}

function setTarget(i) {
  state.target = i === state.active ? null : i;
  render();
  sendPrompt();
}

function sendParamsNow() {
  if (!ready() || !state.player) return;
  state.remote.sendParams(state.values, state.player.positionSec);
}

function readDuration() {
  const v = Number(els.duration.value);
  return Number.isFinite(v) && v > 0 ? Math.min(v, 60) : BANKS[state.bank].duration;
}

async function stop() {
  state.gen += 1;
  if (state.paramsTimer != null) window.clearInterval(state.paramsTimer);
  state.paramsTimer = null;
  try { await state.player?.close(); } catch {}
  try { state.remote?.close(); } catch {}
  Object.assign(state, { player: null, remote: null, analyser: null, tickMs: null });
  setStatus("idle");
}

async function start() {
  await stop();
  const gen = state.gen;
  setStatus("connecting", "loading model...");
  const duration = readDuration();
  const config = {
    telemetry_version: 1,
    backend: "sa3",
    prompt: prompts()[state.active],
    sa3_duration_s: duration,
  };
  try {
    // SA3 at full denoise never hears the source; ship silence of the
    // canvas length so the session geometry is the requested loop.
    const stub = new Float32Array(Math.round(duration * STUB_RATE) * STUB_CHANNELS);
    const remote = new RemoteBackend(wsUrl(), stub, STUB_CHANNELS, config,
      { sliceWorkerUrl: "/sdk/sliceDecoder.worker.js" });
    state.remote = remote;
    remote.addEventListener("slice", (event) => {
      const d = event.detail;
      const player = state.player;
      if (!player || d.epoch !== player.swapCount) return;
      if (typeof d.tickMs === "number" && Number.isFinite(d.tickMs)) state.tickMs = d.tickMs;
      const startFrame = Math.floor(d.startSample);
      if (d.flags === SLICE_FLAG_DELTA) player.addDelta(startFrame, d.audio);
      else player.patch(startFrame, d.audio);
    });
    remote.addEventListener("close", () => {
      if (remote.closedByUser || state.remote !== remote) return;
      setStatus("error", "connection lost");
    });
    await remote.connect();
    if (gen !== state.gen) return;
    if (!remote.initialBuffer) throw new Error("server sent no initial buffer");

    const manifest = remote.knobManifest?.knobs ?? {};
    state.knobs = Object.entries(manifest)
      .filter(([, e]) => e.type === "float" || e.type === "int")
      .map(([name, entry]) => ({ name, entry }));
    state.values = Object.fromEntries(
      Object.entries(manifest).map(([n, e]) => [n, defaultValue(e)]),
    );
    renderKnobs();

    const player = new AudioPlayer({ workletUrl: "/sdk/audio-worklet.js?v=5" });
    state.player = player;
    await player.init(remote.initialBuffer, remote.channels);
    await player.resume();
    const analyser = player.ctx.createAnalyser();
    analyser.fftSize = 2048;
    analyser.smoothingTimeConstant = 0.6;
    player.node.connect(analyser);
    state.analyser = analyser;
    state.freq = new Uint8Array(analyser.frequencyBinCount);

    state.paramsTimer = window.setInterval(() => { sendParamsNow(); render(); }, PARAMS_TICK_MS);
    setStatus("ready");
    if (state.target != null) sendPrompt();
  } catch (err) {
    await stop();
    setStatus("error", err instanceof Error ? err.message : "start failed");
  }
}

// --- knobs ---------------------------------------------------------------

function defaultValue(e) {
  if (e.default !== undefined) return e.default;
  if (e.type === "bool") return false;
  if (e.type === "enum") return e.options?.[0] ?? "";
  return e.min ?? 0;
}

function renderKnobs() {
  els.knobs.replaceChildren(...state.knobs.map(({ name, entry }) => {
    const cell = document.createElement("label");
    cell.className = "knob";
    if (entry.description) cell.title = entry.description;
    const isInt = entry.type === "int";
    const min = entry.min ?? 0;
    const max = entry.max ?? 1;
    const fmt = (v) => (isInt ? String(Math.round(v)) : Number(v).toFixed(2));
    const head = document.createElement("span");
    const val = document.createElement("em");
    head.textContent = name.replace(/^sa3_/, "").replace(/_/g, " ") + " ";
    head.append(val);
    const input = document.createElement("input");
    Object.assign(input, { type: "range", min, max, step: isInt ? 1 : (max - min) / 200 });
    input.value = String(state.values[name]);
    val.textContent = fmt(input.value);
    input.dataset.knob = name;
    input.addEventListener("input", () => {
      const v = isInt ? Math.round(Number(input.value)) : Number(input.value);
      state.values = { ...state.values, [name]: v };
      val.textContent = fmt(v);
      sendParamsNow();
    });
    cell.append(head, input);
    return cell;
  }));
}

function reroll() {
  const seed = state.knobs.find((k) => k.name === "seed");
  if (!seed) return;
  const lo = seed.entry.min ?? 0;
  const hi = seed.entry.max ?? 9999;
  const v = Math.floor(lo + Math.random() * (hi - lo + 1));
  const input = els.knobs.querySelector('input[data-knob="seed"]');
  if (input) {
    input.value = String(v);
    input.dispatchEvent(new Event("input"));
  } else {
    state.values = { ...state.values, seed: v };
    sendParamsNow();
  }
}

// --- save the current loop -------------------------------------------------

function saveWav() {
  const player = state.player;
  const mirror = player?.getMirror();
  if (!mirror) return;
  const ch = state.remote?.channels ?? STUB_CHANNELS;
  const rate = player.ctx?.sampleRate ?? STUB_RATE;
  const frames = Math.floor(mirror.length / ch);
  const buf = new ArrayBuffer(44 + frames * ch * 2);
  const dv = new DataView(buf);
  const str = (o, s) => [...s].forEach((c, i) => dv.setUint8(o + i, c.charCodeAt(0)));
  str(0, "RIFF"); dv.setUint32(4, 36 + frames * ch * 2, true); str(8, "WAVE");
  str(12, "fmt "); dv.setUint32(16, 16, true); dv.setUint16(20, 1, true);
  dv.setUint16(22, ch, true); dv.setUint32(24, rate, true);
  dv.setUint32(28, rate * ch * 2, true); dv.setUint16(32, ch * 2, true);
  dv.setUint16(34, 16, true); str(36, "data"); dv.setUint32(40, frames * ch * 2, true);
  for (let i = 0; i < frames * ch; i++) {
    dv.setInt16(44 + i * 2, Math.max(-1, Math.min(1, mirror[i])) * 32767, true);
  }
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([buf], { type: "audio/wav" }));
  const slug = prompts()[state.active].toLowerCase().replace(/[^a-z0-9]+/g, "_").slice(0, 40);
  a.download = `sfx_${slug}_seed${state.values.seed ?? 0}.wav`;
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 1000);
}

// --- visuals: loop waveform + playhead + live spectrum ---------------------

function draw() {
  const c = els.scope;
  const ctx = c.getContext("2d");
  const w = (c.width = Math.max(1, Math.round(c.clientWidth * devicePixelRatio)));
  const h = (c.height = Math.max(1, Math.round(c.clientHeight * devicePixelRatio)));
  const css = getComputedStyle(document.documentElement);
  ctx.fillStyle = css.getPropertyValue("--scope-bg");
  ctx.fillRect(0, 0, w, h);
  const player = state.player;
  const mirror = player?.getMirror();
  if (mirror && mirror.length) {
    const ch = state.remote?.channels ?? STUB_CHANNELS;
    const frames = Math.floor(mirror.length / ch);
    const per = Math.max(1, Math.floor(frames / w));
    ctx.fillStyle = css.getPropertyValue("--wave");
    for (let x = 0; x < w; x++) {
      const f0 = Math.floor((x / w) * frames);
      let peak = 0;
      for (let f = f0; f < f0 + per && f < frames; f += 4) {
        const v = Math.abs(mirror[f * ch]);
        if (v > peak) peak = v;
      }
      const bar = Math.max(1, peak * h * 0.9);
      ctx.fillRect(x, (h - bar) / 2, 1, bar);
    }
    const dur = player.duration || 1;
    const px = ((player.positionSec % dur) / dur) * w;
    ctx.fillStyle = css.getPropertyValue("--accent");
    ctx.fillRect(px - devicePixelRatio, 0, 2 * devicePixelRatio, h);
  }
  if (state.analyser) {
    state.analyser.getByteFrequencyData(state.freq);
    const bins = 96;
    const bw = w / bins;
    ctx.fillStyle = css.getPropertyValue("--spec");
    for (let i = 0; i < bins; i++) {
      // log-spaced bins so the low end is not a single bar
      const idx = Math.min(state.freq.length - 1, Math.floor(Math.pow(state.freq.length, i / bins)));
      const v = state.freq[idx] / 255;
      ctx.fillRect(i * bw, h - v * h * 0.35, bw - 1, v * h * 0.35);
    }
  }
  requestAnimationFrame(draw);
}

// --- wiring ----------------------------------------------------------------

els.transport.addEventListener("click", () => (running() ? void stop() : void start()));
els.bank.addEventListener("change", () => {
  state.bank = els.bank.value;
  state.active = 0;
  state.target = null;
  els.duration.value = String(BANKS[state.bank].duration);
  renderPads();
});
els.prompt.addEventListener("keydown", (ev) => {
  if (ev.key !== "Enter") return;
  const text = els.prompt.value.trim();
  if (!text) return;
  prompts()[state.active] = text;
  renderPads();
  sendPrompt();
});
els.reroll.addEventListener("click", reroll);
els.save.addEventListener("click", saveWav);
els.morph.addEventListener("input", () => {
  const v = Number(els.morph.value);
  els.morphValue.textContent = v.toFixed(2);
  state.remote?.sendSetPromptBlend(v);
});
window.addEventListener("keydown", (ev) => {
  const t = ev.target;
  if (t instanceof HTMLInputElement || t instanceof HTMLSelectElement) return;
  const n = "123456789".indexOf(ev.key);
  if (n >= 0 && n < prompts().length) selectPad(n);
  if (ev.key === "r") reroll();
});
window.addEventListener("beforeunload", () => { try { state.remote?.close(); } catch {} });

renderBanks();
renderPads();
draw();
