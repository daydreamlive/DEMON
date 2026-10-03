import { AudioPlayer, RemoteBackend, SLICE_FLAG_DELTA } from "/sdk/demon-client.js";

// state
const DEFAULT_PROMPT =
  "bpm is 140. key is F, and scale is minor. Darkwave / Coldwave. " +
  "Gothic synth textures, driving bass, cavernous drums.";
const PARAMS_TICK_MS = 100;
const KNOB_PREFIX = "minimax_";
const POINT_COUNT = 256;
// MiniMax ignores the upload (no audio encoder). The SDK always sends a PCM
// frame, which text_only would leave unread, so upload a short silent stub.
const STUB_SAMPLE_RATE = 48000;
const STUB_SECONDS = 2;
const STUB_CHANNELS = 2;
// A healthy session commits a chunk every few s; a long gap = piece ended, tape looping.
const ENDED_AFTER_MS = 15000;
const state = {
  remote: null, player: null, analyser: null, freq: null, wave: null,
  status: "idle", slices: 0, receivedFrames: 0, lastSliceAt: null, rms: 0,
  values: {}, paramsTimer: null, error: "", gen: 0,
};

// DOM
const els = {
  canvas: document.querySelector("#scene"),
  status: document.querySelector("#status"),
  prompt: document.querySelector("#prompt"),
  lyrics: document.querySelector("#lyrics"),
  duration: document.querySelector("#duration"),
  start: document.querySelector("#start"),
  send: document.querySelector("#send"),
  reconnect: document.querySelector("#reconnect"),
  knobs: document.querySelector("#knobs"),
};
els.prompt.value = DEFAULT_PROMPT;

function setStatus(status) {
  state.status = status;
  els.start.textContent = status === "idle" || status === "error" ? "Start" : "Stop";
  els.send.disabled = !state.player;
  els.reconnect.disabled = !state.player;
}

function renderStatusLine() {
  if (state.status === "error" || state.status === "idle" || state.status === "connecting") {
    els.status.textContent = state.error ? `error: ${state.error}` : state.status;
    return;
  }
  if (!state.player || state.slices === 0) {
    els.status.textContent = "waiting for first audio (~10 s)";
    return;
  }
  if (performance.now() - state.lastSliceAt > ENDED_AFTER_MS) {
    els.status.textContent = "piece ended (tape loops): Reconnect for a new one";
    return;
  }
  const received = state.receivedFrames / state.remote.sampleRate;
  const ahead = Math.max(0, received - state.player.positionSec);
  els.status.textContent = `playing, ${ahead.toFixed(1)} s buffered, ${state.slices} slices`;
}

// DEMON connection
function wsUrl() {
  const override = new URLSearchParams(location.search).get("ws");
  return override || `${location.protocol === "https:" ? "wss:" : "ws:"}//${location.host}/`;
}

function buildConfig() {
  const prompt = els.prompt.value.trim() || DEFAULT_PROMPT;
  const config = { telemetry_version: 1, backend: "minimax", prompt, prompt_b: prompt };
  const lyrics = els.lyrics.value.trim();
  if (lyrics) config.minimax_lyrics = lyrics;
  const duration = Number(els.duration.value);
  if (els.duration.value !== "" && Number.isFinite(duration) && duration > 0) {
    config.minimax_duration_s = duration;
  }
  return config;
}

function defaultKnobValue(entry) {
  if (entry.default !== undefined) return entry.default;
  if (entry.type === "bool") return false;
  if (entry.type === "enum") return entry.options?.[0] ?? "";
  return entry.min ?? 0;
}

function handleSlice(event) {
  const detail = event.detail;
  const player = state.player;
  if (!player || detail.epoch !== player.swapCount) return;
  // Playback starts on the first slice so the playhead stays behind the frontier.
  if (state.slices === 0) void player.resume();
  const start = Math.floor(detail.startSample);
  if (detail.flags === SLICE_FLAG_DELTA) player.addDelta(start, detail.audio);
  else player.patch(start, detail.audio);
  state.receivedFrames = Math.max(state.receivedFrames, start + detail.audio.length / state.remote.channels);
  state.slices += 1;
  state.lastSliceAt = performance.now();
}

function sendParams() {
  if (!state.remote || !state.player) return;
  state.remote.sendParams(state.values, state.player.positionSec);
}

// disconnect() bumps state.gen, so a connect() still awaiting drops what it built.
async function connect() {
  const gen = state.gen;
  const remote = new RemoteBackend(
    wsUrl(), new Float32Array(STUB_SAMPLE_RATE * STUB_SECONDS * STUB_CHANNELS), STUB_CHANNELS,
    buildConfig(), { sliceWorkerUrl: "/sdk/sliceDecoder.worker.js" },
  );
  state.remote = remote;
  remote.addEventListener("slice", handleSlice);
  remote.addEventListener("close", () => {
    if (remote.closedByUser || state.remote !== remote) return;
    state.error = "connection lost";
    setStatus("error");
  });
  await remote.connect();
  if (gen !== state.gen) return void remote.close();
  if (!remote.initialBuffer) throw new Error("server sent no initial buffer");
  console.info("minimax session", remote.backendSessionId);

  const manifest = remote.knobManifest?.knobs ?? {};
  const knobs = Object.entries(manifest).filter(([name]) => name.startsWith(KNOB_PREFIX));
  state.values = Object.fromEntries(knobs.map(([name, entry]) => [name, defaultKnobValue(entry)]));
  // Steer a continuous stream (mask the end token) and pivot hard on a new
  // prompt (keep 2.5 s of history), as the original page did.
  if ("minimax_endless" in state.values) state.values.minimax_endless = true;
  if ("minimax_reprompt_history_s" in state.values) state.values.minimax_reprompt_history_s = 2.5;
  renderKnobs(knobs);

  const player = new AudioPlayer({ workletUrl: "/sdk/audio-worklet.js" });
  try {
    await player.init(remote.initialBuffer, remote.channels);
  } catch (err) {
    try { await player.close(); } catch {}
    throw err;
  }
  if (gen !== state.gen) {
    try { await player.close(); } catch {}
    return void remote.close();
  }
  state.player = player;
  attachAnalyser(player);
  state.paramsTimer = window.setInterval(sendParams, PARAMS_TICK_MS);
  setStatus("playing");
}

async function disconnect() {
  state.gen += 1;
  window.clearInterval(state.paramsTimer);
  state.paramsTimer = null;
  try { await state.player?.close(); } catch {}
  try { state.remote?.close(); } catch {}
  Object.assign(state, { remote: null, player: null, analyser: null, rms: 0 });
  Object.assign(state, { slices: 0, receivedFrames: 0, lastSliceAt: null });
  setStatus("idle");
}

async function restart() {
  await disconnect(); // tear down any session left behind by an error
  state.error = "";
  setStatus("connecting");
  const gen = state.gen;
  try {
    await connect();
  } catch (err) {
    if (gen !== state.gen) return; // Stop was pressed mid-connect
    await disconnect();
    state.error = err instanceof Error ? err.message : String(err);
    setStatus("error");
  }
}

function renderKnobs(knobs) {
  els.knobs.replaceChildren(...knobs.map(([name, entry]) => knobControl(name, entry)));
}

function knobControl(name, entry) {
  const label = document.createElement("label");
  if (entry.description) label.title = entry.description;
  const value = document.createElement("span");
  const show = (v) => { value.textContent = ` ${typeof v === "number" ? +v.toFixed(2) : v}`; };
  const commit = (v) => { state.values = { ...state.values, [name]: v }; show(v); sendParams(); };
  let input;
  if (entry.type === "bool") {
    input = document.createElement("input");
    input.type = "checkbox";
    input.checked = Boolean(state.values[name]);
    input.addEventListener("change", () => commit(input.checked));
  } else if (entry.type === "enum") {
    input = document.createElement("select");
    for (const option of entry.options ?? []) input.append(new Option(String(option), String(option)));
    input.value = String(state.values[name]);
    input.addEventListener("change", () => commit(input.value));
  } else {
    const min = entry.min ?? 0;
    const max = entry.max ?? 1;
    input = document.createElement("input");
    Object.assign(input, { type: "range", min: String(min), max: String(max) });
    input.step = entry.type === "int" ? "1" : String((max - min) / 200);
    input.value = String(state.values[name]);
    input.addEventListener("input", () => commit(Number(input.value)));
  }
  show(state.values[name]);
  label.append(name.slice(KNOB_PREFIX.length).replace(/_/g, " "), value, input);
  return label;
}

// audio analyser
function attachAnalyser(player) {
  const analyser = player.ctx.createAnalyser();
  analyser.fftSize = 1024;
  analyser.smoothingTimeConstant = 0.75;
  player.node.connect(analyser);
  state.analyser = analyser;
  state.freq = new Uint8Array(analyser.frequencyBinCount);
  state.wave = new Float32Array(analyser.fftSize);
}

function bandEnergy(from, to) {
  if (!state.analyser) return 0;
  let sum = 0;
  for (let i = from; i < to; i++) sum += state.freq[i];
  return sum / ((to - from) * 255);
}

function readAnalyser() {
  if (!state.analyser) return;
  state.analyser.getByteFrequencyData(state.freq);
  state.analyser.getFloatTimeDomainData(state.wave);
  let sum = 0;
  for (const s of state.wave) sum += s * s;
  state.rms = Math.sqrt(sum / state.wave.length);
}

// ring canvas: a ring of points that swells with the bass and ripples with the mids
const ctx2d = els.canvas.getContext("2d");

function resize() {
  const dpr = Math.min(window.devicePixelRatio, 2);
  els.canvas.width = Math.round(window.innerWidth * dpr);
  els.canvas.height = Math.round(window.innerHeight * dpr);
}

function drawRing(t) {
  const { width, height } = els.canvas;
  ctx2d.fillStyle = "#07070b";
  ctx2d.fillRect(0, 0, width, height);
  const low = bandEnergy(1, 12);
  const mid = bandEnergy(12, 90);
  const high = bandEnergy(90, 400);
  const base = Math.min(width, height) * 0.28 * (1 + low * 0.45);
  const dot = Math.max(1.5, Math.min(width, height) * 0.004 * (1 + high * 2));
  ctx2d.fillStyle = `hsl(${225 - high * 180} 70% ${60 + high * 25}%)`;
  for (let i = 0; i < POINT_COUNT; i++) {
    const angle = (i / POINT_COUNT) * Math.PI * 2 + t * 0.12;
    const r = base * (1 + Math.sin(t * 3 + angle * 6) * mid * 0.15);
    ctx2d.fillRect(width / 2 + Math.cos(angle) * r, height / 2 + Math.sin(angle) * r, dot, dot);
  }
}

// render loop
function frame(ms) {
  readAnalyser();
  drawRing(ms / 1000);
  renderStatusLine();
  requestAnimationFrame(frame);
}

// event wiring
els.start.addEventListener("click", async () => {
  if (state.status !== "idle" && state.status !== "error") await disconnect();
  else await restart();
});
els.reconnect.addEventListener("click", () => void restart());
els.send.addEventListener("click", () => {
  const prompt = els.prompt.value.trim() || DEFAULT_PROMPT;
  state.remote?.sendPrompt(prompt, undefined, undefined, prompt);
});
window.addEventListener("resize", resize);
window.addEventListener("beforeunload", () => {
  try { state.remote?.close(); } catch {}
});
window.__demo = {
  get stats() {
    return {
      connected: Boolean(state.remote && state.player),
      slices: state.slices,
      positionSec: state.player?.positionSec ?? 0,
      rms: state.rms,
    };
  },
};

resize();
setStatus("idle");
requestAnimationFrame(frame);
