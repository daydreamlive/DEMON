import { AudioPlayer, RemoteBackend, SLICE_FLAG_DELTA } from "/sdk/demon-client.js";

// state
const SAMPLE_RATE = 48000;
const STUB_FRAMES = 9600; // the mrt2 create path ignores the handshake audio
const STUB_CHANNELS = 2;
const PARAMS_TICK_MS = 100;
const BAR_COUNT = 64;
const DEFAULT_PROMPT_A = "deep house, warm analog bass, crisp hats";
const DEFAULT_PROMPT_B = "ambient piano, slow strings, tape hiss";
const state = {
  remote: null, player: null, analyser: null, freq: null, wave: null,
  status: "idle", slices: 0, lastEndSec: 0, windowSec: 0, rms: 0,
  knobs: [], values: {}, paramsTimer: null, error: "", gen: 0,
};

// DOM
const els = {
  canvas: document.querySelector("#scene"),
  status: document.querySelector("#status"),
  promptA: document.querySelector("#prompt-a"),
  promptB: document.querySelector("#prompt-b"),
  blend: document.querySelector("#blend"),
  blendValue: document.querySelector("#blend-value"),
  knobs: document.querySelector("#knobs"),
  start: document.querySelector("#start"),
  send: document.querySelector("#send"),
};
els.promptA.value = DEFAULT_PROMPT_A;
els.promptB.value = DEFAULT_PROMPT_B;

function setStatus(status) {
  state.status = status;
  els.start.textContent = status === "idle" || status === "error" ? "Start" : "Stop";
  els.send.disabled = !state.player;
}

function renderStatusLine() {
  if (state.status === "error" || state.status === "idle" || state.status === "connecting") {
    els.status.textContent = state.error ? `error: ${state.error}` : state.status;
    return;
  }
  if (!state.player) {
    els.status.textContent = "ready";
    return;
  }
  if (state.rms <= 0.001) {
    els.status.textContent = `waiting for first audio (${state.slices} slices)`;
    return;
  }
  const play = state.windowSec > 0 ? state.player.positionSec % state.windowSec : state.player.positionSec;
  let ahead = state.lastEndSec - play;
  if (state.windowSec > 0) ahead = ((ahead % state.windowSec) + state.windowSec) % state.windowSec;
  els.status.textContent = `playing, ${ahead.toFixed(2)} s buffered, ${state.slices} slices`;
}

// DEMON connection
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

function promptPair() {
  const a = els.promptA.value.trim() || DEFAULT_PROMPT_A;
  const b = els.promptB.value.trim() || a;
  return [a, b];
}

function handleSlice(event) {
  const detail = event.detail;
  const player = state.player;
  if (!player || detail.epoch !== player.swapCount) return;
  const start = Math.floor(detail.startSample);
  if (detail.flags === SLICE_FLAG_DELTA) player.addDelta(start, detail.audio);
  else player.patch(start, detail.audio);
  state.slices += 1;
  state.lastEndSec = (start + detail.numSamples) / SAMPLE_RATE;
}

function sendParams() {
  if (!state.remote || !state.player) return;
  state.remote.sendParams(state.values, state.player.positionSec);
}

// disconnect() bumps state.gen, so a connect() still awaiting drops what it built.
async function connect() {
  const gen = state.gen;
  const [prompt, promptB] = promptPair();
  const config = { telemetry_version: 1, backend: "mrt2", prompt, prompt_b: promptB };
  const remote = new RemoteBackend(
    wsUrl(), new Float32Array(STUB_FRAMES * STUB_CHANNELS), STUB_CHANNELS, config,
    { sliceWorkerUrl: "/sdk/sliceDecoder.worker.js" },
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
  console.info("mrt2 session", remote.backendSessionId);
  setStatus("ready");

  const manifest = remote.knobManifest?.knobs ?? {};
  state.knobs = Object.entries(manifest)
    .filter(([name, entry]) => name.startsWith("mrt2_") && (entry.type === "float" || entry.type === "int"))
    .map(([name, entry]) => ({ name, entry }));
  state.values = Object.fromEntries(state.knobs.map(({ name, entry }) => [name, entry.default ?? entry.min ?? 0]));
  renderKnobs();

  const player = new AudioPlayer({ workletUrl: "/sdk/audio-worklet.js" });
  try {
    await player.init(remote.initialBuffer, remote.channels);
    await player.resume();
  } catch (err) {
    try { await player.close(); } catch {}
    throw err;
  }
  if (gen !== state.gen) {
    try { await player.close(); } catch {}
    return void remote.close();
  }
  state.player = player;
  state.windowSec = remote.initialBuffer.length / remote.channels / SAMPLE_RATE;
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
  Object.assign(state, { remote: null, player: null, analyser: null, freq: null, wave: null, slices: 0, rms: 0, lastEndSec: 0 });
  setStatus("idle");
}

function renderKnobs() {
  const nodes = state.knobs.map(({ name, entry }) => {
    const label = document.createElement("label");
    label.className = "knob-control";
    const heading = document.createElement("span");
    heading.className = "knob-heading";
    const caption = document.createElement("span");
    caption.className = "knob-label";
    caption.textContent = name.replace(/^mrt2_/, "").replace(/_/g, " ");
    const value = document.createElement("span");
    value.className = "knob-value";
    const input = document.createElement("input");
    input.type = "range";
    input.min = String(entry.min ?? 0);
    input.max = String(entry.max ?? 1);
    input.step = entry.type === "int" ? "1" : String(((entry.max ?? 1) - (entry.min ?? 0)) / 200);
    input.value = String(state.values[name]);
    input.dataset.knob = name;
    if (entry.description) label.title = entry.description;
    value.textContent = ` ${Number(input.value).toFixed(entry.type === "int" ? 0 : 2)}`;
    input.addEventListener("input", () => {
      state.values = { ...state.values, [name]: Number(input.value) };
      value.textContent = ` ${Number(input.value).toFixed(entry.type === "int" ? 0 : 2)}`;
      sendParams();
    });
    heading.append(caption, value);
    label.append(heading, input);
    return label;
  });
  els.knobs.replaceChildren(...nodes);
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

function readAnalyser() {
  if (!state.analyser) return;
  state.analyser.getByteFrequencyData(state.freq);
  state.analyser.getFloatTimeDomainData(state.wave);
  let sum = 0;
  for (const s of state.wave) sum += s * s;
  state.rms = Math.sqrt(sum / state.wave.length);
}

const ctx2d = els.canvas.getContext("2d");
function resize() {
  const dpr = Math.min(window.devicePixelRatio, 2);
  els.canvas.width = Math.round(window.innerWidth * dpr);
  els.canvas.height = Math.round(window.innerHeight * dpr);
}
function drawSpectrum() {
  const { width, height } = els.canvas;
  ctx2d.clearRect(0, 0, width, height);
  const hue = 210 - Number(els.blend.value) * 185; // A = cool blue, B = warm orange
  const bucket = Math.floor(384 / BAR_COUNT);
  const left = width > 760 ? width * 0.42 : width * 0.04;
  const barW = (width - left - width * 0.04) / BAR_COUNT;
  const baseline = height * 0.76;
  const gradient = ctx2d.createLinearGradient(0, baseline - height * 0.42, 0, baseline);
  gradient.addColorStop(0, `hsla(${hue} 78% 70% / 0.9)`);
  gradient.addColorStop(1, `hsla(${hue} 72% 46% / 0.42)`);
  ctx2d.strokeStyle = `hsla(${hue} 46% 66% / 0.1)`;
  ctx2d.lineWidth = 1;
  for (const y of [0.34, 0.55, 0.76]) {
    ctx2d.beginPath();
    ctx2d.moveTo(left, height * y);
    ctx2d.lineTo(width * 0.96, height * y);
    ctx2d.stroke();
  }
  ctx2d.strokeStyle = gradient;
  ctx2d.lineWidth = Math.max(2, barW * 0.43);
  ctx2d.lineCap = "round";
  ctx2d.shadowColor = `hsla(${hue} 85% 62% / 0.5)`;
  ctx2d.shadowBlur = Math.min(18, barW * 1.5);
  for (let i = 0; i < BAR_COUNT; i++) {
    let level = 0;
    if (state.freq) {
      for (let k = 0; k < bucket; k++) level += state.freq[i * bucket + k];
      level /= bucket * 255;
    }
    const h = Math.max(1, level * height * 0.42);
    const x = left + (i + 0.5) * barW;
    ctx2d.beginPath();
    ctx2d.moveTo(x, baseline);
    ctx2d.lineTo(x, baseline - h);
    ctx2d.stroke();
  }
  ctx2d.shadowBlur = 0;
}

// render loop
function frame() {
  readAnalyser();
  drawSpectrum();
  renderStatusLine();
  requestAnimationFrame(frame);
}

// event wiring
els.start.addEventListener("click", async () => {
  if (state.status !== "idle" && state.status !== "error") {
    await disconnect();
    return;
  }
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
});
els.send.addEventListener("click", () => {
  const [prompt, promptB] = promptPair();
  state.remote?.sendPrompt(prompt, undefined, undefined, promptB);
});
els.blend.addEventListener("input", () => {
  els.blendValue.textContent = Number(els.blend.value).toFixed(2);
  state.remote?.sendSetPromptBlend(Number(els.blend.value));
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
