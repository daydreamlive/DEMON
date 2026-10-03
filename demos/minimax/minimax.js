import * as THREE from "three";
import { AudioPlayer, RemoteBackend, SLICE_FLAG_DELTA } from "/sdk/demon-client.js";

// --- state ---
const DEFAULT_PROMPT =
  "bpm is 140. key is F, and scale is minor. Darkwave / Coldwave. " +
  "Gothic synth textures, driving bass, cavernous drums.";
const PARAMS_TICK_MS = 100;
const KNOB_PREFIX = "minimax_";
// MiniMax ignores the upload (no audio encoder). The SDK always sends a PCM
// frame, which text_only would leave unread, so upload a short silent stub.
const STUB_SAMPLE_RATE = 48000;
const STUB_SECONDS = 2;
const STUB_CHANNELS = 2;
// A healthy session commits a chunk every few s; a long gap = piece ended, tape looping.
const ENDED_AFTER_MS = 15000;

const state = {
  remote: null, player: null, analyser: null, freq: null, wave: null,
  status: "idle", connected: false, slices: 0, receivedFrames: 0,
  lastSliceAt: null, values: {}, paramsTimer: null, rms: 0,
};

// --- DOM ---
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
  renderStatus();
}

function renderStatus() {
  const live = state.remote != null;
  els.start.textContent = live ? "Stop" : "Start";
  els.send.disabled = !state.connected;
  els.reconnect.disabled = !live;
  let text = state.status;
  if (state.status === "playing" && state.player) {
    const received = state.receivedFrames / state.remote.sampleRate;
    const ahead = Math.max(0, received - state.player.positionSec);
    text += ` | ${ahead.toFixed(1)} s buffered`;
  }
  els.status.textContent = text;
}

function buildKnobPanel(manifest) {
  const entries = Object.entries(manifest).filter(([name]) => name.startsWith(KNOB_PREFIX));
  els.knobs.replaceChildren(...entries.map(([name, entry]) => knobControl(name, entry)));
}

function knobControl(name, entry) {
  const wrap = document.createElement("label");
  wrap.className = "knob";
  if (entry.description) wrap.title = entry.description;
  const label = document.createElement("span");
  label.textContent = name.slice(KNOB_PREFIX.length).replace(/_/g, " ");
  const readout = document.createElement("span");
  const show = (v) => { readout.textContent = typeof v === "number" ? String(+v.toFixed(2)) : String(v); };
  let input;
  if (entry.type === "bool") {
    input = document.createElement("input");
    input.type = "checkbox";
    input.checked = Boolean(state.values[name]);
    input.addEventListener("change", () => commitKnob(name, input.checked, show));
  } else if (entry.type === "enum") {
    input = document.createElement("select");
    for (const option of entry.options ?? []) input.append(new Option(String(option), String(option)));
    input.value = String(state.values[name]);
    input.addEventListener("change", () => commitKnob(name, input.value, show));
  } else {
    const min = entry.min ?? 0;
    const max = entry.max ?? 1;
    input = document.createElement("input");
    Object.assign(input, { type: "range", min: String(min), max: String(max) });
    input.step = entry.type === "int" ? "1" : String((max - min) / 200);
    input.value = String(state.values[name]);
    input.addEventListener("input", () => commitKnob(name, Number(input.value), show));
  }
  show(state.values[name]);
  wrap.append(label, readout, input);
  return wrap;
}

function commitKnob(name, value, show) {
  state.values[name] = value;
  show(value);
  sendParams();
}

// --- DEMON connection ---
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

async function start() {
  await stop();
  setStatus("connecting");
  try {
    const remote = new RemoteBackend(
      wsUrl(),
      new Float32Array(STUB_SAMPLE_RATE * STUB_SECONDS * STUB_CHANNELS),
      STUB_CHANNELS,
      buildConfig(),
      { sliceWorkerUrl: "/sdk/sliceDecoder.worker.js" },
    );
    state.remote = remote;
    remote.addEventListener("slice", onSlice);
    remote.addEventListener("close", () => {
      if (!remote.closedByUser) setStatus("connection lost");
    });
    await remote.connect();
    if (!remote.initialBuffer) throw new Error("server sent no initial buffer");
    console.info("minimax session", remote.backendSessionId);

    const manifest = remote.knobManifest?.knobs ?? {};
    state.values = Object.fromEntries(
      Object.entries(manifest).map(([name, entry]) => [name, defaultKnobValue(entry)]));
    // Steer a continuous stream (mask the end token) and pivot hard on a new
    // prompt (keep 2.5 s of history), as the original page did.
    if ("minimax_endless" in state.values) state.values.minimax_endless = true;
    if ("minimax_reprompt_history_s" in state.values) state.values.minimax_reprompt_history_s = 2.5;
    buildKnobPanel(manifest);

    const player = new AudioPlayer({ workletUrl: "/sdk/audio-worklet.js" });
    state.player = player;
    await player.init(remote.initialBuffer, remote.channels);
    attachAnalyser(player);
    state.connected = true;
    state.paramsTimer = window.setInterval(onParamsTick, PARAMS_TICK_MS);
    // Playback starts on the first slice (onSlice) so the playhead stays behind the frontier.
    setStatus("waiting for first audio (~10 s)");
  } catch (err) {
    await stop();
    setStatus(`error: ${err instanceof Error ? err.message : "start failed"}`);
  }
}

function onSlice(event) {
  const detail = event.detail;
  const player = state.player;
  if (!player || detail.epoch !== player.swapCount) return;
  if (state.slices === 0) void player.resume();
  const startFrame = Math.floor(detail.startSample);
  if (detail.flags === SLICE_FLAG_DELTA) player.addDelta(startFrame, detail.audio);
  else player.patch(startFrame, detail.audio);
  const end = startFrame + detail.audio.length / state.remote.channels;
  state.receivedFrames = Math.max(state.receivedFrames, end);
  state.slices += 1;
  state.lastSliceAt = performance.now();
  if (state.status !== "playing") setStatus("playing");
}

function onParamsTick() {
  sendParams();
  const stale = state.lastSliceAt != null && performance.now() - state.lastSliceAt > ENDED_AFTER_MS;
  if (stale && state.status === "playing") setStatus("piece ended (tape loops): Reconnect for a new one");
  else renderStatus();
}

function sendParams() {
  if (state.connected) state.remote.sendParams(state.values, state.player.positionSec);
}

async function stop() {
  window.clearInterval(state.paramsTimer);
  state.connected = false; // stops sendParams before the socket closes
  try { await state.player?.close(); } catch {}
  try { state.remote?.close(); } catch {}
  Object.assign(state, { remote: null, player: null, analyser: null, paramsTimer: null });
  Object.assign(state, { slices: 0, receivedFrames: 0, lastSliceAt: null, rms: 0 });
  setStatus("idle");
}

// --- audio analyser ---
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
  let sum = 0;
  for (let i = from; i < to; i++) sum += state.freq[i];
  return sum / ((to - from) * 255);
}

function readBands() {
  if (!state.analyser) return { low: 0, mid: 0, high: 0 };
  state.analyser.getByteFrequencyData(state.freq);
  state.analyser.getFloatTimeDomainData(state.wave);
  let sq = 0;
  for (const s of state.wave) sq += s * s;
  state.rms = Math.sqrt(sq / state.wave.length);
  return { low: bandEnergy(1, 12), mid: bandEnergy(12, 90), high: bandEnergy(90, 400) };
}

// --- three scene ---
const renderer = new THREE.WebGLRenderer({ canvas: els.canvas, antialias: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(55, 1, 0.1, 100);
camera.position.set(0, 0, 4.2);
scene.add(new THREE.AmbientLight(0x404060, 1.2));
const sun = new THREE.DirectionalLight(0xffffff, 1.6);
sun.position.set(3, 4, 5);
scene.add(sun);

const sphereGeometry = new THREE.IcosahedronGeometry(1.2, 5);
const basePositions = sphereGeometry.attributes.position.array.slice();
const pointsMaterial = new THREE.PointsMaterial({ size: 0.022, color: 0x8fb6ff });
const points = new THREE.Points(sphereGeometry, pointsMaterial);
scene.add(points);
const core = new THREE.Mesh(
  new THREE.IcosahedronGeometry(0.55, 2),
  new THREE.MeshStandardMaterial({ color: 0x24204a, emissive: 0x3a1f6a, roughness: 0.6, flatShading: true }),
);
scene.add(core);

function resize() {
  renderer.setSize(window.innerWidth, window.innerHeight, false);
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
}

function updateScene(bands, t) {
  const pos = sphereGeometry.attributes.position.array;
  const swell = 1 + bands.low * 0.45;
  for (let i = 0; i < pos.length; i += 3) {
    const x = basePositions[i], y = basePositions[i + 1], z = basePositions[i + 2];
    const k = swell * (1 + Math.sin(t * 3 + y * 6 + x * 2) * bands.mid * 0.12);
    pos[i] = x * k; pos[i + 1] = y * k; pos[i + 2] = z * k;
  }
  sphereGeometry.attributes.position.needsUpdate = true;
  pointsMaterial.color.setHSL(0.62 - bands.high * 0.5, 0.7, 0.6 + bands.high * 0.25);
  points.rotation.y = t * 0.12;
  points.rotation.x = Math.sin(t * 0.2) * 0.25;
  core.rotation.y = -t * 0.3;
  core.scale.setScalar(1 + bands.low * 0.3);
  core.material.emissiveIntensity = 0.4 + bands.high * 2;
}

// --- render loop ---
function frame(ms) {
  updateScene(readBands(), ms / 1000);
  renderer.render(scene, camera);
  requestAnimationFrame(frame);
}

// --- event wiring ---
els.start.addEventListener("click", () => (state.remote ? void stop() : void start()));
els.reconnect.addEventListener("click", () => void start());
els.send.addEventListener("click", () => {
  const prompt = els.prompt.value.trim() || DEFAULT_PROMPT;
  state.remote?.sendPrompt(prompt, undefined, undefined, prompt);
});
window.addEventListener("resize", resize);
window.addEventListener("beforeunload", () => { try { state.remote?.close(); } catch {} });
window.__demo = {
  get stats() {
    const { connected, slices, rms } = state;
    return { connected, slices, positionSec: state.player?.positionSec ?? 0, rms };
  },
};
resize();
renderStatus();
requestAnimationFrame(frame);
