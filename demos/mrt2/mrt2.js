import * as THREE from "three";
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
  knobs: [], values: {}, paramsTimer: null, error: "",
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

async function connect() {
  const [prompt, promptB] = promptPair();
  const config = { telemetry_version: 1, backend: "mrt2", prompt, prompt_b: promptB };
  const remote = new RemoteBackend(
    wsUrl(), new Float32Array(STUB_FRAMES * STUB_CHANNELS), STUB_CHANNELS, config,
    { sliceWorkerUrl: "/sdk/sliceDecoder.worker.js" },
  );
  state.remote = remote;
  remote.addEventListener("slice", handleSlice);
  remote.addEventListener("close", () => {
    if (remote.closedByUser) return;
    state.error = "connection lost";
    setStatus("error");
  });
  await remote.connect();
  console.info("mrt2 session", remote.backendSessionId);
  setStatus("ready");

  const manifest = remote.knobManifest?.knobs ?? {};
  state.knobs = Object.entries(manifest)
    .filter(([name, entry]) => name.startsWith("mrt2_") && (entry.type === "float" || entry.type === "int"))
    .map(([name, entry]) => ({ name, entry }));
  state.values = Object.fromEntries(state.knobs.map(({ name, entry }) => [name, entry.default ?? entry.min ?? 0]));
  renderKnobs();

  const player = new AudioPlayer({ workletUrl: "/sdk/audio-worklet.js" });
  await player.init(remote.initialBuffer, remote.channels);
  await player.resume();
  state.player = player;
  state.windowSec = remote.initialBuffer.length / remote.channels / SAMPLE_RATE;
  attachAnalyser(player);
  state.paramsTimer = window.setInterval(sendParams, PARAMS_TICK_MS);
  setStatus("playing");
}

async function disconnect() {
  window.clearInterval(state.paramsTimer);
  state.paramsTimer = null;
  try { await state.player?.close(); } catch {}
  try { state.remote?.close(); } catch {}
  Object.assign(state, { remote: null, player: null, analyser: null, slices: 0, rms: 0, lastEndSec: 0 });
  setStatus("idle");
}

function renderKnobs() {
  const nodes = state.knobs.map(({ name, entry }) => {
    const label = document.createElement("label");
    const value = document.createElement("span");
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
    label.append(name.replace(/^mrt2_/, "").replace(/_/g, " "), value, input);
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

function readBands() {
  if (!state.analyser) return { low: 0, mid: 0, high: 0 };
  state.analyser.getByteFrequencyData(state.freq);
  state.analyser.getFloatTimeDomainData(state.wave);
  let sum = 0;
  for (const s of state.wave) sum += s * s;
  state.rms = Math.sqrt(sum / state.wave.length);
  const avg = (a, b) => {
    let t = 0;
    for (let i = a; i < b; i++) t += state.freq[i];
    return t / ((b - a) * 255);
  };
  return { low: avg(1, 12), mid: avg(12, 96), high: avg(96, 384) };
}

// three scene
const renderer = new THREE.WebGLRenderer({ canvas: els.canvas, antialias: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x07080b);
const camera = new THREE.PerspectiveCamera(55, 1, 0.1, 100);
camera.position.set(0, 0, 14);
scene.add(new THREE.AmbientLight(0xffffff, 0.35));
const sun = new THREE.DirectionalLight(0xffffff, 1.6);
sun.position.set(4, 6, 10);
scene.add(sun);

const ring = new THREE.Group();
const bars = [];
const barGeometry = new THREE.BoxGeometry(0.32, 1, 0.32);
barGeometry.translate(0, 0.5, 0); // grow outward from the ring radius
for (let i = 0; i < BAR_COUNT; i++) {
  const pivot = new THREE.Group();
  pivot.rotation.z = (i / BAR_COUNT) * Math.PI * 2;
  const bar = new THREE.Mesh(barGeometry, new THREE.MeshStandardMaterial({ roughness: 0.4 }));
  bar.position.y = 3.2;
  pivot.add(bar);
  ring.add(pivot);
  bars.push(bar);
}
scene.add(ring);

function resize() {
  renderer.setSize(window.innerWidth, window.innerHeight, false);
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
}

function updateScene(bands, dt) {
  const hue = 0.58 - Number(els.blend.value) * 0.52; // A = cool blue, B = warm orange
  const bucket = state.freq ? Math.floor(384 / BAR_COUNT) : 0;
  bars.forEach((bar, i) => {
    let level = 0;
    if (state.freq) {
      for (let k = 0; k < bucket; k++) level += state.freq[i * bucket + k];
      level /= bucket * 255;
    }
    bar.scale.y = 0.15 + level * 4.5;
    bar.material.color.setHSL(hue + level * 0.08, 0.75, 0.35 + level * 0.3);
  });
  ring.rotation.z += dt * (0.1 + bands.mid * 0.9);
  camera.position.z = 14 - bands.low * 3;
  sun.intensity = 1.2 + bands.high * 3;
}

// render loop
let lastFrame = performance.now();
function frame(now) {
  const dt = Math.min((now - lastFrame) / 1000, 0.1);
  lastFrame = now;
  updateScene(readBands(), dt);
  renderer.render(scene, camera);
  renderStatusLine();
  requestAnimationFrame(frame);
}

// event wiring
els.start.addEventListener("click", async () => {
  if (state.status !== "idle" && state.status !== "error") {
    await disconnect();
    return;
  }
  state.error = "";
  setStatus("connecting");
  try {
    await connect();
  } catch (err) {
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
