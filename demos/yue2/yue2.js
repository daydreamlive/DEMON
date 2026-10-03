import { AudioPlayer, RemoteBackend, SLICE_FLAG_DELTA } from "/sdk/demon-client.js";

// state
const SAMPLE_RATE = 48000;
const STUB_FRAMES = 9600; // the yue2 create path composes its own song and ignores this
const STUB_CHANNELS = 2;
const PARAMS_TICK_MS = 100;
const BAR_COUNT = 64;
const DEFAULT_PROMPT_A = "English, warm piano pop, female vocal, 90 BPM";
const DEFAULT_PROMPT_B = ""; // empty = B follows A; a distinct B composes a second song (more create time)
const DEFAULT_LYRICS =
  "[Verse]\nCity lights are calling me home\nEvery street I walk alone\n[Chorus]\nSing it loud\n[Outro]\n";
const DEFAULT_DURATION_S = 60;
const state = {
  remote: null, player: null, analyser: null, freq: null, wave: null,
  status: "idle", slices: 0, lastEndSec: 0, windowSec: 0, rms: 0,
  knobs: [], values: {}, paramsTimer: null, error: "", notice: "", gen: 0, numGens: 0,
};

// DOM
const els = {
  canvas: document.querySelector("#scene"),
  status: document.querySelector("#status"),
  promptA: document.querySelector("#prompt-a"),
  promptB: document.querySelector("#prompt-b"),
  blend: document.querySelector("#blend"),
  lyrics: document.querySelector("#lyrics"),
  duration: document.querySelector("#duration"),
  blendValue: document.querySelector("#blend-value"),
  knobs: document.querySelector("#knobs"),
  start: document.querySelector("#start"),
  send: document.querySelector("#send"),
};
els.promptA.value = DEFAULT_PROMPT_A;
els.promptB.value = DEFAULT_PROMPT_B;
els.lyrics.value = DEFAULT_LYRICS;
els.duration.value = String(DEFAULT_DURATION_S);

function setStatus(status) {
  state.status = status;
  els.start.textContent = status === "idle" || status === "error" ? "Start" : "Stop";
  els.send.disabled = !state.player;
  // Lyrics and song length are composed into the song at connect.
  const locked = status !== "idle" && status !== "error";
  els.lyrics.disabled = locked;
  els.duration.disabled = locked;
}

function renderStatusLine() {
  if (state.status === "connecting") {
    els.status.textContent = "composing (score + semantic tokens, about 15-20 s)";
    return;
  }
  if (state.status === "error" || state.status === "idle") {
    els.status.textContent = state.error ? `error: ${state.error}` : state.status;
    return;
  }
  if (!state.player) {
    els.status.textContent = "ready";
    return;
  }
  const play = state.windowSec > 0 ? state.player.positionSec % state.windowSec : state.player.positionSec;
  els.status.textContent =
    `playing ${play.toFixed(1)} / ${state.windowSec.toFixed(1)} s, ${state.slices} slices` +
    (state.notice ? ` (${state.notice})` : "");
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
  state.numGens = detail.numGens ?? state.numGens;
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
  const duration = Math.max(10, Math.min(100, Number(els.duration.value) || DEFAULT_DURATION_S));
  const config = {
    telemetry_version: 1, backend: "yue2", prompt, prompt_b: promptB,
    yue2_lyrics: els.lyrics.value, yue2_duration_s: duration,
  };
  const remote = new RemoteBackend(
    wsUrl(), new Float32Array(STUB_FRAMES * STUB_CHANNELS), STUB_CHANNELS, config,
    { sliceWorkerUrl: "/sdk/sliceDecoder.worker.js" },
  );
  state.remote = remote;
  remote.addEventListener("slice", handleSlice);
  // Non-fatal server errors (a failed re-compose): the old song keeps playing.
  remote.addEventListener("server_error", (event) => {
    state.notice = event.detail?.message || event.detail?.code || "server error";
  });
  remote.addEventListener("close", () => {
    if (remote.closedByUser || state.remote !== remote) return;
    state.error = "connection lost";
    setStatus("error");
  });
  await remote.connect();
  if (gen !== state.gen) return void remote.close();
  if (!remote.initialBuffer) throw new Error("server sent no initial buffer");
  console.info("yue2 session", remote.backendSessionId);
  setStatus("ready");

  const manifest = remote.knobManifest?.knobs ?? {};
  state.knobs = Object.entries(manifest)
    .filter(([, entry]) => entry.type === "float" || entry.type === "int")
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
  Object.assign(state, {
    remote: null, player: null, analyser: null, freq: null, wave: null, slices: 0, rms: 0, lastEndSec: 0, notice: "", numGens: 0,
  });
  setStatus("idle");
}

function knobInput(name, entry) {
  const input = document.createElement("input");
  if (name === "seed") {
    // A seed is an identity, not a quantity: a number box, not a 0..2^32 slider.
    input.type = "number";
    input.step = "1";
  } else {
    input.type = "range";
    input.step = entry.type === "int" ? "1" : String(((entry.max ?? 1) - (entry.min ?? 0)) / 200);
  }
  input.min = String(entry.min ?? 0);
  input.max = String(entry.max ?? 1);
  input.value = String(state.values[name]);
  input.dataset.knob = name;
  return input;
}

function renderKnobs() {
  const nodes = state.knobs.map(({ name, entry }) => {
    const label = document.createElement("label");
    label.className = "knob-control";
    const heading = document.createElement("span"); heading.className = "knob-heading";
    const caption = document.createElement("span"); caption.className = "knob-label"; caption.textContent = name.replace(/^yue2_/, "").replace(/_/g, " ");
    const value = document.createElement("span"); value.className = "knob-value";
    const input = knobInput(name, entry);
    const digits = entry.type === "int" ? 0 : 2;
    if (entry.description) label.title = entry.description;
    if (input.type === "range") value.textContent = ` ${Number(input.value).toFixed(digits)}`;
    input.addEventListener("input", () => {
      const v = Number(input.value);
      if (!Number.isFinite(v)) return;
      state.values = { ...state.values, [name]: entry.type === "int" ? Math.round(v) : v };
      if (input.type === "range") value.textContent = ` ${v.toFixed(digits)}`;
      sendParams();
    });
    heading.append(caption, value); label.append(heading, input);
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

// spectrum canvas
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
  const startX = width > 760 ? width * 0.52 : width * 0.08;
  const span = width > 760 ? width * 0.43 : width * 0.84;
  const barW = span / BAR_COUNT;
  const baseline = height * 0.72, maxH = height * 0.42;
  ctx2d.strokeStyle = "rgb(130 177 197 / 12%)";
  ctx2d.beginPath(); ctx2d.moveTo(startX, baseline + 1); ctx2d.lineTo(startX + span, baseline + 1); ctx2d.stroke();
  const glow = ctx2d.createLinearGradient(0, baseline - maxH, 0, baseline);
  glow.addColorStop(0, `hsl(${hue + 25} 64% 68%)`); glow.addColorStop(1, `hsl(${hue} 48% 35%)`);
  for (let i = 0; i < BAR_COUNT; i++) {
    let level = 0;
    if (state.freq) {
      for (let k = 0; k < bucket; k++) level += state.freq[i * bucket + k];
      level /= bucket * 255;
    }
    const h = Math.max(1, level * maxH); // flat baseline when idle; real bars only from the analyser
    ctx2d.fillStyle = glow;
    ctx2d.shadowColor = `hsl(${hue} 70% 55% / 65%)`;
    ctx2d.shadowBlur = level > 0.08 ? 12 : 0;
    ctx2d.fillRect(startX + i * barW + 1, baseline - h, Math.max(1, barW - 3), h);
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
  els.blendValue.textContent = Number(els.blend.value) >= 0.5 ? "B" : "A"; // hard switch at 0.5
  state.remote?.sendSetPromptBlend(Number(els.blend.value));
});
window.addEventListener("resize", resize);
window.addEventListener("beforeunload", () => {
  try { state.remote?.close(); } catch {}
});
window.__demo = {
  get stats() {
    return {
      status: state.status,
      connected: Boolean(state.remote && state.player),
      windowSec: state.windowSec,
      slices: state.slices,
      numGens: state.numGens,
      notice: state.notice,
      positionSec: state.player?.positionSec ?? 0,
      rms: state.rms,
    };
  },
};

resize();
setStatus("idle");
requestAnimationFrame(frame);
