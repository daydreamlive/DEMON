import {
  AnalysisStore,
  AudioPlayer,
  MicCapture,
  RemoteBackend,
  SLICE_FLAG_DELTA,
  notesToSmf,
  playheadForNotes,
} from "/sdk/demon-client.js";

// Live notes, one `notes` analysis tap drawn as a piano roll. Three sources:
//
// - Microphone (default): a `live_input` session created without a source;
//   MicCapture streams 100 ms frames as `audio_in` and the server appends
//   them at its write frontier. No generator and no playback: the only model
//   on the card is the analyser. The roll scrolls with the frontier the
//   server acknowledges.
// - File: a `live_input` session whose canvas is the uploaded clip, looped
//   through AudioPlayer.
// - Generator: an SA3 session on a server fixture, as before.
//
// File and Generator stream `params.playback_pos` every PARAMS_TICK_MS (the
// server analyses ahead of the playhead it learns from it). Microphone sends
// none: the server playhead follows the frontier.

const DEFAULT_PROMPT =
  "driving cinematic synthwave, analog arpeggios, gated reverb snare, " +
  "wide saw-lead, 152 bpm, G minor, 4/4";
const DEFAULT_FIXTURE = "low_fi_Gm_loop_60s_gnm.wav";
const DEFAULT_BACKEND = "sa3";
const LIVE_INPUT_BACKEND = "live_input";
const STUB_FRAMES = 9600;
const STUB_CHANNELS = 2;
const UPLOAD_RATE = 48000;
const PARAMS_TICK_MS = 80;
const TAP = "notes";
const BEHIND_S = 4;
const MIC_AHEAD_S = 1;
// An ack arrives at most every 200 ms; the drawn frontier extrapolates by
// wall time between acks, never further than this past the last one.
const MIC_EXTRAPOLATE_MAX_S = 0.4;
const MIN_PITCH_SPAN = 24;
const MODES = ["mic", "file", "generator"];

const els = {
  analysisMode: document.querySelector("#analysis-mode"),
  dropped: document.querySelector("#analysis-dropped"),
  error: document.querySelector("#analysis-error"),
  file: document.querySelector("#file"),
  fixture: document.querySelector("#fixture"),
  horizon: document.querySelector("#horizon"),
  horizonValue: document.querySelector("#horizon-value"),
  instruments: document.querySelector("#instruments"),
  levelPeak: document.querySelector("#level-peak"),
  levelRms: document.querySelector("#level-rms"),
  loop: document.querySelector("#loop"),
  loopEnd: document.querySelector("#loop-end"),
  loopRow: document.querySelector("#loop-row"),
  loopStart: document.querySelector("#loop-start"),
  mode: document.querySelector("#mode"),
  modePanels: document.querySelectorAll(".mode-panel"),
  notesOn: document.querySelector("#notes-on"),
  prompt: document.querySelector("#prompt"),
  roll: document.querySelector("#roll"),
  saveMidi: document.querySelector("#save-midi"),
  size: document.querySelector("#size"),
  statusDot: document.querySelector("#status-dot"),
  statusText: document.querySelector("#status-text"),
  take: document.querySelector("#take"),
  transport: document.querySelector("#transport"),
};

const store = new AnalysisStore(TAP);

const state = {
  fixtures: [],
  status: "idle",
  message: "",
  remote: null,
  player: null,
  mic: null,
  paramsTimer: null,
  frame: null,
  detachStore: null,
  notesAvailable: false,
  analysis: null,
  // Seconds the newest analysis_update trailed the playhead (microphone:
  // the frontier), or null before the first one.
  lagS: null,
  loopBand: null,
  // Microphone: the last acknowledged frontier (canvas seconds of the
  // current take), when it arrived, the take's canvas length and number.
  frontierS: 0,
  frontierWall: 0,
  takeS: 0,
  take: 0,
  sourceName: "",
};

els.prompt.value = DEFAULT_PROMPT;

function query(name) {
  return new URLSearchParams(window.location.search).get(name);
}

function wsUrl() {
  const override = query("ws");
  if (override) return override;
  const url = new URL(window.location.href);
  url.protocol = url.protocol === "https:" ? "wss:" : "ws:";
  url.pathname = "/";
  url.search = "";
  url.hash = "";
  return url.toString();
}

function mode() {
  return els.mode.value;
}

function setStatus(status, message = "") {
  state.status = status;
  state.message = message;
  renderControls();
}

function running() {
  return state.status === "ready" || state.status === "connecting";
}

function horizonS() {
  return Number(els.horizon.value);
}

function instrumentList() {
  return els.instruments.value
    .split(/[\s,]+/)
    .map((name) => name.trim())
    .filter(Boolean);
}

function tapOptions() {
  const options = { size: els.size.value, horizon_s: horizonS() };
  const instruments = instrumentList();
  if (instruments.length > 0) options.instruments = instruments;
  return options;
}

function renderTake() {
  els.take.textContent =
    mode() === "mic" && state.status === "ready"
      ? `take ${state.take} · ${micFrontier().toFixed(1)} s`
      : "";
}

function renderControls() {
  const ready = state.status === "ready";
  const m = mode();
  els.transport.textContent = running() ? "Stop" : "Start";
  els.transport.classList.toggle("power-on", running());
  els.mode.disabled = running();
  els.fixture.disabled = running();
  els.prompt.disabled = running();
  els.file.disabled = running();
  for (const panel of els.modePanels) panel.hidden = panel.dataset.mode !== m;
  // No playback in microphone mode, so no loop band.
  els.loopRow.hidden = m === "mic";
  els.notesOn.disabled = !ready || !state.notesAvailable;
  els.loop.disabled = !ready || m === "mic";
  els.saveMidi.disabled = store.settledItems().length === 0;
  els.statusDot.className = `status-dot status-${state.status}`;
  let lag = "";
  if (ready && state.lagS != null) {
    const word = state.lagS >= 0 ? "lag" : "ahead";
    lag = ` · ${word} ~${Math.abs(state.lagS).toFixed(1)} s`;
  }
  els.statusText.textContent = (state.message || state.status) + lag;
  renderTake();
}

function renderAnalysisState() {
  const a = state.analysis;
  els.analysisMode.textContent = a && a.enabled ? a.mode : "off";
  els.dropped.textContent = a && a.dropped > 0 ? `dropped ${a.dropped}` : "";
  els.error.textContent = a?.error ?? "";
  els.error.hidden = !a?.error;
}

function renderLevel(level) {
  const pct = (v) => `${Math.min(100, Math.max(0, v * 100)).toFixed(1)}%`;
  // RMS of full-scale speech or music sits well under 0.5; double it so
  // the bar uses its width. The peak tick is unscaled.
  els.levelRms.style.width = pct(level ? level.rms * 2 : 0);
  els.levelPeak.style.left = pct(level ? level.peak : 0);
}

function renderFixtures() {
  const selected = els.fixture.value || DEFAULT_FIXTURE;
  const names = state.fixtures.length > 0 ? state.fixtures : [selected];
  els.fixture.replaceChildren(
    ...names.map((name) => {
      const option = document.createElement("option");
      option.value = name;
      option.textContent = name;
      return option;
    }),
  );
  els.fixture.value = names.includes(selected) ? selected : names[0];
}

async function fetchFixtures() {
  try {
    const res = await fetch("/api/server-info");
    const info = await res.json();
    state.fixtures = Array.isArray(info.server_side_fixtures)
      ? info.server_side_fixtures
      : [];
  } catch {
    state.fixtures = [];
  }
  renderFixtures();
}

// ---- piano roll ---------------------------------------------------------

const INSTRUMENT_HUES = [205, 32, 140, 280, 0, 55, 175, 320, 95, 245];
const hueCache = new Map();

function hueFor(instrument) {
  let hue = hueCache.get(instrument);
  if (hue === undefined) {
    hue = INSTRUMENT_HUES[hueCache.size % INSTRUMENT_HUES.length];
    hueCache.set(instrument, hue);
  }
  return hue;
}

function pitchRange(items) {
  let lo = Infinity;
  let hi = -Infinity;
  for (const item of items) {
    if (item.instrument === "drums") continue;
    lo = Math.min(lo, item.pitch);
    hi = Math.max(hi, item.pitch);
  }
  if (!Number.isFinite(lo)) return [48, 48 + MIN_PITCH_SPAN];
  const pad = Math.max(0, MIN_PITCH_SPAN - (hi - lo)) / 2;
  return [Math.floor(lo - pad) - 1, Math.ceil(hi + pad) + 1];
}

function sizeCanvas(canvas) {
  const dpr = window.devicePixelRatio || 1;
  const w = Math.round(canvas.clientWidth * dpr);
  const h = Math.round(canvas.clientHeight * dpr);
  if (canvas.width !== w || canvas.height !== h) {
    canvas.width = w;
    canvas.height = h;
  }
  return dpr;
}

function micFrontier() {
  if (state.status !== "ready") return state.frontierS;
  const elapsed = (performance.now() - state.frontierWall) / 1000;
  return state.frontierS + Math.min(Math.max(0, elapsed), MIC_EXTRAPOLATE_MAX_S);
}

function audiblePlayhead() {
  const player = state.player;
  if (!player) return 0;
  const latency = player.ctx?.outputLatency ?? 0;
  return playheadForNotes(
    player.positionSec,
    latency,
    state.loopBand,
    player.duration,
  );
}

// The drawn window. Microphone mode shows the take behind the frontier
// (a cell's notes arrive once it is fully written); playback modes centre
// on the audible playhead with the analysis horizon ahead.
function view() {
  if (mode() === "mic") {
    const ph = micFrontier();
    return {
      ph,
      t0: ph - BEHIND_S - horizonS(),
      t1: ph + MIC_AHEAD_S,
      duration: state.takeS,
      wrap: null,
    };
  }
  const duration = state.player?.duration ?? 0;
  const ph = audiblePlayhead();
  return {
    ph,
    t0: ph - BEHIND_S,
    t1: ph + horizonS(),
    duration,
    wrap: state.loopBand ?? (duration > 0 ? { start_s: 0, end_s: duration } : null),
  };
}

function drawRoll() {
  const canvas = els.roll;
  const dpr = sizeCanvas(canvas);
  const ctx = canvas.getContext("2d");
  const W = canvas.width;
  const H = canvas.height;
  const overviewH = Math.round(10 * dpr);
  const rollH = H - overviewH - Math.round(4 * dpr);

  ctx.fillStyle = "#0b0c0e";
  ctx.fillRect(0, 0, W, H);

  const { ph, t0, t1, duration, wrap } = view();
  const xOf = (t) => ((t - t0) / (t1 - t0)) * W;

  const [lo, hi] = pitchRange(store.items());
  const rowH = rollH / (hi - lo + 1);
  const yOf = (pitch) => rollH - (pitch - lo + 1) * rowH;

  // Octave lines at every C.
  ctx.strokeStyle = "rgb(255 255 255 / 7%)";
  ctx.lineWidth = 1;
  ctx.fillStyle = "rgb(255 255 255 / 35%)";
  ctx.font = `${Math.round(9 * dpr)}px ui-sans-serif, system-ui, sans-serif`;
  for (let p = Math.ceil(lo / 12) * 12; p <= hi; p += 12) {
    const y = Math.round(yOf(p) + rowH) + 0.5;
    ctx.beginPath();
    ctx.moveTo(0, y);
    ctx.lineTo(W, y);
    ctx.stroke();
    ctx.fillText(`C${p / 12 - 1}`, 3 * dpr, y - 2 * dpr);
  }

  // Loop seams inside the unrolled window.
  if (state.loopBand) {
    const { start_s, end_s } = state.loopBand;
    const period = end_s - start_s;
    ctx.strokeStyle = "rgb(107 160 205 / 45%)";
    ctx.setLineDash([4 * dpr, 4 * dpr]);
    for (let k = Math.floor((t0 - end_s) / period); end_s + k * period <= t1; k++) {
      const x = Math.round(xOf(end_s + k * period)) + 0.5;
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, rollH);
      ctx.stroke();
    }
    ctx.setLineDash([]);
  }

  // Notes: settled filled, provisional hollow. Drums sit on the bottom row.
  ctx.lineWidth = Math.max(1, dpr);
  for (const { item, shift_s } of store.itemsIn(t0, t1, wrap)) {
    const x = xOf(item.start_s + shift_s);
    const w = Math.max(2 * dpr, xOf(item.end_s + shift_s) - x);
    const isDrum = item.instrument === "drums";
    const y = isDrum ? rollH - rowH : yOf(item.pitch);
    const h = Math.max(2, rowH - 1);
    const hue = hueFor(item.instrument);
    if (item.basis === "settled") {
      ctx.fillStyle = `hsl(${hue} 70% 62%)`;
      ctx.fillRect(x, y, w, h);
    } else {
      ctx.strokeStyle = `hsl(${hue} 70% 62%)`;
      ctx.strokeRect(x + 0.5, y + 0.5, w - 1, h - 1);
    }
  }

  // Playhead (microphone: the write frontier).
  const px = Math.round(xOf(ph)) + 0.5;
  ctx.strokeStyle = "#f1f1ed";
  ctx.beginPath();
  ctx.moveTo(px, 0);
  ctx.lineTo(px, rollH);
  ctx.stroke();

  // Canvas overview: whole duration, loop band shaded, playhead tick.
  const oy = H - overviewH;
  ctx.fillStyle = "#16181b";
  ctx.fillRect(0, oy, W, overviewH);
  if (duration > 0) {
    const ox = (t) => (t / duration) * W;
    if (state.loopBand) {
      ctx.fillStyle = "rgb(107 160 205 / 35%)";
      ctx.fillRect(
        ox(state.loopBand.start_s),
        oy,
        ox(state.loopBand.end_s) - ox(state.loopBand.start_s),
        overviewH,
      );
    }
    ctx.fillStyle = "#f1f1ed";
    ctx.fillRect(Math.round(ox(ph)), oy, Math.max(1, dpr), overviewH);
  }
}

function frameLoop() {
  drawRoll();
  renderTake();
  state.frame = window.requestAnimationFrame(frameLoop);
}

// ---- session ------------------------------------------------------------

function sendParamsNow() {
  if (!state.remote || !state.player || state.status !== "ready") return;
  state.remote.sendParams({}, state.player.positionSec);
}

function sendTap() {
  if (!state.remote || state.status !== "ready" || !state.notesAvailable) return;
  if (els.notesOn.checked) {
    state.remote.sendAnalysisTap(TAP, true, tapOptions());
  } else {
    state.remote.sendAnalysisTap(TAP, false);
  }
}

function applyLoopBand() {
  const player = state.player;
  const remote = state.remote;
  if (!player || !remote) return;
  // Only a generator needs the band server-side (it renders into it); a
  // live_input clip is static and would answer loop_band with
  // command_failed.
  const serverBand = remote.capabilities?.loop_band === true;
  const duration = player.duration;
  const start = Math.max(0, Number(els.loopStart.value) || 0);
  const end = Math.min(duration, Number(els.loopEnd.value) || 0);
  if (els.loop.checked && end - start >= 0.5) {
    state.loopBand = { start_s: start, end_s: end };
    player.setLoopBand(start, end);
    if (serverBand) remote.sendLoopBand(start, end);
  } else {
    state.loopBand = null;
    player.clearLoopBand();
    if (serverBand) remote.sendLoopBand(null, null);
  }
}

async function stop() {
  if (state.paramsTimer != null) {
    window.clearInterval(state.paramsTimer);
    state.paramsTimer = null;
  }
  if (state.frame != null) {
    window.cancelAnimationFrame(state.frame);
    state.frame = null;
  }
  try {
    await state.mic?.stop();
  } catch {}
  state.detachStore?.();
  state.detachStore = null;
  try {
    await state.player?.close();
  } catch {}
  try {
    state.remote?.close();
  } catch {}
  state.mic = null;
  state.player = null;
  state.remote = null;
  state.notesAvailable = false;
  state.analysis = null;
  state.lagS = null;
  state.loopBand = null;
  els.notesOn.checked = false;
  els.loop.checked = false;
  renderLevel(null);
  renderAnalysisState();
  setStatus("idle");
}

// Decode the chosen file at the canvas rate, interleaved, at most stereo.
async function decodeFile(file) {
  const ctx = new AudioContext({ sampleRate: UPLOAD_RATE });
  try {
    const audio = await ctx.decodeAudioData(await file.arrayBuffer());
    const channels = Math.min(2, audio.numberOfChannels);
    const n = audio.length;
    const interleaved = new Float32Array(n * channels);
    for (let c = 0; c < channels; c++) {
      const data = audio.getChannelData(c);
      for (let i = 0; i < n; i++) interleaved[i * channels + c] = data[i];
    }
    return { interleaved, channels };
  } finally {
    await ctx.close();
  }
}

// What each source sends at connect: the upload and the session config.
async function sessionSource() {
  const m = mode();
  if (m === "mic") {
    // An empty upload asks a live_input session for a silent capture
    // canvas (the family's live_input_capture_s, default 240 s).
    return {
      interleaved: new Float32Array(0),
      channels: 1,
      config: { telemetry_version: 1, backend: LIVE_INPUT_BACKEND },
      name: "microphone",
    };
  }
  if (m === "file") {
    const file = els.file.files?.[0];
    if (!file) throw new Error("Choose an audio file first.");
    const { interleaved, channels } = await decodeFile(file);
    if (interleaved.length === 0) throw new Error("That file holds no audio.");
    return {
      interleaved,
      channels,
      config: { telemetry_version: 1, backend: LIVE_INPUT_BACKEND },
      name: file.name,
    };
  }
  return {
    interleaved: new Float32Array(STUB_FRAMES * STUB_CHANNELS),
    channels: STUB_CHANNELS,
    config: {
      telemetry_version: 1,
      backend: query("backend") || DEFAULT_BACKEND,
      prompt: els.prompt.value.trim() || DEFAULT_PROMPT,
      use_server_fixture: true,
      fixture_name: els.fixture.value || DEFAULT_FIXTURE,
    },
    name: els.fixture.value || DEFAULT_FIXTURE,
  };
}

function fail(message) {
  void stop().then(() => setStatus("error", message));
}

function wireEvents(remote) {
  remote.addEventListener("slice", (event) => {
    const detail = event.detail;
    const player = state.player;
    if (!player || detail.epoch !== player.swapCount) return;
    const startFrame = Math.floor(detail.startSample);
    if (detail.flags === SLICE_FLAG_DELTA) {
      player.addDelta(startFrame, detail.audio);
    } else {
      player.patch(startFrame, detail.audio);
    }
  });
  remote.addEventListener("swap_ready", (event) => {
    const detail = event.detail;
    if (state.player) {
      state.player.swap(detail.interleaved, detail.channels);
      return;
    }
    // Microphone: the capture canvas filled (or the stream restarted) and
    // the server opened a new take. The store clears on the matching
    // analysis_update; the silent canvas itself is not kept.
    state.take += 1;
    state.takeS = detail.duration ?? state.takeS;
    state.frontierS = 0;
    state.frontierWall = performance.now();
  });
  remote.addEventListener("audio_in_ack", (event) => {
    state.frontierS = event.detail.frontier_s;
    state.frontierWall = performance.now();
  });
  remote.addEventListener("analysis_update", (event) => {
    const u = event.detail;
    if (u?.tap !== TAP || u.clear) return;
    // How far behind the listener this range arrived: the microphone has
    // no playback, so measure from the frontier to the range end.
    if (mode() === "mic") {
      state.lagS = micFrontier() - (u.range_start_s + u.span_s);
    } else if (u.playhead_s != null) {
      state.lagS = u.playhead_s - u.range_start_s;
    } else {
      return;
    }
    renderControls();
  });
  remote.addEventListener("command_failed", (event) => {
    if (event.detail?.command === "audio_in") {
      fail(event.detail.error || "The server refused live audio input.");
    }
  });
  remote.addEventListener("analysis_state", (event) => {
    if (event.detail?.tap !== TAP) return;
    state.analysis = event.detail;
    if (!event.detail.enabled) els.notesOn.checked = false;
    renderAnalysisState();
  });
  remote.addEventListener("close", () => {
    if (remote.closedByUser) return;
    fail("Connection lost.");
  });
}

async function startMic(remote) {
  if (remote.capabilities?.audio_in !== true) {
    throw new Error("This server cannot take live audio input.");
  }
  // The capture canvas is silent server state; nothing here plays it, so
  // do not hold the copy the handshake delivered.
  remote.initialBuffer = null;
  state.take = 1;
  state.takeS = remote.duration ?? 0;
  state.frontierS = 0;
  state.frontierWall = performance.now();
  const mic = new MicCapture({
    onFrame: (frame) => {
      remote.sendAudioIn(frame.pcm, frame.seq, { startS: frame.startS });
    },
    onLevel: renderLevel,
  });
  state.mic = mic;
  await mic.start();
}

async function startPlayer(remote) {
  if (!remote.initialBuffer) throw new Error("server sent no initial buffer");
  const player = new AudioPlayer({ workletUrl: "/sdk/audio-worklet.js?v=5" });
  state.player = player;
  await player.init(remote.initialBuffer, remote.channels);
  await player.resume();
  els.loopEnd.max = String(player.duration);
  els.loopStart.max = String(player.duration);
  state.paramsTimer = window.setInterval(sendParamsNow, PARAMS_TICK_MS);
}

async function start() {
  await stop();
  store.reset();
  setStatus("connecting", "Connecting...");

  try {
    const source = await sessionSource();
    state.sourceName = source.name;
    const remote = new RemoteBackend(
      wsUrl(),
      source.interleaved,
      source.channels,
      source.config,
      { sliceWorkerUrl: "/sdk/sliceDecoder.worker.js" },
    );
    state.remote = remote;
    wireEvents(remote);
    state.detachStore = store.attach(remote);

    await remote.connect();
    state.notesAvailable = remote.analysisTaps.includes(TAP);

    if (mode() === "mic") await startMic(remote);
    else await startPlayer(remote);

    state.frame = window.requestAnimationFrame(frameLoop);
    setStatus(
      "ready",
      state.notesAvailable ? "" : "Notes analysis is not available on this server.",
    );
    // Microphone and file sessions exist only to be analysed: tap on.
    if (state.notesAvailable && mode() !== "generator") {
      els.notesOn.checked = true;
      sendTap();
    }
  } catch (err) {
    await stop();
    setStatus("error", err instanceof Error ? err.message : "Start failed");
  }
}

function saveMidi() {
  // Microphone: the whole take. Playback: the loop band when one is set.
  const band = mode() === "mic" ? null : state.loopBand;
  const notes = store
    .settledItems()
    .filter((n) => !band || (n.start_s >= band.start_s && n.start_s < band.end_s));
  if (notes.length === 0) return;
  const bytes = notesToSmf(notes, { originS: band ? band.start_s : 0 });
  const blob = new Blob([bytes], { type: "audio/midi" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  const stem =
    mode() === "mic"
      ? `microphone-take${state.take}`
      : (state.sourceName || "canvas").replace(/\.[^.]+$/, "");
  a.href = url;
  a.download = `${stem}-notes.mid`;
  a.click();
  window.setTimeout(() => URL.revokeObjectURL(url), 1000);
}

els.transport.addEventListener("click", () => {
  if (running()) void stop();
  else void start();
});

els.mode.addEventListener("change", () => {
  store.reset();
  renderControls();
  drawRoll();
});
els.notesOn.addEventListener("change", sendTap);
els.horizon.addEventListener("input", () => {
  els.horizonValue.textContent = `${horizonS()} s`;
});
els.horizon.addEventListener("change", () => {
  if (els.notesOn.checked) sendTap();
});
// A new enable replaces the running tap, so a size change takes effect
// at once.
els.size.addEventListener("change", () => {
  state.lagS = null;
  if (els.notesOn.checked) sendTap();
});
els.instruments.addEventListener("change", () => {
  if (els.notesOn.checked) sendTap();
});
els.loop.addEventListener("change", applyLoopBand);
els.loopStart.addEventListener("change", () => {
  if (els.loop.checked) applyLoopBand();
});
els.loopEnd.addEventListener("change", () => {
  if (els.loop.checked) applyLoopBand();
});
els.saveMidi.addEventListener("click", saveMidi);

store.subscribe(renderControls);

window.addEventListener("beforeunload", () => {
  try {
    state.remote?.close();
  } catch {}
});

const requested = query("source");
if (MODES.includes(requested)) els.mode.value = requested;

renderControls();
renderAnalysisState();
renderFixtures();
renderLevel(null);
drawRoll();
void fetchFixtures();
