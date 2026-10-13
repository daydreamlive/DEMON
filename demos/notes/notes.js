import {
  AnalysisStore,
  AudioPlayer,
  RemoteBackend,
  SLICE_FLAG_DELTA,
  notesToSmf,
  playheadForNotes,
} from "/sdk/demon-client.js";

// Live notes of the streaming canvas: one `notes` analysis tap drawn as a
// piano roll around the audible playhead. The server analyses ahead of
// the playhead it learns from `params.playback_pos`, so this page streams
// that every PARAMS_TICK_MS even though it drives no knobs.

const DEFAULT_PROMPT =
  "driving cinematic synthwave, analog arpeggios, gated reverb snare, " +
  "wide saw-lead, 152 bpm, G minor, 4/4";
const DEFAULT_FIXTURE = "low_fi_Gm_loop_60s_gnm.wav";
const DEFAULT_BACKEND = "sa3";
const STUB_FRAMES = 9600;
const STUB_CHANNELS = 2;
const PARAMS_TICK_MS = 80;
const TAP = "notes";
const BEHIND_S = 4;
const MIN_PITCH_SPAN = 24;

const els = {
  dropped: document.querySelector("#analysis-dropped"),
  error: document.querySelector("#analysis-error"),
  fixture: document.querySelector("#fixture"),
  horizon: document.querySelector("#horizon"),
  horizonValue: document.querySelector("#horizon-value"),
  instruments: document.querySelector("#instruments"),
  licence: document.querySelector("#licence"),
  loop: document.querySelector("#loop"),
  loopEnd: document.querySelector("#loop-end"),
  loopStart: document.querySelector("#loop-start"),
  mode: document.querySelector("#analysis-mode"),
  notesOn: document.querySelector("#notes-on"),
  prompt: document.querySelector("#prompt"),
  roll: document.querySelector("#roll"),
  saveMidi: document.querySelector("#save-midi"),
  statusDot: document.querySelector("#status-dot"),
  statusText: document.querySelector("#status-text"),
  transport: document.querySelector("#transport"),
};

const store = new AnalysisStore(TAP);

const state = {
  fixtures: [],
  status: "idle",
  message: "",
  remote: null,
  player: null,
  paramsTimer: null,
  frame: null,
  detachStore: null,
  notesAvailable: false,
  analysis: null,
  loopBand: null,
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
  const options = { horizon_s: horizonS() };
  const instruments = instrumentList();
  if (instruments.length > 0) options.instruments = instruments;
  return options;
}

function renderControls() {
  const ready = state.status === "ready";
  els.transport.textContent = running() ? "Stop" : "Start";
  els.transport.classList.toggle("power-on", running());
  els.fixture.disabled = running();
  els.prompt.disabled = running();
  els.notesOn.disabled = !ready || !state.notesAvailable;
  els.loop.disabled = !ready;
  els.saveMidi.disabled = store.settledItems().length === 0;
  els.statusDot.className = `status-dot status-${state.status}`;
  els.statusText.textContent = state.message || state.status;
}

function renderAnalysisState() {
  const a = state.analysis;
  els.mode.textContent = a && a.enabled ? a.mode : "off";
  els.dropped.textContent = a && a.dropped > 0 ? `dropped ${a.dropped}` : "";
  const licence = a?.licence ? `${a.engine} ${a.size}: ${a.licence}` : "";
  els.licence.textContent = licence;
  els.licence.hidden = !licence;
  els.error.textContent = a?.error ?? "";
  els.error.hidden = !a?.error;
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

  const duration = state.player?.duration ?? 0;
  const ph = audiblePlayhead();
  const t0 = ph - BEHIND_S;
  const t1 = ph + horizonS();
  const xOf = (t) => ((t - t0) / (t1 - t0)) * W;
  const wrap =
    state.loopBand ?? (duration > 0 ? { start_s: 0, end_s: duration } : null);

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

  // Audible playhead.
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
  const duration = player.duration;
  const start = Math.max(0, Number(els.loopStart.value) || 0);
  const end = Math.min(duration, Number(els.loopEnd.value) || 0);
  if (els.loop.checked && end - start >= 0.5) {
    state.loopBand = { start_s: start, end_s: end };
    player.setLoopBand(start, end);
    remote.sendLoopBand(start, end);
  } else {
    state.loopBand = null;
    player.clearLoopBand();
    remote.sendLoopBand(null, null);
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
  state.detachStore?.();
  state.detachStore = null;
  try {
    await state.player?.close();
  } catch {}
  try {
    state.remote?.close();
  } catch {}
  state.player = null;
  state.remote = null;
  state.notesAvailable = false;
  state.analysis = null;
  state.loopBand = null;
  els.notesOn.checked = false;
  els.loop.checked = false;
  renderAnalysisState();
  setStatus("idle");
}

function buildConfig() {
  return {
    telemetry_version: 1,
    backend: query("backend") || DEFAULT_BACKEND,
    prompt: els.prompt.value.trim() || DEFAULT_PROMPT,
    use_server_fixture: true,
    fixture_name: els.fixture.value || DEFAULT_FIXTURE,
  };
}

async function start() {
  await stop();
  store.reset();
  setStatus("connecting", "Connecting...");

  try {
    const remote = new RemoteBackend(
      wsUrl(),
      new Float32Array(STUB_FRAMES * STUB_CHANNELS),
      STUB_CHANNELS,
      buildConfig(),
      { sliceWorkerUrl: "/sdk/sliceDecoder.worker.js" },
    );
    state.remote = remote;

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
      state.player?.swap(detail.interleaved, detail.channels);
    });
    remote.addEventListener("analysis_state", (event) => {
      if (event.detail?.tap !== TAP) return;
      state.analysis = event.detail;
      if (!event.detail.enabled) els.notesOn.checked = false;
      renderAnalysisState();
    });
    remote.addEventListener("close", () => {
      if (remote.closedByUser) return;
      setStatus("error", "Connection lost.");
    });
    state.detachStore = store.attach(remote);

    await remote.connect();
    if (!remote.initialBuffer) throw new Error("server sent no initial buffer");

    state.notesAvailable = remote.analysisTaps.includes(TAP);

    const player = new AudioPlayer({ workletUrl: "/sdk/audio-worklet.js?v=5" });
    state.player = player;
    await player.init(remote.initialBuffer, remote.channels);
    await player.resume();

    els.loopEnd.max = String(player.duration);
    els.loopStart.max = String(player.duration);
    state.paramsTimer = window.setInterval(sendParamsNow, PARAMS_TICK_MS);
    state.frame = window.requestAnimationFrame(frameLoop);
    setStatus(
      "ready",
      state.notesAvailable ? "" : "Notes analysis is not available on this server.",
    );
  } catch (err) {
    await stop();
    setStatus("error", err instanceof Error ? err.message : "Start failed");
  }
}

function saveMidi() {
  const band = state.loopBand;
  const notes = store
    .settledItems()
    .filter((n) => !band || (n.start_s >= band.start_s && n.start_s < band.end_s));
  if (notes.length === 0) return;
  const bytes = notesToSmf(notes, { originS: band ? band.start_s : 0 });
  const blob = new Blob([bytes], { type: "audio/midi" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  const stem = (els.fixture.value || "canvas").replace(/\.[^.]+$/, "");
  a.href = url;
  a.download = `${stem}-notes.mid`;
  a.click();
  window.setTimeout(() => URL.revokeObjectURL(url), 1000);
}

els.transport.addEventListener("click", () => {
  if (running()) void stop();
  else void start();
});

els.notesOn.addEventListener("change", sendTap);
els.horizon.addEventListener("input", () => {
  els.horizonValue.textContent = `${horizonS()} s`;
});
els.horizon.addEventListener("change", () => {
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

renderControls();
renderAnalysisState();
renderFixtures();
drawRoll();
void fetchFixtures();
