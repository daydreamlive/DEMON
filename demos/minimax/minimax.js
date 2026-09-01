import {
  AudioPlayer,
  RemoteBackend,
  SLICE_FLAG_DELTA,
} from "/sdk/demon-client.js";

const DEFAULT_PROMPT =
  "bpm is 140. key is F, and scale is minor. Darkwave / Coldwave. " +
  "Gothic synth textures, driving bass, cavernous drums.";
const PARAMS_TICK_MS = 80;

// MiniMax is an append-only family with no audio encoder: the create
// upload is ignored and the tape starts silent (see
// acestep/streaming/minimax_session.py). The wire protocol still wants
// an audio frame, so send the shortest silence the pool alignment
// accepts instead of naming a fixture the server would pointlessly load.
const UPLOAD_SAMPLE_RATE = 48000;
const UPLOAD_SECONDS = 2;
const UPLOAD_CHANNELS = 2;

const els = {
  knobs: document.querySelector("#knobs"),
  lyrics: document.querySelector("#lyrics"),
  prompt: document.querySelector("#prompt"),
  sendPrompt: document.querySelector("#send-prompt"),
  statusDot: document.querySelector("#status-dot"),
  statusText: document.querySelector("#status-text"),
  tick: document.querySelector("#tick"),
  transport: document.querySelector("#transport"),
  window: document.querySelector("#window"),
};

const state = {
  knobs: [],
  values: {},
  status: "idle",
  message: "",
  tickMs: null,
  remote: null,
  player: null,
  paramsTimer: null,
  lastSliceAt: null,
  ended: false,
  noteTimer: null,
};

// The piece ends when the LM emits its end token (30-115 s observed);
// after that the tape loops and knob/prompt changes have nothing to
// steer. There is no wire event for it, but a healthy session commits a
// chunk every ~4 s, so a long slice gap is an unambiguous tell.
const ENDED_AFTER_MS = 15000;

els.prompt.value = DEFAULT_PROMPT;

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

function setStatus(status, message = "") {
  state.status = status;
  state.message = message;
  renderStatus();
}

function running() {
  return state.status === "ready" || state.status === "connecting";
}

function knobLabel(name) {
  return name.replace(/^minimax_/, "").replace(/_/g, " ");
}

function valueFromEntry(entry) {
  if (entry.default !== undefined) return entry.default;
  if (entry.type === "bool") return false;
  if (entry.type === "enum") return entry.options?.[0] ?? "";
  return entry.min ?? 0;
}

function formatValue(entry, value) {
  if (entry.type === "int") return String(Math.round(Number(value)));
  if (entry.type === "float") return Number(value).toFixed(2);
  if (entry.type === "bool") return value ? "on" : "off";
  return String(value);
}

function renderStatus() {
  els.transport.textContent = running() ? "Stop" : "Start";
  els.transport.classList.toggle("power-on", running());
  els.sendPrompt.disabled = state.status !== "ready";
  els.lyrics.disabled = running();
  els.window.disabled = running();
  els.tick.textContent =
    state.tickMs == null ? "--.-" : Number(state.tickMs).toFixed(1);
  els.statusDot.className = `status-dot status-${state.status}`;
  els.statusText.textContent = state.message || state.status;
}

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

// Commit a knob value: update only the live state map and ship it. The
// knob node updates its own visuals in place (see numericKnob), so we
// never rebuild the grid mid-interaction — rebuilding would destroy the
// element the user is dragging and drop the gesture.
function commitKnobValue(name, entry, value) {
  const next = entry.type === "int" ? Math.round(Number(value)) : value;
  state.values = { ...state.values, [name]: next };
  sendParamsNow();
}

function numericKnob(name, entry) {
  const min = entry.min ?? 0;
  const max = entry.max ?? 1;
  const span = max - min || 1;
  const isInt = entry.type === "int";
  const defaultValue = clamp(Number(valueFromEntry(entry)), min, max);

  // Increment ladder shared by wheel + keyboard. Drag uses a continuous
  // pixel→value mapping instead (see below), so it isn't on this ladder.
  const coarse = isInt ? 1 : span / 100;
  const fine = isInt ? 1 : span / 1000;
  const page = isInt ? Math.max(1, Math.round(span / 10)) : span / 10;

  const cell = document.createElement("div");
  cell.className = "knob-cell";
  if (entry.description) cell.title = entry.description;

  const wrap = document.createElement("div");
  wrap.className = "knob-wrap";

  const knob = document.createElement("div");
  knob.className = "knob";
  knob.tabIndex = 0;
  knob.setAttribute("role", "slider");
  knob.setAttribute("aria-label", knobLabel(name));
  knob.setAttribute("aria-valuemin", String(min));
  knob.setAttribute("aria-valuemax", String(max));

  const rotor = document.createElement("div");
  rotor.className = "pointer-rotor";
  const pointer = document.createElement("div");
  pointer.className = "knob-pointer";
  rotor.append(pointer);
  knob.append(rotor);
  wrap.append(knob);

  const valueEl = document.createElement("div");
  valueEl.className = "knob-value";

  const label = document.createElement("div");
  label.className = "knob-label";
  label.textContent = knobLabel(name);

  cell.append(wrap, valueEl, label);

  // `current` is the quantized, committed value; `accum` is an
  // unquantized float so sub-step drag motion accumulates rather than
  // being rounded away every frame.
  let current = clamp(Number(state.values[name] ?? defaultValue), min, max);
  let accum = current;

  function paint() {
    const norm = clamp((current - min) / span, 0, 1);
    rotor.style.transform = `rotate(${-135 + norm * 270}deg)`;
    valueEl.textContent = formatValue(entry, current);
    knob.setAttribute("aria-valuenow", String(current));
    knob.setAttribute("aria-valuetext", formatValue(entry, current));
  }

  function setValue(next) {
    const q = clamp(isInt ? Math.round(next) : next, min, max);
    accum = clamp(next, min, max);
    if (q === current) return;
    current = q;
    paint();
    commitKnobValue(name, entry, q);
  }

  paint();

  // --- DAW-style vertical drag ---------------------------------------
  // Relative motion (not click-to-position): the value tracks how far
  // the pointer has moved since press, not where it landed. A full
  // min→max sweep takes ~PIXELS_PER_SPAN px of upward travel; Shift
  // drops sensitivity 5x for fine trims. Pointer capture keeps the
  // gesture alive when the cursor leaves the 76px knob.
  const PIXELS_PER_SPAN = 200;
  let dragging = false;
  let lastY = 0;

  knob.addEventListener("pointerdown", (event) => {
    if (event.button !== 0) return;
    event.preventDefault();
    knob.focus();
    dragging = true;
    accum = current;
    lastY = event.clientY;
    knob.classList.add("dragging");
    document.body.classList.add("knob-dragging");
    knob.setPointerCapture(event.pointerId);
  });

  knob.addEventListener("pointermove", (event) => {
    if (!dragging) return;
    const perPixel = (span / PIXELS_PER_SPAN) / (event.shiftKey ? 5 : 1);
    const dy = lastY - event.clientY; // up = increase
    lastY = event.clientY;
    setValue(accum + dy * perPixel);
  });

  function endDrag(event) {
    if (!dragging) return;
    dragging = false;
    knob.classList.remove("dragging");
    document.body.classList.remove("knob-dragging");
    try {
      knob.releasePointerCapture(event.pointerId);
    } catch {}
  }
  knob.addEventListener("pointerup", endDrag);
  knob.addEventListener("pointercancel", endDrag);

  // Double-click restores the knob's declared default — the DAW reset.
  knob.addEventListener("dblclick", (event) => {
    event.preventDefault();
    setValue(defaultValue);
  });

  knob.addEventListener(
    "wheel",
    (event) => {
      event.preventDefault();
      const inc = (event.shiftKey ? fine : coarse) * (event.deltaY < 0 ? 1 : -1);
      setValue(current + inc);
    },
    { passive: false },
  );

  knob.addEventListener("keydown", (event) => {
    const inc = event.shiftKey ? fine : coarse;
    let next;
    switch (event.key) {
      case "ArrowUp":
      case "ArrowRight":
        next = current + inc;
        break;
      case "ArrowDown":
      case "ArrowLeft":
        next = current - inc;
        break;
      case "PageUp":
        next = current + page;
        break;
      case "PageDown":
        next = current - page;
        break;
      case "Home":
        next = min;
        break;
      case "End":
        next = max;
        break;
      default:
        return;
    }
    event.preventDefault();
    setValue(next);
  });

  return cell;
}

function enumKnob(name, entry) {
  const cell = document.createElement("label");
  cell.className = "knob-cell";
  if (entry.description) cell.title = entry.description;

  const select = document.createElement("select");
  select.className = "select-knob";
  const options = entry.options ?? [];
  for (const optionValue of options) {
    const option = document.createElement("option");
    option.value = String(optionValue);
    option.textContent = String(optionValue);
    select.append(option);
  }
  select.value = String(state.values[name] ?? valueFromEntry(entry));

  const valueEl = document.createElement("div");
  valueEl.className = "knob-value";
  valueEl.textContent = formatValue(entry, select.value);

  select.addEventListener("change", () => {
    valueEl.textContent = formatValue(entry, select.value);
    commitKnobValue(name, entry, select.value);
  });

  const label = document.createElement("div");
  label.className = "knob-label";
  label.textContent = knobLabel(name);

  cell.append(select, valueEl, label);
  return cell;
}

function boolKnob(name, entry) {
  const cell = document.createElement("label");
  cell.className = "knob-cell";
  if (entry.description) cell.title = entry.description;

  const wrap = document.createElement("span");
  wrap.className = "bool-knob";
  const input = document.createElement("input");
  input.type = "checkbox";
  input.checked = Boolean(state.values[name] ?? valueFromEntry(entry));
  wrap.append(input);

  const valueEl = document.createElement("div");
  valueEl.className = "knob-value";
  valueEl.textContent = formatValue(entry, input.checked);

  input.addEventListener("change", () => {
    valueEl.textContent = formatValue(entry, input.checked);
    commitKnobValue(name, entry, input.checked);
  });

  const label = document.createElement("div");
  label.className = "knob-label";
  label.textContent = knobLabel(name);

  cell.append(wrap, valueEl, label);
  return cell;
}

function renderKnobs() {
  if (state.knobs.length === 0) {
    const placeholder = document.createElement("div");
    placeholder.className = "knob-placeholder";
    placeholder.textContent =
      state.status === "connecting"
        ? "loading knob bank..."
        : "knobs appear when the session starts";
    els.knobs.replaceChildren(placeholder);
    return;
  }

  const nodes = state.knobs.map(({ name, entry }) => {
    if (entry.type === "enum") return enumKnob(name, entry);
    if (entry.type === "bool") return boolKnob(name, entry);
    return numericKnob(name, entry);
  });
  els.knobs.replaceChildren(...nodes);
}

function sendParamsNow() {
  if (!state.remote || !state.player || state.status !== "ready") return;
  state.remote.sendParams(state.values, state.player.positionSec);
}

// Show a transient status note without disturbing the status itself.
function flashNote(text, holdMs = 5000) {
  if (state.noteTimer != null) window.clearTimeout(state.noteTimer);
  state.message = text;
  renderStatus();
  state.noteTimer = window.setTimeout(() => {
    state.noteTimer = null;
    if (!state.ended) {
      state.message = "";
      renderStatus();
    }
  }, holdMs);
}

async function stop() {
  if (state.paramsTimer != null) {
    window.clearInterval(state.paramsTimer);
    state.paramsTimer = null;
  }
  if (state.noteTimer != null) {
    window.clearTimeout(state.noteTimer);
    state.noteTimer = null;
  }
  state.lastSliceAt = null;
  state.ended = false;
  try {
    await state.player?.close();
  } catch {}
  try {
    state.remote?.close();
  } catch {}
  state.player = null;
  state.remote = null;
  state.tickMs = null;
  setStatus("idle");
}

function readWindowSeconds() {
  const parsed = Number(els.window.value);
  return Number.isFinite(parsed) && parsed > 0 ? parsed : null;
}

function buildConfig() {
  const prompt = els.prompt.value.trim() || DEFAULT_PROMPT;
  const lyrics = els.lyrics.value.trim();
  const config = {
    telemetry_version: 1,
    backend: "minimax",
    prompt,
  };
  if (lyrics) config.minimax_lyrics = lyrics;
  const windowS = readWindowSeconds();
  if (windowS != null) config.minimax_duration_s = windowS;
  return config;
}

async function start() {
  await stop();
  setStatus("connecting", "Connecting...");
  renderKnobs();

  try {
    const remote = new RemoteBackend(
      wsUrl(),
      new Float32Array(UPLOAD_SAMPLE_RATE * UPLOAD_SECONDS * UPLOAD_CHANNELS),
      UPLOAD_CHANNELS,
      buildConfig(),
      { sliceWorkerUrl: "/sdk/sliceDecoder.worker.js" },
    );
    state.remote = remote;

    remote.addEventListener("slice", (event) => {
      const detail = event.detail;
      const player = state.player;
      if (!player || detail.epoch !== player.swapCount) return;
      if (typeof detail.tickMs === "number" && Number.isFinite(detail.tickMs)) {
        state.tickMs = detail.tickMs;
        renderStatus();
      }
      // Hold playback until real audio exists at the head of the tape.
      // Starting the playhead at connect (the refining families' way)
      // sends it running through silence the generator then has to
      // chase; starting it on the first slice keeps the frontier ahead
      // from the first audible sample.
      if (state.status === "connecting") {
        setStatus("ready");
        void player.resume();
      }
      const startFrame = Math.floor(detail.startSample);
      if (detail.flags === SLICE_FLAG_DELTA) {
        player.addDelta(startFrame, detail.audio);
      } else {
        player.patch(startFrame, detail.audio);
      }
      state.lastSliceAt = performance.now();
      if (state.ended) {
        state.ended = false;
        state.message = "";
      }
    });
    remote.addEventListener("params", (event) => {
      const next = event.detail?.tick_ms;
      if (typeof next === "number" && Number.isFinite(next)) {
        state.tickMs = next;
        renderStatus();
      }
    });
    remote.addEventListener("close", () => {
      if (remote.closedByUser) return;
      setStatus("error", "Connection lost.");
    });

    await remote.connect();
    if (!remote.initialBuffer) throw new Error("server sent no initial buffer");

    const manifest = remote.knobManifest?.knobs ?? {};
    state.knobs = Object.entries(manifest).map(([name, entry]) => ({ name, entry }));
    state.values = Object.fromEntries(
      state.knobs.map(({ name, entry }) => [name, valueFromEntry(entry)]),
    );
    // This page exists to steer a CONTINUOUS stream, so it opts into
    // masking the model's end-of-audio token (see the knob's help text);
    // untick `endless` to let pieces cadence naturally.
    if ("minimax_endless" in state.values) state.values.minimax_endless = true;
    // Send Prompt should pivot hard: keep only the last couple of
    // seconds of the piece's history at a swap so the new caption
    // outweighs the old material (0 = keep all = slow ~20 s morph).
    // Measured: 10 s pivots up in energy but not down; 2.5 s pivots
    // both directions in ~1-4 s.
    if ("minimax_reprompt_history_s" in state.values)
      state.values.minimax_reprompt_history_s = 2.5;
    renderKnobs();

    const player = new AudioPlayer({ workletUrl: "/sdk/audio-worklet.js?v=5" });
    state.player = player;
    await player.init(remote.initialBuffer, remote.channels);
    // No resume here: the slice handler starts playback when the first
    // audio lands (~10 s: 200 AR frames plus one chunk render).

    state.paramsTimer = window.setInterval(() => {
      sendParamsNow();
      if (
        !state.ended &&
        state.status === "ready" &&
        state.lastSliceAt != null &&
        performance.now() - state.lastSliceAt > ENDED_AFTER_MS
      ) {
        state.ended = true;
        state.message = "piece ended (tape loops) — Stop, then Start for a new one";
        renderStatus();
      }
    }, PARAMS_TICK_MS);
    setStatus("connecting", "Warming up — first audio in ~10 s");
  } catch (err) {
    await stop();
    setStatus("error", err instanceof Error ? err.message : "Start failed");
  }
}

els.transport.addEventListener("click", () => {
  if (running()) void stop();
  else void start();
});

els.sendPrompt.addEventListener("click", () => {
  if (state.ended) {
    flashNote("Piece already ended — Stop, then Start for a new one");
    return;
  }
  const prompt = els.prompt.value.trim() || DEFAULT_PROMPT;
  state.remote?.sendPrompt(prompt);
  flashNote("Prompt sent — reaches the ear in a few seconds");
});

window.addEventListener("beforeunload", () => {
  try {
    state.remote?.close();
  } catch {}
});

renderStatus();
