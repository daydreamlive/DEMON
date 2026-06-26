import {
  AudioPlayer,
  RemoteBackend,
  SLICE_FLAG_DELTA,
} from "/sdk/demon-client.js";

const DEFAULT_PROMPT =
  "driving cinematic synthwave, analog arpeggios, gated reverb snare, " +
  "wide saw-lead, 152 bpm, G minor, 4/4";
const DEFAULT_FIXTURE = "low_fi_Gm_loop_60s_gnm.wav";
const STUB_FRAMES = 9600;
const STUB_CHANNELS = 2;
const PARAMS_TICK_MS = 80;

const els = {
  blend: document.querySelector("#blend"),
  blendValue: document.querySelector("#blend-value"),
  duration: document.querySelector("#duration"),
  fixture: document.querySelector("#fixture"),
  knobs: document.querySelector("#knobs"),
  promptA: document.querySelector("#prompt-a"),
  promptB: document.querySelector("#prompt-b"),
  sendPrompt: document.querySelector("#send-prompt"),
  statusDot: document.querySelector("#status-dot"),
  statusText: document.querySelector("#status-text"),
  tick: document.querySelector("#tick"),
  transport: document.querySelector("#transport"),
};

const state = {
  fixtures: [],
  knobs: [],
  values: {},
  status: "idle",
  message: "",
  tickMs: null,
  remote: null,
  player: null,
  paramsTimer: null,
};

els.promptA.value = DEFAULT_PROMPT;

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
  return name.replace(/^sa3_/, "").replace(/_/g, " ");
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
  els.blend.disabled = state.status !== "ready" || !els.promptB.value.trim();
  els.fixture.disabled = running();
  els.duration.disabled = running();
  els.tick.textContent =
    state.tickMs == null ? "--.-" : Number(state.tickMs).toFixed(1);
  els.statusDot.className = `status-dot status-${state.status}`;
  els.statusText.textContent = state.message || state.status;
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

async function stop() {
  if (state.paramsTimer != null) {
    window.clearInterval(state.paramsTimer);
    state.paramsTimer = null;
  }
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

function readDuration() {
  const parsed = Number(els.duration.value);
  return Number.isFinite(parsed) && parsed > 0 ? parsed : null;
}

function buildConfig() {
  const prompt = els.promptA.value.trim() || DEFAULT_PROMPT;
  const promptB = els.promptB.value.trim();
  const config = {
    telemetry_version: 1,
    backend: "sa3",
    prompt,
    use_server_fixture: true,
    fixture_name: els.fixture.value || DEFAULT_FIXTURE,
  };
  if (promptB && promptB !== prompt) config.prompt_b = promptB;
  const durationS = readDuration();
  if (durationS != null) config.sa3_duration_s = durationS;
  return config;
}

async function start() {
  await stop();
  setStatus("connecting", "Connecting...");
  renderKnobs();

  try {
    els.blend.value = "0";
    els.blendValue.textContent = "0.00";

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
      if (typeof detail.tickMs === "number" && Number.isFinite(detail.tickMs)) {
        state.tickMs = detail.tickMs;
        renderStatus();
      }
      const startFrame = Math.floor(detail.startSample);
      if (detail.flags === SLICE_FLAG_DELTA) {
        player.addDelta(startFrame, detail.audio);
      } else {
        player.patch(startFrame, detail.audio);
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
    renderKnobs();

    const player = new AudioPlayer({ workletUrl: "/sdk/audio-worklet.js?v=5" });
    state.player = player;
    await player.init(remote.initialBuffer, remote.channels);
    await player.resume();

    state.paramsTimer = window.setInterval(sendParamsNow, PARAMS_TICK_MS);
    setStatus("ready");
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
  const prompt = els.promptA.value.trim() || DEFAULT_PROMPT;
  const promptB = els.promptB.value.trim();
  state.remote?.sendPrompt(
    prompt,
    undefined,
    undefined,
    promptB && promptB !== prompt ? promptB : undefined,
  );
});

els.blend.addEventListener("input", () => {
  const value = Number(els.blend.value);
  els.blendValue.textContent = value.toFixed(2);
  state.remote?.sendSetPromptBlend(value);
});

els.promptB.addEventListener("input", renderStatus);

window.addEventListener("beforeunload", () => {
  try {
    state.remote?.close();
  } catch {}
});

renderStatus();
renderFixtures();
void fetchFixtures();
