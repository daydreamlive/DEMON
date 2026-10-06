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
// Steering packs arrive as steer_<name> entries in the session manifest.
const STEER_PREFIX = "steer_";
// One pedal per pack category (entry.meta.category), in board order.
// `cats` lists the manifest categories that land on the pedal; `hue`
// tints the faceplate. Packs with no category land on MISC.
const PEDALS = [
  { id: "genre", title: "GENRE", cats: ["genre"], hue: 12 },
  { id: "instrument", title: "INSTRUMENT", cats: ["instrument"], hue: 38 },
  { id: "sound_effect", title: "SFX", cats: ["sound_effect", "sfx"], hue: 88 },
  { id: "production", title: "PRODUCTION", cats: ["production"], hue: 150 },
  { id: "mood", title: "MOOD", cats: ["mood"], hue: 330 },
  { id: "space", title: "SPACE", cats: ["space"], hue: 245 },
  { id: "tone", title: "TONE", cats: ["timbre", "dynamics", "rhythm", "articulation"], hue: 200 },
  { id: "abstract", title: "ABSTRACT", cats: ["abstract"], hue: 280 },
  { id: "misc", title: "MISC", cats: [], hue: 210 },
];
// The first five packs predate the category field; they are timbre,
// dynamics and rhythm controls, so they sit on the TONE pedal.
const LEGACY_PACK_PEDAL = {
  steer_bright: "tone",
  steer_warm: "tone",
  steer_percussive: "tone",
  steer_rough: "tone",
  steer_density: "tone",
};
// Split only the two largest categories. A name not listed stays on its
// parent pedal. Each sub-pedal has its own bypass.
const SUB_PEDALS = {
  sound_effect: {
    environment: { title: "ENVIRONMENT / IMPACTS", names: [
      "frogs", "rain_on_surface", "sfx_insects", "sfx_ocean_waves",
      "dishes", "explosion", "impact_boom", "smash", "riser", "magic_sparkle",
    ] },
    machines: { title: "MACHINES / CROWD", names: [
      "jet", "revving", "alarm_clock", "phone_ring", "sfx_telephone", "ui_click", "laser_zap", "glitch_sfx",
      "sfx_applause", "sfx_baby_cry",
    ] },
  },
  tone: {
    texture: { title: "COLOR / GRIT", names: [
      "bright", "warm", "fizzy_highs", "shimmering", "muted_horn", "nylon_soft", "woody_body",
      "rough", "gritty", "fuzzy", "dissonant", "thick_unison", "percussive",
    ] },
    motion: { title: "RHYTHM / PLAYING", names: [
      "accelerando", "arpeggiated", "clave_pattern", "dense_arrangement", "density", "locked_groove",
      "polyrhythmic", "pulsing_synth", "quantized", "sfx_event_rate", "shaker_pulse", "swing", "tom_patterns",
      "hammered_notes", "legato_phrasing", "marcato", "palm_muted", "spiccato", "virtuosic_runs",
      "intense", "solo_build", "solo_crest",
    ] },
  },
};
const SUB_PEDAL_OF = new Map(
  Object.entries(SUB_PEDALS).flatMap(([parent, subs]) =>
    Object.entries(subs).flatMap(([sub, { names }]) =>
      names.map((n) => [`${parent}:${STEER_PREFIX}${n}`, `${parent}/${sub}`]),
    ),
  ),
);
// Calibrated knobs reach their fidelity cutoff at 1/headroom of the
// throw (server: STEERING_PACK_HEADROOM = 1.25, so 80%).
const DEFAULT_HEADROOM = 1.25;

const els = {
  blend: document.querySelector("#blend"),
  blendValue: document.querySelector("#blend-value"),
  duration: document.querySelector("#duration"),
  fixture: document.querySelector("#fixture"),
  board: document.querySelector("#board"),
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
  steer: [],
  bypassed: new Set(),
  values: {},
  status: "idle",
  message: "",
  tickMs: null,
  remote: null,
  player: null,
  paramsTimer: null,
};
let sendFeedbackTimer = null;

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

function knobLabel(name, entry) {
  if (entry?.meta?.label) return String(entry.meta.label);
  if (name.startsWith(STEER_PREFIX)) {
    return name.slice(STEER_PREFIX.length).replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());
  }
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

// Bipolar knobs (min < 0 < max) put 0 at 12 o'clock and map each sign
// to its own half of the throw, so an asymmetric range (a steering
// pack's per-sign calibrated gain) still reads as "full CW = max, full
// CCW = min". Other knobs map linearly. `pos` is the 0..1 throw.
function knobScale(min, max) {
  if (min < 0 && max > 0) {
    return {
      bipolar: true,
      toPos: (v) => (v >= 0 ? 0.5 + 0.5 * (v / max) : 0.5 - 0.5 * (v / min)),
      fromPos: (p) => (p >= 0.5 ? (p - 0.5) * 2 * max : (0.5 - p) * 2 * min),
    };
  }
  const span = max - min || 1;
  return {
    bipolar: false,
    toPos: (v) => (v - min) / span,
    fromPos: (p) => min + p * span,
  };
}

function knobTick(pos, className) {
  const tick = document.createElement("div");
  tick.className = `knob-tick ${className}`.trim();
  tick.style.transform = `rotate(${-135 + pos * 270}deg)`;
  return tick;
}

// User-facing tooltip. Catalogue packs carry internal labelling notes in
// meta.blurb (scorer, CLAP, spot-check remarks), so it is never shown for
// them: line 1 is the +/- anchors (or the label), line 2 the calibrated
// range and cutoff per sign, line 3 the pack description. The five legacy
// packs keep their original blurb + description.
function knobTooltip(name, entry, min, max) {
  const meta = entry.meta ?? {};
  const description = meta.description || entry.description || "";
  if (!name.startsWith(STEER_PREFIX)) return description;
  if (name in LEGACY_PACK_PEDAL && !meta.category) {
    return [meta.blurb, entry.description].filter(Boolean).join("\n\n");
  }
  const lines = [];
  if (meta.pos_anchor || meta.neg_anchor) {
    lines.push(`+ ${meta.pos_anchor || "-"}  /  - ${meta.neg_anchor || "-"}`);
  } else {
    lines.push(knobLabel(name, entry));
  }
  if (meta.calibrated) {
    const reached = meta.cutoff_reached ?? {};
    const cut = meta.cutoff ?? {};
    const side = (sign, limit, c, ok) =>
      `${sign}${Math.abs(Number(limit)).toFixed(1)}` +
      (c != null ? ` (cutoff ${Number(c).toFixed(1)}, ${ok === false ? "not reached" : "reached"})` : "");
    lines.push(`range ${side("+", max, cut.pos, reached.pos)}  /  ${side("-", min, cut.neg, reached.neg)}`);
  } else {
    lines.push(`range ${Number(min).toFixed(1)} to +${Number(max).toFixed(1)} (uncalibrated)`);
  }
  if (description) lines.push(description);
  return lines.join("\n");
}

function numericKnob(name, entry) {
  const min = entry.min ?? 0;
  const max = entry.max ?? 1;
  const span = max - min || 1;
  const isInt = entry.type === "int";
  const scale = knobScale(min, max);
  const defaultValue = clamp(Number(valueFromEntry(entry)), min, max);
  const meta = entry.meta ?? {};
  const label = knobLabel(name, entry);

  // Increment ladder shared by wheel + keyboard: throw (0..1) units for
  // floats, value units for ints. Drag uses a continuous pixel-to-throw
  // mapping instead (see below).
  const coarse = isInt ? 1 : 1 / 100;
  const fine = isInt ? 1 : 1 / 1000;
  const page = isInt ? Math.max(1, Math.round(span / 10)) : 1 / 10;

  const cell = document.createElement("div");
  cell.className = "knob-cell";
  const tip = knobTooltip(name, entry, min, max);
  if (tip) cell.title = tip;

  const wrap = document.createElement("div");
  wrap.className = "knob-wrap";

  const knob = document.createElement("div");
  knob.className = "knob";
  knob.tabIndex = 0;
  knob.setAttribute("role", "slider");
  knob.setAttribute("aria-label", label);
  knob.setAttribute("aria-valuemin", String(min));
  knob.setAttribute("aria-valuemax", String(max));

  const rotor = document.createElement("div");
  rotor.className = "pointer-rotor";
  const pointer = document.createElement("div");
  pointer.className = "knob-pointer";
  rotor.append(pointer);
  knob.append(rotor);
  wrap.append(knob);

  // Calibrated steering knobs: a tick at the fidelity cutoff on each
  // side (1/headroom = 80% of the throw). A sign whose screening never
  // reached the cutoff gets a hollow tick and a dot on the label.
  const unreached = [];
  if (meta.calibrated && scale.bipolar) {
    const frac = 1 / Number(meta.headroom || DEFAULT_HEADROOM);
    const reached = meta.cutoff_reached ?? {};
    wrap.append(
      knobTick(0.5 + 0.5 * frac, reached.pos === false ? "tick-unreached" : ""),
      knobTick(0.5 - 0.5 * frac, reached.neg === false ? "tick-unreached" : ""),
    );
    if (reached.pos === false) unreached.push("+");
    if (reached.neg === false) unreached.push("-");
  }

  const valueEl = document.createElement("div");
  valueEl.className = "knob-value";

  const labelEl = document.createElement("div");
  labelEl.className = "knob-label";
  labelEl.textContent = label;
  if (unreached.length) {
    cell.classList.add("knob-unreached");
    const dot = document.createElement("span");
    dot.className = "unreached-dot";
    dot.title =
      `cutoff not reached in screening (${unreached.join(" ")}): ` +
      "range uses the largest probe";
    labelEl.append(dot);
  }

  cell.append(wrap, valueEl, labelEl);

  // `current` is the quantized, committed value; `accum` is an
  // unquantized throw position so sub-step drag motion accumulates
  // rather than being rounded away every frame.
  let current = clamp(Number(state.values[name] ?? defaultValue), min, max);
  let accum = scale.toPos(current);

  function paint() {
    const norm = clamp(scale.toPos(current), 0, 1);
    rotor.style.transform = `rotate(${-135 + norm * 270}deg)`;
    valueEl.textContent = formatValue(entry, current);
    knob.setAttribute("aria-valuenow", String(current));
    knob.setAttribute("aria-valuetext", formatValue(entry, current));
  }

  function setValue(next) {
    const v = clamp(next, min, max);
    const q = isInt ? Math.round(v) : v;
    accum = clamp(scale.toPos(v), 0, 1);
    if (q === current) return;
    current = q;
    paint();
    commitKnobValue(name, entry, q);
  }

  function setPos(pos) {
    const p = clamp(pos, 0, 1);
    setValue(scale.fromPos(p));
    accum = p;
  }

  function step(inc) {
    if (isInt) setValue(current + inc);
    else setPos(scale.toPos(current) + inc);
  }

  paint();

  // --- DAW-style vertical drag ---------------------------------------
  // Relative motion (not click-to-position): the value tracks how far
  // the pointer has moved since press, not where it landed. A full
  // sweep takes ~PIXELS_PER_SPAN px of upward travel; Shift drops
  // sensitivity 5x for fine trims. Pointer capture keeps the gesture
  // alive when the cursor leaves the knob.
  const PIXELS_PER_SPAN = 200;
  let dragging = false;
  let lastY = 0;

  knob.addEventListener("pointerdown", (event) => {
    if (event.button !== 0) return;
    event.preventDefault();
    knob.focus();
    dragging = true;
    accum = scale.toPos(current);
    lastY = event.clientY;
    knob.classList.add("dragging");
    document.body.classList.add("knob-dragging");
    knob.setPointerCapture(event.pointerId);
  });

  knob.addEventListener("pointermove", (event) => {
    if (!dragging) return;
    const perPixel = 1 / PIXELS_PER_SPAN / (event.shiftKey ? 5 : 1);
    const dy = lastY - event.clientY; // up = increase
    lastY = event.clientY;
    setPos(accum + dy * perPixel);
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
      step((event.shiftKey ? fine : coarse) * (event.deltaY < 0 ? 1 : -1));
    },
    { passive: false },
  );

  knob.addEventListener("keydown", (event) => {
    const inc = event.shiftKey ? fine : coarse;
    switch (event.key) {
      case "ArrowUp":
      case "ArrowRight":
        step(inc);
        break;
      case "ArrowDown":
      case "ArrowLeft":
        step(-inc);
        break;
      case "PageUp":
        step(page);
        break;
      case "PageDown":
        step(-page);
        break;
      case "Home":
        setValue(min);
        break;
      case "End":
        setValue(max);
        break;
      default:
        return;
    }
    event.preventDefault();
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
  label.textContent = knobLabel(name, entry);

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
  label.textContent = knobLabel(name, entry);

  cell.append(wrap, valueEl, label);
  return cell;
}

function parentPedalId(name, entry) {
  const cat = String(entry.meta?.category ?? "").trim().toLowerCase();
  if (cat) {
    const pedal = PEDALS.find((p) => p.cats.includes(cat));
    return pedal ? pedal.id : "misc";
  }
  return LEGACY_PACK_PEDAL[name] ?? "misc";
}

// The pedal (or sub-pedal, "<parent>/<sub>") a steer knob lands on;
// bypass is keyed by this id.
function pedalIdFor(name, entry) {
  const parent = parentPedalId(name, entry);
  return SUB_PEDAL_OF.get(`${parent}:${name}`) ?? parent;
}

// Board order: each parent pedal's sub-pedals in table order, then the
// parent itself for any unlisted names.
function pedalDescriptors() {
  const out = [];
  for (const p of PEDALS) {
    for (const [sub, { title }] of Object.entries(SUB_PEDALS[p.id] ?? {})) {
      out.push({ id: `${p.id}/${sub}`, title: `${p.title}: ${title}`, hue: p.hue });
    }
    out.push(p);
  }
  return out;
}

function steerPedal(pedal, knobs) {
  const section = document.createElement("section");
  section.className = "plugin plugin-steer";
  section.dataset.pedal = pedal.id;
  section.style.setProperty("--hue", String(pedal.hue));
  section.setAttribute("aria-label", `${pedal.title} steering pedal`);
  const bypassed = state.bypassed.has(pedal.id);
  section.classList.toggle("bypassed", bypassed);

  const screws = document.createElement("div");
  screws.className = "screws";
  screws.setAttribute("aria-hidden", "true");
  screws.append(...Array.from({ length: 4 }, () => document.createElement("span")));

  // Up to 8 knobs on one row; more wrap into balanced rows.
  const grid = document.createElement("div");
  grid.className = "knob-grid";
  const rows = Math.ceil(knobs.length / 8);
  grid.style.setProperty("--cols", String(Math.ceil(knobs.length / rows)));
  grid.append(...knobs.map(({ name, entry }) => numericKnob(name, entry)));

  // Footswitch + LED: LED lit = engaged. Bypass sends 0 for this
  // pedal's knobs and keeps their positions for when it is re-engaged.
  const foot = document.createElement("div");
  foot.className = "pedal-foot";
  const led = document.createElement("span");
  led.className = "pedal-led";
  led.setAttribute("aria-hidden", "true");
  const title = document.createElement("div");
  title.className = "title";
  title.textContent = pedal.title;
  const sw = document.createElement("button");
  sw.type = "button";
  sw.className = "footswitch";
  sw.setAttribute("aria-pressed", String(!bypassed));
  sw.setAttribute("aria-label", `${pedal.title} bypass`);
  sw.title = "Bypass: sends 0 for this pedal's knobs and keeps their settings";
  sw.addEventListener("click", () => {
    const now = !state.bypassed.has(pedal.id);
    if (now) state.bypassed.add(pedal.id);
    else state.bypassed.delete(pedal.id);
    section.classList.toggle("bypassed", now);
    sw.setAttribute("aria-pressed", String(!now));
    sendParamsNow();
  });
  foot.append(led, title, sw);

  section.append(screws, grid, foot);
  return section;
}

// Pedals sit in two wings either side of the main pedal, each a dense
// wrap of pedals sized to their knob count. Pedals go to the lighter
// wing in board order so the main pedal stays centred.
function renderPedals() {
  for (const node of els.board.querySelectorAll(".pedal-wing")) node.remove();
  const descs = pedalDescriptors();
  const groups = new Map(descs.map((p) => [p.id, []]));
  for (const item of state.steer) groups.get(pedalIdFor(item.name, item.entry))?.push(item);
  const pedals = descs.filter((p) => groups.get(p.id).length > 0);
  els.board.classList.toggle("has-pedals", pedals.length > 0);
  if (pedals.length === 0) return;
  const wings = [document.createElement("div"), document.createElement("div")];
  const load = [0, 0];
  wings[0].className = "pedal-wing pedal-wing-left";
  wings[1].className = "pedal-wing pedal-wing-right";
  for (const p of pedals) {
    const knobs = groups.get(p.id);
    const side = load[0] <= load[1] ? 0 : 1;
    load[side] += knobs.length + 2;
    wings[side].append(steerPedal(p, knobs));
  }
  const main = els.board.querySelector(".plugin-main");
  els.board.insertBefore(wings[0], main);
  els.board.append(wings[1]);
}

function renderKnobs() {
  renderPedals();

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

// What goes on the wire: the knob values, with every knob on a
// bypassed pedal sent as 0 (its stored value is kept for un-bypass).
function wireValues() {
  if (state.bypassed.size === 0) return state.values;
  const out = { ...state.values };
  for (const { name, entry } of state.steer) {
    if (state.bypassed.has(pedalIdFor(name, entry))) out[name] = 0;
  }
  return out;
}

function sendParamsNow() {
  if (!state.remote || !state.player || state.status !== "ready") return;
  state.remote.sendParams(wireValues(), state.player.positionSec);
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
  window.clearTimeout(sendFeedbackTimer);
  els.sendPrompt.textContent = "Send Prompt";
  els.sendPrompt.classList.remove("sent");
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
  state.steer = [];
  renderKnobs();
  setStatus("idle");
}

function readDuration() {
  const parsed = Number(els.duration.value);
  return Number.isFinite(parsed) && parsed > 0 ? parsed : null;
}

function buildConfig() {
  const prompt = els.promptA.value.trim();
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
    const all = Object.entries(manifest).map(([name, entry]) => ({ name, entry }));
    state.values = Object.fromEntries(
      all.map(({ name, entry }) => [name, valueFromEntry(entry)]),
    );
    state.knobs = all.filter(({ name }) => !name.startsWith(STEER_PREFIX));
    state.steer = all.filter(({ name }) => name.startsWith(STEER_PREFIX));
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
  const prompt = els.promptA.value.trim();
  const promptB = els.promptB.value.trim();
  state.remote?.sendPrompt(
    prompt,
    undefined,
    undefined,
    promptB && promptB !== prompt ? promptB : undefined,
  );
  els.sendPrompt.textContent = "Sent";
  els.sendPrompt.classList.add("sent");
  window.clearTimeout(sendFeedbackTimer);
  sendFeedbackTimer = window.setTimeout(() => {
    els.sendPrompt.textContent = "Send Prompt";
    els.sendPrompt.classList.remove("sent");
  }, 1200);
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
