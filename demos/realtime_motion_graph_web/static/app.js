// Video-first performance UI. Auto-starts on page load.

import { RemoteBackend, SAMPLE_RATE, SLICE_FLAG_DELTA } from "./protocol.js";
import { AudioPlayer } from "./audio.js";
import { HUD } from "./hud.js";
import { VideoLayer } from "./video.js";
import { GraphRenderer } from "./graph.js";
import { initCursor } from "./cursor.js";
import { EffectsRenderer } from "./effects.js";
import { initRibbons, tickRibbons } from "./ribbons.js";

// Custom cursor inflates/deflates with the existing cursor-idle timer.
// Initialise before the await fetch below so the bulge is live as soon
// as the user moves the mouse on the start overlay.
initCursor();

// ── DOM handles ──

const $ = (id) => document.getElementById(id);
const perfScreen = $("performance");
const startOverlay = $("start-overlay");
const playBtn = $("play-btn");
const wsUrlInput = $("ws-url");
const hudCanvas = $("hud");
const statusBar = $("status-bar");
const videoA = $("video-a");
const videoB = $("video-b");
const promptAInput = $("prompt-a");
const promptBInput = $("prompt-b");
const blendSlider = $("prompt-blend");
const blendValueEl = $("blend-value");
const sendPromptBtn = $("send-prompt");
const seedBtn = $("seed-btn");
const seedValueEl = $("seed-value");
const pauseBtn = $("pause-btn");
const modeToggle = $("mode-toggle");
const graphCanvas = $("graph");
const effectsCanvas = $("effects-canvas");

// ── Config ──
// All initial values come from config.json (next to this file). Edit + refresh
// to re-tune the installation; no rebuild needed. The schema is:
//   ws_url_default — string; takes effect when ?ws= isn't passed.
//   engine         — passed verbatim to the WebSocket /init handshake;
//                    notably engine.enabled_loras is the list of LoRA ids
//                    (filename stems) to auto-enable at session start.
//   prompts.{a, b, blend}
//   controls.<param> — initial value for each slider.
//   controls.lora_default_strength — initial value for any LoRA strength
//                                    slider (per-id defaults are not used).
//   seed, video_bpm
// SLIDER_META below is *behavior* (max range / arrow-key step / pro flag) —
// not a starting value, so it stays in code.

const userConfig = await fetch(`config.json?t=${Date.now()}`).then((r) => {
  if (!r.ok) throw new Error(`config.json fetch failed: ${r.status}`);
  return r.json();
});

// Server-info: when the server runs with --no-backend, this returns
// {no_backend: true} and the Play handler skips the WebSocket path.
// Best-effort: if the endpoint isn't there (older server) we treat it as
// backend-present and let WS try as usual.
const serverInfo = await fetch("/api/server-info")
  .then((r) => (r.ok ? r.json() : { no_backend: false }))
  .catch(() => ({ no_backend: false }));

// LoRA catalog comes from /api/loras — a cheap filesystem scan of
// MODELS_DIR/loras/ that runs on the static-file server, so we can
// render the Library panel immediately on page load (no Play required,
// no WebSocket required, no model load required).  When Play kicks
// off, the WS init config carries `enabled_loras` derived from
// whatever rows the operator has toggled on, so the pipeline starts
// with exactly the right LoRAs from the jump.
const loraInitial = await fetch("/api/loras")
  .then((r) => (r.ok ? r.json() : { dir: "", loras: [] }))
  .catch(() => ({ dir: "", loras: [] }));

const CONFIG = {
  ...userConfig.engine,
  prompt: userConfig.prompts.a,
};

// LoRA UI defaults.  Per-id slider behavior is uniform; the catalog drives
// identity, the user names them on disk via the filename stem.
const LORA_SLIDER_MAX = 2.0;
const LORA_SLIDER_STEP = 0.2;
const LORA_DEFAULT_STRENGTH = userConfig.controls?.lora_default_strength ?? 0.0;

const VIDEO_BPM = userConfig.video_bpm;

// WS URL: ?ws= query param > config.ws_url_default > same-host fallback.
function resolveDefaultWsUrl() {
  const params = new URLSearchParams(location.search);
  if (params.has("ws")) return params.get("ws");
  if (userConfig.ws_url_default) return userConfig.ws_url_default;
  const proto = location.protocol === "https:" ? "wss:" : "ws:";
  return `${proto}//${location.host}`;
}
wsUrlInput.value = resolveDefaultWsUrl();

// Slider behavior (max range, arrow-key step, pro-pane membership). Initial
// values are loaded separately from userConfig.controls below.  Per-LoRA
// `lora_str_<id>` sliders are added dynamically when the server's catalog
// arrives — they are NOT listed here.
const SLIDER_META = {
  denoise:       { max: 1.0, step: 0.10 },
  hint_strength: { max: 2.0, step: 0.20 },

  feedback:      { max: 1.0, step: 0.10, pro: true },
  shift:         { max: 1.0, step: 0.10, pro: true },
  noise_share:   { max: 1.0, step: 0.10, pro: true },
  ode_noise:     { max: 0.5, step: 0.05, pro: true },

  ch_g0: { max: 3.0, step: 0.15, pro: true },
  ch_g1: { max: 3.0, step: 0.15, pro: true },
  ch_g2: { max: 3.0, step: 0.15, pro: true },
  ch_g3: { max: 3.0, step: 0.15, pro: true },
  ch_g4: { max: 3.0, step: 0.15, pro: true },
  ch_g5: { max: 3.0, step: 0.15, pro: true },
  ch_g6: { max: 3.0, step: 0.15, pro: true },
  ch_g7: { max: 3.0, step: 0.15, pro: true },

  ch13: { max: 3.0, step: 0.15, pro: true },
  ch14: { max: 3.0, step: 0.15, pro: true },
  ch19: { max: 3.0, step: 0.15, pro: true },
  ch23: { max: 3.0, step: 0.15, pro: true },
  ch29: { max: 3.0, step: 0.15, pro: true },
  ch56: { max: 3.0, step: 0.15, pro: true },

  // DCW (wavelet-domain post-step correction). Numeric knobs only;
  // dcw_enabled / dcw_mode / dcw_wavelet are non-slider controls handled
  // via dcwState below.
  dcw_scaler:      { max: 0.20, step: 0.01, pro: true },
  dcw_high_scaler: { max: 0.10, step: 0.01, pro: true },
};

// Non-slider DCW state. Sent every tick alongside the slider raw dict so
// toggle/dropdown changes take effect on the very next tick (no rebuild,
// piggybacks on the existing per-tick params message).
const DCW_MODES = ["low", "high", "double", "pix"];
const DCW_WAVELETS = ["haar", "db4", "sym8", "db8"];

const SLIDER_DEFS = Object.fromEntries(
  Object.entries(SLIDER_META).map(([name, meta]) => [
    name,
    { ...meta, default: userConfig.controls[name] ?? 0 },
  ])
);

// LoRA library state.  Populated from the server's `ready` payload and any
// later `lora_catalog` updates.  Source of truth for what the UI renders
// in the Library tile and which entries the perimeter / hotkey / MIDI / FX
// slots target.  Each entry: {id, name, state, strength}.
let loraCatalog = [];
const loraStrengths = {};  // id -> client-side slider value

function enabledLoraIds() {
  return loraCatalog.filter((d) => d.state === "enabled").map((d) => d.id);
}

function loraStrengthKey(id) {
  return `lora_str_${id}`;
}

// Hotkey (a/s/d/g) -> slider param name.  `s` and `d` are dynamic: they
// resolve to the first / second currently enabled LoRA at lookup time, so
// toggling LoRAs in the Library panel doesn't dangle the keyboard
// shortcuts.  Returns null when no LoRA fills that slot.
const STATIC_SLIDER_KEYS = { a: "denoise", g: "hint_strength" };

function resolveSliderKey(letter) {
  if (letter in STATIC_SLIDER_KEYS) return STATIC_SLIDER_KEYS[letter];
  if (letter === "s" || letter === "d") {
    const ids = enabledLoraIds();
    const idx = letter === "s" ? 0 : 1;
    return ids[idx] ? loraStrengthKey(ids[idx]) : null;
  }
  return null;
}

const SLIDER_KEYS_LETTERS = ["a", "s", "d", "g"];

// Display modes. Graph is the default; the toggle button cycles to the
// video subtree. CSS swaps visibility based on body[data-mode].
const DISPLAY_MODES = ["graph", "video"];
let currentMode = "graph";
function setMode(mode) {
  currentMode = mode;
  document.body.dataset.mode = mode;
  if (modeToggle) modeToggle.textContent = mode.toUpperCase();
}
modeToggle?.addEventListener("click", () => {
  const i = DISPLAY_MODES.indexOf(currentMode);
  setMode(DISPLAY_MODES[(i + 1) % DISPLAY_MODES.length]);
});
setMode("graph");

// ── State ──

let session = null;

const sliderValues = {};
for (const [name, def] of Object.entries(SLIDER_DEFS)) {
  sliderValues[name] = def.default;
}

// Seed is not a slider; it's a random value set by button press.
let seedValue = userConfig.seed;

// DCW non-numeric state (boolean toggle + two enum strings). Mirrors the
// dcw_enabled / dcw_mode / dcw_wavelet kwargs read by StreamDenoise.execute().
const dcwState = {
  enabled: userConfig.controls.dcw_enabled ?? false,
  mode: userConfig.controls.dcw_mode ?? "double",
  wavelet: userConfig.controls.dcw_wavelet ?? "haar",
};

function randomizeSeed() {
  seedValue = Math.random();
  seedValueEl.textContent = seedValue.toFixed(2);
}

let activePrompt = CONFIG.prompt;
let activeKey = userConfig.engine.key ?? "G# minor";
let blendValue = userConfig.prompts.blend;
let heldKeys = new Set();

// Apply config-driven initial values to the DOM. Slider tracks/values get
// populated by initSliders() from sliderValues, which is itself seeded from
// SLIDER_DEFS.default, so this only needs to handle elements outside the
// slider system: prompt textareas, blend slider/value, seed display.
function applyConfigToDom() {
  promptAInput.value = userConfig.prompts.a;
  promptBInput.value = userConfig.prompts.b;
  blendSlider.value = String(blendValue);
  blendValueEl.textContent = blendValue.toFixed(2);
  seedValueEl.textContent = seedValue.toFixed(2);
}

// ── Idle reset ──
// After `reset_seconds` of no user interaction (pointer, key, MIDI), revert
// every control to its config default and re-send the active prompt. For an
// installation this keeps the next visitor from inheriting the previous
// operator's tweaks. Set reset_seconds to 0 in config.json to disable.
let idleTimer = null;

function resetToDefaults() {
  for (const name of Object.keys(SLIDER_DEFS)) {
    sliderValues[name] = SLIDER_DEFS[name].default;
    updateSliderUI(name);
  }
  seedValue = userConfig.seed;
  blendValue = userConfig.prompts.blend;
  activeKey = userConfig.engine.key ?? activeKey;
  // DCW non-numeric controls back to config defaults; re-render reflects
  // the change in the (possibly closed) Advanced sheet on next open.
  dcwState.enabled = userConfig.controls.dcw_enabled ?? false;
  dcwState.mode = userConfig.controls.dcw_mode ?? "double";
  dcwState.wavelet = userConfig.controls.dcw_wavelet ?? "haar";
  const dcwToggle = document.querySelector('[data-role="dcw-enabled"]');
  if (dcwToggle) {
    dcwToggle.textContent = `DCW: ${dcwState.enabled ? "ON" : "OFF"}`;
    dcwToggle.classList.toggle("active", dcwState.enabled);
  }
  for (const sel of document.querySelectorAll(".dcw-select")) {
    const parentLabel = sel.previousElementSibling?.textContent || "";
    const key = parentLabel.includes("mode") ? "mode" : "wavelet";
    sel.value = dcwState[key];
  }

  promptAInput.value = userConfig.prompts.a;
  promptBInput.value = userConfig.prompts.b;
  blendSlider.value = String(blendValue);
  blendValueEl.textContent = blendValue.toFixed(2);
  seedValueEl.textContent = seedValue.toFixed(2);

  // Close the Advanced drawer so a left-open operator panel doesn't greet
  // the next visitor. Goes through setDrawerOpen so the body's `drawer-open`
  // class (used by the graph-mode squeeze) stays in sync.
  setDrawerOpen(false);

  // Re-send the default prompt so audio character actually reverts.
  activePrompt = computeBlendedPrompt();
  session?.remote?.sendPrompt(activePrompt, activeKey);
}

function bumpIdleTimer() {
  const secs = userConfig.reset_seconds ?? 0;
  if (secs <= 0) return;
  clearTimeout(idleTimer);
  idleTimer = setTimeout(resetToDefaults, secs * 1000);
}

function initIdleReset() {
  // Capture-phase so we count interactions even when handlers stopPropagation.
  document.addEventListener("pointerdown", bumpIdleTimer, true);
  document.addEventListener("keydown",     bumpIdleTimer, true);
  // MIDI events are wired separately in bindAccess() — they call bumpIdleTimer
  // directly because they don't surface as DOM events.
  bumpIdleTimer();
}

// ── Cursor auto-hide ──
// After a short idle window with no mouse motion, hide the cursor (kiosk
// hygiene — a stuck pointer ruins the installation aesthetic). Any movement
// brings it back. Keyboard / MIDI / touch don't toggle it; only real mouse
// motion does, which matches operator expectation.
const CURSOR_IDLE_MS = 2000;
let cursorIdleTimer = null;

function showCursor() {
  document.body.classList.remove("cursor-idle");
  clearTimeout(cursorIdleTimer);
  cursorIdleTimer = setTimeout(() => {
    document.body.classList.add("cursor-idle");
  }, CURSOR_IDLE_MS);
}

function initCursorAutoHide() {
  document.addEventListener("mousemove", showCursor, { passive: true });
  showCursor();
}

// Global timer state -- persists across reconnects.
let globalTimerStart = null;
let globalTimerInterval = null;
let globalPausedElapsed = 0;
let globalTimerPaused = false;

// ── Helpers ──

function showStatus(msg) {
  statusBar.textContent = msg;
  statusBar.classList.remove("hidden");
}
function hideStatus() { statusBar.classList.add("hidden"); }

// ── Audio decoding (works for both audio and video containers) ──

async function decodeAudioBuffer(arrayBuf) {
  const tmpCtx = new (window.OfflineAudioContext || window.webkitOfflineAudioContext)(1, 1, SAMPLE_RATE);
  const decoded = await tmpCtx.decodeAudioData(arrayBuf.slice(0));
  const outChannels = Math.min(2, decoded.numberOfChannels);
  const targetFrames = Math.ceil(decoded.duration * SAMPLE_RATE);
  const offline = new OfflineAudioContext(outChannels, targetFrames, SAMPLE_RATE);
  const src = offline.createBufferSource();
  src.buffer = decoded;
  src.connect(offline.destination);
  src.start();
  const rendered = await offline.startRendering();

  let frames = rendered.length;
  frames = Math.min(frames, Math.floor(60.0 * SAMPLE_RATE));
  const pool = 1920 * 5;
  frames = frames - (frames % pool);

  const interleaved = new Float32Array(frames * outChannels);
  for (let c = 0; c < outChannels; c++) {
    const data = rendered.getChannelData(c);
    for (let i = 0; i < frames; i++) interleaved[i * outChannels + c] = data[i];
  }
  return { interleaved, channels: outChannels, frames };
}

// ── Slider UI ──

const sliderEls = new Map();

function bindSlider(group) {
  const param = group.dataset.param;
  const track = group.querySelector(".slider-track");
  const fill = group.querySelector(".slider-fill");
  const thumb = group.querySelector(".slider-thumb");
  const valEl = group.querySelector(".slider-value");

  sliderEls.set(param, { group, track, fill, thumb, valEl });
  updateSliderUI(param);

  let dragging = false;
  // Timestamp of the most recent pointerdown on this track. Used to detect
  // a quick second click as a "reset to default" gesture — standard
  // double-click-reset behavior in mixer / DAW UIs. Tracking it ourselves
  // (instead of leaning on the synthetic `dblclick` event) is reliable in
  // the presence of pointer capture and cross-browser quirks.
  let lastDownAt = 0;
  const DBLCLICK_MS = 350;

  const setFromY = (clientY) => {
    const rect = track.getBoundingClientRect();
    const frac = 1 - Math.max(0, Math.min(1, (clientY - rect.top) / rect.height));
    sliderValues[param] = frac * SLIDER_DEFS[param].max;
    updateSliderUI(param);
  };

  track.addEventListener("pointerdown", (e) => {
    // Right-click is reserved for MIDI learn (handled at the document level
    // via contextmenu); bail before any drag state is set so the learn flow
    // takes over cleanly.
    if (e.button !== 0) return;
    const now = performance.now();
    if (now - lastDownAt < DBLCLICK_MS) {
      // Second click within the dblclick window: reset to default.
      sliderValues[param] = SLIDER_DEFS[param].default;
      updateSliderUI(param);
      bumpIdleTimer();
      lastDownAt = 0;
      dragging = false;
      group.classList.remove("dragging");
      return;
    }
    lastDownAt = now;
    dragging = true;
    group.classList.add("dragging");
    track.setPointerCapture(e.pointerId);
    setFromY(e.clientY);
  });
  track.addEventListener("pointermove", (e) => { if (dragging) setFromY(e.clientY); });
  const endDrag = () => {
    dragging = false;
    group.classList.remove("dragging");
  };
  track.addEventListener("pointerup", endDrag);
  track.addEventListener("pointercancel", endDrag);
}

const DISPLAY_NAMES = {
  noise_share: "nshare",
  ode_noise: "ode",
  hint_strength: "structure strength",
  dcw_scaler: "DCW low",
  dcw_high_scaler: "DCW high",
};

function displayNameFor(param) {
  if (param in DISPLAY_NAMES) return DISPLAY_NAMES[param];
  if (param.startsWith("lora_str_")) {
    const id = param.slice("lora_str_".length);
    const entry = loraCatalog.find((d) => d.id === id);
    return entry?.name || id;
  }
  return param.replace(/_/g, " ");
}

function createSliderGroup(param) {
  const def = SLIDER_DEFS[param];
  const frac = (sliderValues[param] / def.max * 100).toFixed(1);
  const label = displayNameFor(param);
  const group = document.createElement("div");
  group.className = "slider-group";
  group.dataset.param = param;
  group.innerHTML = `
    <div class="slider-label">${label}</div>
    <div class="slider-track">
      <div class="slider-fill" style="height: ${frac}%"></div>
      <div class="slider-thumb" style="bottom: ${frac}%"></div>
    </div>
    <div class="slider-value">${sliderValues[param].toFixed(2)}</div>
  `;
  return group;
}

// Build a labeled tile that contains a row of fader strips. Tiles are
// the unit of grouping inside the drawer — the DOM hierarchy is
// `.mixer-tiles > .mixer-tile > .mixer-channels > .slider-group`.
function buildChannelTile(label, params) {
  const tile = document.createElement("div");
  tile.className = "mixer-tile";
  const labelEl = document.createElement("div");
  labelEl.className = "mixer-tile-label";
  labelEl.textContent = label;
  const channels = document.createElement("div");
  channels.className = "mixer-channels";
  for (const p of params) channels.appendChild(createSliderGroup(p));
  tile.appendChild(labelEl);
  tile.appendChild(channels);
  return { tile, channelEls: Array.from(channels.children) };
}

function initSliders() {
  // Bind the four hardcoded faders in the Main tile.
  for (const group of document.querySelectorAll("#sliders .slider-group")) {
    bindSlider(group);
  }

  // Build and insert the dynamic tiles between the Main tile and the
  // (already-in-HTML) Seed tile. Order: Main → Engine → Channel Gains →
  // Channels → DCW → Seed → Prompts.
  const tilesContainer = document.getElementById("mixer-tiles");
  const seedTile = tilesContainer.querySelector('[data-tile="seed"]');

  const insertTile = (label, params) => {
    const { tile, channelEls } = buildChannelTile(label, params);
    tilesContainer.insertBefore(tile, seedTile);
    for (const el of channelEls) bindSlider(el);
  };
  insertTile("Engine",        ["feedback", "shift", "noise_share", "ode_noise"]);
  insertTile("Channel Gains", ["ch_g0", "ch_g1", "ch_g2", "ch_g3", "ch_g4", "ch_g5", "ch_g6", "ch_g7"]);
  insertTile("Channels",      ["ch13", "ch14", "ch19", "ch23", "ch29", "ch56"]);

  // DCW tile groups its two faders (low/high scaler) with the on/off
  // toggle and mode/wavelet selectors — DCW belongs together as one
  // logical block, not split across the mixer.
  const dcwTile = document.createElement("div");
  dcwTile.className = "mixer-tile";
  dcwTile.dataset.tile = "dcw";
  const dcwLabel = document.createElement("div");
  dcwLabel.className = "mixer-tile-label";
  dcwLabel.textContent = "DCW";
  const dcwChannels = document.createElement("div");
  dcwChannels.className = "mixer-channels";
  const dcwLow = createSliderGroup("dcw_scaler");
  const dcwHigh = createSliderGroup("dcw_high_scaler");
  dcwChannels.appendChild(dcwLow);
  dcwChannels.appendChild(dcwHigh);
  dcwChannels.appendChild(buildDcwPanel());
  dcwTile.appendChild(dcwLabel);
  dcwTile.appendChild(dcwChannels);
  tilesContainer.insertBefore(dcwTile, seedTile);
  bindSlider(dcwLow);
  bindSlider(dcwHigh);

  // Library tile placeholder.  Filled in by applyLoraCatalog() once the
  // server publishes its catalog.  Inserted after Main so it sits next
  // to the user-facing faders.
  const libTile = document.createElement("div");
  libTile.className = "mixer-tile mixer-tile-library";
  libTile.dataset.tile = "lora-library";
  const libLabel = document.createElement("div");
  libLabel.className = "mixer-tile-label";
  libLabel.textContent = "LoRA library";
  const libBody = document.createElement("div");
  libBody.className = "lora-library-body";
  libBody.dataset.role = "lora-library-body";
  libTile.appendChild(libLabel);
  libTile.appendChild(libBody);
  const mainTile = tilesContainer.querySelector('[data-tile="main"]');
  tilesContainer.insertBefore(libTile, mainTile?.nextSibling ?? seedTile);
}

// ── LoRA library state mgmt ──
// Source of truth is `loraCatalog`.  Lifecycle:
//   1. /api/loras at page load -> initial catalog (all REGISTERED).  The
//      first DEFAULT_AUTO_ENABLE_COUNT entries are flipped to ENABLED in
//      the local UI by default — that's the demo policy, the server
//      doesn't impose it.
//   2. User toggles in the Library panel pre-Play -> mutates
//      loraCatalog locally (no network).
//   3. On Play, the WS init config carries `enabled_loras` derived from
//      whatever is currently ENABLED in loraCatalog.
//   4. Post-Play, toggling sends enable_lora / disable_lora over the WS;
//      the server pushes a `lora_catalog` update and we re-render from
//      that authoritative state.
const DEFAULT_AUTO_ENABLE_COUNT = 2;

function applyLoraCatalog(newCatalog) {
  loraCatalog = Array.isArray(newCatalog) ? newCatalog.slice() : [];
  const seen = new Set();
  for (const d of loraCatalog) {
    seen.add(d.id);
    const key = loraStrengthKey(d.id);
    if (!(key in SLIDER_DEFS)) {
      SLIDER_DEFS[key] = {
        max: LORA_SLIDER_MAX,
        step: LORA_SLIDER_STEP,
        default: LORA_DEFAULT_STRENGTH,
      };
    }
    if (!(key in sliderValues)) {
      // Server sometimes echoes back a non-zero strength (e.g. when a
      // LoRA was already enabled on a prior connect).  Honor it.
      sliderValues[key] = typeof d.strength === "number" && d.strength > 0
        ? d.strength : LORA_DEFAULT_STRENGTH;
    }
  }
  // Drop SLIDER_DEFS / sliderValues entries for ids that left the catalog.
  for (const k of Object.keys(SLIDER_DEFS)) {
    if (!k.startsWith("lora_str_")) continue;
    const id = k.slice("lora_str_".length);
    if (!seen.has(id)) {
      delete SLIDER_DEFS[k];
      delete sliderValues[k];
    }
  }
  rebuildLoraLibraryUI();
  refreshLoraEdgeBars();
}

// Local-only mutation: flip a row's enabled state in the catalog without
// touching the network.  Used pre-Play and as the optimistic update
// while a WS message is in flight.
function setLoraEnabledLocal(id, enabled) {
  const desc = loraCatalog.find((d) => d.id === id);
  if (!desc) return;
  desc.state = enabled ? "enabled" : "registered";
  rebuildLoraLibraryUI();
  refreshLoraEdgeBars();
}

function getEnabledLoraIdsForConfig() {
  return loraCatalog.filter((d) => d.state === "enabled").map((d) => d.id);
}

// Strength dict keyed by id, for the WS init config.  The server uses
// this to enable each LoRA at its target strength in one refit, so the
// first decoded window already has the LoRA contributing — without
// this, enabling at 0 and waiting for per-tick set_strength to ramp it
// up produces a "first ~5s sounds like the LoRA is missing" glitch
// because the streaming pipeline depth covers several decoded seconds.
function getEnabledLoraStrengthsForConfig() {
  const out = {};
  for (const d of loraCatalog) {
    if (d.state !== "enabled") continue;
    const v = sliderValues[loraStrengthKey(d.id)];
    if (typeof v === "number") out[d.id] = v;
  }
  return out;
}

function rebuildLoraLibraryUI() {
  const body = document.querySelector('[data-role="lora-library-body"]');
  if (!body) return;
  body.innerHTML = "";
  if (loraCatalog.length === 0) {
    const empty = document.createElement("div");
    empty.className = "lora-library-empty";
    empty.textContent = "No LoRAs in MODELS_DIR/loras/";
    body.appendChild(empty);
    return;
  }
  for (const d of loraCatalog) {
    body.appendChild(buildLoraRow(d));
  }
}

// One row per catalog entry.  Layout:
//   [switch]  name              [horizontal slider]  0.30
// Switch and slider are interactive in both pre-Play and post-Play; the
// slider is locked while the LoRA is off so the operator can see the
// stored value but can't accidentally drag it.
function buildLoraRow(desc) {
  const enabled = desc.state === "enabled";
  const param = loraStrengthKey(desc.id);

  const row = document.createElement("div");
  row.className = "lora-row";
  row.dataset.loraId = desc.id;
  row.dataset.state = enabled ? "enabled" : "disabled";
  row.dataset.param = param;  // makes the row a MIDI-learn target

  // Toggle switch.  Visual: pill with a sliding thumb.  No text label
  // (text labels for ON/OFF read as redundant when the thumb position
  // already says it).  aria-checked carries semantics for screen readers.
  const sw = document.createElement("button");
  sw.type = "button";
  sw.className = "lora-switch";
  sw.setAttribute("role", "switch");
  sw.setAttribute("aria-checked", enabled ? "true" : "false");
  sw.setAttribute("aria-label", `${desc.name || desc.id} — ${enabled ? "enabled" : "disabled"}, click to toggle`);
  sw.innerHTML = `<span class="lora-switch-thumb" aria-hidden="true"></span>`;
  sw.addEventListener("click", () => {
    const wasEnabled = loraCatalog.find((d) => d.id === desc.id)?.state === "enabled";
    const willEnable = !wasEnabled;
    // Optimistic local update so the UI feels instant.  Pre-Play this is
    // the only update; post-Play the server's lora_catalog broadcast
    // confirms or corrects.
    setLoraEnabledLocal(desc.id, willEnable);
    if (session?.remote) {
      if (willEnable) {
        // Pass the slider's current value so the server refits at it
        // in one shot, avoiding the "first decode window sounds like
        // the LoRA is missing" glitch.
        const strength = sliderValues[loraStrengthKey(desc.id)];
        session.remote.sendEnableLora(desc.id, strength);
      } else {
        session.remote.sendDisableLora(desc.id);
      }
    }
    bumpIdleTimer();
  });

  const name = document.createElement("span");
  name.className = "lora-row-name";
  name.textContent = desc.name || desc.id;
  // Click the name to toggle too — bigger hit target than the 32px switch.
  name.addEventListener("click", () => sw.click());

  // Horizontal strength slider.  Backed by sliderValues[param] so the
  // rest of the system (sendParams, graph history, MIDI binding,
  // perimeter bars, FX) reads it like any other slider.
  const slider = createLoraStrengthSlider(param);
  row.dataset.strengthSlider = "1";  // marker for findStrengthSlider

  row.appendChild(sw);
  row.appendChild(name);
  row.appendChild(slider);
  return row;
}

// Horizontal slider tailored for the Library rows.  Doesn't reuse
// .slider-group because that one is vertical and carries its own label.
// Registered with sliderEls so updateSliderUI() drives both this and the
// (hypothetical) vertical slider.
function createLoraStrengthSlider(param) {
  const wrap = document.createElement("div");
  wrap.className = "lora-strength";
  wrap.dataset.param = param;

  const track = document.createElement("div");
  track.className = "lora-strength-track";
  const fill = document.createElement("div");
  fill.className = "lora-strength-fill";
  track.appendChild(fill);

  const valEl = document.createElement("div");
  valEl.className = "lora-strength-value";

  wrap.appendChild(track);
  wrap.appendChild(valEl);

  // Adapter that exposes the same shape updateSliderUI() expects.
  // ``thumb`` is null because this slider has no separate thumb element
  // (the right edge of the fill is the visual position indicator).
  sliderEls.set(param, {
    group: wrap, track, fill, thumb: null, valEl,
    horizontal: true,
  });
  updateSliderUI(param);

  bindHorizontalSlider(wrap, param);
  return wrap;
}

function bindHorizontalSlider(wrap, param) {
  const track = wrap.querySelector(".lora-strength-track");
  const def = SLIDER_DEFS[param];
  if (!track || !def) return;

  let dragging = false;
  let lastDownAt = 0;
  const DBLCLICK_MS = 280;

  const setFromX = (clientX) => {
    const rect = track.getBoundingClientRect();
    const frac = Math.max(0, Math.min(1, (clientX - rect.left) / rect.width));
    sliderValues[param] = frac * def.max;
    updateSliderUI(param);
    bumpIdleTimer();
  };

  track.addEventListener("pointerdown", (e) => {
    // Only respond when the row is enabled.  Reading data-state at
    // pointerdown time so the lock follows the catalog state without
    // needing to rebind on toggle.
    const row = wrap.closest(".lora-row");
    if (row?.dataset.state !== "enabled") return;
    if (e.button !== 0) return;
    const now = performance.now();
    if (now - lastDownAt < DBLCLICK_MS) {
      sliderValues[param] = SLIDER_DEFS[param].default;
      updateSliderUI(param);
      bumpIdleTimer();
      lastDownAt = 0;
      return;
    }
    lastDownAt = now;
    dragging = true;
    track.setPointerCapture(e.pointerId);
    setFromX(e.clientX);
  });
  track.addEventListener("pointermove", (e) => {
    if (dragging) setFromX(e.clientX);
  });
  const endDrag = () => { dragging = false; };
  track.addEventListener("pointerup", endDrag);
  track.addEventListener("pointercancel", endDrag);
}

// DCW control strip: enable toggle + mode + wavelet selects. Numeric
// scalers are regular sliders (already in SLIDER_META); these three are
// the non-numeric companion controls.
function buildDcwPanel() {
  const wrap = document.createElement("div");
  wrap.className = "dcw-panel";

  const toggle = document.createElement("button");
  toggle.type = "button";
  toggle.className = "dcw-toggle";
  toggle.dataset.role = "dcw-enabled";
  const renderToggle = () => {
    toggle.textContent = `DCW: ${dcwState.enabled ? "ON" : "OFF"}`;
    toggle.classList.toggle("active", dcwState.enabled);
  };
  renderToggle();
  toggle.addEventListener("click", () => {
    dcwState.enabled = !dcwState.enabled;
    renderToggle();
  });

  const makeSelect = (label, options, key) => {
    const row = document.createElement("label");
    row.className = "dcw-row";
    const tag = document.createElement("span");
    tag.className = "dcw-row-label";
    tag.textContent = label;
    const sel = document.createElement("select");
    sel.className = "dcw-select";
    for (const opt of options) {
      const o = document.createElement("option");
      o.value = opt;
      o.textContent = opt;
      if (dcwState[key] === opt) o.selected = true;
      sel.appendChild(o);
    }
    sel.addEventListener("change", () => { dcwState[key] = sel.value; });
    row.appendChild(tag);
    row.appendChild(sel);
    return row;
  };

  wrap.appendChild(toggle);
  wrap.appendChild(makeSelect("DCW mode", DCW_MODES, "mode"));
  wrap.appendChild(makeSelect("wavelet", DCW_WAVELETS, "wavelet"));
  return wrap;
}

function updateSliderUI(param) {
  const el = sliderEls.get(param);
  if (el) {
    const def = SLIDER_DEFS[param];
    const frac = Math.max(0, Math.min(1, sliderValues[param] / def.max));
    const pct = (frac * 100).toFixed(1) + "%";
    if (el.horizontal) {
      el.fill.style.width = pct;
    } else {
      el.fill.style.height = pct;
      if (el.thumb) el.thumb.style.bottom = pct;
    }
    el.valEl.textContent = sliderValues[param].toFixed(2);
  }
  updateInstallBar(param);
}

// ── Installation HUD bars: pointer-driven scrub ──
// Click/drag on a perimeter bar scrubs the corresponding slider value.
// The whole bar surface is the hit target — top fills left→right (x axis),
// left and right fill bottom→top (y axis, inverted). Every set funnels
// through the same updateSliderUI path the sliders + MIDI use, so the
// Advanced sheet, the bar fill, and any backend sendParams all stay in
// sync. bumpIdleTimer() keeps the 60s reset clock alive.
// Edge bar binding.  Top bar is fixed to "denoise"; left/right bars track
// the first and second currently-enabled LoRA at pointer-time.  The
// resolver runs on every drag, so toggling the catalog re-aims the bars
// without rebinding.
function initEdgeBars() {
  const bindStatic = (selector, param, axis) => {
    const edge = document.querySelector(selector);
    const def = SLIDER_DEFS[param];
    if (!edge || !def) return;
    let dragging = false;
    const setFromPointer = (clientX, clientY) => {
      const rect = edge.getBoundingClientRect();
      const frac = axis === "x"
        ? Math.max(0, Math.min(1, (clientX - rect.left) / rect.width))
        : 1 - Math.max(0, Math.min(1, (clientY - rect.top) / rect.height));
      sliderValues[param] = frac * def.max;
      updateSliderUI(param);
      bumpIdleTimer();
    };
    edge.addEventListener("pointerdown", (e) => {
      dragging = true;
      edge.setPointerCapture(e.pointerId);
      setFromPointer(e.clientX, e.clientY);
    });
    edge.addEventListener("pointermove", (e) => {
      if (dragging) setFromPointer(e.clientX, e.clientY);
    });
    edge.addEventListener("pointerup", () => { dragging = false; });
    edge.addEventListener("pointercancel", () => { dragging = false; });
  };
  const bindLoraSlot = (selector, slotIdx) => {
    const edge = document.querySelector(selector);
    if (!edge) return;
    let dragging = false;
    const setFromPointer = (clientY) => {
      const ids = enabledLoraIds();
      const id = ids[slotIdx];
      if (!id) return;
      const param = loraStrengthKey(id);
      const def = SLIDER_DEFS[param];
      if (!def) return;
      const rect = edge.getBoundingClientRect();
      const frac = 1 - Math.max(0, Math.min(1, (clientY - rect.top) / rect.height));
      sliderValues[param] = frac * def.max;
      updateSliderUI(param);
      bumpIdleTimer();
    };
    edge.addEventListener("pointerdown", (e) => {
      dragging = true;
      edge.setPointerCapture(e.pointerId);
      setFromPointer(e.clientY);
    });
    edge.addEventListener("pointermove", (e) => {
      if (dragging) setFromPointer(e.clientY);
    });
    edge.addEventListener("pointerup", () => { dragging = false; });
    edge.addEventListener("pointercancel", () => { dragging = false; });
  };
  bindStatic(".install-edge-top", "denoise", "x");
  bindLoraSlot(".install-edge-left",  0);
  bindLoraSlot(".install-edge-right", 1);
}

// Refresh `data-bar` on the left/right perimeter edges so they point at
// the first/second enabled LoRA's id.  Called whenever the catalog state
// changes so updateInstallBar's query still hits the right element.
function refreshLoraEdgeBars() {
  const ids = enabledLoraIds();
  const left = document.querySelector(".install-edge-left");
  const right = document.querySelector(".install-edge-right");
  const updateEdge = (edge, slotIdx) => {
    if (!edge) return;
    const id = ids[slotIdx];
    edge.dataset.bar = id ? loraStrengthKey(id) : "";
    edge.classList.toggle("install-edge-empty", !id);
    const labelEl = edge.querySelector(".install-edge-label");
    if (labelEl) {
      const desc = id ? loraCatalog.find((d) => d.id === id) : null;
      labelEl.textContent = desc ? (desc.name || id) : "";
    }
  };
  updateEdge(left, 0);
  updateEdge(right, 1);
  if (ids[0]) updateInstallBar(loraStrengthKey(ids[0]));
  if (ids[1]) updateInstallBar(loraStrengthKey(ids[1]));
}

// ── Installation HUD bars ──
// Mirror sliderValues onto the screen-edge HUD frame for the public-facing
// view. Pulses on change so the active bar is visible from across a room.
function updateInstallBar(param) {
  const def = SLIDER_DEFS[param];
  if (!def) return;
  const edge = document.querySelector(`.install-edge[data-bar="${param}"]`);
  if (!edge) return;
  const frac = Math.max(0, Math.min(1, sliderValues[param] / def.max));
  edge.style.setProperty("--fill", frac);
  const valEl = edge.querySelector(`[data-bar-val="${param}"]`);
  if (valEl) valEl.textContent = sliderValues[param].toFixed(2);
  edge.classList.remove("install-edge-pulse");
  void edge.offsetWidth;
  edge.classList.add("install-edge-pulse");
  clearTimeout(edge._installPulseT);
  edge._installPulseT = setTimeout(() => edge.classList.remove("install-edge-pulse"), 600);
}

// Advanced drawer open/close. The drawer pulls up from the bottom and
// only ever closes via an explicit user action: clicking the handle bar,
// pressing Escape, or pressing `o`. There is no click-outside-to-close —
// the drawer stays put while the operator works in it.
//
// Whenever the drawer toggles we mirror the state into a `drawer-open`
// class on <body>. CSS uses that class to shrink #install-stage in graph
// mode, so the entire interface (HUD edges, graph, side blurs — all of
// it) lifts above the drawer instead of being overlaid by it. In video
// mode the drawer keeps overlaying as before.
function setDrawerOpen(open) {
  const sheet = document.getElementById("install-sheet");
  if (!sheet) return;
  sheet.classList.toggle("open", open);
  document.body.classList.toggle("drawer-open", open);
}
function toggleDrawer() {
  const sheet = document.getElementById("install-sheet");
  if (!sheet) return;
  setDrawerOpen(!sheet.classList.contains("open"));
}
function initAdvancedSheet() {
  const sheet = document.getElementById("install-sheet");
  const handle = document.getElementById("install-adv-handle");
  if (!sheet) return;
  handle?.addEventListener("click", (e) => {
    e.stopPropagation();
    toggleDrawer();
  });
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape") {
      setDrawerOpen(false);
      return;
    }
    // Skip the `o` hotkey while the operator is typing into a text field
    // so they can write 'o' in a prompt without toggling the drawer.
    const tag = e.target?.tagName;
    if (tag === "INPUT" || tag === "TEXTAREA") return;
    if (e.key === "o" || e.key === "O") {
      e.preventDefault();
      toggleDrawer();
    }
  });
}
initAdvancedSheet();

function adjustSlider(param, direction) {
  const def = SLIDER_DEFS[param];
  sliderValues[param] = Math.max(0, Math.min(def.max, sliderValues[param] + direction * def.step));
  updateSliderUI(param);
}

// ── Prompt blend UI ──

function computeBlendedPrompt() {
  const a = promptAInput.value.trim();
  const b = promptBInput.value.trim();
  if (blendValue < 0.5) return a;
  if (blendValue > 0.5) return b;
  return [a, b].filter(Boolean).join(", ");
}

function sendCurrentPrompt() {
  activePrompt = computeBlendedPrompt();
  session?.remote?.sendPrompt(activePrompt, activeKey);
}

function initPrompts() {
  promptAInput.addEventListener("keydown", (e) => e.stopPropagation());
  promptBInput.addEventListener("keydown", (e) => e.stopPropagation());

  // Slider just updates the value display; doesn't auto-send.
  blendSlider.addEventListener("input", () => {
    blendValue = parseFloat(blendSlider.value);
    blendValueEl.textContent = blendValue.toFixed(2);
  });
  blendSlider.addEventListener("change", () => {
    blendValue = parseFloat(blendSlider.value);
    blendValueEl.textContent = blendValue.toFixed(2);
  });

  // Send button sends the computed prompt.
  sendPromptBtn.addEventListener("click", sendCurrentPrompt);
}

// ── Keyboard shortcuts ──

function initKeyboard() {
  window.addEventListener("keydown", (e) => {
    if (e.target.tagName === "INPUT" || e.target.tagName === "TEXTAREA") return;
    if (perfScreen.classList.contains("hidden")) return;

    const key = e.key.toLowerCase();

    if (SLIDER_KEYS_LETTERS.includes(key)) {
      const param = resolveSliderKey(key);
      if (param) {
        heldKeys.add(key);
        const el = sliderEls.get(param);
        if (el) el.group.classList.add("active");
      }
      return;
    }

    if (key === "arrowup" || key === "arrowdown") {
      e.preventDefault();
      const dir = key === "arrowup" ? 1 : -1;
      for (const k of heldKeys) {
        const param = resolveSliderKey(k);
        if (param) adjustSlider(param, dir);
      }
      return;
    }

    if (key === "f") {
      randomizeSeed();
      return;
    }

  });

  window.addEventListener("keyup", (e) => {
    const key = e.key.toLowerCase();
    heldKeys.delete(key);
    if (SLIDER_KEYS_LETTERS.includes(key)) {
      const param = resolveSliderKey(key);
      if (param) {
        const el = sliderEls.get(param);
        if (el) el.group.classList.remove("active");
      }
    }
  });
}

// ── Global timer ──
// Disabled for the installation: the demo runs indefinitely, no expiry,
// no times-up screen. The pause logic still touches globalTimerStart but
// is harmless when it stays null.

function startGlobalTimer() { /* no-op */ }
function updateGlobalTimer() { /* no-op */ }

// ── Session (connection + video + render loop) ──

class Session {
  constructor(remote, audio, videos) {
    this.remote = remote;
    this.audio = audio;
    this.videos = videos;
    this.hud = new HUD(hudCanvas);
    this.running = true;
    this.paused = false;
    this._lastWavSwap = -1;
    this._sendInterval = null;
    this._renderRaf = null;

    this.videoLayer = new VideoLayer({ videoA, videoB, bpm: VIDEO_BPM });
    this.graph = new GraphRenderer(graphCanvas);

    // Auto-fit the focal wrapper to the actual source video aspect.
    // tokens.css ships a hardcoded 1.7778 (16:9) which stretches any
    // non-16:9 source — square footage in particular gets squashed
    // wide. Read the real ratio off the <video> element when its
    // metadata lands and re-tint --source-aspect on :root. Square-ish
    // sources also collapse the side-blur strips to zero width since
    // there are no outpainted gutters to mask.
    const updateAspect = (el) => {
      if (!el.videoWidth || !el.videoHeight) return;
      const aspect = el.videoWidth / el.videoHeight;
      document.documentElement.style.setProperty("--source-aspect", aspect.toFixed(4));
      if (Math.abs(aspect - 1) < 0.05) {
        document.documentElement.style.setProperty("--side-blur-width", "0px");
      } else {
        document.documentElement.style.removeProperty("--side-blur-width");
      }
    };
    videoA.addEventListener("loadedmetadata", () => updateAspect(videoA));
    videoB.addEventListener("loadedmetadata", () => updateAspect(videoB));

    // Multi-color ribbons painted along each slider edge (replaces the
    // old amber level meter). Tied to the slider's --fill plus the
    // shared kick value; rendered each rAF frame.
    this.ribbons = initRibbons();

    // Audio-reactive video pipeline (parallax + bloom-on-kick). Best-effort:
    // if WebGL2 is unavailable we hide the canvas and fall through to the
    // raw <video> elements behind it.
    const fx = userConfig.effects ?? {};
    this._bloomOnKick = fx.bloom_on_kick ?? 0.3;
    // CSS-side bloom is much weaker visually than additive RGB shader bloom,
    // so the UI signal gets a separate multiplier on top of bloom_on_kick.
    this._uiBloomMul = fx.ui_bloom_multiplier ?? 5.0;
    this.effects = null;
    try {
      this.effects = new EffectsRenderer(effectsCanvas);
      if (fx.parallax_strength !== undefined) this.effects.setParallaxStrength(fx.parallax_strength);
      if (fx.bloom_on_kick !== undefined)     this.effects.setBloomOnKick(fx.bloom_on_kick);
      if (fx.bloom_threshold !== undefined)   this.effects.setBloomThreshold(fx.bloom_threshold);
      if (fx.warp_strength !== undefined)     this.effects.setWarpStrength(fx.warp_strength);
    } catch (e) {
      console.error("EffectsRenderer init failed; falling back to plain video:", e);
      effectsCanvas.style.display = "none";
    }

    // Kick onset detector state. We run a 40..180 Hz band-pass on the
    // PCM mirror, take its RMS, then trigger on positive spectral flux
    // above an adaptive threshold (mean + k·std of recent flux). Fast
    // envelope decay so each hit reads as a discrete pulse instead of
    // a continuous bass amplitude (which is what the previous detector
    // was producing — picking up sustained bass / sub rumble as kicks).
    this._kLpDC = 0;        // ~30 Hz tracker; subtracted from x for high-pass
    this._kLp1 = 0;         // ~180 Hz two-pole low-pass, stage 1
    this._kLp2 = 0;         // ~180 Hz two-pole low-pass, stage 2
    this._kPrevInstant = 0; // last frame's band RMS for flux
    this._kFluxHist = [];   // ring of recent flux values (~1s @ 60fps)
    this._kLastTriggerMs = 0;
    this._kickEnv = 0;
  }

  async start() {
    pauseBtn.innerHTML = "&#9646;&#9646;";

    this.videoLayer.setVideos(this.videos);
    this.videoLayer.play(this.videos[0], "none");

    // Mirror the playing video into the blurred ambient back layer so
    // the side gutters fill with motion-matched ambience.
    const ambient = document.getElementById("install-ambient");
    if (ambient && this.videos[0]) {
      ambient.src = `videos/${this.videos[0]}`;
      ambient.muted = true;
      ambient.loop = true;
      ambient.play().catch(() => {});
    }

    if (this.remote) {
      this.remote.addEventListener("slice", (e) => {
        const d = e.detail;
        if (d.flags === SLICE_FLAG_DELTA) {
          this.audio.addDelta(d.startSample, d.audio);
        } else {
          this.audio.patch(d.startSample, d.audio);
        }
      });

      this.remote.addEventListener("close", () => {
        if (this.running) {
          this.running = false;
          showStatus("Server disconnected.");
        }
      });

      this._sendInterval = setInterval(() => this._sendTick(), 8);
    }

    this._renderRaf = requestAnimationFrame(() => this._renderLoop());
    startGlobalTimer();
  }

  async togglePause() {
    if (this.paused) {
      globalTimerStart = Date.now() - globalPausedElapsed * 1000;
      globalTimerPaused = false;
      await this.audio.resume();
      this.paused = false;
      pauseBtn.innerHTML = "&#9646;&#9646;";
    } else {
      globalPausedElapsed = (Date.now() - globalTimerStart) / 1000;
      globalTimerPaused = true;
      if (this.audio.ctx) await this.audio.ctx.suspend();
      this.paused = true;
      pauseBtn.innerHTML = "&#9654;";
    }
  }

  _sendTick() {
    if (!this.running || !this.remote || this.paused) return;
    const raw = { seed: seedValue };
    for (const name of Object.keys(SLIDER_DEFS)) {
      raw[name] = sliderValues[name];
    }
    // Non-slider DCW state (boolean toggle + two enum strings). The Python
    // side reads raw.get("dcw_enabled") etc. and forwards to pipe.set_dcw().
    raw.dcw_enabled = dcwState.enabled;
    raw.dcw_mode = dcwState.mode;
    raw.dcw_wavelet = dcwState.wavelet;
    this.remote.sendParams(raw, this.audio.positionSec);
  }

  _renderLoop() {
    if (!this.running) return;

    if (this.audio.swapCount !== this._lastWavSwap) {
      this._lastWavSwap = this.audio.swapCount;
      const mirror = this.audio.getMirror();
      if (mirror) this.hud.updateWaveform(mirror, this.audio.channels);
    }

    const dur = this.audio.duration || 1;
    const frac = Math.max(0, Math.min(1, this.audio.positionSec / dur));
    const kick = this._computeKick();

    // Bloom amplitude exposed to CSS so the perimeter HUD bars and the
    // cursor halo glow in lockstep with the shader bloom on the video.
    // The UI multiplier compensates for box-shadow being a much weaker
    // visual than additive bloom in pixel space.
    document.documentElement.style.setProperty(
      "--bloom-amount", (kick * this._bloomOnKick * this._uiBloomMul).toFixed(3)
    );

    // History stays warm in all modes so mode switches never show a blank canvas.
    this.graph.sample({ ...sliderValues, seed: seedValue }, { ...SLIDER_DEFS, seed: { max: 1 } });

    if (currentMode === "graph") {
      this.graph.draw(frac, kick);
    } else {
      this.hud.draw(frac);
    }

    const t = performance.now() / 1000;
    if (this.effects) {
      // FX flavors slot off the first/second currently-enabled LoRA:
      // slot 0 -> daft (rose bloom tint), slot 1 -> dubstep (chromatic
      // aberration).  Decoupled from any specific LoRA id so toggling
      // the catalog re-aims them.  Both default to 0 when no LoRA fills
      // the slot.
      const ids = enabledLoraIds();
      const slot0 = ids[0]
        ? sliderValues[loraStrengthKey(ids[0])] / LORA_SLIDER_MAX : 0;
      const slot1 = ids[1]
        ? sliderValues[loraStrengthKey(ids[1])] / LORA_SLIDER_MAX : 0;
      this.effects.setDaftPunk(slot0);
      this.effects.setDubstep(slot1);
      this.effects.tick(this.videoLayer.activeVideo, t, kick);
    }
    tickRibbons(this.ribbons, t, kick);

    this._renderRaf = requestAnimationFrame(() => this._renderLoop());
  }

  stop() {
    this.running = false;
    clearInterval(this._sendInterval);
    cancelAnimationFrame(this._renderRaf);
    this.videoLayer.destroy();
    this.graph.destroy();
    if (this.effects) this.effects.destroy();
    document.documentElement.style.removeProperty("--bloom-amount");
    if (this.remote) this.remote.close();
  }

  // Kick onset detector. Pipeline per rAF frame:
  //   1. Band-pass the recent PCM window to 40..180 Hz (HP via x − lp_slow,
  //      LP via two cascaded one-poles at ~180 Hz). 40 Hz HP kills DC and
  //      sub rumble; 180 Hz LP rejects the snare body.
  //   2. RMS over the window -> "instant" band amplitude.
  //   3. Spectral flux = max(0, instant − prev) — only positive changes
  //      register, so sustained bass doesn't read as a kick.
  //   4. Adaptive threshold = mean + k·std over the last ~1 s of flux.
  //   5. Trigger on flux > threshold (with absolute floor + 100 ms
  //      refractory). On trigger, kickEnv jumps to 1.0; otherwise it
  //      decays fast so each hit is a discrete pulse.
  _computeKick() {
    const mirror = this.audio.getMirror();
    if (!mirror) return this._kickEnv;
    const ch = this.audio.channels || 1;
    const win = 1024;  // ~21 ms @ 48k — short enough to catch transients
    const posFrame = Math.floor(this.audio.positionSec * SAMPLE_RATE);
    const framesTotal = Math.floor(mirror.length / ch);
    const endFrame = Math.min(framesTotal, posFrame);
    const startFrame = Math.max(0, endFrame - win);
    const n = endFrame - startFrame;
    if (n <= 0) return this._kickEnv;

    // Filter coefficients (alpha = 1 − exp(−2π·fc/fs) at 48 kHz).
    const alphaHP = 0.004;   // ~30 Hz lp tracker (used as DC blocker)
    const alphaLP = 0.024;   // ~190 Hz lp cutoff per pole
    let lpDC = this._kLpDC;
    let lp1 = this._kLp1;
    let lp2 = this._kLp2;
    let sum = 0;
    for (let f = startFrame; f < endFrame; f++) {
      let x = 0;
      for (let c = 0; c < ch; c++) x += mirror[f * ch + c];
      x /= ch;
      lpDC += alphaHP * (x - lpDC);
      const hpx = x - lpDC;
      lp1 += alphaLP * (hpx - lp1);
      lp2 += alphaLP * (lp1 - lp2);
      sum += lp2 * lp2;
    }
    this._kLpDC = lpDC;
    this._kLp1 = lp1;
    this._kLp2 = lp2;
    const instant = Math.sqrt(sum / n);

    const flux = Math.max(0, instant - this._kPrevInstant);
    this._kPrevInstant = instant;

    this._kFluxHist.push(flux);
    if (this._kFluxHist.length > 60) this._kFluxHist.shift();
    const m = this._kFluxHist.length;
    let mean = 0;
    for (let i = 0; i < m; i++) mean += this._kFluxHist[i];
    mean /= m;
    let varSum = 0;
    for (let i = 0; i < m; i++) {
      const d = this._kFluxHist[i] - mean;
      varSum += d * d;
    }
    const std = Math.sqrt(varSum / m);
    const threshold = mean + 1.8 * std;

    const now = performance.now();
    if (flux > threshold && flux > 0.003 && now - this._kLastTriggerMs > 100) {
      this._kickEnv = 1.0;
      this._kLastTriggerMs = now;
    } else {
      // 0.82^5 ≈ 0.37, ^10 ≈ 0.14 — about ten frames (~167 ms) to fade
      // visibly to zero. Tight enough that each kick reads as its own pulse.
      this._kickEnv *= 0.82;
    }

    return this._kickEnv > 1 ? 1 : this._kickEnv;
  }

  getMirror() {
    return this.audio.getMirror();
  }

  closeAudio() {
    this.audio.close();
  }
}

// ── Session lifecycle ──

async function startSession(interleaved, channels, frames, videos) {
  if (session) {
    session.stop();
    session.closeAudio();
    session = null;
  }

  let remote = null;
  let initialBuffer, initialChannels;

  if (serverInfo.no_backend) {
    // No-backend mode: skip the WebSocket entirely. The video's own audio
    // (decoded above into `interleaved`) drives the AudioPlayer, so video
    // and sound play together; Session runs with remote=null and is
    // already null-safe in that path.
    showStatus(`Loaded ${(frames / SAMPLE_RATE).toFixed(1)}s audio. UI-only mode.`);
    initialBuffer = interleaved;
    initialChannels = channels;
    applyLoraCatalog([]);
  } else {
    showStatus(`Loaded ${(frames / SAMPLE_RATE).toFixed(1)}s audio. Connecting...`);
    try {
      // Carry the operator's pre-Play LoRA selections through to the
      // pipeline.  Server enables exactly these ids on first tick — no
      // mid-stream "now turn it on" round-trip needed.
      const config = {
        ...CONFIG,
        prompt: activePrompt,
        key: activeKey,
        enabled_loras: getEnabledLoraIdsForConfig(),
        lora_strengths: getEnabledLoraStrengthsForConfig(),
      };
      remote = new RemoteBackend(wsUrlInput.value.trim(), interleaved, channels, config);
      await remote.connect();
    } catch (e) {
      showStatus(`Connection failed: ${e.message}`);
      return;
    }
    showStatus("Connected. Starting audio...");
    initialBuffer = remote.initialBuffer;
    initialChannels = remote.channels;
    // Server is now the source of truth.  Reconcile our local catalog
    // with what the server actually loaded (handles cases where the
    // server registered ad-hoc paths or a row in our pre-Play list
    // was missing from the catalog).
    if (Array.isArray(remote.loraCatalog) && remote.loraCatalog.length) {
      applyLoraCatalog(remote.loraCatalog);
    }
    remote.addEventListener("lora_catalog", (e) => {
      applyLoraCatalog(e.detail);
    });
  }

  const player = new AudioPlayer();
  await player.init(initialBuffer, initialChannels);
  hideStatus();
  session = new Session(remote, player, videos);

  await new Promise((r) => requestAnimationFrame(r));
  session.hud.resize();
  session.start();
}

// ── Init ──

playBtn.addEventListener("click", async () => {
  startOverlay.classList.add("hidden");

  showStatus("Loading video...");
  try {
    const resp = await fetch("/api/videos");
    if (!resp.ok) throw new Error("Could not fetch video list");
    const videos = await resp.json();
    if (videos.length === 0) throw new Error("No videos found on server");

    showStatus("Decoding audio from video...");
    const videoResp = await fetch(`videos/${videos[0]}`);
    if (!videoResp.ok) throw new Error(`Failed to fetch video: ${videoResp.status}`);
    const arrayBuf = await videoResp.arrayBuffer();
    const { interleaved, channels, frames } = await decodeAudioBuffer(arrayBuf);

    await startSession(interleaved, channels, frames, videos);
  } catch (e) {
    showStatus(`Error: ${e.message}`);
  }
});

seedBtn.addEventListener("click", () => randomizeSeed());

pauseBtn.addEventListener("click", () => {
  if (session) session.togglePause();
});

// ── MIDI (Web MIDI API, default mapping for Akai MPK Mini) ──
// MPK Mini default preset: knobs 1-8 send CC 70-77; pads 1-2 send notes 36-37.
// The mapping is mutable: right-click a control to enter MIDI learn mode and
// the next CC/Note overwrites the binding. Persisted in localStorage so the
// last mapping survives a reload. Reset by clearing the entry or via the
// midi-status badge (shift-click toggles diag, includes a reset hint).

// CC 71/72 use the magic strings "lora_slot_0" / "lora_slot_1"; the
// handler resolves them at message time to the first / second currently
// enabled LoRA so toggling the catalog re-aims the knobs without a
// remap.  All other ccs are static slider names.
const LORA_SLOT_MARKER = ["lora_slot_0", "lora_slot_1"];

const DEFAULT_MIDI_MAP = {
  cc: {
    70: "denoise",
    71: LORA_SLOT_MARKER[0],
    72: LORA_SLOT_MARKER[1],
    73: "hint_strength",
    74: "feedback",
    75: "shift",
    76: "noise_share",
    77: "ode_noise",
  },
  notes: {
    36: "seed",
    37: "send_prompt",
  },
};

function resolveCcParam(name) {
  if (name === LORA_SLOT_MARKER[0] || name === LORA_SLOT_MARKER[1]) {
    const ids = enabledLoraIds();
    const idx = name === LORA_SLOT_MARKER[0] ? 0 : 1;
    return ids[idx] ? loraStrengthKey(ids[idx]) : null;
  }
  return name;
}

// Note actions are looked up by name (the value in midiMap.notes) so
// localStorage can serialize the binding without holding function refs.
const NOTE_ACTIONS = {
  seed:        () => randomizeSeed(),
  send_prompt: () => sendCurrentPrompt(),
  mode_toggle: () => modeToggle?.click(),
  pause:       () => pauseBtn?.click(),
};

const MIDI_STORAGE_KEY = "demon_midi_map_v1";

function loadMidiMap() {
  try {
    const stored = JSON.parse(localStorage.getItem(MIDI_STORAGE_KEY) || "null");
    if (stored && stored.cc && stored.notes) {
      return { cc: { ...stored.cc }, notes: { ...stored.notes } };
    }
  } catch (e) {
    console.warn("Failed to read MIDI map from localStorage:", e);
  }
  return { cc: { ...DEFAULT_MIDI_MAP.cc }, notes: { ...DEFAULT_MIDI_MAP.notes } };
}

function saveMidiMap() {
  try {
    localStorage.setItem(MIDI_STORAGE_KEY, JSON.stringify(midiMap));
  } catch (e) {
    console.warn("Failed to persist MIDI map:", e);
  }
}

let midiMap = loadMidiMap();

// ── Knob mode auto-detect (absolute vs relative 2's complement) ──
// Some MIDI controllers (MPK Mini in increment/decrement mode, many endless
// encoders) send relative deltas instead of absolute positions: 1 = +1 CW,
// 127 = -1 CCW (2's complement); 65..127 = -(128 - value). Treating those as
// absolute (value/127) maps every CW tick to ~0.0 and every CCW tick to 1.0,
// so the slider snaps between extremes — that's the bug being fixed here.
//
// Strategy: classify per-CC. The first 1-2 messages may apply optimistically
// as relative if the value is at an extreme (which is the bug pattern); if a
// mid-range value arrives we lock to absolute. Once we have ≥3 samples we
// commit to whichever mode fits (all-extreme → relative; otherwise absolute).
const knobState = new Map();

function decodeKnob(cc, value) {
  let s = knobState.get(cc);
  if (!s) {
    s = { history: [], mode: "unknown" };
    knobState.set(cc, s);
  }
  s.history.push(value);
  if (s.history.length > 6) s.history.shift();

  if (s.mode === "unknown" && s.history.length >= 3) {
    const allExtreme = s.history.every((v) => v <= 4 || v >= 123);
    s.mode = allExtreme ? "relative" : "absolute";
  }

  if (s.mode === "absolute") {
    return { mode: "absolute", absolute: value / 127 };
  }
  if (s.mode === "relative") {
    let delta = 0;
    if (value > 0 && value < 64) delta = value;
    else if (value > 64 && value <= 127) delta = value - 128;
    return { mode: "relative", delta };
  }
  // Unknown: an extreme reading is almost certainly a relative tick, so apply
  // it that way; a mid-range reading commits to absolute right now so the
  // first sweep doesn't lose its starting samples.
  if (value <= 4 || value >= 123) {
    let delta = 0;
    if (value > 0 && value < 64) delta = value;
    else if (value > 64 && value <= 127) delta = value - 128;
    return { mode: "relative", delta };
  }
  s.mode = "absolute";
  return { mode: "absolute", absolute: value / 127 };
}

function handleMidiCC(cc, value) {
  const rawName = midiMap.cc[cc];
  if (!rawName) return;
  const param = resolveCcParam(rawName);
  if (!param) return;  // dynamic slot with no LoRA enabled
  const def = SLIDER_DEFS[param];
  if (!def) return;
  const decoded = decodeKnob(cc, value);
  if (decoded.mode === "relative") {
    if (!decoded.delta) return;
    const next = sliderValues[param] + decoded.delta * def.step;
    sliderValues[param] = Math.max(0, Math.min(def.max, next));
  } else {
    sliderValues[param] = decoded.absolute * def.max;
  }
  updateSliderUI(param);
}

function handleMidiNote(note) {
  const action = midiMap.notes[note];
  if (!action) return;
  NOTE_ACTIONS[action]?.();
}

// ── MIDI learn ──
// Right-click a slider strip or any element with [data-midi-learn] to bind
// the next incoming CC/Note to that target. The previous binding for that
// target is cleared so each target maps to exactly one input. Esc cancels.
let midiLearn = null;

function startMidiLearn(kind, target, el) {
  cancelMidiLearn();
  midiLearn = { kind, target, el };
  if (el) el.classList.add("midi-learning");
  setMidiStatus(`Learn: ${target} — twist or press`, "midi-warn");
}

function cancelMidiLearn() {
  if (midiLearn?.el) midiLearn.el.classList.remove("midi-learning");
  midiLearn = null;
}

function applyMidiLearn(kind, num) {
  if (!midiLearn || midiLearn.kind !== kind) return false;
  const map = kind === "cc" ? midiMap.cc : midiMap.notes;
  for (const k of Object.keys(map)) {
    if (map[k] === midiLearn.target) delete map[k];
  }
  // Per-CC knob mode (relative vs absolute) is a property of the hardware
  // encoder, not the binding, so we keep whatever classification was learned
  // for `num` across remaps — no need to reset knobState here.
  map[String(num)] = midiLearn.target;
  saveMidiMap();
  setMidiStatus(`Learned ${midiLearn.target} ← ${kind} ${num}`, "midi-ok");
  cancelMidiLearn();
  return true;
}

function initMidiLearnUI() {
  document.addEventListener("contextmenu", (e) => {
    const sliderGroup = e.target.closest?.(".slider-group");
    if (sliderGroup?.dataset?.param) {
      e.preventDefault();
      startMidiLearn("cc", sliderGroup.dataset.param, sliderGroup);
      return;
    }
    // LoRA library rows are slider-group-equivalent for MIDI learn:
    // right-click anywhere on the row (switch, name, slider track) to
    // bind the next CC to that LoRA's strength.  data-param on the row
    // is set by buildLoraRow to lora_str_<id>.
    const loraRow = e.target.closest?.(".lora-row");
    if (loraRow?.dataset?.param) {
      e.preventDefault();
      startMidiLearn("cc", loraRow.dataset.param, loraRow);
      return;
    }
    const learnEl = e.target.closest?.("[data-midi-learn]");
    if (learnEl) {
      e.preventDefault();
      startMidiLearn("note", learnEl.dataset.midiLearn, learnEl);
    }
  });
  document.addEventListener("keydown", (e) => {
    if (e.key !== "Escape" || !midiLearn) return;
    cancelMidiLearn();
    // Swallow the Escape so the drawer's own Esc-to-close handler doesn't
    // also fire — cancelling learn shouldn't close the panel underneath it.
    e.stopPropagation();
  }, true);
}

// ── MIDI status badge + diagnostic log ──
// Both elements live inside the Advanced sheet now. They used to hang off
// document.body pinned to the bottom-left of the viewport; that was visible
// during the show. Operator opens the sheet to see MIDI state.
const midiSlot = document.getElementById("install-midi-slot") || document.body;

const midiStatus = document.createElement("div");
midiStatus.id = "midi-status";
midiStatus.textContent = "MIDI: --";
midiSlot.appendChild(midiStatus);

const midiDiag = document.createElement("pre");
midiDiag.id = "midi-diag";
midiDiag.className = "hidden";
midiSlot.appendChild(midiDiag);

function setMidiStatus(text, state) {
  midiStatus.textContent = text;
  midiStatus.classList.remove("midi-ok", "midi-warn", "midi-error");
  if (state) midiStatus.classList.add(state);
}

function midiLog(msg) {
  const ts = new Date().toLocaleTimeString();
  midiDiag.textContent += `[${ts}] ${msg}\n`;
  midiDiag.scrollTop = midiDiag.scrollHeight;
  console.log("[MIDI]", msg);
}

function bindAccess(access) {
  let inputCount = 0;

  const bind = (input) => {
    input.onmidimessage = (e) => {
      if (!e.data || e.data.length < 3) return;
      const [status, data1, data2] = e.data;
      const cmd = status & 0xf0;
      if (cmd === 0xb0) {
        if (applyMidiLearn("cc", data1)) return;
        setMidiStatus(`CC ${data1}: ${data2}`, "midi-ok");
        handleMidiCC(data1, data2);
        bumpIdleTimer();
      }
      if (cmd === 0x90 && data2 > 0) {
        if (applyMidiLearn("note", data1)) return;
        setMidiStatus(`Note ${data1} vel ${data2}`, "midi-ok");
        handleMidiNote(data1);
        bumpIdleTimer();
      }
    };
    inputCount++;
  };

  for (const input of access.inputs.values()) {
    midiLog(`  input: "${input.name}" state=${input.state} id=${input.id}`);
    bind(input);
  }
  for (const output of access.outputs.values()) {
    midiLog(`  output: "${output.name}" state=${output.state}`);
  }

  access.onstatechange = (e) => {
    midiLog(`statechange: ${e.port.type} "${e.port.name}" -> ${e.port.state}`);
    if (e.port.type === "input") {
      if (e.port.state === "connected") {
        bind(e.port);
        setMidiStatus(`MIDI: ${e.port.name}`, "midi-ok");
      } else {
        setMidiStatus("MIDI: disconnected", "midi-warn");
      }
    }
  };

  if (inputCount > 0) {
    const firstName = access.inputs.values().next().value.name;
    setMidiStatus(`MIDI: ${firstName}`, "midi-ok");
  } else {
    setMidiStatus("MIDI: 0 devices [click]", "midi-warn");
  }

  return inputCount;
}

function initMidi(opts) {
  const label = opts ? `sysex:${opts.sysex}` : "no-opts";
  if (!navigator.requestMIDIAccess) {
    setMidiStatus("MIDI: not supported", "midi-error");
    midiLog("navigator.requestMIDIAccess is undefined");
    return;
  }

  midiLog(`--- initMidi(${label}) ---`);
  midiLog(`secure: ${window.isSecureContext}, proto: ${location.protocol}, origin: ${location.origin}`);
  setMidiStatus("MIDI: requesting...");

  const p = opts ? navigator.requestMIDIAccess(opts) : navigator.requestMIDIAccess();
  p.then((access) => {
    midiLog(`resolved. inputs.size=${access.inputs.size} outputs.size=${access.outputs.size} sysexEnabled=${access.sysexEnabled}`);
    const count = bindAccess(access);
    if (count === 0) {
      midiLog("0 inputs. Will try sysex:true on next click.");
    }
  }).catch((err) => {
    midiLog(`REJECTED: ${err.name}: ${err.message}`);
    setMidiStatus("MIDI: denied [click]", "midi-error");
  });
}

let _clickCount = 0;
midiStatus.addEventListener("click", (e) => {
  _clickCount++;
  // Toggle diagnostic panel on shift-click or triple-click.
  if (e.shiftKey || _clickCount >= 3) {
    _clickCount = 0;
    midiDiag.classList.toggle("hidden");
    return;
  }
  // First click: retry without sysex. Second click: try WITH sysex (triggers permission prompt).
  if (_clickCount % 2 === 1) {
    setMidiStatus("MIDI: retrying...");
    initMidi();
  } else {
    setMidiStatus("MIDI: retrying (sysex)...");
    initMidi({ sysex: true });
  }
});

// Init UI -- each in try/catch so one failure can't kill the rest.
try { applyConfigToDom(); } catch (e) { console.error("applyConfigToDom failed:", e); }
try { initSliders(); } catch (e) { console.error("initSliders failed:", e); }
// Library tile must be populated AFTER initSliders() (which inserts the
// tile placeholder) but before initEdgeBars() (which queries the
// perimeter slot bindings).  Demo policy: pre-toggle the first
// DEFAULT_AUTO_ENABLE_COUNT rows on so a fresh page-load looks staged
// and ready; the operator can flip them off before pressing Play.
try {
  const initial = (loraInitial.loras || []).map((d, i) => ({
    ...d,
    state: i < DEFAULT_AUTO_ENABLE_COUNT ? "enabled" : "registered",
  }));
  applyLoraCatalog(initial);
} catch (e) { console.error("applyLoraCatalog(initial) failed:", e); }
try { initEdgeBars(); } catch (e) { console.error("initEdgeBars failed:", e); }
try { initPrompts(); } catch (e) { console.error("initPrompts failed:", e); }
try { initKeyboard(); } catch (e) { console.error("initKeyboard failed:", e); }
try { initIdleReset(); } catch (e) { console.error("initIdleReset failed:", e); }
try { initCursorAutoHide(); } catch (e) { console.error("initCursorAutoHide failed:", e); }
try { initMidiLearnUI(); } catch (e) { console.error("initMidiLearnUI failed:", e); }
initMidi();
