// Headless wire smoke for the /sfx sound-design board (demos/sfx).
//
// Drives the page's own wire module (demos/sfx/layer-core.js) through the
// Node build of the client SDK, so the configs, prompt/blend commands and
// playhead reports are byte-for-byte what the page sends. Audio is kept
// in a per-layer mirror exactly as AudioPlayer keeps it (patch = write,
// delta = add), and a simulated listener copies out what the playhead
// plays, so the WAVs written here are what a browser would have heard.
//
// Scenarios (all against one running server):
//   four:    four layers live at once (2 ambience, 2 one-shot) for --secs,
//            with slice landing lead per layer (negative = late = audible
//            glitch), tick ms and slice rate.
//   blend:   layer 1 rain -> storm, blend swept 0 -> 1 like a slider drag.
//   trigger: one-shot layer re-triggered, plus a seed re-roll.
//   drift:   60 s, three ambience layers at once, the mirror (the whole
//            20 s canvas) snapshotted every 5 s: a text layer at denoise
//            1.0 (v3), the v2 "evolve" setting (sa3_denoise 0.3 +
//            feedback 1, kept only to measure the feedback loop), and a
//            layer with the click + hum input at denoise 0.3; that layer
//            then goes to denoise 0.9 for 30 s more. Snapshots are scored
//            by scripts/sa3/sfx_demo_v3_signal.py.
//   card:    (with drift) an example-card click, one prompt slot set and
//            nothing else, must reach the server as a prompt swap
//            (prompt_applied), for slot A and slot B.
//   oneshot_input: a 2.5 s input on a one-shot layer: the canvas must
//            follow the input length and the tail decay to silence.
//
// Run (repo root, server already up):
//   node scripts/sa3/sfx_demo_wire_smoke.mjs --ws ws://127.0.0.1:1318/
//        --out E:/Projects/sa3-variants/sfx/demo_v3 --only drift,card,oneshot_input

import { createRequire } from "node:module";
import { mkdirSync, writeFileSync, readFileSync } from "node:fs";
import { join } from "node:path";
import {
  CHANNELS, PARAMS_TICK_MS, SAMPLE_RATE, LayerWire, VirtualPlayhead, encodeWav, fitInput, mixLayers,
} from "../../demos/sfx/layer-core.js";

const require = createRequire(import.meta.url);
const { RemoteBackend, SLICE_FLAG_DELTA } = require("../../packages/demon-client/dist/demon-client.node.cjs");

const args = Object.fromEntries(
  process.argv.slice(2).reduce((acc, a, i, all) => (a.startsWith("--") ? [...acc, [a.slice(2), all[i + 1]]] : acc), []),
);
const WS = args.ws ?? "ws://127.0.0.1:1318/";
const OUT = args.out ?? "out/sfx_demo_v3";
const SECS = Number(args.secs ?? 20);
const ONLY = args.only ? args.only.split(",") : null;
mkdirSync(OUT, { recursive: true });

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const now = () => performance.now() / 1000;

// One headless layer: the page's Layer minus DOM and speakers.
class HeadlessLayer {
  constructor(name, opts) {
    this.name = name;
    this.opts = opts;
    this.mode = opts.mode;
    this.slices = 0;
    this.leads = [];
    this.ticks = [];
    this.heard = [];
    this.errors = [];
    this.wire = new LayerWire(RemoteBackend, {
      wsUrl: WS,
      onSlice: (d) => this.applySlice(d),
      onClose: () => this.errors.push("connection lost"),
    });
  }

  async open() {
    const t0 = now();
    await this.wire.open(this.opts);
    this.readyS = now() - t0;
    const remote = this.wire.remote;
    this.channels = remote.channels;
    this.mirror = remote.initialBuffer.slice();
    this.frames = this.mirror.length / this.channels;
    this.duration = this.frames / SAMPLE_RATE;
    this.playhead = new VirtualPlayhead(this.duration);
    this.lastPos = 0;
    // One-shot: parked until trigger (the page seeks to the end).
    this.playing = this.mode !== "oneshot";
    this.timer = setInterval(() => this.tick(), PARAMS_TICK_MS);
  }

  // The playhead the page reports: the player's own (a looping 1x clock)
  // for ambience, the free-running virtual one for a one-shot.
  position() {
    return this.playhead.position();
  }

  tick() {
    const pos = this.position();
    this.wire.report(pos);
    this.listen(pos);
  }

  // Copy what the playhead played since the last tick into `heard`.
  listen(pos) {
    if (!this.playing) { this.lastPos = pos; return; }
    const a = Math.floor(this.lastPos * SAMPLE_RATE);
    const b = Math.floor(pos * SAMPLE_RATE);
    const ch = this.channels;
    const take = (from, to) => this.heard.push(this.mirror.slice(from * ch, to * ch));
    if (b >= a) take(a, b);
    else { take(a, this.frames); take(0, b); }
    this.lastPos = pos;
    if (this.mode === "oneshot" && this.heardFrames() >= this.frames) this.playing = false;
  }

  heardFrames() {
    return this.heard.reduce((n, x) => n + x.length / this.channels, 0);
  }

  applySlice(d) {
    if (d.epoch !== 0) return;
    this.slices += 1;
    if (typeof d.tickMs === "number" && Number.isFinite(d.tickMs)) this.ticks.push(d.tickMs);
    const start = Math.floor(d.startSample);
    const ch = this.channels;
    const base = start * ch;
    const n = Math.min(d.audio.length, this.mirror.length - base);
    if (d.flags === SLICE_FLAG_DELTA) for (let i = 0; i < n; i++) this.mirror[base + i] += d.audio[i];
    else for (let i = 0; i < n; i++) this.mirror[base + i] = d.audio[i];
    // Landing lead: how far ahead of the reported playhead the slice
    // starts, folded into (-dur/2, dur/2]. Negative = it landed behind.
    let lead = start / SAMPLE_RATE - this.position();
    while (lead > this.duration / 2) lead -= this.duration;
    while (lead <= -this.duration / 2) lead += this.duration;
    this.leads.push(lead);
  }

  trigger() {
    this.playhead.restart();
    this.heard = [];
    this.lastPos = 0;
    this.playing = true;
    this.triggeredAt = now();
  }

  takeHeard() {
    const total = this.heard.reduce((n, x) => n + x.length, 0);
    const out = new Float32Array(total);
    let o = 0;
    for (const x of this.heard) { out.set(x, o); o += x.length; }
    this.heard = [];
    return out;
  }

  stats() {
    const sorted = (xs) => [...xs].sort((a, b) => a - b);
    const pct = (xs, p) => (xs.length ? sorted(xs)[Math.min(xs.length - 1, Math.floor(p * xs.length))] : null);
    return {
      name: this.name, mode: this.mode, duration_s: this.duration, ready_s: +this.readyS.toFixed(2),
      slices: this.slices, tick_ms_p50: pct(this.ticks, 0.5), tick_ms_p95: pct(this.ticks, 0.95),
      lead_s_min: this.leads.length ? Math.min(...this.leads) : null, lead_s_p50: pct(this.leads, 0.5),
      late_slices: this.leads.filter((l) => l < 0).length, errors: this.errors,
    };
  }

  saveMirror(file) { writeFileSync(join(OUT, file), Buffer.from(encodeWav(this.mirror, this.channels, SAMPLE_RATE))); }

  close() { clearInterval(this.timer); this.wire.close(); }
}

const save = (file, pcm) => writeFileSync(join(OUT, file), Buffer.from(encodeWav(pcm, CHANNELS, SAMPLE_RATE)));
const report = {};
const want = (s) => !ONLY || ONLY.includes(s);

// --- four layers at once ---------------------------------------------------
const layers = [
  new HeadlessLayer("L1 rain/storm", { mode: "ambience", promptA: "Steady heavy rain on a tin roof",
    promptB: "Thunderstorm with rolling thunder and heavy rain", seed: 11 }),
  new HeadlessLayer("L2 crowd", { mode: "ambience", promptA: "Crowd murmur in a large indoor hall", seed: 12 }),
  new HeadlessLayer("L3 door", { mode: "oneshot", promptA: "Heavy metal door slam, single impact, reverberant warehouse", seed: 13 }),
  new HeadlessLayer("L4 laser", { mode: "oneshot", promptA: "Futuristic laser blast, sharp energy pulse, arcade style", seed: 14 }),
];
// The page's Play starts every layer at once.
if (["four", "blend", "trigger"].some(want)) await Promise.all(layers.map((l) => l.open()));
console.log("ready:", layers.filter((l) => l.readyS != null).map((l) => `${l.name} ${l.readyS.toFixed(2)}s`).join(", "));

if (want("four")) {
  await sleep(SECS * 1000);
  report.four = { secs: SECS, layers: layers.map((l) => l.stats()) };
  layers.forEach((l, i) => l.saveMirror(`four_L${i + 1}_mirror.wav`));
  save("four_L1_heard.wav", layers[0].takeHeard());
  save("four_L2_heard.wav", layers[1].takeHeard());
  save("four_mix.wav", mixLayers(layers.map((l) => ({
    mirror: l.mirror, channels: l.channels, gain: 0.8, loop: l.mode === "ambience",
  })), 20));
  console.log("four:", JSON.stringify(report.four.layers.map((s) => [s.name, s.slices, s.tick_ms_p50, s.lead_s_min, s.late_slices])));
}

// --- blend sweep on layer 1 ------------------------------------------------
if (want("blend")) {
  const L1 = layers[0];
  L1.takeHeard();
  const hold = 6;
  await sleep(hold * 1000); // blend 0 (rain)
  const sweep = 10;
  const steps = Math.round((sweep * 1000) / PARAMS_TICK_MS);
  for (let i = 1; i <= steps; i++) { L1.wire.setBlend(i / steps); await sleep(PARAMS_TICK_MS); }
  await sleep(hold * 1000); // blend 1 (storm)
  save("blend_L1_heard.wav", L1.takeHeard());
  report.blend = { hold_s: hold, sweep_s: sweep, note: "heard = 6 s at 0, 10 s linear 0->1, 6 s at 1" };
  console.log("blend: done");
}

// --- one-shot trigger + seed re-roll ---------------------------------------
if (want("trigger")) {
  const L3 = layers[2];
  const trig = [];
  for (const label of ["first", "again"]) {
    L3.trigger();
    await sleep((L3.duration + 0.3) * 1000);
    const pcm = L3.takeHeard();
    save(`trigger_L3_${label}.wav`, pcm);
    trig.push(label);
  }
  const before = L3.mirror.slice();
  const t0 = now();
  L3.wire.setSeed(4242);
  // Time until most of the clip differs from the old seed's render.
  let changedS = null;
  while (now() - t0 < 8) {
    await sleep(20);
    let diff = 0;
    let energy = 0;
    for (let i = 0; i < before.length; i += 16) {
      diff += Math.abs(L3.mirror[i] - before[i]);
      energy += Math.abs(before[i]);
    }
    if (energy > 0 && diff / energy > 0.5) { changedS = now() - t0; break; }
  }
  L3.trigger();
  await sleep((L3.duration + 0.3) * 1000);
  save("trigger_L3_reroll.wav", L3.takeHeard());
  report.trigger = { triggers: trig, reroll_changed_after_s: changedS };
  console.log("trigger: reroll changed after", changedS);
}

const opened = layers.filter((l) => l.readyS != null);
opened.forEach((l) => l.close());
if (opened.length) report.streams = opened.map((l) => l.stats());

// --- input audio -------------------------------------------------------------
// The v2 click + hum WAV (2.5 s: 110 Hz hum with harmonics at -18 dBFS, a
// sharp click every 0.5 s), read from disk so the file is what was sent.
function loadInput(file) {
  const wav = readFileSync(file);
  const pcm16 = new Int16Array(wav.buffer.slice(wav.byteOffset + 44, wav.byteOffset + wav.length));
  const interleaved = Float32Array.from(pcm16, (v) => v / 32767);
  return { interleaved, channels: 2, seconds: interleaved.length / 2 / SAMPLE_RATE };
}
const INPUT_WAV = args.input ?? "E:/Projects/sa3-variants/sfx/demo_v2/a2a_source_click_hum.wav";

// Mirror snapshots every `every` s for `secs` s, written as WAVs.
async function snapshots(list, tag, secs, every = 5) {
  for (let t = every; t <= secs; t += every) {
    await sleep(every * 1000);
    for (const [key, l] of list) l.saveMirror(`${tag}_${key}_t${String(t).padStart(3, "0")}.wav`);
  }
}

if (want("drift")) {
  const input = loadInput(INPUT_WAV);
  save("input_ambience_fitted.wav", fitInput(input, "ambience").pcm);
  const prompt = "Steady heavy rain on a tin roof";
  const d10 = new HeadlessLayer("text denoise 1.0", { mode: "ambience", promptA: prompt, seed: 31 });
  const evo = new HeadlessLayer("v2 evolve 0.3 (feedback 1)", { mode: "ambience", promptA: prompt, seed: 31 });
  const inp = new HeadlessLayer("input denoise 0.3", {
    mode: "ambience", promptA: "Crowd murmur in a large indoor hall", seed: 32, input, denoise: 0.3,
  });
  await Promise.all([d10.open(), evo.open(), inp.open()]);
  // The v2 evolve knob, reproduced verbatim for the diagnosis only.
  evo.wire.values = { ...evo.wire.values, sa3_denoise: 0.3, feedback: 1 };
  const secs = Number(args.driftSecs ?? 60);
  const named = [["d10", d10], ["evo03", evo], ["in03", inp]];
  for (const [key, l] of named) l.saveMirror(`drift_${key}_t000.wav`);
  await snapshots(named, "drift", secs);
  inp.wire.setDenoise(0.9);
  await snapshots([["in09", inp]], "drift", Number(args.highSecs ?? 30));
  report.drift = { secs, layers: [d10, evo, inp].map((l) => ({ ...l.stats(), values: l.wire.values })) };
  // Card click: one slot set, nothing else, on a live layer.
  if (want("card")) {
    const cards = [];
    for (const [slot, text] of [["a", "Wind howling through a narrow canyon"], ["b", "Ocean waves crashing on rocks"]]) {
      const t0 = now();
      const remote = d10.wire.remote;
      const applied = new Promise((resolve) => {
        const on = (ev) => { remote.removeEventListener("prompt_applied", on); resolve(ev.detail ?? true); };
        remote.addEventListener("prompt_applied", on);
        setTimeout(() => resolve(null), 5000);
      });
      const sent = d10.wire.setPromptSlot(slot, text);
      const tags = await applied;
      cards.push({ slot, text, sent, prompt_applied: tags, ack_ms: tags == null ? null : Math.round((now() - t0) * 1000) });
    }
    report.card = cards;
    console.log("card:", JSON.stringify(cards));
  }
  [d10, evo, inp].forEach((l) => l.close());
  console.log("drift: done");
}

if (want("oneshot_input")) {
  const input = loadInput(INPUT_WAV);
  const L = new HeadlessLayer("one-shot 2.5 s input", {
    mode: "oneshot", promptA: "Heavy metal door slam, single impact, reverberant warehouse", seed: 41, input, denoise: 0.5,
  });
  await L.open();
  await sleep(8000);
  L.trigger();
  await sleep((L.duration + 0.3) * 1000);
  save("oneshot_input_heard.wav", L.takeHeard());
  L.saveMirror("oneshot_input_mirror.wav");
  report.oneshot_input = { input_s: input.seconds, canvas_s: L.duration, fit: L.wire.fit.note, ...L.stats() };
  console.log("oneshot_input: canvas", L.duration);
  L.close();
}

writeFileSync(join(OUT, args.report ?? "wire_smoke.json"), JSON.stringify(report, null, 2));
console.log("wrote", join(OUT, "wire_smoke.json"));
await sleep(500);
process.exit(0);
