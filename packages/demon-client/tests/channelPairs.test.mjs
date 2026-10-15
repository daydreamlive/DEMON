// Channel-pair mix: the worklet's pair sum and level reports, the
// stereo path left as it was, and the main-thread helpers the
// ScriptProcessor fallback and loudness matcher use.
//
// Run: npm test (node --test; needs Node 22.6+ for .ts type stripping).

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";
import vm from "node:vm";

import {
  mixChannelPairs,
  sanitizeChannelGains,
  smoothPairGains,
} from "../audio/channelPairs.ts";

const SR = 48000;
const BLOCK = 128;

// Load assets/audio-worklet.js into a sandbox with the AudioWorklet
// globals stubbed; return the registered processor class.
function loadProcessor() {
  const src = readFileSync(
    new URL("../assets/audio-worklet.js", import.meta.url),
    "utf8",
  );
  let registered = null;
  class AudioWorkletProcessor {
    constructor() {
      this.port = { posted: [], postMessage(m) { this.posted.push(m); } };
    }
  }
  const ctx = vm.createContext({
    AudioWorkletProcessor,
    sampleRate: SR,
    registerProcessor: (_name, cls) => { registered = cls; },
    Float32Array, Float64Array, Math, Number, Array, ArrayBuffer, Object,
  });
  vm.runInContext(src, ctx);
  return registered;
}

function makeProcessor(buffer, channels) {
  const Proc = loadProcessor();
  const p = new Proc();
  p._onmessage({ type: "init", buffer, channels });
  return p;
}

function runBlock(p, outChannels = 2) {
  const out = [];
  for (let c = 0; c < outChannels; c++) out.push(new Float32Array(BLOCK));
  p.process([], [out]);
  return out;
}

// Deterministic interleaved buffer, distinct value per (frame, channel).
function makeBuffer(frames, channels) {
  const b = new Float32Array(frames * channels);
  for (let f = 0; f < frames; f++) {
    for (let c = 0; c < channels; c++) {
      b[f * channels + c] = Math.sin(0.001 * (f + 1) * (c + 1)) * 0.1 * (c + 1);
    }
  }
  return b;
}

const FRAMES = SR; // 1 s: far from the loop seam for the first blocks

test("stereo buffer plays the samples unchanged and posts no pairLevels", () => {
  const buf = makeBuffer(FRAMES, 2);
  const p = makeProcessor(buf.slice(), 2);
  p._onmessage({ type: "setChannelGains", gains: [0.25] });
  const out = runBlock(p);
  for (let i = 0; i < BLOCK; i++) {
    assert.equal(out[0][i], buf[i * 2]);
    assert.equal(out[1][i], buf[i * 2 + 1]);
  }
  runBlock(p);
  const pos = p.port.posted.filter((m) => m.type === "position");
  assert.ok(pos.length > 0);
  for (const m of pos) assert.equal(m.pairLevels, undefined);
});

test("10-channel buffer at unity gains plays the sum of the five pairs", () => {
  const ch = 10;
  const buf = makeBuffer(FRAMES, ch);
  const p = makeProcessor(buf.slice(), ch);
  const out = runBlock(p);
  for (let i = 0; i < BLOCK; i++) {
    for (let c = 0; c < 2; c++) {
      let want = 0;
      for (let k = 0; k < 5; k++) want += buf[i * ch + 2 * k + c];
      assert.ok(Math.abs(out[c][i] - want) < 1e-6, `frame ${i} ch ${c}`);
    }
  }
});

test("pair gains converge (smoothed) and missing entries stay at unity", () => {
  const ch = 10;
  const buf = makeBuffer(FRAMES, ch);
  const p = makeProcessor(buf.slice(), ch);
  const gains = [0, 0.5, 2]; // pairs 3 and 4 missing -> 1
  p._onmessage({ type: "setChannelGains", gains });
  // First frame moves only one IIR step away from unity (no jump).
  const first = runBlock(p);
  let unity0 = 0;
  for (let k = 0; k < 5; k++) unity0 += buf[2 * k];
  assert.ok(Math.abs(first[0][0] - unity0) < Math.abs(unity0) * 0.01 + 1e-6);
  // ~0.5 s of blocks: well past the 25 ms time constant.
  let out = first;
  const blocks = 190;
  for (let b = 1; b < blocks; b++) out = runBlock(p);
  const full = [0, 0.5, 2, 1, 1];
  const base = (blocks - 1) * BLOCK;
  for (let i = 0; i < BLOCK; i++) {
    for (let c = 0; c < 2; c++) {
      let want = 0;
      for (let k = 0; k < 5; k++) want += full[k] * buf[(base + i) * ch + 2 * k + c];
      assert.ok(Math.abs(out[c][i] - want) < 1e-4, `frame ${i} ch ${c}`);
    }
  }
});

test("pairLevels report pre-gain RMS per pair", () => {
  const ch = 6;
  const frames = FRAMES;
  const buf = new Float32Array(frames * ch);
  const amps = [0.5, 0.1, 0]; // constant DC per pair -> RMS = amplitude
  for (let f = 0; f < frames; f++) {
    for (let k = 0; k < 3; k++) {
      buf[f * ch + 2 * k] = amps[k];
      buf[f * ch + 2 * k + 1] = amps[k];
    }
  }
  const p = makeProcessor(buf, ch);
  p._onmessage({ type: "setChannelGains", gains: [0, 0, 0] });
  runBlock(p);
  runBlock(p);
  const msg = p.port.posted.find((m) => m.type === "position");
  assert.ok(msg && msg.pairLevels, "position carries pairLevels");
  assert.equal(msg.pairLevels.length, 3);
  for (let k = 0; k < 3; k++) {
    assert.ok(Math.abs(msg.pairLevels[k] - amps[k]) < 1e-6, `pair ${k}`);
  }
});

test("mixChannelPairs sums pairs with gains, missing = 1", () => {
  const ch = 6;
  const frames = 4;
  const buf = makeBuffer(frames, ch);
  const out = mixChannelPairs(buf, ch, [2, 0]);
  assert.equal(out.length, frames * 2);
  for (let f = 0; f < frames; f++) {
    for (let c = 0; c < 2; c++) {
      const want = 2 * buf[f * ch + c] + 0 * buf[f * ch + 2 + c] + buf[f * ch + 4 + c];
      assert.ok(Math.abs(out[f * 2 + c] - want) < 1e-7);
    }
  }
});

test("sanitizeChannelGains and smoothPairGains", () => {
  assert.deepEqual(sanitizeChannelGains([1, -1, NaN, 0.5]), [1, 0, 1, 0.5]);
  const live = new Float32Array([1, 1, 1]);
  smoothPairGains(live, [0, 2], 0.5); // third target missing -> 1
  assert.deepEqual(Array.from(live), [0.5, 1.5, 1]);
  for (let i = 0; i < 60; i++) smoothPairGains(live, [0, 2], 0.5);
  assert.ok(Math.abs(live[0]) < 1e-6 && Math.abs(live[1] - 2) < 1e-6);
});
