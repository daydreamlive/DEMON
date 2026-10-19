// One-shot renders: the wire frame, the int16 block layout of the answer, and
// the promise round trip against a stand-in RemoteBackend.
//
// Run: npm test (node --test; needs Node 22.6+ for .ts type stripping).

import assert from "node:assert/strict";
import { test } from "node:test";

import {
  decodeOneshot,
  int16ToFloat32,
  oneshotCommand,
  renderOneshot,
} from "../oneshot.ts";

function meta(over = {}) {
  return {
    type: "oneshot_ready",
    request_id: "r1",
    ids: ["a", "b"],
    seeds: [5, 6],
    streams: [0, 1, 2],
    sample_rate: 48000,
    channels: 2,
    frames: 4,
    encoding: "pcm_s16le",
    ms: { generate: 1.5, decode: 2, pack: 0.5, total: 4 },
    redo_of: null,
    ...over,
  };
}

/** Block (v, s) filled with v*100 + s*10 + i, i = sample index in the block. */
function payload(m) {
  const block = m.frames * m.channels;
  const a = new Int16Array(m.ids.length * m.streams.length * block);
  for (let v = 0; v < m.ids.length; v++)
    for (let s = 0; s < m.streams.length; s++)
      for (let i = 0; i < block; i++) a[(v * m.streams.length + s) * block + i] = v * 100 + s * 10 + i;
  return a.buffer;
}

test("oneshotCommand: seeds and knobs, or a redo", () => {
  const c = oneshotCommand({ requestId: "x", seeds: [1, 2], knobs: { seed: 3 } });
  assert.deepEqual(c, { type: "oneshot", request_id: "x", seeds: [1, 2], knobs: { seed: 3 } });
  const r = oneshotCommand({ requestId: "y", redo: { id: "k", stream: 2, seed: 9 } });
  assert.deepEqual(r, { type: "oneshot", request_id: "y", redo: { id: "k", stream: 2, seed: 9 } });
  assert.match(oneshotCommand({ seeds: [0] }).request_id, /^oneshot-/);
});

test("decodeOneshot: version-major, then stream, each block interleaved", () => {
  const m = meta();
  const res = decodeOneshot(m, payload(m));
  assert.equal(res.versions.length, 2);
  assert.deepEqual(res.streams, [0, 1, 2]);
  assert.equal(res.versions[1].id, "b");
  assert.equal(res.versions[1].seed, 6);
  assert.deepEqual(Array.from(res.versions[1].audio[2]), [120, 121, 122, 123, 124, 125, 126, 127]);
  assert.deepEqual(Array.from(res.versions[0].audio[0]).slice(0, 3), [0, 1, 2]);
  assert.equal(res.ms.generate, 1.5);
  assert.equal(res.redoOf, null);
});

test("decodeOneshot: a redo carries its one stream", () => {
  const m = meta({ ids: ["n"], seeds: [9], streams: [3], redo_of: "a" });
  const res = decodeOneshot(m, payload(m));
  assert.deepEqual(Object.keys(res.versions[0].audio), ["3"]);
  assert.equal(res.redoOf, "a");
});

test("decodeOneshot: a frame of the wrong size is an error", () => {
  assert.throws(() => decodeOneshot(meta(), new ArrayBuffer(10)), /expected/);
});

test("int16ToFloat32", () => {
  assert.deepEqual(Array.from(int16ToFloat32(new Int16Array([-32768, 0, 16384]))), [-1, 0, 0.5]);
});

class Stand extends EventTarget {
  constructor(answer) {
    super();
    this.answer = answer;
    this.sent = [];
  }
  sendOneshot(req) {
    this.sent.push(req);
    queueMicrotask(() => this.answer(this, req));
    return req.requestId;
  }
}

test("renderOneshot resolves with its own answer only", async () => {
  const m = meta();
  const remote = new Stand((r, req) => {
    r.dispatchEvent(new CustomEvent("oneshot", { detail: { ...decodeOneshot(m, payload(m)), requestId: "other" } }));
    r.dispatchEvent(new CustomEvent("oneshot", { detail: { ...decodeOneshot(m, payload(m)), requestId: req.requestId } }));
  });
  const res = await renderOneshot(remote, { requestId: "mine", seeds: [5, 6] });
  assert.equal(res.requestId, "mine");
  assert.equal(remote.sent[0].seeds.length, 2);
});

test("renderOneshot rejects on oneshot_failed and on close", async () => {
  const failing = new Stand((r, req) =>
    r.dispatchEvent(new CustomEvent("oneshot_failed", { detail: { request_id: req.requestId, error: "nope" } })));
  await assert.rejects(renderOneshot(failing, { seeds: [1] }), /nope/);
  const closing = new Stand((r) => r.dispatchEvent(new CustomEvent("close")));
  await assert.rejects(renderOneshot(closing, { seeds: [1] }), /connection lost/);
});
