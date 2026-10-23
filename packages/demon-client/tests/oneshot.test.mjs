// One-shot renders: the wire frame, the int16 block layout of the answer, and
// the promise round trip against a stand-in RemoteBackend.
//
// Run: npm test (node --test; needs Node 22.6+ for .ts type stripping).

import assert from "node:assert/strict";
import { test } from "node:test";

import {
  OneshotFrames,
  decodeOneshot,
  int16ToFloat32,
  oneshotCommand,
  oneshotLatentFrame,
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

// ---- slow DEMON: several streams, denoise, client-held latents, latents back ----

test("oneshotCommand: streams, denoise, return_latents and a client latent", () => {
  const c = oneshotCommand({
    requestId: "z",
    returnLatents: true,
    redo: { streams: [2, 0], seed: 4, denoise: [0.5, 1] },
  });
  assert.deepEqual(c, {
    type: "oneshot", request_id: "z", return_latents: true,
    redo: { streams: [2, 0], seed: 4, denoise: [0.5, 1] },
  });
  const data = new Uint16Array([1, 2, 3, 4, 5, 6]);
  const l = oneshotCommand({
    requestId: "l",
    redo: { stream: 1, seed: 2, denoise: 0.3, latent: { data, nStreams: 3, channels: 2, frames: 1 } },
  });
  assert.deepEqual(l.redo, {
    stream: 1, seed: 2, denoise: 0.3, latent: { dtype: "float16", shape: [3, 2, 1] },
  });
  assert.throws(
    () => oneshotCommand({ redo: { stream: 0, seed: 1, latent: { data, nStreams: 2, channels: 2, frames: 2 } } }),
    /expected/,
  );
});

test("oneshotLatentFrame: the latent's bytes (its own view only), or null", () => {
  const big = new Uint16Array([9, 1, 2, 3, 9]);
  const frame = oneshotLatentFrame({
    redo: { stream: 0, seed: 1, latent: { data: big.subarray(1, 4), nStreams: 1, channels: 3, frames: 1 } },
  });
  assert.deepEqual(Array.from(new Uint16Array(frame)), [1, 2, 3]);
  assert.equal(oneshotLatentFrame({ seeds: [1] }), null);
  assert.equal(oneshotLatentFrame({ redo: { id: "k", stream: 0, seed: 1 } }), null);
});

function latentMeta(over = {}) {
  return meta({
    latent_header: { n_streams: 3, channels: 2, frames: 2, seconds: 1.5 },
    latents: { dtype: "float16", shape: [2, 3, 2, 2], bytes: 2 * 12 * 2 },
    ...over,
  });
}

test("decodeOneshot: latents header and per-version latent views", () => {
  const m = latentMeta();
  const lat = new Uint16Array(24).map((_, i) => i);
  const res = decodeOneshot(m, payload(m), lat.buffer);
  assert.deepEqual(res.latentHeader, { nStreams: 3, channels: 2, frames: 2, seconds: 1.5 });
  assert.deepEqual(res.latents.shape, [2, 3, 2, 2]);
  assert.equal(res.versions[1].latent.length, 12);
  assert.equal(res.versions[1].latent[0], 12);
  assert.equal(res.versions[1].latent.buffer, lat.buffer);    // a view, no copy
  assert.throws(() => decodeOneshot(m, payload(m), new ArrayBuffer(10)), /expected/);
  assert.throws(() => decodeOneshot(m, payload(m)), /missing/);
  const plain = decodeOneshot(meta(), payload(meta()));
  assert.equal(plain.latents, undefined);
  assert.equal(plain.versions[0].latent, undefined);
});

test("OneshotFrames: one frame without latents, two with", () => {
  const f = new OneshotFrames();
  assert.equal(f.waiting, false);
  f.begin(meta());
  assert.equal(f.waiting, true);
  const one = f.push(payload(meta()));
  assert.equal(one.result.versions.length, 2);
  assert.equal(f.waiting, false);

  const m = latentMeta();
  f.begin(m);
  assert.equal(f.push(payload(m)), null);                      // PCM in, latents to come
  assert.equal(f.waiting, true);
  const two = f.push(new Uint16Array(24).buffer);
  assert.deepEqual(two.result.latents.shape, [2, 3, 2, 2]);
  assert.equal(f.waiting, false);

  f.begin(m);
  f.push(payload(m));
  const bad = f.push(new ArrayBuffer(4));
  assert.equal(bad.result, undefined);
  assert.match(bad.error, /expected/);
  assert.equal(bad.meta.request_id, "r1");
});
