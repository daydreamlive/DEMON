// Lean windows and swaps: a derived_windows session builds a silent initial
// buffer and takes a silent or kept swap_ready without a binary frame, and
// a compact swap_source frame carries only the channels with audio, as
// 16-bit samples, where the server says it takes one (compact_pcm).
//
// Run: npm test (node --test; needs Node 22.6+ for .ts type stripping).

import assert from "node:assert/strict";
import { test } from "node:test";

// The built bundle: the source's extensionless imports need a bundler.
import { packCompactPcmFrame, RemoteBackend } from "../dist/demon-client.js";

/** A pod over a fake socket: it records what the client sends and answers what the test scripts. */
class FakePod {
  static OPEN = 1;
  static last = null;
  readyState = 0;
  binaryType = "";
  bufferedAmount = 0;
  sent = [];
  constructor() {
    FakePod.last = this;
    setTimeout(() => ((this.readyState = 1), this.onopen?.()));
  }
  send(x) {
    this.sent.push(x);
  }
  close() {
    this.readyState = 3;
  }
  say(x) {
    this.onmessage?.({ data: typeof x === "string" ? x : x });
  }
}

const ready = (extra = {}) =>
  JSON.stringify({ type: "ready", duration: 0.5, sample_rate: 48000, channels: 10, capabilities: { swap: true, compact_pcm: true }, ...extra });

async function open(readyExtra) {
  const remote = new RemoteBackend("ws://pod", new Float32Array(48000), 2, { derived_windows: true }, { WebSocketConstructor: FakePod });
  const connected = remote.connect();
  await new Promise((r) => setTimeout(r, 5));
  FakePod.last.say(ready(readyExtra));
  return { remote, connected };
}

test("a silent initial window is built, not waited for", async () => {
  const { remote, connected } = await open({ window: "silent", frames: 24000 });
  await connected;
  assert.equal(remote.initialBuffer.length, 24000 * 10);
  assert.ok(remote.initialBuffer.every((v) => v === 0));
  assert.equal(JSON.parse(FakePod.last.sent[0]).derived_windows, true, "the config asks for it");
});

test("without window, the binary frame follows as always", async () => {
  const { remote, connected } = await open({});
  let done = false;
  connected.then(() => (done = true));
  await new Promise((r) => setTimeout(r, 5));
  assert.equal(done, false, "still waiting for the frame");
  FakePod.last.say(new Uint16Array(20).buffer);
  await connected;
  assert.equal(remote.initialBuffer.length, 20);
});

test("a silent or kept swap_ready lands at once; a plain one waits for its frame", async () => {
  const { remote, connected } = await open({ window: "silent", frames: 4 });
  await connected;
  const got = [];
  remote.addEventListener("swap_ready", (e) => got.push(e.detail));
  const swap = { type: "swap_ready", duration: 0.5, sample_rate: 48000, channels: 10 };
  FakePod.last.say(JSON.stringify({ ...swap, window: "silent", frames: 3, source_epoch: 1 }));
  FakePod.last.say(JSON.stringify({ ...swap, window: "kept", frames: 3, source_epoch: 2 }));
  assert.equal(got.length, 2);
  assert.deepEqual([...got[0].interleaved], new Array(30).fill(0));
  assert.equal(got[1].interleaved, null, "kept: the listener keeps its own buffer");
  assert.equal(remote.sliceEpoch ?? remote._sliceEpoch, 2);
  FakePod.last.say(JSON.stringify({ ...swap, source_epoch: 3 }));
  assert.equal(got.length, 2, "a plain swap_ready waits for its frame");
  FakePod.last.say(new Uint16Array(10).buffer);
  assert.equal(got.length, 3);
  assert.equal(got[2].interleaved.length, 10);
});

test("a compact frame carries only the channels with audio, as 16-bit", () => {
  const ch = 10, n = 4;
  const a = new Float32Array(n * ch);
  for (let f = 0; f < n; f++) (a[f * ch + 4] = 0.5), (a[f * ch + 5] = -1.5);
  const { frame, present } = packCompactPcmFrame(a, ch);
  assert.deepEqual(present, [4, 5]);
  const dv = new DataView(frame.buffer);
  assert.deepEqual([dv.getUint32(0, true), dv.getUint32(4, true)], [2, n]);
  assert.equal(frame.byteLength, 8 + n * 2 * 2);
  const pcm = new Int16Array(frame.buffer, 8);
  assert.deepEqual([pcm[0], pcm[1]], [16384, -32768], "scaled by 32768, clamped");
  assert.deepEqual(packCompactPcmFrame(new Float32Array(8), 2).present, [0], "a silent buffer sends one channel");
});

test("sendSwapSourceCompact sends the compact frame only where the server takes it", async () => {
  const { remote, connected } = await open({ window: "silent", frames: 4 });
  await connected;
  const a = new Float32Array(4 * 10);
  a[3] = 0.25;
  const pod = FakePod.last;
  pod.sent.length = 0;
  assert.equal(remote.sendSwapSourceCompact(a, 10, "jazz"), true);
  const cmd = JSON.parse(pod.sent[0]);
  assert.deepEqual(cmd, { type: "swap_source", pcm: { encoding: "s16le", channels: 10, present: [3] }, tags: "jazz" });
  assert.equal(pod.sent[1].byteLength, 8 + 4 * 2);
  remote.capabilities = { swap: true };
  pod.sent.length = 0;
  remote.sendSwapSourceCompact(a, 10);
  assert.equal(JSON.parse(pod.sent[0]).pcm, undefined, "no compact_pcm: the full float32 frame");
  assert.equal(pod.sent[1].byteLength, 8 + a.byteLength);
});
