// RemoteBackend.sendParams must report whether the message actually reached
// the wire. The 125 Hz param sync (useParamSync) consumes a one-shot
// worst-slice-lead sample that clears on read; if sendParams silently drops
// the tick (socket not open, or the backpressure gate fires) the caller has
// to re-arm that sample, so it needs a truthful success signal. These tests
// lock the boolean contract — including the reviewer's repro: a negative lead
// must not be reported as sent when bufferedAmount sits above the gate.

import { afterEach, beforeEach, describe, expect, it } from "vitest";

import { RemoteBackend } from "@demon/client";

import { FakeWebSocket, installFakeWebSocket } from "../replay/harness";

// Mirror of PARAMS_BACKPRESSURE_BYTES (8 KiB) in protocol.ts; 9000 sits above
// it, matching the reviewer's reproduction.
const OVER_GATE = 9000;

async function openBackend(): Promise<{
  remote: RemoteBackend;
  ws: FakeWebSocket;
}> {
  const remote = new RemoteBackend(
    "ws://test.invalid/",
    new Float32Array(0),
    2,
    { use_server_fixture: true, fixture_name: "x.wav" },
  );
  const p = remote.connect();
  const ws = FakeWebSocket.last;
  ws.emitOpen();
  ws.emitMessage(
    JSON.stringify({
      type: "ready",
      duration: 0.02,
      sample_rate: 48000,
      channels: 2,
    }),
  );
  // A zero-filled buffer frame: header (see makeBufferFrame) is implicit in
  // the encoder, but RemoteBackend only needs *a* buffer frame to leave the
  // initial-buffer phase. Empty interleaved data is fine for this test — we
  // never read the audio back.
  ws.emitMessage(new ArrayBuffer(0));
  await p.catch(() => {});
  return { remote, ws };
}

function paramsSent(ws: FakeWebSocket): string[] {
  return ws.sent.filter(
    (d): d is string => typeof d === "string" && d.includes('"type":"params"'),
  );
}

describe("RemoteBackend.sendParams backpressure / success signal", () => {
  let restore: () => void;
  beforeEach(() => {
    restore = installFakeWebSocket();
  });
  afterEach(() => {
    restore();
  });

  it("returns true and writes the wire on an open, drained socket", async () => {
    const { remote, ws } = await openBackend();
    ws.bufferedAmount = 0;
    const before = paramsSent(ws).length;
    const ok = remote.sendParams({ seed: 1 }, 0.5, -0.75);
    expect(ok).toBe(true);
    const after = paramsSent(ws);
    expect(after.length).toBe(before + 1);
    // The negative lead reached the wire so the server can widen its lead.
    expect(JSON.parse(after[after.length - 1]).slice_lead_s).toBe(-0.75);
  });

  it("returns false and writes nothing when bufferedAmount is over the gate", async () => {
    const { remote, ws } = await openBackend();
    ws.bufferedAmount = OVER_GATE;
    const before = paramsSent(ws).length;
    // Reviewer's repro: a worst lead of -0.75 with a congested socket.
    const ok = remote.sendParams({ seed: 1 }, 0.5, -0.75);
    expect(ok).toBe(false);
    expect(paramsSent(ws).length).toBe(before);
  });

  it("returns false when the socket is not open", async () => {
    const { remote, ws } = await openBackend();
    ws.bufferedAmount = 0;
    ws.readyState = FakeWebSocket.CLOSED;
    expect(remote.sendParams({ seed: 1 }, 0.5, -0.75)).toBe(false);
  });
});
