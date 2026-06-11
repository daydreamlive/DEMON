// Tier A: source-buffer epoch semantics on RemoteBackend — the
// swap-bleed bug class. Slices are stamped with the epoch current at WS
// receipt; the epoch bumps BEFORE swap_ready dispatch so anything
// arriving after the swap buffer is tagged for the new source, while
// stale slices keep the old epoch and get dropped by the app's
// `epoch !== player.swapCount` guard.

import { afterEach, beforeEach, describe, expect, it } from "vitest";

import { RemoteBackend, SLICE_FLAG_DELTA } from "@demon/client";
import type { AudioSlice, SwapReadyDetail } from "@demon/client";

import {
  FakeWebSocket,
  installFakeWebSocket,
} from "../replay/harness";
import { f16ExactRamp, makeBufferFrame, makeSliceFrame } from "./wire";

const READY = {
  type: "ready",
  duration: 0.02,
  sample_rate: 48000,
  channels: 2,
};

interface Session {
  remote: RemoteBackend;
  ws: FakeWebSocket;
  slices: AudioSlice[];
  swaps: SwapReadyDetail[];
}

async function openSession(initialFrames = 960): Promise<Session> {
  const remote = new RemoteBackend(
    "ws://test.invalid/",
    new Float32Array(0),
    2,
    { use_server_fixture: true, fixture_name: "x.wav" },
  );
  const slices: AudioSlice[] = [];
  const swaps: SwapReadyDetail[] = [];
  remote.addEventListener("slice", (e) => {
    slices.push((e as CustomEvent<AudioSlice>).detail);
  });
  remote.addEventListener("swap_ready", (e) => {
    swaps.push((e as CustomEvent<SwapReadyDetail>).detail);
  });
  const p = remote.connect();
  const ws = FakeWebSocket.last;
  ws.emitOpen();
  ws.emitMessage(JSON.stringify(READY));
  ws.emitMessage(makeBufferFrame(f16ExactRamp(initialFrames * 2)));
  await p;
  return { remote, ws, slices, swaps };
}

function emitSwap(ws: FakeWebSocket, frames = 480): void {
  ws.emitMessage(
    JSON.stringify({
      type: "swap_ready",
      duration: frames / 48000,
      channels: 2,
      fixture_name: "y.wav",
    }),
  );
  ws.emitMessage(makeBufferFrame(f16ExactRamp(frames * 2)));
}

describe("RemoteBackend slice epochs", () => {
  let restore: () => void;
  beforeEach(() => {
    restore = installFakeWebSocket();
  });
  afterEach(() => {
    restore();
  });

  it("stamps pre-swap slices 0 and post-swap slices 1", async () => {
    const { ws, slices, swaps } = await openSession();
    ws.emitMessage(
      makeSliceFrame({ startSample: 0, audio: f16ExactRamp(64) }),
    );
    expect(slices).toHaveLength(1);
    expect(slices[0].epoch).toBe(0);

    emitSwap(ws);
    expect(swaps).toHaveLength(1);

    ws.emitMessage(
      makeSliceFrame({ startSample: 32, audio: f16ExactRamp(64) }),
    );
    expect(slices).toHaveLength(2);
    expect(slices[1].epoch).toBe(1);
  });

  it("bumps the epoch BEFORE dispatching swap_ready", async () => {
    // The invariant the swap-bleed fix rests on: a slice handed to the
    // WS from inside the swap_ready listener itself (i.e. the instant
    // the swap completes) must already carry the new epoch.
    const { remote, ws, slices } = await openSession();
    let epochInsideListener = -1;
    remote.addEventListener("swap_ready", () => {
      ws.emitMessage(
        makeSliceFrame({ startSample: 0, audio: f16ExactRamp(8) }),
      );
      epochInsideListener = slices[slices.length - 1].epoch;
    });
    emitSwap(ws);
    expect(epochInsideListener).toBe(1);
  });

  it("setSliceEpoch realigns stamping after a reconnect", async () => {
    // Reconnect path: a fresh RemoteBackend starts at epoch 0, but
    // player.swap() bumped swapCount on the surviving AudioPlayer.
    // useStartSession calls setSliceEpoch(player.swapCount) so the
    // listener's equality guard doesn't drop every recovered slice.
    const { remote, ws, slices } = await openSession();
    remote.setSliceEpoch(3);
    ws.emitMessage(
      makeSliceFrame({ startSample: 0, audio: f16ExactRamp(8) }),
    );
    expect(slices[slices.length - 1].epoch).toBe(3);
  });

  it("decodes RAW and DELTA payloads identically on the fallback path", async () => {
    const { ws, slices } = await openSession();
    const audio = f16ExactRamp(128);
    ws.emitMessage(makeSliceFrame({ startSample: 16, audio }));
    ws.emitMessage(
      makeSliceFrame({
        startSample: 16,
        audio,
        flags: SLICE_FLAG_DELTA,
      }),
    );
    expect(slices).toHaveLength(2);
    const [raw, delta] = slices;
    expect(raw.flags).toBe(0);
    expect(delta.flags).toBe(SLICE_FLAG_DELTA);
    expect(raw.numSamples).toBe(64);
    expect(delta.numSamples).toBe(64);
    expect(Array.from(delta.audio)).toEqual(Array.from(raw.audio));
    expect(Array.from(raw.audio)).toEqual(Array.from(audio));
  });

  it("routes the post-swap_ready binary to swap_ready, not to slices", async () => {
    const { ws, slices, swaps } = await openSession();
    emitSwap(ws, 240);
    expect(swaps).toHaveLength(1);
    expect(slices).toHaveLength(0);
    expect(swaps[0].interleaved).toHaveLength(240 * 2);
    expect(swaps[0].duration).toBeCloseTo(240 / 48000, 10);
  });
});
