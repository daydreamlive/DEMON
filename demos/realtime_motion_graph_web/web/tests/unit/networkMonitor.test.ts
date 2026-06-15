// Connection-quality monitor: landing-lead math + the bleed→unstable
// flip. The bleed watchdog flags "unstable" when slices keep arriving
// but land behind the playhead (raw source audible) — the case the
// stall watchdog (no slices at all) misses.

import {
  afterEach,
  beforeEach,
  describe,
  expect,
  it,
  vi,
} from "vitest";

import { SAMPLE_RATE } from "@demon/client";
import type { AudioPlayer, RemoteBackend } from "@demon/client";

import {
  createNetworkMonitor,
  landingLeadSeconds,
} from "@/engine/networkMonitor";
import { useNetworkStore } from "@/store/useNetworkStore";

describe("landingLeadSeconds", () => {
  const DUR = 60;

  it("is positive when the slice lands ahead of the playhead", () => {
    // playhead at 10s, slice written at 10.2s → +0.2s ahead.
    const startSample = 10.2 * SAMPLE_RATE;
    expect(landingLeadSeconds(startSample, 10, DUR)).toBeCloseTo(0.2, 5);
  });

  it("is negative when the slice lands behind the playhead", () => {
    // playhead at 10s, slice written at 9.8s → 0.2s behind (raw heard).
    const startSample = 9.8 * SAMPLE_RATE;
    expect(landingLeadSeconds(startSample, 10, DUR)).toBeCloseTo(-0.2, 5);
  });

  it("folds the loop-wrap pre-write to a small positive lead", () => {
    // Playhead near the end (59.9s), slice pre-written at the head
    // (0.1s). Raw diff is -59.8s, but across the loop seam the playhead
    // wraps to 0 in 0.1s then reaches 0.1s — so the slice is really
    // +0.2s ahead. The fold must report +0.2, not a huge negative.
    const startSample = 0.1 * SAMPLE_RATE;
    expect(landingLeadSeconds(startSample, 59.9, DUR)).toBeCloseTo(0.2, 5);
  });

  it("skips the fold when duration is unknown (<= 0)", () => {
    const startSample = 0.1 * SAMPLE_RATE;
    expect(landingLeadSeconds(startSample, 59.9, 0)).toBeCloseTo(-59.8, 4);
  });
});

describe("createNetworkMonitor bleed watchdog", () => {
  const EVAL_MS = 500;
  let nowMs: number;
  let remote: RemoteBackend;
  let player: { positionSec: number; duration: number };
  let monitor: { stop(): void };

  // Dispatch a slice whose write position is `leadS` seconds from the
  // playhead, then advance one evaluate tick.
  const tick = (leadS: number) => {
    nowMs += EVAL_MS;
    const startSample = Math.round((player.positionSec + leadS) * SAMPLE_RATE);
    remote.dispatchEvent(
      new CustomEvent("slice", { detail: { startSample } }),
    );
    vi.advanceTimersByTime(EVAL_MS);
  };

  // Advance a tick with NO slice (stall path).
  const tickSilent = () => {
    nowMs += EVAL_MS;
    vi.advanceTimersByTime(EVAL_MS);
  };

  const quality = () => useNetworkStore.getState().quality;

  beforeEach(() => {
    vi.stubGlobal("window", globalThis);
    nowMs = 1000;
    vi.spyOn(performance, "now").mockImplementation(() => nowMs);
    // Fake only the interval timer; performance.now stays on our spy.
    vi.useFakeTimers({ toFake: ["setInterval", "clearInterval"] });
    useNetworkStore.getState().reset();
    remote = new EventTarget() as unknown as RemoteBackend;
    player = { positionSec: 10, duration: 60 };
    monitor = createNetworkMonitor(remote, player as unknown as AudioPlayer);
  });

  afterEach(() => {
    monitor.stop();
    vi.useRealTimers();
    vi.restoreAllMocks();
    vi.unstubAllGlobals();
    useNetworkStore.getState().reset();
  });

  it("stays healthy while slices land ahead of the playhead", () => {
    for (let i = 0; i < 20; i++) tick(+0.2);
    expect(quality()).toBe("healthy");
  });

  it("does not flip on a brief transient dip (< escalate window)", () => {
    for (let i = 0; i < 4; i++) tick(-0.3); // ~2s of bleed, below the 6s show
    expect(quality()).toBe("healthy");
  });

  it("flips to unstable after sustained bleed, then recovers", () => {
    // 12 ticks (6s) of behind-the-playhead slices → show.
    for (let i = 0; i < 12; i++) tick(-0.3);
    expect(quality()).toBe("unstable");

    // 16 ticks (8s) of healthy leads → hide again.
    for (let i = 0; i < 16; i++) tick(+0.2);
    expect(quality()).toBe("healthy");
  });

  it("treats a tiny sub-margin negative lead as healthy", () => {
    // -0.02s is inside BLEED_LEAD_S (-0.05): noise, not a bleed.
    for (let i = 0; i < 16; i++) tick(-0.02);
    expect(quality()).toBe("healthy");
  });

  it("still flips on a total stall (no slices arriving)", () => {
    tick(+0.2); // establish a baseline slice
    // ~6 silent ticks to cross STALL_MS, then 12 more to escalate.
    for (let i = 0; i < 20; i++) tickSilent();
    expect(quality()).toBe("unstable");
  });
});
