// Tier A: AudioPlayer buffer semantics on the ScriptProcessor fallback
// path (no AudioWorklet in Node, same as a non-secure browser context).
//
// The regression this guards hardest: AudioPlayer.swap() used to
// hard-zero _spPosition, which silently overrode the
// restart_song_on_swap=false gate (whether a swap restarts from 0 is
// owned solely by useFixtureSwap's seek(0)). swap() must CLAMP the
// playhead into the new buffer, never reset it.

import { afterEach, beforeEach, describe, expect, it } from "vitest";

import { AudioPlayer, SAMPLE_RATE } from "@demon/client";

// ── minimal AudioContext stub (SP fallback) ────────────────────────────

class FakeGainParam {
  value = 1.0;
  cancelScheduledValues(): void {}
  setTargetAtTime(): void {}
}

class FakeNode {
  gain = new FakeGainParam();
  onaudioprocess: ((e: unknown) => void) | null = null;
  connect(): void {}
  disconnect(): void {}
}

class FakeAudioContext {
  sampleRate = SAMPLE_RATE;
  state = "running";
  currentTime = 0;
  destination = new FakeNode();
  // audioWorklet intentionally absent -> AudioPlayer takes the
  // ScriptProcessor fallback.
  createScriptProcessor(): FakeNode {
    return new FakeNode();
  }
  createGain(): FakeNode {
    return new FakeNode();
  }
  async close(): Promise<void> {}
}

type PlayerInternals = {
  _spPosition: number;
  _spBuffer: Float32Array | null;
};

function internals(p: AudioPlayer): PlayerInternals {
  return p as unknown as PlayerInternals;
}

function ramp(frames: number, channels = 2, offset = 0): Float32Array {
  const out = new Float32Array(frames * channels);
  for (let i = 0; i < out.length; i++) out[i] = offset + i;
  return out;
}

describe("AudioPlayer (ScriptProcessor fallback)", () => {
  let restoreCtx: unknown;
  let player: AudioPlayer;

  beforeEach(async () => {
    restoreCtx = (globalThis as { AudioContext?: unknown }).AudioContext;
    (globalThis as { AudioContext: unknown }).AudioContext = FakeAudioContext;
    player = new AudioPlayer();
    await player.init(ramp(1000), 2);
  });

  afterEach(() => {
    (globalThis as { AudioContext?: unknown }).AudioContext = restoreCtx;
  });

  it("initializes frameCount / channels / mirror from the buffer", () => {
    expect(player.frameCount).toBe(1000);
    expect(player.channels).toBe(2);
    expect(player.duration).toBeCloseTo(1000 / SAMPLE_RATE, 12);
    expect(player.getMirror()).toHaveLength(2000);
    expect(internals(player)._spBuffer).toHaveLength(2000);
  });

  // ── swap playhead semantics (the _spPosition regression) ────────────

  it("swap to a SHORTER buffer clamps the playhead, never resets to 0", () => {
    internals(player)._spPosition = 900;
    player.swap(ramp(300), 2);
    expect(player.frameCount).toBe(300);
    expect(internals(player)._spPosition).toBe(299); // clamped, NOT 0
  });

  it("swap to a LONGER buffer preserves the playhead phase", () => {
    internals(player)._spPosition = 500;
    player.swap(ramp(5000), 2);
    expect(internals(player)._spPosition).toBe(500);
  });

  it("swap bumps swapCount exactly once per swap", () => {
    expect(player.swapCount).toBe(0);
    player.swap(ramp(500), 2);
    player.swap(ramp(400), 2);
    expect(player.swapCount).toBe(2);
  });

  it("patch / addDelta must NOT bump swapCount (epoch is swap-only)", () => {
    // swapCount doubles as the source-buffer epoch the slice listener
    // compares against; bumping it on ordinary writes drops in-flight
    // slices (the seam wrap-decode desync bug).
    player.patch(0, ramp(10));
    player.addDelta(0, ramp(10));
    expect(player.swapCount).toBe(0);
  });

  // ── buffer writes ────────────────────────────────────────────────────

  it("patch overwrites; addDelta accumulates (mirror + SP buffer)", () => {
    const audio = new Float32Array([1, 2, 3, 4]); // 2 frames stereo
    player.patch(10, audio);
    let m = player.getMirror()!;
    expect(Array.from(m.subarray(20, 24))).toEqual([1, 2, 3, 4]);

    player.addDelta(10, new Float32Array([0.5, 0.5, 0.5, 0.5]));
    m = player.getMirror()!;
    expect(Array.from(m.subarray(20, 24))).toEqual([1.5, 2.5, 3.5, 4.5]);

    const sp = internals(player)._spBuffer!;
    expect(Array.from(sp.subarray(20, 24))).toEqual([1.5, 2.5, 3.5, 4.5]);
  });

  it("clamps writes that run past the end of the buffer", () => {
    const audio = new Float32Array(20).fill(7); // 10 frames
    player.patch(995, audio); // only 5 frames fit
    const m = player.getMirror()!;
    expect(Array.from(m.subarray(1990, 2000))).toEqual(
      new Array(10).fill(7),
    );
    expect(m).toHaveLength(2000); // no growth, no throw
  });

  it("ignores writes starting wholly past the end", () => {
    const before = player.getMirror()!.slice();
    player.patch(2000, new Float32Array([9, 9]));
    expect(Array.from(player.getMirror()!)).toEqual(Array.from(before));
  });

  it("notifies mirror listeners on writes and swaps", () => {
    let fired = 0;
    const off = player.onMirrorChange(() => fired++);
    player.patch(0, ramp(4));
    player.addDelta(0, ramp(4));
    player.swap(ramp(100), 2);
    expect(fired).toBe(3);
    off();
    player.patch(0, ramp(4));
    expect(fired).toBe(3);
  });

  // ── seek ─────────────────────────────────────────────────────────────

  it("seek clamps into the buffer", () => {
    player.seek(1e9);
    expect(internals(player)._spPosition).toBe(999);
    expect(player.positionSec).toBeCloseTo(999 / SAMPLE_RATE, 12);
    player.seek(-5);
    expect(internals(player)._spPosition).toBe(0);
    expect(player.positionSec).toBe(0);
  });

  it("swap can change the channel count", () => {
    player.swap(ramp(600, 1), 1);
    expect(player.channels).toBe(1);
    expect(player.frameCount).toBe(600);
  });
});
