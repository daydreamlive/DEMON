// Live audio input client: the audio_in frame encoder, the audio_in_ack
// CustomEvent, PcmFramer framing/resampling, and MicCapture over mocked
// audio APIs (ScriptProcessor and AudioWorklet paths).

import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import {
  AUDIO_IN_SAMPLE_RATE,
  MicCapture,
  PcmFramer,
  RemoteBackend,
  type PcmFrame,
} from "@demon/client";

import { FakeWebSocket, installFakeWebSocket } from "../replay/harness";

async function openBackend() {
  const remote = new RemoteBackend(
    "ws://test.invalid/",
    new Float32Array(0),
    1,
    { backend: "live_input", text_only: true },
  );
  const p = remote.connect();
  const ws = FakeWebSocket.last;
  ws.emitOpen();
  ws.emitMessage(
    JSON.stringify({
      type: "ready",
      duration: 0.02,
      sample_rate: 48000,
      channels: 1,
      capabilities: { audio_in: true, generates_audio: false },
    }),
  );
  ws.emitMessage(new ArrayBuffer(0));
  await p.catch(() => {});
  return { remote, ws };
}

describe("RemoteBackend.sendAudioIn", () => {
  let restore: () => void;
  beforeEach(() => {
    restore = installFakeWebSocket();
  });
  afterEach(() => restore());

  it("sends the audio_in header then one <II> + float32 PCM frame", async () => {
    const { remote, ws } = await openBackend();
    const before = ws.sent.length;
    // A subarray of a larger buffer: only the view's bytes may go out.
    const big = new Float32Array(10).map((_, i) => i / 10);
    const frame = big.subarray(2, 6);
    expect(remote.sendAudioIn(frame, 7, { startS: 0.7 })).toBe(true);
    expect(remote.sendAudioIn(new Float32Array(4), 8)).toBe(true);
    const sent = ws.sent.slice(before);
    expect(JSON.parse(sent[0] as string)).toEqual({
      type: "audio_in",
      seq: 7,
      start_s: 0.7,
    });
    const bin = sent[1] as ArrayBuffer;
    const dv = new DataView(bin);
    expect(dv.getUint32(0, true)).toBe(1); // channels
    expect(dv.getUint32(4, true)).toBe(4); // samples
    expect(Array.from(new Float32Array(bin, 8))).toEqual(Array.from(frame));
    expect(bin.byteLength).toBe(8 + 4 * 4);
    expect(JSON.parse(sent[2] as string)).toEqual({ type: "audio_in", seq: 8 });
  });

  it("dispatches audio_in_ack as a typed CustomEvent", async () => {
    const { remote, ws } = await openBackend();
    const acks: unknown[] = [];
    remote.addEventListener("audio_in_ack", (e) =>
      acks.push((e as CustomEvent).detail),
    );
    ws.emitMessage(
      JSON.stringify({ type: "audio_in_ack", seq: 12, frontier_s: 1.3 }),
    );
    expect(acks).toEqual([{ type: "audio_in_ack", seq: 12, frontier_s: 1.3 }]);
  });

  it("refuses to send on a closed socket", async () => {
    const { remote, ws } = await openBackend();
    ws.close();
    expect(remote.sendAudioIn(new Float32Array(4), 0)).toBe(false);
  });
});

describe("PcmFramer", () => {
  it("frames 48 kHz input into 100 ms frames across block boundaries", () => {
    const f = new PcmFramer({ inputRate: 48000 });
    expect(f.frameSamples).toBe(4800);
    const frames: PcmFrame[] = [];
    for (let i = 0; i < 10; i++) frames.push(...f.push(new Float32Array(1000).fill(0.5)));
    expect(frames.map((x) => x.seq)).toEqual([0, 1]);
    expect(frames.map((x) => x.startS)).toEqual([0, 0.1]);
    expect(frames[0].pcm.length).toBe(4800);
    expect(frames[0].rms).toBeCloseTo(0.5, 6);
    expect(frames[0].peak).toBeCloseTo(0.5, 6);
    // 400 samples carried; 4400 more completes the third frame.
    expect(f.push(new Float32Array(4399))).toEqual([]);
    expect(f.push(new Float32Array(1))[0].seq).toBe(2);
  });

  it("restarts numbering on reset", () => {
    const f = new PcmFramer({ inputRate: 48000, frameMs: 10 });
    expect(f.push(new Float32Array(480 * 3)).map((x) => x.seq)).toEqual([0, 1, 2]);
    f.push(new Float32Array(100));
    f.reset();
    const next = f.push(new Float32Array(480));
    expect(next.map((x) => [x.seq, x.startS])).toEqual([[0, 0]]);
  });

  it("resamples a 44.1 kHz context to 48 kHz frames", () => {
    const f = new PcmFramer({ inputRate: 44100 });
    const frames: PcmFrame[] = [];
    // One second of a ramp in odd-sized blocks.
    const n = 44100;
    const ramp = new Float32Array(n).map((_, i) => i / n);
    for (let i = 0; i < n; i += 997) frames.push(...f.push(ramp.subarray(i, i + 997)));
    // ~48000 output samples => 9 complete frames (the 10th needs the
    // final interpolation sample).
    expect(frames.length).toBe(9);
    const out = new Float32Array(frames.length * 4800);
    frames.forEach((fr, k) => out.set(fr.pcm, k * 4800));
    // Linear interpolation of a ramp is the ramp, continuous across blocks.
    for (let i = 1; i < out.length; i++) {
      expect(out[i] - out[i - 1]).toBeCloseTo(44100 / 48000 / n, 6);
    }
  });
});

// ---- MicCapture over mocked audio APIs ---------------------------------------

function fakeNode() {
  return { connect: vi.fn(), disconnect: vi.fn() };
}

function fakeContext(sampleRate: number, withWorklet: boolean) {
  const ctx = {
    sampleRate,
    state: "running",
    destination: {},
    closed: false,
    processor: null as null | { onaudioprocess: ((ev: unknown) => void) | null },
    createMediaStreamSource: vi.fn(() => fakeNode()),
    createGain: vi.fn(() => ({ ...fakeNode(), gain: { value: 1 } })),
    createScriptProcessor: vi.fn(() => {
      const p = { ...fakeNode(), onaudioprocess: null };
      ctx.processor = p;
      return p;
    }),
    resume: vi.fn(async () => {}),
    close: vi.fn(async () => {
      ctx.closed = true;
    }),
    audioWorklet: withWorklet ? { addModule: vi.fn(async () => {}) } : undefined,
  };
  return ctx;
}

function fakeMedia() {
  const track = { stop: vi.fn() };
  const media = {
    getUserMedia: vi.fn(async (_c: MediaStreamConstraints) => ({
      getTracks: () => [track],
    })),
  };
  return { media, track };
}

function processEvent(channels: Float32Array[]) {
  return {
    inputBuffer: {
      numberOfChannels: channels.length,
      getChannelData: (c: number) => channels[c],
    },
  };
}

describe("MicCapture", () => {
  it("opens an unprocessed mono mic and frames the ScriptProcessor path", async () => {
    const ctx = fakeContext(48000, false);
    const { media, track } = fakeMedia();
    const frames: PcmFrame[] = [];
    const levels: { rms: number; peak: number }[] = [];
    const mic = new MicCapture({
      onFrame: (f) => frames.push(f),
      onLevel: (l) => levels.push(l),
      mediaDevices: media as unknown as MediaDevices,
      audioContextFactory: (o) => {
        expect(o.sampleRate).toBe(AUDIO_IN_SAMPLE_RATE);
        return ctx as unknown as AudioContext;
      },
    });
    await mic.start();
    expect(mic.running).toBe(true);
    expect(mic.usingWorklet).toBe(false);
    const constraints = media.getUserMedia.mock.calls[0][0];
    expect(constraints.audio).toMatchObject({
      channelCount: 1,
      echoCancellation: false,
      noiseSuppression: false,
      autoGainControl: false,
    });
    // Stereo blocks are averaged to mono.
    for (let i = 0; i < 5; i++) {
      ctx.processor!.onaudioprocess!(
        processEvent([new Float32Array(2048).fill(0.2), new Float32Array(2048).fill(0.6)]),
      );
    }
    expect(frames.map((f) => f.seq)).toEqual([0, 1]);
    expect(frames[0].pcm[0]).toBeCloseTo(0.4, 6);
    expect(levels).toHaveLength(2);
    expect(levels[0].rms).toBeCloseTo(0.4, 6);

    await mic.stop();
    expect(mic.running).toBe(false);
    expect(track.stop).toHaveBeenCalled();
    expect(ctx.closed).toBe(true);
    // Late callbacks after stop are ignored.
    ctx.processor!.onaudioprocess?.(processEvent([new Float32Array(9600)]));
    expect(frames).toHaveLength(2);
  });

  it("uses the AudioWorklet when available and restarts seq after stop", async () => {
    const ports: { onmessage: ((ev: { data: Float32Array }) => void) | null }[] = [];
    class FakeWorkletNode {
      port = { onmessage: null as ((ev: { data: Float32Array }) => void) | null };
      connect = vi.fn();
      disconnect = vi.fn();
      constructor(_ctx: unknown, name: string) {
        expect(name).toBe("demon-mic-tap");
        ports.push(this.port);
      }
    }
    vi.stubGlobal("AudioWorkletNode", FakeWorkletNode);
    try {
      const frames: PcmFrame[] = [];
      const ctxs: ReturnType<typeof fakeContext>[] = [];
      const mic = new MicCapture({
        onFrame: (f) => frames.push(f),
        mediaDevices: fakeMedia().media as unknown as MediaDevices,
        audioContextFactory: () => {
          const c = fakeContext(48000, true);
          ctxs.push(c);
          return c as unknown as AudioContext;
        },
      });
      await mic.start();
      expect(mic.usingWorklet).toBe(true);
      expect(ctxs[0].audioWorklet!.addModule).toHaveBeenCalledTimes(1);
      for (let i = 0; i < 5; i++) ports[0].onmessage!({ data: new Float32Array(1024).fill(0.1) });
      expect(frames.map((f) => f.seq)).toEqual([0]);
      await mic.stop();
      await mic.start();
      for (let i = 0; i < 5; i++) ports[1].onmessage!({ data: new Float32Array(1024) });
      expect(frames.map((f) => f.seq)).toEqual([0, 0]);
      await mic.stop();
    } finally {
      vi.unstubAllGlobals();
    }
  });

  it("releases the microphone when the audio graph fails to build", async () => {
    const { media, track } = fakeMedia();
    const mic = new MicCapture({
      onFrame: () => {},
      mediaDevices: media as unknown as MediaDevices,
      audioContextFactory: () => {
        throw new Error("no audio device");
      },
    });
    await expect(mic.start()).rejects.toThrow("no audio device");
    expect(track.stop).toHaveBeenCalled();
    expect(mic.running).toBe(false);
  });
});
