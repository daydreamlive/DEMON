// Tier B: drive the REAL RemoteBackend with recorded server traffic.
//
// For each cached golden scenario this test replays the wire transcript
// through a fake global WebSocket, re-issues every recorded client send
// through the real RemoteBackend send methods (asserting byte/JSON wire
// equality), reconstructs the song buffer the way the app's slice
// listener + AudioPlayer do, and compares the canonical position-aligned
// region against `canonical.f32.raw` from the reference bundle.
//
// Unlike the live golden suite (tests/golden, GPU server, tolerance
// thresholds), the replay is fully deterministic: the input bytes ARE
// the recording, so the reconstructed region must match the canonical
// reference exactly. Any diff is a real client decode/apply regression
// (float16 conversion, zstd delta path, slice routing, epoch handling).
//
// Prereq: `python -m tests.golden.refs_store fetch` (run from the repo
// root via the repo venv) to populate ~/.cache/demon/test-refs/.
// Scenarios missing from the cache are reported as skipped.

import { describe, expect, it } from "vitest";

import { RemoteBackend } from "@demon/client";
import type {
  AudioSlice,
  SessionConfig,
  SwapReadyDetail,
} from "@demon/client";

import {
  BufferMirror,
  FakeWebSocket,
  diffRegions,
  installFakeWebSocket,
  loadScenario,
  scenarioCached,
  summarizeDecode,
  writeReport,
  type ScenarioBundle,
  type TranscriptBinEntry,
  type TranscriptJsonEntry,
} from "./harness";

const SCENARIOS = [
  "baseline_stream",
  "knob_step",
  "lora_enable",
  "prompt_change",
  "swap_fixture",
  "upload_path",
];

interface SliceMeta {
  flags: number;
  startSample: number;
  numSamples: number;
  epoch: number;
}

interface ReplayOutcome {
  remote: RemoteBackend;
  mirror: BufferMirror;
  ws: FakeWebSocket;
  events: string[];
  slices: SliceMeta[];
  decodeMsAll: number[];
  decodeMsRaw: number[];
  decodeMsDelta: number[];
  sliceBytes: number;
  recvJsonCounts: Record<string, number>;
  connectError: unknown;
  /** duration/channels/sampleRate snapshotted at the ready event (a
   *  later swap_ready legitimately overwrites the live fields). */
  atReady: { duration: number; channels: number; sampleRate: number } | null;
}

/** Parse the recorded pcm_upload frame (<II header + interleaved f32). */
function parsePcmUpload(buf: ArrayBuffer): {
  interleaved: Float32Array;
  channels: number;
} {
  const dv = new DataView(buf);
  const channels = dv.getUint32(0, true);
  const samples = dv.getUint32(4, true);
  const interleaved = new Float32Array(buf, 8, channels * samples).slice();
  return { interleaved, channels };
}

function bytesEqual(a: ArrayBuffer, b: ArrayBuffer): boolean {
  if (a.byteLength !== b.byteLength) return false;
  const ua = new Uint8Array(a);
  const ub = new Uint8Array(b);
  for (let i = 0; i < ua.length; i++) if (ua[i] !== ub[i]) return false;
  return true;
}

/** Re-issue a recorded client send through the real RemoteBackend send
 *  method for its message type, then assert the JSON that actually hit
 *  the wire deep-equals the recording. */
function driveRecordedSend(
  remote: RemoteBackend,
  ws: FakeWebSocket,
  entry: TranscriptJsonEntry,
): void {
  const data = entry.data;
  const before = ws.sent.length;
  switch (data.type) {
    case "params":
      remote.sendParams(
        data.raw as Record<string, number | string | boolean>,
        data.playback_pos as number,
      );
      break;
    case "prompt":
      remote.sendPrompt(
        data.tags as string,
        data.key as string | undefined,
        data.time_signature as string | undefined,
        data.tags_b as string | undefined,
      );
      break;
    case "enable_lora":
      remote.sendEnableLora(
        data.id as string,
        data.strength as number | undefined,
      );
      break;
    case "disable_lora":
      remote.sendDisableLora(data.id as string);
      break;
    case "swap_source":
      if (data.use_server_source === true) {
        remote.sendSwapSourceByName(
          data.fixture_name as string,
          data.tags as string | undefined,
          data.key as string | undefined,
          data.time_signature as string | undefined,
          data.stem_source_mode as
            | "full"
            | "vocals"
            | "instruments"
            | undefined,
        );
      } else {
        throw new Error(
          "recorded PCM swap_source replay is not wired up yet " +
            "(no golden scenario records one)",
        );
      }
      break;
    default:
      throw new Error(
        `transcript contains a send type the replay driver doesn't ` +
          `map: ${String(data.type)} — protocol drift between the ` +
          `golden client and RemoteBackend?`,
      );
  }
  expect(ws.sent.length, `send ${String(data.type)} reached the wire`).toBe(
    before + 1,
  );
  const wire = ws.sent[ws.sent.length - 1];
  expect(typeof wire).toBe("string");
  expect(JSON.parse(wire as string)).toEqual(data);
}

async function replayScenario(bundle: ScenarioBundle): Promise<ReplayOutcome> {
  const restore = installFakeWebSocket();
  try {
    // upload_path records the PCM frame the client sent; feed the same
    // samples into RemoteBackend so its auto-upload reproduces it.
    let interleaved: Float32Array = new Float32Array(0);
    let channels = 2;
    const pcmEntry = bundle.entries.find(
      (e): e is TranscriptBinEntry =>
        e.kind === "bin" && e.dir === "send" && e.role === "pcm_upload",
    );
    if (pcmEntry) {
      const parsed = parsePcmUpload(bundle.readBlob(pcmEntry));
      interleaved = parsed.interleaved;
      channels = parsed.channels;
    }

    const remote = new RemoteBackend(
      "ws://replay.invalid/session",
      interleaved,
      channels,
      bundle.config as SessionConfig,
    );

    const mirror = new BufferMirror();
    const events: string[] = [];
    const slices: SliceMeta[] = [];
    // Listener wiring mirrors useStartSession: bind BEFORE connect so the
    // first slices after ready are never missed.
    for (const type of [
      "ready",
      "slice",
      "swap_ready",
      "swap_failed",
      "params",
      "prompt_applied",
      "lora_catalog",
      "stem_assets",
      "stem_failed",
      "depth_applied",
      "error",
      "close",
      "json",
    ]) {
      remote.addEventListener(type, () => events.push(type));
    }
    let atReady: ReplayOutcome["atReady"] = null;
    remote.addEventListener("ready", () => {
      mirror.init(remote.initialBuffer!, remote.channels);
      atReady = {
        duration: remote.duration,
        channels: remote.channels,
        sampleRate: remote.sampleRate,
      };
    });
    remote.addEventListener("slice", (e) => {
      const detail = (e as CustomEvent<AudioSlice>).detail;
      slices.push({
        flags: detail.flags,
        startSample: detail.startSample,
        numSamples: detail.numSamples,
        epoch: detail.epoch,
      });
      mirror.onSlice(detail);
    });
    remote.addEventListener("swap_ready", (e) => {
      mirror.onSwapReady((e as CustomEvent<SwapReadyDetail>).detail);
    });

    let connectError: unknown;
    const connectP = remote.connect().then(
      () => {},
      (err) => {
        connectError = err;
      },
    );

    const ws = FakeWebSocket.last;
    ws.emitOpen();

    // The open handler sent the config (+ the PCM frame on the upload
    // path). Verify both against the recording before streaming.
    expect(ws.sent.length).toBe(pcmEntry ? 2 : 1);
    expect(JSON.parse(ws.sent[0] as string)).toEqual(bundle.config);
    if (pcmEntry) {
      expect(
        bytesEqual(ws.sent[1] as ArrayBuffer, bundle.readBlob(pcmEntry)),
        "client PCM upload frame matches the recorded bytes",
      ).toBe(true);
    }

    const decodeMsAll: number[] = [];
    const decodeMsRaw: number[] = [];
    const decodeMsDelta: number[] = [];
    let sliceBytes = 0;
    const recvJsonCounts: Record<string, number> = {};

    for (const entry of bundle.entries) {
      if (entry.dir === "send") {
        if (entry.kind === "bin") continue; // pcm_upload: auto-sent above
        if (entry.data === bundle.config) continue; // session config
        driveRecordedSend(remote, ws, entry);
        continue;
      }
      if (entry.kind === "json") {
        const t = String(entry.data.type ?? "");
        recvJsonCounts[t] = (recvJsonCounts[t] ?? 0) + 1;
        ws.emitMessage(JSON.stringify(entry.data));
        continue;
      }
      // recv bin — initial / slice / swap_buffer / stem
      const buf = bundle.readBlob(entry);
      if (entry.role === "slice") {
        const nBefore = slices.length;
        const t0 = performance.now();
        ws.emitMessage(buf);
        const dt = performance.now() - t0;
        // ms/slice includes header parse + zstd + f16->f32 + buffer apply
        // — the full main-thread cost of one wire slice frame.
        decodeMsAll.push(dt);
        sliceBytes += entry.bytes;
        if (slices.length === nBefore + 1) {
          (slices[slices.length - 1].flags === 1
            ? decodeMsDelta
            : decodeMsRaw
          ).push(dt);
        }
      } else {
        ws.emitMessage(buf);
      }
    }

    await connectP;
    return {
      remote,
      mirror,
      ws,
      events,
      slices,
      decodeMsAll,
      decodeMsRaw,
      decodeMsDelta,
      sliceBytes,
      recvJsonCounts,
      connectError,
      atReady,
    };
  } finally {
    restore();
  }
}

describe("RemoteBackend transcript replay", () => {
  for (const scenario of SCENARIOS) {
    it.skipIf(!scenarioCached(scenario))(scenario, async () => {
      const bundle = loadScenario(scenario);
      const out = await replayScenario(bundle);

      // ── connection + event sequence ────────────────────────────────
      expect(out.connectError, "connect() resolved").toBeUndefined();
      expect(out.remote.ready).toBe(true);
      expect(out.atReady?.duration).toBe(bundle.metrics.ready.duration);
      expect(out.atReady?.channels).toBe(bundle.metrics.ready.channels);
      expect(out.atReady?.sampleRate).toBe(bundle.metrics.ready.sample_rate);

      expect(out.events.filter((e) => e === "error")).toHaveLength(0);
      expect(out.events.filter((e) => e === "swap_failed")).toHaveLength(0);
      expect(out.events.filter((e) => e === "close")).toHaveLength(0);
      expect(out.events.filter((e) => e === "ready")).toHaveLength(1);
      // "ready" precedes every slice.
      expect(out.events.indexOf("ready")).toBeLessThan(
        out.events.indexOf("slice"),
      );

      // Every recorded slice frame decoded and dispatched.
      const recordedSlices = bundle.entries.filter(
        (e) => e.kind === "bin" && e.dir === "recv" && e.role === "slice",
      ).length;
      expect(out.slices).toHaveLength(recordedSlices);
      expect(out.slices.length).toBe(bundle.metrics.n_slices);

      // Every params_update surfaced as a params event.
      expect(out.events.filter((e) => e === "params")).toHaveLength(
        out.recvJsonCounts["params_update"] ?? 0,
      );

      // ── swap / epoch bookkeeping ───────────────────────────────────
      const recordedSwaps = bundle.entries.filter(
        (e) => e.kind === "bin" && e.dir === "recv" && e.role === "swap_buffer",
      ).length;
      expect(out.events.filter((e) => e === "swap_ready")).toHaveLength(
        recordedSwaps,
      );
      expect(out.mirror.swapCount).toBe(recordedSwaps);
      if (recordedSwaps > 0) {
        // Slices are stamped with the source epoch at WS receipt: epoch 0
        // before the swap buffer landed, 1 after — and with the listener
        // and decode running synchronously here, none may be dropped.
        const epochs = new Set(out.slices.map((s) => s.epoch));
        expect(epochs).toEqual(new Set([0, 1]));
        const firstNew = out.slices.findIndex((s) => s.epoch === 1);
        expect(
          out.slices.slice(0, firstNew).every((s) => s.epoch === 0),
          "epoch increases monotonically across the swap",
        ).toBe(true);
        expect(
          out.slices.slice(firstNew).every((s) => s.epoch === 1),
          "no stale-epoch slice after the swap",
        ).toBe(true);
      }
      if (recordedSwaps > 0) {
        // Live fields track the swap target after a swap_ready.
        const swapMsg = bundle.entries.find(
          (e) => e.kind === "json" && e.data.type === "swap_ready",
        ) as TranscriptJsonEntry | undefined;
        expect(out.remote.duration).toBe(swapMsg?.data.duration);
      }
      expect(out.mirror.droppedEpoch).toBe(0);
      expect(out.mirror.applied).toBeGreaterThan(0);

      // ── audio: canonical region must reproduce exactly ─────────────
      const region = bundle.metrics.canonical_region;
      const got = out.mirror.region(region.start_frame, region.end_frame);
      expect(got.length).toBe(bundle.canonical.length);
      const diff = diffRegions(got, bundle.canonical);
      expect(
        diff.mismatches,
        `reconstructed buffer deviates from canonical reference ` +
          `(maxAbsDiff=${diff.maxAbsDiff}, first at flat index ` +
          `${diff.firstMismatch})`,
      ).toBe(0);

      // ── perf artifact ──────────────────────────────────────────────
      const all = summarizeDecode(out.decodeMsAll);
      const report = {
        scenario,
        node: process.version,
        decode_path: "main-thread (no Worker in Node)",
        n_slices: out.slices.length,
        slice_bytes_total: out.sliceBytes,
        decode_ms: all,
        decode_ms_raw: summarizeDecode(out.decodeMsRaw),
        decode_ms_delta: summarizeDecode(out.decodeMsDelta),
        // Realtime budget: each slice covers numSamples/48000 sec of
        // audio; decode must stay far below that to keep up.
        realtime_factor:
          out.slices.reduce((acc, s) => acc + s.numSamples, 0) /
          48000 /
          Math.max(1e-9, all.total_ms / 1000),
        region_diff: diff,
      };
      const file = writeReport(`replay-${scenario}.json`, report);
      // eslint-disable-next-line no-console
      console.log(
        `[replay] ${scenario}: ${out.slices.length} slices, ` +
          `${all.mean_ms.toFixed(3)} ms/slice mean ` +
          `(p95 ${all.p95_ms.toFixed(3)}), ` +
          `${all.slices_per_sec.toFixed(0)} slices/sec, ` +
          `rt x${report.realtime_factor.toFixed(0)} -> ${file}`,
      );

      // Coarse keep-up ceiling, deliberately loose (CI-safe): decoding
      // must run at least 10x faster than the audio it represents.
      expect(report.realtime_factor).toBeGreaterThan(10);
    });
  }

  it.skipIf(SCENARIOS.some((s) => scenarioCached(s)))(
    "refs cache present",
    () => {
      throw new Error(
        "No golden reference bundles cached. Run (from the repo root, " +
          "via the repo venv): python -m tests.golden.refs_store fetch",
      );
    },
  );
});
