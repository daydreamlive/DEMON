// Replay harness for driving the REAL `RemoteBackend` (packages/demon-client/protocol.ts)
// from recorded golden-session transcripts — no GPU, no server, no network.
//
// Transcripts come from the golden harness (tests/golden/client.py in the
// repo root): one `transcript.jsonl` + `blobs/` per scenario, cached at
// `~/.cache/demon/test-refs/<scenario>/` by
// `python -m tests.golden.refs_store fetch` (run from the repo root via
// the repo venv).
//
// Each transcript line is {t: <ms>, dir: "send"|"recv", kind: "json"|"bin",
// ...}; binary frames reference a blob file and carry a `role` tag
// (initial / slice / swap_buffer / stem / pcm_upload) assigned by the
// recorder's pending-state machine, so the replay can re-frame them
// without re-deriving protocol state.
//
// In Node there is no `Worker`, so RemoteBackend falls back to the
// main-thread slice decode path (_parseSlice: fzstd + float16->float32) —
// which is exactly the code we want to regression-test and measure.

import * as fs from "node:fs";
import * as os from "node:os";
import * as path from "node:path";
import { fileURLToPath } from "node:url";

export const REFS_CACHE_DIR = path.join(
  os.homedir(),
  ".cache",
  "demon",
  "test-refs",
);

// ── transcript types ───────────────────────────────────────────────────

export interface TranscriptJsonEntry {
  t: number;
  dir: "send" | "recv";
  kind: "json";
  data: Record<string, unknown> & { type?: string };
}

export interface TranscriptBinEntry {
  t: number;
  dir: "send" | "recv";
  kind: "bin";
  role: "initial" | "slice" | "swap_buffer" | "stem" | "pcm_upload";
  bytes: number;
  blob?: string;
}

export type TranscriptEntry = TranscriptJsonEntry | TranscriptBinEntry;

export interface CanonicalRegion {
  start_frame: number;
  end_frame: number;
  start_s: number;
  len_s: number;
}

export interface ScenarioBundle {
  scenario: string;
  dir: string;
  entries: TranscriptEntry[];
  /** Recorded session config (the first send json line). */
  config: Record<string, unknown>;
  metrics: {
    scenario: string;
    ready: {
      duration: number;
      channels: number;
      sample_rate: number;
      [k: string]: unknown;
    };
    canonical_region: CanonicalRegion;
    coverage: [number, number][];
    n_slices: number;
    [k: string]: unknown;
  };
  /** Position-aligned reference audio region, interleaved float32. */
  canonical: Float32Array;
  /** Read a binary entry's blob as a standalone ArrayBuffer. */
  readBlob(entry: TranscriptBinEntry): ArrayBuffer;
}

export function scenarioCached(scenario: string): boolean {
  return fs.existsSync(
    path.join(REFS_CACHE_DIR, scenario, "transcript.jsonl"),
  );
}

export function listCachedScenarios(): string[] {
  if (!fs.existsSync(REFS_CACHE_DIR)) return [];
  return fs
    .readdirSync(REFS_CACHE_DIR)
    .filter((name) => scenarioCached(name))
    .sort();
}

export function loadScenario(scenario: string): ScenarioBundle {
  const dir = path.join(REFS_CACHE_DIR, scenario);
  const lines = fs
    .readFileSync(path.join(dir, "transcript.jsonl"), "utf-8")
    .split("\n")
    .filter((l) => l.trim().length > 0);
  const entries = lines.map((l) => JSON.parse(l) as TranscriptEntry);
  const first = entries[0];
  if (first.kind !== "json" || first.dir !== "send") {
    throw new Error(
      `${scenario}: transcript does not start with the config send`,
    );
  }
  const metrics = JSON.parse(
    fs.readFileSync(path.join(dir, "metrics.json"), "utf-8"),
  ) as ScenarioBundle["metrics"];
  const rawBytes = fs.readFileSync(path.join(dir, "canonical.f32.raw"));
  const canonical = new Float32Array(
    rawBytes.buffer.slice(
      rawBytes.byteOffset,
      rawBytes.byteOffset + rawBytes.byteLength,
    ),
  );
  return {
    scenario,
    dir,
    entries,
    config: first.data,
    metrics,
    canonical,
    readBlob(entry: TranscriptBinEntry): ArrayBuffer {
      if (!entry.blob) {
        throw new Error(`${scenario}: bin entry has no blob reference`);
      }
      const buf = fs.readFileSync(path.join(dir, entry.blob));
      // Standalone ArrayBuffer copy: RemoteBackend's onmessage branches on
      // `ev.data instanceof ArrayBuffer`, and a Node Buffer's backing store
      // is a pooled ArrayBuffer with a nonzero byteOffset.
      return buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength);
    },
  };
}

// ── fake WebSocket ─────────────────────────────────────────────────────

type WsData = string | ArrayBuffer;

/** Minimal scripted stand-in for the browser WebSocket, with just the
 *  surface RemoteBackend touches: handler properties, readyState,
 *  binaryType, send(), close(), and the OPEN/... statics. The test feeds
 *  recorded frames through `emitOpen()` / `emitMessage()` synchronously. */
export class FakeWebSocket {
  static CONNECTING = 0;
  static OPEN = 1;
  static CLOSING = 2;
  static CLOSED = 3;

  static instances: FakeWebSocket[] = [];
  static reset(): void {
    FakeWebSocket.instances = [];
  }
  static get last(): FakeWebSocket {
    const inst = FakeWebSocket.instances[FakeWebSocket.instances.length - 1];
    if (!inst) throw new Error("no FakeWebSocket constructed yet");
    return inst;
  }

  url: string;
  binaryType = "blob";
  readyState = FakeWebSocket.CONNECTING;
  /** Mirrors the real socket's send-queue depth. Tests set this to drive
   *  RemoteBackend.sendParams's backpressure gate; send() never auto-bumps
   *  it (the fake drains instantly). */
  bufferedAmount = 0;
  onopen: (() => void) | null = null;
  onmessage: ((ev: { data: WsData }) => void) | null = null;
  onerror: ((e: unknown) => void) | null = null;
  onclose:
    | ((e: { code: number; reason: string; wasClean: boolean }) => void)
    | null = null;

  /** Everything the client wrote to the socket, in order. */
  sent: WsData[] = [];

  constructor(url: string) {
    this.url = url;
    FakeWebSocket.instances.push(this);
  }

  send(data: WsData | Uint8Array): void {
    if (this.readyState !== FakeWebSocket.OPEN) {
      throw new Error("send on non-open FakeWebSocket");
    }
    if (typeof data === "string") {
      this.sent.push(data);
    } else if (data instanceof Uint8Array) {
      // Copy out so later mutation of the source buffer can't rewrite
      // what we assert against.
      const copy = new ArrayBuffer(data.byteLength);
      new Uint8Array(copy).set(data);
      this.sent.push(copy);
    } else {
      this.sent.push(data.slice(0));
    }
  }

  close(code = 1000, reason = ""): void {
    if (this.readyState === FakeWebSocket.CLOSED) return;
    this.readyState = FakeWebSocket.CLOSED;
    this.onclose?.({ code, reason, wasClean: true });
  }

  // ── test-side drivers ────────────────────────────────────────────────

  emitOpen(): void {
    this.readyState = FakeWebSocket.OPEN;
    this.onopen?.();
  }

  emitMessage(data: WsData): void {
    this.onmessage?.({ data });
  }

  emitClose(code: number, reason = "", wasClean = false): void {
    this.readyState = FakeWebSocket.CLOSED;
    this.onclose?.({ code, reason, wasClean });
  }
}

/** Install FakeWebSocket as the global WebSocket. Returns a restore fn. */
export function installFakeWebSocket(): () => void {
  const g = globalThis as { WebSocket?: unknown };
  const prev = g.WebSocket;
  g.WebSocket = FakeWebSocket;
  FakeWebSocket.reset();
  return () => {
    g.WebSocket = prev;
    FakeWebSocket.reset();
  };
}

// ── client-mirror buffer reconstruction ────────────────────────────────

import type { AudioSlice, SwapReadyDetail } from "@demon/client";

/** Reconstructs the song buffer the way the app's slice listener +
 *  AudioPlayer do (useStartSession.wireRemoteListeners): RAW slices
 *  overwrite their region, DELTA slices add into it, a swap_ready
 *  buffer replaces it wholesale, and slices whose epoch doesn't match
 *  the current swap generation are dropped. */
export class BufferMirror {
  buffer: Float32Array | null = null;
  channels = 2;
  swapCount = 0;
  /** Slices applied / dropped-by-epoch / dropped-out-of-range. */
  applied = 0;
  droppedEpoch = 0;
  droppedRange = 0;

  init(interleaved: Float32Array, channels: number): void {
    this.buffer = interleaved.slice();
    this.channels = channels;
  }

  onSwapReady(detail: SwapReadyDetail): void {
    this.buffer = detail.interleaved.slice();
    this.channels = detail.channels;
    this.swapCount++;
  }

  onSlice(slice: AudioSlice): void {
    // Mirrors the app's epoch guard (slice listener drops stale-source
    // slices; AudioPlayer.swap bumps swapCount in lockstep with the
    // protocol's _sliceEpoch).
    if (slice.epoch !== this.swapCount) {
      this.droppedEpoch++;
      return;
    }
    const buf = this.buffer;
    if (!buf) return;
    const ch = this.channels;
    const frames = (buf.length / ch) | 0;
    const s = Math.floor(slice.startSample);
    const n = slice.numSamples;
    if (s + n > frames) {
      this.droppedRange++;
      return;
    }
    const base = s * ch;
    const audio = slice.audio;
    if (slice.flags === 1 /* SLICE_FLAG_DELTA */) {
      for (let i = 0; i < audio.length; i++) buf[base + i] += audio[i];
    } else {
      for (let i = 0; i < audio.length; i++) buf[base + i] = audio[i];
    }
    this.applied++;
  }

  /** Interleaved view of [startFrame, endFrame). */
  region(startFrame: number, endFrame: number): Float32Array {
    if (!this.buffer) throw new Error("mirror not initialized");
    return this.buffer.subarray(
      startFrame * this.channels,
      endFrame * this.channels,
    );
  }
}

export interface RegionDiff {
  n: number;
  mismatches: number;
  maxAbsDiff: number;
  firstMismatch: number | null;
}

export function diffRegions(a: Float32Array, b: Float32Array): RegionDiff {
  if (a.length !== b.length) {
    throw new Error(`region length mismatch: ${a.length} vs ${b.length}`);
  }
  let mismatches = 0;
  let maxAbsDiff = 0;
  let firstMismatch: number | null = null;
  for (let i = 0; i < a.length; i++) {
    const d = Math.abs(a[i] - b[i]);
    if (d > 0 || a[i] !== b[i]) {
      mismatches++;
      if (firstMismatch === null) firstMismatch = i;
      if (d > maxAbsDiff) maxAbsDiff = d;
    }
  }
  return { n: a.length, mismatches, maxAbsDiff, firstMismatch };
}

// ── perf reporting ─────────────────────────────────────────────────────

export function percentile(sorted: number[], p: number): number {
  if (sorted.length === 0) return NaN;
  const idx = Math.min(
    sorted.length - 1,
    Math.max(0, Math.round((p / 100) * (sorted.length - 1))),
  );
  return sorted[idx];
}

export interface DecodeStats {
  n: number;
  p50_ms: number;
  p95_ms: number;
  max_ms: number;
  mean_ms: number;
  total_ms: number;
  slices_per_sec: number;
}

export function summarizeDecode(perSliceMs: number[]): DecodeStats {
  const sorted = [...perSliceMs].sort((a, b) => a - b);
  const total = perSliceMs.reduce((acc, v) => acc + v, 0);
  return {
    n: perSliceMs.length,
    p50_ms: round3(percentile(sorted, 50)),
    p95_ms: round3(percentile(sorted, 95)),
    max_ms: round3(sorted[sorted.length - 1] ?? NaN),
    mean_ms: round3(total / Math.max(1, perSliceMs.length)),
    total_ms: round3(total),
    slices_per_sec: round3((perSliceMs.length / Math.max(1e-9, total)) * 1000),
  };
}

function round3(v: number): number {
  return Math.round(v * 1000) / 1000;
}

/** Report dir: <repo>/runs/web-replay-reports (runs/ is gitignored). */
export function reportDir(): string {
  const here = path.dirname(fileURLToPath(import.meta.url));
  const webRoot = path.resolve(here, "..", "..");
  const repoRoot = path.resolve(webRoot, "..", "..", "..");
  const dir = path.join(repoRoot, "runs", "web-replay-reports");
  fs.mkdirSync(dir, { recursive: true });
  return dir;
}

export function writeReport(name: string, payload: unknown): string {
  const file = path.join(reportDir(), name);
  fs.writeFileSync(file, JSON.stringify(payload, null, 2) + "\n");
  return file;
}
