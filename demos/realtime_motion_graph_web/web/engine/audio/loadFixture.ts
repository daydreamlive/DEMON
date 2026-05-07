// Fetch + decode an audio fixture from the DEMON pod, returning interleaved
// float32 PCM at the audio context sample rate. Uses Web Audio's
// decodeAudioData() so any WAV/MP3/FLAC the pod ships with works without a
// custom decoder.
//
// Also handles user-uploaded tracks: useCustomTracksStore caches their
// decoded buffers; loadFixtureAudio() checks that cache first, so the
// existing Play / fixture-swap paths work unchanged when the active
// fixture is an upload.

import { podHttp } from "@/engine/podUrl";
import { SAMPLE_RATE } from "@/engine/protocol";

export interface DecodedFixture {
  interleaved: Float32Array;
  channels: number;
  frames: number;
  sampleRate: number;
}

/** Decoder runs on a short-lived real AudioContext at SAMPLE_RATE so the
 *  PCM matches what the pod's pipeline expects. We previously used
 *  OfflineAudioContext here; recent Chromium builds occasionally never
 *  resolve OfflineAudioContext.decodeAudioData(), leaving the UI stuck on
 *  "Loading fixture…". A regular AudioContext is the documented path and
 *  is safe because Play is a user gesture. */
async function decodeArrayBuffer(bytes: ArrayBuffer): Promise<DecodedFixture> {
  const Ctx: typeof AudioContext =
    (window.AudioContext as typeof AudioContext) ||
    ((window as unknown as { webkitAudioContext: typeof AudioContext })
      .webkitAudioContext as typeof AudioContext);
  const tmpCtx = new Ctx({ sampleRate: SAMPLE_RATE });
  let audioBuffer: AudioBuffer;
  try {
    // decodeAudioData mutates the input ArrayBuffer in some browsers, so
    // we pass a copy via .slice(0).
    audioBuffer = await tmpCtx.decodeAudioData(bytes.slice(0));
  } finally {
    void tmpCtx.close();
  }

  // Always emit exactly 2 channels: mono → duplicate, stereo → pass
  // through, >2 → take front L/R only (Web Audio puts front-L=0,
  // front-R=1 for any layout).
  const srcChannels = audioBuffer.numberOfChannels;
  const frames = audioBuffer.length;
  const channels = 2;
  const interleaved = new Float32Array(frames * channels);

  if (srcChannels === 1) {
    const m = audioBuffer.getChannelData(0);
    for (let i = 0; i < frames; i++) {
      const v = m[i];
      interleaved[i * 2] = v;
      interleaved[i * 2 + 1] = v;
    }
  } else {
    const l = audioBuffer.getChannelData(0);
    const r = audioBuffer.getChannelData(1);
    for (let i = 0; i < frames; i++) {
      interleaved[i * 2] = l[i];
      interleaved[i * 2 + 1] = r[i];
    }
  }

  return { interleaved, channels, frames, sampleRate: audioBuffer.sampleRate };
}

export async function loadFixtureAudio(name: string): Promise<DecodedFixture> {
  // Custom uploads short-circuit the pod fetch — they live in memory only.
  // Imported lazily to avoid a Zustand cycle at module load.
  const { useCustomTracksStore } = await import("@/store/useCustomTracksStore");
  const cached = useCustomTracksStore.getState().decoded.get(name);
  if (cached) return cached;

  const url = podHttp(`/fixtures/${encodeURIComponent(name)}`);
  const res = await fetch(url);
  if (!res.ok) {
    throw new Error(`Fixture fetch failed: ${res.status} ${res.statusText}`);
  }
  const bytes = await res.arrayBuffer();
  return decodeArrayBuffer(bytes);
}

// DEMON's WebSocket server caps incoming frames at ~50 MiB
// (websockets.serve(max_size=…)). swap_source sends an 8-byte header +
// interleaved Float32 PCM, so the decoded buffer must fit under that cap.
// Practical limit at 48 kHz stereo Float32 (8 B/frame) ≈ 130 s of audio.
const MAX_SWAP_SOURCE_BYTES = 50 * 1024 * 1024;
const SWAP_SOURCE_HEADER_BYTES = 8;

function ensureFitsSwapLimit(decoded: DecodedFixture): void {
  const totalBytes = decoded.interleaved.byteLength + SWAP_SOURCE_HEADER_BYTES;
  if (totalBytes <= MAX_SWAP_SOURCE_BYTES) return;
  const seconds = decoded.frames / decoded.sampleRate;
  const bytesPerSecond = decoded.sampleRate * decoded.channels * 4;
  const maxSeconds = Math.floor(
    (MAX_SWAP_SOURCE_BYTES - SWAP_SOURCE_HEADER_BYTES) / bytesPerSecond,
  );
  throw new Error(
    `Track too long for the engine — ${Math.round(seconds)} s decoded ` +
      `(${(totalBytes / 1024 / 1024).toFixed(1)} MB), engine accepts ` +
      `up to ~${maxSeconds} s. Trim the file or pick a shorter clip.`,
  );
}

/** Decode a user-supplied audio File (mp3, wav, flac, ogg — anything the
 *  browser supports). Used by the OperatorStrip upload affordance.
 *  Throws if the decoded PCM would exceed DEMON's WS frame cap (~50 MiB,
 *  ≈ 130 s at 48 kHz stereo). */
export async function decodeAudioFile(file: File): Promise<DecodedFixture> {
  const bytes = await file.arrayBuffer();
  const decoded = await decodeArrayBuffer(bytes);
  ensureFitsSwapLimit(decoded);
  return decoded;
}

/** Fetch the pod's whitelist of fixture names. */
export async function listFixtures(): Promise<string[]> {
  const res = await fetch(podHttp("/api/fixtures"));
  if (!res.ok) throw new Error(`Fixture list failed: ${res.status}`);
  const json = (await res.json()) as string[];
  return json;
}
