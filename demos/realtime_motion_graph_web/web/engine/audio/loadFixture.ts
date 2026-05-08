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
  const rawFrames = audioBuffer.length;
  const channels = 2;

  // Length normalize: trim to a whole-second multiple. Browsers'
  // decodeAudioData honours the mp3 encoder-padding header and returns
  // a non-integer-second sample count for many real-world files (e.g.
  // a 142.96 s mp3 with 23 ms of priming silence at the head). The
  // server-side VAE encode then computes a latent count off that ragged
  // tail and can underflow into a negative time dim — we saw
  // `Trying to create tensor with negative dimension -1: [1, 128, -1]`
  // on a track with exactly that shape. Whole-second alignment also
  // matches what every TRT engine profile expects (60/120/240 s
  // boundaries) and is what server-side fixture caches assume.
  // Trimming the partial tail loses < 1 s of audio; padding with
  // zeros would risk an audible click.
  const sr = audioBuffer.sampleRate;
  const frames = Math.floor(rawFrames / sr) * sr;
  if (frames < sr) {
    throw new Error(
      `Audio too short — need ≥ 1 second, got ${(rawFrames / sr).toFixed(2)} s.`,
    );
  }

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

  return { interleaved, channels, frames, sampleRate: sr };
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

// Cap user-supplied audio at the smallest TRT engine profile (60 s; see
// acestep/paths.py:_TRT_ENGINE_PROFILES). Larger profiles (120 s, 240 s)
// require materially more VRAM at session init — we've been seeing 1011s
// on cold-loads when the bigger profiles spin up. Operators that want
// longer engines can raise this and the server's websockets.serve
// max_size accordingly.
const MAX_FIXTURE_DURATION_S = 60;

function ensureFitsSwapLimit(decoded: DecodedFixture): void {
  const seconds = decoded.frames / decoded.sampleRate;
  if (seconds <= MAX_FIXTURE_DURATION_S) return;
  throw new Error(
    `Track too long — ${Math.round(seconds)}s (max ${MAX_FIXTURE_DURATION_S}s). Trim it.`,
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
