// Fetch + decode an audio fixture from the DEMON pod, returning interleaved
// float32 PCM at the audio context sample rate. Uses Web Audio's
// decodeAudioData() so any WAV/MP3/FLAC the pod ships with works without a
// custom decoder.
//
// Also handles user-uploaded tracks: useCustomTracksStore caches their
// decoded buffers; loadFixtureAudio() checks that cache first, so the
// existing Play / fixture-swap paths work unchanged when the active
// fixture is an upload.

import { fetchWithRetry } from "@/engine/fetchWithRetry";
import { defaultWsUrl, podHttp } from "@/engine/podUrl";
import { SAMPLE_RATE } from "@/engine/protocol";
import { getApiKey } from "@/engine/rtmgConfig";

export interface DecodedFixture {
  interleaved: Float32Array;
  channels: number;
  frames: number;
  sampleRate: number;
}

export type StemSourceMode = "full" | "vocals" | "instruments";
export type StemOverlayKind = "vocals" | "instruments";

export interface DecodedStemAssets {
  vocals: DecodedFixture;
  instruments: DecodedFixture;
}

// Server-side latent pool size (1920 * 5 = 9600 samples = 0.2 s at
// 48 kHz). The streaming session and the sidecar precompute both align to this;
// trimming the decoded fixture to the same boundary keeps the runtime
// `samples` count matching the sidecar's recorded `samples` field, so
// `_try_load_sidecar` accepts the cached BPM / key / latents instead
// of falling back to live CNN detection.
const SAMPLE_POOL = 9600;

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

  // Length normalize: trim to a multiple of the server's latent pool
  // (1920 * 5 = 9600 samples = 0.2 s at 48 kHz, mirroring the
  // streaming session's `pool` and scripts/precompute_fixture_sidecars.py's POOL). Browsers'
  // decodeAudioData honours the mp3 encoder-padding header and returns
  // a non-pool-aligned sample count for many real-world files (e.g. a
  // 142.96 s mp3 with 23 ms of priming silence at the head). The
  // server-side VAE encode then computes a latent count off that ragged
  // tail and can underflow into a negative time dim — we saw
  // `Trying to create tensor with negative dimension -1: [1, 128, -1]`
  // on a track with exactly that shape. Pool alignment is what every
  // server step (VAE encode, sidecar samples field, TRT engine
  // selection) actually requires; aligning to whole seconds (the
  // previous rule) was strictly coarser and broke fixtures whose
  // natural pool-aligned length isn't a whole second — e.g. the lo-fi
  // loop is 57.6 s, so the whole-second trim shaved it to 57.0 s and
  // missed the sidecar lookup, falling back to CNN key detection.
  const sr = audioBuffer.sampleRate;
  const frames = Math.floor(rawFrames / SAMPLE_POOL) * SAMPLE_POOL;
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

async function fetchAndDecode(url: string): Promise<DecodedFixture> {
  const res = await fetch(url);
  if (!res.ok) {
    throw new Error(`Audio fetch failed: ${res.status} ${res.statusText}`);
  }
  const bytes = await res.arrayBuffer();
  return decodeArrayBuffer(bytes);
}

async function fetchAndDecodeOptional(
  url: string,
): Promise<DecodedFixture | null> {
  const res = await fetch(url);
  if (res.status === 404) return null;
  if (!res.ok) {
    throw new Error(`Audio fetch failed: ${res.status} ${res.statusText}`);
  }
  const bytes = await res.arrayBuffer();
  return decodeArrayBuffer(bytes);
}

export async function loadFixtureAudio(name: string): Promise<DecodedFixture> {
  // Custom uploads short-circuit the pod fetch — they live in memory only.
  // Imported lazily to avoid a Zustand cycle at module load.
  const { useCustomTracksStore } = await import("@/store/useCustomTracksStore");
  const store = useCustomTracksStore.getState();
  const cached = store.tracks.get(name)?.decoded;
  if (cached) return cached;

  const encoded = encodeURIComponent(name);
  // Only probe /user_uploads for names the store knows as uploads (fresh
  // uploads and seeded-persisted tracks). Built-in fixtures skip it, so the
  // common default-fixture load doesn't eat a guaranteed 404 round-trip.
  if (store.has(name)) {
    const fromUpload = await fetchAndDecodeOptional(
      podHttp(`/user_uploads/${encoded}`),
    );
    if (fromUpload) return fromUpload;
  }
  return fetchAndDecode(podHttp(`/fixtures/${encoded}`));
}

// Hard upper bound on upload length. The interactive trim dialog
// (WaveformTrimDialog) lets the user pick any window of up to
// engine.max_source_duration_s, so tracks longer than this just give
// the user more room to pick FROM — they aren't passed to the engine
// at full length. The cap here exists purely to keep one giant decode
// from exhausting the browser's memory; 10 min of 48 kHz stereo
// Float32 is already ~115 MiB.
export const MAX_UPLOAD_DURATION_S = 600;

/** Decode a user-supplied audio File (mp3, wav, flac, ogg — anything the
 *  browser supports). Used by the upload affordances. Returns the
 *  full-length decoded buffer; the caller (TrackPicker /
 *  AudioSourceCrate) drives the interactive trim dialog and only the
 *  trimmed slice ever reaches the engine. */
export async function decodeAudioFile(file: File): Promise<DecodedFixture> {
  const bytes = await file.arrayBuffer();
  const decoded = await decodeArrayBuffer(bytes);
  const seconds = decoded.frames / decoded.sampleRate;
  if (seconds > MAX_UPLOAD_DURATION_S) {
    throw new Error(
      `Upload too long — max ${Math.round(MAX_UPLOAD_DURATION_S / 60)} min, got ${seconds.toFixed(0)} s.`,
    );
  }
  return decoded;
}

/** Fetch the pod's whitelist of fixture names.
 *
 *  Uses fetchWithRetry so a backend still coming up (502 from the Next
 *  dev proxy) is transparently waited on instead of leaving the fixture
 *  list empty until the operator refreshes. */
export async function listFixtures(): Promise<string[]> {
  const res = await fetchWithRetry(podHttp("/api/fixtures"));
  if (!res.ok) throw new Error(`Fixture list failed: ${res.status}`);
  const json = (await res.json()) as string[];
  return json;
}

export async function listUserUploads(): Promise<string[]> {
  const res = await fetchWithRetry(podHttp("/api/user_uploads"));
  if (!res.ok) throw new Error(`User upload list failed: ${res.status}`);
  const json = (await res.json()) as string[];
  return json;
}

export interface UploadTrackResult {
  name: string;
  bpm: number;
  key: string;
  timeSignature: string;
  durationS: number;
  samples: number;
}

export interface UploadTrackOptions {
  key?: string | null;
  timeSignature?: string | null;
  /** Aborts the in-flight upload (e.g. the user closed the dialog). */
  signal?: AbortSignal;
  /** Overall deadline; the server can accept the socket but never reply
   *  (encode hang / unsupported duration), which would hang the caller. */
  timeoutMs?: number;
}

async function resolveUploadWsUrl(): Promise<string> {
  let url = defaultWsUrl();
  const { useSessionStore } = await import("@/store/useSessionStore");
  const serverUrl = useSessionStore.getState().wsUrl;
  if (serverUrl) url = serverUrl;
  const apiKey = getApiKey();
  if (apiKey) {
    const sep = url.includes("?") ? "&" : "?";
    url = `${url}${sep}apiKey=${encodeURIComponent(apiKey)}`;
  }
  return url;
}

// Returns Uint8Array<ArrayBuffer> (not the wider Uint8Array<ArrayBufferLike>
// that's the default in TS 5.7+'s libdom) so ws.send accepts it without
// a cast — BufferSource excludes SharedArrayBuffer-backed views, and
// the `new Uint8Array(byteLength)` allocation here is always ArrayBuffer-
// backed. Without the narrow return type, consumers compiling on TS
// 5.7+ hit "Uint8Array<ArrayBufferLike> not assignable to BufferSource".
function pcmFrame(decoded: DecodedFixture): Uint8Array<ArrayBuffer> {
  const hdr = new ArrayBuffer(8);
  const dv = new DataView(hdr);
  dv.setUint32(0, decoded.channels, true);
  dv.setUint32(4, decoded.frames, true);
  const pcm = new Uint8Array(
    decoded.interleaved.buffer,
    decoded.interleaved.byteOffset,
    decoded.interleaved.byteLength,
  );
  const combined = new Uint8Array(hdr.byteLength + pcm.byteLength);
  combined.set(new Uint8Array(hdr), 0);
  combined.set(pcm, hdr.byteLength);
  return combined;
}

export async function uploadTrackToServer(
  name: string,
  decoded: DecodedFixture,
  options: UploadTrackOptions = {},
): Promise<UploadTrackResult> {
  const wsUrl = await resolveUploadWsUrl();
  const timeoutMs = options.timeoutMs ?? 120_000;
  return new Promise((resolve, reject) => {
    if (options.signal?.aborted) {
      reject(new DOMException("upload aborted", "AbortError"));
      return;
    }
    let settled = false;
    const ws = new WebSocket(wsUrl);
    ws.binaryType = "arraybuffer";

    const finish = (fn: () => void) => {
      if (settled) return;
      settled = true;
      clearTimeout(timer);
      options.signal?.removeEventListener("abort", onAbort);
      try {
        ws.close();
      } catch {}
      fn();
    };

    // A pod can accept the upload socket but never reply (encode hang,
    // missing/old handler) and never close it — without a deadline the caller
    // hangs forever (the "Encoding…" / stuck-dialog bug). Fail recoverably.
    const timer = setTimeout(() => {
      finish(() =>
        reject(new Error("Upload timed out — the server didn't respond.")),
      );
    }, timeoutMs);

    const onAbort = () =>
      finish(() => reject(new DOMException("upload aborted", "AbortError")));
    options.signal?.addEventListener("abort", onAbort);

    ws.onopen = () => {
      try {
        ws.send(JSON.stringify({
          type: "upload_track",
          name,
          ...(options.key ? { key: options.key } : {}),
          ...(options.timeSignature
            ? { time_signature: options.timeSignature }
            : {}),
        }));
        ws.send(pcmFrame(decoded));
      } catch (e) {
        finish(() => reject(e instanceof Error ? e : new Error(String(e))));
      }
    };

    ws.onmessage = (ev) => {
      if (typeof ev.data !== "string") return;
      let msg: Record<string, unknown>;
      try {
        msg = JSON.parse(ev.data) as Record<string, unknown>;
      } catch {
        return;
      }
      if (msg.type === "upload_ok") {
        finish(() =>
          resolve({
            name: String(msg.name),
            bpm: Number(msg.bpm ?? 0),
            key: String(msg.key ?? ""),
            timeSignature: String(msg.time_signature ?? "4"),
            durationS: Number(msg.duration_s ?? 0),
            samples: Number(msg.samples ?? 0),
          }),
        );
      } else if (msg.type === "upload_failed") {
        finish(() => reject(new Error(String(msg.error ?? "upload failed"))));
      }
    };

    ws.onerror = () => {
      finish(() => reject(new Error("upload connection failed")));
    };

    ws.onclose = (ev) => {
      if (settled) return;
      finish(() =>
        reject(new Error(ev.reason || `upload connection closed (${ev.code})`)),
      );
    };
  });
}

// Preferred default the UI picks when no fixture is yet selected. Falls
// back to names[0] if the catalog doesn't contain it (e.g. removed from
// KNOWN_FIXTURES upstream).
export const PREFERRED_DEFAULT_FIXTURE = "low_fi_Gm_loop_60s_gnm.wav";

export function pickDefaultFixture(names: readonly string[]): string {
  if (names.includes(PREFERRED_DEFAULT_FIXTURE)) return PREFERRED_DEFAULT_FIXTURE;
  return names[0] ?? "";
}
