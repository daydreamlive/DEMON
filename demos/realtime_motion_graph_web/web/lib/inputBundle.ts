"use client";

// Serialize the three audio inputs (input track, timbre ref, structure
// ref) into / out of an exported config file. The OperatorStrip's Export
// dialog opts into this; Import consumes it.
//
// Why this lives separate from lib/config.ts: an RtmgConfig is pure JSON
// describing the engine + sliders, and round-trips losslessly through
// localStorage / the operator-editable public/config.json. Audio inputs
// are big binary blobs that only make sense as a session attachment —
// keeping them in their own module (and their own top-level `inputs`
// key on the exported object) means an export WITHOUT inputs is byte-for-
// byte the old format, and an old DEMON build importing a file WITH
// inputs just ignores the unknown key (mergeConfig drops it).
//
// Wire shape: a clip input embeds its PCM as a base64 16-bit WAV. A
// library-fixture input is server-resolvable by name, so it carries no
// audio. The input track may be either; timbre / structure refs mirror
// the RefSource mode already tracked in usePerformanceStore.
//
// Export carries ONLY the audio — never the encoded latent or ripped
// stems an upload accrues server-side. Those are re-derived on IMPORT:
// when a pod is reachable the clip is pushed back through the normal
// upload pipeline (uploadTrackToServer → server VAE-encode + stem-rip +
// sidecar persist) so it ends up identical to a freshly uploaded track;
// when no pod is connected yet the clip registers in-memory and the
// server derives its sidecar live on first swap (sendSwapSource ships
// PCM and CNN-detects on a miss). Either way the audio is the only thing
// that travels in the file.

import {
  decodeAudioFile,
  loadFixtureAudio,
  uploadTrackToServer,
  type DecodedFixture,
  type StemSourceMode,
} from "@/engine/audio/loadFixture";
import { getConfig } from "@/lib/config";
import { trimAudioBuffer } from "@/lib/audio/trimAudioBuffer";
import { useCustomTracksStore } from "@/store/useCustomTracksStore";
import { usePerformanceStore, type RefSource } from "@/store/usePerformanceStore";
import { useSessionStore } from "@/store/useSessionStore";

// Fallback ref-duration cap when engine.max_source_duration_s is unset.
// Mirrors RefControl / TrackPicker.
const DEFAULT_TRIM_CAP_S = 120;

/** One serialized input. `fixture` is a library track the server can
 *  load by name (no audio on the wire). `clip` embeds the trimmed PCM as
 *  a base64 WAV so an upload survives the round-trip even on a machine
 *  that never had the original file. */
export type SerializedInput =
  | { kind: "fixture"; name: string }
  | {
      kind: "clip";
      name: string;
      sourceMode?: StemSourceMode;
      /** 16-bit PCM WAV, base64-encoded. Sample rate + channel count
       *  ride in the WAV header, so decode re-derives them. */
      wavBase64: string;
    };

/** The three inputs as captured for export. A field is null when that
 *  input axis simply has nothing active. */
export interface SerializedInputs {
  track?: SerializedInput | null;
  timbre?: SerializedInput | null;
  structure?: SerializedInput | null;
}

// ── WAV + base64 codec ─────────────────────────────────────────────────

function writeAscii(view: DataView, offset: number, text: string): void {
  for (let i = 0; i < text.length; i++) {
    view.setUint8(offset + i, text.charCodeAt(i));
  }
}

/** Encode already-interleaved float32 PCM as a 16-bit WAV ArrayBuffer.
 *  Parallels lib/audio/encodeWav.ts (which takes an AudioBuffer); here
 *  the source is the interleaved Float32Array a DecodedFixture already
 *  holds, so we skip the channel de-interleave. */
function encodeWavInterleaved(
  interleaved: Float32Array,
  channels: number,
  sampleRate: number,
): ArrayBuffer {
  const frames = Math.floor(interleaved.length / channels);
  const bytesPerSample = 2;
  const dataLen = frames * channels * bytesPerSample;
  const out = new ArrayBuffer(44 + dataLen);
  const view = new DataView(out);

  writeAscii(view, 0, "RIFF");
  view.setUint32(4, 36 + dataLen, true);
  writeAscii(view, 8, "WAVE");
  writeAscii(view, 12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true); // PCM
  view.setUint16(22, channels, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * channels * bytesPerSample, true);
  view.setUint16(32, channels * bytesPerSample, true);
  view.setUint16(34, 8 * bytesPerSample, true);
  writeAscii(view, 36, "data");
  view.setUint32(40, dataLen, true);

  const pcm = new Int16Array(out, 44, frames * channels);
  const n = frames * channels;
  for (let i = 0; i < n; i++) {
    const s = interleaved[i];
    const c = s < -1 ? -1 : s > 1 ? 1 : s;
    // Asymmetric scaling (mirrors encodeWav.ts) keeps the full negative
    // range without wrapping.
    pcm[i] = c < 0 ? c * 0x8000 : c * 0x7fff;
  }
  return out;
}

function arrayBufferToBase64(buf: ArrayBuffer): string {
  const bytes = new Uint8Array(buf);
  // Chunk to stay clear of the argument-count ceiling on
  // String.fromCharCode for multi-MB clips.
  const chunk = 0x8000;
  let binary = "";
  for (let i = 0; i < bytes.length; i += chunk) {
    binary += String.fromCharCode(...bytes.subarray(i, i + chunk));
  }
  return btoa(binary);
}

function base64ToArrayBuffer(b64: string): ArrayBuffer {
  const binary = atob(b64);
  const len = binary.length;
  const bytes = new Uint8Array(len);
  for (let i = 0; i < len; i++) bytes[i] = binary.charCodeAt(i);
  return bytes.buffer;
}

function encodeClip(
  name: string,
  decoded: DecodedFixture,
  sourceMode?: StemSourceMode,
): SerializedInput {
  const wav = encodeWavInterleaved(
    decoded.interleaved,
    decoded.channels,
    decoded.sampleRate,
  );
  return {
    kind: "clip",
    name,
    ...(sourceMode ? { sourceMode } : {}),
    wavBase64: arrayBufferToBase64(wav),
  };
}

// ── Capture (export) ───────────────────────────────────────────────────

/** Resolve a custom track's decoded PCM for embedding. Freshly uploaded
 *  tracks carry it in the store; a seeded "persisted" upload (added via
 *  addPersisted from useSeedUserUploads) has no in-memory buffer until
 *  it's played, and loadFixtureAudio never writes one back — so fetch +
 *  decode it from the pod's /user_uploads on demand. Keeps the export's
 *  portability promise (the audio always travels, even for an upload the
 *  user only re-selected from a prior session). */
async function resolveDecoded(
  name: string,
  cached: DecodedFixture | undefined,
): Promise<DecodedFixture> {
  return cached ?? (await loadFixtureAudio(name));
}

/** Snapshot the active input track. A custom upload (present in
 *  useCustomTracksStore) embeds its PCM; a library fixture serializes by
 *  name. Returns null when nothing is loaded. */
async function captureTrack(): Promise<SerializedInput | null> {
  const name = usePerformanceStore.getState().fixture;
  if (!name) return null;
  const track = useCustomTracksStore.getState().tracks.get(name);
  if (!track) return { kind: "fixture", name };
  const decoded = await resolveDecoded(name, track.decoded);
  return encodeClip(name, decoded, track.sourceMode);
}

/** Snapshot a timbre / structure RefSource. Clip refs embed PCM pulled
 *  from useCustomTracksStore (fetched from the pod if the upload was
 *  only seeded this session); a clip whose upload record is gone is
 *  dropped (null) rather than exported as an unloadable name. */
async function captureRef(ref: RefSource | null): Promise<SerializedInput | null> {
  if (!ref) return null;
  if (ref.mode === "fixture") return { kind: "fixture", name: ref.name };
  const track = useCustomTracksStore.getState().tracks.get(ref.name);
  if (!track) return null;
  try {
    const decoded = await resolveDecoded(ref.name, track.decoded);
    return encodeClip(ref.name, decoded, track.sourceMode);
  } catch {
    // Upload record present but audio no longer fetchable — drop it
    // rather than ship a ref the importer can't decode.
    return null;
  }
}

/** Build the `inputs` object for an export — captures every active input
 *  (track, timbre ref, structure ref). Async: a seeded upload's PCM may
 *  need a pod fetch before it can be embedded. */
export async function captureInputs(): Promise<SerializedInputs> {
  const perf = usePerformanceStore.getState();
  const [track, timbre, structure] = await Promise.all([
    captureTrack(),
    captureRef(perf.timbreRef),
    captureRef(perf.structRef),
  ]);
  return { track, timbre, structure };
}

/** Whether any input axis is currently active — gates the Export
 *  dialog's "Serialize inputs" checkbox. */
export function anyInputPresent(): boolean {
  const perf = usePerformanceStore.getState();
  return Boolean(perf.fixture || perf.timbreRef || perf.structRef);
}

// ── Apply (import) ─────────────────────────────────────────────────────

/** Decode a clip's embedded WAV and register it as a custom track,
 *  best-effort pre-encoding it through the upload pipeline first.
 *
 *  When a pod is reachable, uploadTrackToServer makes the server
 *  VAE-encode the latent, rip stems, and persist both as sidecars
 *  (mirrors commitUploadedTrack) — the imported clip then behaves
 *  exactly like a freshly uploaded track, and the server owns the
 *  canonical de-duplicated name.
 *
 *  The pre-encode is an OPTIMIZATION, not a requirement: the swap path
 *  (sendSwapSource) and the ref path (sendSetTimbreSource) both ship raw
 *  PCM, and the server encodes + CNN-detects live on a sidecar miss. So
 *  if the upload can't run (no pod connected yet) we fall back to a
 *  plain in-memory register under a client-de-duplicated name — the clip
 *  is still selectable and swappable, its sidecar just gets derived
 *  lazily on first use. This keeps import-before-connect working with no
 *  regression instead of dropping the input. Returns the chosen name. */
async function registerClip(input: {
  name: string;
  sourceMode?: StemSourceMode;
  wavBase64: string;
}): Promise<string> {
  const bytes = base64ToArrayBuffer(input.wavBase64);
  const file = new File([bytes], input.name, { type: "audio/wav" });
  // decodeAudioFile re-applies pool alignment + the browser-memory
  // length ceiling, exactly as a fresh upload would. A decode failure
  // (bad base64 / too short) is genuinely unusable, so it propagates and
  // the caller skips this input.
  const decoded = await decodeAudioFile(file);
  const custom = useCustomTracksStore.getState();
  const sourceMode = input.sourceMode ?? "full";

  // Surface progress on the session status bar: the server encode can
  // take a few seconds and the import toast only fires once we resolve.
  const { setStatus, status } = useSessionStore.getState();
  setStatus(status, `Encoding ${input.name}...`);
  try {
    const uploaded = await uploadTrackToServer(input.name, decoded);
    custom.add(uploaded.name, decoded, file, sourceMode);
    return uploaded.name;
  } catch {
    // No pod reachable (or encode error) — register in-memory; the
    // sidecar is derived live when the clip is first swapped in.
    let chosen = input.name;
    let i = 1;
    while (custom.has(chosen)) chosen = `${input.name} (${i++})`;
    custom.add(chosen, decoded, file, sourceMode);
    return chosen;
  } finally {
    const s = useSessionStore.getState();
    s.setStatus(s.status, "");
  }
}

/** Head-trim a decoded ref to the configured source-duration cap, the
 *  same clamp RefControl applies before shipping a ref over the WS. */
function clampRefDuration(decoded: DecodedFixture, capS: number): DecodedFixture {
  const durS = decoded.frames / decoded.sampleRate;
  if (durS <= capS) return decoded;
  return trimAudioBuffer(decoded, 0, capS);
}

async function applyTrack(input: SerializedInput): Promise<void> {
  const perf = usePerformanceStore.getState();
  if (input.kind === "fixture") {
    perf.setFixture(input.name);
    return;
  }
  const name = await registerClip(input);
  // Writing perf.fixture drives useFixtureSwap: live sessions hot-swap
  // the source; a not-yet-started session picks it up on the next Play
  // (resolveFixtureForConnect reads perf.fixture, loadFixtureAudio finds
  // the clip in the custom-tracks cache).
  perf.setFixture(name);
}

/** Apply a timbre / structure ref. Returns true if it reached the
 *  server. Clip audio is always registered so it shows up in the
 *  dropdowns, but a ref can only take effect against a LIVE session
 *  (the server boots with no overrides and refs aren't part of
 *  SessionConfig) — so when no session is ready we register + report
 *  false, matching how the manual ref pickers behave. */
async function applyRef(
  kind: "timbre" | "structure",
  input: SerializedInput,
): Promise<boolean> {
  const session = useSessionStore.getState();
  const perf = usePerformanceStore.getState();
  const ready = session.status === "ready" && session.remote != null;
  const setRef =
    kind === "timbre" ? perf.setTimbreRef : perf.setStructRef;

  if (input.kind === "fixture") {
    if (!ready || !session.remote) return false;
    if (kind === "timbre") session.remote.sendSetTimbreFixture(input.name);
    else session.remote.sendSetStructureFixture(input.name);
    setRef({ mode: "fixture", name: input.name });
    return true;
  }

  // Clip: register first so it's selectable regardless of session state.
  const name = await registerClip(input);
  if (!ready || !session.remote) return false;
  const decoded = useCustomTracksStore.getState().tracks.get(name)?.decoded;
  if (!decoded) return false;
  const capS = getConfig().engine.max_source_duration_s ?? DEFAULT_TRIM_CAP_S;
  const clamped = clampRefDuration(decoded, capS);
  const ok =
    kind === "timbre"
      ? session.remote.sendSetTimbreSource(
          clamped.interleaved,
          clamped.channels,
          name,
        )
      : session.remote.sendSetStructureSource(
          clamped.interleaved,
          clamped.channels,
          name,
        );
  if (ok) {
    setRef({ mode: "clip", name });
    return true;
  }
  return false;
}

export interface ApplyInputsResult {
  /** Inputs that took full effect. */
  applied: string[];
  /** Refs whose audio was registered but couldn't be sent because no
   *  session is live yet — the caller can hint the user to press Play. */
  needSession: string[];
}

/** Apply a deserialized `inputs` object to the live stores + session.
 *  Best-effort: a malformed or unresolvable input is skipped rather than
 *  aborting the whole import. */
export async function applyInputs(
  inputs: SerializedInputs,
): Promise<ApplyInputsResult> {
  const applied: string[] = [];
  const needSession: string[] = [];

  if (inputs.track) {
    try {
      await applyTrack(inputs.track);
      applied.push("track");
    } catch {
      // Unresolvable track (bad base64 / too short) — leave the current
      // source untouched.
    }
  }

  const refs: Array<["timbre" | "structure", SerializedInput | null | undefined]> = [
    ["timbre", inputs.timbre],
    ["structure", inputs.structure],
  ];
  for (const [kind, ref] of refs) {
    if (!ref) continue;
    try {
      const ok = await applyRef(kind, ref);
      if (ok) applied.push(kind);
      else needSession.push(kind);
    } catch {
      // Skip a ref that fails to decode / send.
    }
  }

  return { applied, needSession };
}

/** True when an imported, parsed object actually carries any input. */
export function hasInputs(inputs: SerializedInputs | null | undefined): boolean {
  if (!inputs) return false;
  return Boolean(inputs.track || inputs.timbre || inputs.structure);
}
