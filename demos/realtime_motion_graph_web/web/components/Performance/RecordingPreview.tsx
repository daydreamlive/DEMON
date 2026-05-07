"use client";

import { useRecordingStore } from "@/store/useRecordingStore";
import { encodeWav } from "@/lib/audio/encodeWav";

function isoStamp(): string {
  return new Date().toISOString().replace(/[:.]/g, "-").replace(/Z$/, "");
}

function fmtDuration(ms: number): string {
  const s = Math.max(0, Math.round(ms / 1000));
  const mm = Math.floor(s / 60);
  const ss = s % 60;
  return `${mm}:${ss.toString().padStart(2, "0")}`;
}

type Prepared = { blob: Blob; filename: string; mime: string };

// Re-encode the captured Opus/AAC blob to WAV so users get a DAW-friendly file.
// Falls back silently to the original blob if decoding fails (rare).
async function prepareDownload(
  source: { blob: Blob; ext: string; mime: string },
): Promise<Prepared> {
  const stamp = isoStamp();
  let ctx: AudioContext | null = null;
  try {
    ctx = new AudioContext();
    const buf = await ctx.decodeAudioData(await source.blob.arrayBuffer());
    return {
      blob: encodeWav(buf),
      filename: `daydream-${stamp}.wav`,
      mime: "audio/wav",
    };
  } catch (err) {
    console.warn("[RecordingPreview] WAV encode failed; falling back", err);
    return {
      blob: source.blob,
      filename: `daydream-${stamp}.${source.ext}`,
      mime: source.mime,
    };
  } finally {
    try {
      ctx?.close();
    } catch {}
  }
}

function triggerDownload(blob: Blob, filename: string) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}

export function RecordingPreview() {
  const state = useRecordingStore((s) => s.state);

  if (state.kind !== "preview") return null;

  function dismiss() {
    document.dispatchEvent(new CustomEvent("dd:dismiss-record-preview"));
  }

  async function save() {
    if (state.kind !== "preview") return;
    const prepared = await prepareDownload({
      blob: state.blob,
      ext: state.ext,
      mime: state.mime,
    });
    triggerDownload(prepared.blob, prepared.filename);
  }

  async function share() {
    if (state.kind !== "preview") return;
    const nav = navigator as Navigator & {
      canShare?: (data: ShareData) => boolean;
    };
    const prepared = await prepareDownload({
      blob: state.blob,
      ext: state.ext,
      mime: state.mime,
    });
    try {
      const file = new File([prepared.blob], prepared.filename, {
        type: prepared.mime,
      });
      const data: ShareData = { files: [file], title: "Daydream clip" };
      if (nav.canShare?.(data)) {
        await nav.share(data);
        return;
      }
    } catch (err) {
      // User cancellation throws AbortError — just swallow.
      if ((err as Error).name === "AbortError") return;
      console.warn("[RecordingPreview] share failed", err);
    }
    triggerDownload(prepared.blob, prepared.filename);
  }

  const canShare =
    typeof navigator !== "undefined" &&
    "share" in navigator &&
    "canShare" in navigator;

  return (
    <div className="recording-preview" role="dialog" aria-label="Saved clip">
      <div className="recording-preview-header">
        <span className="recording-preview-title">New clip</span>
        <span className="recording-preview-meta">
          {fmtDuration(state.durationMs)} · WAV
        </span>
      </div>
      <audio
        className="recording-preview-audio"
        src={state.url}
        controls
        preload="metadata"
      />
      <div className="recording-preview-actions">
        <button
          type="button"
          className="recording-preview-btn recording-preview-btn--primary"
          onClick={save}
        >
          Save
        </button>
        {canShare && (
          <button
            type="button"
            className="recording-preview-btn"
            onClick={share}
          >
            Share
          </button>
        )}
        <button
          type="button"
          className="recording-preview-btn recording-preview-btn--ghost"
          onClick={dismiss}
        >
          Discard
        </button>
      </div>
    </div>
  );
}
