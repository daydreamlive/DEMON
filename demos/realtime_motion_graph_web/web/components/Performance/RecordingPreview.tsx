"use client";

import { useEffect, useState } from "react";

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

type Prepared = {
  blob: Blob;
  filename: string;
  mime: string;
  kind: "audio" | "video";
};

// Re-encode the captured Opus/AAC blob to WAV so users get a DAW-friendly file.
// Falls back silently to the original blob if decoding fails (rare).
async function prepareAudioDownload(
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
      kind: "audio",
    };
  } catch (err) {
    console.warn("[RecordingPreview] WAV encode failed; falling back", err);
    return {
      blob: source.blob,
      filename: `daydream-${stamp}.${source.ext}`,
      mime: source.mime,
      kind: "audio",
    };
  } finally {
    try {
      ctx?.close();
    } catch {}
  }
}

function prepareVideoDownload(source: {
  blob: Blob;
  ext: string;
  mime: string;
}): Prepared {
  const stamp = isoStamp();
  return {
    blob: source.blob,
    filename: `daydream-${stamp}.${source.ext}`,
    mime: source.mime,
    kind: "video",
  };
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
  // Toggle defaults to ON when a video blob is present; the user can
  // flip to audio-only before clicking Save. State lives in the
  // component because it's UI-local — a fresh preview always starts
  // with the video preferred.
  const hasVideo = state.kind === "preview" && !!state.videoBlob;
  const [includeVideo, setIncludeVideo] = useState(true);
  useEffect(() => {
    setIncludeVideo(hasVideo);
  }, [hasVideo, state.kind === "preview" ? state.url : null]);

  if (state.kind !== "preview") return null;

  function dismiss() {
    document.dispatchEvent(new CustomEvent("dd:dismiss-record-preview"));
  }

  function notifySaved(prepared: Prepared, durationMs: number) {
    // Lets the host webapp persist the clip alongside its own session
    // metadata (see demon-public-demo's saved-sessions feature). Fired
    // after the user-visible Save/Share completes so the listener
    // doesn't race with the download. No-op if nobody is listening.
    document.dispatchEvent(
      new CustomEvent("dd:recording-saved", {
        detail: {
          blob: prepared.blob,
          mime: prepared.mime,
          filename: prepared.filename,
          durationMs,
          kind: prepared.kind,
        },
      }),
    );
  }

  async function prepare(): Promise<Prepared> {
    if (state.kind !== "preview") throw new Error("not in preview state");
    if (includeVideo && state.videoBlob && state.videoMime && state.videoExt) {
      return prepareVideoDownload({
        blob: state.videoBlob,
        ext: state.videoExt,
        mime: state.videoMime,
      });
    }
    return prepareAudioDownload({
      blob: state.blob,
      ext: state.ext,
      mime: state.mime,
    });
  }

  async function save() {
    if (state.kind !== "preview") return;
    const prepared = await prepare();
    triggerDownload(prepared.blob, prepared.filename);
    notifySaved(prepared, state.durationMs);
    dismiss();
  }

  async function share() {
    if (state.kind !== "preview") return;
    const nav = navigator as Navigator & {
      canShare?: (data: ShareData) => boolean;
    };
    const prepared = await prepare();
    try {
      const file = new File([prepared.blob], prepared.filename, {
        type: prepared.mime,
      });
      const data: ShareData = { files: [file], title: "Daydream clip" };
      if (nav.canShare?.(data)) {
        await nav.share(data);
        notifySaved(prepared, state.durationMs);
        dismiss();
        return;
      }
    } catch (err) {
      // User cancellation throws AbortError — leave the preview open so
      // they can try a different action (Save / different share target).
      if ((err as Error).name === "AbortError") return;
      console.warn("[RecordingPreview] share failed", err);
    }
    triggerDownload(prepared.blob, prepared.filename);
    notifySaved(prepared, state.durationMs);
    dismiss();
  }

  const canShare =
    typeof navigator !== "undefined" &&
    "share" in navigator &&
    "canShare" in navigator;

  const metaLabel = includeVideo && hasVideo ? "Video" : "WAV";

  return (
    <div className="recording-preview" role="dialog" aria-label="Saved clip">
      <div className="recording-preview-header">
        <span className="recording-preview-title">New clip</span>
        <span className="recording-preview-meta">
          {fmtDuration(state.durationMs)} · {metaLabel}
        </span>
      </div>
      <div className="recording-preview-media">
        <audio
          className="recording-preview-audio"
          src={state.url}
          controls
          preload="metadata"
        />
        {hasVideo && state.videoUrl && (
          <video
            className="recording-preview-video-chip"
            src={state.videoUrl}
            playsInline
            muted
            loop
            autoPlay
            aria-label="Graph capture preview"
          />
        )}
      </div>
      {hasVideo && (
        <label className="recording-preview-format">
          <input
            type="checkbox"
            checked={includeVideo}
            onChange={(e) => setIncludeVideo(e.target.checked)}
          />
          <span>Include video</span>
        </label>
      )}
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
