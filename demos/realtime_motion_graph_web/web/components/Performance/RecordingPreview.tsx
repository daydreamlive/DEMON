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

type SaveKind = "audio" | "video";

type Prepared = {
  blob: Blob;
  filename: string;
  mime: string;
  kind: SaveKind;
};

// Re-encode the captured Opus/AAC blob to WAV so users get a DAW-friendly file.
// Falls back silently to the original blob if decoding fails (rare).
async function prepareAudioDownload(
  source: { blob: Blob; ext: string; mime: string },
  stamp: string,
): Promise<Prepared> {
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

function prepareVideoDownload(
  source: { blob: Blob; ext: string; mime: string },
  stamp: string,
): Prepared {
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

// Three download modes:
//   "video" — muxed webm/mp4 (audio + graph video in one container)
//   "audio" — WAV only (existing behavior; works on every browser)
//   "both"  — fire both downloads in sequence (browser may surface a
//             "this site is downloading multiple files" permission
//             prompt on first use; that's a one-time accept)
type Format = "video" | "audio" | "both";

export function RecordingPreview() {
  const state = useRecordingStore((s) => s.state);
  const hasVideo = state.kind === "preview" && !!state.videoBlob;
  // Default to "video" when video was captured (so the user lands on
  // the richer-by-default option), "audio" otherwise. Reset whenever a
  // new preview lands.
  const [format, setFormat] = useState<Format>(hasVideo ? "video" : "audio");
  useEffect(() => {
    setFormat(hasVideo ? "video" : "audio");
  }, [hasVideo, state.kind === "preview" ? state.url : null]);

  if (state.kind !== "preview") return null;

  function dismiss() {
    document.dispatchEvent(new CustomEvent("dd:dismiss-record-preview"));
  }

  function notifySaved(prepared: Prepared, durationMs: number) {
    // Lets the host webapp persist the clip alongside its own session
    // metadata. Fires once per saved file — for "both", two events
    // fire back-to-back so the listener can persist each independently.
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

  async function preparedItems(): Promise<Prepared[]> {
    if (state.kind !== "preview") return [];
    const stamp = isoStamp();
    const out: Prepared[] = [];
    if (format === "video" || format === "both") {
      if (state.videoBlob && state.videoMime && state.videoExt) {
        out.push(
          prepareVideoDownload(
            {
              blob: state.videoBlob,
              ext: state.videoExt,
              mime: state.videoMime,
            },
            stamp,
          ),
        );
      }
    }
    if (format === "audio" || format === "both") {
      out.push(
        await prepareAudioDownload(
          { blob: state.blob, ext: state.ext, mime: state.mime },
          stamp,
        ),
      );
    }
    return out;
  }

  async function save() {
    if (state.kind !== "preview") return;
    const items = await preparedItems();
    for (const p of items) {
      triggerDownload(p.blob, p.filename);
      notifySaved(p, state.durationMs);
    }
    dismiss();
  }

  async function share() {
    if (state.kind !== "preview") return;
    const nav = navigator as Navigator & {
      canShare?: (data: ShareData) => boolean;
    };
    // Share only meaningfully handles a single file at a time on most
    // platforms; for "both" we fall through to downloading both.
    if (format !== "both") {
      const items = await preparedItems();
      const prepared = items[0];
      if (!prepared) return;
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
        // User cancellation throws AbortError — leave the preview open
        // so they can try a different action (Save / different share
        // target).
        if ((err as Error).name === "AbortError") return;
        console.warn("[RecordingPreview] share failed", err);
      }
      triggerDownload(prepared.blob, prepared.filename);
      notifySaved(prepared, state.durationMs);
      dismiss();
      return;
    }
    // "both" path — defer to save (two downloads).
    void save();
  }

  const canShare =
    typeof navigator !== "undefined" &&
    "share" in navigator &&
    "canShare" in navigator;

  const metaLabel =
    format === "video"
      ? "Video"
      : format === "both"
        ? "Video + WAV"
        : "WAV";

  return (
    <div className="recording-preview" role="dialog" aria-label="Saved clip">
      <div className="recording-preview-header">
        <span className="recording-preview-title">New clip</span>
        <span className="recording-preview-meta">
          {fmtDuration(state.durationMs)} · {metaLabel}
        </span>
      </div>
      <div className="recording-preview-media">
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
        <audio
          className="recording-preview-audio"
          src={state.url}
          controls
          preload="metadata"
        />
      </div>
      {hasVideo && (
        <div
          className="recording-preview-format"
          role="radiogroup"
          aria-label="Download format"
        >
          <FormatButton
            value="video"
            active={format}
            onPick={setFormat}
            label="Video + audio"
            sublabel="One file"
          />
          <FormatButton
            value="audio"
            active={format}
            onPick={setFormat}
            label="Audio only"
            sublabel="WAV"
          />
          <FormatButton
            value="both"
            active={format}
            onPick={setFormat}
            label="Both"
            sublabel="Two files"
          />
        </div>
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

function FormatButton({
  value,
  active,
  onPick,
  label,
  sublabel,
}: {
  value: Format;
  active: Format;
  onPick: (v: Format) => void;
  label: string;
  sublabel: string;
}) {
  const isActive = active === value;
  return (
    <button
      type="button"
      role="radio"
      aria-checked={isActive}
      className={`recording-preview-format-btn${
        isActive ? " recording-preview-format-btn--active" : ""
      }`}
      onClick={() => onPick(value)}
    >
      <span className="recording-preview-format-btn-label">{label}</span>
      <span className="recording-preview-format-btn-sublabel">{sublabel}</span>
    </button>
  );
}
