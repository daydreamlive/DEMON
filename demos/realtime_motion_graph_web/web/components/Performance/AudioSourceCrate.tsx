"use client";

import { useEffect, useRef, useState, type CSSProperties } from "react";

import {
  decodeAudioFile,
  listFixtures,
  pickDefaultFixture,
  type DecodedFixture,
} from "@/engine/audio/loadFixture";
import { LOCAL_MODE } from "@/lib/runtime";
import { useCustomTracksStore } from "@/store/useCustomTracksStore";
import { usePerformanceStore } from "@/store/usePerformanceStore";
import { useSessionStore } from "@/store/useSessionStore";
import type { TimeSignature } from "@/types/engine";

import { AlmostReadyDialog } from "./AlmostReadyDialog";
import { MicRecorder } from "./MicRecorder";

// Bottom-left counterpart to the bottom-right turntable. Replaces the plain
// fixture <select> as the primary track-picker — that dropdown hid the fact
// that multiple preloaded tracks exist. The placard is the entire affordance:
// always shows the current track at rest, click to fan out a column of
// sleeves with a permanent "upload your own" sleeve pinned at the bottom.
//
// All track switching (built-in or upload) goes through usePerformanceStore's
// setFixture(); useFixtureSwap (subscribed in <Performance/>) handles the
// mid-session crossfade. The advanced-controls strip carries an unchanged
// dropdown + upload icon for power users.

interface TrackOption {
  name: string;
  kind: "fixture" | "custom";
}

function UploadIcon({ size = 14 }: { size?: number }) {
  return (
    <svg
      viewBox="0 0 16 16"
      width={size}
      height={size}
      fill="none"
      stroke="currentColor"
      strokeWidth={1.4}
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      <path d="M8 10V2" />
      <path d="M4.5 5.5L8 2l3.5 3.5" />
      <path d="M2.5 10v3a1 1 0 0 0 1 1h9a1 1 0 0 0 1-1v-3" />
    </svg>
  );
}

function MicIcon({ size = 14 }: { size?: number }) {
  return (
    <svg
      viewBox="0 0 16 16"
      width={size}
      height={size}
      fill="none"
      stroke="currentColor"
      strokeWidth={1.4}
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      <rect x="6" y="2" width="4" height="8" rx="2" />
      <path d="M3.5 8.5a4.5 4.5 0 0 0 9 0" />
      <path d="M8 13v1.5" />
      <path d="M6 14.5h4" />
    </svg>
  );
}

export function AudioSourceCrate() {
  const fixture = usePerformanceStore((s) => s.fixture);
  const setFixture = usePerformanceStore((s) => s.setFixture);
  const kiosk = usePerformanceStore((s) => s.kiosk);
  const sessionWsUrl = useSessionStore((s) => s.wsUrl);

  const [fixtures, setFixtures] = useState<string[]>([]);
  const customNames = useCustomTracksStore((s) => s.names);
  const addCustomTrack = useCustomTracksStore((s) => s.add);

  const [open, setOpen] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [micOpen, setMicOpen] = useState(false);
  // After decode, the dialog gates the actual fixture swap so the user
  // can confirm the trim (if any) and pick a key before playback
  // crossfades.
  const [pending, setPending] = useState<{
    decoded: DecodedFixture;
    fileName: string;
    wasTrimmed: boolean;
    originalFile: File;
  } | null>(null);

  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const placardRef = useRef<HTMLButtonElement | null>(null);
  const uploadBtnRef = useRef<HTMLButtonElement | null>(null);
  const fanRef = useRef<HTMLDivElement | null>(null);

  // Daydream-webapp queue-admit gate: /api/pod/* returns 401 pre-admit,
  // so prod waits for wsUrl before fetching. Standalone DEMON has no
  // queue (LOCAL_MODE), so we skip the wait there.
  useEffect(() => {
    if (!sessionWsUrl && !LOCAL_MODE) return;
    void listFixtures()
      .then((names) => {
        setFixtures(names);
        const def = pickDefaultFixture(names);
        if (!usePerformanceStore.getState().fixture && def) {
          setFixture(def);
        }
      })
      .catch(() => setFixtures([]));
  }, [setFixture, sessionWsUrl]);

  useEffect(() => {
    if (!open) return;
    function onPointer(e: PointerEvent) {
      const t = e.target as Node | null;
      if (!t) return;
      if (placardRef.current?.contains(t)) return;
      if (uploadBtnRef.current?.contains(t)) return;
      if (fanRef.current?.contains(t)) return;
      setOpen(false);
    }
    function onKey(e: KeyboardEvent) {
      if (e.key === "Escape") setOpen(false);
    }
    document.addEventListener("pointerdown", onPointer);
    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("pointerdown", onPointer);
      document.removeEventListener("keydown", onKey);
    };
  }, [open]);

  async function onFilePicked(file: File) {
    const { setStatus } = useSessionStore.getState();
    setUploading(true);
    setStatus(useSessionStore.getState().status, `Loading ${file.name}…`);
    try {
      const { decoded, wasTrimmed } = await decodeAudioFile(file);
      const baseName = file.name;
      let chosen = baseName;
      let i = 1;
      while (useCustomTracksStore.getState().has(chosen)) {
        chosen = `${baseName} (${i++})`;
      }
      // Hand the decoded buffer + trim flag to the dialog. We DON'T add
      // it to the custom-tracks store yet, and we DON'T setFixture()
      // yet — that all happens on Continue so the previously playing
      // song keeps playing if the user cancels.
      setPending({ decoded, fileName: chosen, wasTrimmed, originalFile: file });
      setStatus(useSessionStore.getState().status, "");
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e);
      setStatus(useSessionStore.getState().status, `Upload failed: ${msg}`);
    } finally {
      setUploading(false);
    }
  }

  function commitPending(
    keyOverride: string | null,
    timeSignatureOverride: TimeSignature | null,
  ) {
    if (!pending) return;
    const { decoded, fileName, originalFile } = pending;
    addCustomTrack(fileName, decoded, originalFile);
    const perf = usePerformanceStore.getState();
    if (keyOverride) {
      perf.setPendingKeyOverride(keyOverride);
      // Pre-set activeKey so the swap_source send carries the override
      // as the model hint — useFixtureSwap reads activeKey when calling
      // remote.sendSwapSource().
      perf.setKey(keyOverride);
    }
    if (timeSignatureOverride) {
      // Mirror the keyscale override: stash a one-shot value so the
      // swap_ready handler in useFixtureSwap can re-apply it (and tell
      // the server) even though the server's own resolver won't have
      // it during the in-flight swap. Pre-set activeTimeSignature so
      // the same UI control reflects the choice immediately.
      perf.setPendingTimeSignatureOverride(timeSignatureOverride);
      perf.setTimeSignature(timeSignatureOverride);
    }
    setFixture(fileName);
    setPending(null);
  }

  if (kiosk) return null;

  const tracks: TrackOption[] = [
    ...fixtures.map((name) => ({ name, kind: "fixture" as const })),
    ...customNames.map((name) => ({ name, kind: "custom" as const })),
  ];
  const displayedName = fixture || (tracks[0]?.name ?? "—");

  return (
    <>
      <div className="audio-source-dock">
        <button
          ref={placardRef}
          type="button"
          className={`audio-source-crate${open ? " audio-source-crate--open" : ""}`}
          onClick={() => setOpen((v) => !v)}
          aria-haspopup="menu"
          aria-expanded={open}
          aria-label={`Pick audio track. Current: ${displayedName}`}
          data-dd-tooltip={`Audio source: ${displayedName}`}
        >
          <span className="audio-source-marquee-rows">
            <span className="audio-source-marquee-label">
              {open ? "Pick a track" : "▶ Now playing"}
            </span>
            <span className="audio-source-marquee-name" title={displayedName}>
              {open ? "or upload your own" : displayedName}
            </span>
          </span>
          <span className="audio-source-crate-caret" aria-hidden="true">
            <svg viewBox="0 0 10 10" width={10} height={10} fill="none" stroke="currentColor" strokeWidth={1.6} strokeLinecap="round" strokeLinejoin="round">
              <path d="M2 6.5L5 3.5L8 6.5" />
            </svg>
          </span>
        </button>
        {/* Always-visible upload affordance. Discoverability beats the
            "Upload your own" sleeve hidden inside the fan — same handler,
            same dialog gate, just one click closer. */}
        <button
          ref={uploadBtnRef}
          type="button"
          className="audio-source-upload-btn"
          disabled={uploading}
          onClick={() => fileInputRef.current?.click()}
          aria-label="Upload your own audio track"
          data-dd-tooltip={uploading ? "Decoding…" : "Upload your own audio track"}
        >
          <UploadIcon size={16} />
          <span className="audio-source-upload-label">
            {uploading ? "Decoding…" : "Upload"}
          </span>
        </button>
        <button
          type="button"
          className="audio-source-mic-btn"
          disabled={uploading}
          onClick={() => setMicOpen(true)}
          aria-label="Record audio from microphone"
          data-dd-tooltip="Record audio from your microphone"
        >
          <MicIcon size={16} />
          <span className="audio-source-upload-label">Mic</span>
        </button>
      </div>

      {open && (
        <div ref={fanRef} className="audio-source-fan" role="menu">
          <div className="audio-source-fan-scroll">
            {tracks.length === 0 && (
              <div className="audio-source-fan-empty">Loading tracks…</div>
            )}
            {tracks.map((t, i) => {
              const isCurrent = t.name === fixture;
              return (
                <button
                  key={`${t.kind}:${t.name}`}
                  type="button"
                  role="menuitem"
                  className={[
                    "audio-source-sleeve",
                    isCurrent ? "audio-source-sleeve--current" : "",
                    t.kind === "custom" ? "audio-source-sleeve--custom" : "",
                  ]
                    .filter(Boolean)
                    .join(" ")}
                  style={{ "--idx": i } as CSSProperties}
                  onClick={() => {
                    setFixture(t.name);
                    setOpen(false);
                  }}
                  title={t.name}
                >
                  <span className="audio-source-sleeve-art" aria-hidden="true" />
                  <span className="audio-source-sleeve-label">{t.name}</span>
                </button>
              );
            })}
          </div>
          {/* Upload sleeve is pinned outside the scroll region so it stays
              visible regardless of fixture count. Always rendered. */}
          <button
            type="button"
            role="menuitem"
            className="audio-source-sleeve audio-source-sleeve--upload"
            disabled={uploading}
            onClick={() => fileInputRef.current?.click()}
            data-dd-tooltip="Upload your own audio track"
          >
            <span
              className="audio-source-sleeve-art audio-source-sleeve-art--upload"
              aria-hidden="true"
            >
              <UploadIcon />
            </span>
            <span className="audio-source-sleeve-label">
              {uploading ? "Decoding…" : "Upload your own"}
            </span>
          </button>
        </div>
      )}

      <input
        ref={fileInputRef}
        type="file"
        accept="audio/*,.mp3,.wav,.flac,.ogg,.m4a,.aac"
        style={{ display: "none" }}
        onChange={(e) => {
          const file = e.target.files?.[0];
          // Reset so picking the same file twice still fires onChange.
          e.target.value = "";
          if (file) void onFilePicked(file);
        }}
      />

      {pending && (
        <AlmostReadyDialog
          fileName={pending.fileName}
          wasTrimmed={pending.wasTrimmed}
          defaultKey={usePerformanceStore.getState().activeKey}
          defaultTimeSignature={
            usePerformanceStore.getState().activeTimeSignature
          }
          onContinue={({ keyOverride, timeSignatureOverride }) =>
            commitPending(keyOverride, timeSignatureOverride)
          }
          onPickAnother={() => {
            setPending(null);
            // Re-open the picker on the next tick so the file input
            // change handler isn't racing the close.
            setTimeout(() => fileInputRef.current?.click(), 0);
          }}
          onClose={() => setPending(null)}
        />
      )}

      {micOpen && (
        <MicRecorder
          onComplete={(file) => {
            setMicOpen(false);
            void onFilePicked(file);
          }}
          onClose={() => setMicOpen(false)}
        />
      )}
    </>
  );
}
