"use client";

import { useEffect, useRef, useState, type CSSProperties } from "react";

import {
  decodeAudioFile,
  listFixtures,
  pickDefaultFixture,
  type DecodedFixture,
  type StemSourceMode,
} from "@/engine/audio/loadFixture";
import { useActionGate } from "@/hooks/useActionGate";
import { useSeedUserUploads } from "@/hooks/useSeedUserUploads";
import { commitUploadedTrack } from "@/lib/audio/commitUploadedTrack";
import { useUploadOnboardingHint } from "@/hooks/useUploadOnboardingHint";
import { trimAudioBuffer } from "@/lib/audio/trimAudioBuffer";
import { useConfig } from "@/lib/config";
import { LOCAL_MODE } from "@/lib/runtime";
import { useCustomTracksStore } from "@/store/useCustomTracksStore";
import { usePerformanceStore } from "@/store/usePerformanceStore";
import { useSessionStore } from "@/store/useSessionStore";
import type { TimeSignature } from "@/types/engine";

import { AlmostReadyDialog } from "./AlmostReadyDialog";
import { MicRecorder } from "./MicRecorder";
import { UploadOnboardingHint } from "./UploadOnboardingHint";
import { WaveformTrimDialog } from "./WaveformTrimDialog";

const DEFAULT_TRIM_CAP_S = 120;

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

function NoteIcon({ size = 16 }: { size?: number }) {
  return (
    <svg
      viewBox="0 0 24 24"
      width={size}
      height={size}
      fill="currentColor"
      aria-hidden="true"
    >
      <path d="M12 3v10.55c-.59-.34-1.27-.55-2-.55-2.21 0-4 1.79-4 4s1.79 4 4 4 4-1.79 4-4V7h4V3h-6z" />
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
  // Upload flow is two-stage: ``trimming`` opens the interactive
  // WaveformTrimDialog right after decode, ``pending`` opens the
  // AlmostReadyDialog (source-mode + key) once the user has chosen
  // their trim window. Splitting them keeps the previously-playing
  // song alive through both — neither addCustomTrack nor setFixture
  // runs until the AlmostReadyDialog "Continue" lands.
  const [trimming, setTrimming] = useState<{
    decoded: DecodedFixture;
    fileName: string;
    originalFile: File;
  } | null>(null);
  const [pending, setPending] = useState<{
    decoded: DecodedFixture;
    fileName: string;
    originalFile: File;
  } | null>(null);
  const trimCapS =
    useConfig().engine.max_source_duration_s ?? DEFAULT_TRIM_CAP_S;

  // Auto-collapse: the dock folds to a small play bubble 2s after it's
  // left alone, and expands back on hover / focus. `collapsed` drives
  // the CSS cross-fade; `hovered` is the pointer-over signal.
  const [collapsed, setCollapsed] = useState(false);
  const [hovered, setHovered] = useState(false);

  // First-visit nudge pointing at the Upload button — sequel to the
  // StrengthOnboardingHint. See useUploadOnboardingHint for the
  // reveal/dismiss state machine. We fold its `visible` into
  // `dockActive` below so the dock can't auto-collapse out from
  // under the hint mid-display.
  const uploadHint = useUploadOnboardingHint();

  const fileInputRef = useRef<HTMLInputElement | null>(null);
  // Aborts an in-flight upload when the Almost-Ready dialog is closed.
  const uploadAbortRef = useRef<AbortController | null>(null);
  const placardRef = useRef<HTMLButtonElement | null>(null);
  const uploadBtnRef = useRef<HTMLButtonElement | null>(null);
  const fanRef = useRef<HTMLDivElement | null>(null);
  useSeedUserUploads();
  // Host-supplied gate for high-intent actions (track change + upload).
  // Defaults to allow-all when no provider is mounted; demon-public-demo
  // overrides this to show a sign-up dialog for anonymous users.
  const gate = useActionGate();

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

  // Expand whenever the dock is being used (hovered, or a picker /
  // dialog / upload is in flight); otherwise collapse to the bubble
  // 2s after the last interaction. The timeout is cleared on any
  // re-activation so it only fires once the dock is truly idle.
  const dockActive =
    hovered ||
    open ||
    micOpen ||
    uploading ||
    pending !== null ||
    trimming !== null ||
    uploadHint.visible;
  useEffect(() => {
    if (dockActive) {
      setCollapsed(false);
      return;
    }
    const t = window.setTimeout(() => setCollapsed(true), 2000);
    return () => window.clearTimeout(t);
  }, [dockActive]);

  async function onFilePicked(file: File) {
    const { setStatus } = useSessionStore.getState();
    setUploading(true);
    setStatus(useSessionStore.getState().status, `Loading ${file.name}…`);
    try {
      const decoded = await decodeAudioFile(file);
      const baseName = file.name;
      let chosen = baseName;
      let i = 1;
      while (useCustomTracksStore.getState().has(chosen)) {
        chosen = `${baseName} (${i++})`;
      }
      // Hand the full-length decoded buffer to the trim dialog. We
      // don't touch the custom-tracks store or setFixture() until
      // after BOTH the trim dialog and the AlmostReadyDialog confirm,
      // so cancelling either step keeps the previous song playing.
      setTrimming({ decoded, fileName: chosen, originalFile: file });
      setStatus(useSessionStore.getState().status, "");
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e);
      setStatus(useSessionStore.getState().status, `Upload failed: ${msg}`);
    } finally {
      setUploading(false);
    }
  }

  function onTrimConfirm(startS: number, endS: number) {
    if (!trimming) return;
    const trimmed = trimAudioBuffer(trimming.decoded, startS, endS);
    setPending({
      decoded: trimmed,
      fileName: trimming.fileName,
      originalFile: trimming.originalFile,
    });
    setTrimming(null);
  }

  async function commitPending(
    keyOverride: string | null,
    timeSignatureOverride: TimeSignature | null,
    sourceMode: StemSourceMode,
  ) {
    if (!pending) return;
    const controller = new AbortController();
    uploadAbortRef.current = controller;
    await commitUploadedTrack({
      pending,
      keyOverride,
      timeSignatureOverride,
      sourceMode,
      addCustomTrack,
      setFixture,
      setPending,
      setUploading,
      signal: controller.signal,
    });
  }

  if (kiosk) return null;

  const tracks: TrackOption[] = [
    ...fixtures.map((name) => ({ name, kind: "fixture" as const })),
    ...customNames.map((name) => ({ name, kind: "custom" as const })),
  ];
  const displayedName = fixture || (tracks[0]?.name ?? "—");

  return (
    <>
      <div
        className="audio-source-dock"
        data-collapsed={collapsed ? "" : undefined}
        onMouseEnter={() => setHovered(true)}
        onMouseLeave={() => setHovered(false)}
      >
        {/* Collapsed face — a music-note bubble wreathed by the same
            writhing ribbon ring as the HaloBadge (useRenderLoop finds
            this `.halo-ribbons` canvas by selector and ticks it).
            Hovering it fires the dock's onMouseEnter and expands;
            onClick/onFocus cover touch + keyboard. */}
        <button
          type="button"
          className="audio-source-bubble"
          aria-label="Show now-playing controls"
          onClick={() => setHovered(true)}
          onFocus={() => setHovered(true)}
        >
          <canvas className="halo-ribbons" aria-hidden="true" />
          <NoteIcon size={18} />
        </button>
        <div className="audio-source-dock-body">
        {/* Placard + Upload form the "track controls" pair (the two
            ways to swap input audio — pick from library, or upload).
            Wrapping them in a relative-positioned span lets
            UploadOnboardingHint anchor above their combined width,
            since the hint's copy ("Select another track or upload
            your own") covers BOTH affordances. Mic stays outside the
            wrapper — it's a separate input mode, not a track-swap. */}
        <span className="audio-source-track-controls">
          <UploadOnboardingHint visible={uploadHint.visible} />
          <button
            ref={placardRef}
            type="button"
            className={`audio-source-crate${open ? " audio-source-crate--open" : ""}`}
            onClick={() => {
              // Placard click opens the fan picker — that's the
              // "select another track" half of the hint's copy, so
              // clicking it counts as discovery (matches the upload
              // button's behaviour).
              uploadHint.dismiss();
              setOpen((v) => !v);
            }}
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
          {/* Always-visible upload affordance. Discoverability beats
              the "Upload your own" sleeve hidden inside the fan —
              same handler, same dialog gate, just one click closer. */}
          <button
            ref={uploadBtnRef}
            type="button"
            className="audio-source-upload-btn"
            disabled={uploading}
            onClick={async () => {
              // Clicking dismisses the onboarding hint — the user has
              // clearly found the button, even if they cancel the
              // picker.
              uploadHint.dismiss();
              if (!(await gate("upload"))) return;
              fileInputRef.current?.click();
            }}
            aria-label="Upload your own audio track"
            data-dd-tooltip={uploading ? "Decoding…" : "Upload your own audio track"}
          >
            <UploadIcon size={16} />
            <span className="audio-source-upload-label">
              {uploading ? "Decoding…" : "Upload"}
            </span>
          </button>
        </span>
        <button
          type="button"
          className="audio-source-mic-btn"
          disabled={uploading}
          onClick={async () => {
            if (!(await gate("mic"))) return;
            setMicOpen(true);
          }}
          aria-label="Record audio from microphone"
          data-dd-tooltip="Record audio from your microphone"
        >
          <MicIcon size={16} />
          <span className="audio-source-upload-label">Mic</span>
        </button>
        </div>
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
                  onClick={async () => {
                    // Close the fan optimistically so the gate dialog
                    // (if any) doesn't compete with the open menu for
                    // the user's attention. Gate denial leaves the fan
                    // closed; user can re-open after signing up.
                    setOpen(false);
                    if (!(await gate("track_change"))) return;
                    setFixture(t.name);
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
            onClick={async () => {
              if (!(await gate("upload"))) return;
              fileInputRef.current?.click();
            }}
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

      {trimming && (
        <WaveformTrimDialog
          decoded={trimming.decoded}
          fileName={trimming.fileName}
          capS={trimCapS}
          onConfirm={onTrimConfirm}
          onCancel={() => setTrimming(null)}
        />
      )}
      {pending && (
        <AlmostReadyDialog
          fileName={pending.fileName}
          wasTrimmed={false}
          defaultKey={usePerformanceStore.getState().activeKey}
          defaultTimeSignature={
            usePerformanceStore.getState().activeTimeSignature
          }
          onContinue={({ keyOverride, timeSignatureOverride, sourceMode }) =>
            commitPending(keyOverride, timeSignatureOverride, sourceMode)
          }
          onPickAnother={() => {
            uploadAbortRef.current?.abort();
            setPending(null);
            // Re-open the picker on the next tick so the file input
            // change handler isn't racing the close.
            setTimeout(() => fileInputRef.current?.click(), 0);
          }}
          onClose={() => {
            uploadAbortRef.current?.abort();
            setPending(null);
          }}
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
