"use client";

import { useEffect, useRef, useState, type CSSProperties } from "react";

import { decodeAudioFile, listFixtures } from "@/engine/audio/loadFixture";
import { useCustomTracksStore } from "@/store/useCustomTracksStore";
import { usePerformanceStore } from "@/store/usePerformanceStore";
import { useSessionStore } from "@/store/useSessionStore";

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

function UploadIcon() {
  return (
    <svg
      viewBox="0 0 16 16"
      width={14}
      height={14}
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

  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const placardRef = useRef<HTMLButtonElement | null>(null);
  const fanRef = useRef<HTMLDivElement | null>(null);

  // Fetch the catalog only AFTER the queue admits us. The pod proxy at
  // /api/pod/* returns 401 without a session id, so calling pre-admit
  // would just spam 401s in the network tab for no benefit.
  useEffect(() => {
    if (!sessionWsUrl) return;
    void listFixtures()
      .then((names) => {
        setFixtures(names);
        if (!usePerformanceStore.getState().fixture && names[0]) {
          setFixture(names[0]);
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
      const decoded = await decodeAudioFile(file);
      const baseName = file.name;
      let chosen = baseName;
      let i = 1;
      while (useCustomTracksStore.getState().has(chosen)) {
        chosen = `${baseName} (${i++})`;
      }
      addCustomTrack(chosen, decoded);
      setFixture(chosen);
      setStatus(useSessionStore.getState().status, "");
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e);
      setStatus(useSessionStore.getState().status, `Upload failed: ${msg}`);
    } finally {
      setUploading(false);
    }
  }

  if (kiosk) return null;

  const tracks: TrackOption[] = [
    ...fixtures.map((name) => ({ name, kind: "fixture" as const })),
    ...customNames.map((name) => ({ name, kind: "custom" as const })),
  ];
  const displayedName = fixture || (tracks[0]?.name ?? "—");

  return (
    <>
      <button
        ref={placardRef}
        type="button"
        className={`audio-source-crate${open ? " audio-source-crate--open" : ""}`}
        onClick={() => setOpen((v) => !v)}
        aria-haspopup="menu"
        aria-expanded={open}
        aria-label={`Pick audio track. Current: ${displayedName}`}
        title={`Audio source: ${displayedName}`}
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
            title="Upload your own audio track"
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
    </>
  );
}
