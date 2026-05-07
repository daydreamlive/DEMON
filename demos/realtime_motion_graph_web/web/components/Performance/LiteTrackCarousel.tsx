"use client";

import { useEffect, useRef, useState } from "react";

import { decodeAudioFile, listFixtures } from "@/engine/audio/loadFixture";
import { LOCAL_MODE } from "@/lib/runtime";
import { useCustomTracksStore } from "@/store/useCustomTracksStore";
import { usePerformanceStore } from "@/store/usePerformanceStore";
import { useSessionStore } from "@/store/useSessionStore";

// Mobile Lite-controls track picker. A horizontal scroll-snap row of fixture
// chips followed by an "Upload your own" chip. Reuses the same fixture
// catalog, custom-tracks store, and decodeAudioFile path as AudioSourceCrate
// so a track switch from either surface looks identical to useFixtureSwap.

interface TrackOption {
  name: string;
  kind: "fixture" | "custom";
}

function UploadIcon() {
  return (
    <svg
      viewBox="0 0 16 16"
      width={18}
      height={18}
      fill="none"
      stroke="currentColor"
      strokeWidth={1.6}
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

export function LiteTrackCarousel() {
  const fixture = usePerformanceStore((s) => s.fixture);
  const setFixture = usePerformanceStore((s) => s.setFixture);
  const sessionWsUrl = useSessionStore((s) => s.wsUrl);

  const [fixtures, setFixtures] = useState<string[]>([]);
  const customNames = useCustomTracksStore((s) => s.names);
  const addCustomTrack = useCustomTracksStore((s) => s.add);

  const [uploading, setUploading] = useState(false);
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const scrollerRef = useRef<HTMLDivElement | null>(null);

  // Daydream-webapp queue-admit gate: standalone DEMON has no queue
  // (LOCAL_MODE), so we skip the wait there.
  useEffect(() => {
    if (!sessionWsUrl && !LOCAL_MODE) return;
    void listFixtures()
      .then((names) => {
        setFixtures(names);
        if (!usePerformanceStore.getState().fixture && names[0]) {
          setFixture(names[0]);
        }
      })
      .catch(() => setFixtures([]));
  }, [setFixture, sessionWsUrl]);

  // Auto-scroll the current chip into view when fixture changes from
  // elsewhere (e.g. AudioSourceCrate, MobileFullSheet config tab).
  useEffect(() => {
    const root = scrollerRef.current;
    if (!root || !fixture) return;
    const target = root.querySelector<HTMLElement>(
      `[data-track-name="${CSS.escape(fixture)}"]`,
    );
    if (target)
      target.scrollIntoView({
        behavior: "smooth",
        block: "nearest",
        inline: "center",
      });
  }, [fixture]);

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

  const tracks: TrackOption[] = [
    ...fixtures.map((name) => ({ name, kind: "fixture" as const })),
    ...customNames.map((name) => ({ name, kind: "custom" as const })),
  ];

  return (
    <div className="lite-track-carousel" role="tablist" aria-label="Audio track">
      <div ref={scrollerRef} className="lite-track-carousel-scroll">
        {tracks.length === 0 && (
          <div className="lite-track-carousel-empty">Loading…</div>
        )}
        {tracks.map((t) => {
          const isCurrent = t.name === fixture;
          return (
            <button
              key={`${t.kind}:${t.name}`}
              type="button"
              role="tab"
              aria-selected={isCurrent}
              data-track-name={t.name}
              className={[
                "lite-track-chip",
                isCurrent ? "lite-track-chip--current" : "",
                t.kind === "custom" ? "lite-track-chip--custom" : "",
              ]
                .filter(Boolean)
                .join(" ")}
              onClick={() => setFixture(t.name)}
              title={t.name}
            >
              <span className="lite-track-chip-label">{t.name}</span>
            </button>
          );
        })}
        <button
          type="button"
          className="lite-track-chip lite-track-chip--upload"
          disabled={uploading}
          onClick={() => fileInputRef.current?.click()}
          title="Upload audio track"
          aria-label="Upload audio track"
        >
          <span className="lite-track-chip-icon" aria-hidden="true">
            <UploadIcon />
          </span>
          <span className="lite-track-chip-label">
            {uploading ? "Decoding…" : "Upload"}
          </span>
        </button>
      </div>

      <input
        ref={fileInputRef}
        type="file"
        accept="audio/*,.mp3,.wav,.flac,.ogg,.m4a,.aac"
        style={{ display: "none" }}
        onChange={(e) => {
          const file = e.target.files?.[0];
          e.target.value = "";
          if (file) void onFilePicked(file);
        }}
      />
    </div>
  );
}
