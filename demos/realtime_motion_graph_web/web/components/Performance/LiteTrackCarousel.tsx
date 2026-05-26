"use client";

import { useEffect, useRef, useState } from "react";

import {
  decodeAudioFile,
  listFixtures,
  pickDefaultFixture,
  type DecodedFixture,
  type StemSourceMode,
} from "@/engine/audio/loadFixture";
import { generateLegoStemsPreflight } from "@/engine/lego/generateLegoStems";
import { useStartSession } from "@/hooks/useStartSession";
import { LOCAL_MODE } from "@/lib/runtime";
import { useCustomTracksStore } from "@/store/useCustomTracksStore";
import { usePerformanceStore } from "@/store/usePerformanceStore";
import { useSessionStore } from "@/store/useSessionStore";
import type { TimeSignature } from "@/types/engine";
import type { LegoLayerConfig } from "@/types/lego";

import { AlmostReadyDialog } from "./AlmostReadyDialog";

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
  const startSession = useStartSession();

  const [fixtures, setFixtures] = useState<string[]>([]);
  const customNames = useCustomTracksStore((s) => s.names);
  const addCustomTrack = useCustomTracksStore((s) => s.add);

  const [uploading, setUploading] = useState(false);
  const [pending, setPending] = useState<{
    decoded: DecodedFixture;
    fileName: string;
    wasTrimmed: boolean;
    originalFile: File;
  } | null>(null);
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const scrollerRef = useRef<HTMLDivElement | null>(null);

  // Daydream-webapp queue-admit gate: standalone DEMON has no queue
  // (LOCAL_MODE), so we skip the wait there.
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
      const { decoded, wasTrimmed } = await decodeAudioFile(file);
      const baseName = file.name;
      let chosen = baseName;
      let i = 1;
      while (useCustomTracksStore.getState().has(chosen)) {
        chosen = `${baseName} (${i++})`;
      }
      // Defer commit to the AlmostReadyDialog so cancel leaves the
      // previously playing track alone.
      setPending({ decoded, fileName: chosen, wasTrimmed, originalFile: file });
      setStatus(useSessionStore.getState().status, "");
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e);
      setStatus(useSessionStore.getState().status, `Upload failed: ${msg}`);
    } finally {
      setUploading(false);
    }
  }

  async function commitPending(
    keyOverride: string | null,
    timeSignatureOverride: TimeSignature | null,
    sourceMode: StemSourceMode,
    legoLayers: LegoLayerConfig[],
  ) {
    if (!pending) return;
    const { decoded, fileName, originalFile } = pending;
    addCustomTrack(fileName, decoded, originalFile, sourceMode);
    const restartAfterLego =
      legoLayers.length > 0 && useSessionStore.getState().status !== "idle";
    const perf = usePerformanceStore.getState();
    if (keyOverride) {
      perf.setPendingKeyOverride(keyOverride);
      perf.setKey(keyOverride);
    }
    if (timeSignatureOverride) {
      perf.setPendingTimeSignatureOverride(timeSignatureOverride);
      perf.setTimeSignature(timeSignatureOverride);
    }
    if (legoLayers.length > 0) {
      if (restartAfterLego) {
        const session = useSessionStore.getState();
        session.setStatus(session.status, "Stopping playback for LEGO generation…");
        try {
          await session.player?.close();
        } catch {}
        try {
          session.remote?.close();
        } catch {}
        session.reset();
        await new Promise<void>((resolve) => window.setTimeout(resolve, 2000));
      }
      const custom = useCustomTracksStore.getState();
      const session = useSessionStore.getState();
      for (const layer of legoLayers) {
        custom.setLegoPrompt(fileName, layer.track, layer.prompt);
        custom.setLegoStatus(fileName, layer.track, "queued");
      }
      session.setStatus(session.status, "Generating LEGO layers…");
      try {
        const stems = await generateLegoStemsPreflight({
          wsUrl: sessionWsUrl,
          fixtureName: fileName,
          decoded,
          layers: legoLayers,
          onStatus: (track, status, error) => {
            if (track && track !== "__job__") {
              useCustomTracksStore
                .getState()
                .setLegoStatus(
                  fileName,
                  track,
                  status as "queued" | "running" | "ready" | "failed",
                  error,
                );
            }
          },
        });
        custom.setLegoStems(
          fileName,
          stems,
          Object.fromEntries(legoLayers.map((l) => [l.track, l.prompt])),
        );
        useSessionStore.getState().setStatus(useSessionStore.getState().status, "");
      } catch (e) {
        const msg = e instanceof Error ? e.message : String(e);
        for (const layer of legoLayers) {
          custom.setLegoStatus(fileName, layer.track, "failed", msg);
        }
        useSessionStore
          .getState()
          .setStatus(useSessionStore.getState().status, `LEGO failed: ${msg}`);
      }
    }
    setFixture(fileName);
    setPending(null);
    if (restartAfterLego) {
      await startSession();
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
          data-dd-tooltip="Upload audio track"
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

      {pending && (
        <AlmostReadyDialog
          fileName={pending.fileName}
          wasTrimmed={pending.wasTrimmed}
          defaultKey={usePerformanceStore.getState().activeKey}
          defaultTimeSignature={
            usePerformanceStore.getState().activeTimeSignature
          }
          onContinue={({ keyOverride, timeSignatureOverride, sourceMode, legoLayers }) =>
            commitPending(keyOverride, timeSignatureOverride, sourceMode, legoLayers)
          }
          onPickAnother={() => {
            setPending(null);
            setTimeout(() => fileInputRef.current?.click(), 0);
          }}
          onClose={() => setPending(null)}
        />
      )}
    </div>
  );
}
