"use client";

import { useEffect, useRef, useState } from "react";

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

// Inline track picker — a dropdown + upload button living at the top of
// the CORE tab so power users don't have to leave the panel to swap
// input audio. Mirrors AudioSourceCrate's upload flow (decode →
// AlmostReadyDialog gate → addCustomTrack + setFixture), just packaged
// as a compact form row instead of the floating placard.

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

export function TrackPicker() {
  const fixture = usePerformanceStore((s) => s.fixture);
  const setFixture = usePerformanceStore((s) => s.setFixture);
  const sessionWsUrl = useSessionStore((s) => s.wsUrl);

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
      perf.setKey(keyOverride);
    }
    if (timeSignatureOverride) {
      perf.setPendingTimeSignatureOverride(timeSignatureOverride);
      perf.setTimeSignature(timeSignatureOverride);
    }
    setFixture(fileName);
    setPending(null);
  }

  return (
    <div className="track-picker">
      <label className="track-picker-label" htmlFor="core-fixture-select">
        Track
      </label>
      <div className="track-picker-row">
        <select
          id="core-fixture-select"
          className="fixture-select"
          title="Audio source — pick a track or one of your uploaded tracks"
          value={fixture}
          onChange={(e) => setFixture(e.target.value)}
        >
          {fixtures.length === 0 && customNames.length === 0 && <option>—</option>}
          {customNames.length > 0 && (
            <optgroup label="Your uploads">
              {customNames.map((n) => (
                <option key={`u:${n}`} value={n}>
                  {n}
                </option>
              ))}
            </optgroup>
          )}
          {fixtures.length > 0 && (
            <optgroup label="Tracks">
              {fixtures.map((f) => (
                <option key={f} value={f}>
                  {f}
                </option>
              ))}
            </optgroup>
          )}
        </select>
        <button
          type="button"
          className="track-picker-upload"
          data-dd-tooltip={uploading ? "Decoding…" : "Upload audio track"}
          aria-label="Upload audio track"
          disabled={uploading}
          onClick={() => fileInputRef.current?.click()}
        >
          <UploadIcon size={14} />
          <span className="track-picker-upload-label">
            {uploading ? "Decoding…" : "Upload"}
          </span>
        </button>
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
            setTimeout(() => fileInputRef.current?.click(), 0);
          }}
          onClose={() => setPending(null)}
        />
      )}
    </div>
  );
}
