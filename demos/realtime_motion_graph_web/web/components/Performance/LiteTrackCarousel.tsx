"use client";

import { useEffect, useRef, useState } from "react";

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
import { trimAudioBuffer } from "@/lib/audio/trimAudioBuffer";
import { useConfig } from "@/lib/config";
import { LOCAL_MODE } from "@/lib/runtime";
import { useCustomTracksStore } from "@/store/useCustomTracksStore";
import { usePerformanceStore } from "@/store/usePerformanceStore";
import { useSessionStore } from "@/store/useSessionStore";
import type { TimeSignature } from "@/types/engine";

import { AlmostReadyDialog } from "./AlmostReadyDialog";
import { MicRecorder } from "./MicRecorder";
import { WaveformTrimDialog } from "./WaveformTrimDialog";

const DEFAULT_TRIM_CAP_S = 120;

// Mobile Lite-controls track picker. Renders as a native <select> so
// the OS picker (iOS wheel / Android dialog) opens on tap — the user
// gets a one-thumb friendly chooser with built-in scrolling, search,
// and accessibility instead of the horizontal scroll-snap row of chips
// the old carousel forced. The library + uploads + record-from-mic +
// the "Upload your own…" sentinel all live in the same select, with
// optgroups to keep them visually grouped. Reuses the same fixture
// catalog, custom-tracks store, decodeAudioFile path, and MicRecorder
// modal as AudioSourceCrate so a track switch from either surface
// looks identical to useFixtureSwap.
//
// (Component name kept as ``LiteTrackCarousel`` to preserve the import
// site in ``LiteControls.tsx``; the carousel itself is gone.)

// Sentinel values: selecting them triggers a side-effect instead of a
// fixture swap. The select stays controlled on ``fixture``, so React
// re-renders and snaps the visible selection back to the active track
// immediately while the modal opens in front.
const UPLOAD_VALUE = "__upload__";
const MIC_VALUE = "__mic__";

export function LiteTrackCarousel() {
  const fixture = usePerformanceStore((s) => s.fixture);
  const setFixture = usePerformanceStore((s) => s.setFixture);
  const sessionWsUrl = useSessionStore((s) => s.wsUrl);

  const [fixtures, setFixtures] = useState<string[]>([]);
  const customNames = useCustomTracksStore((s) => s.names);
  const addCustomTrack = useCustomTracksStore((s) => s.add);

  const [uploading, setUploading] = useState(false);
  const [micOpen, setMicOpen] = useState(false);
  // Mirrors AudioSourceCrate's two-stage flow: trim first, then the
  // AlmostReadyDialog. Keeps the previously playing track alive
  // through both steps.
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
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  // Aborts an in-flight upload when the Almost-Ready dialog is closed.
  const uploadAbortRef = useRef<AbortController | null>(null);
  useSeedUserUploads();
  // Host-supplied gate for track-change + upload (see useActionGate).
  // Default is allow-all; demon-public-demo overrides to require sign-up.
  const gate = useActionGate();

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

  // Native <select> handles its own current-option-in-view, so the
  // scroll-into-view dance the carousel needed is gone.

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
      // Interactive trim first; ``onTrimConfirm`` below slices the
      // window and hands the trimmed buffer to the AlmostReadyDialog
      // for the source-mode + key step.
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
    return commitUploadedTrack({
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

  const totalTracks = fixtures.length + customNames.length;

  return (
    <div className="lite-track-picker">
      <label
        htmlFor="lite-track-select"
        className="lite-track-picker-label"
      >
        Track
      </label>
      <div className="lite-track-picker-field">
        <select
          id="lite-track-select"
          className="lite-track-picker-select"
          value={fixture ?? ""}
          disabled={uploading}
          aria-label="Audio track"
          onChange={async (e) => {
            const v = e.target.value;
            if (v === UPLOAD_VALUE) {
              // Don't actually setFixture("__upload__"); the controlled
              // ``value=fixture`` re-render snaps the visible selection
              // back to the active track while the file picker opens
              // in front.
              if (!(await gate("upload"))) return;
              fileInputRef.current?.click();
              return;
            }
            if (v === MIC_VALUE) {
              // Same sentinel pattern as UPLOAD_VALUE — open the
              // MicRecorder modal; the controlled select snaps back
              // to the active track on the next render.
              if (!(await gate("mic"))) return;
              setMicOpen(true);
              return;
            }
            if (v) {
              if (!(await gate("track_change"))) return;
              setFixture(v);
            }
          }}
        >
          {totalTracks === 0 && (
            <option value="" disabled>
              Loading…
            </option>
          )}
          {fixtures.length > 0 && (
            <optgroup label="Library">
              {fixtures.map((name) => (
                <option key={`fixture:${name}`} value={name}>
                  {name}
                </option>
              ))}
            </optgroup>
          )}
          {customNames.length > 0 && (
            <optgroup label="Your tracks">
              {customNames.map((name) => (
                <option key={`custom:${name}`} value={name}>
                  {name}
                </option>
              ))}
            </optgroup>
          )}
          <option value={MIC_VALUE}>
            ●  Record from microphone…
          </option>
          <option value={UPLOAD_VALUE}>
            {uploading ? "Decoding…" : "↑  Upload your own…"}
          </option>
        </select>
        <span className="lite-track-picker-chevron" aria-hidden="true">
          ▾
        </span>
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
            // Recorded clip flows through the same decode →
            // AlmostReadyDialog → addCustomTrack pipeline as a file
            // upload. MicRecorder emits a wav File so onFilePicked's
            // decodeAudioFile handles it identically.
            void onFilePicked(file);
          }}
          onClose={() => setMicOpen(false)}
        />
      )}
    </div>
  );
}
