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
import { RefSelect } from "./RefSelect";
import { WaveformTrimDialog } from "./WaveformTrimDialog";

const DEFAULT_TRIM_CAP_S = 120;

// Inline track picker — a caption + dropdown + sibling upload icon
// living at the top of the CORE tab so power users don't have to leave
// the panel to swap input audio. Shares the RefSelect chrome with the
// timbre + structure reference pickers immediately below it, so the
// three controls (track / timbre ref / structure ref) read as a
// coherent row of source-material selectors.
//
// Upload flow mirrors AudioSourceCrate: decode locally → gate with
// AlmostReadyDialog → addCustomTrack + setFixture on Continue.

export function TrackPicker() {
  const fixture = usePerformanceStore((s) => s.fixture);
  const setFixture = usePerformanceStore((s) => s.setFixture);
  const sessionWsUrl = useSessionStore((s) => s.wsUrl);

  const [fixtures, setFixtures] = useState<string[]>([]);
  const customNames = useCustomTracksStore((s) => s.names);
  const addCustomTrack = useCustomTracksStore((s) => s.add);

  const [uploading, setUploading] = useState(false);
  // Upload flow is two-stage: first the interactive trim dialog
  // (``trimming``), then the AlmostReadyDialog (``pending``). The
  // trim step is always on — short uploads still get the dialog so
  // users can pick a section of their 30 s loop instead of the head.
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
      const decoded = await decodeAudioFile(file);
      const baseName = file.name;
      let chosen = baseName;
      let i = 1;
      while (useCustomTracksStore.getState().has(chosen)) {
        chosen = `${baseName} (${i++})`;
      }
      // First stop: the user picks the trim window. The pre-trim
      // DecodedFixture stays in memory only as long as this dialog
      // is open; ``onTrimConfirm`` below slices it down to the
      // selected window and drops the original.
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

  return (
    <>
      <RefSelect
        label="input track"
        value={fixture || ""}
        pinned={[]}
        groups={[
          {
            label: "Library",
            options: fixtures.map((n) => ({ value: n, label: n })),
          },
          {
            label: "Your tracks",
            options: customNames.map((n) => ({ value: n, label: n })),
          },
        ]}
        onSelect={async (v) => {
          if (!(await gate("track_change"))) return;
          setFixture(v);
        }}
        disabled={uploading}
        ariaLabel="Input track"
        onUpload={async () => {
          if (!(await gate("upload"))) return;
          fileInputRef.current?.click();
        }}
        uploadLabel={uploading ? "Decoding…" : "Upload audio track"}
        tooltip="The input song the model is processing. Pick a built-in fixture, one of your uploads, or click the upload icon to drop in a new file. New uploads decode locally, then the Almost-Ready dialog gates the swap so the previous track keeps playing if you cancel."
      />
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
    </>
  );
}
