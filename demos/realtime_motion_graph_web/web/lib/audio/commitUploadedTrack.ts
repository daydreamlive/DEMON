"use client";

import {
  uploadTrackToServer,
  type DecodedFixture,
  type StemSourceMode,
} from "@/engine/audio/loadFixture";
import { usePerformanceStore } from "@/store/usePerformanceStore";
import { useSessionStore } from "@/store/useSessionStore";
import type { TimeSignature } from "@/types/engine";

export interface PendingTrackUpload {
  decoded: DecodedFixture;
  fileName: string;
  originalFile: File;
}

interface CommitUploadedTrackArgs {
  pending: PendingTrackUpload;
  keyOverride: string | null;
  timeSignatureOverride: TimeSignature | null;
  sourceMode: StemSourceMode;
  addCustomTrack: (
    name: string,
    decoded: DecodedFixture,
    file?: File,
    sourceMode?: StemSourceMode,
    persisted?: boolean,
  ) => void;
  setFixture: (name: string) => void;
  setPending: (pending: PendingTrackUpload | null) => void;
  setUploading: (uploading: boolean) => void;
  /** Aborts the upload when the dialog is closed mid-encode. */
  signal?: AbortSignal;
}

export async function commitUploadedTrack({
  pending,
  keyOverride,
  timeSignatureOverride,
  sourceMode,
  addCustomTrack,
  setFixture,
  setPending,
  setUploading,
  signal,
}: CommitUploadedTrackArgs): Promise<void> {
  const { decoded, fileName, originalFile } = pending;
  // Keep `pending` set until the upload actually succeeds: encoding can
  // fail (bad audio, server/network), and clearing it up front would throw
  // away the user's trimmed selection with no way to retry.
  setUploading(true);
  const { setStatus } = useSessionStore.getState();
  setStatus(useSessionStore.getState().status, `Encoding ${fileName}...`);
  try {
    const uploaded = await uploadTrackToServer(fileName, decoded, {
      key: keyOverride,
      timeSignature: timeSignatureOverride,
      signal,
    });
    // The server persisted audio + sidecars + stems to disk before
    // replying upload_ok, so swaps to this track can load by name.
    addCustomTrack(uploaded.name, decoded, originalFile, sourceMode, true);
    const perf = usePerformanceStore.getState();
    if (keyOverride) {
      perf.setPendingKeyOverride(keyOverride);
      perf.setKey(keyOverride);
    }
    if (timeSignatureOverride) {
      perf.setPendingTimeSignatureOverride(timeSignatureOverride);
      perf.setTimeSignature(timeSignatureOverride);
    }
    setFixture(uploaded.name);
    setPending(null);
    setStatus(useSessionStore.getState().status, "");
  } catch (e) {
    // User closed the dialog mid-encode → just clear the status, no error.
    if (signal?.aborted || (e instanceof DOMException && e.name === "AbortError")) {
      setStatus(useSessionStore.getState().status, "");
    } else {
      const msg = e instanceof Error ? e.message : String(e);
      setStatus(useSessionStore.getState().status, `Upload failed: ${msg}`);
      // `pending` is intentionally left in place so the user can retry.
    }
  } finally {
    setUploading(false);
  }
}
