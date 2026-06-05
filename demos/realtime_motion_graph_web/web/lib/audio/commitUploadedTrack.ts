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

/** What the AlmostReadyDialog needs to know after the await resolves:
 *  success unmounts the dialog (parent cleared `pending`); an error
 *  leaves it mounted so the dialog can show the message inline and let
 *  the user retry; an abort means the user already closed it. */
export type UploadOutcome =
  | { ok: true }
  | { ok: false; aborted: true }
  | { ok: false; aborted?: false; error: string };

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
}: CommitUploadedTrackArgs): Promise<UploadOutcome> {
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
    return { ok: true };
  } catch (e) {
    // User closed the dialog mid-encode → just clear the status, no error.
    if (signal?.aborted || (e instanceof DOMException && e.name === "AbortError")) {
      setStatus(useSessionStore.getState().status, "");
      return { ok: false, aborted: true };
    }
    const msg = e instanceof Error ? e.message : String(e);
    setStatus(useSessionStore.getState().status, `Upload failed: ${msg}`);
    // `pending` is intentionally left in place so the user can retry, and
    // the dialog surfaces `error` inline (the status bar is hidden behind
    // the modal, so returning it is the only way the user sees it).
    return { ok: false, error: msg };
  } finally {
    setUploading(false);
  }
}
