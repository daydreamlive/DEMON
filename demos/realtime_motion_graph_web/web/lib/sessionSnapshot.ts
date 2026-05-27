"use client";

import {
  decodeAudioFile,
  type DecodedFixture,
  type DecodedStemAssets,
  type StemOverlayKind,
} from "@/engine/audio/loadFixture";
import { trimAudioBuffer } from "@/lib/audio/trimAudioBuffer";
import { applyConfig, captureRtmgConfig, type RtmgConfig } from "@/lib/config";
import {
  getSessionAudioAsset,
  getSessionUploadFile,
  hasSessionAudioAsset,
  hasSessionUploadFile,
  putSessionAudioAsset,
  putSessionUploadFile,
} from "@/lib/sessionAudioAssets";
import {
  useCustomTracksStore,
  type CustomTrackAssetMetadata,
  type HydratedCustomTrack,
} from "@/store/useCustomTracksStore";
import { usePerformanceStore } from "@/store/usePerformanceStore";
import { useSessionStore } from "@/store/useSessionStore";
import { useStemOverlayStore } from "@/store/useStemOverlayStore";

export const SESSION_SNAPSHOT_VERSION = 1;

export type SessionCompletenessStatus =
  | "complete"
  | "saving-assets"
  | "stems-not-ready"
  | "missing-audio-asset"
  | "unsupported-version"
  | "quota-error";

export interface SessionCompleteness {
  status: SessionCompletenessStatus;
  message: string;
  missingAssetIds: string[];
}

export interface SessionSnapshotV1 {
  version: typeof SESSION_SNAPSHOT_VERSION;
  capturedAt: number;
  config: RtmgConfig;
  fixture: string;
  customTracks: CustomTrackAssetMetadata[];
  stemOverlay: {
    enabled: Record<StemOverlayKind, boolean>;
    volumes: Record<StemOverlayKind, number>;
  };
}

export type SessionSnapshot = SessionSnapshotV1;

export type RestoreProgress =
  | "validating"
  | "loading-assets"
  | "restoring-controls"
  | "restoring-stems"
  | "ready";

export function captureSessionSnapshot(): SessionSnapshot {
  const perf = usePerformanceStore.getState();
  const stemOverlay = useStemOverlayStore.getState();
  return {
    version: SESSION_SNAPSHOT_VERSION,
    capturedAt: Date.now(),
    config: captureRtmgConfig(),
    fixture: perf.fixture,
    customTracks: useCustomTracksStore.getState().exportMetadata(),
    stemOverlay: {
      enabled: { ...stemOverlay.enabled },
      volumes: { ...stemOverlay.volumes },
    },
  };
}

export function validateSessionSnapshotShape(
  value: unknown,
): value is SessionSnapshot {
  if (typeof value !== "object" || value === null || Array.isArray(value)) {
    return false;
  }
  const snapshot = value as Partial<SessionSnapshot>;
  return (
    snapshot.version === SESSION_SNAPSHOT_VERSION &&
    typeof snapshot.capturedAt === "number" &&
    typeof snapshot.fixture === "string" &&
    typeof snapshot.config === "object" &&
    snapshot.config !== null &&
    Array.isArray(snapshot.customTracks) &&
    typeof snapshot.stemOverlay === "object" &&
    snapshot.stemOverlay !== null
  );
}

export async function checkSessionCompleteness(
  snapshot: unknown,
): Promise<SessionCompleteness> {
  if (!validateSessionSnapshotShape(snapshot)) {
    return {
      status: "unsupported-version",
      message: "This saved session uses an unsupported format.",
      missingAssetIds: [],
    };
  }

  const missing: string[] = [];
  for (const track of snapshot.customTracks) {
    if (
      !(await hasSessionAudioAsset(track.assetId)) &&
      !(await hasSessionUploadFile(track.assetId))
    ) {
      missing.push(track.assetId);
    }
    if (track.sourceMode !== "full" && !track.stemAssetIds) {
      return {
        status: "stems-not-ready",
        message: "Wait for stems to finish, then save again.",
        missingAssetIds: [],
      };
    }
    if (track.stemAssetIds) {
      for (const assetId of Object.values(track.stemAssetIds)) {
        if (!(await hasSessionAudioAsset(assetId))) missing.push(assetId);
      }
    }
  }

  if (missing.length > 0) {
    return {
      status: "missing-audio-asset",
      message:
        missing.length === 1
          ? "Uploaded source missing on this device."
          : `${missing.length} uploaded sources missing on this device.`,
      missingAssetIds: missing,
    };
  }

  return {
    status: "complete",
    message:
      snapshot.customTracks.length > 0
        ? "All local audio assets saved."
        : "Controls saved.",
    missingAssetIds: [],
  };
}

export async function persistSessionSnapshotAssets(
  snapshot: SessionSnapshot,
): Promise<SessionCompleteness> {
  const custom = useCustomTracksStore.getState();
  try {
    for (const trackMeta of snapshot.customTracks) {
      const track = custom.tracks.get(trackMeta.name);
      if (!track) {
        return {
          status: "missing-audio-asset",
          message: `Uploaded source missing: ${trackMeta.name}`,
          missingAssetIds: [trackMeta.assetId],
        };
      }
      await putSessionAudioAsset(trackMeta.assetId, track.decoded);
      if (track.originalFile) {
        await putSessionUploadFile(trackMeta.assetId, track.originalFile);
      }
      if (track.stems && track.stemAssetIds) {
        await putSessionAudioAsset(track.stemAssetIds.vocals, track.stems.vocals);
        await putSessionAudioAsset(
          track.stemAssetIds.instruments,
          track.stems.instruments,
        );
      } else if (trackMeta.sourceMode !== "full") {
        return {
          status: "stems-not-ready",
          message: "Wait for stem extraction to finish before saving this session.",
          missingAssetIds: [],
        };
      }
    }
  } catch (e) {
    const msg = e instanceof Error ? e.message : String(e);
    return {
      status: "quota-error",
      message: `Could not save uploaded audio: ${msg}`,
      missingAssetIds: snapshot.customTracks.map((t) => t.assetId),
    };
  }

  return checkSessionCompleteness(snapshot);
}

async function decodeSavedUploadForMetadata(
  metadata: CustomTrackAssetMetadata,
  file: File,
): Promise<DecodedFixture> {
  const decoded = await decodeAudioFile(file);
  const duration = decoded.frames / decoded.sampleRate;
  if (
    typeof metadata.trimStartS === "number" &&
    typeof metadata.trimEndS === "number" &&
    metadata.trimEndS > metadata.trimStartS &&
    duration + 0.05 >= metadata.trimEndS
  ) {
    return trimAudioBuffer(decoded, metadata.trimStartS, metadata.trimEndS);
  }
  if (decoded.frames !== metadata.frames) {
    const expectedS = metadata.frames / metadata.sampleRate;
    if (duration > expectedS + 0.05) {
      return trimAudioBuffer(decoded, 0, expectedS);
    }
  }
  return decoded;
}

async function hydrateCustomTracks(
  snapshot: SessionSnapshot,
): Promise<SessionCompleteness> {
  const hydrated: HydratedCustomTrack[] = [];
  const missing: string[] = [];
  for (const metadata of snapshot.customTracks) {
    let source = await getSessionAudioAsset(metadata.assetId);
    const originalFile = await getSessionUploadFile(metadata.assetId);
    if (!source) {
      if (!originalFile) {
        missing.push(metadata.assetId);
        continue;
      }
      source = await decodeSavedUploadForMetadata(metadata, originalFile);
      await putSessionAudioAsset(metadata.assetId, source);
    }
    let stems: DecodedStemAssets | undefined;
    if (metadata.stemAssetIds) {
      const vocals = await getSessionAudioAsset(metadata.stemAssetIds.vocals);
      const instruments = await getSessionAudioAsset(
        metadata.stemAssetIds.instruments,
      );
      if (!vocals || !instruments) {
        missing.push(
          ...[
            !vocals ? metadata.stemAssetIds.vocals : null,
            !instruments ? metadata.stemAssetIds.instruments : null,
          ].filter((id): id is string => id !== null),
        );
        continue;
      }
      stems = { vocals, instruments };
    }
    if (metadata.sourceMode !== "full" && !stems) {
      return {
        status: "stems-not-ready",
        message: "Saved stems are missing. Re-upload and save after stems finish.",
        missingAssetIds: [],
      };
    }
    const decoded =
      metadata.sourceMode !== "full" && stems
        ? stems[metadata.sourceMode]
        : source;
    const hasSavedStemAssets = Boolean(stems);
    const skipStemExtraction =
      metadata.sourceMode !== "full" || hasSavedStemAssets;
    hydrated.push({
      metadata,
      decoded,
      ...(originalFile ? { originalFile } : {}),
      ...(stems ? { stems } : {}),
      // Avoid re-running MelFormer when saved stems already exist or the
      // selected inference PCM came from a saved stem. Legacy/full saves
      // without cached stems are allowed to re-rip on play for overlays.
      ...(skipStemExtraction ? { skipStemExtraction } : {}),
    });
  }

  if (missing.length > 0) {
    return {
      status: "missing-audio-asset",
      message: "Uploaded source missing on this device.",
      missingAssetIds: missing,
    };
  }

  useCustomTracksStore.getState().hydrateSavedTracks(hydrated);
  return {
    status: "complete",
    message: "Uploaded audio loaded.",
    missingAssetIds: [],
  };
}

function flashSessionStatus(message: string): void {
  const session = useSessionStore.getState();
  session.setStatus(session.status, message);
}

export async function applySessionSnapshot(
  snapshot: unknown,
  opts: {
    onProgress?: (progress: RestoreProgress) => void;
  } = {},
): Promise<SessionCompleteness> {
  opts.onProgress?.("validating");
  const completeness = await checkSessionCompleteness(snapshot);
  if (completeness.status !== "complete") return completeness;
  if (!validateSessionSnapshotShape(snapshot)) return completeness;

  opts.onProgress?.("loading-assets");
  flashSessionStatus("Loading uploaded audio...");
  const hydrated = await hydrateCustomTracks(snapshot);
  if (hydrated.status !== "complete") return hydrated;

  opts.onProgress?.("restoring-controls");
  flashSessionStatus("Restoring controls...");
  const perf = usePerformanceStore.getState();
  perf.setSkipNextDenoiseGate(true);
  applyConfig(snapshot.config);
  usePerformanceStore.getState().setFixture(snapshot.fixture);

  const stem = useStemOverlayStore.getState();
  (Object.keys(snapshot.stemOverlay.enabled) as StemOverlayKind[]).forEach((kind) => {
    stem.setEnabled(kind, snapshot.stemOverlay.enabled[kind]);
  });
  (Object.keys(snapshot.stemOverlay.volumes) as StemOverlayKind[]).forEach((kind) => {
    stem.setVolume(kind, snapshot.stemOverlay.volumes[kind]);
  });

  opts.onProgress?.("restoring-stems");
  if (
    snapshot.customTracks.some(
      (track) => track.name === snapshot.fixture && track.stemAssetIds,
    )
  ) {
    flashSessionStatus("Restored saved stem audio.");
  }

  opts.onProgress?.("ready");
  window.setTimeout(() => {
    const session = useSessionStore.getState();
    if (
      session.message === "Restoring controls..." ||
      session.message === "Loading uploaded audio..." ||
      session.message === "Restored saved stem audio."
    ) {
      session.setStatus(session.status, "");
    }
  }, 1800);

  return {
    status: "complete",
    message: "Session restored.",
    missingAssetIds: [],
  };
}

export async function relinkSessionAudioAsset(
  snapshot: SessionSnapshot,
  assetId: string,
  file: File,
): Promise<DecodedFixture> {
  const metadata = snapshot.customTracks.find((track) => track.assetId === assetId);
  if (!metadata) throw new Error("Saved session does not reference this audio asset.");

  const asset = await decodeSavedUploadForMetadata(metadata, file);

  await putSessionAudioAsset(assetId, asset);
  await putSessionUploadFile(assetId, file);
  return asset;
}

export function sessionSnapshotSignature(snapshot: SessionSnapshot): string {
  return JSON.stringify({
    version: snapshot.version,
    config: snapshot.config,
    fixture: snapshot.fixture,
    customTracks: snapshot.customTracks,
    stemOverlay: snapshot.stemOverlay,
  });
}
