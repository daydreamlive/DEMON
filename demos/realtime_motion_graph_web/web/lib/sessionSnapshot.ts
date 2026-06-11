"use client";

import {
  decodeAudioFile,
  loadFixtureAudio,
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
import {
  captureDeckStateSnapshot,
  createDefaultDeckStateSnapshot,
  DECK_STATE_SNAPSHOT_VERSION,
  useDeckStore,
  type DeckId,
  type DeckStateSnapshot,
} from "@/store/useDeckStore";
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
  deckState?: DeckStateSnapshot;
  stemOverlay: {
    enabled: Record<StemOverlayKind, boolean>;
    volumes: Record<StemOverlayKind, number>;
  };
}

export type SessionSnapshot = SessionSnapshotV1;

export function captureSessionSnapshot(): SessionSnapshot {
  const perf = usePerformanceStore.getState();
  const stemOverlay = useStemOverlayStore.getState();
  return {
    version: SESSION_SNAPSHOT_VERSION,
    capturedAt: Date.now(),
    config: captureRtmgConfig(),
    fixture: perf.fixture,
    customTracks: useCustomTracksStore.getState().exportMetadata(),
    deckState: captureDeckStateSnapshot(),
    stemOverlay: {
      enabled: { ...stemOverlay.enabled },
      volumes: { ...stemOverlay.volumes },
    },
  };
}

function validateDeckStateSnapshotShape(value: unknown): value is DeckStateSnapshot {
  if (typeof value !== "object" || value === null || Array.isArray(value)) {
    return false;
  }
  const snapshot = value as Partial<DeckStateSnapshot>;
  return (
    snapshot.version === DECK_STATE_SNAPSHOT_VERSION &&
    Array.isArray(snapshot.deckIds) &&
    typeof snapshot.decks === "object" &&
    snapshot.decks !== null &&
    typeof snapshot.crossfade === "number" &&
    typeof snapshot.inferenceEnabled === "boolean"
  );
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
    (typeof snapshot.deckState === "undefined" ||
      validateDeckStateSnapshotShape(snapshot.deckState)) &&
    typeof snapshot.stemOverlay === "object" &&
    snapshot.stemOverlay !== null
  );
}

function sourceAssetIdsFor(
  metadata: CustomTrackAssetMetadata,
): Partial<Record<"full" | "vocals" | "instruments", string>> & { full: string } {
  return {
    full: metadata.sourceAssetIds?.full ?? metadata.assetId,
    ...(metadata.sourceAssetIds?.vocals || metadata.stemAssetIds?.vocals
      ? {
          vocals: metadata.sourceAssetIds?.vocals ?? metadata.stemAssetIds?.vocals,
        }
      : {}),
    ...(metadata.sourceAssetIds?.instruments || metadata.stemAssetIds?.instruments
      ? {
          instruments:
            metadata.sourceAssetIds?.instruments ??
            metadata.stemAssetIds?.instruments,
        }
      : {}),
  };
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
    const sourceAssetIds = sourceAssetIdsFor(track);
    if (
      !(await hasSessionAudioAsset(sourceAssetIds.full)) &&
      !(await hasSessionUploadFile(sourceAssetIds.full))
    ) {
      missing.push(sourceAssetIds.full);
    }
    const selectedAssetId = sourceAssetIds[track.sourceMode];
    if (track.sourceMode !== "full" && !selectedAssetId) {
      return {
        status: "stems-not-ready",
        message: "Wait for stems to finish, then save again.",
        missingAssetIds: [],
      };
    }
    for (const mode of ["vocals", "instruments"] as const) {
      const assetId = sourceAssetIds[mode];
      if (assetId && !(await hasSessionAudioAsset(assetId))) missing.push(assetId);
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
      const sourceAssetIds = sourceAssetIdsFor(trackMeta);
      // Persist the full mix under the track's assetId (not the active stem),
      // so the saved "source" asset is always the complete upload regardless
      // of which inference source is currently selected.
      const fullPcm = track?.full ?? track?.decoded;
      if (!track || !fullPcm) {
        return {
          status: "missing-audio-asset",
          message: `Uploaded source missing: ${trackMeta.name}`,
          missingAssetIds: [sourceAssetIds.full],
        };
      }
      await putSessionAudioAsset(sourceAssetIds.full, fullPcm);
      if (track.originalFile) {
        await putSessionUploadFile(sourceAssetIds.full, track.originalFile);
      }
      if (track.stems && sourceAssetIds.vocals && sourceAssetIds.instruments) {
        await putSessionAudioAsset(sourceAssetIds.vocals, track.stems.vocals);
        await putSessionAudioAsset(
          sourceAssetIds.instruments,
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
    const sourceAssetIds = sourceAssetIdsFor(metadata);
    let source = await getSessionAudioAsset(sourceAssetIds.full);
    const originalFile = await getSessionUploadFile(sourceAssetIds.full);
    if (!source) {
      if (!originalFile) {
        missing.push(sourceAssetIds.full);
        continue;
      }
      source = await decodeSavedUploadForMetadata(metadata, originalFile);
      await putSessionAudioAsset(sourceAssetIds.full, source);
    }
    let stems: DecodedStemAssets | undefined;
    if (sourceAssetIds.vocals && sourceAssetIds.instruments) {
      const vocals = await getSessionAudioAsset(sourceAssetIds.vocals);
      const instruments = await getSessionAudioAsset(sourceAssetIds.instruments);
      if (!vocals || !instruments) {
        missing.push(
          ...[
            !vocals ? sourceAssetIds.vocals : null,
            !instruments ? sourceAssetIds.instruments : null,
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
    const hasSavedStemAssets = Boolean(stems);
    const skipStemExtraction =
      metadata.sourceMode !== "full" || hasSavedStemAssets;
    hydrated.push({
      metadata,
      full: source,
      ...(originalFile ? { originalFile } : {}),
      ...(stems ? { stems } : {}),
      // Avoid re-running MelFormer when saved stems already exist. The store
      // derives the active inference PCM (full mix or cached stem) from
      // metadata.sourceMode. Legacy/full saves without cached stems re-rip on
      // play for overlays.
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

function deckTrackRef(
  deckState: DeckStateSnapshot,
  id: DeckId | null,
): { mode: "fixture" | "clip"; name: string } | null {
  if (!id) return null;
  const name = deckState.decks[id]?.trackName;
  if (!name) return null;
  const mode = useCustomTracksStore.getState().has(name) ? "clip" : "fixture";
  return { mode, name };
}

async function restoreDeckReferences(deckState: DeckStateSnapshot): Promise<void> {
  const perf = usePerformanceStore.getState();
  const timbreRef = deckTrackRef(deckState, deckState.timbreDeckId);
  const structRef = deckTrackRef(deckState, deckState.structureDeckId);
  perf.setTimbreRef(timbreRef);
  perf.setStructRef(structRef);

  const session = useSessionStore.getState();
  if (session.status !== "ready" || !session.remote) return;
  const apply = async (
    ref: { mode: "fixture" | "clip"; name: string } | null,
    sendFixture: (name: string) => void,
    sendSource: (i: Float32Array, c: number, n: string) => boolean,
  ): Promise<void> => {
    if (!ref) return;
    try {
      if (ref.mode === "fixture") {
        sendFixture(ref.name);
        return;
      }
      const decoded = await loadFixtureAudio(ref.name);
      sendSource(decoded.interleaved, decoded.channels, ref.name);
    } catch {
      // Missing/deleted ref sources should not block restoring the session.
    }
  };
  await apply(
    timbreRef,
    (name) => session.remote?.sendSetTimbreFixture(name),
    (interleaved, channels, name) =>
      session.remote?.sendSetTimbreSource(interleaved, channels, name) ?? false,
  );
  await apply(
    structRef,
    (name) => session.remote?.sendSetStructureFixture(name),
    (interleaved, channels, name) =>
      session.remote?.sendSetStructureSource(interleaved, channels, name) ?? false,
  );
}

export async function applySessionSnapshot(
  snapshot: unknown,
): Promise<SessionCompleteness> {
  const completeness = await checkSessionCompleteness(snapshot);
  if (completeness.status !== "complete") return completeness;
  if (!validateSessionSnapshotShape(snapshot)) return completeness;

  flashSessionStatus("Loading uploaded audio...");
  const hydrated = await hydrateCustomTracks(snapshot);
  if (hydrated.status !== "complete") return hydrated;

  flashSessionStatus("Restoring controls...");
  const perf = usePerformanceStore.getState();
  perf.setSkipNextDenoiseGate(true);
  applyConfig(snapshot.config);
  usePerformanceStore.getState().setFixture(snapshot.fixture);
  const deckState =
    snapshot.deckState ?? createDefaultDeckStateSnapshot(snapshot.fixture || null);
  useDeckStore.getState().restoreDeckState(deckState);
  await restoreDeckReferences(deckState);

  const stem = useStemOverlayStore.getState();
  (Object.keys(snapshot.stemOverlay.enabled) as StemOverlayKind[]).forEach((kind) => {
    stem.setEnabled(kind, snapshot.stemOverlay.enabled[kind]);
  });
  (Object.keys(snapshot.stemOverlay.volumes) as StemOverlayKind[]).forEach((kind) => {
    stem.setVolume(kind, snapshot.stemOverlay.volumes[kind]);
  });

  if (
    snapshot.customTracks.some(
      (track) => track.name === snapshot.fixture && track.stemAssetIds,
    )
  ) {
    flashSessionStatus("Restored saved stem audio.");
  }

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

export function sessionSnapshotSignature(snapshot: SessionSnapshot): string {
  return JSON.stringify({
    version: snapshot.version,
    config: snapshot.config,
    fixture: snapshot.fixture,
    customTracks: snapshot.customTracks,
    deckState: snapshot.deckState,
    stemOverlay: snapshot.stemOverlay,
  });
}
