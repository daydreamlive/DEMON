"use client";

import { create } from "zustand";

import type {
  DecodedFixture,
  DecodedStemAssets,
  StemOverlayKind,
  StemSourceMode,
} from "@/engine/audio/loadFixture";

// In-memory cache for active user-uploaded tracks. Local saved sessions mirror
// the decoded PCM, original upload File, and MelFormer stems into IndexedDB;
// this store is the live view the engine reads while the page is running.

export type StemStatus = "idle" | "processing" | "ready" | "failed";

export interface CustomTrackAssetMetadata {
  name: string;
  assetId: string;
  stemAssetIds?: Record<StemOverlayKind, string>;
  sourceMode: StemSourceMode;
  originalFileName?: string;
  trimStartS?: number;
  trimEndS?: number;
  frames: number;
  channels: number;
  sampleRate: number;
  addedAt: number;
}

export interface HydratedCustomTrack {
  metadata: CustomTrackAssetMetadata;
  decoded: DecodedFixture;
  originalFile?: File;
  stems?: DecodedStemAssets;
  skipStemExtraction?: boolean;
}

export interface CustomTrack {
  decoded: DecodedFixture;
  /** Original encoded upload, when available from the file-picker path. */
  originalFile?: File;
  /** Stable key for the trimmed PCM stored in IndexedDB. */
  assetId: string;
  /** Stable keys for MelFormer stem PCM stored in IndexedDB. */
  stemAssetIds?: Record<StemOverlayKind, string>;
  /** Original filename retained for relinking after browser storage loss. */
  originalFileName?: string;
  /** Trim window that produced `decoded`; used when relinking originals. */
  trimStartS?: number;
  trimEndS?: number;
  /** Creation time used only for local-session metadata. */
  addedAt: number;
  /** Which version of the uploaded track should feed model inference. */
  sourceMode: StemSourceMode;
  /** Model-ripped stems returned by the backend. */
  stems?: DecodedStemAssets;
  /** Restored sessions hydrate stem PCM locally; do not ask backend to re-rip. */
  skipStemExtraction?: boolean;
  stemStatus: StemStatus;
  stemError?: string;
}

interface CustomTracksState {
  /** Names in upload order. Reactive — components subscribe to this. */
  names: string[];
  /** Upload records keyed by name. Read via getState() from non-React code. */
  tracks: Map<string, CustomTrack>;

  add: (
    name: string,
    decoded: DecodedFixture,
    file?: File,
    sourceMode?: StemSourceMode,
    metadata?: Partial<
      Pick<
        CustomTrack,
        "assetId" | "originalFileName" | "trimStartS" | "trimEndS" | "addedAt"
      >
    >,
  ) => void;
  setStemStatus: (
    name: string,
    status: StemStatus,
    error?: string,
  ) => void;
  setSourceMode: (name: string, sourceMode: StemSourceMode) => void;
  setStems: (name: string, stems: DecodedStemAssets) => void;
  resolveSourceMode: (name: string) => StemSourceMode | undefined;
  resolveBackendSourceMode: (name: string) => StemSourceMode | undefined;
  shouldSkipStemExtraction: (name: string) => boolean;
  exportMetadata: () => CustomTrackAssetMetadata[];
  hydrateSavedTracks: (tracks: HydratedCustomTrack[]) => void;
  has: (name: string) => boolean;
}

function createAssetId(name: string): string {
  const slug = name
    .toLowerCase()
    .replace(/[^a-z0-9._-]+/g, "-")
    .replace(/^-+|-+$/g, "")
    .slice(0, 72) || "upload";
  const suffix =
    typeof crypto !== "undefined" && "randomUUID" in crypto
      ? crypto.randomUUID()
      : `${Date.now().toString(36)}-${Math.random().toString(36).slice(2)}`;
  return `custom:${slug}:${suffix}`;
}

export const useCustomTracksStore = create<CustomTracksState>((set, get) => ({
  names: [],
  tracks: new Map(),

  add: (name, decoded, file, sourceMode = "full", metadata = {}) =>
    set((s) => {
      const nextTracks = new Map(s.tracks);
      nextTracks.set(name, {
        decoded,
        ...(file ? { originalFile: file } : {}),
        assetId: metadata.assetId ?? createAssetId(name),
        originalFileName: metadata.originalFileName ?? file?.name,
        ...(typeof metadata.trimStartS === "number"
          ? { trimStartS: metadata.trimStartS }
          : {}),
        ...(typeof metadata.trimEndS === "number"
          ? { trimEndS: metadata.trimEndS }
          : {}),
        addedAt: metadata.addedAt ?? Date.now(),
        sourceMode,
        stemStatus: "idle",
      });
      const nextNames = s.names.includes(name) ? s.names : [...s.names, name];
      return {
        names: nextNames,
        tracks: nextTracks,
      };
    }),

  setStemStatus: (name, status, error) =>
    set((s) => {
      const track = s.tracks.get(name);
      if (!track) return {};
      const nextTracks = new Map(s.tracks);
      nextTracks.set(name, {
        ...track,
        stemStatus: status,
        ...(error ? { stemError: error } : { stemError: undefined }),
      });
      return { tracks: nextTracks };
    }),

  setSourceMode: (name, sourceMode) =>
    set((s) => {
      const track = s.tracks.get(name);
      if (!track) return {};
      const nextTracks = new Map(s.tracks);
      nextTracks.set(name, { ...track, sourceMode });
      return { tracks: nextTracks };
    }),

  setStems: (name, stems) =>
    set((s) => {
      const track = s.tracks.get(name);
      if (!track) return {};
      const nextTracks = new Map(s.tracks);
      nextTracks.set(name, {
        ...track,
        stems,
        stemAssetIds: track.stemAssetIds ?? {
          vocals: `${track.assetId}:stem:vocals`,
          instruments: `${track.assetId}:stem:instruments`,
        },
        stemStatus: "ready",
        stemError: undefined,
      });
      return { tracks: nextTracks };
    }),

  resolveSourceMode: (name) => {
    return get().tracks.get(name)?.sourceMode;
  },

  resolveBackendSourceMode: (name) => {
    const track = get().tracks.get(name);
    if (!track || track.skipStemExtraction) return undefined;
    return track.sourceMode;
  },

  shouldSkipStemExtraction: (name) => {
    return get().tracks.get(name)?.skipStemExtraction === true;
  },

  exportMetadata: () => {
    return Array.from(get().tracks.entries()).map(([name, track]) => ({
      name,
      assetId: track.assetId,
      ...(track.stemAssetIds ? { stemAssetIds: track.stemAssetIds } : {}),
      sourceMode: track.sourceMode,
      ...(track.originalFileName ? { originalFileName: track.originalFileName } : {}),
      ...(typeof track.trimStartS === "number" ? { trimStartS: track.trimStartS } : {}),
      ...(typeof track.trimEndS === "number" ? { trimEndS: track.trimEndS } : {}),
      frames: track.decoded.frames,
      channels: track.decoded.channels,
      sampleRate: track.decoded.sampleRate,
      addedAt: track.addedAt,
    }));
  },

  hydrateSavedTracks: (tracks) =>
    set((s) => {
      const nextTracks = new Map(s.tracks);
      const nextNames = [...s.names];
      for (const {
        metadata,
        decoded,
        originalFile,
        stems,
        skipStemExtraction,
      } of tracks) {
        nextTracks.set(metadata.name, {
          decoded,
          ...(originalFile ? { originalFile } : {}),
          assetId: metadata.assetId,
          stemAssetIds: metadata.stemAssetIds,
          originalFileName: metadata.originalFileName,
          trimStartS: metadata.trimStartS,
          trimEndS: metadata.trimEndS,
          addedAt: metadata.addedAt,
          sourceMode: metadata.sourceMode,
          ...(stems ? { stems } : {}),
          skipStemExtraction,
          stemStatus: stems ? "ready" : "idle",
        });
        if (!nextNames.includes(metadata.name)) nextNames.push(metadata.name);
      }
      return {
        names: nextNames,
        tracks: nextTracks,
      };
    }),

  has: (name) => get().tracks.has(name),
}));
