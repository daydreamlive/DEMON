"use client";

import { create } from "zustand";

import type {
  DecodedFixture,
  DecodedStemAssets,
  StemOverlayKind,
  StemSourceMode,
} from "@/engine/audio/loadFixture";
import { defaultSwapSourceMode } from "@/lib/config";

// In-memory cache for active user-uploaded tracks. Local saved sessions mirror
// the decoded PCM, original upload File, and MelFormer stems into IndexedDB;
// this store is the live view the engine reads while the page is running.

export type StemStatus = "idle" | "processing" | "ready" | "failed";

export interface CustomTrackAssetMetadata {
  name: string;
  assetId: string;
  sourceAssetIds?: Partial<Record<StemSourceMode, string>> & { full: string };
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
  /** Full-mix PCM. The active inference source is derived from this plus
   *  `stems` and `metadata.sourceMode`. */
  full: DecodedFixture;
  originalFile?: File;
  stems?: DecodedStemAssets;
  skipStemExtraction?: boolean;
}

export interface CustomTrack {
  /** The PCM that currently feeds inference — derived from `full`/`stems`
   *  for the active `sourceMode`. Kept in sync by setSourceMode/setStems. */
  decoded?: DecodedFixture;
  /** The full-mix PCM, retained so the inference source can switch back to
   *  "full" and so all three sources stay available without re-ripping. */
  full?: DecodedFixture;
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
  /**
   * True once the track's audio + sidecars + stems exist on the pod's
   * disk (seeded from the server, or persisted by a successful
   * uploadTrackToServer). Lets a swap to this track load by name on the
   * server instead of re-uploading PCM and re-ripping stems. Tracks that
   * only ever lived in browser memory (no-pod fallback, MCP mirror) stay
   * false and keep the client-supplied-PCM swap path.
   */
  persisted: boolean;
}

type CustomTrackMetadata = Partial<
  Pick<
    CustomTrack,
    "assetId" | "originalFileName" | "trimStartS" | "trimEndS" | "addedAt"
  >
>;

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
    metadataOrPersisted?: CustomTrackMetadata | boolean,
    persisted?: boolean,
  ) => void;
  addPersisted: (name: string, sourceMode?: StemSourceMode) => void;
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
  /**
   * Is this track loadable by name on the server? True for built-in
   * fixtures (everything the dropdown shows that isn't a custom track is
   * a pod-resident fixture) and for persisted uploads. Drives the
   * server-side swap fast path.
   */
  isServerResident: (name: string) => boolean;
}

/** The PCM that should feed inference for the track's active source mode.
 *  "full" uses the full mix; "vocals"/"instruments" use the matching cached
 *  stem when available, falling back to the full mix until stems arrive. */
function resolveActiveDecoded(track: CustomTrack): DecodedFixture | undefined {
  if (track.sourceMode !== "full" && track.stems) {
    return track.stems[track.sourceMode];
  }
  return track.full ?? track.decoded;
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

  add: (
    name,
    decoded,
    file,
    sourceMode = defaultSwapSourceMode(),
    metadataOrPersisted = {},
    persistedArg = false,
  ) =>
    set((s) => {
      const metadata =
        typeof metadataOrPersisted === "boolean" ? {} : metadataOrPersisted;
      const persisted =
        typeof metadataOrPersisted === "boolean"
          ? metadataOrPersisted
          : persistedArg;
      const nextTracks = new Map(s.tracks);
      nextTracks.set(name, {
        decoded,
        full: decoded,
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
        persisted,
      });
      const nextNames = s.names.includes(name) ? s.names : [...s.names, name];
      return {
        names: nextNames,
        tracks: nextTracks,
      };
    }),

  addPersisted: (name, sourceMode = defaultSwapSourceMode()) =>
    set((s) => {
      if (s.tracks.has(name)) return {};
      const nextTracks = new Map(s.tracks);
      nextTracks.set(name, {
        assetId: createAssetId(name),
        addedAt: Date.now(),
        sourceMode,
        stemStatus: "idle",
        persisted: true,
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
      const next: CustomTrack = { ...track, sourceMode };
      // Re-point the inference PCM at the newly selected source so the swap
      // actually feeds vocals/instruments/full — not just relabels it.
      next.decoded = resolveActiveDecoded(next) ?? next.decoded;
      const nextTracks = new Map(s.tracks);
      nextTracks.set(name, next);
      return { tracks: nextTracks };
    }),

  setStems: (name, stems) =>
    set((s) => {
      const track = s.tracks.get(name);
      if (!track) return {};
      const next: CustomTrack = {
        ...track,
        stems,
        stemAssetIds: track.stemAssetIds ?? {
          vocals: `${track.assetId}:stem:vocals`,
          instruments: `${track.assetId}:stem:instruments`,
        },
        stemStatus: "ready",
        stemError: undefined,
      };
      // Now that the matching stem PCM exists, make sure the active source
      // mode points at it (e.g. a track uploaded as "vocals" switches its
      // inference PCM from the full mix to the vocal stem on arrival).
      next.decoded = resolveActiveDecoded(next) ?? next.decoded;
      const nextTracks = new Map(s.tracks);
      nextTracks.set(name, next);
      return { tracks: nextTracks };
    }),

  resolveSourceMode: (name) => {
    return get().tracks.get(name)?.sourceMode;
  },

  // The source mode is always informative to the backend: on the
  // server-resident path it selects the cached per-mode sidecar, and on the
  // client-PCM path it is ignored when skip_stem_extraction is set. There is
  // no longer a reason to suppress it for restored sessions.
  resolveBackendSourceMode: (name) => {
    return get().tracks.get(name)?.sourceMode;
  },

  shouldSkipStemExtraction: (name) => {
    return get().tracks.get(name)?.skipStemExtraction === true;
  },

  exportMetadata: () => {
    const out: CustomTrackAssetMetadata[] = [];
    for (const [name, track] of get().tracks.entries()) {
      const full = track.full ?? track.decoded;
      if (!full) continue;
      const sourceAssetIds: CustomTrackAssetMetadata["sourceAssetIds"] = {
        full: track.assetId,
        ...(track.stemAssetIds
          ? {
              vocals: track.stemAssetIds.vocals,
              instruments: track.stemAssetIds.instruments,
            }
          : {}),
      };
      out.push({
        name,
        assetId: track.assetId,
        sourceAssetIds,
        ...(track.stemAssetIds ? { stemAssetIds: track.stemAssetIds } : {}),
        sourceMode: track.sourceMode,
        ...(track.originalFileName
          ? { originalFileName: track.originalFileName }
          : {}),
        ...(typeof track.trimStartS === "number"
          ? { trimStartS: track.trimStartS }
          : {}),
        ...(typeof track.trimEndS === "number"
          ? { trimEndS: track.trimEndS }
          : {}),
        frames: full.frames,
        channels: full.channels,
        sampleRate: full.sampleRate,
        addedAt: track.addedAt,
      });
    }
    return out;
  },

  hydrateSavedTracks: (tracks) =>
    set((s) => {
      const nextTracks = new Map(s.tracks);
      const nextNames = [...s.names];
      for (const {
        metadata,
        full,
        originalFile,
        stems,
        skipStemExtraction,
      } of tracks) {
        const next: CustomTrack = {
          decoded: full,
          full,
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
          persisted: false,
        };
        // Restore the saved inference source (full mix or a cached stem).
        next.decoded = resolveActiveDecoded(next) ?? full;
        nextTracks.set(metadata.name, next);
        if (!nextNames.includes(metadata.name)) nextNames.push(metadata.name);
      }
      return {
        names: nextNames,
        tracks: nextTracks,
      };
    }),

  has: (name) => get().tracks.has(name),

  isServerResident: (name) => {
    const track = get().tracks.get(name);
    // Not a custom track → it's a built-in fixture, which always lives on
    // the pod. A custom track is server-loadable only once persisted.
    if (!track) return true;
    return track.persisted;
  },
}));
