"use client";

import { create } from "zustand";

import type {
  DecodedFixture,
  DecodedStemAssets,
  StemSourceMode,
} from "@/engine/audio/loadFixture";
import { defaultSwapSourceMode } from "@/lib/config";

// In-memory cache for user-uploaded tracks. The decoded PCM and related upload
// metadata live in one non-persistent Map (Float32Array and File don't survive
// JSON / localStorage), and the names are mirrored into a reactive list so the
// fixture dropdown re-renders when an upload completes. Cleared on page reload
// — uploads are session-scoped, matching how the pod treats fixtures.

export type StemStatus = "idle" | "processing" | "ready" | "failed";

export interface CustomTrack {
  decoded?: DecodedFixture;
  /** Original encoded upload, when available from the file-picker path. */
  originalFile?: File;
  /** Which version of the uploaded track should feed model inference. */
  sourceMode: StemSourceMode;
  /** Model-ripped stems returned by the backend. */
  stems?: DecodedStemAssets;
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
  has: (name: string) => boolean;
  /**
   * Is this track loadable by name on the server? True for built-in
   * fixtures (everything the dropdown shows that isn't a custom track is
   * a pod-resident fixture) and for persisted uploads. Drives the
   * server-side swap fast path.
   */
  isServerResident: (name: string) => boolean;
}

export const useCustomTracksStore = create<CustomTracksState>((set, get) => ({
  names: [],
  tracks: new Map(),

  add: (name, decoded, file, sourceMode = defaultSwapSourceMode(), persisted = false) =>
    set((s) => {
      const nextTracks = new Map(s.tracks);
      nextTracks.set(name, {
        decoded,
        ...(file ? { originalFile: file } : {}),
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
        stemStatus: "ready",
        stemError: undefined,
      });
      return { tracks: nextTracks };
    }),

  resolveSourceMode: (name) => {
    return get().tracks.get(name)?.sourceMode;
  },

  has: (name) => get().tracks.has(name),

  isServerResident: (name) => {
    const track = get().tracks.get(name);
    // Not a custom track → it's a built-in fixture, which always lives on
    // the pod. A custom track is server-loadable only once persisted.
    if (!track) return true;
    return track.persisted;
  },
}));
