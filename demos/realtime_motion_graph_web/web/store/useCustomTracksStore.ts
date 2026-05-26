"use client";

import { create } from "zustand";

import type {
  DecodedFixture,
  DecodedStemAssets,
  StemSourceMode,
} from "@/engine/audio/loadFixture";

// In-memory cache for user-uploaded tracks. The decoded PCM and related upload
// metadata live in one non-persistent Map (Float32Array and File don't survive
// JSON / localStorage), and the names are mirrored into a reactive list so the
// fixture dropdown re-renders when an upload completes. Cleared on page reload
// — uploads are session-scoped, matching how the pod treats fixtures.

export type StemStatus = "idle" | "queued" | "processing" | "running" | "ready" | "failed";

export interface CustomTrack {
  decoded: DecodedFixture;
  /** Original encoded upload, when available from the file-picker path. */
  originalFile?: File;
  /** Which version of the uploaded track should feed model inference. */
  sourceMode: StemSourceMode;
  /** Model-ripped stems returned by the backend. */
  stems?: DecodedStemAssets;
  stemStatus: StemStatus;
  stemError?: string;
  /** Base-model LEGO layers generated from this uploaded track. */
  legoStems?: DecodedStemAssets;
  legoStatus: Record<string, StemStatus>;
  legoErrors: Record<string, string | undefined>;
  legoPrompts: Record<string, string>;
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
  ) => void;
  setStemStatus: (
    name: string,
    status: StemStatus,
    error?: string,
  ) => void;
  setSourceMode: (name: string, sourceMode: StemSourceMode) => void;
  setStems: (name: string, stems: DecodedStemAssets) => void;
  setLegoStatus: (
    name: string,
    track: string,
    status: StemStatus,
    error?: string,
  ) => void;
  setLegoStems: (
    name: string,
    stems: DecodedStemAssets,
    prompts?: Record<string, string>,
  ) => void;
  setLegoPrompt: (name: string, track: string, prompt: string) => void;
  resolveSourceMode: (name: string) => StemSourceMode | undefined;
  has: (name: string) => boolean;
}

export const useCustomTracksStore = create<CustomTracksState>((set, get) => ({
  names: [],
  tracks: new Map(),

  add: (name, decoded, file, sourceMode = "full") =>
    set((s) => {
      const nextTracks = new Map(s.tracks);
      nextTracks.set(name, {
        decoded,
        ...(file ? { originalFile: file } : {}),
        sourceMode,
        stemStatus: "idle",
        legoStatus: {},
        legoErrors: {},
        legoPrompts: {},
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

  setLegoStatus: (name, legoTrack, status, error) =>
    set((s) => {
      const track = s.tracks.get(name);
      if (!track) return {};
      const nextTracks = new Map(s.tracks);
      nextTracks.set(name, {
        ...track,
        legoStatus: { ...track.legoStatus, [legoTrack]: status },
        legoErrors: {
          ...track.legoErrors,
          [legoTrack]: error,
        },
      });
      return { tracks: nextTracks };
    }),

  setLegoStems: (name, stems, prompts) =>
    set((s) => {
      const track = s.tracks.get(name);
      if (!track) return {};
      const nextLegoStems = { ...(track.legoStems ?? {}), ...stems };
      const nextStatus = { ...track.legoStatus };
      const nextErrors = { ...track.legoErrors };
      for (const stemName of Object.keys(stems)) {
        nextStatus[stemName] = "ready";
        nextErrors[stemName] = undefined;
      }
      const nextTracks = new Map(s.tracks);
      nextTracks.set(name, {
        ...track,
        legoStems: nextLegoStems,
        legoStatus: nextStatus,
        legoErrors: nextErrors,
        legoPrompts: { ...track.legoPrompts, ...(prompts ?? {}) },
      });
      return { tracks: nextTracks };
    }),

  setLegoPrompt: (name, legoTrack, prompt) =>
    set((s) => {
      const track = s.tracks.get(name);
      if (!track) return {};
      const nextTracks = new Map(s.tracks);
      nextTracks.set(name, {
        ...track,
        legoPrompts: { ...track.legoPrompts, [legoTrack]: prompt },
      });
      return { tracks: nextTracks };
    }),

  resolveSourceMode: (name) => {
    return get().tracks.get(name)?.sourceMode;
  },

  has: (name) => get().tracks.has(name),
}));
