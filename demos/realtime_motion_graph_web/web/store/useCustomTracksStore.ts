"use client";

import { create } from "zustand";

import type { DecodedFixture } from "@/engine/audio/loadFixture";

// In-memory cache for user-uploaded tracks. The decoded PCM lives in a
// non-reactive Map (Float32Array doesn't survive JSON / localStorage), and
// the names are mirrored into a reactive list so the fixture dropdown
// re-renders when an upload completes. Cleared on page reload — uploads
// are session-scoped, matching how the pod treats fixtures (it only ever
// sees the decoded PCM, never the file).

interface CustomTracksState {
  /** Names in upload order. Reactive — components subscribe to this. */
  names: string[];
  /** Decoded buffers keyed by name. Read directly via getState() from
   *  non-React code (loadFixtureAudio); updates don't re-render. */
  decoded: Map<string, DecodedFixture>;

  add: (name: string, decoded: DecodedFixture) => void;
  has: (name: string) => boolean;
}

export const useCustomTracksStore = create<CustomTracksState>((set, get) => ({
  names: [],
  decoded: new Map(),

  add: (name, decoded) =>
    set((s) => {
      const nextDecoded = new Map(s.decoded);
      nextDecoded.set(name, decoded);
      const nextNames = s.names.includes(name) ? s.names : [...s.names, name];
      return { names: nextNames, decoded: nextDecoded };
    }),

  has: (name) => get().decoded.has(name),
}));
