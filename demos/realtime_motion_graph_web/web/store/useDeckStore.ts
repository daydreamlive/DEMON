"use client";

import { create } from "zustand";

import type { StemOverlayKind, StemSourceMode } from "@/engine/audio/loadFixture";

export type DeckId = "A" | "B" | "C" | "D";
export type DeckCrossfadeSide = "left" | "right" | null;

export interface DeckSlot {
  id: DeckId;
  trackName: string | null;
  color: string;
  sourcePart: StemSourceMode;
  stemOverlayLevels: Record<StemOverlayKind, number>;
  volume: number;
  muted: boolean;
  solo: boolean;
  playing: boolean;
  cueSec: number;
  positionSec: number;
  lastStartedAtMs: number | null;
  crossfadeSide: DeckCrossfadeSide;
}

interface DeckStoreState {
  decks: Record<DeckId, DeckSlot>;
  deckIds: DeckId[];
  timbreDeckId: DeckId | null;
  structureDeckId: DeckId | null;
  crossfade: number;
  monitorEnabled: boolean;
  inferenceEnabled: boolean;
  mixRevision: number;
  ensureInitialDeck: (trackName: string) => void;
  addDeck: (trackName: string) => DeckId | null;
  removeDeck: (id: DeckId) => void;
  setTimbreDeck: (id: DeckId | null) => void;
  setStructureDeck: (id: DeckId | null) => void;
  setTrack: (id: DeckId, trackName: string) => void;
  setSourcePart: (id: DeckId, part: StemSourceMode) => void;
  setStemOverlayLevel: (
    id: DeckId,
    kind: StemOverlayKind,
    level: number,
  ) => void;
  setVolume: (id: DeckId, volume: number) => void;
  setMuted: (id: DeckId, muted: boolean) => void;
  toggleMuted: (id: DeckId) => void;
  setSolo: (id: DeckId, solo: boolean) => void;
  toggleSolo: (id: DeckId) => void;
  setPlaying: (id: DeckId, playing: boolean, nowMs?: number) => void;
  seek: (id: DeckId, positionSec: number, nowMs?: number) => void;
  setCue: (id: DeckId, cueSec: number) => void;
  jumpToCue: (id: DeckId, nowMs?: number) => void;
  setCrossfadeSide: (id: DeckId, side: DeckCrossfadeSide) => void;
  setCrossfade: (value: number) => void;
  setMonitorEnabled: (enabled: boolean) => void;
  setInferenceEnabled: (enabled: boolean) => void;
}

export const DECK_IDS: DeckId[] = ["A", "B", "C", "D"];
export const MAX_DECKS = 4;
const DECK_COLORS: Record<DeckId, string> = {
  A: "oklch(0.72 0.16 42)",
  B: "oklch(0.72 0.15 158)",
  C: "oklch(0.70 0.16 305)",
  D: "oklch(0.74 0.13 215)",
};
export const DECK_STEM_OVERLAY_MAX = 6.0;

function now(): number {
  return typeof performance !== "undefined" ? performance.now() : Date.now();
}

function clamp01(value: number): number {
  if (!Number.isFinite(value)) return 0;
  return Math.max(0, Math.min(1, value));
}

function clampStemOverlay(value: number): number {
  if (!Number.isFinite(value)) return 0;
  return Math.max(0, Math.min(DECK_STEM_OVERLAY_MAX, value));
}

function clampSec(value: number): number {
  if (!Number.isFinite(value)) return 0;
  return Math.max(0, value);
}

function makeDeck(id: DeckId, crossfadeSide: DeckCrossfadeSide): DeckSlot {
  return {
    id,
    trackName: null,
    color: DECK_COLORS[id],
    sourcePart: "full",
    stemOverlayLevels: { vocals: 0, instruments: 0 },
    volume: 1,
    muted: false,
    solo: false,
    playing: false,
    cueSec: 0,
    positionSec: 0,
    lastStartedAtMs: null,
    crossfadeSide,
  };
}

function currentDeckPosition(deck: DeckSlot, nowMs = now()): number {
  if (!deck.playing || deck.lastStartedAtMs === null) return deck.positionSec;
  return deck.positionSec + Math.max(0, nowMs - deck.lastStartedAtMs) / 1000;
}

function patchDeck(
  id: DeckId,
  patcher: (deck: DeckSlot, state: DeckStoreState) => DeckSlot,
  bumpRevision = true,
) {
  return (state: DeckStoreState): Partial<DeckStoreState> => ({
    decks: {
      ...state.decks,
      [id]: patcher(state.decks[id], state),
    },
    ...(bumpRevision ? { mixRevision: state.mixRevision + 1 } : {}),
  });
}

export function deckPositionSec(deck: DeckSlot, nowMs = now()): number {
  return currentDeckPosition(deck, nowMs);
}

export const useDeckStore = create<DeckStoreState>((set) => ({
  decks: {
    A: makeDeck("A", "left"),
    B: makeDeck("B", "right"),
    C: makeDeck("C", null),
    D: makeDeck("D", null),
  },
  deckIds: ["A"],
  timbreDeckId: null,
  structureDeckId: null,
  crossfade: 0.5,
  monitorEnabled: false,
  inferenceEnabled: true,
  mixRevision: 0,

  ensureInitialDeck: (trackName) =>
    set((state) => {
      const firstId = state.deckIds[0] ?? "A";
      const firstDeck = state.decks[firstId];
      if (firstDeck?.trackName) return {};
      return {
        deckIds: state.deckIds.length > 0 ? state.deckIds : [firstId],
        decks: {
          ...state.decks,
          [firstId]: {
            ...firstDeck,
            trackName,
            positionSec: 0,
            cueSec: 0,
            lastStartedAtMs: firstDeck.playing ? now() : null,
          },
        },
        mixRevision: state.mixRevision + 1,
      };
    }),

  addDeck: (trackName) => {
    let added: DeckId | null = null;
    set((state) => {
      if (state.deckIds.length >= MAX_DECKS) return {};
      const id = DECK_IDS.find((candidate) => !state.deckIds.includes(candidate));
      if (!id) return {};
      added = id;
      const hasLeft = state.deckIds.some(
        (deckId) => state.decks[deckId].crossfadeSide === "left",
      );
      const hasRight = state.deckIds.some(
        (deckId) => state.decks[deckId].crossfadeSide === "right",
      );
      const crossfadeSide: DeckCrossfadeSide = !hasLeft
        ? "left"
        : !hasRight
          ? "right"
          : null;
      return {
        deckIds: [...state.deckIds, id],
        decks: {
          ...state.decks,
          [id]: {
            ...state.decks[id],
            trackName,
            sourcePart: "full",
            stemOverlayLevels: { vocals: 0, instruments: 0 },
            volume: 1,
            muted: false,
            solo: false,
            playing: false,
            cueSec: 0,
            positionSec: 0,
            lastStartedAtMs: null,
            crossfadeSide,
          },
        },
        mixRevision: state.mixRevision + 1,
      };
    });
    return added;
  },

  removeDeck: (id) =>
    set((state) => {
      if (state.deckIds.length <= 1 || !state.deckIds.includes(id)) return {};
      const nextDeckIds = state.deckIds.filter((deckId) => deckId !== id);
      return {
        deckIds: nextDeckIds,
        timbreDeckId: state.timbreDeckId === id ? null : state.timbreDeckId,
        structureDeckId:
          state.structureDeckId === id ? null : state.structureDeckId,
        decks: {
          ...state.decks,
          [id]: {
            ...state.decks[id],
            trackName: null,
            stemOverlayLevels: { vocals: 0, instruments: 0 },
            playing: false,
            muted: false,
            solo: false,
            lastStartedAtMs: null,
            positionSec: 0,
            cueSec: 0,
          },
        },
        mixRevision: state.mixRevision + 1,
      };
    }),
  setTimbreDeck: (id) =>
    set((state) =>
      id === null || state.deckIds.includes(id)
        ? { timbreDeckId: id, mixRevision: state.mixRevision + 1 }
        : {},
    ),
  setStructureDeck: (id) =>
    set((state) =>
      id === null || state.deckIds.includes(id)
        ? { structureDeckId: id, mixRevision: state.mixRevision + 1 }
        : {},
    ),

  setTrack: (id, trackName) =>
    set(
      patchDeck(id, (deck) => ({
        ...deck,
        trackName,
        positionSec: 0,
        cueSec: 0,
        lastStartedAtMs: deck.playing ? now() : null,
      })),
    ),
  setSourcePart: (id, part) =>
    set(patchDeck(id, (deck) => ({ ...deck, sourcePart: part }))),
  setStemOverlayLevel: (id, kind, level) =>
    set(
      patchDeck(id, (deck) => ({
        ...deck,
        stemOverlayLevels: {
          ...deck.stemOverlayLevels,
          [kind]: clampStemOverlay(level),
        },
      })),
    ),
  setVolume: (id, volume) =>
    set(patchDeck(id, (deck) => ({ ...deck, volume: clamp01(volume) }))),
  setMuted: (id, muted) =>
    set(patchDeck(id, (deck) => ({ ...deck, muted }))),
  toggleMuted: (id) =>
    set(patchDeck(id, (deck) => ({ ...deck, muted: !deck.muted }))),
  setSolo: (id, solo) =>
    set(patchDeck(id, (deck) => ({ ...deck, solo }))),
  toggleSolo: (id) =>
    set(patchDeck(id, (deck) => ({ ...deck, solo: !deck.solo }))),
  setPlaying: (id, playing, nowMs = now()) =>
    set(
      patchDeck(id, (deck) => {
        const pos = currentDeckPosition(deck, nowMs);
        return {
          ...deck,
          playing,
          positionSec: pos,
          lastStartedAtMs: playing ? nowMs : null,
        };
      }),
    ),
  seek: (id, positionSec, nowMs = now()) =>
    set(
      patchDeck(id, (deck) => ({
        ...deck,
        positionSec: clampSec(positionSec),
        lastStartedAtMs: deck.playing ? nowMs : null,
      })),
    ),
  setCue: (id, cueSec) =>
    set(patchDeck(id, (deck) => ({ ...deck, cueSec: clampSec(cueSec) }))),
  jumpToCue: (id, nowMs = now()) =>
    set(
      patchDeck(id, (deck) => ({
        ...deck,
        positionSec: deck.cueSec,
        lastStartedAtMs: deck.playing ? nowMs : null,
      })),
    ),
  setCrossfadeSide: (id, side) =>
    set((state) => {
      if (!state.deckIds.includes(id)) return {};
      const nextDecks = { ...state.decks };
      for (const deckId of state.deckIds) {
        if (deckId !== id && nextDecks[deckId].crossfadeSide === side) {
          nextDecks[deckId] = { ...nextDecks[deckId], crossfadeSide: null };
        }
      }
      nextDecks[id] = { ...nextDecks[id], crossfadeSide: side };
      return {
        decks: nextDecks,
        mixRevision: state.mixRevision + 1,
      };
    }),
  setCrossfade: (value) =>
    set((state) => ({ crossfade: clamp01(value), mixRevision: state.mixRevision + 1 })),
  setMonitorEnabled: (enabled) => set({ monitorEnabled: enabled }),
  setInferenceEnabled: (enabled) => set({ inferenceEnabled: enabled }),
}));
