"use client";

import { create } from "zustand";

import type { StemOverlayKind, StemSourceMode } from "@/engine/audio/loadFixture";

export type DeckId = "A" | "B" | "C" | "D";
export type DeckCrossfadeSide = "left" | "right" | null;
export const DECK_STATE_SNAPSHOT_VERSION = 1;

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

export interface DeckSlotSnapshot {
  id: DeckId;
  trackName: string | null;
  sourcePart: StemSourceMode;
  stemOverlayLevels: Record<StemOverlayKind, number>;
  volume: number;
  muted: boolean;
  solo: boolean;
  cueSec: number;
  positionSec: number;
  crossfadeSide: DeckCrossfadeSide;
}

export interface DeckStateSnapshotV1 {
  version: typeof DECK_STATE_SNAPSHOT_VERSION;
  deckIds: DeckId[];
  decks: Record<DeckId, DeckSlotSnapshot>;
  timbreDeckId: DeckId | null;
  structureDeckId: DeckId | null;
  crossfade: number;
  inferenceEnabled: boolean;
}

export type DeckStateSnapshot = DeckStateSnapshotV1;

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
  restoreDeckState: (snapshot: DeckStateSnapshot) => void;
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

function makeDefaultDecks(): Record<DeckId, DeckSlot> {
  return {
    A: makeDeck("A", "left"),
    B: makeDeck("B", "right"),
    C: makeDeck("C", null),
    D: makeDeck("D", null),
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

function deckSlotSnapshot(deck: DeckSlot): DeckSlotSnapshot {
  return {
    id: deck.id,
    trackName: deck.trackName,
    sourcePart: deck.sourcePart,
    stemOverlayLevels: { ...deck.stemOverlayLevels },
    volume: deck.volume,
    muted: deck.muted,
    solo: deck.solo,
    cueSec: deck.cueSec,
    positionSec: deckPositionSec(deck),
    crossfadeSide: deck.crossfadeSide,
  };
}

function isDeckId(value: unknown): value is DeckId {
  return value === "A" || value === "B" || value === "C" || value === "D";
}

function isSourcePart(value: unknown): value is StemSourceMode {
  return value === "full" || value === "vocals" || value === "instruments";
}

function isCrossfadeSide(value: unknown): value is DeckCrossfadeSide {
  return value === "left" || value === "right" || value === null;
}

function restoreDeckSlot(
  id: DeckId,
  snapshot: Partial<DeckSlotSnapshot> | undefined,
  fallback: DeckSlot,
): DeckSlot {
  const stemLevels = snapshot?.stemOverlayLevels ?? fallback.stemOverlayLevels;
  return {
    ...fallback,
    trackName:
      typeof snapshot?.trackName === "string" || snapshot?.trackName === null
        ? snapshot.trackName
        : fallback.trackName,
    sourcePart: isSourcePart(snapshot?.sourcePart)
      ? snapshot.sourcePart
      : fallback.sourcePart,
    stemOverlayLevels: {
      vocals: clampStemOverlay(stemLevels.vocals),
      instruments: clampStemOverlay(stemLevels.instruments),
    },
    volume:
      typeof snapshot?.volume === "number"
        ? clamp01(snapshot.volume)
        : fallback.volume,
    muted:
      typeof snapshot?.muted === "boolean" ? snapshot.muted : fallback.muted,
    solo: typeof snapshot?.solo === "boolean" ? snapshot.solo : fallback.solo,
    playing: false,
    cueSec:
      typeof snapshot?.cueSec === "number"
        ? clampSec(snapshot.cueSec)
        : fallback.cueSec,
    positionSec:
      typeof snapshot?.positionSec === "number"
        ? clampSec(snapshot.positionSec)
        : fallback.positionSec,
    lastStartedAtMs: null,
    crossfadeSide: isCrossfadeSide(snapshot?.crossfadeSide)
      ? snapshot.crossfadeSide
      : fallback.crossfadeSide,
    id,
  };
}

export function captureDeckStateSnapshot(): DeckStateSnapshot {
  const state = useDeckStore.getState();
  return {
    version: DECK_STATE_SNAPSHOT_VERSION,
    deckIds: [...state.deckIds],
    decks: {
      A: deckSlotSnapshot(state.decks.A),
      B: deckSlotSnapshot(state.decks.B),
      C: deckSlotSnapshot(state.decks.C),
      D: deckSlotSnapshot(state.decks.D),
    },
    timbreDeckId: state.timbreDeckId,
    structureDeckId: state.structureDeckId,
    crossfade: state.crossfade,
    inferenceEnabled: state.inferenceEnabled,
  };
}

export function createDefaultDeckStateSnapshot(
  trackName: string | null = null,
): DeckStateSnapshot {
  const decks = makeDefaultDecks();
  decks.A.trackName = trackName;
  return {
    version: DECK_STATE_SNAPSHOT_VERSION,
    deckIds: ["A"],
    decks: {
      A: deckSlotSnapshot(decks.A),
      B: deckSlotSnapshot(decks.B),
      C: deckSlotSnapshot(decks.C),
      D: deckSlotSnapshot(decks.D),
    },
    timbreDeckId: null,
    structureDeckId: null,
    crossfade: 0.5,
    inferenceEnabled: true,
  };
}

export const useDeckStore = create<DeckStoreState>((set) => ({
  decks: makeDefaultDecks(),
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
  restoreDeckState: (snapshot) =>
    set((state) => {
      if (snapshot.version !== DECK_STATE_SNAPSHOT_VERSION) return {};
      const defaults = makeDefaultDecks();
      const deckIds = snapshot.deckIds.filter(isDeckId);
      const nextDeckIds = Array.from(new Set(deckIds));
      if (nextDeckIds.length === 0) nextDeckIds.push("A");
      const decks = {
        A: restoreDeckSlot("A", snapshot.decks.A, defaults.A),
        B: restoreDeckSlot("B", snapshot.decks.B, defaults.B),
        C: restoreDeckSlot("C", snapshot.decks.C, defaults.C),
        D: restoreDeckSlot("D", snapshot.decks.D, defaults.D),
      };
      const validRole = (id: DeckId | null): DeckId | null =>
        id && nextDeckIds.includes(id) ? id : null;
      return {
        decks,
        deckIds: nextDeckIds,
        timbreDeckId: validRole(snapshot.timbreDeckId),
        structureDeckId: validRole(snapshot.structureDeckId),
        crossfade: clamp01(snapshot.crossfade),
        monitorEnabled: false,
        inferenceEnabled: snapshot.inferenceEnabled,
        mixRevision: state.mixRevision + 1,
      };
    }),
}));
