"use client";

import { useEffect, useMemo } from "react";

import {
  DECK_IDS,
  type DeckId,
  type DeckSlot,
} from "@/store/useDeckStore";
import { useSessionStore } from "@/store/useSessionStore";

const DECK_PARAM_DEBOUNCE_MS = 40;

type WireDeck = {
  id: string;
  track_name: string;
  source_part: "full" | "vocals" | "instruments";
  volume: number;
  muted: boolean;
  playing: boolean;
  side: "left" | "right";
};

function sourcePartForWire(part: DeckSlot["sourcePart"]): WireDeck["source_part"] {
  if (part === "vocals" || part === "instruments") return part;
  return "full";
}

export function useDeckServerSync({
  decks,
  crossfade,
  enabled,
  revision,
}: {
  decks: Record<DeckId, DeckSlot>;
  crossfade: number;
  enabled: boolean;
  revision: number;
}) {
  const payload = useMemo(() => {
    return DECK_IDS.flatMap((id): WireDeck[] => {
      const deck = decks[id];
      if (!deck?.trackName || deck.crossfadeSide === null) return [];
      return [{
        id,
        track_name: deck.trackName,
        source_part: sourcePartForWire(deck.sourcePart),
        volume: deck.volume,
        muted: deck.muted,
        playing: deck.playing,
        side: deck.crossfadeSide,
      }];
    });
  }, [decks, revision]);

  useEffect(() => {
    if (!enabled || payload.length === 0) return;
    const timer = window.setTimeout(() => {
      const { remote, status } = useSessionStore.getState();
      if (status !== "ready" || !remote) return;
      remote.sendDeckMixState(payload, crossfade);
    }, DECK_PARAM_DEBOUNCE_MS);
    return () => window.clearTimeout(timer);
  }, [crossfade, enabled, payload, revision]);
}
