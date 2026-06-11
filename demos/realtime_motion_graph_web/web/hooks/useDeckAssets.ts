"use client";

import { useEffect, useMemo, useRef, useState } from "react";

import {
  loadDeckTrackAssets,
  type DeckTrackAssets,
} from "@/lib/audio/deckAssets";
import {
  DECK_IDS,
  type DeckId,
  type DeckSlot,
} from "@/store/useDeckStore";

export type DeckAssetStatus = "idle" | "loading" | "ready" | "failed";

export interface UseDeckAssetsResult {
  assetsByDeck: Partial<Record<DeckId, DeckTrackAssets>>;
  statuses: Partial<Record<string, DeckAssetStatus>>;
  errors: Partial<Record<string, string>>;
}

export function useDeckAssets(
  decks: Record<DeckId, DeckSlot>,
): UseDeckAssetsResult {
  const trackNames = useMemo(
    () =>
      Array.from(
        new Set(
          DECK_IDS.map((id) => decks[id].trackName).filter(
            (name): name is string => Boolean(name),
          ),
        ),
      ).sort(),
    [decks],
  );
  const [assets, setAssets] = useState<Record<string, DeckTrackAssets>>({});
  const [statuses, setStatuses] = useState<Partial<Record<string, DeckAssetStatus>>>({});
  const [errors, setErrors] = useState<Partial<Record<string, string>>>({});
  const loadingRef = useRef<Set<string>>(new Set());
  const assetsRef = useRef<Record<string, DeckTrackAssets>>({});
  const mountedRef = useRef(true);

  useEffect(() => {
    mountedRef.current = true;
    return () => {
      mountedRef.current = false;
    };
  }, []);

  useEffect(() => {
    assetsRef.current = assets;
  }, [assets]);

  useEffect(() => {
    for (const name of trackNames) {
      if (assetsRef.current[name] || loadingRef.current.has(name)) continue;
      loadingRef.current.add(name);
      setStatuses((s) => ({ ...s, [name]: "loading" }));
      void loadDeckTrackAssets(name)
        .then((loaded) => {
          if (!mountedRef.current) return;
          setAssets((s) => ({ ...s, [name]: loaded }));
          setStatuses((s) => ({ ...s, [name]: "ready" }));
          setErrors((s) => ({ ...s, [name]: undefined }));
        })
        .catch((e) => {
          if (!mountedRef.current) return;
          setStatuses((s) => ({ ...s, [name]: "failed" }));
          setErrors((s) => ({
            ...s,
            [name]: e instanceof Error ? e.message : String(e),
          }));
        })
        .finally(() => {
          loadingRef.current.delete(name);
        });
    }
  }, [trackNames]);

  const assetsByDeck = useMemo(() => {
    const out: Partial<Record<DeckId, DeckTrackAssets>> = {};
    for (const id of DECK_IDS) {
      const name = decks[id].trackName;
      if (name && assets[name]) out[id] = assets[name];
    }
    return out;
  }, [assets, decks]);

  return { assetsByDeck, statuses, errors };
}
