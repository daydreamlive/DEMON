"use client";

import { useEffect } from "react";

import { listFixtures, pickDefaultFixture } from "@/engine/audio/loadFixture";
import { LOCAL_MODE } from "@/lib/runtime";
import { useDeckStore } from "@/store/useDeckStore";
import { usePerformanceStore } from "@/store/usePerformanceStore";
import { useSessionStore } from "@/store/useSessionStore";

export function useSeedDecks(): void {
  const sessionWsUrl = useSessionStore((s) => s.wsUrl);
  const activeFixture = usePerformanceStore((s) => s.fixture);

  useEffect(() => {
    if (!sessionWsUrl && !LOCAL_MODE) return;
    if (activeFixture) {
      useDeckStore.getState().ensureInitialDeck(activeFixture);
      return;
    }
    void listFixtures()
      .then((names) => {
        const fallback = pickDefaultFixture(names);
        if (fallback) {
          usePerformanceStore.getState().setFixture(fallback);
          useDeckStore.getState().ensureInitialDeck(fallback);
        }
      })
      .catch(() => {});
  }, [activeFixture, sessionWsUrl]);
}
