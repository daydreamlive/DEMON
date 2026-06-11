"use client";

import { useDeckAssets } from "@/hooks/useDeckAssets";
import { useDeckMonitor } from "@/hooks/useDeckMonitor";
import { useDeckServerSync } from "@/hooks/useDeckServerSync";
import { useDeckStore } from "@/store/useDeckStore";

export function useDeckRuntime(): void {
  const decks = useDeckStore((s) => s.decks);
  const crossfade = useDeckStore((s) => s.crossfade);
  const monitorEnabled = useDeckStore((s) => s.monitorEnabled);
  const inferenceEnabled = useDeckStore((s) => s.inferenceEnabled);
  const revision = useDeckStore((s) => s.mixRevision);
  const { assetsByDeck } = useDeckAssets(decks);

  useDeckMonitor({ decks, assetsByDeck, crossfade, enabled: monitorEnabled });
  useDeckServerSync({
    decks,
    crossfade,
    enabled: inferenceEnabled,
    revision,
  });
}
