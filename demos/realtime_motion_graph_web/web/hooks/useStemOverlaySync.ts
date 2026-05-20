"use client";

import { useEffect } from "react";

import { useCustomTracksStore } from "@/store/useCustomTracksStore";
import { usePerformanceStore } from "@/store/usePerformanceStore";
import { useSessionStore } from "@/store/useSessionStore";
import { useStemOverlayStore } from "@/store/useStemOverlayStore";

export function useStemOverlaySync() {
  const fixture = usePerformanceStore((s) => s.fixture);
  const player = useSessionStore((s) => s.player);
  const stems = useCustomTracksStore((s) =>
    fixture ? s.tracks.get(fixture)?.stems : undefined,
  );
  const vocalsEnabled = useStemOverlayStore((s) => s.enabled.vocals);
  const instrumentsEnabled = useStemOverlayStore((s) => s.enabled.instruments);
  const vocalsVolume = useStemOverlayStore((s) => s.volumes.vocals);
  const instrumentsVolume = useStemOverlayStore((s) => s.volumes.instruments);

  useEffect(() => {
    if (!player) return;
    if (!stems) {
      player.clearStemOverlays();
      return;
    }
    player.setStemOverlay("vocals", stems.vocals.interleaved, stems.vocals.channels);
    player.setStemOverlay(
      "instruments",
      stems.instruments.interleaved,
      stems.instruments.channels,
    );
  }, [fixture, player, stems]);

  useEffect(() => {
    if (!player) return;
    player.setStemOverlayVolume(
      "vocals",
      vocalsEnabled ? vocalsVolume : 0,
    );
    player.setStemOverlayVolume(
      "instruments",
      instrumentsEnabled ? instrumentsVolume : 0,
    );
  }, [
    player,
    vocalsEnabled,
    instrumentsEnabled,
    vocalsVolume,
    instrumentsVolume,
  ]);
}
