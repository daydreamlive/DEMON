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
  const legoStems = useCustomTracksStore((s) =>
    fixture ? s.tracks.get(fixture)?.legoStems : undefined,
  );
  const enabled = useStemOverlayStore((s) => s.enabled);
  const volumes = useStemOverlayStore((s) => s.volumes);

  useEffect(() => {
    if (!player) return;
    if (!stems && !legoStems) {
      player.clearStemOverlays();
      return;
    }
    player.clearStemOverlays();
    if (stems) {
      for (const [kind, stem] of Object.entries(stems)) {
        player.setStemOverlay(kind, stem.interleaved, stem.channels);
      }
    }
    if (legoStems) {
      for (const [kind, stem] of Object.entries(legoStems)) {
        player.setStemOverlay(`lego:${kind}`, stem.interleaved, stem.channels);
      }
    }
  }, [fixture, player, stems, legoStems]);

  useEffect(() => {
    if (!player) return;
    const keys = new Set([...Object.keys(enabled), ...Object.keys(volumes)]);
    keys.forEach((kind) => {
      const volume = volumes[kind] ?? 0.65;
      player.setStemOverlayVolume(kind, enabled[kind] ? volume : 0);
    });
  }, [player, enabled, volumes]);
}
