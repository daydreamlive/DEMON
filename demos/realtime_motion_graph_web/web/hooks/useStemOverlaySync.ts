"use client";

import { useEffect } from "react";

import { SAMPLE_RATE } from "@demon/client";
import { useDeckAssets } from "@/hooks/useDeckAssets";
import type { DecodedFixture, StemOverlayKind } from "@/engine/audio/loadFixture";
import { useCustomTracksStore } from "@/store/useCustomTracksStore";
import { DECK_IDS, useDeckStore } from "@/store/useDeckStore";
import { usePerformanceStore } from "@/store/usePerformanceStore";
import { useSessionStore } from "@/store/useSessionStore";
import { useStemOverlayStore } from "@/store/useStemOverlayStore";

interface StemOverlayPart {
  source: DecodedFixture;
  gain: number;
}

function readSample(
  source: DecodedFixture,
  frame: number,
  channel: 0 | 1,
): number {
  if (source.frames <= 0) return 0;
  const wrapped = ((frame % source.frames) + source.frames) % source.frames;
  const sourceChannel = Math.min(channel, source.channels - 1);
  return source.interleaved[wrapped * source.channels + sourceChannel] ?? 0;
}

function mixStemOverlayParts(
  parts: StemOverlayPart[],
  targetFrames: number,
): DecodedFixture | null {
  const active = parts.filter((part) => part.gain > 0 && part.source.frames > 0);
  if (active.length === 0 || targetFrames <= 0) return null;

  const interleaved = new Float32Array(targetFrames * 2);
  for (const { source, gain } of active) {
    for (let frame = 0; frame < targetFrames; frame++) {
      const base = frame * 2;
      interleaved[base] += readSample(source, frame, 0) * gain;
      interleaved[base + 1] += readSample(source, frame, 1) * gain;
    }
  }
  return {
    interleaved,
    channels: 2,
    frames: targetFrames,
    sampleRate: SAMPLE_RATE,
  };
}

export function useStemOverlaySync() {
  const fixture = usePerformanceStore((s) => s.fixture);
  const player = useSessionStore((s) => s.player);
  const decks = useDeckStore((s) => s.decks);
  const { assetsByDeck } = useDeckAssets(decks);
  const stems = useCustomTracksStore((s) =>
    fixture ? s.tracks.get(fixture)?.stems : undefined,
  );
  const vocalsEnabled = useStemOverlayStore((s) => s.enabled.vocals);
  const instrumentsEnabled = useStemOverlayStore((s) => s.enabled.instruments);
  const vocalsVolume = useStemOverlayStore((s) => s.volumes.vocals);
  const instrumentsVolume = useStemOverlayStore((s) => s.volumes.instruments);

  useEffect(() => {
    if (!player) return;

    const targetFrames = Math.max(player.frameCount, 1);
    const buildParts = (kind: StemOverlayKind): StemOverlayPart[] => {
      const parts: StemOverlayPart[] = [];
      if (stems) {
        const gain =
          kind === "vocals"
            ? vocalsEnabled
              ? vocalsVolume
              : 0
            : instrumentsEnabled
              ? instrumentsVolume
              : 0;
        if (gain > 0) parts.push({ source: stems[kind], gain });
      }
      for (const id of DECK_IDS) {
        const deck = decks[id];
        if (!deck.trackName || deck.muted) continue;
        const source = assetsByDeck[id]?.stems[kind];
        const gain = deck.stemOverlayLevels?.[kind] ?? 0;
        if (source && gain > 0) parts.push({ source, gain });
      }
      return parts;
    };

    (["vocals", "instruments"] as const).forEach((kind) => {
      const mixed = mixStemOverlayParts(buildParts(kind), targetFrames);
      if (!mixed) {
        player.clearStemOverlay(kind);
        return;
      }
      player.setStemOverlay(kind, mixed.interleaved, mixed.channels);
      player.setStemOverlayVolume(kind, 1);
    });
  }, [
    assetsByDeck,
    decks,
    fixture,
    player,
    stems,
    vocalsEnabled,
    instrumentsEnabled,
    vocalsVolume,
    instrumentsVolume,
  ]);
}
