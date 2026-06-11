"use client";

import { useEffect, useRef } from "react";

import { SAMPLE_RATE } from "@demon/client";
import { deckAssetSource, type DeckTrackAssets } from "@/lib/audio/deckAssets";
import { effectiveDeckGain } from "@/lib/audio/deckMixer";
import {
  DECK_IDS,
  deckPositionSec,
  type DeckId,
  type DeckSlot,
} from "@/store/useDeckStore";

interface MonitorEntry {
  key: string;
  source: AudioBufferSourceNode;
  gain: GainNode;
}

function toAudioBuffer(ctx: AudioContext, decoded: {
  interleaved: Float32Array;
  channels: number;
  frames: number;
}): AudioBuffer {
  const buffer = ctx.createBuffer(2, decoded.frames, SAMPLE_RATE);
  const left = buffer.getChannelData(0);
  const right = buffer.getChannelData(1);
  for (let i = 0; i < decoded.frames; i++) {
    left[i] = decoded.interleaved[i * decoded.channels] ?? 0;
    right[i] = decoded.interleaved[i * decoded.channels + Math.min(1, decoded.channels - 1)] ?? left[i];
  }
  return buffer;
}

function stopEntry(entry: MonitorEntry | undefined): void {
  if (!entry) return;
  try {
    entry.source.stop();
  } catch {}
  try {
    entry.source.disconnect();
  } catch {}
  try {
    entry.gain.disconnect();
  } catch {}
}

export function useDeckMonitor({
  decks,
  assetsByDeck,
  crossfade,
  enabled,
}: {
  decks: Record<DeckId, DeckSlot>;
  assetsByDeck: Partial<Record<DeckId, DeckTrackAssets>>;
  crossfade: number;
  enabled: boolean;
}) {
  const ctxRef = useRef<AudioContext | null>(null);
  const entriesRef = useRef<Partial<Record<DeckId, MonitorEntry>>>({});

  useEffect(() => {
    if (!enabled) {
      for (const id of DECK_IDS) {
        stopEntry(entriesRef.current[id]);
        delete entriesRef.current[id];
      }
      return;
    }

    if (!ctxRef.current) {
      ctxRef.current = new AudioContext({
        sampleRate: SAMPLE_RATE,
        latencyHint: "interactive",
      });
    }
    const ctx = ctxRef.current;
    void ctx.resume().catch(() => {});

    const anySolo = Object.values(decks).some((deck) => deck.solo);
    for (const id of DECK_IDS) {
      const deck = decks[id];
      const assets = assetsByDeck[id];
      const source = deckAssetSource(assets, deck.sourcePart);
      const gainValue = effectiveDeckGain(deck, crossfade, anySolo);
      const key = [
        deck.trackName ?? "",
        deck.sourcePart,
        source?.frames ?? 0,
        source?.interleaved.byteLength ?? 0,
        deck.playing ? "play" : "stop",
        deck.positionSec.toFixed(3),
        deck.lastStartedAtMs ?? "",
      ].join("|");
      const existing = entriesRef.current[id];

      if (!source || !deck.playing) {
        stopEntry(existing);
        delete entriesRef.current[id];
        continue;
      }

      if (!existing || existing.key !== key) {
        stopEntry(existing);
        const audioBuffer = toAudioBuffer(ctx, source);
        const node = ctx.createBufferSource();
        const gain = ctx.createGain();
        node.buffer = audioBuffer;
        node.loop = true;
        gain.gain.value = gainValue;
        node.connect(gain);
        gain.connect(ctx.destination);
        const offset = deckPositionSec(deck) % Math.max(0.001, audioBuffer.duration);
        try {
          node.start(0, offset);
        } catch {}
        entriesRef.current[id] = { key, source: node, gain };
      } else {
        existing.gain.gain.setTargetAtTime(gainValue, ctx.currentTime, 0.025);
      }
    }
  }, [assetsByDeck, crossfade, decks, enabled]);

  useEffect(() => {
    return () => {
      for (const id of DECK_IDS) stopEntry(entriesRef.current[id]);
      entriesRef.current = {};
      void ctxRef.current?.close().catch(() => {});
      ctxRef.current = null;
    };
  }, []);
}
