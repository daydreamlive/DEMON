import type { DecodedFixture } from "@/engine/audio/loadFixture";
import { SAMPLE_RATE } from "@demon/client";
import {
  deckPositionSec,
  type DeckId,
  type DeckSlot,
} from "@/store/useDeckStore";

import { deckAssetSource, type DeckTrackAssets } from "./deckAssets";

export const DECK_MIX_POOL_FRAMES = 9600;

export interface DeckMixInput {
  decks: Record<DeckId, DeckSlot>;
  assets: Partial<Record<DeckId, DeckTrackAssets>>;
  crossfade: number;
  maxDurationS: number;
  nowMs?: number;
  normalizePeak?: number;
  requirePlaying?: boolean;
}

export interface DeckMixResult {
  interleaved: Float32Array;
  channels: 2;
  frames: number;
  sampleRate: number;
  activeDecks: DeckId[];
  peak: number;
  appliedGain: number;
}

function clamp01(value: number): number {
  if (!Number.isFinite(value)) return 0;
  return Math.max(0, Math.min(1, value));
}

function poolAlign(frames: number): number {
  return Math.floor(frames / DECK_MIX_POOL_FRAMES) * DECK_MIX_POOL_FRAMES;
}

function crossfadeGain(deck: DeckSlot, crossfade: number): number {
  if (deck.crossfadeSide === null) return 0;
  const x = clamp01(crossfade);
  return deck.crossfadeSide === "left" ? 1 - x : x;
}

export function effectiveDeckGain(
  deck: DeckSlot,
  crossfade: number,
  anySolo: boolean,
  requirePlaying = true,
): number {
  if ((requirePlaying && !deck.playing) || !deck.trackName || deck.muted) return 0;
  if (anySolo && !deck.solo) return 0;
  return clamp01(deck.volume) * crossfadeGain(deck, crossfade);
}

function readSample(
  source: DecodedFixture,
  frame: number,
  channel: 0 | 1,
): number {
  if (source.frames <= 0) return 0;
  const wrapped = ((frame % source.frames) + source.frames) % source.frames;
  const srcChannel = Math.min(channel, source.channels - 1);
  return source.interleaved[wrapped * source.channels + srcChannel] ?? 0;
}

function measurePeak(buffer: Float32Array): number {
  let peak = 0;
  for (let i = 0; i < buffer.length; i++) {
    const abs = Math.abs(buffer[i]);
    if (abs > peak) peak = abs;
  }
  return peak;
}

export function renderDeckMix(input: DeckMixInput): DeckMixResult | null {
  const nowMs = input.nowMs;
  const anySolo = Object.values(input.decks).some((d) => d.solo);
  const active = Object.values(input.decks)
    .map((deck) => {
      const assets = input.assets[deck.id];
      const source = deckAssetSource(assets, deck.sourcePart);
      const gain = effectiveDeckGain(
        deck,
        input.crossfade,
        anySolo,
        input.requirePlaying ?? false,
      );
      return source && gain > 0 ? { deck, source, gain } : null;
    })
    .filter((v): v is { deck: DeckSlot; source: DecodedFixture; gain: number } =>
      v !== null,
    );

  if (active.length === 0) return null;

  const longest = Math.max(...active.map(({ source }) => source.frames));
  const capFrames = Math.max(
    DECK_MIX_POOL_FRAMES,
    Math.floor(input.maxDurationS * SAMPLE_RATE),
  );
  const frames = poolAlign(Math.min(longest, capFrames));
  if (frames <= 0) return null;

  const out = new Float32Array(frames * 2);
  for (const { deck, source, gain } of active) {
    const startFrame = Math.floor(deckPositionSec(deck, nowMs) * SAMPLE_RATE);
    for (let frame = 0; frame < frames; frame++) {
      const srcFrame = startFrame + frame;
      const base = frame * 2;
      out[base] += readSample(source, srcFrame, 0) * gain;
      out[base + 1] += readSample(source, srcFrame, 1) * gain;
    }
  }

  const peak = measurePeak(out);
  const normalizePeak = input.normalizePeak ?? 0.98;
  const appliedGain = peak > normalizePeak ? normalizePeak / peak : 1;
  if (appliedGain < 1) {
    for (let i = 0; i < out.length; i++) out[i] *= appliedGain;
  }

  return {
    interleaved: out,
    channels: 2,
    frames,
    sampleRate: SAMPLE_RATE,
    activeDecks: active.map(({ deck }) => deck.id),
    peak,
    appliedGain,
  };
}
