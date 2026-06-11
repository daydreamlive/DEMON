"use client";

import {
  fetchAndDecodeAudio,
  loadFixtureAudio,
  type DecodedFixture,
  type StemSourceMode,
} from "@/engine/audio/loadFixture";
import { podHttp } from "@/engine/podUrl";
import { useCustomTracksStore } from "@/store/useCustomTracksStore";

export interface DeckTrackManifest {
  name: string;
  stems: Partial<Record<"vocals" | "instruments", string>>;
  metadata: Record<string, unknown>;
}

export interface DeckTrackAssets {
  name: string;
  full: DecodedFixture;
  stems: Partial<Record<"vocals" | "instruments", DecodedFixture>>;
  manifest: DeckTrackManifest | null;
}

const assetCache = new Map<string, Promise<DeckTrackAssets>>();

interface LocalCustomTrackAssets {
  full: DecodedFixture;
  stems: Partial<Record<"vocals" | "instruments", DecodedFixture>>;
  persisted: boolean;
}

function sourceFor(
  assets: DeckTrackAssets,
  part: StemSourceMode,
): DecodedFixture | null {
  if (part === "full") return assets.full;
  return assets.stems[part] ?? null;
}

function loadLocalCustomTrackAssets(name: string): LocalCustomTrackAssets | null {
  const track = useCustomTracksStore.getState().tracks.get(name);
  if (!track) return null;
  const full = track.full ?? track.decoded;
  if (!full) return null;
  return {
    full,
    stems: track.stems
      ? {
          vocals: track.stems.vocals,
          instruments: track.stems.instruments,
        }
      : {},
    persisted: track.persisted,
  };
}

export function deckAssetSource(
  assets: DeckTrackAssets | undefined,
  part: StemSourceMode,
): DecodedFixture | null {
  return assets ? sourceFor(assets, part) : null;
}

async function fetchManifest(name: string): Promise<DeckTrackManifest | null> {
  const res = await fetch(
    podHttp(`/api/track_asset?name=${encodeURIComponent(name)}`),
  );
  if (res.status === 404) return null;
  if (!res.ok) {
    throw new Error(`Track asset lookup failed: ${res.status}`);
  }
  return (await res.json()) as DeckTrackManifest;
}

async function fetchStem(
  name: string,
  mode: "vocals" | "instruments",
): Promise<DecodedFixture | null> {
  const url = podHttp(
    `/api/track_stem?name=${encodeURIComponent(name)}&mode=${mode}`,
  );
  const res = await fetch(url);
  if (res.status === 404) return null;
  if (!res.ok) throw new Error(`Stem fetch failed: ${res.status}`);
  const bytes = await res.arrayBuffer();
  const blobUrl = URL.createObjectURL(new Blob([bytes]));
  try {
    return await fetchAndDecodeAudio(blobUrl);
  } finally {
    URL.revokeObjectURL(blobUrl);
  }
}

async function loadDeckTrackAssetsUncached(name: string): Promise<DeckTrackAssets> {
  const local = loadLocalCustomTrackAssets(name);
  if (
    local &&
    (!local.persisted || (local.stems.vocals && local.stems.instruments))
  ) {
    return { name, full: local.full, stems: local.stems, manifest: null };
  }

  const [full, manifest] = await Promise.all([
    local ? Promise.resolve(local.full) : loadFixtureAudio(name),
    fetchManifest(name).catch(() => null),
  ]);

  const stems: DeckTrackAssets["stems"] = { ...(local?.stems ?? {}) };
  const wanted: Array<"vocals" | "instruments"> = ["vocals", "instruments"];
  await Promise.all(
    wanted.map(async (mode) => {
      if (stems[mode]) return;
      if (manifest && !(mode in manifest.stems)) return;
      const decoded = await fetchStem(name, mode).catch(() => null);
      if (decoded) stems[mode] = decoded;
    }),
  );

  return { name, full, stems, manifest };
}

export function loadDeckTrackAssets(name: string): Promise<DeckTrackAssets> {
  const cached = assetCache.get(name);
  if (cached) return cached;
  const promise = loadDeckTrackAssetsUncached(name).catch((error) => {
    assetCache.delete(name);
    throw error;
  });
  assetCache.set(name, promise);
  return promise;
}

export function clearDeckAssetCache(name?: string): void {
  if (name) assetCache.delete(name);
  else assetCache.clear();
}
