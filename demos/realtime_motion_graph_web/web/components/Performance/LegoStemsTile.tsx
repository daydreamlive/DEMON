"use client";

import { useCustomTracksStore } from "@/store/useCustomTracksStore";
import { usePerformanceStore } from "@/store/usePerformanceStore";
import { LEGO_TRACKS, labelForLegoTrack } from "@/types/lego";

export function LegoStemsTile() {
  const fixture = usePerformanceStore((s) => s.fixture);
  const customTrack = useCustomTracksStore((s) =>
    fixture ? s.tracks.get(fixture) : undefined,
  );

  const isCustom = Boolean(fixture && customTrack);

  const helper = !fixture
    ? "Pick or upload a track first."
    : !isCustom
      ? "LEGO generation is available when uploading your own tracks."
      : "Generate LEGO layers from the upload dialog before playback so the base model has VRAM headroom.";

  return (
    <div className="mixer-tile lego-tile" data-tile="lego">
      <div className="mixer-tile-label">LEGO Layers</div>
      <p className="lego-tile-note">{helper}</p>
      <div className="lego-layer-list">
        {LEGO_TRACKS.map((track) => {
          const prompt = customTrack?.legoPrompts?.[track] ?? "";
          const hasStem = Boolean(customTrack?.legoStems?.[track]);
          const trackStatus = customTrack?.legoStatus?.[track];
          const error = customTrack?.legoErrors?.[track];
          return (
            <div
              key={track}
              className={`lego-layer-row${hasStem ? " is-selected" : ""}`}
            >
              <div className="lego-layer-check">
                <span>{labelForLegoTrack(track)}</span>
              </div>
              <input
                className="lego-layer-prompt"
                type="text"
                value={prompt}
                readOnly
                aria-label={`${labelForLegoTrack(track)} LEGO prompt`}
                placeholder="Not generated"
              />
              <span
                className={`lego-layer-status${trackStatus ? ` is-${trackStatus}` : ""}`}
                title={error || undefined}
              >
                {trackStatus ?? "idle"}
              </span>
            </div>
          );
        })}
      </div>
    </div>
  );
}
