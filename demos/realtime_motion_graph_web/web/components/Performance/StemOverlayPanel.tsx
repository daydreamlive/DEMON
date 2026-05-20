"use client";

import { useCustomTracksStore } from "@/store/useCustomTracksStore";
import { usePerformanceStore } from "@/store/usePerformanceStore";
import { useStemOverlayStore } from "@/store/useStemOverlayStore";
import type { StemOverlayKind } from "@/engine/audio/loadFixture";

const LABELS: Record<StemOverlayKind, string> = {
  vocals: "Vocals",
  instruments: "Instruments",
};

export function StemOverlayPanel() {
  const fixture = usePerformanceStore((s) => s.fixture);
  const kiosk = usePerformanceStore((s) => s.kiosk);
  const track = useCustomTracksStore((s) =>
    fixture ? s.tracks.get(fixture) : undefined,
  );
  const sourceMode = track?.sourceMode;
  const status = track?.stemStatus;
  const error = track?.stemError;
  const stemsReady = Boolean(track?.stems);
  const enabled = useStemOverlayStore((s) => s.enabled);
  const volumes = useStemOverlayStore((s) => s.volumes);
  const toggle = useStemOverlayStore((s) => s.toggle);
  const setEnabled = useStemOverlayStore((s) => s.setEnabled);
  const setVolume = useStemOverlayStore((s) => s.setVolume);

  if (kiosk || !sourceMode) return null;

  const summary =
    status === "processing"
      ? "Ripping stems..."
      : status === "failed"
        ? "Stem rip failed"
        : stemsReady
          ? `Inference source: ${sourceMode}`
          : "Stems will load on play";

  const setLayerVolume = (kind: StemOverlayKind, value: number) => {
    setVolume(kind, value);
    setEnabled(kind, value > 0);
  };

  return (
    <section className="stem-overlay-panel" aria-label="Stem overlay layers">
      <div className="stem-overlay-head">
        <span className="stem-overlay-title">Stem layers</span>
        <span className="stem-overlay-summary" title={error || summary}>
          {summary}
        </span>
      </div>
      <div className="stem-overlay-controls">
        {(["vocals", "instruments"] as StemOverlayKind[]).map((kind) => (
          <div key={kind} className="stem-overlay-row">
            <button
              type="button"
              className={`stem-overlay-toggle${enabled[kind] ? " active" : ""}`}
              disabled={!stemsReady}
              aria-pressed={enabled[kind]}
              onClick={() => toggle(kind)}
            >
              {LABELS[kind]}
            </button>
            <input
              className="stem-overlay-volume"
              type="range"
              min={0}
              max={1.5}
              step={0.01}
              value={enabled[kind] ? volumes[kind] : 0}
              disabled={!stemsReady}
              aria-label={`${LABELS[kind]} overlay volume`}
              onChange={(e) => setLayerVolume(kind, Number(e.target.value))}
            />
          </div>
        ))}
      </div>
    </section>
  );
}
