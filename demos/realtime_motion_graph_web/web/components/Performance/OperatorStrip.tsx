"use client";

import { useEffect, useRef, useState } from "react";

import { decodeAudioFile, listFixtures } from "@/engine/audio/loadFixture";
import { togglePauseAndAudio } from "@/engine/audio/togglePauseAndAudio";
import { useCurveStore } from "@/store/useCurveStore";
import { useCustomTracksStore } from "@/store/useCustomTracksStore";
import { usePerformanceStore } from "@/store/usePerformanceStore";
import { useSessionStore } from "@/store/useSessionStore";
import { VALID_KEYSCALES } from "@/types/engine";

import { MidiBadge } from "./MidiBadge";
import { RecordToggle } from "./RecordToggle";

export function OperatorStrip() {
  const [fixtures, setFixtures] = useState<string[]>([]);
  const fixture = usePerformanceStore((s) => s.fixture);
  const activeKey = usePerformanceStore((s) => s.activeKey);
  const kiosk = usePerformanceStore((s) => s.kiosk);
  const paused = usePerformanceStore((s) => s.paused);
  const showKbdHints = usePerformanceStore((s) => s.showKbdHints);
  const smooth = usePerformanceStore((s) => s.smooth);
  const smoothMs = usePerformanceStore((s) => s.smoothMs);
  const setFixture = usePerformanceStore((s) => s.setFixture);
  const setKey = usePerformanceStore((s) => s.setKey);
  const toggleKiosk = usePerformanceStore((s) => s.toggleKiosk);
  const overlayOpen = useCurveStore((s) => s.overlayOpen);
  const toggleOverlay = useCurveStore((s) => s.toggleOverlay);
  const toggleKbdHints = usePerformanceStore((s) => s.toggleKbdHints);
  const toggleSmooth = usePerformanceStore((s) => s.toggleSmooth);
  const setSmoothMs = usePerformanceStore((s) => s.setSmoothMs);

  const customNames = useCustomTracksStore((s) => s.names);
  const addCustomTrack = useCustomTracksStore((s) => s.add);

  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const [uploading, setUploading] = useState(false);

  // The bottom-left <AudioSourceCrate /> is the primary track-picker; we
  // keep this dropdown + upload icon here as the power-user fallback that
  // power users will keep relying on while the advanced controls strip
  // is open. Both surfaces drive the same setFixture() / addCustomTrack()
  // path, so picking from either re-triggers useFixtureSwap identically.
  const sessionWsUrl = useSessionStore((s) => s.wsUrl);
  useEffect(() => {
    // Pre-admit the proxy returns 401, so don't fetch until the queue
    // hands us a wsUrl.
    if (!sessionWsUrl) return;
    void listFixtures()
      .then((names) => {
        setFixtures(names);
        if (!usePerformanceStore.getState().fixture && names[0]) {
          setFixture(names[0]);
        }
      })
      .catch(() => setFixtures([]));
  }, [setFixture, sessionWsUrl]);

  async function onFilePicked(file: File) {
    const { setStatus } = useSessionStore.getState();
    setUploading(true);
    setStatus(useSessionStore.getState().status, `Loading ${file.name}…`);
    try {
      const decoded = await decodeAudioFile(file);
      // De-collide names: appending an index when the same filename is
      // uploaded twice keeps prior uploads selectable while letting the
      // new one win the dropdown. The decoded buffer is what the pod
      // actually consumes, so the displayed name is purely for the UI.
      const baseName = file.name;
      let chosen = baseName;
      let i = 1;
      while (useCustomTracksStore.getState().has(chosen)) {
        chosen = `${baseName} (${i++})`;
      }
      addCustomTrack(chosen, decoded);
      setFixture(chosen);
      setStatus(useSessionStore.getState().status, "");
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e);
      setStatus(useSessionStore.getState().status, `Upload failed: ${msg}`);
    } finally {
      setUploading(false);
    }
  }

  // The pod's WS URL is allocated by the queue and not user-editable.
  return (
    <div className="install-section-operator">
      <select
        id="fixture-select"
        className="fixture-select"
        title="Audio source — pick a track or one of your uploaded tracks"
        value={fixture}
        onChange={(e) => setFixture(e.target.value)}
      >
        {fixtures.length === 0 && customNames.length === 0 && <option>—</option>}
        {customNames.length > 0 && (
          <optgroup label="Your uploads">
            {customNames.map((n) => (
              <option key={`u:${n}`} value={n}>
                {n}
              </option>
            ))}
          </optgroup>
        )}
        {fixtures.length > 0 && (
          <optgroup label="Tracks">
            {fixtures.map((f) => (
              <option key={f} value={f}>
                {f}
              </option>
            ))}
          </optgroup>
        )}
      </select>
      <button
        type="button"
        className="pause-btn"
        title={uploading ? "Decoding…" : "Upload audio track"}
        aria-label="Upload audio track"
        disabled={uploading}
        onClick={() => fileInputRef.current?.click()}
      >
        <svg
          viewBox="0 0 16 16"
          width={14}
          height={14}
          fill="none"
          stroke="currentColor"
          strokeWidth={1.4}
          strokeLinecap="round"
          strokeLinejoin="round"
          aria-hidden="true"
        >
          <path d="M8 10V2" />
          <path d="M4.5 5.5L8 2l3.5 3.5" />
          <path d="M2.5 10v3a1 1 0 0 0 1 1h9a1 1 0 0 0 1-1v-3" />
        </svg>
      </button>
      <input
        ref={fileInputRef}
        type="file"
        accept="audio/*,.mp3,.wav,.flac,.ogg,.m4a,.aac"
        style={{ display: "none" }}
        onChange={(e) => {
          const file = e.target.files?.[0];
          // Reset the input so picking the same file twice still fires
          // onChange (browsers debounce identical selections otherwise).
          e.target.value = "";
          if (file) void onFilePicked(file);
        }}
      />
      <select
        id="key-select"
        className="fixture-select"
        title="Musical key — auto-detected; override re-sends prompt"
        value={activeKey}
        onChange={(e) => setKey(e.target.value)}
      >
        {VALID_KEYSCALES.map((k) => (
          <option key={k} value={k}>
            {k}
          </option>
        ))}
      </select>
      <button
        id="kiosk-toggle"
        className={`pause-btn${kiosk ? " active" : ""}`}
        data-midi-learn="kiosk_toggle"
        title="Toggle kiosk mode — auto-hide cursor + idle reset (right-click to MIDI-learn)"
        type="button"
        onClick={toggleKiosk}
      >
        KIOSK
      </button>
      <button
        type="button"
        className={`pause-btn${overlayOpen ? " active" : ""}`}
        data-midi-learn="schedule_curves_toggle"
        title="Open the curve scheduler — draw param automation against the track (right-click to MIDI-learn)"
        onClick={toggleOverlay}
      >
        SCHEDULE CURVES
      </button>
      <div id="install-midi-slot">
        <MidiBadge />
      </div>
      <button
        type="button"
        className={`pause-btn${smooth ? " active" : ""}`}
        title={
          smooth
            ? `Smooth slider transitions over ${smoothMs} ms — click to disable`
            : "Smooth slider transitions: slider drags + MIDI knob movement glide to their target over the chosen duration. The visual stays instant."
        }
        aria-pressed={smooth}
        onClick={toggleSmooth}
      >
        SMOOTH: {smooth ? `${(smoothMs / 1000).toFixed(smoothMs < 1000 ? 2 : 1)}s` : "OFF"}
      </button>
      <select
        className="key-select"
        value={String(smoothMs)}
        disabled={!smooth}
        onChange={(e) => setSmoothMs(parseInt(e.target.value, 10))}
        title="Slider transition duration. Only applies when SMOOTH is ON."
      >
        {[100, 250, 500, 1000, 1500, 2000, 3000, 5000].map((ms) => (
          <option key={ms} value={ms}>
            {ms < 1000 ? `${ms}ms` : `${ms / 1000}s`}
          </option>
        ))}
      </select>
      <button
        type="button"
        className={`pause-btn${showKbdHints ? " active" : ""}`}
        title="Show keyboard-shortcut hints under each slider"
        aria-pressed={showKbdHints}
        onClick={toggleKbdHints}
      >
        KBD: {showKbdHints ? "ON" : "OFF"}
      </button>
      <RecordToggle />
      <button
        id="pause-btn"
        className="pause-btn pause-btn--right"
        data-midi-learn="pause"
        title="Pause/Resume (right-click to MIDI-learn)"
        type="button"
        onClick={togglePauseAndAudio}
      >
        {paused ? "▶" : "⏸"}
      </button>
    </div>
  );
}
