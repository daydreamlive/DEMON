"use client";

import { useEffect, useRef, useState } from "react";

import {
  decodeAudioFile,
  listFixtures,
  pickDefaultFixture,
  type DecodedFixture,
} from "@/engine/audio/loadFixture";
import { togglePauseAndAudio } from "@/engine/audio/togglePauseAndAudio";
import {
  applyConfig,
  captureRtmgConfig,
  getConfig,
  mergeConfig,
  type RtmgConfig,
} from "@/lib/config";
import { LOCAL_MODE } from "@/lib/runtime";
import { confirm } from "@/store/useConfirmStore";
import { useCurveStore } from "@/store/useCurveStore";
import { useCustomTracksStore } from "@/store/useCustomTracksStore";
import { useLoraStore } from "@/store/useLoraStore";
import { usePerformanceStore } from "@/store/usePerformanceStore";
import { useSessionStore } from "@/store/useSessionStore";
import {
  TIME_SIGNATURE_LABELS,
  VALID_KEYSCALES,
  VALID_TIME_SIGNATURES,
  isTimeSignature,
  type TimeSignature,
} from "@/types/engine";

import { AlmostReadyDialog } from "./AlmostReadyDialog";
import { MidiBadge } from "./MidiBadge";
import { RecordToggle } from "./RecordToggle";

export function OperatorStrip() {
  const [fixtures, setFixtures] = useState<string[]>([]);
  const fixture = usePerformanceStore((s) => s.fixture);
  const activeKey = usePerformanceStore((s) => s.activeKey);
  const activeTimeSignature = usePerformanceStore((s) => s.activeTimeSignature);
  const kiosk = usePerformanceStore((s) => s.kiosk);
  const paused = usePerformanceStore((s) => s.paused);
  const showKbdHints = usePerformanceStore((s) => s.showKbdHints);
  const smooth = usePerformanceStore((s) => s.smooth);
  const smoothMs = usePerformanceStore((s) => s.smoothMs);
  const lufsOn = usePerformanceStore((s) => s.lufsOn);
  const loopOn = usePerformanceStore((s) => s.loopOn);
  const setFixture = usePerformanceStore((s) => s.setFixture);
  const setKey = usePerformanceStore((s) => s.setKey);
  const setTimeSignature = usePerformanceStore((s) => s.setTimeSignature);
  const toggleKiosk = usePerformanceStore((s) => s.toggleKiosk);
  const overlayOpen = useCurveStore((s) => s.overlayOpen);
  const toggleOverlay = useCurveStore((s) => s.toggleOverlay);
  const toggleKbdHints = usePerformanceStore((s) => s.toggleKbdHints);
  const toggleSmooth = usePerformanceStore((s) => s.toggleSmooth);
  const setSmoothMs = usePerformanceStore((s) => s.setSmoothMs);
  const toggleLufs = usePerformanceStore((s) => s.toggleLufs);
  const toggleLoop = usePerformanceStore((s) => s.toggleLoop);

  const customNames = useCustomTracksStore((s) => s.names);
  const addCustomTrack = useCustomTracksStore((s) => s.add);

  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const configFileInputRef = useRef<HTMLInputElement | null>(null);
  const [uploading, setUploading] = useState(false);
  // Mirror of AudioSourceCrate's pending-upload state. The "Almost
  // Ready" dialog gates the actual fixture swap so the previously
  // playing track keeps playing if the user cancels.
  const [pending, setPending] = useState<{
    decoded: DecodedFixture;
    fileName: string;
    wasTrimmed: boolean;
    originalFile: File;
  } | null>(null);

  // The bottom-left <AudioSourceCrate /> is the primary track-picker; we
  // keep this dropdown + upload icon here as the power-user fallback that
  // power users will keep relying on while the advanced controls strip
  // is open. Both surfaces drive the same setFixture() / addCustomTrack()
  // path, so picking from either re-triggers useFixtureSwap identically.
  // Daydream-webapp queue-admit gate: standalone DEMON has no queue
  // (LOCAL_MODE), so we skip the wait there.
  const sessionWsUrl = useSessionStore((s) => s.wsUrl);
  useEffect(() => {
    if (!sessionWsUrl && !LOCAL_MODE) return;
    void listFixtures()
      .then((names) => {
        setFixtures(names);
        const def = pickDefaultFixture(names);
        if (!usePerformanceStore.getState().fixture && def) {
          setFixture(def);
        }
      })
      .catch(() => setFixtures([]));
  }, [setFixture, sessionWsUrl]);

  // Push LUFS state to the live AudioPlayer. Re-runs whenever the user
  // toggles, and whenever a new player instance appears (session
  // start / restart) so the setting carries across sessions without
  // the user re-toggling.
  const player = useSessionStore((s) => s.player);
  useEffect(() => {
    if (!player) return;
    player.setLufs(lufsOn);
  }, [player, lufsOn]);

  // Same pattern for loop. Default is on; flipping off makes the worklet
  // freeze at end-of-buffer and emit silence instead of wrapping.
  useEffect(() => {
    if (!player) return;
    player.setLoop(loopOn);
  }, [player, loopOn]);

  // End-of-buffer → auto-pause. Only fires when loop is off (the
  // worklet's one-shot only emits in that mode). Suspends the audio
  // context and flips the performance store's paused flag so the
  // play/pause button immediately shows ▶.
  useEffect(() => {
    if (!player) return;
    return player.onEndOfBuffer(() => {
      void player.ctx?.suspend();
      usePerformanceStore.getState().setPaused(true);
    });
  }, [player]);

  // Import config — load an exported RtmgConfig JSON (or a demon-public-
  // demo share file, which is RtmgConfig + a `tracks` extension that
  // mergeConfig silently drops). Layers on top of the live config so
  // operator-edited fields the user didn't touch in the imported file
  // keep their current values.
  async function onConfigFilePicked(file: File): Promise<void> {
    const { setStatus } = useSessionStore.getState();
    try {
      const text = await file.text();
      const parsed = JSON.parse(text) as Partial<RtmgConfig>;
      if (typeof parsed !== "object" || parsed === null || Array.isArray(parsed)) {
        throw new Error("config root must be an object");
      }
      const merged = mergeConfig(getConfig(), parsed);
      applyConfig(merged);
      setStatus(useSessionStore.getState().status, `Imported ${file.name}`);
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e);
      setStatus(useSessionStore.getState().status, `Import failed: ${msg}`);
    }
  }

  // Export config — snapshot the live stores into an RtmgConfig and
  // trigger a JSON download. Filename includes a timestamp so multiple
  // exports in a session don't collide.
  function onExportConfig(): void {
    const snapshot = captureRtmgConfig();
    const blob = new Blob([JSON.stringify(snapshot, null, 2)], {
      type: "application/json",
    });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    const ts = new Date().toISOString().replace(/[:.]/g, "-").slice(0, 19);
    a.download = `demon-config-${ts}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
    const { setStatus } = useSessionStore.getState();
    setStatus(useSessionStore.getState().status, `Exported config`);
  }

  async function onFilePicked(file: File) {
    const { setStatus } = useSessionStore.getState();
    setUploading(true);
    setStatus(useSessionStore.getState().status, `Loading ${file.name}…`);
    try {
      const { decoded, wasTrimmed } = await decodeAudioFile(file);
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
      // Defer addCustomTrack + setFixture to commitPending so cancelling
      // leaves the previously playing track intact.
      setPending({ decoded, fileName: chosen, wasTrimmed, originalFile: file });
      setStatus(useSessionStore.getState().status, "");
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e);
      setStatus(useSessionStore.getState().status, `Upload failed: ${msg}`);
    } finally {
      setUploading(false);
    }
  }

  function commitPending(
    keyOverride: string | null,
    timeSignatureOverride: TimeSignature | null,
  ) {
    if (!pending) return;
    const { decoded, fileName, originalFile } = pending;
    addCustomTrack(fileName, decoded, originalFile);
    const perf = usePerformanceStore.getState();
    if (keyOverride) {
      perf.setPendingKeyOverride(keyOverride);
      perf.setKey(keyOverride);
    }
    if (timeSignatureOverride) {
      perf.setPendingTimeSignatureOverride(timeSignatureOverride);
      perf.setTimeSignature(timeSignatureOverride);
    }
    setFixture(fileName);
    setPending(null);
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
        data-dd-tooltip={uploading ? "Decoding…" : "Upload audio track"}
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
        title="Musical key — sidecar / auto-detected; changes apply immediately"
        value={activeKey}
        onChange={async (e) => {
          const newKey = e.target.value;
          if (newKey === activeKey) return;
          // Capture the element synchronously: after the await the
          // SyntheticEvent's currentTarget may be cleared.
          const select = e.currentTarget;
          // Surface what this control actually does. Users were reading
          // it as a song-pitch transposer (which it isn't) — the
          // confirm is a one-off "are you sure" with the explanation
          // attached, so the action stays one click away but the
          // misconception gets corrected before the change applies.
          const ok = await confirm({
            title: "Change key",
            message: `Change key to "${newKey}"?\n\nThis tells the model what key the song is in. It does NOT change the song's pitch or transpose the audio.`,
            confirmLabel: "Change key",
          });
          if (!ok) {
            // Bounce the <select> back to the previous value so the UI
            // reflects the cancelled state. Direct DOM write because
            // the controlled `value` prop didn't change, so React
            // won't re-sync on its own.
            select.value = activeKey;
            return;
          }
          setKey(newKey);
          const remote = useSessionStore.getState().remote;
          if (remote) {
            const { promptA } = usePerformanceStore.getState();
            remote.sendPrompt(promptA, newKey);
          }
        }}
      >
        {VALID_KEYSCALES.map((k) => (
          <option key={k} value={k}>
            {k}
          </option>
        ))}
      </select>
      <select
        id="time-sig-select"
        className="fixture-select"
        title="Time signature — sidecar / default; tells the model the song's meter (does not change tempo or beat grid)"
        value={activeTimeSignature}
        onChange={async (e) => {
          const newTs = e.target.value;
          if (!isTimeSignature(newTs) || newTs === activeTimeSignature) return;
          const select = e.currentTarget;
          // Same confirm-on-change UX as the key dropdown: meter is a
          // model hint, not a tempo/beat-grid edit. The wording mirrors
          // the keyscale confirm exactly so operators read both
          // controls the same way.
          const ok = await confirm({
            title: "Change time signature",
            message: `Change time signature to "${TIME_SIGNATURE_LABELS[newTs]}"?\n\nThis tells the model the song's meter. It does NOT change the song's tempo or beat grid.`,
            confirmLabel: "Change time signature",
          });
          if (!ok) {
            select.value = activeTimeSignature;
            return;
          }
          setTimeSignature(newTs);
          const remote = useSessionStore.getState().remote;
          if (remote) {
            const { promptA, activeKey: ak } = usePerformanceStore.getState();
            remote.sendPrompt(promptA, ak, newTs);
          }
        }}
      >
        {VALID_TIME_SIGNATURES.map((ts) => (
          <option key={ts} value={ts}>
            {TIME_SIGNATURE_LABELS[ts]}
          </option>
        ))}
      </select>
      <button
        id="kiosk-toggle"
        className={`pause-btn${kiosk ? " active" : ""}`}
        data-midi-learn="kiosk_toggle"
        data-dd-tooltip="Toggle kiosk mode — auto-hide cursor + idle reset (right-click to MIDI-learn)"
        type="button"
        onClick={toggleKiosk}
      >
        KIOSK
      </button>
      <button
        type="button"
        className={`pause-btn${overlayOpen ? " active" : ""}`}
        data-midi-learn="schedule_curves_toggle"
        data-dd-tooltip="Open the curve scheduler — draw param automation against the track (right-click to MIDI-learn)"
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
        data-dd-tooltip={
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
        className="fixture-select"
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
        className={`pause-btn${lufsOn ? " active" : ""}`}
        data-dd-tooltip={
          lufsOn
            ? "Loudness match ON — quieter passages are boosted to match the loudest seen (peak-clamped at –1 dBTP). Resets on track change. Click to disable."
            : "Loudness match: continuously meter LUFS, track the running max, boost quieter passages so they hit the loudest level seen this track. Never attenuates."
        }
        aria-pressed={lufsOn}
        onClick={toggleLufs}
      >
        LUFS: {lufsOn ? "MATCH" : "OFF"}
      </button>
      <button
        type="button"
        className={`pause-btn${showKbdHints ? " active" : ""}`}
        data-dd-tooltip="Show keyboard-shortcut hints under each slider"
        aria-pressed={showKbdHints}
        onClick={toggleKbdHints}
      >
        KBD: {showKbdHints ? "ON" : "OFF"}
      </button>
      <button
        type="button"
        className="pause-btn"
        data-dd-tooltip="Reset all sliders + LoRAs to defaults. Does NOT touch MIDI mapping, automation curves, or persisted UI prefs."
        onClick={async () => {
          const ok = await confirm({
            title: "Reset",
            message: "Reset sliders and LoRAs to defaults?",
            confirmLabel: "Reset",
            variant: "danger",
          });
          if (!ok) return;
          usePerformanceStore.getState().resetToDefaults();
          useLoraStore.getState().reset();
        }}
      >
        RESET
      </button>
      <RecordToggle />
      <button
        type="button"
        className="pause-btn pause-btn--right"
        data-midi-learn="seek_start"
        data-dd-tooltip="Seek to beginning (right-click to MIDI-learn)"
        aria-label="Seek to beginning"
        onClick={() => {
          const p = useSessionStore.getState().player;
          p?.seek(0);
        }}
      >
        ⏮
      </button>
      <button
        id="pause-btn"
        className="pause-btn"
        data-midi-learn="pause"
        data-dd-tooltip="Pause/Resume (right-click to MIDI-learn)"
        type="button"
        onClick={togglePauseAndAudio}
      >
        {paused ? "▶" : "⏸"}
      </button>
      <button
        type="button"
        className={`pause-btn${loopOn ? " active" : ""}`}
        data-midi-learn="loop_toggle"
        data-dd-tooltip={
          loopOn
            ? "Loop ON — playhead wraps at end-of-buffer (right-click to MIDI-learn)"
            : "Loop OFF — playback stops at end-of-buffer; click ⏮ to restart"
        }
        aria-label="Toggle loop"
        aria-pressed={loopOn}
        onClick={toggleLoop}
      >
        ↻
      </button>

      {/* Import + Export config — far right of the strip, local-mode
          only. These are developer / power-user affordances for
          dialing in a pod's config.json without SSH-editing; the live
          deployed product hides them entirely (LOCAL_MODE === false).
          mergeConfig drops unknown top-level keys so demon-public-demo
          share files (with the `tracks` extension) import as a base
          config without error. */}
      {LOCAL_MODE && (
        <>
          <button
            type="button"
            className="pause-btn"
            style={{ marginLeft: "auto" }}
            data-dd-tooltip="Import config from JSON"
            aria-label="Import config"
            onClick={() => configFileInputRef.current?.click()}
          >
            Import
          </button>
          <button
            type="button"
            className="pause-btn"
            data-dd-tooltip="Download current config as JSON"
            aria-label="Export config"
            onClick={onExportConfig}
          >
            Export
          </button>
          <input
            ref={configFileInputRef}
            type="file"
            accept=".json,application/json"
            style={{ display: "none" }}
            onChange={(e) => {
              const file = e.target.files?.[0];
              e.target.value = "";
              if (file) void onConfigFilePicked(file);
            }}
          />
        </>
      )}

      {pending && (
        <AlmostReadyDialog
          fileName={pending.fileName}
          wasTrimmed={pending.wasTrimmed}
          defaultKey={activeKey}
          defaultTimeSignature={activeTimeSignature}
          onContinue={({ keyOverride, timeSignatureOverride }) =>
            commitPending(keyOverride, timeSignatureOverride)
          }
          onPickAnother={() => {
            setPending(null);
            setTimeout(() => fileInputRef.current?.click(), 0);
          }}
          onClose={() => setPending(null)}
        />
      )}
    </div>
  );
}
