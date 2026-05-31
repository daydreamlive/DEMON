"use client";

import { useEffect, useRef, useState } from "react";

import { togglePauseAndAudio } from "@/engine/audio/togglePauseAndAudio";
import {
  applyConfig,
  captureRtmgConfig,
  getConfig,
  mergeConfig,
  type RtmgConfig,
} from "@/lib/config";
import {
  applyInputs,
  captureInputs,
  hasInputs,
  anyInputPresent,
  type SerializedInputs,
} from "@/lib/inputBundle";
import { confirm } from "@/store/useConfirmStore";
import {
  INTERP_PATHS,
  INTERP_PATH_LABELS,
  useInterpStore,
  type InterpMethod,
  type InterpPath,
} from "@/store/useInterpStore";
import { useLoraStore } from "@/store/useLoraStore";
import { usePerformanceStore } from "@/store/usePerformanceStore";
import { useSessionStore } from "@/store/useSessionStore";
import {
  TIME_SIGNATURE_LABELS,
  VALID_KEYSCALES,
  VALID_TIME_SIGNATURES,
  isTimeSignature,
} from "@/types/engine";

import { ExportDialog } from "./ExportDialog";
import { MidiBadge } from "./MidiBadge";
import { RefSelect } from "./RefSelect";

// Exported file shape: an RtmgConfig plus the optional embedded inputs.
// Old DEMON builds importing this just ignore `inputs` (mergeConfig
// drops unknown keys); a config exported WITHOUT inputs is byte-identical
// to the legacy format.
type DemonExport = RtmgConfig & { inputs?: SerializedInputs };

// Show a transient status message that clears itself after 2s — unless a
// newer message replaced it meanwhile. Used for the import/export toasts,
// which otherwise stuck on screen indefinitely.
function flashStatus(message: string): void {
  const ss = useSessionStore.getState();
  ss.setStatus(ss.status, message);
  window.setTimeout(() => {
    const cur = useSessionStore.getState();
    if (cur.message === message) cur.setStatus(cur.status, "");
  }, 2000);
}

const INTERP_METHOD_OPTIONS: { value: InterpMethod; label: string }[] = [
  { value: "slerp", label: "Slerp" },
  { value: "linear", label: "Linear" },
];

const INTERP_PATH_TOOLTIPS: Record<InterpPath, string> = {
  structure:
    "How structural (semantic-hint) guidance blends in. Slerp holds the latent's norm constant; linear averages.",
  timbre:
    "How the timbre reference blends from silence to full. Slerp holds the conditioning norm constant; linear averages.",
  prompt:
    "How prompt A crossfades to prompt B. Slerp avoids the washed-out midpoint a linear average produces between unrelated prompts.",
  feedback:
    "How the latent feedback tap mixes into the source. Slerp holds the latent's norm constant; linear averages.",
};

// One labelled dropdown per live blend path, using the same RefSelect
// component as the core input / structure / timbre pickers so they match
// visually. Subscribes narrowly to its own method so flipping one path
// doesn't re-render the others.
function InterpRow({ path }: { path: InterpPath }) {
  const method = useInterpStore((s) => s.methods[path]);
  const setMethod = useInterpStore((s) => s.setMethod);
  return (
    <RefSelect
      label={INTERP_PATH_LABELS[path]}
      value={method}
      pinned={INTERP_METHOD_OPTIONS}
      groups={[]}
      onSelect={(v) => setMethod(path, v as InterpMethod)}
      ariaLabel={`${INTERP_PATH_LABELS[path]} interpolation method`}
      tooltip={INTERP_PATH_TOOLTIPS[path]}
    />
  );
}

export function OperatorStrip() {
  const activeKey = usePerformanceStore((s) => s.activeKey);
  const activeTimeSignature = usePerformanceStore((s) => s.activeTimeSignature);
  const kiosk = usePerformanceStore((s) => s.kiosk);
  const paused = usePerformanceStore((s) => s.paused);
  const showKbdHints = usePerformanceStore((s) => s.showKbdHints);
  const smooth = usePerformanceStore((s) => s.smooth);
  const smoothMs = usePerformanceStore((s) => s.smoothMs);
  const lufsOn = usePerformanceStore((s) => s.lufsOn);
  const loopOn = usePerformanceStore((s) => s.loopOn);
  const setKey = usePerformanceStore((s) => s.setKey);
  const setTimeSignature = usePerformanceStore((s) => s.setTimeSignature);
  const toggleKiosk = usePerformanceStore((s) => s.toggleKiosk);
  const toggleKbdHints = usePerformanceStore((s) => s.toggleKbdHints);
  const toggleSmooth = usePerformanceStore((s) => s.toggleSmooth);
  const setSmoothMs = usePerformanceStore((s) => s.setSmoothMs);
  const toggleLufs = usePerformanceStore((s) => s.toggleLufs);
  const toggleLoop = usePerformanceStore((s) => s.toggleLoop);

  const configFileInputRef = useRef<HTMLInputElement | null>(null);

  // Export dialog: opening it snapshots whether any input is active so
  // the dialog knows whether to show the "Serialize inputs" checkbox.
  const [exportOpen, setExportOpen] = useState(false);
  const [exportHasInputs, setExportHasInputs] = useState(false);

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
    try {
      const text = await file.text();
      const parsed = JSON.parse(text) as Partial<DemonExport>;
      if (typeof parsed !== "object" || parsed === null || Array.isArray(parsed)) {
        throw new Error("config root must be an object");
      }
      // mergeConfig only reads the known RtmgConfig keys, so the `inputs`
      // attachment rides through untouched and is applied separately.
      const merged = mergeConfig(getConfig(), parsed);
      applyConfig(merged);

      const inputs = parsed.inputs;
      if (hasInputs(inputs)) {
        const { applied, needSession } = await applyInputs(
          inputs as SerializedInputs,
        );
        let msg = `Imported ${file.name}`;
        if (applied.length) msg += ` + inputs (${applied.join(", ")})`;
        if (needSession.length) {
          msg += ` — press Play to apply ${needSession.join(", ")}`;
        }
        flashStatus(msg);
      } else {
        flashStatus(`Imported ${file.name}`);
      }
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e);
      flashStatus(`Import failed: ${msg}`);
    }
  }

  // Open the export dialog — snapshot whether any input is live so the
  // dialog knows whether to show the "Serialize inputs" checkbox.
  function openExportDialog(): void {
    setExportHasInputs(anyInputPresent());
    setExportOpen(true);
  }

  // Export config — snapshot the live stores into an RtmgConfig, embed
  // the active inputs when the operator opted in, and trigger a JSON
  // download. Filename includes a timestamp so multiple exports in a
  // session don't collide.
  async function runExport(serializeInputs: boolean): Promise<void> {
    setExportOpen(false);
    const snapshot: DemonExport = captureRtmgConfig();
    const inputs = serializeInputs ? await captureInputs() : {};
    const includeInputs = hasInputs(inputs);
    if (includeInputs) snapshot.inputs = inputs;
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
    flashStatus(includeInputs ? "Exported config + inputs" : "Exported config");
  }

  // The pod's WS URL is allocated by the queue and not user-editable.
  return (
    <div className="operator-strip">
      {/* ── Key & Time ──────────────────────────────────────────────
          The "what's playing" group: key + time signature. Track
          dropdown + upload now live in the CORE tab via <TrackPicker/>. */}
      <section className="operator-section">
        <h3 className="operator-section-label">Key &amp; Time</h3>
        <div className="operator-row">
          <select
            id="key-select"
            className="fixture-select"
            title="Musical key — sidecar / auto-detected; changes apply immediately"
            value={activeKey}
            onChange={async (e) => {
              const newKey = e.target.value;
              if (newKey === activeKey) return;
              const select = e.currentTarget;
              const ok = await confirm({
                title: "Change key",
                message: `Change key to "${newKey}"?\n\nThis tells the model what key the song is in. It will affect the pitch of the output, but it does NOT perfectly transpose the audio.`,
                confirmLabel: "Change key",
              });
              if (!ok) {
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
        </div>
      </section>

      {/* ── Transport ──────────────────────────────────────────────
          Standard transport row: seek-start / play-pause / loop / reset. */}
      <section className="operator-section">
        <h3 className="operator-section-label">Transport</h3>
        <div className="operator-row">
          <button
            type="button"
            className="pause-btn"
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
        </div>
      </section>

      {/* ── Playback prefs ─────────────────────────────────────────
          How the engine plays back: glide smoothing, loudness match,
          keyboard hints, kiosk lock. */}
      <section className="operator-section">
        <h3 className="operator-section-label">Playback prefs</h3>
        <div className="operator-row">
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
            id="kiosk-toggle"
            className={`pause-btn${kiosk ? " active" : ""}`}
            data-midi-learn="kiosk_toggle"
            data-dd-tooltip="Toggle kiosk mode — auto-hide cursor + idle reset (right-click to MIDI-learn)"
            type="button"
            onClick={toggleKiosk}
          >
            KIOSK
          </button>
        </div>
      </section>

      {/* ── Interpolation ──────────────────────────────────────────
          Per-path blend method (slerp vs linear) for the four live
          blends. Changes apply immediately via set_interp_method. */}
      <section className="operator-section">
        <h3 className="operator-section-label">Interpolation</h3>
        {INTERP_PATHS.map((path) => (
          <InterpRow key={path} path={path} />
        ))}
      </section>

      {/* ── Status ─────────────────────────────────────────────────
          At-a-glance state indicators (MIDI device, etc.). */}
      <section className="operator-section">
        <h3 className="operator-section-label">Status</h3>
        <div className="operator-row">
          <div id="install-midi-slot">
            <MidiBadge />
          </div>
        </div>
      </section>

      <section className="operator-section">
        <h3 className="operator-section-label">Config</h3>
        <div className="operator-row">
          <button
            type="button"
            className="pause-btn"
            data-dd-tooltip="Import config from JSON"
            aria-label="Import config"
            onClick={() => configFileInputRef.current?.click()}
          >
            Import
          </button>
          <button
            type="button"
            className="pause-btn"
            data-dd-tooltip="Download current config as JSON (optionally with input audio)"
            aria-label="Export config"
            onClick={openExportDialog}
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
        </div>
      </section>

      {exportOpen && (
        <ExportDialog
          hasInputs={exportHasInputs}
          onCancel={() => setExportOpen(false)}
          onConfirm={runExport}
        />
      )}
    </div>
  );
}
