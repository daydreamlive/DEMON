"use client";

import { useEffect, useState } from "react";

import type { KnobManifestEntry } from "@demon/client";
import styles from "./sa3.module.css";

import { useSA3Session } from "./useSA3Session";

// Stable Audio 3 frontend — the hardware-pedal chassis the magenta
// route established, laid out for the sa3 family's v1 surface: the
// knob bank straight from ready.knob_manifest (sa3_denoise, seed,
// steps_override today), a single prompt, and the audio-to-audio
// source picker (pod-side fixture + optional fixed duration). Nothing
// acestep-shaped exists here — no LoRAs, no timbre/structure, no A/B
// blend (sa3 v1 holds one conditioning bundle at a time).

const DEFAULT_PROMPT =
  "driving cinematic synthwave, analog arpeggios, gated reverb snare, " +
  "wide saw-lead, 152 bpm, G minor, 4/4";
const DEFAULT_FIXTURE = "low_fi_Gm_loop_60s_gnm.wav";

function knobLabel(name: string): string {
  return name.replace(/^sa3_/, "").replace(/_/g, " ");
}

function formatValue(entry: KnobManifestEntry, value: number): string {
  return entry.type === "int" ? String(Math.round(value)) : value.toFixed(2);
}

interface RotorKnobProps {
  name: string;
  entry: KnobManifestEntry;
  value: number;
  onChange: (value: number) => void;
}

// Small rotary control — invisible range input over a CSS pointer disc;
// range/step come from the manifest entry, -135°..+135° sweep.
function RotorKnob({ name, entry, value, onChange }: RotorKnobProps) {
  const min = entry.min ?? 0;
  const max = entry.max ?? 1;
  const span = max - min || 1;
  const norm = Math.max(0, Math.min(1, (value - min) / span));
  const angle = -135 + norm * 270;
  const step = entry.type === "int" ? 1 : span / 200;

  return (
    <div className={styles.knobCell} title={entry.description}>
      <div className={styles.knobWrap}>
        <input
          className={styles.knobInput}
          type="range"
          min={min}
          max={max}
          step={step}
          value={value}
          onChange={(event) => onChange(Number(event.target.value))}
          aria-label={knobLabel(name)}
        />
        <div className={styles.knob}>
          <div
            className={styles.pointerRotor}
            style={{ transform: `rotate(${angle}deg)` }}
          >
            <div className={styles.knobPointer} />
          </div>
        </div>
      </div>
      <div className={styles.knobValue}>{formatValue(entry, value)}</div>
      <div className={styles.knobLabel}>{knobLabel(name)}</div>
    </div>
  );
}

export function SA3Panel() {
  const session = useSA3Session();
  const [prompt, setPrompt] = useState(DEFAULT_PROMPT);
  const [fixture, setFixture] = useState(DEFAULT_FIXTURE);
  const [durationS, setDurationS] = useState("");

  // The pod's fixture list arrives async; if the preferred default
  // isn't on this pod, settle on the first available name.
  useEffect(() => {
    if (session.fixtures.length === 0) return;
    if (!session.fixtures.includes(fixture)) {
      setFixture(session.fixtures[0]);
    }
  }, [session.fixtures, fixture]);

  const running = session.status === "ready" || session.status === "connecting";

  function onToggle() {
    if (running) {
      void session.stop();
    } else {
      const parsed = Number(durationS);
      void session.start(
        prompt,
        fixture,
        Number.isFinite(parsed) && parsed > 0 ? parsed : null,
      );
    }
  }

  return (
    <main className={styles.shell}>
      <section className={styles.plugin} aria-label="Daydream SA3">
        <div className={styles.brandRow}>
          <div className={styles.brand}>Daydream</div>
        </div>

        <div className={styles.knobGrid}>
          {session.knobs.length > 0 ? (
            session.knobs.map(({ name, entry }) => (
              <RotorKnob
                key={name}
                name={name}
                entry={entry}
                value={session.values[name] ?? 0}
                onChange={(v) => session.setKnob(name, v)}
              />
            ))
          ) : (
            <div className={styles.knobPlaceholder}>
              {session.status === "connecting"
                ? "loading knob bank…"
                : "knobs appear when the session starts"}
            </div>
          )}
        </div>

        <div className={styles.controls}>
          <label className={styles.fieldSlot}>
            <span>Prompt</span>
            <textarea
              rows={2}
              value={prompt}
              onChange={(event) => setPrompt(event.target.value)}
            />
          </label>
          <div className={styles.sourceRow}>
            <label className={styles.fieldSlot}>
              <span>Source</span>
              <select
                value={fixture}
                disabled={running}
                onChange={(event) => setFixture(event.target.value)}
              >
                {(session.fixtures.length > 0
                  ? session.fixtures
                  : [fixture]
                ).map((name) => (
                  <option key={name} value={name}>
                    {name}
                  </option>
                ))}
              </select>
            </label>
            <label className={`${styles.fieldSlot} ${styles.durationSlot}`}>
              <span>Duration s</span>
              <input
                type="number"
                min={1}
                max={120}
                placeholder="auto"
                value={durationS}
                disabled={running}
                onChange={(event) => setDurationS(event.target.value)}
              />
            </label>
          </div>
          <button
            type="button"
            className={styles.sendBtn}
            disabled={session.status !== "ready"}
            onClick={() => session.sendPrompt(prompt)}
          >
            Send Prompt
          </button>
        </div>

        <div className={styles.title}>sa3</div>

        <div className={styles.transport}>
          <button
            type="button"
            className={`${styles.powerBtn}${running ? ` ${styles.powerOn}` : ""}`}
            onClick={onToggle}
          >
            {running ? "Stop" : "Start"}
          </button>
          <span
            className={`${styles.statusDot} ${styles[`status_${session.status}`]}`}
            aria-hidden="true"
          />
          <span className={styles.statusText}>
            {session.message || session.status}
          </span>
        </div>
      </section>
    </main>
  );
}
