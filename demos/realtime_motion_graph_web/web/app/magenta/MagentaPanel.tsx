"use client";

import { useState } from "react";

import type { KnobManifestEntry } from "@demon/client";
import styles from "./magenta.module.css";

import { useMagentaSession } from "./useMagentaSession";

// Magenta RT 2 frontend — the hardware-pedal chassis from the
// ambient-oneknob POC (daydream-ambien-oneknob branch), but instead of
// one macro knob it lays out every control the mrt2 family actually
// declares: the knob bank straight from ready.knob_manifest, Tags A/B,
// and the A↔B blend. Nothing acestep-shaped exists here — no fixtures,
// no LoRAs, no timbre/structure.

const DEFAULT_TAGS_A = "warm analog synthwave, steady beat";
const DEFAULT_TAGS_B = "";

function knobLabel(name: string): string {
  return name.replace(/^mrt2_/, "").replace(/_/g, " ");
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

// Small rotary control — the POC's rotor pattern (invisible range input
// over a CSS pointer disc) at satellite size. Range/step come from the
// manifest entry; the -135°..+135° sweep matches the POC's big knob.
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

export function MagentaPanel() {
  const session = useMagentaSession();
  const [tagsA, setTagsA] = useState(DEFAULT_TAGS_A);
  const [tagsB, setTagsB] = useState(DEFAULT_TAGS_B);
  const [blend, setBlendState] = useState(0);

  const running = session.status === "ready" || session.status === "connecting";

  function onToggle() {
    if (running) {
      void session.stop();
    } else {
      void session.start(tagsA, tagsB, blend);
    }
  }

  function onBlend(value: number) {
    setBlendState(value);
    session.setBlend(value);
  }

  return (
    <main className={styles.shell}>
      <section className={styles.plugin} aria-label="Daydream Magenta">
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

        <div className={styles.tags}>
          <label className={styles.tagSlot}>
            <span>Tags A</span>
            <textarea
              rows={2}
              value={tagsA}
              onChange={(event) => setTagsA(event.target.value)}
            />
          </label>
          <div className={styles.blendRow}>
            <span>A</span>
            <input
              type="range"
              min={0}
              max={1}
              step={0.01}
              value={blend}
              onChange={(event) => onBlend(Number(event.target.value))}
              aria-label="Prompt blend"
            />
            <span>B</span>
          </div>
          <label className={styles.tagSlot}>
            <span>Tags B</span>
            <textarea
              rows={2}
              value={tagsB}
              onChange={(event) => setTagsB(event.target.value)}
            />
          </label>
          <button
            type="button"
            className={styles.sendBtn}
            disabled={session.status !== "ready"}
            onClick={() => session.sendTags(tagsA, tagsB)}
          >
            Send Tags
          </button>
        </div>

        <div className={styles.title}>magenta</div>

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
