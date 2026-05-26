"use client";

import { useEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";

import { type StemSourceMode } from "@/engine/audio/loadFixture";
import {
  TIME_SIGNATURE_LABELS,
  VALID_KEYSCALES,
  VALID_TIME_SIGNATURES,
  isTimeSignature,
  type TimeSignature,
} from "@/types/engine";
import {
  DEFAULT_LEGO_PROMPTS,
  LEGO_TRACKS,
  labelForLegoTrack,
  type LegoLayerConfig,
  type LegoTrack,
} from "@/types/lego";

// Three-step confirm dialog between file-pick and fixture-swap:
//   Step 1 — inference source: full track / instruments / vocals.
//   Step 2 — optional base-model LEGO layers generated before playback.
//   Step 3 — key + time signature. Each is "Auto-detect" or an explicit
//            override that wins over the server's resolver for this
//            swap (see useFixtureSwap.ts).
// If the source was longer than 240 s the parent has already
// auto-trimmed it; step 1 surfaces that with a one-click "pick another"
// escape.

type Step = 1 | 2 | 3;

interface SourceOption {
  mode: StemSourceMode;
  title: string;
  hint: string;
}

const SOURCE_OPTIONS: SourceOption[] = [
  {
    mode: "full",
    title: "Full track",
    hint: "Feed the whole upload to inference. Stems are still ripped automatically for the realtime layers.",
  },
  {
    mode: "instruments",
    title: "Instruments",
    hint: "Auto-rip stems, then feed only the instrumental bed to inference.",
  },
  {
    mode: "vocals",
    title: "Vocals",
    hint: "Auto-rip stems, then feed only the vocal stem to inference.",
  },
];

const AUTO = "auto";

export interface AlmostReadyDialogProps {
  fileName: string;
  wasTrimmed: boolean;
  /** Retained for call-site compatibility; the step-2 selects default
   *  to "Auto-detect" rather than pre-filling a prior pick. */
  defaultKey: string;
  defaultTimeSignature: TimeSignature;
  onContinue: (opts: {
    keyOverride: string | null;
    timeSignatureOverride: TimeSignature | null;
    sourceMode: StemSourceMode;
    legoLayers: LegoLayerConfig[];
  }) => void | Promise<void>;
  /** Only invoked when wasTrimmed is true; parent re-opens the file
   *  picker so the user can swap to a shorter source. */
  onPickAnother: () => void;
  onClose: () => void;
}

export function AlmostReadyDialog({
  fileName,
  wasTrimmed,
  onContinue,
  onPickAnother,
  onClose,
}: AlmostReadyDialogProps) {
  const [mounted, setMounted] = useState(false);
  const [step, setStep] = useState<Step>(1);
  const [sourceMode, setSourceMode] = useState<StemSourceMode>("full");
  const [keyChoice, setKeyChoice] = useState<string>(AUTO);
  const [tsChoice, setTsChoice] = useState<string>(AUTO);
  const [legoSelected, setLegoSelected] = useState<Set<LegoTrack>>(
    () => new Set(),
  );
  const [legoPrompts, setLegoPrompts] =
    useState<Record<LegoTrack, string>>(DEFAULT_LEGO_PROMPTS);
  const [continuing, setContinuing] = useState(false);
  const primaryRef = useRef<HTMLButtonElement | null>(null);

  useEffect(() => setMounted(true), []);

  async function finish() {
    if (continuing) return;
    setContinuing(true);
    try {
      await onContinue({
        keyOverride: keyChoice === AUTO ? null : keyChoice,
        timeSignatureOverride:
          tsChoice !== AUTO && isTimeSignature(tsChoice) ? tsChoice : null,
        sourceMode,
        legoLayers: Array.from(legoSelected).map((track) => ({
          track,
          prompt: legoPrompts[track],
        })),
      });
    } finally {
      setContinuing(false);
    }
  }

  // Esc closes; Enter advances (step 1 → next, step 2 → start) unless
  // focus is on a SELECT whose Enter has its own meaning.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        e.stopPropagation();
        e.preventDefault();
        onClose();
        return;
      }
      if (e.key === "Enter") {
        const tag = (e.target as HTMLElement | null)?.tagName;
        if (tag === "SELECT") return;
        e.preventDefault();
        if (continuing) return;
        if (step === 1) setStep(2);
        else if (step === 2) setStep(3);
        else finish();
      }
    };
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [step, sourceMode, keyChoice, tsChoice, continuing, onClose, onContinue]);

  // Focus the primary button on mount and on every step change so
  // Enter / Space fire it.
  useEffect(() => {
    if (mounted) primaryRef.current?.focus();
  }, [mounted, step]);

  if (!mounted) return null;

  return createPortal(
    <div className="almost-ready-backdrop" onClick={onClose} role="presentation">
      <div
        className="almost-ready-modal"
        role="dialog"
        aria-modal="true"
        aria-labelledby="almost-ready-title"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="config-modal-accent" aria-hidden="true" />

        <div className="almost-ready-header">
          <h2 id="almost-ready-title" className="almost-ready-title">
            Almost ready
          </h2>
          <button
            type="button"
            className="config-modal-close"
            onClick={onClose}
            aria-label="Cancel upload"
          >
            ×
          </button>
        </div>

        <div className="almost-ready-body">
          <div className="almost-ready-filename" title={fileName}>
            {fileName}
          </div>

          <div className="almost-ready-steps" aria-hidden="true">
            <span
              className={`almost-ready-step-dot${step === 1 ? " is-active" : ""}`}
            />
            <span
              className={`almost-ready-step-dot${step === 2 ? " is-active" : ""}`}
            />
            <span
              className={`almost-ready-step-dot${step === 3 ? " is-active" : ""}`}
            />
          </div>

          {step === 1 ? (
            <>
              <div className="almost-ready-step-head">
                <span className="almost-ready-step-num">Step 1 of 3</span>
                <h3 className="almost-ready-step-title">Inference source</h3>
              </div>

              {wasTrimmed && (
                <p className="almost-ready-trim-msg">
                  Trimmed to the 240-second upload limit.
                </p>
              )}

              <div
                className="almost-ready-cards"
                role="radiogroup"
                aria-label="Inference source"
              >
                {SOURCE_OPTIONS.map((opt) => (
                  <button
                    key={opt.mode}
                    type="button"
                    role="radio"
                    aria-checked={sourceMode === opt.mode}
                    className={`almost-ready-card${sourceMode === opt.mode ? " is-selected" : ""}`}
                    onClick={() => setSourceMode(opt.mode)}
                  >
                    <span className="almost-ready-card-radio" aria-hidden="true" />
                    <span className="almost-ready-card-text">
                      <span className="almost-ready-card-title">
                        {opt.title}
                      </span>
                      <span className="almost-ready-card-hint">{opt.hint}</span>
                    </span>
                  </button>
                ))}
              </div>

              <p className="almost-ready-note">
                New tech with limits — songs without key or tempo changes
                work best.
              </p>
            </>
          ) : step === 2 ? (
            <>
              <div className="almost-ready-step-head">
                <span className="almost-ready-step-num">Step 2 of 3</span>
                <h3 className="almost-ready-step-title">LEGO layers</h3>
              </div>
              <p className="almost-ready-note">
                Optional: generate base-model instrument layers now, before
                the realtime model loads into VRAM.
              </p>
              <div className="almost-ready-lego-list">
                {LEGO_TRACKS.map((track) => {
                  const checked = legoSelected.has(track);
                  return (
                    <div
                      key={track}
                      className={`almost-ready-lego-row${checked ? " is-selected" : ""}`}
                    >
                      <label className="almost-ready-lego-check">
                        <input
                          type="checkbox"
                          checked={checked}
                          onChange={() => {
                            setLegoSelected((prev) => {
                              const next = new Set(prev);
                              if (next.has(track)) next.delete(track);
                              else next.add(track);
                              return next;
                            });
                          }}
                        />
                        <span>{labelForLegoTrack(track)}</span>
                      </label>
                      <input
                        className="almost-ready-lego-prompt"
                        type="text"
                        value={legoPrompts[track]}
                        onChange={(e) =>
                          setLegoPrompts((prev) => ({
                            ...prev,
                            [track]: e.target.value,
                          }))
                        }
                        disabled={!checked}
                        aria-label={`${labelForLegoTrack(track)} LEGO prompt`}
                      />
                    </div>
                  );
                })}
              </div>
            </>
          ) : (
            <>
              <div className="almost-ready-step-head">
                <span className="almost-ready-step-num">Step 3 of 3</span>
                <h3 className="almost-ready-step-title">
                  Key &amp; time signature
                </h3>
              </div>

              <div className="almost-ready-field">
                <label className="almost-ready-field-label" htmlFor="ar-key">
                  Key
                </label>
                <select
                  id="ar-key"
                  className="almost-ready-select fixture-select"
                  value={keyChoice}
                  onChange={(e) => setKeyChoice(e.target.value)}
                >
                  <option value={AUTO}>Auto-detect</option>
                  {VALID_KEYSCALES.map((k) => (
                    <option key={k} value={k}>
                      {k}
                    </option>
                  ))}
                </select>
                <span className="almost-ready-field-hint">
                  Tells the model the song&apos;s key — it doesn&apos;t
                  repitch the audio.
                </span>
              </div>

              <div className="almost-ready-field">
                <label className="almost-ready-field-label" htmlFor="ar-ts">
                  Time signature
                </label>
                <select
                  id="ar-ts"
                  className="almost-ready-select fixture-select"
                  value={tsChoice}
                  onChange={(e) => setTsChoice(e.target.value)}
                >
                  <option value={AUTO}>Auto-detect</option>
                  {VALID_TIME_SIGNATURES.map((ts) => (
                    <option key={ts} value={ts}>
                      {TIME_SIGNATURE_LABELS[ts]}
                    </option>
                  ))}
                </select>
                <span className="almost-ready-field-hint">
                  Tells the model the song&apos;s meter — it doesn&apos;t
                  change the tempo or beat grid.
                </span>
              </div>
            </>
          )}
        </div>

        <div className="almost-ready-footer">
          {step === 1 ? (
            <>
              {wasTrimmed && (
                <button
                  type="button"
                  className="almost-ready-btn almost-ready-btn--ghost"
                  onClick={onPickAnother}
                >
                  Pick another
                </button>
              )}
              <button
                type="button"
                className="almost-ready-btn almost-ready-btn--secondary"
                onClick={onClose}
              >
                Cancel
              </button>
              <button
                ref={primaryRef}
                type="button"
                className="almost-ready-btn almost-ready-btn--primary"
                onClick={() => setStep(2)}
              >
                Next →
              </button>
            </>
          ) : step === 2 ? (
            <>
              <button
                type="button"
                className="almost-ready-btn almost-ready-btn--secondary"
                onClick={() => setStep(1)}
              >
                ← Back
              </button>
              <button
                ref={primaryRef}
                type="button"
                className="almost-ready-btn almost-ready-btn--primary"
                onClick={() => setStep(3)}
              >
                Next →
              </button>
            </>
          ) : (
            <>
              <button
                type="button"
                className="almost-ready-btn almost-ready-btn--secondary"
                onClick={() => setStep(2)}
                disabled={continuing}
              >
                ← Back
              </button>
              <button
                ref={primaryRef}
                type="button"
                className="almost-ready-btn almost-ready-btn--primary"
                onClick={finish}
                disabled={continuing}
              >
                {continuing
                  ? legoSelected.size > 0
                    ? "Generating..."
                    : "Starting..."
                  : "Start"}
              </button>
            </>
          )}
        </div>
      </div>
    </div>,
    document.body,
  );
}
