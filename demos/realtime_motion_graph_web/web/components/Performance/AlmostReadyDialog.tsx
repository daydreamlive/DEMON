"use client";

import { useEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";

import { type StemSourceMode } from "@/engine/audio/loadFixture";
import { type UploadOutcome } from "@/lib/audio/commitUploadedTrack";
import { defaultSwapSourceMode } from "@/lib/config";
import {
  TIME_SIGNATURE_LABELS,
  VALID_KEYSCALES,
  VALID_TIME_SIGNATURES,
  isTimeSignature,
  type TimeSignature,
} from "@/types/engine";

// Two-step confirm dialog between file-pick and fixture-swap:
//   Step 1 — inference source: full track / instruments / vocals.
//   Step 2 — key + time signature. Each is "Auto-detect" or an explicit
//            override that wins over the server's resolver for this
//            swap (see useFixtureSwap.ts).
// If the source was longer than 240 s the parent has already
// auto-trimmed it; step 1 surfaces that with a one-click "pick another"
// escape.

type Step = 1 | 2;

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
  }) => void | Promise<UploadOutcome | void>;
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
  const [sourceMode, setSourceMode] = useState<StemSourceMode>(() =>
    defaultSwapSourceMode(),
  );
  const [keyChoice, setKeyChoice] = useState<string>(AUTO);
  const [tsChoice, setTsChoice] = useState<string>(AUTO);
  // Encoding happens on the server (GPU stem-rip + source prep) and can
  // take many seconds. Without an in-modal busy state the user clicks
  // Start and sees nothing change — the dialog looks frozen — because the
  // "Encoding…" status and any error land in the status bar *behind* the
  // modal. Track it here so Start shows progress and surfaces failures.
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const primaryRef = useRef<HTMLButtonElement | null>(null);
  // Guards the post-await state writes: a successful upload (or the user
  // hitting ×) unmounts this dialog, so we must not setState afterwards.
  const aliveRef = useRef(true);

  useEffect(() => setMounted(true), []);
  useEffect(() => {
    aliveRef.current = true;
    return () => {
      aliveRef.current = false;
    };
  }, []);

  async function finish() {
    if (submitting) return;
    setSubmitting(true);
    setError(null);
    const outcome = await onContinue({
      keyOverride: keyChoice === AUTO ? null : keyChoice,
      timeSignatureOverride:
        tsChoice !== AUTO && isTimeSignature(tsChoice) ? tsChoice : null,
      sourceMode,
    });
    // Success/abort unmounts us (parent cleared `pending`); only a failure
    // leaves the dialog mounted, in which case show the reason and re-arm
    // Start so the user can retry without re-picking the file.
    if (!aliveRef.current) return;
    setSubmitting(false);
    if (outcome && outcome.ok === false && !outcome.aborted) {
      setError(outcome.error || "Upload failed. Please try again.");
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
        if (step === 1) setStep(2);
        else finish();
      }
    };
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [step, sourceMode, keyChoice, tsChoice, onClose, onContinue]);

  // Focus the primary button on mount and on every step change so
  // Enter / Space fire it.
  useEffect(() => {
    if (mounted) primaryRef.current?.focus();
  }, [mounted, step]);

  if (!mounted) return null;

  return createPortal(
    // Backdrop click closes — but not mid-encode, so a stray click can't
    // silently abort a running upload. The × stays live as the explicit out.
    <div
      className="almost-ready-backdrop"
      onClick={submitting ? undefined : onClose}
      role="presentation"
    >
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
            aria-label={submitting ? "Cancel encoding" : "Cancel upload"}
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
          </div>

          {step === 1 ? (
            <>
              <div className="almost-ready-step-head">
                <span className="almost-ready-step-num">Step 1 of 2</span>
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
          ) : (
            <>
              <div className="almost-ready-step-head">
                <span className="almost-ready-step-num">Step 2 of 2</span>
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

        {(submitting || error) && (
          <div
            className={`almost-ready-uploading${error ? " is-error" : ""}`}
            role="status"
            aria-live="polite"
          >
            {submitting ? (
              <>
                <span
                  className="almost-ready-spinner"
                  aria-hidden="true"
                />
                <span>
                  Encoding on the server — this can take up to a minute.
                </span>
              </>
            ) : (
              <span>{error}</span>
            )}
          </div>
        )}

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
          ) : (
            <>
              <button
                type="button"
                className="almost-ready-btn almost-ready-btn--secondary"
                onClick={() => setStep(1)}
                disabled={submitting}
              >
                ← Back
              </button>
              <button
                ref={primaryRef}
                type="button"
                className="almost-ready-btn almost-ready-btn--primary"
                onClick={finish}
                disabled={submitting}
                aria-busy={submitting}
              >
                {submitting ? "Encoding…" : error ? "Retry" : "Start"}
              </button>
            </>
          )}
        </div>
      </div>
    </div>,
    document.body,
  );
}
