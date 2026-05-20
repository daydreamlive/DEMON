"use client";

import { useEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";

import {
  type StemSourceMode,
} from "@/engine/audio/loadFixture";
import {
  DEFAULT_TIME_SIGNATURE,
  TIME_SIGNATURE_LABELS,
  VALID_KEYSCALES,
  VALID_TIME_SIGNATURES,
  isTimeSignature,
  type TimeSignature,
} from "@/types/engine";

// Sits between file-pick and fixture-swap so users can confirm the
// upload before playback transitions. Three jobs:
//   1. If the source was longer than 240 s, the parent has already
//      auto-trimmed it; we surface that fact and offer "Pick another
//      song" as a one-click escape.
//   2. Let the user choose a key for the model: default is "Auto-detect"
//      (server CNN runs as part of the swap and its result populates the
//      activeKey). "Set manually" sets a one-shot override that wins
//      over the CNN's result for this swap (see useFixtureSwap.ts).
//   3. Same auto/manual choice for time signature. There's no audio-
//      domain detector for meter today — "Auto-detect" actually means
//      "let the server pick (sidecar value on a hit, "4" otherwise)";
//      "Set manually" tags the upload with the operator's pick which
//      wins over the server's default.

type KeyMode = "auto" | "manual";
type TimeSigMode = "auto" | "manual";

export interface AlmostReadyDialogProps {
  fileName: string;
  wasTrimmed: boolean;
  /** Default value for the manual-mode dropdown. Usually the user's
   *  current activeKey so they don't have to re-pick if they had a
   *  preference. */
  defaultKey: string;
  /** Default for the time-signature manual dropdown. Same posture as
   *  defaultKey: the operator's current activeTimeSignature carries
   *  forward so they don't lose their pick on every upload. */
  defaultTimeSignature: TimeSignature;
  onContinue: (opts: {
    keyOverride: string | null;
    timeSignatureOverride: TimeSignature | null;
    sourceMode: StemSourceMode;
  }) => void;
  /** Only invoked when wasTrimmed is true; parent re-opens the file
   *  picker so the user can swap to a shorter source instead of
   *  accepting the trim. */
  onPickAnother: () => void;
  onClose: () => void;
}

export function AlmostReadyDialog({
  fileName,
  wasTrimmed,
  defaultKey,
  defaultTimeSignature,
  onContinue,
  onPickAnother,
  onClose,
}: AlmostReadyDialogProps) {
  const [mounted, setMounted] = useState(false);
  const [mode, setMode] = useState<KeyMode>("auto");
  const [manualKey, setManualKey] = useState(
    VALID_KEYSCALES.includes(defaultKey) ? defaultKey : "C major",
  );
  const [timeSigMode, setTimeSigMode] = useState<TimeSigMode>("auto");
  const [sourceMode, setSourceMode] = useState<StemSourceMode>("full");
  const [manualTimeSig, setManualTimeSig] = useState<TimeSignature>(
    isTimeSignature(defaultTimeSignature)
      ? defaultTimeSignature
      : DEFAULT_TIME_SIGNATURE,
  );
  const continueRef = useRef<HTMLButtonElement | null>(null);

  useEffect(() => setMounted(true), []);

  // Esc closes; preventDefault so an open AdvancedDrawer underneath
  // doesn't also toggle.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        e.stopPropagation();
        e.preventDefault();
        onClose();
      } else if (e.key === "Enter") {
        // Enter = primary action, but only when focus isn't already
        // on a form control whose Enter has its own meaning.
        const tag = (e.target as HTMLElement | null)?.tagName;
        if (tag === "SELECT" || tag === "TEXTAREA") return;
        e.preventDefault();
        onContinue({
          keyOverride: mode === "manual" ? manualKey : null,
          timeSignatureOverride:
            timeSigMode === "manual" ? manualTimeSig : null,
          sourceMode,
        });
      }
    };
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, [mode, manualKey, timeSigMode, manualTimeSig, sourceMode, onClose, onContinue]);

  // Move keyboard focus to the primary button when the dialog mounts so
  // Enter / Space immediately fires Continue.
  useEffect(() => {
    if (!mounted) return;
    continueRef.current?.focus();
  }, [mounted]);

  if (!mounted) return null;

  return createPortal(
    <div
      className="almost-ready-backdrop"
      onClick={onClose}
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
            Almost Ready
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
          <p className="almost-ready-disclaimer">
            This is new tech and has some limitations. For the best
            results, choose songs without key or tempo changes.
          </p>

          <p className="almost-ready-filename" title={fileName}>
            {fileName}
          </p>

          {wasTrimmed && (
            <p className="almost-ready-trim-msg">
              Uploads are limited to 240 seconds. We&apos;ve trimmed your
              upload to fit within this limit.
            </p>
          )}

          <fieldset className="almost-ready-key-section">
            <legend className="almost-ready-key-legend">Inference source</legend>

            <label className="almost-ready-key-mode">
              <input
                type="radio"
                name="source-mode"
                value="full"
                checked={sourceMode === "full"}
                onChange={() => setSourceMode("full")}
              />
              <span>
                <strong>Use full track</strong>
                <span className="almost-ready-key-mode-hint">
                  Feed the full upload to inference. Stems are still ripped
                  automatically for realtime layers.
                </span>
              </span>
            </label>

            <label className="almost-ready-key-mode">
              <input
                type="radio"
                name="source-mode"
                value="vocals"
                checked={sourceMode === "vocals"}
                onChange={() => setSourceMode("vocals")}
              />
              <span>
                <strong>Use vocals</strong>
                <span className="almost-ready-key-mode-hint">
                  Auto-rip stems, then feed only the vocal stem to inference.
                </span>
              </span>
            </label>

            <label className="almost-ready-key-mode">
              <input
                type="radio"
                name="source-mode"
                value="instruments"
                checked={sourceMode === "instruments"}
                onChange={() => setSourceMode("instruments")}
              />
              <span>
                <strong>Use instruments</strong>
                <span className="almost-ready-key-mode-hint">
                  Auto-rip stems, then feed the instrumental bed to inference.
                </span>
              </span>
            </label>
          </fieldset>

          <fieldset className="almost-ready-key-section">
            <legend className="almost-ready-key-legend">Key</legend>

            <label className="almost-ready-key-mode">
              <input
                type="radio"
                name="key-mode"
                value="auto"
                checked={mode === "auto"}
                onChange={() => setMode("auto")}
              />
              <span>
                <strong>Auto-detect</strong>
                <span className="almost-ready-key-mode-hint">
                  We&apos;ll detect the song&apos;s key automatically.
                </span>
              </span>
            </label>

            <label className="almost-ready-key-mode">
              <input
                type="radio"
                name="key-mode"
                value="manual"
                checked={mode === "manual"}
                onChange={() => setMode("manual")}
              />
              <span>
                <strong>Set manually</strong>
                <span className="almost-ready-key-mode-hint">
                  Tells the model the song&apos;s key. Does not change
                  the song&apos;s pitch.
                </span>
              </span>
            </label>

            {mode === "manual" && (
              <select
                className="almost-ready-key-select fixture-select"
                value={manualKey}
                onChange={(e) => setManualKey(e.target.value)}
                aria-label="Pick a key"
              >
                {VALID_KEYSCALES.map((k) => (
                  <option key={k} value={k}>
                    {k}
                  </option>
                ))}
              </select>
            )}
          </fieldset>

          <fieldset className="almost-ready-key-section">
            <legend className="almost-ready-key-legend">Time signature</legend>

            <label className="almost-ready-key-mode">
              <input
                type="radio"
                name="time-sig-mode"
                value="auto"
                checked={timeSigMode === "auto"}
                onChange={() => setTimeSigMode("auto")}
              />
              <span>
                <strong>Auto-detect</strong>
                <span className="almost-ready-key-mode-hint">
                  We&apos;ll use the server&apos;s default (4/4 unless a
                  matching fixture sidecar says otherwise).
                </span>
              </span>
            </label>

            <label className="almost-ready-key-mode">
              <input
                type="radio"
                name="time-sig-mode"
                value="manual"
                checked={timeSigMode === "manual"}
                onChange={() => setTimeSigMode("manual")}
              />
              <span>
                <strong>Set manually</strong>
                <span className="almost-ready-key-mode-hint">
                  Tells the model the song&apos;s meter. Does not change
                  the song&apos;s tempo or beat grid.
                </span>
              </span>
            </label>

            {timeSigMode === "manual" && (
              <select
                className="almost-ready-key-select fixture-select"
                value={manualTimeSig}
                onChange={(e) => {
                  const v = e.target.value;
                  if (isTimeSignature(v)) setManualTimeSig(v);
                }}
                aria-label="Pick a time signature"
              >
                {VALID_TIME_SIGNATURES.map((ts) => (
                  <option key={ts} value={ts}>
                    {TIME_SIGNATURE_LABELS[ts]}
                  </option>
                ))}
              </select>
            )}
          </fieldset>
        </div>

        <div className="almost-ready-footer">
          {wasTrimmed && (
            <button
              type="button"
              className="almost-ready-btn almost-ready-btn--secondary"
              onClick={onPickAnother}
            >
              Pick another song
            </button>
          )}
          <button
            ref={continueRef}
            type="button"
            className="almost-ready-btn almost-ready-btn--primary"
            onClick={() =>
              onContinue({
                keyOverride: mode === "manual" ? manualKey : null,
                timeSignatureOverride:
                  timeSigMode === "manual" ? manualTimeSig : null,
                sourceMode,
              })
            }
          >
            Continue
          </button>
        </div>
      </div>
    </div>,
    document.body,
  );
}
