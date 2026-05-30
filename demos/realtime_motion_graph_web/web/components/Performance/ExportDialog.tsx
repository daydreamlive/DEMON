"use client";

import { useEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";

// Export options dialog. Opened by the OperatorStrip's Export button.
// Reuses the confirm-dialog-* chrome for visual parity with the rest of
// the app's modals.
//
// "Serialize inputs" is pre-checked: clicking Export and pressing Enter
// gives the operator a config file that carries the input track + timbre
// ref + structure ref (uploads embedded as audio, library fixtures by
// name). Unchecking it falls back to the legacy config-only export.

interface Props {
  /** Whether any input is currently active. When false the checkbox is
   *  hidden — there's nothing to serialize. */
  hasInputs: boolean;
  onCancel: () => void;
  onConfirm: (serializeInputs: boolean) => void;
}

export function ExportDialog({ hasInputs, onCancel, onConfirm }: Props) {
  const [mounted, setMounted] = useState(false);
  const [serialize, setSerialize] = useState(true);
  const exportRef = useRef<HTMLButtonElement | null>(null);
  const previouslyFocusedRef = useRef<HTMLElement | null>(null);

  useEffect(() => setMounted(true), []);

  useEffect(() => {
    previouslyFocusedRef.current = document.activeElement as HTMLElement | null;
    exportRef.current?.focus();
    return () => {
      previouslyFocusedRef.current?.focus?.();
    };
  }, []);

  function confirm() {
    onConfirm(hasInputs && serialize);
  }

  // Stable ref so the keydown listener doesn't re-bind every render.
  const confirmRef = useRef(confirm);
  confirmRef.current = confirm;

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        e.stopPropagation();
        e.preventDefault();
        onCancel();
      } else if (e.key === "Enter") {
        const tag = (e.target as HTMLElement | null)?.tagName;
        if (tag === "SELECT" || tag === "TEXTAREA") return;
        e.preventDefault();
        confirmRef.current();
      }
    };
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, [onCancel]);

  if (!mounted) return null;

  return createPortal(
    <div
      className="confirm-dialog-backdrop"
      onClick={onCancel}
      role="presentation"
    >
      <div
        className="confirm-dialog-modal"
        role="dialog"
        aria-modal="true"
        aria-labelledby="export-dialog-title"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="config-modal-accent" aria-hidden="true" />

        <div className="confirm-dialog-header">
          <h2 id="export-dialog-title" className="confirm-dialog-title">
            Export config
          </h2>
          <button
            type="button"
            className="config-modal-close"
            onClick={onCancel}
            aria-label="Cancel"
          >
            ×
          </button>
        </div>

        <div className="confirm-dialog-body">
          <p className="confirm-dialog-message">
            Download the current setup as a JSON file.
          </p>

          {hasInputs && (
            <>
              <label
                style={{
                  display: "flex",
                  alignItems: "center",
                  gap: "0.5rem",
                  marginTop: "0.5rem",
                  cursor: "pointer",
                }}
              >
                <input
                  type="checkbox"
                  checked={serialize}
                  onChange={(e) => setSerialize(e.target.checked)}
                />
                <span>Serialize inputs</span>
              </label>
              <p
                className="confirm-dialog-message"
                style={{ marginTop: "0.25rem", opacity: 0.7, fontSize: "0.85em" }}
              >
                Embed the active input track, timbre ref, and structure ref
                so this file replays the same inputs anywhere. Uploads are
                stored in the file; library tracks are referenced by name.
              </p>
            </>
          )}
        </div>

        <div className="confirm-dialog-footer">
          <button
            type="button"
            className="confirm-dialog-btn confirm-dialog-btn--secondary"
            onClick={onCancel}
          >
            Cancel
          </button>
          <button
            ref={exportRef}
            type="button"
            className="confirm-dialog-btn confirm-dialog-btn--primary"
            onClick={confirm}
          >
            Export
          </button>
        </div>
      </div>
    </div>,
    document.body,
  );
}
