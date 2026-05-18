"use client";

import { useEffect, useMemo, useState } from "react";
import { createPortal } from "react-dom";

import { KEY_BINDINGS } from "@/engine/keyboard/bindings";
import { LORA_SLOT_MARKER, type NoteAction } from "@/engine/midi/types";
import { useMidiStore } from "@/store/useMidiStore";
import { useUiStore } from "@/store/useUiStore";

interface RowDef {
  kind: "cc" | "note";
  target: string;
  label: string;
}

const CC_ROWS: RowDef[] = [
  { kind: "cc", target: "denoise", label: "Remix strength" },
  { kind: "cc", target: "hint_strength", label: "Structure strength" },
  { kind: "cc", target: "feedback", label: "Feedback" },
  { kind: "cc", target: "shift", label: "Shift" },
  { kind: "cc", target: LORA_SLOT_MARKER[0], label: "LoRA slot 1 strength" },
  { kind: "cc", target: LORA_SLOT_MARKER[1], label: "LoRA slot 2 strength" },
];

const NOTE_ACTIONS: { target: NoteAction; label: string }[] = [
  { target: "seed", label: "Randomize seed" },
  { target: "send_prompt", label: "Send prompt" },
  { target: "pause", label: "Pause / resume" },
  { target: "mode_toggle", label: "Toggle display mode" },
  { target: "kiosk_toggle", label: "Toggle kiosk" },
];

const NOTE_ROWS: RowDef[] = NOTE_ACTIONS.map((n) => ({
  kind: "note",
  target: n.target,
  label: n.label,
}));

const ALL_ROWS = [...CC_ROWS, ...NOTE_ROWS];

export function ConfigModal() {
  const open = useUiStore((s) => s.configOpen);
  const setConfigOpen = useUiStore((s) => s.setConfigOpen);

  const [tab, setTab] = useState<"midi" | "keyboard">("midi");
  const [mounted, setMounted] = useState(false);

  useEffect(() => setMounted(true), []);

  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        e.stopPropagation();
        setConfigOpen(false);
      }
    };
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, [open, setConfigOpen]);

  // Cancel any in-flight MIDI learn when the modal closes.
  useEffect(() => {
    if (open) return;
    const learn = useMidiStore.getState().learn;
    if (learn) useMidiStore.getState().cancelLearn();
  }, [open]);

  if (!mounted || !open) return null;

  return createPortal(
    <div
      className="config-modal-backdrop"
      onClick={() => setConfigOpen(false)}
      role="presentation"
    >
      <div
        className="config-modal"
        role="dialog"
        aria-modal="true"
        aria-label="MIDI and keyboard configuration"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="config-modal-accent" aria-hidden="true" />

        <div className="config-modal-header">
          <h2 className="config-modal-title">Configuration</h2>
          <button
            type="button"
            className="config-modal-close"
            onClick={() => setConfigOpen(false)}
            aria-label="Close"
          >
            ×
          </button>
        </div>

        <div className="config-modal-tabs" role="tablist">
          <button
            type="button"
            role="tab"
            aria-selected={tab === "midi"}
            className={`config-modal-tab${tab === "midi" ? " config-modal-tab--active" : ""}`}
            onClick={() => setTab("midi")}
          >
            MIDI
          </button>
          <button
            type="button"
            role="tab"
            aria-selected={tab === "keyboard"}
            className={`config-modal-tab${tab === "keyboard" ? " config-modal-tab--active" : ""}`}
            onClick={() => setTab("keyboard")}
          >
            Keyboard
          </button>
        </div>

        <div className="config-modal-body">
          {tab === "midi" ? <MidiTab /> : <KeyboardTab />}
        </div>
      </div>
    </div>,
    document.body,
  );
}

function MidiTab() {
  const map = useMidiStore((s) => s.map);
  const learn = useMidiStore((s) => s.learn);
  const status = useMidiStore((s) => s.status);
  const available = useMidiStore((s) => s.available);
  const startLearn = useMidiStore((s) => s.startLearn);
  const cancelLearn = useMidiStore((s) => s.cancelLearn);
  const clearBinding = useMidiStore((s) => s.clearBinding);
  const resetMap = useMidiStore((s) => s.resetMap);

  const findNum = useMemo(() => {
    return (kind: "cc" | "note", target: string): string | null => {
      const slot = kind === "cc" ? map.cc : map.notes;
      for (const [num, t] of Object.entries(slot)) {
        if (t === target) return num;
      }
      return null;
    };
  }, [map]);

  return (
    <div className="config-midi">
      <p className="config-midi-tip">
        Right-click any slider in Full Controls to learn directly.
      </p>

      <div
        className={`config-midi-status config-midi-status--${status.tone}`}
      >
        <span className="config-midi-status-dot" aria-hidden="true" />
        <span>{status.message}</span>
        {!available && (
          <span className="config-midi-status-hint">
            (no devices detected — connect a controller)
          </span>
        )}
      </div>

      <div className="config-midi-table">
        <div className="config-midi-row config-midi-row--head">
          <span>Parameter</span>
          <span>Type</span>
          <span>#</span>
          <span />
        </div>

        {ALL_ROWS.map((row) => {
          const num = findNum(row.kind, row.target);
          const isLearning =
            learn?.kind === row.kind && learn.target === row.target;
          return (
            <div className="config-midi-row" key={`${row.kind}:${row.target}`}>
              <span className="config-midi-cell-label">{row.label}</span>
              <span className="config-midi-cell-kind">
                {row.kind === "cc" ? "CC" : "Note"}
              </span>
              <span className="config-midi-cell-num">
                {isLearning ? "…" : (num ?? "—")}
              </span>
              <span className="config-midi-cell-actions">
                <button
                  type="button"
                  className={`config-midi-btn${isLearning ? " config-midi-btn--learning" : ""}`}
                  onClick={() => {
                    if (isLearning) {
                      cancelLearn();
                    } else {
                      startLearn(row.kind, row.target, null);
                    }
                  }}
                >
                  {isLearning ? "Cancel" : "Learn"}
                </button>
                <button
                  type="button"
                  className="config-midi-btn config-midi-btn--ghost"
                  onClick={() => clearBinding(row.kind, row.target)}
                  disabled={!num}
                >
                  Clear
                </button>
              </span>
            </div>
          );
        })}
      </div>

      <div className="config-midi-footer">
        <button
          type="button"
          className="config-midi-btn config-midi-btn--ghost"
          onClick={() => resetMap()}
        >
          Reset to defaults
        </button>
      </div>
    </div>
  );
}

function KeyboardTab() {
  return (
    <div className="config-keyboard">
      <div className="config-keyboard-list">
        {KEY_BINDINGS.map((b) => (
          <div className="config-keyboard-row" key={b.combo}>
            <kbd className="config-keyboard-combo">{b.combo}</kbd>
            <span className="config-keyboard-desc">{b.description}</span>
          </div>
        ))}
      </div>
      <p className="config-keyboard-note">
        Rebinding will land in a future update. Current shortcuts are fixed.
      </p>
    </div>
  );
}
