"use client";

import { useEffect, useMemo, useState } from "react";
import { createPortal } from "react-dom";

import { KEY_BINDINGS } from "@/engine/keyboard/bindings";
import {
  MIDI_CC_ROWS,
  MIDI_ENUM_ROWS,
  MIDI_MODE_OPTIONS,
  MIDI_NOTE_ROWS,
  MIDI_TARGET_ROWS,
  midiTargetLabel,
  type MidiTargetKind,
} from "@/engine/midi/targetRegistry";
import type { CcMode } from "@/engine/midi/types";
import { useMidiStore } from "@/store/useMidiStore";
import { useUiStore } from "@/store/useUiStore";

type RowKind = MidiTargetKind;

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
  const setCcMode = useMidiStore((s) => s.setCcMode);
  const resetMap = useMidiStore((s) => s.resetMap);
  const [learnModes, setLearnModes] = useState<Record<string, CcMode>>({});

  const selectedModeFor = (target: string, num: string | null): CcMode =>
    (num ? map.ccMode?.[num] : undefined) ?? learnModes[target] ?? "absolute";

  const setSelectedMode = (target: string, mode: CcMode, num: string | null) => {
    if (num) setCcMode(num, mode);
    setLearnModes((s) => ({ ...s, [target]: mode }));
  };

  // Find the bound controller number for a row, searching every bucket
  // the target can legitimately live in. `cc` is a continuous slider param,
  // `enum` a dropdown (ccEnum / noteEnum), `note` an action (notes /
  // ccActions). Returns the number plus how it's bound so the table can
  // label it.
  const findBinding = useMemo(() => {
    return (
      kind: RowKind,
      target: string,
    ): { num: string; via: "cc" | "note" } | null => {
      const search = (
        slot: Record<string, string> | undefined,
        via: "cc" | "note",
      ): { num: string; via: "cc" | "note" } | null => {
        if (!slot) return null;
        for (const [num, t] of Object.entries(slot)) {
          if (t === target) return { num, via };
        }
        return null;
      };
      if (kind === "cc") {
        return search(map.cc, "cc");
      }
      if (kind === "enum") {
        return search(map.ccEnum, "cc") ?? search(map.noteEnum, "note");
      }
      return search(map.notes, "note") ?? search(map.ccActions, "cc");
    };
  }, [map]);

  // Bindings made by right-clicking a control that has no curated row
  // above — channel faders, LoRA strength, the blend slider, etc. Listed
  // here so their binding mode (the spring modes a mod wheel needs) is
  // reachable. Keyed by what's NOT covered by the curated targets.
  const extraRows = useMemo(() => {
    const curatedCc = new Set(MIDI_CC_ROWS.map((r) => r.target));
    const curatedNote = new Set(MIDI_NOTE_ROWS.map((r) => r.target));
    const curatedEnum = new Set(MIDI_ENUM_ROWS.map((r) => r.target));
    const out: {
      kind: RowKind;
      target: string;
      num: string;
      via: "cc" | "note";
    }[] = [];
    const sweep = (
      slot: Record<string, string> | undefined,
      kind: RowKind,
      via: "cc" | "note",
      curated: Set<string>,
    ) => {
      for (const [num, t] of Object.entries(slot ?? {})) {
        if (!curated.has(t)) out.push({ kind, target: t, num, via });
      }
    };
    sweep(map.cc, "cc", "cc", curatedCc);
    sweep(map.ccActions, "note", "cc", curatedNote);
    sweep(map.notes, "note", "note", curatedNote);
    sweep(map.ccEnum, "enum", "cc", curatedEnum);
    sweep(map.noteEnum, "enum", "note", curatedEnum);
    return out;
  }, [map]);

  return (
    <div className="config-midi">
      <p className="config-midi-tip">
        Right-click any slider, knob, dropdown, or toggle in Full Controls
        to learn directly. For a CC, pick what the physical control does:
        position sets the value, clicks nudge it, center returns to default,
        or bottom returns to default.
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

        {MIDI_TARGET_ROWS.map((row) => {
          const binding = findBinding(row.kind, row.target);
          const num = binding?.num ?? null;
          const isLearning =
            learn?.kind === row.kind && learn.target === row.target;
          // Type cell: a continuous CC, an enum (knob or pad), or an
          // action note. For enums show how it actually landed.
          const typeLabel =
            row.kind === "cc"
              ? "CC"
              : row.kind === "enum"
                ? binding
                  ? binding.via === "cc"
                    ? "Enum · CC"
                    : "Enum · Note"
                  : "Enum"
                : "Note";
          return (
            <div className="config-midi-row" key={`${row.kind}:${row.target}`}>
              <span className="config-midi-cell-label">{row.label}</span>
              <span className="config-midi-cell-kind">{typeLabel}</span>
              <span className="config-midi-cell-num">
                {isLearning ? "…" : (num ?? "—")}
              </span>
              <span className="config-midi-cell-actions">
                {/* Binding mode — only meaningful for a CC-bound slider
                    (and only once a controller is assigned). Enum knobs
                    and action notes have no continuous mode. */}
                {row.kind === "cc" && (
                  <select
                    className="config-midi-mode"
                    title="Choose what the physical control does before learning it."
                    value={selectedModeFor(row.target, binding?.via === "cc" ? num : null)}
                    onChange={(e) =>
                      setSelectedMode(
                        row.target,
                        e.target.value as CcMode,
                        binding?.via === "cc" ? num : null,
                      )
                    }
                  >
                    {MIDI_MODE_OPTIONS.map((o) => (
                      <option key={o.value} value={o.value}>
                        {o.label}
                      </option>
                    ))}
                  </select>
                )}
                <button
                  type="button"
                  className={`config-midi-btn${isLearning ? " config-midi-btn--learning" : ""}`}
                  onClick={() => {
                    if (isLearning) {
                      cancelLearn();
                    } else {
                      startLearn(
                        row.kind,
                        row.target,
                        null,
                        row.kind === "cc"
                          ? selectedModeFor(row.target, binding?.via === "cc" ? num : null)
                          : undefined,
                      );
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

      {extraRows.length > 0 && (
        <div className="config-midi-table">
          <div className="config-midi-row config-midi-row--head">
            <span>Mapped control</span>
            <span>Type</span>
            <span>#</span>
            <span />
          </div>
          {extraRows.map((row) => {
            const typeLabel =
              row.kind === "cc"
                ? "CC"
                : row.kind === "enum"
                  ? row.via === "cc"
                    ? "Enum · CC"
                    : "Enum · Note"
                  : row.via === "cc"
                    ? "Note · CC"
                    : "Note";
            return (
              <div
                className="config-midi-row"
                key={`extra:${row.kind}:${row.via}:${row.num}`}
              >
                <span className="config-midi-cell-label">
                  {midiTargetLabel(row.target)}
                </span>
                <span className="config-midi-cell-kind">{typeLabel}</span>
                <span className="config-midi-cell-num">{row.num}</span>
                <span className="config-midi-cell-actions">
                  {row.kind === "cc" && row.via === "cc" && (
                    <select
                      className="config-midi-mode"
                      title="Choose what the physical control does."
                      value={map.ccMode?.[row.num] ?? "absolute"}
                      onChange={(e) =>
                        setCcMode(row.num, e.target.value as CcMode)
                      }
                    >
                      {MIDI_MODE_OPTIONS.map((o) => (
                        <option key={o.value} value={o.value}>
                          {o.label}
                        </option>
                      ))}
                    </select>
                  )}
                  <button
                    type="button"
                    className="config-midi-btn config-midi-btn--ghost"
                    onClick={() => clearBinding(row.kind, row.target)}
                  >
                    Clear
                  </button>
                </span>
              </div>
            );
          })}
        </div>
      )}

      <div className="config-midi-footer">
        <button
          type="button"
          className="config-midi-btn config-midi-btn--ghost"
          onClick={() => {
            resetMap();
            // Drop locally-remembered learn modes so a row's dropdown can't
            // keep showing a stale mode after the binding reverts to default.
            setLearnModes({});
          }}
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
