"use client";

import { useEffect, useRef, useState } from "react";

import {
  addAndFocusPromptSlot,
  focusPromptSlot,
  removePromptSlot,
  setBlendEndpointA,
  setBlendPartner,
} from "@/lib/promptDeck";
import { usePerformanceStore } from "@/store/usePerformanceStore";
import { useSessionStore } from "@/store/useSessionStore";

// Three independent concerns, deliberately decoupled:
//
//   - Tap a chip in the deck strip → focuses it for editing
//     (textarea binds to that slot). Crossfader endpoints DO NOT
//     change. Slots loaded as A or B carry their respective badge.
//   - Click the crossfader's A label → popover lists slots → pick
//     one to load into engine A. The picked slot also becomes
//     focused so the operator can edit what they just loaded.
//   - Click the crossfader's B label → same shape, loads to engine B.
//
// The "active prompt" textarea binds to whichever slot is focused —
// whether or not that slot is currently loaded as A or B. If it IS
// loaded, edits mirror into engine.promptA/promptB and the engine
// hears them. If not, edits stay in the slot until the operator
// loads it. This is what stops the crossfader from shifting under
// the operator every time they tap a chip — A and B are stable
// "decks", the chip strip is a library + editing surface.

export function PromptsTile() {
  const slots = usePerformanceStore((s) => s.promptSlots);
  const currentSlotId = usePerformanceStore((s) => s.currentSlotId);
  const blendPartnerId = usePerformanceStore((s) => s.blendPartnerId);
  const focusedSlotId = usePerformanceStore((s) => s.focusedSlotId);
  const activeKey = usePerformanceStore((s) => s.activeKey);
  const activeTimeSignature = usePerformanceStore((s) => s.activeTimeSignature);
  const promptA = usePerformanceStore((s) => s.promptA);
  const promptB = usePerformanceStore((s) => s.promptB);
  const blend = usePerformanceStore(
    (s) => s.sliderTargets.prompt_blend ?? 0,
  );
  const setPromptSlotText = usePerformanceStore((s) => s.setPromptSlotText);
  const renamePromptSlot = usePerformanceStore((s) => s.renamePromptSlot);
  const setSlider = usePerformanceStore((s) => s.setSlider);

  const focusedSlot = slots.find((s) => s.id === focusedSlotId) ?? slots[0];
  const currentSlot = slots.find((s) => s.id === currentSlotId) ?? null;
  const partnerSlot = slots.find((s) => s.id === blendPartnerId) ?? null;

  // Local rename state.
  const [renamingId, setRenamingId] = useState<string | null>(null);
  const [renameDraft, setRenameDraft] = useState("");
  const renameInputRef = useRef<HTMLInputElement | null>(null);
  useEffect(() => {
    if (renamingId && renameInputRef.current) {
      renameInputRef.current.focus();
      renameInputRef.current.select();
    }
  }, [renamingId]);

  // Two independent popover states — one for the A endpoint, one for
  // the B endpoint. Shared outside-click handler closes whichever is
  // open. Refs point at the picker container so the outside-click
  // ignore region is precise.
  const [aPickerOpen, setAPickerOpen] = useState(false);
  const [bPickerOpen, setBPickerOpen] = useState(false);
  const aPickerRef = useRef<HTMLDivElement | null>(null);
  const bPickerRef = useRef<HTMLDivElement | null>(null);
  useEffect(() => {
    if (!aPickerOpen && !bPickerOpen) return;
    const onDown = (ev: MouseEvent) => {
      const t = ev.target as Node;
      if (aPickerOpen && aPickerRef.current && !aPickerRef.current.contains(t)) {
        setAPickerOpen(false);
      }
      if (bPickerOpen && bPickerRef.current && !bPickerRef.current.contains(t)) {
        setBPickerOpen(false);
      }
    };
    document.addEventListener("mousedown", onDown);
    return () => document.removeEventListener("mousedown", onDown);
  }, [aPickerOpen, bPickerOpen]);

  function sendPrompt() {
    const remote = useSessionStore.getState().remote;
    if (remote) {
      remote.sendPrompt(
        promptA,
        activeKey,
        activeTimeSignature,
        blendPartnerId !== null ? promptB : undefined,
      );
    }
  }

  function startRename(id: string, currentLabel: string) {
    setRenamingId(id);
    setRenameDraft(currentLabel);
  }
  function commitRename() {
    if (renamingId) {
      const trimmed = renameDraft.trim();
      if (trimmed) renamePromptSlot(renamingId, trimmed);
    }
    setRenamingId(null);
  }

  // A popover excludes whatever's loaded as B (and vice-versa) — the
  // engine can't blend a slot with itself. The currently-loaded
  // endpoint shows up in its OWN popover as "selected" so the user
  // sees the current state and can pick a sibling.
  const aCandidates = slots.filter((s) => s.id !== blendPartnerId);
  const bCandidates = slots.filter((s) => s.id !== currentSlotId);

  return (
    <div className="mixer-tile mixer-tile-prompts" data-tile="prompts">
      <div className="mixer-tile-label">Tags</div>
      <div id="prompt-section">
        <div className="prompt-slot">
          <label
            className="prompt-label"
            htmlFor="prompt-active"
            data-dd-tooltip="Edits the focused slot's text (★ in the deck). If that slot is loaded as A or B, the engine hears your edit. Otherwise edits stay in the slot until you load it."
            data-dd-tooltip-wide=""
          >
            Editing: {focusedSlot?.label ?? "—"}
          </label>
          <textarea
            id="prompt-active"
            className="prompt-input"
            rows={3}
            value={focusedSlot?.text ?? ""}
            onChange={(e) =>
              focusedSlot && setPromptSlotText(focusedSlot.id, e.target.value)
            }
          />
        </div>
        <div
          className="prompt-coupling-hint"
          data-dd-tooltip="Prompts steer the model strongly only when Strength is high (model has freedom) and Structure is low (less anchored to the source). Outside that window you'll hear minor variations, not the prompt's character."
          data-dd-tooltip-wide=""
        >
          Hits hardest at high Strength + low Structure.
        </div>
        <div className="prompt-deck" role="tablist" aria-label="Prompt slots">
          {slots.map((slot) => {
            const isA = slot.id === currentSlotId;
            const isB = slot.id === blendPartnerId;
            const isFocused = slot.id === focusedSlotId;
            const isRenaming = slot.id === renamingId;
            return (
              <div
                key={slot.id}
                className={[
                  "prompt-deck-slot",
                  isFocused ? "prompt-deck-slot--focused" : "",
                  isA ? "prompt-deck-slot--a" : "",
                  isB ? "prompt-deck-slot--b" : "",
                  isRenaming ? "prompt-deck-slot--renaming" : "",
                ].filter(Boolean).join(" ")}
                role="tab"
                aria-selected={isFocused}
              >
                {isRenaming ? (
                  <input
                    ref={renameInputRef}
                    className="prompt-deck-slot-rename"
                    type="text"
                    value={renameDraft}
                    onChange={(e) => setRenameDraft(e.target.value)}
                    onBlur={commitRename}
                    onKeyDown={(e) => {
                      if (e.key === "Enter") {
                        e.preventDefault();
                        commitRename();
                      } else if (e.key === "Escape") {
                        e.preventDefault();
                        setRenamingId(null);
                      }
                    }}
                  />
                ) : (
                  <button
                    type="button"
                    className="prompt-deck-slot-label"
                    onClick={() => {
                      if (!isFocused) focusPromptSlot(slot.id);
                    }}
                    onDoubleClick={() => startRename(slot.id, slot.label)}
                    title={`Double-click to rename. ${slot.text || "(empty)"}`}
                  >
                    {isFocused && (
                      <span className="prompt-deck-slot-focus-mark" aria-hidden="true">★</span>
                    )}
                    {slot.label}
                    {isA && (
                      <span
                        className="prompt-deck-slot-endpoint-badge prompt-deck-slot-endpoint-badge--a"
                        aria-label="Loaded as A"
                      >
                        A
                      </span>
                    )}
                    {isB && (
                      <span
                        className="prompt-deck-slot-endpoint-badge prompt-deck-slot-endpoint-badge--b"
                        aria-label="Loaded as B"
                      >
                        B
                      </span>
                    )}
                  </button>
                )}
                {slots.length > 1 && !isRenaming && (
                  <button
                    type="button"
                    className="prompt-deck-slot-remove"
                    onClick={(e) => {
                      e.stopPropagation();
                      removePromptSlot(slot.id);
                    }}
                    aria-label={`Delete ${slot.label}`}
                    title="Delete slot"
                  >
                    ×
                  </button>
                )}
              </div>
            );
          })}
          <button
            type="button"
            className="prompt-deck-add"
            onClick={() => {
              const newId = addAndFocusPromptSlot();
              const newSlot = usePerformanceStore
                .getState()
                .promptSlots.find((s) => s.id === newId);
              if (newSlot) startRename(newSlot.id, newSlot.label);
            }}
            aria-label="Add prompt slot"
            title="Add prompt slot"
          >
            +
          </button>
        </div>
        {blendPartnerId !== null && partnerSlot && currentSlot && (
          <div
            className="prompt-deck-crossfader"
            data-param="prompt_blend"
            data-dd-tooltip="Crossfade between the A endpoint and the B endpoint. 0 = pure A, 1 = pure B. Right-click to MIDI-learn. Click either label to load a different slot."
            data-dd-tooltip-wide=""
          >
            <div className="prompt-deck-crossfader-end-picker" ref={aPickerRef}>
              <button
                type="button"
                className={`prompt-deck-crossfader-end prompt-deck-crossfader-end--a${aPickerOpen ? " prompt-deck-crossfader-end--open" : ""}`}
                onClick={() => {
                  setAPickerOpen((v) => !v);
                  setBPickerOpen(false);
                }}
                aria-haspopup="menu"
                aria-expanded={aPickerOpen}
                title={`Click to pick a different A endpoint. Current: ${currentSlot.label}`}
              >
                {currentSlot.label}
                <span className="prompt-deck-crossfader-end-caret" aria-hidden="true">▾</span>
              </button>
              {aPickerOpen && aCandidates.length > 0 && (
                <div className="prompt-deck-blend-menu prompt-deck-blend-menu--a" role="menu">
                  {aCandidates.map((slot) => (
                    <button
                      key={slot.id}
                      type="button"
                      role="menuitem"
                      className={`prompt-deck-blend-menu-item${slot.id === currentSlotId ? " prompt-deck-blend-menu-item--selected" : ""}`}
                      onClick={() => {
                        setBlendEndpointA(slot.id);
                        setAPickerOpen(false);
                      }}
                    >
                      {slot.label}
                    </button>
                  ))}
                </div>
              )}
            </div>
            <input
              type="range"
              className="prompt-deck-crossfader-slider"
              min="0"
              max="1"
              step="0.01"
              value={blend}
              onChange={(e) => setSlider("prompt_blend", parseFloat(e.target.value))}
              aria-label={`Blend between ${currentSlot.label} and ${partnerSlot.label}`}
            />
            <div className="prompt-deck-crossfader-end-picker" ref={bPickerRef}>
              <button
                type="button"
                className={`prompt-deck-crossfader-end prompt-deck-crossfader-end--b${bPickerOpen ? " prompt-deck-crossfader-end--open" : ""}`}
                onClick={() => {
                  setBPickerOpen((v) => !v);
                  setAPickerOpen(false);
                }}
                aria-haspopup="menu"
                aria-expanded={bPickerOpen}
                title={`Click to pick a different B endpoint. Current: ${partnerSlot.label}`}
              >
                {partnerSlot.label}
                <span className="prompt-deck-crossfader-end-caret" aria-hidden="true">▾</span>
              </button>
              {bPickerOpen && bCandidates.length > 0 && (
                <div className="prompt-deck-blend-menu prompt-deck-blend-menu--b" role="menu">
                  {bCandidates.map((slot) => (
                    <button
                      key={slot.id}
                      type="button"
                      role="menuitem"
                      className={`prompt-deck-blend-menu-item${slot.id === blendPartnerId ? " prompt-deck-blend-menu-item--selected" : ""}`}
                      onClick={() => {
                        setBlendPartner(slot.id);
                        setBPickerOpen(false);
                      }}
                    >
                      {slot.label}
                    </button>
                  ))}
                </div>
              )}
            </div>
            <span className="prompt-deck-crossfader-value">
              {blend.toFixed(2)}
            </span>
            <kbd className="desktop-only prompt-deck-crossfader-kbd">B + ▲▼</kbd>
          </div>
        )}
        <button
          id="send-prompt"
          className="send-prompt-btn"
          data-midi-learn="send_prompt"
          data-dd-tooltip="Send tags — Enter (out of textarea) or ⌘/Ctrl + Enter (in textarea); right-click to MIDI-learn"
          type="button"
          onClick={sendPrompt}
        >
          Send Tags
          <kbd className="desktop-only send-kbd">⏎</kbd>
        </button>
      </div>
    </div>
  );
}
