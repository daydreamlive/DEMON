"use client";

import { useEffect, useRef, useState } from "react";

import {
  addAndSwitchToPromptSlot,
  removePromptSlot,
  switchToPromptSlot,
} from "@/lib/promptDeck";
import { usePerformanceStore } from "@/store/usePerformanceStore";
import { useSessionStore } from "@/store/useSessionStore";

// Deck UX: many named slots, only one active at a time. The engine only
// has A/B; lib/promptDeck.ts ping-pongs the active slot through the two
// engine slots on switch and tweens the prompt_blend slider to make the
// transition smooth. The crossfade slider that used to live here is
// gone — operators reported it was hard to read; the deck switch is the
// only visible interaction.
//
// Editing the active slot's text mirrors to promptA or promptB via the
// store (setPromptSlotText), but does NOT re-encode automatically — the
// "Send Tags" button is the explicit commit, matching the pre-deck flow.

export function PromptsTile() {
  const slots = usePerformanceStore((s) => s.promptSlots);
  const currentSlotId = usePerformanceStore((s) => s.currentSlotId);
  const activeKey = usePerformanceStore((s) => s.activeKey);
  const activeTimeSignature = usePerformanceStore((s) => s.activeTimeSignature);
  const promptA = usePerformanceStore((s) => s.promptA);
  const promptB = usePerformanceStore((s) => s.promptB);
  const setPromptSlotText = usePerformanceStore((s) => s.setPromptSlotText);
  const renamePromptSlot = usePerformanceStore((s) => s.renamePromptSlot);

  const currentSlot = slots.find((s) => s.id === currentSlotId) ?? slots[0];

  // Local rename state: which slot id is in rename mode, and its draft
  // label. Committing flushes to the store and exits; Escape cancels.
  const [renamingId, setRenamingId] = useState<string | null>(null);
  const [renameDraft, setRenameDraft] = useState("");
  const renameInputRef = useRef<HTMLInputElement | null>(null);
  useEffect(() => {
    if (renamingId && renameInputRef.current) {
      renameInputRef.current.focus();
      renameInputRef.current.select();
    }
  }, [renamingId]);

  function sendPrompt() {
    const remote = useSessionStore.getState().remote;
    if (remote) {
      remote.sendPrompt(promptA, activeKey, activeTimeSignature, promptB);
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

  return (
    <div className="mixer-tile mixer-tile-prompts" data-tile="prompts">
      <div className="mixer-tile-label">Tags</div>
      <div id="prompt-section">
        <div className="prompt-slot">
          <label
            className="prompt-label"
            htmlFor="prompt-active"
            data-dd-tooltip="Active slot's text. The model conditions on this. Edit, then click Send Tags to commit. Switch slots with the strip below — the engine blends smoothly."
            data-dd-tooltip-wide=""
          >
            Active prompt
          </label>
          <textarea
            id="prompt-active"
            className="prompt-input"
            rows={3}
            value={currentSlot?.text ?? ""}
            onChange={(e) =>
              currentSlot && setPromptSlotText(currentSlot.id, e.target.value)
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
            const isActive = slot.id === currentSlotId;
            const isRenaming = slot.id === renamingId;
            return (
              <div
                key={slot.id}
                className={`prompt-deck-slot${isActive ? " prompt-deck-slot--active" : ""}${isRenaming ? " prompt-deck-slot--renaming" : ""}`}
                role="tab"
                aria-selected={isActive}
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
                      if (!isActive) switchToPromptSlot(slot.id);
                    }}
                    onDoubleClick={() => startRename(slot.id, slot.label)}
                    title={`Double-click to rename. ${slot.text || "(empty)"}`}
                  >
                    {slot.label}
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
              const newId = addAndSwitchToPromptSlot();
              // Drop straight into rename mode on add so the user names
              // it before forgetting why they made it.
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
