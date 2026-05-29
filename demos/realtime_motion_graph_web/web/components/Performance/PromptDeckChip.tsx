"use client";

import { switchToPromptSlot } from "@/lib/promptDeck";
import { usePerformanceStore } from "@/store/usePerformanceStore";

// Hero-bay surface for the prompt deck: a compact slot strip that lets
// the operator switch slots without opening Full Controls. Editing slot
// text + add/remove still live in the full PromptsTile — the chip is
// switch-only. The "..." button opens the drawer to the Styles tab
// (which hosts PromptsTile) so the operator can dive deeper without
// hunting for the toggle.

export function PromptDeckChip() {
  const slots = usePerformanceStore((s) => s.promptSlots);
  const currentSlotId = usePerformanceStore((s) => s.currentSlotId);
  if (slots.length === 0) return null;

  return (
    <div className="hero-macros-prompts">
      <div
        className="hero-macros-group-label"
        data-dd-tooltip="Active prompt slot. Tap a name to switch — the engine blends smoothly between the old and new tags. Open Full Controls (or click the ⋯) to edit slot text or add new ones."
        data-dd-tooltip-wide=""
        data-dd-tooltip-title="Prompt deck"
      >
        Prompts
      </div>
      <div className="hero-prompt-deck" role="tablist" aria-label="Prompt slots">
        {slots.map((slot) => {
          const isActive = slot.id === currentSlotId;
          return (
            <button
              key={slot.id}
              type="button"
              className={`hero-prompt-deck-slot${isActive ? " hero-prompt-deck-slot--active" : ""}`}
              role="tab"
              aria-selected={isActive}
              onClick={() => {
                if (!isActive) switchToPromptSlot(slot.id);
              }}
              title={slot.text || "(empty)"}
            >
              {slot.label}
            </button>
          );
        })}
        <button
          type="button"
          className="hero-prompt-deck-more"
          onClick={() => {
            // Dispatch a directed open-event so the drawer routes to
            // Styles (which hosts PromptsTile) in one move. AdvancedDrawer
            // listens for dd:open-drawer-tab and combines "open" + "set
            // tab" into one handler, no localStorage round-trip.
            document.dispatchEvent(
              new CustomEvent("dd:open-drawer-tab", {
                detail: { tab: "styles" },
              }),
            );
          }}
          aria-label="Edit prompts in Full Controls"
          title="Edit prompts (opens Full Controls)"
        >
          ⋯
        </button>
      </div>
    </div>
  );
}
