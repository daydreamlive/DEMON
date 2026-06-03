"use client";

import { usePerformanceStore } from "@/store/usePerformanceStore";
import { useSessionStore } from "@/store/useSessionStore";

// Orchestration for the prompt deck.
//
// THREE concepts kept deliberately separate:
//
//   focusedSlotId  → UI focus (textarea binding only; no engine effect
//                    until that slot is also loaded as A or B).
//   currentSlotId  → engine A endpoint. Only changes when the operator
//                    explicitly picks via the crossfader's A label
//                    popover.
//   blendPartnerId → engine B endpoint. Only changes via the
//                    crossfader's B label popover. Auto-backfilled on
//                    deletes so the crossfader stays valid for 2+
//                    slots.
//
// Decoupling focus from the engine endpoints is what stops the
// crossfader's A/B labels from shifting under the operator every time
// they tap a chip in the deck. Tap a chip → textarea updates. Slot
// stays loaded where it was. To actually swap the playing prompt,
// you click the A or B label on the crossfader.

function sendCurrent(): void {
  const remote = useSessionStore.getState().remote;
  if (!remote) return;
  const s = usePerformanceStore.getState();
  remote.sendPrompt(
    s.promptA,
    s.activeKey,
    s.activeTimeSignature,
    s.blendPartnerId !== null ? s.promptB : undefined,
  );
}

/**
 * Internal: keep blendPartnerId pointing at a valid non-current slot
 * whenever there are 2+ slots; clear it (and promptB) when only 1
 * remains. Called after every mutation that could invalidate the
 * partner (add / remove / explicit re-point). Mutates store state in
 * place — caller runs sendCurrent() afterward.
 */
function ensureValidPartner(): void {
  const s = usePerformanceStore.getState();
  if (s.promptSlots.length < 2) {
    if (s.blendPartnerId !== null) {
      s.setBlendPartner(null);
      s.setPromptB("");
      s.setSlider("prompt_blend", 0);
    }
    return;
  }
  const partnerExists =
    s.blendPartnerId !== null &&
    s.promptSlots.some((slot) => slot.id === s.blendPartnerId);
  const partnerIsCurrent = s.blendPartnerId === s.currentSlotId;
  if (partnerExists && !partnerIsCurrent) return;
  const pick = s.promptSlots.find((slot) => slot.id !== s.currentSlotId);
  if (!pick) return;
  s.setBlendPartner(pick.id);
  s.setPromptB(pick.text);
}

/**
 * Tap-on-chip handler. Pure UI focus shift — binds the textarea to
 * the target slot. Does NOT touch engine A / B / blend. The operator
 * is saying "I want to edit / read this slot," not "I want to hear
 * it." To make the slot audible, they click the A or B label on the
 * crossfader.
 */
export function focusPromptSlot(targetId: string): void {
  const perf = usePerformanceStore.getState();
  if (perf.focusedSlotId === targetId) return;
  if (!perf.promptSlots.some((s) => s.id === targetId)) return;
  perf.setFocusedSlot(targetId);
}

/**
 * Load `targetId` into engine A — used by the crossfader's A-label
 * popover. Updates promptA, re-sends, and also moves the textarea
 * focus to the target so the operator can immediately edit what
 * they just loaded. Does NOT touch prompt_blend; the operator's
 * slider position (or MIDI knob) is preserved.
 */
export function setBlendEndpointA(targetId: string): void {
  const perf = usePerformanceStore.getState();
  if (targetId === perf.currentSlotId) return;
  if (targetId === perf.blendPartnerId) return; // can't blend with self
  const target = perf.promptSlots.find((s) => s.id === targetId);
  if (!target) return;
  perf.setPromptA(target.text);
  perf.setCurrentSlot(targetId);
  perf.setFocusedSlot(targetId);
  ensureValidPartner();
  sendCurrent();
}

/**
 * Load `targetId` into engine B — used by the crossfader's B-label
 * popover. Same shape as setBlendEndpointA but for the B side. Also
 * moves textarea focus to the loaded slot.
 */
export function setBlendPartner(targetId: string): void {
  const perf = usePerformanceStore.getState();
  if (targetId === perf.blendPartnerId) return;
  if (targetId === perf.currentSlotId) return;
  const target = perf.promptSlots.find((s) => s.id === targetId);
  if (!target) return;
  perf.setBlendPartner(targetId);
  perf.setPromptB(target.text);
  perf.setFocusedSlot(targetId);
  sendCurrent();
}

/**
 * Add a new slot and focus it (for immediate rename). Does not load
 * to A or B — the new slot is empty, so the operator authors it
 * first and explicitly loads it when ready. Returns the new slot id.
 */
export function addAndFocusPromptSlot(label?: string): string {
  const id = usePerformanceStore.getState().addPromptSlot(label, "");
  ensureValidPartner();
  focusPromptSlot(id);
  return id;
}

/**
 * Remove a slot. The store re-points currentSlotId / blendPartnerId /
 * focusedSlotId if any pointed at the removed slot; this wrapper
 * then re-loads promptA/promptB from the new endpoints, runs
 * ensureValidPartner to backfill a partner when needed, and re-sends.
 */
export function removePromptSlot(id: string): void {
  const before = usePerformanceStore.getState();
  const wasCurrent = before.currentSlotId === id;
  const wasPartner = before.blendPartnerId === id;
  before.removePromptSlot(id);

  if (!wasCurrent && !wasPartner) return;

  const after = usePerformanceStore.getState();
  if (wasCurrent) {
    const newCurrent = after.promptSlots.find(
      (s) => s.id === after.currentSlotId,
    );
    if (newCurrent) after.setPromptA(newCurrent.text);
  }
  ensureValidPartner();
  sendCurrent();
}
