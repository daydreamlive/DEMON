"use client";

import { usePerformanceStore } from "@/store/usePerformanceStore";
import { useSessionStore } from "@/store/useSessionStore";

// Orchestration for the prompt deck. The engine only knows two slots
// (A/B) and a lerp between them. The deck is a many-slot UX layered on
// top: at any moment one logical slot is "current" and lives in one of
// the two engine slots; switching to another logical slot loads it into
// the inactive engine slot and tweens prompt_blend toward it.
//
// State lives in usePerformanceStore: promptSlots / currentSlotId /
// physicalSlot (0 = current in A, 1 = current in B). This module is
// stateless — it reads/writes the store and pokes the WS.

/**
 * Switch the active logical slot to `targetId`. No-op if `targetId` is
 * already current or doesn't exist. Side effects, in order:
 *   1. Load the target's text into the *inactive* engine slot
 *      (promptA if physicalSlot is currently 1; promptB if 0).
 *   2. sendPrompt(A, B) so the server re-encodes the new pair.
 *   3. setSlider("prompt_blend", target) — the smoothing tween animates
 *      the engine from the old logical slot to the new one. If the
 *      Smooth toggle is off, the move is instant.
 *   4. Commit physicalSlot + currentSlotId.
 */
export function switchToPromptSlot(targetId: string): void {
  const perf = usePerformanceStore.getState();
  if (perf.currentSlotId === targetId) return;
  const target = perf.promptSlots.find((s) => s.id === targetId);
  if (!target) return;
  const current = perf.promptSlots.find((s) => s.id === perf.currentSlotId);
  if (!current) return;

  const inactive: 0 | 1 = perf.physicalSlot === 0 ? 1 : 0;
  if (inactive === 1) {
    // Current sits in A; target loads into B.
    perf.setPromptA(current.text);
    perf.setPromptB(target.text);
  } else {
    // Current sits in B; target loads into A.
    perf.setPromptA(target.text);
    perf.setPromptB(current.text);
  }

  const remote = useSessionStore.getState().remote;
  if (remote) {
    const tagsA = inactive === 1 ? current.text : target.text;
    const tagsB = inactive === 1 ? target.text : current.text;
    remote.sendPrompt(tagsA, perf.activeKey, perf.activeTimeSignature, tagsB);
  }

  // Blend target == inactive engine slot. The smoothing tween (or
  // instant jump, if Smooth is off) carries the engine from the old
  // current to the new one. usePromptBlendSync throttles the wire
  // sends.
  perf.setSlider("prompt_blend", inactive);
  perf.setCurrentSlot(targetId, inactive);
}

/**
 * Add a new slot and switch to it. Convenience wrapper used by the "+"
 * affordance in PromptsTile — adding without switching would leave the
 * new (empty) slot dangling in the deck. Returns the new slot id.
 */
export function addAndSwitchToPromptSlot(label?: string): string {
  const id = usePerformanceStore.getState().addPromptSlot(label, "");
  switchToPromptSlot(id);
  return id;
}

/**
 * Remove a slot. If it's the active one, the store's removePromptSlot
 * already advances currentSlotId to a neighbor; we then push the new
 * current slot's text to the engine so the wire matches. If the removed
 * slot was *only* the inactive engine partner, nothing on the wire
 * changes — the next switch loads a fresh partner anyway.
 */
export function removePromptSlot(id: string): void {
  const before = usePerformanceStore.getState();
  if (before.currentSlotId !== id) {
    before.removePromptSlot(id);
    return;
  }
  before.removePromptSlot(id);
  const after = usePerformanceStore.getState();
  const newCurrent = after.promptSlots.find((s) => s.id === after.currentSlotId);
  if (!newCurrent) return;
  // The removed slot was the active one; the store already chose a
  // neighbor as currentSlotId but physicalSlot still points at the
  // engine slot that held the deleted text. Re-load the new current
  // into that slot and re-send so the engine reflects what the UI
  // shows. Blend stays where it is — no transition.
  if (after.physicalSlot === 0) after.setPromptA(newCurrent.text);
  else after.setPromptB(newCurrent.text);
  const remote = useSessionStore.getState().remote;
  if (remote) {
    const fresh = usePerformanceStore.getState();
    remote.sendPrompt(
      fresh.promptA,
      fresh.activeKey,
      fresh.activeTimeSignature,
      fresh.promptB,
    );
  }
}
