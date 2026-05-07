"use client";

// Tiny chip that shows MIDI device count + last learn message; populated
// inside #install-midi-slot (already styled by globals.css). Uses a portal-
// ish injection: just render into the slot via a regular React subtree
// inside <OperatorStrip />.

import { useMidiStore } from "@/store/useMidiStore";

export function MidiBadge() {
  const status = useMidiStore((s) => s.status);
  const cls = `midi-badge midi-${status.tone}`;
  return (
    <div className={cls} title="MIDI status — right-click any slider to learn">
      {status.message}
    </div>
  );
}
