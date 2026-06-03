"use client";

// Tiny chip that shows MIDI device count + last learn message; populated
// inside #install-midi-slot (already styled by globals.css). Uses a portal-
// ish injection: just render into the slot via a regular React subtree
// inside <OperatorStrip />.

import { useMidiStore } from "@/store/useMidiStore";
import { useUiStore } from "@/store/useUiStore";

export function MidiBadge() {
  const status = useMidiStore((s) => s.status);
  const cls = `midi-badge midi-${status.tone}`;
  // Clickable to open configuration, but keep the <div> look — the
  // `.midi-badge` class owns the chrome, so a <button> would fight UA
  // styles. role/tabIndex keep it keyboard-reachable.
  return (
    <div
      className={cls}
      role="button"
      tabIndex={0}
      style={{ cursor: "pointer" }}
      data-dd-tooltip="MIDI status — click to open configuration; right-click any control to learn"
      onClick={() => useUiStore.getState().setConfigOpen(true)}
      onKeyDown={(e) => {
        if (e.key === "Enter" || e.key === " ") {
          e.preventDefault();
          useUiStore.getState().setConfigOpen(true);
        }
      }}
    >
      {status.message}
    </div>
  );
}
