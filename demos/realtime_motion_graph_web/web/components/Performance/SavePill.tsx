"use client";

import type { LocalSavedSessionsController } from "@/hooks/useLocalSavedSessions";
import { useSessionStore } from "@/store/useSessionStore";

interface Props {
  sessions: LocalSavedSessionsController;
}

function formatAge(timestamp: number): string {
  const seconds = Math.max(0, Math.floor((Date.now() - timestamp) / 1000));
  if (seconds < 60) return "just now";
  const minutes = Math.floor(seconds / 60);
  if (minutes < 60) return `${minutes}m ago`;
  const hours = Math.floor(minutes / 60);
  if (hours < 24) return `${hours}h ago`;
  const days = Math.floor(hours / 24);
  return `${days}d ago`;
}

export function SavePill({ sessions }: Props) {
  const status = useSessionStore((s) => s.status);
  if (status === "idle") return null;

  const dirty = sessions.dirty || !sessions.activeSessionId;
  const label = sessions.busy
    ? "Saving..."
    : dirty
      ? "Unsaved changes"
      : sessions.lastSavedAt
        ? `Saved ${formatAge(sessions.lastSavedAt)}`
        : "Local saves";

  return (
    <div className={`save-pill${dirty ? " save-pill--dirty" : ""}`}>
      {dirty && <span className="save-pill-dot" aria-hidden="true" />}
      <span className="save-pill-label">{label}</span>
      {dirty && (
        <button
          type="button"
          className="save-pill-btn"
          disabled={sessions.busy}
          onClick={() => void sessions.save()}
        >
          Save
        </button>
      )}
    </div>
  );
}
