"use client";

import { useState } from "react";

import type { LocalSavedSessionsController } from "@/hooks/useLocalSavedSessions";
import { confirm } from "@/store/useConfirmStore";

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

export function SessionsTile({ sessions }: Props) {
  const [editingId, setEditingId] = useState<string | null>(null);
  const [draftName, setDraftName] = useState("");

  function beginRename(id: string, name: string): void {
    setEditingId(id);
    setDraftName(name);
  }

  function commitRename(): void {
    if (!editingId) return;
    sessions.rename(editingId, draftName);
    setEditingId(null);
    setDraftName("");
  }

  async function deleteSession(id: string, name: string): Promise<void> {
    const ok = await confirm({
      title: "Delete Saved Session",
      message: `Delete "${name}" from this browser? Audio cached for other saved sessions is left untouched.`,
      confirmLabel: "Delete",
      variant: "danger",
    });
    if (ok) sessions.deleteSession(id);
  }

  return (
    <div className="sessions-tile">
      <div className="sessions-tile-header">
        <h3 className="sessions-tile-title">Local Saved Sessions</h3>
        <div className="sessions-tile-actions">
          <button
            type="button"
            className="sessions-action-btn sessions-action-btn--secondary"
            disabled={sessions.busy}
            onClick={() => void sessions.saveAsNew()}
          >
            Save copy
          </button>
          <button
            type="button"
            className="sessions-action-btn"
            disabled={sessions.busy}
            onClick={() => void sessions.save()}
          >
            {sessions.busy ? "Saving" : "Save"}
          </button>
        </div>
      </div>

      {sessions.error && (
        <div className="sessions-error" role="status">
          {sessions.error}
        </div>
      )}

      {sessions.sessions.length === 0 ? (
        <div className="sessions-empty">
          <p>No local saved sessions yet.</p>
          <button
            type="button"
            className="sessions-action-btn"
            disabled={sessions.busy}
            onClick={() => void sessions.save()}
          >
            Save this session
          </button>
        </div>
      ) : (
        <div className="sessions-list">
          {sessions.sessions.map((session) => {
            const current = session.id === sessions.activeSessionId;
            const editing = session.id === editingId;
            return (
              <div
                key={session.id}
                className={`sessions-row${current ? " sessions-row--current" : ""}`}
              >
                <div className="sessions-row-main">
                  {editing ? (
                    <input
                      className="sessions-row-name-input"
                      value={draftName}
                      autoFocus
                      onChange={(e) => setDraftName(e.target.value)}
                      onBlur={commitRename}
                      onKeyDown={(e) => {
                        if (e.key === "Enter") commitRename();
                        if (e.key === "Escape") {
                          setEditingId(null);
                          setDraftName("");
                        }
                      }}
                    />
                  ) : (
                    <button
                      type="button"
                      className="sessions-row-name"
                      onClick={() => void sessions.open(session.id)}
                      title={session.name}
                    >
                      {session.name}
                    </button>
                  )}
                  <span className="sessions-row-age">
                    {current ? "Current" : "Saved"} {formatAge(session.updatedAt)}
                  </span>
                </div>
                <div className="sessions-row-actions">
                  <button
                    type="button"
                    className="sessions-row-btn sessions-row-btn--open"
                    disabled={sessions.busy}
                    onClick={() => void sessions.open(session.id)}
                  >
                    Open
                  </button>
                  <button
                    type="button"
                    className="sessions-row-btn"
                    disabled={sessions.busy || editing}
                    onClick={() => beginRename(session.id, session.name)}
                  >
                    Rename
                  </button>
                  <button
                    type="button"
                    className="sessions-row-btn sessions-row-btn--delete"
                    disabled={sessions.busy}
                    onClick={() => void deleteSession(session.id, session.name)}
                  >
                    Delete
                  </button>
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
