"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import {
  applySessionSnapshot,
  captureSessionSnapshot,
  persistSessionSnapshotAssets,
  sessionSnapshotSignature,
  validateSessionSnapshotShape,
  type SessionSnapshot,
} from "@/lib/sessionSnapshot";
import {
  deleteLocalSavedSessionRecord,
  getLocalSavedSessionRecord,
  listLocalSavedSessionRecords,
  putLocalSavedSessionRecord,
} from "@/lib/sessionAudioAssets";
import { useCurveStore } from "@/store/useCurveStore";
import { useCustomTracksStore } from "@/store/useCustomTracksStore";
import { useLoraStore } from "@/store/useLoraStore";
import { usePerformanceStore } from "@/store/usePerformanceStore";
import { useStemOverlayStore } from "@/store/useStemOverlayStore";

const STORAGE_KEY = "demon:local-saved-sessions:v1";
const ACTIVE_ID_KEY = "demon:local-saved-sessions:active-id";

export interface LocalSavedSessionRecord {
  id: string;
  name: string;
  updatedAt: number;
  snapshot: SessionSnapshot;
}

export interface LocalSavedSessionsController {
  sessions: LocalSavedSessionRecord[];
  activeSessionId: string | null;
  dirty: boolean;
  busy: boolean;
  error: string | null;
  lastSavedAt: number | null;
  save: () => Promise<boolean>;
  saveAsNew: () => Promise<boolean>;
  open: (id: string) => Promise<boolean>;
  rename: (id: string, name: string) => void;
  deleteSession: (id: string) => void;
  clearError: () => void;
}

function createId(): string {
  if (typeof crypto !== "undefined" && "randomUUID" in crypto) {
    return crypto.randomUUID();
  }
  return `${Date.now().toString(36)}-${Math.random().toString(36).slice(2)}`;
}

function sessionName(snapshot: SessionSnapshot): string {
  const fixture = snapshot.fixture.trim();
  if (!fixture) return "Untitled session";
  return fixture.replace(/\.[a-z0-9]+$/i, "") || fixture;
}

function readLegacySessions(): LocalSavedSessionRecord[] {
  if (typeof localStorage === "undefined") return [];
  try {
    const parsed = JSON.parse(localStorage.getItem(STORAGE_KEY) || "[]") as unknown;
    if (!Array.isArray(parsed)) return [];
    return parsed
      .filter((entry): entry is LocalSavedSessionRecord => {
        if (typeof entry !== "object" || entry === null) return false;
        const record = entry as Partial<LocalSavedSessionRecord>;
        return (
          typeof record.id === "string" &&
          typeof record.name === "string" &&
          typeof record.updatedAt === "number" &&
          validateSessionSnapshotShape(record.snapshot)
        );
      })
      .sort((a, b) => b.updatedAt - a.updatedAt);
  } catch {
    return [];
  }
}

async function readSessions(): Promise<LocalSavedSessionRecord[]> {
  const stored = await listLocalSavedSessionRecords<LocalSavedSessionRecord>();
  const sessions = stored.filter((record) =>
    validateSessionSnapshotShape(record.snapshot),
  );
  if (sessions.length > 0) {
    return sessions.sort((a, b) => b.updatedAt - a.updatedAt);
  }
  return readLegacySessions();
}

function readActiveId(): string | null {
  if (typeof localStorage === "undefined") return null;
  return localStorage.getItem(ACTIVE_ID_KEY);
}

function writeActiveId(id: string | null): void {
  if (typeof localStorage === "undefined") return;
  if (id) localStorage.setItem(ACTIVE_ID_KEY, id);
  else localStorage.removeItem(ACTIVE_ID_KEY);
}

export function useLocalSavedSessions(): LocalSavedSessionsController {
  const [sessions, setSessions] = useState<LocalSavedSessionRecord[]>([]);
  const [activeSessionId, setActiveSessionId] = useState<string | null>(null);
  const [activeSignature, setActiveSignature] = useState<string | null>(null);
  const [dirty, setDirty] = useState(true);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [lastSavedAt, setLastSavedAt] = useState<number | null>(null);

  const sessionsRef = useRef(sessions);
  const activeSessionIdRef = useRef(activeSessionId);
  const activeSignatureRef = useRef(activeSignature);
  const dirtyTimerRef = useRef<number | null>(null);

  useEffect(() => {
    sessionsRef.current = sessions;
  }, [sessions]);
  useEffect(() => {
    activeSessionIdRef.current = activeSessionId;
  }, [activeSessionId]);
  useEffect(() => {
    activeSignatureRef.current = activeSignature;
  }, [activeSignature]);

  const syncDirty = useCallback(() => {
    const signature = activeSignatureRef.current;
    if (!signature) {
      setDirty(true);
      return;
    }
    try {
      setDirty(sessionSnapshotSignature(captureSessionSnapshot()) !== signature);
    } catch {
      setDirty(true);
    }
  }, []);

  // Store mutations (knob turns, MIDI, automation) can fire many times per
  // frame during a performance. Coalesce them into one signature check per
  // window instead of stringifying the full snapshot on every emission.
  const scheduleSyncDirty = useCallback(() => {
    if (dirtyTimerRef.current !== null) return;
    dirtyTimerRef.current = window.setTimeout(() => {
      dirtyTimerRef.current = null;
      syncDirty();
    }, 200);
  }, [syncDirty]);

  useEffect(() => {
    let cancelled = false;
    void readSessions().then((loaded) => {
      if (cancelled) return;
      const storedActiveId = readActiveId();
      setSessions(loaded);
      if (storedActiveId) {
        const active = loaded.find((session) => session.id === storedActiveId);
        if (active) {
          setActiveSessionId(active.id);
          setActiveSignature(sessionSnapshotSignature(active.snapshot));
          setLastSavedAt(active.updatedAt);
        } else {
          writeActiveId(null);
        }
      }
    });
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    syncDirty();
  }, [activeSignature, syncDirty]);

  useEffect(() => {
    const unsubscribers = [
      usePerformanceStore.subscribe(scheduleSyncDirty),
      useLoraStore.subscribe(scheduleSyncDirty),
      useCurveStore.subscribe(scheduleSyncDirty),
      useCustomTracksStore.subscribe(scheduleSyncDirty),
      useStemOverlayStore.subscribe(scheduleSyncDirty),
    ];
    return () => {
      unsubscribers.forEach((unsubscribe) => unsubscribe());
      if (dirtyTimerRef.current !== null) {
        window.clearTimeout(dirtyTimerRef.current);
        dirtyTimerRef.current = null;
      }
    };
  }, [scheduleSyncDirty]);

  const replaceSessionSummaries = useCallback((next: LocalSavedSessionRecord[]) => {
    const ordered = [...next].sort((a, b) => b.updatedAt - a.updatedAt);
    setSessions(ordered);
  }, []);

  const saveRecord = useCallback(
    async (mode: "current" | "new"): Promise<boolean> => {
      setBusy(true);
      setError(null);
      try {
        const snapshot = captureSessionSnapshot();
        const now = Date.now();
        const currentId = activeSessionIdRef.current;
        const existing =
          mode === "current" && currentId
            ? sessionsRef.current.find((session) => session.id === currentId)
            : undefined;
        const record: LocalSavedSessionRecord = {
          id: existing?.id ?? createId(),
          name: existing?.name ?? sessionName(snapshot),
          updatedAt: now,
          snapshot,
        };
        const persisted = await persistSessionSnapshotAssets(snapshot);
        if (persisted.status !== "complete") {
          setError(persisted.message);
          return false;
        }
        await putLocalSavedSessionRecord(record);
        const next = existing
          ? sessionsRef.current.map((session) =>
              session.id === existing.id ? record : session,
            )
          : [record, ...sessionsRef.current];
        replaceSessionSummaries(next);
        writeActiveId(record.id);
        setActiveSessionId(record.id);
        setActiveSignature(sessionSnapshotSignature(snapshot));
        setLastSavedAt(now);
        setDirty(false);
        return true;
      } catch (e) {
        setError(e instanceof Error ? e.message : String(e));
        return false;
      } finally {
        setBusy(false);
      }
    },
    [replaceSessionSummaries],
  );

  const open = useCallback(async (id: string): Promise<boolean> => {
    const summary = sessionsRef.current.find((session) => session.id === id);
    if (!summary) return false;
    setBusy(true);
    setError(null);
    try {
      const stored =
        await getLocalSavedSessionRecord<LocalSavedSessionRecord>(id);
      const record = stored ?? summary;
      const restored = await applySessionSnapshot(record.snapshot);
      if (restored.status !== "complete") {
        setError(restored.message);
        return false;
      }
      writeActiveId(record.id);
      setActiveSessionId(record.id);
      setActiveSignature(sessionSnapshotSignature(record.snapshot));
      setLastSavedAt(record.updatedAt);
      setDirty(false);
      return true;
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
      return false;
    } finally {
      setBusy(false);
    }
  }, []);

  const rename = useCallback(
    (id: string, name: string) => {
      const trimmed = name.trim();
      if (!trimmed) return;
      const next = sessionsRef.current.map((session) =>
        session.id === id ? { ...session, name: trimmed } : session,
      );
      replaceSessionSummaries(next);
      void getLocalSavedSessionRecord<LocalSavedSessionRecord>(id).then(
        (record) => {
          if (record) {
            void putLocalSavedSessionRecord({ ...record, name: trimmed });
          }
        },
      );
    },
    [replaceSessionSummaries],
  );

  const deleteSession = useCallback(
    (id: string) => {
      void deleteLocalSavedSessionRecord(id);
      replaceSessionSummaries(
        sessionsRef.current.filter((session) => session.id !== id),
      );
      if (activeSessionIdRef.current === id) {
        writeActiveId(null);
        setActiveSessionId(null);
        setActiveSignature(null);
        setLastSavedAt(null);
        setDirty(true);
      }
    },
    [replaceSessionSummaries],
  );

  return useMemo(
    () => ({
      sessions,
      activeSessionId,
      dirty,
      busy,
      error,
      lastSavedAt,
      save: () => saveRecord("current"),
      saveAsNew: () => saveRecord("new"),
      open,
      rename,
      deleteSession,
      clearError: () => setError(null),
    }),
    [
      activeSessionId,
      busy,
      deleteSession,
      dirty,
      error,
      lastSavedAt,
      open,
      rename,
      saveRecord,
      sessions,
    ],
  );
}
