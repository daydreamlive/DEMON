"use client";

import { useEffect } from "react";

import { usePerformanceStore } from "@/store/usePerformanceStore";
import { useSessionStore } from "@/store/useSessionStore";

// Mirror of the server's timbre_* / structure_* acks into the perf
// store. The actual upload kickoff lives in the RefControl component;
// this hook only handles incoming acks.
//
// Failure semantics differ between the two kinds and match the backend:
//
//   - timbre: backend rolls back to the prior override on failure, so
//     the displayed name is still correct as-is. We just surface the
//     error message in the status bar.
//   - structure: backend fully clears any prior override on failure
//     (the error path zeroes struct_audio_ref / struct_context_ref and
//     restores stream.source). We mirror that by clearing structName.
//
// Both clear their displayed name when the session leaves "ready" so a
// stale name from a previous session doesn't carry over to the next —
// the backend always boots with no override.

export type RefKind = "timbre" | "structure";

interface KindConfig {
  setEvent: string;
  clearedEvent: string;
  failedEvent: string;
  statusPrefix: string;
  setName: (name: string | null) => void;
  defaultName: string;
  clearOnFail: boolean;
}

function configFor(kind: RefKind): KindConfig {
  if (kind === "timbre") {
    return {
      setEvent: "timbre_set",
      clearedEvent: "timbre_cleared",
      failedEvent: "timbre_failed",
      statusPrefix: "Timbre",
      setName: (n) => usePerformanceStore.getState().setTimbreName(n),
      defaultName: "timbre",
      clearOnFail: false,
    };
  }
  return {
    setEvent: "structure_set",
    clearedEvent: "structure_cleared",
    failedEvent: "structure_failed",
    statusPrefix: "Structure",
    setName: (n) => usePerformanceStore.getState().setStructName(n),
    defaultName: "structure",
    clearOnFail: true,
  };
}

export function useRefSourceAcks(kind: RefKind) {
  useEffect(() => {
    const cfg = configFor(kind);
    let attached: { remote: EventTarget; off: () => void } | null = null;

    const attach = (remote: EventTarget) => {
      const onSet = (e: Event) => {
        const detail = (e as CustomEvent<{ name?: string }>).detail;
        cfg.setName(detail?.name ?? cfg.defaultName);
        useSessionStore.getState().setStatus("ready", "");
      };
      const onCleared = () => {
        cfg.setName(null);
        useSessionStore.getState().setStatus("ready", "");
      };
      const onFailed = (e: Event) => {
        const err = (e as CustomEvent<string>).detail || "upload failed";
        if (cfg.clearOnFail) cfg.setName(null);
        useSessionStore.getState().setStatus(
          "ready", `${cfg.statusPrefix}: ${err}`,
        );
      };
      remote.addEventListener(cfg.setEvent, onSet);
      remote.addEventListener(cfg.clearedEvent, onCleared);
      remote.addEventListener(cfg.failedEvent, onFailed);
      const off = () => {
        remote.removeEventListener(cfg.setEvent, onSet);
        remote.removeEventListener(cfg.clearedEvent, onCleared);
        remote.removeEventListener(cfg.failedEvent, onFailed);
      };
      attached = { remote, off };
    };

    const detach = () => {
      attached?.off();
      attached = null;
    };

    const apply = (remote: EventTarget | null) => {
      if (attached?.remote === remote) return;
      detach();
      if (remote) attach(remote);
    };

    apply(useSessionStore.getState().remote ?? null);

    const unsub = useSessionStore.subscribe((s, prev) => {
      if (s.remote !== prev.remote) {
        apply(s.remote ?? null);
      }
      if (prev.status === "ready" && s.status !== "ready") {
        cfg.setName(null);
      }
    });

    return () => {
      detach();
      unsub();
    };
  }, [kind]);
}
