"use client";

import { useEffect } from "react";

import { useRecordingStore } from "@/store/useRecordingStore";
import { useSessionStore } from "@/store/useSessionStore";

// Owns the MediaRecorder lifecycle. UI components dispatch
// document.dispatchEvent(new CustomEvent("dd:toggle-record")) — keyboard
// shortcut handler does the same. Mirrors the dd:toggle-drawer pattern.
//
// Strategy notes:
//   • Audio-only. Stream comes from AudioPlayer.getRecordingStream() — a
//     MediaStreamAudioDestinationNode tee'd off the same worklet output the
//     user hears, at 48 kHz / 2 ch.
//   • MIME ladder picks Opus-in-WebM first (transparent at 192 kbps), falls
//     back to AAC-in-MP4 for Safari/iOS, gracefully disables otherwise.
//   • start(1000) yields 1-second chunks so a tab crash loses at most ~1s.
//   • Visibility / AudioContext statechange handlers auto-pause and surface
//     status into the existing StatusBar via useSessionStore.setStatus.

const SOFT_CAP_MS = 60 * 60 * 1000; // 60 minutes
const SOFT_CAP_BYTES = 150 * 1024 * 1024; // ~150 MB

interface MimeChoice {
  mime: string;
  ext: string;
  bitrate: number;
}

const MIME_LADDER: MimeChoice[] = [
  { mime: "audio/webm;codecs=opus", ext: "webm", bitrate: 192_000 },
  { mime: "audio/webm", ext: "webm", bitrate: 192_000 },
  { mime: "audio/mp4;codecs=mp4a.40.2", ext: "m4a", bitrate: 256_000 },
  { mime: "audio/mp4", ext: "m4a", bitrate: 256_000 },
];

function pickMime(): MimeChoice | null {
  if (typeof window === "undefined") return null;
  if (typeof MediaRecorder === "undefined") return null;
  for (const choice of MIME_LADDER) {
    try {
      if (MediaRecorder.isTypeSupported(choice.mime)) return choice;
    } catch {
      // Some browsers throw on bare types; keep walking the ladder.
    }
  }
  return null;
}

export function isRecordingSupported(): boolean {
  return pickMime() !== null;
}

export function useRecording() {
  useEffect(() => {
    let recorder: MediaRecorder | null = null;
    let chunks: Blob[] = [];
    let pickedMime: MimeChoice | null = null;
    let startedAt = 0;
    let pausedTotalMs = 0;
    let pausedSinceMs = 0;
    let capTimer: number | null = null;
    let visibilityPaused = false;

    function setMessage(msg: string) {
      // Reuse the StatusBar — it shows whenever message != "" and != "Playing".
      useSessionStore.getState().setStatus(
        useSessionStore.getState().status,
        msg,
      );
    }

    function clearCapTimer() {
      if (capTimer !== null) {
        window.clearTimeout(capTimer);
        capTimer = null;
      }
    }

    function teardown() {
      clearCapTimer();
      if (recorder) {
        try {
          recorder.ondataavailable = null;
          recorder.onstop = null;
          recorder.onerror = null;
        } catch {}
      }
      recorder = null;
      chunks = [];
      pickedMime = null;
      startedAt = 0;
      pausedTotalMs = 0;
      pausedSinceMs = 0;
      visibilityPaused = false;
    }

    function finalize() {
      if (!recorder || !pickedMime) return;
      const { mime, ext } = pickedMime;

      useRecordingStore.getState().set({ kind: "finalizing" });

      const onstop = () => {
        const blob = new Blob(chunks, { type: mime });
        const url = URL.createObjectURL(blob);
        const now = performance.now();
        const elapsed = now - startedAt - pausedTotalMs;

        useRecordingStore.getState().set({
          kind: "preview",
          blob,
          url,
          mime,
          ext,
          durationMs: Math.max(0, elapsed),
        });
        // Brief celebratory note in the StatusBar.
        const seconds = Math.max(1, Math.round(elapsed / 1000));
        setMessage(`Saved ${seconds}s clip`);
        teardown();
      };

      try {
        if (recorder.state !== "inactive") {
          recorder.onstop = onstop;
          recorder.stop();
        } else {
          onstop();
        }
      } catch (err) {
        console.warn("[useRecording] stop failed", err);
        teardown();
        useRecordingStore.getState().set({
          kind: "error",
          reason: "Could not finalize recording",
        });
      }
    }

    function start() {
      const cur = useRecordingStore.getState().state;
      if (cur.kind !== "idle" && cur.kind !== "preview" && cur.kind !== "error") {
        return;
      }

      // If a previous preview exists, revoke its url before starting fresh.
      if (cur.kind === "preview") {
        try {
          URL.revokeObjectURL(cur.url);
        } catch {}
      }

      const session = useSessionStore.getState();
      if (session.status === "idle" || !session.player) {
        useRecordingStore.getState().set({ kind: "idle" });
        // Show the warning anchored to the record buttons themselves —
        // do NOT overwrite session.message, which is reserved for
        // loading/connecting/error progress in the global StatusBar.
        const { setWarning } = useRecordingStore.getState();
        setWarning("Wait for connection first");
        window.setTimeout(() => {
          if (
            useRecordingStore.getState().warning ===
            "Wait for connection first"
          ) {
            setWarning(null);
          }
        }, 2000);
        return;
      }

      const choice = pickMime();
      if (!choice) {
        useRecordingStore.getState().set({
          kind: "error",
          reason: "Recording not supported on this browser",
        });
        return;
      }

      const stream = session.player.getRecordingStream();
      if (!stream) {
        useRecordingStore.getState().set({
          kind: "error",
          reason: "Audio engine not ready",
        });
        return;
      }

      useRecordingStore.getState().set({ kind: "arming" });

      try {
        recorder = new MediaRecorder(stream, {
          mimeType: choice.mime,
          audioBitsPerSecond: choice.bitrate,
        });
      } catch (err) {
        console.warn("[useRecording] MediaRecorder init failed", err);
        useRecordingStore.getState().set({
          kind: "error",
          reason: "Could not start recorder",
        });
        teardown();
        return;
      }

      pickedMime = choice;
      chunks = [];
      pausedTotalMs = 0;
      pausedSinceMs = 0;
      startedAt = performance.now();

      recorder.ondataavailable = (e: BlobEvent) => {
        if (!e.data || e.data.size === 0) return;
        chunks.push(e.data);
        useRecordingStore.getState().bumpBytes(e.data.size);
        const cur = useRecordingStore.getState().state;
        if (cur.kind === "recording" && cur.bytes >= SOFT_CAP_BYTES) {
          setMessage("Long clip saved — start a new one to keep going");
          finalize();
        }
      };
      recorder.onerror = (e: Event) => {
        console.warn("[useRecording] recorder error", e);
        useRecordingStore.getState().set({
          kind: "error",
          reason: "Recorder error",
        });
        teardown();
      };

      try {
        recorder.start(1000);
      } catch (err) {
        console.warn("[useRecording] start() failed", err);
        useRecordingStore.getState().set({
          kind: "error",
          reason: "Could not start recorder",
        });
        teardown();
        return;
      }

      useRecordingStore.getState().set({
        kind: "recording",
        startedAt,
        bytes: 0,
        pausedMs: 0,
      });

      capTimer = window.setTimeout(() => {
        if (useRecordingStore.getState().state.kind === "recording") {
          setMessage("Long clip saved — start a new one to keep going");
          finalize();
        }
      }, SOFT_CAP_MS);
    }

    function stop() {
      const cur = useRecordingStore.getState().state;
      if (cur.kind !== "recording" && cur.kind !== "paused") return;
      if (cur.kind === "paused") {
        // Closing on a paused recorder loses the in-flight chunk on Safari;
        // resume momentarily so onstop fires cleanly.
        try {
          recorder?.resume();
        } catch {}
        pausedTotalMs += performance.now() - cur.pausedAt;
      }
      finalize();
    }

    function togglePause() {
      const cur = useRecordingStore.getState().state;
      if (!recorder) return;
      if (cur.kind === "recording") {
        try {
          recorder.pause();
        } catch {
          return;
        }
        pausedSinceMs = performance.now();
        useRecordingStore.getState().set({
          kind: "paused",
          startedAt: cur.startedAt,
          bytes: cur.bytes,
          pausedMs: cur.pausedMs,
          pausedAt: pausedSinceMs,
        });
      } else if (cur.kind === "paused") {
        try {
          recorder.resume();
        } catch {
          return;
        }
        const delta = performance.now() - cur.pausedAt;
        pausedTotalMs += delta;
        useRecordingStore.getState().set({
          kind: "recording",
          startedAt: cur.startedAt,
          bytes: cur.bytes,
          pausedMs: cur.pausedMs + delta,
        });
      }
    }

    function dismissPreview() {
      const cur = useRecordingStore.getState().state;
      if (cur.kind === "preview") {
        try {
          URL.revokeObjectURL(cur.url);
        } catch {}
      }
      useRecordingStore.getState().set({ kind: "idle" });
      if (useSessionStore.getState().message.startsWith("Saved ")) {
        setMessage("");
      }
    }

    function onToggle() {
      const cur = useRecordingStore.getState().state;
      if (cur.kind === "idle" || cur.kind === "error") {
        start();
      } else if (cur.kind === "recording" || cur.kind === "paused") {
        stop();
      } else if (cur.kind === "preview") {
        dismissPreview();
        start();
      }
      // arming/finalizing: ignore — debounce double-taps.
    }

    function onPause() {
      togglePause();
    }

    function onDismiss() {
      dismissPreview();
    }

    function onVisibilityChange() {
      const cur = useRecordingStore.getState().state;
      if (document.visibilityState === "hidden") {
        if (cur.kind === "recording") {
          togglePause();
          visibilityPaused = true;
          setMessage("Paused — tab hidden");
        }
      } else if (document.visibilityState === "visible" && visibilityPaused) {
        if (useRecordingStore.getState().state.kind === "paused") {
          togglePause();
          if (useSessionStore.getState().message === "Paused — tab hidden") {
            setMessage("");
          }
        }
        visibilityPaused = false;
      }
    }

    document.addEventListener("dd:toggle-record", onToggle);
    document.addEventListener("dd:pause-record", onPause);
    document.addEventListener("dd:dismiss-record-preview", onDismiss);
    document.addEventListener("visibilitychange", onVisibilityChange);

    // Auto-stop if the AudioContext is interrupted (e.g. iOS phone call).
    const unsubscribeSession = useSessionStore.subscribe((s, prev) => {
      if (s.player !== prev.player) {
        // Session reset (or new session) — finalize anything in-flight.
        const cur = useRecordingStore.getState().state;
        if (cur.kind === "recording" || cur.kind === "paused") {
          setMessage("Audio dropped — saved what we had");
          finalize();
        }
      }
    });

    return () => {
      document.removeEventListener("dd:toggle-record", onToggle);
      document.removeEventListener("dd:pause-record", onPause);
      document.removeEventListener("dd:dismiss-record-preview", onDismiss);
      document.removeEventListener("visibilitychange", onVisibilityChange);
      unsubscribeSession();
      // If the page is unmounting mid-record, ditch the in-progress data.
      if (recorder && recorder.state !== "inactive") {
        try {
          recorder.stop();
        } catch {}
      }
      teardown();
    };
  }, []);
}
