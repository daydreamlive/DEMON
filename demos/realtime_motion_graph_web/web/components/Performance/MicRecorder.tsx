"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";

import { togglePauseAndAudio } from "@/engine/audio/togglePauseAndAudio";
import { usePerformanceStore } from "@/store/usePerformanceStore";

// "Record audio" path for the AudioSourceCrate. Opens a modal, asks for
// the mic, PAUSES playback so the daydream output isn't bleeding into
// the recording, captures up to 60 s, and hands back a File that flows
// through the exact same decode→AlmostReadyDialog→swap path as an
// upload (the caller passes our File to onFilePicked).
//
// Same MIME ladder as useRecording (Opus-in-WebM, AAC-in-MP4 fallback)
// so the produced File decodes everywhere decodeAudioFile runs.

const MAX_MS = 60_000;

const MIME_LADDER: { mime: string; ext: string; bitrate: number }[] = [
  { mime: "audio/webm;codecs=opus", ext: "webm", bitrate: 192_000 },
  { mime: "audio/webm", ext: "webm", bitrate: 192_000 },
  { mime: "audio/mp4;codecs=mp4a.40.2", ext: "m4a", bitrate: 256_000 },
  { mime: "audio/mp4", ext: "m4a", bitrate: 256_000 },
];

function pickMime(): { mime: string; ext: string; bitrate: number } | null {
  if (typeof MediaRecorder === "undefined") return null;
  for (const c of MIME_LADDER) {
    try {
      if (MediaRecorder.isTypeSupported(c.mime)) return c;
    } catch {
      /* isTypeSupported can throw on old impls — keep laddering */
    }
  }
  return null;
}

type Phase = "init" | "ready" | "recording" | "error";

// Synchronous capability check so the initial state is correct without
// a setState-in-effect (the effect only does the async getUserMedia).
function initialSupport(): { phase: Phase; error: string } {
  if (
    typeof navigator === "undefined" ||
    !navigator.mediaDevices?.getUserMedia
  ) {
    return {
      phase: "error",
      error: "Microphone capture isn't supported in this browser.",
    };
  }
  if (!pickMime()) {
    return {
      phase: "error",
      error: "Audio recording isn't supported in this browser.",
    };
  }
  return { phase: "init", error: "" };
}

export function MicRecorder({
  onComplete,
  onClose,
}: {
  onComplete: (file: File) => void;
  onClose: () => void;
}) {
  const [init] = useState(initialSupport);
  const [phase, setPhase] = useState<Phase>(init.phase);
  const [error, setError] = useState<string>(init.error);
  const [elapsedMs, setElapsedMs] = useState(0);

  const streamRef = useRef<MediaStream | null>(null);
  const recorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<BlobPart[]>([]);
  const tickRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const startedAtRef = useRef(0);
  // Whether WE paused playback (so we only resume what we paused).
  const weePausedRef = useRef(false);
  const doneRef = useRef(false);

  const stopTracks = useCallback(() => {
    streamRef.current?.getTracks().forEach((t) => t.stop());
    streamRef.current = null;
  }, []);

  const clearTick = useCallback(() => {
    if (tickRef.current) {
      clearInterval(tickRef.current);
      tickRef.current = null;
    }
  }, []);

  // Restore playback only if we were the ones who paused it.
  const restorePlayback = useCallback(() => {
    if (weePausedRef.current && usePerformanceStore.getState().paused) {
      togglePauseAndAudio();
    }
    weePausedRef.current = false;
  }, []);

  // Acquire mic + pause playback on mount. Capability is already
  // resolved into initial state — skip if unsupported.
  useEffect(() => {
    if (init.phase !== "init") return;
    let cancelled = false;
    (async () => {
      try {
        // Raw-ish audio for music fidelity; fall back to plain audio
        // if the constraint object is rejected.
        let stream: MediaStream;
        try {
          stream = await navigator.mediaDevices.getUserMedia({
            audio: {
              echoCancellation: false,
              noiseSuppression: false,
              autoGainControl: false,
            },
          });
        } catch {
          stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        }
        if (cancelled) {
          stream.getTracks().forEach((t) => t.stop());
          return;
        }
        streamRef.current = stream;
        // Pause the daydream output so it doesn't bleed into the mic.
        if (!usePerformanceStore.getState().paused) {
          togglePauseAndAudio();
          weePausedRef.current = true;
        }
        setPhase("ready");
      } catch (e) {
        if (cancelled) return;
        const name = e instanceof DOMException ? e.name : "";
        setError(
          name === "NotAllowedError" || name === "SecurityError"
            ? "Microphone access was denied."
            : name === "NotFoundError"
              ? "No microphone found."
              : `Couldn't open the microphone${name ? ` (${name})` : ""}.`,
        );
        setPhase("error");
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [init.phase]);

  // Hard cleanup on unmount (covers any exit path).
  useEffect(() => {
    return () => {
      clearTick();
      try {
        if (recorderRef.current && recorderRef.current.state !== "inactive") {
          recorderRef.current.stop();
        }
      } catch {
        /* already stopped */
      }
      stopTracks();
      // Don't leave the user paused if we paused them and they bailed.
      if (!doneRef.current) restorePlayback();
    };
  }, [clearTick, stopTracks, restorePlayback]);

  const finish = useCallback(
    (file: File | null) => {
      if (doneRef.current) return;
      doneRef.current = true;
      clearTick();
      stopTracks();
      restorePlayback();
      if (file) onComplete(file);
      onClose();
    },
    [clearTick, stopTracks, restorePlayback, onComplete, onClose],
  );

  const startRecording = useCallback(() => {
    const stream = streamRef.current;
    const choice = pickMime();
    if (!stream || !choice) return;
    chunksRef.current = [];
    let rec: MediaRecorder;
    try {
      rec = new MediaRecorder(stream, {
        mimeType: choice.mime,
        audioBitsPerSecond: choice.bitrate,
      });
    } catch {
      setPhase("error");
      setError("Couldn't start the recorder.");
      return;
    }
    recorderRef.current = rec;
    rec.ondataavailable = (ev) => {
      if (ev.data && ev.data.size > 0) chunksRef.current.push(ev.data);
    };
    rec.onstop = () => {
      const blob = new Blob(chunksRef.current, { type: choice.mime });
      if (blob.size === 0) {
        finish(null);
        return;
      }
      const ts = new Date().toISOString().slice(0, 19).replace(/[:T]/g, "-");
      finish(
        new File([blob], `mic-recording-${ts}.${choice.ext}`, {
          type: choice.mime,
        }),
      );
    };
    rec.start(1000); // 1 s chunks — a crash loses at most ~1 s
    startedAtRef.current = performance.now();
    setElapsedMs(0);
    setPhase("recording");
    tickRef.current = setInterval(() => {
      const ms = performance.now() - startedAtRef.current;
      setElapsedMs(ms);
      if (ms >= MAX_MS) {
        // auto-stop at the 60 s cap
        try {
          if (recorderRef.current?.state === "recording") {
            recorderRef.current.stop();
          }
        } catch {
          /* noop */
        }
        clearTick();
      }
    }, 100);
  }, [finish, clearTick]);

  const stopRecording = useCallback(() => {
    clearTick();
    const rec = recorderRef.current;
    if (rec && rec.state !== "inactive") {
      try {
        rec.stop(); // onstop → finish(file)
      } catch {
        finish(null);
      }
    } else {
      finish(null);
    }
  }, [clearTick, finish]);

  // Escape cancels (cleanup runs in finish/unmount).
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") finish(null);
    };
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, [finish]);

  const secs = Math.min(60, Math.floor(elapsedMs / 1000));
  const pct = Math.min(100, (elapsedMs / MAX_MS) * 100);

  return createPortal(
    <div
      className="almost-ready-backdrop"
      onClick={() => finish(null)}
      role="presentation"
    >
      <div
        className="almost-ready-modal mic-rec-modal"
        role="dialog"
        aria-modal="true"
        aria-labelledby="mic-rec-title"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="config-modal-accent" aria-hidden="true" />
        <div className="almost-ready-header">
          <h2 id="mic-rec-title" className="almost-ready-title">
            Record audio
          </h2>
          <button
            type="button"
            className="config-modal-close"
            onClick={() => finish(null)}
            aria-label="Cancel recording"
          >
            ×
          </button>
        </div>

        <div className="almost-ready-body mic-rec-body">
          {phase === "init" && (
            <p className="mic-rec-hint">Requesting microphone…</p>
          )}

          {phase === "error" && (
            <>
              <p className="mic-rec-error">{error}</p>
              <button
                type="button"
                className="mic-rec-btn"
                onClick={() => finish(null)}
              >
                Close
              </button>
            </>
          )}

          {(phase === "ready" || phase === "recording") && (
            <>
              <p className="mic-rec-hint">
                {phase === "recording"
                  ? "Recording… playback is paused so it won't bleed in."
                  : "Playback is paused. Record up to 60 seconds — this becomes your track."}
              </p>

              <div
                className={`mic-rec-timer${
                  phase === "recording" ? " mic-rec-timer--live" : ""
                }`}
              >
                <span className="mic-rec-dot" aria-hidden="true" />
                {String(secs).padStart(2, "0")}
                <span className="mic-rec-timer-max"> / 60s</span>
              </div>

              <div className="mic-rec-progress" aria-hidden="true">
                <div
                  className="mic-rec-progress-fill"
                  style={{ width: `${pct}%` }}
                />
              </div>

              {phase === "ready" ? (
                <button
                  type="button"
                  className="mic-rec-btn mic-rec-btn--rec"
                  onClick={startRecording}
                >
                  ● Start recording
                </button>
              ) : (
                <button
                  type="button"
                  className="mic-rec-btn mic-rec-btn--stop"
                  onClick={stopRecording}
                >
                  ■ Stop &amp; use
                </button>
              )}
            </>
          )}
        </div>
      </div>
    </div>,
    document.body,
  );
}
