"use client";

import { useEffect } from "react";

import { RecordingCompositor } from "@/engine/recording/RecordingCompositor";
import { useRecordingStore } from "@/store/useRecordingStore";
import { useSessionStore } from "@/store/useSessionStore";

// Owns the MediaRecorder lifecycle. UI components dispatch
// document.dispatchEvent(new CustomEvent("dd:toggle-record")) — keyboard
// shortcut handler does the same. Mirrors the dd:toggle-drawer pattern.
//
// Strategy notes:
//   • Two parallel recorders run for every capture: an audio-only one
//     (existing behavior — feeds the WAV download path) and an optional
//     muxed video+audio one whose video track comes from the graph
//     canvas via `canvas.captureStream(30)`. The save dialog lets the
//     user pick which blob downloads; the audio-only path stays
//     bit-identical to the previous behavior when video isn't wanted.
//   • Audio stream comes from AudioPlayer.getRecordingStream() — a
//     MediaStreamAudioDestinationNode tee'd off the same worklet output the
//     user hears, at 48 kHz / 2 ch.
//   • Audio MIME ladder picks Opus-in-WebM first (transparent at 192 kbps),
//     falls back to AAC-in-MP4 for Safari/iOS, gracefully disables otherwise.
//   • Video MIME ladder prefers webm/vp9 then webm/vp8 then mp4. If none
//     are supported (older Safari) the video path silently no-ops and
//     the save dialog presents the audio-only path as the only choice.
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

const VIDEO_MIME_LADDER: { mime: string; ext: string }[] = [
  { mime: "video/webm;codecs=vp9,opus", ext: "webm" },
  { mime: "video/webm;codecs=vp8,opus", ext: "webm" },
  { mime: "video/webm", ext: "webm" },
  { mime: "video/mp4;codecs=avc1,mp4a", ext: "mp4" },
  { mime: "video/mp4", ext: "mp4" },
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

function pickVideoMime(): { mime: string; ext: string } | null {
  if (typeof window === "undefined") return null;
  if (typeof MediaRecorder === "undefined") return null;
  for (const choice of VIDEO_MIME_LADDER) {
    try {
      if (MediaRecorder.isTypeSupported(choice.mime)) return choice;
    } catch {}
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
    let videoRecorder: MediaRecorder | null = null;
    let videoChunks: Blob[] = [];
    let pickedVideoMime: { mime: string; ext: string } | null = null;
    let videoStream: MediaStream | null = null;
    let compositor: RecordingCompositor | null = null;
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
      if (videoRecorder) {
        try {
          videoRecorder.ondataavailable = null;
          videoRecorder.onstop = null;
          videoRecorder.onerror = null;
        } catch {}
      }
      // Release the captureStream tracks so the compositor canvas
      // doesn't keep feeding a recorder we've forgotten about. The
      // video track ends on the next animation frame after the last
      // reference drops. Compositor's own rAF gets cancelled too.
      if (videoStream) {
        try {
          for (const t of videoStream.getTracks()) t.stop();
        } catch {}
      }
      if (compositor) {
        try {
          compositor.stop();
        } catch {}
      }
      recorder = null;
      chunks = [];
      pickedMime = null;
      videoRecorder = null;
      videoChunks = [];
      pickedVideoMime = null;
      videoStream = null;
      compositor = null;
      startedAt = 0;
      pausedTotalMs = 0;
      pausedSinceMs = 0;
      visibilityPaused = false;
    }

    function finalize() {
      if (!recorder || !pickedMime) return;
      const { mime, ext } = pickedMime;

      useRecordingStore.getState().set({ kind: "finalizing" });

      // Two recorders may need to finalize: audio (always present) and
      // video (only if the canvas + MIME were available at start). Both
      // must hand back their onstop before we publish the preview —
      // otherwise the video chip would render with a half-filled blob.
      let audioStopped = false;
      let videoStopped = videoRecorder === null;

      const tryPublish = () => {
        if (!audioStopped || !videoStopped) return;
        const audioBlob = new Blob(chunks, { type: mime });
        const url = URL.createObjectURL(audioBlob);
        const now = performance.now();
        const elapsed = now - startedAt - pausedTotalMs;

        let videoBlob: Blob | undefined;
        let videoUrl: string | undefined;
        let videoMime: string | undefined;
        let videoExt: string | undefined;
        if (pickedVideoMime && videoChunks.length > 0) {
          videoBlob = new Blob(videoChunks, { type: pickedVideoMime.mime });
          videoUrl = URL.createObjectURL(videoBlob);
          videoMime = pickedVideoMime.mime;
          videoExt = pickedVideoMime.ext;
        }

        useRecordingStore.getState().set({
          kind: "preview",
          blob: audioBlob,
          url,
          mime,
          ext,
          durationMs: Math.max(0, elapsed),
          videoBlob,
          videoUrl,
          videoMime,
          videoExt,
        });
        const seconds = Math.max(1, Math.round(elapsed / 1000));
        setMessage(`Saved ${seconds}s clip`);
        teardown();
      };

      const onAudioStop = () => {
        audioStopped = true;
        tryPublish();
      };
      const onVideoStop = () => {
        videoStopped = true;
        tryPublish();
      };

      try {
        if (recorder.state !== "inactive") {
          recorder.onstop = onAudioStop;
          recorder.stop();
        } else {
          onAudioStop();
        }
        if (videoRecorder) {
          if (videoRecorder.state !== "inactive") {
            videoRecorder.onstop = onVideoStop;
            videoRecorder.stop();
          } else {
            onVideoStop();
          }
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

      // Optional parallel video recorder — best-effort. Composites the
      // waveform strip + the motion graph into one offscreen canvas
      // and captureStream's that, so the recording matches the
      // performance the user is making. If anything in the chain fails
      // (MIME, captureStream, MediaRecorder ctor) we silently fall
      // through to audio-only (existing behavior).
      pickedVideoMime = null;
      videoRecorder = null;
      videoStream = null;
      videoChunks = [];
      compositor = null;
      try {
        const videoMime = pickVideoMime();
        if (videoMime) {
          compositor = new RecordingCompositor();
          videoStream = compositor.start();
          const combined = new MediaStream([
            ...videoStream.getVideoTracks(),
            ...stream.getAudioTracks(),
          ]);
          videoRecorder = new MediaRecorder(combined, { mimeType: videoMime.mime });
          pickedVideoMime = videoMime;
          videoRecorder.ondataavailable = (e: BlobEvent) => {
            if (!e.data || e.data.size === 0) return;
            videoChunks.push(e.data);
          };
          videoRecorder.onerror = (e: Event) => {
            // Video is best-effort — tear it down but keep the audio
            // recorder running so the user still gets a clip.
            console.warn("[useRecording] video recorder error", e);
            try {
              videoRecorder?.stop();
            } catch {}
            try {
              compositor?.stop();
            } catch {}
            videoRecorder = null;
            pickedVideoMime = null;
            if (videoStream) {
              try {
                for (const t of videoStream.getTracks()) t.stop();
              } catch {}
              videoStream = null;
            }
            compositor = null;
            videoChunks = [];
          };
          videoRecorder.start(1000);
        }
      } catch (err) {
        console.warn("[useRecording] video capture init failed", err);
        try {
          compositor?.stop();
        } catch {}
        videoRecorder = null;
        pickedVideoMime = null;
        if (videoStream) {
          try {
            for (const t of videoStream.getTracks()) t.stop();
          } catch {}
          videoStream = null;
        }
        compositor = null;
      }

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
        // Mirror onto the video recorder if it exists — keeps both
        // tracks aligned so the muxed file doesn't drift.
        try {
          videoRecorder?.pause();
        } catch {}
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
        try {
          videoRecorder?.resume();
        } catch {}
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
        if (cur.videoUrl) {
          try {
            URL.revokeObjectURL(cur.videoUrl);
          } catch {}
        }
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
      if (videoRecorder && videoRecorder.state !== "inactive") {
        try {
          videoRecorder.stop();
        } catch {}
      }
      teardown();
    };
  }, []);
}
