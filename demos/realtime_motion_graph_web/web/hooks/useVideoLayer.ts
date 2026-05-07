"use client";

import { useEffect, useRef, type RefObject } from "react";

import { listVideos } from "@/engine/video/listVideos";
import { VideoLayer } from "@/engine/video/VideoLayer";
import { useSessionStore } from "@/store/useSessionStore";

interface Refs {
  videoA: RefObject<HTMLVideoElement | null>;
  videoB: RefObject<HTMLVideoElement | null>;
}

// Mounts VideoLayer once both <video> refs are populated. Wires the audio
// position feed from the live AudioPlayer; queues + plays the first
// available video. Phase 11 / future: per-prompt video selection.

export function useVideoLayer(refs: Refs) {
  const layerRef = useRef<VideoLayer | null>(null);

  // Effect 1 — VideoLayer lifecycle. Stays bound to the refs only, not
  // session state, so we don't recreate the layer on every queue tick.
  useEffect(() => {
    const a = refs.videoA.current;
    const b = refs.videoB.current;
    if (!a || !b) return;

    const layer = new VideoLayer({
      videoA: a,
      videoB: b,
      bpm: 134, // Server overrides via setBpm() once detected.
      crossfadeDuration: 1.5,
      useMarkers: false, // Drift-correction fallback by default.
    });
    layerRef.current = layer;

    // Hook into the live audio player whenever the session changes.
    const sessionUnsub = useSessionStore.subscribe((s) => {
      if (s.player) {
        layer.setAudioSource(() => s.player?.positionSec ?? 0, true);
      } else {
        layer.setAudioSource(() => 0, false);
      }
    });
    {
      const initial = useSessionStore.getState().player;
      if (initial) {
        layer.setAudioSource(() => initial.positionSec, true);
      }
    }

    return () => {
      sessionUnsub();
      layer.destroy();
      layerRef.current = null;
    };
  }, [refs.videoA, refs.videoB]);

  // Effect 2 — fetch the available video list once the queue admits us
  // (sessionWsUrl flips truthy). The pod proxy at /api/pod/* refuses
  // requests pre-admit, so calling listVideos() on mount before the
  // queue completes would 401. Re-runs if wsUrl changes (re-admission).
  const sessionWsUrl = useSessionStore((s) => s.wsUrl);
  useEffect(() => {
    if (!sessionWsUrl) return;
    const layer = layerRef.current;
    if (!layer) return;
    let cancelled = false;
    void listVideos()
      .then((names) => {
        if (cancelled || names.length === 0) return;
        layer.setVideos(names);
        layer.play(names[0]);
      })
      .catch(() => {
        // Pod may not have any videos; silently fall back to no video.
      });
    return () => {
      cancelled = true;
    };
  }, [sessionWsUrl]);
}
