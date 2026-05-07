"use client";

import { useEffect, useRef } from "react";

import { CustomCursor } from "@/components/CustomCursor";
import { useBodyAttributes } from "@/hooks/useBodyAttributes";
import { useCursor } from "@/hooks/useCursor";
import { useEdgeLoraBinding } from "@/hooks/useEdgeLoraBinding";
import { useFixtureSwap } from "@/hooks/useFixtureSwap";
import { useIdleReset } from "@/hooks/useIdleReset";
import { useIsMobile } from "@/hooks/useIsMobile";
import { useKeyboardShortcuts } from "@/hooks/useKeyboardShortcuts";
import { useMidi } from "@/hooks/useMidi";
import { useParamSync } from "@/hooks/useParamSync";
import { useRecording } from "@/hooks/useRecording";
import { useRenderLoop } from "@/hooks/useRenderLoop";
import { useScheduledCurves } from "@/hooks/useScheduledCurves";
import { useStartSession } from "@/hooks/useStartSession";
import { useVideoLayer } from "@/hooks/useVideoLayer";
import { useCurveStore } from "@/store/useCurveStore";
import { usePerformanceStore } from "@/store/usePerformanceStore";
import { useSessionStore } from "@/store/useSessionStore";

import { AdvancedDrawer } from "./AdvancedDrawer";
import { AudioSourceCrate } from "./AudioSourceCrate";
import { ConfigModal } from "./ConfigModal";
import { DesktopEdgeDrag } from "./DesktopEdgeDrag";
import { HUDFrame } from "./HUDFrame";
import { InstallStage } from "./InstallStage";
import { MobileRemixRail } from "./MobileRemixRail";
import { RecordButton } from "./RecordButton";
import { RecordingPreview } from "./RecordingPreview";
import { StartOverlay } from "./StartOverlay";
import { StatusBar } from "./StatusBar";

// Demo shell — wires the package's hooks + components into a working app.
//
// Stripped vs. the daydream webapp's Performance.tsx:
//   - no useAuth / useQueue / useCredits (no account system in DEMON)
//   - no QueueScene / PaywallSignedIn (no queue, no payments)
//   - no HaloBadge (brand badge with sign-in/out menu)
//
// What remains is the engine UI itself: stage, HUD, graph, drawer, MIDI,
// recording, custom cursor.

export function PerformanceShell() {
  useBodyAttributes();
  useParamSync();
  useScheduledCurves();
  useEffect(() => {
    usePerformanceStore.getState().hydratePersistedPrefs();
    useCurveStore.getState().hydratePersistedCurves();
  }, []);
  useCursor();
  useMidi();
  useKeyboardShortcuts();
  useRecording();
  useFixtureSwap();
  useEdgeLoraBinding();
  useIdleReset(0);

  const startSession = useStartSession();
  const status = useSessionStore((s) => s.status);
  const isMobile = useIsMobile();

  const ambientRef = useRef<HTMLVideoElement | null>(null);
  const videoARef = useRef<HTMLVideoElement | null>(null);
  const videoBRef = useRef<HTMLVideoElement | null>(null);
  const effectsCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const hudCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const graphCanvasRef = useRef<HTMLCanvasElement | null>(null);

  useRenderLoop({
    hudCanvas: hudCanvasRef,
    graphCanvas: graphCanvasRef,
    effectsCanvas: effectsCanvasRef,
    videoA: videoARef,
    videoB: videoBRef,
  });
  useVideoLayer({ videoA: videoARef, videoB: videoBRef });

  const started = status !== "idle";

  return (
    <div id="performance" className="screen">
      {status === "ready" && <AudioSourceCrate />}
      <RecordButton />

      <StartOverlay
        onPlay={() => {
          void startSession();
        }}
        hidden={started}
      />

      <InstallStage
        refs={{
          ambient: ambientRef,
          videoA: videoARef,
          videoB: videoBRef,
          effectsCanvas: effectsCanvasRef,
          hudCanvas: hudCanvasRef,
          graphCanvas: graphCanvasRef,
        }}
      />
      <HUDFrame />
      {isMobile && <MobileRemixRail />}
      {!isMobile && (
        <>
          <DesktopEdgeDrag side="top" />
          <DesktopEdgeDrag side="left" />
          <DesktopEdgeDrag side="right" />
        </>
      )}

      <AdvancedDrawer />
      <ConfigModal />

      <StatusBar />

      <RecordingPreview />

      <CustomCursor />
    </div>
  );
}
