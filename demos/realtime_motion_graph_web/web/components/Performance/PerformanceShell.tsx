"use client";

import { useEffect, useRef } from "react";

import { CustomCursor } from "@/components/CustomCursor";
import { useBodyAttributes } from "@/hooks/useBodyAttributes";
import { useCursor } from "@/hooks/useCursor";
import { useEdgeLoraBinding } from "@/hooks/useEdgeLoraBinding";
import { useFixtureSwap } from "@/hooks/useFixtureSwap";
import { useIdleReset } from "@/hooks/useIdleReset";
import { useInterpSync } from "@/hooks/useInterpSync";
import { useIsMobile } from "@/hooks/useIsMobile";
import { useKeyboardShortcuts } from "@/hooks/useKeyboardShortcuts";
import { useMcpMirror } from "@/hooks/useMcpMirror";
import { useLoraTriggerSync } from "@/hooks/useLoraTriggerSync";
import { useMidi } from "@/hooks/useMidi";
import { useParamSync } from "@/hooks/useParamSync";
import { usePromptBlendSync } from "@/hooks/usePromptBlendSync";
import { useRecording } from "@/hooks/useRecording";
import { useRenderLoop } from "@/hooks/useRenderLoop";
import { useScheduledCurves } from "@/hooks/useScheduledCurves";
import { useRefSourceAcks } from "@/hooks/useRefSourceAcks";
import { useStartSession } from "@/hooks/useStartSession";
import { useStemOverlaySync } from "@/hooks/useStemOverlaySync";
import { useTimbreSync } from "@/hooks/useTimbreSync";
import { useVideoLayer } from "@/hooks/useVideoLayer";
import { useConfig } from "@/lib/config";
import { useCurveStore } from "@/store/useCurveStore";
import { usePerformanceStore } from "@/store/usePerformanceStore";
import { useSessionStore } from "@/store/useSessionStore";

import { AdvancedDrawer } from "./AdvancedDrawer";
import { AudioSourceCrate } from "./AudioSourceCrate";
import { ConfigModal } from "./ConfigModal";
import { ConfirmDialog } from "./ConfirmDialog";
import { DemonBrandMark } from "./DemonBrandMark";
import { FooterLinks } from "./FooterLinks";
import { HeroMacros } from "./HeroMacros";
import { HeroMacrosTooltip } from "./HeroMacrosTooltip";
import { HUDFrame } from "./HUDFrame";
import { HudHelpReadout } from "./HudHelpReadout";
import { InstallStage } from "./InstallStage";
import { LiveIndicator } from "./LiveIndicator";
import {
  MobileLoraBlendStepper,
  MobileRemixStepper,
} from "./MobileStepperRail";
import { NetworkIndicator } from "./NetworkIndicator";
import { PortraitLockOverlay } from "./PortraitLockOverlay";
import { RecordingPreview } from "./RecordingPreview";
import { StartOverlay } from "./StartOverlay";
import { StatusBar } from "./StatusBar";
import { WaveformScrubBox } from "./WaveformScrubBox";

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
  useTimbreSync();
  usePromptBlendSync();
  useInterpSync();
  useLoraTriggerSync();
  useStemOverlaySync();
  useRefSourceAcks("timbre");
  useRefSourceAcks("structure");
  useMcpMirror();
  const config = useConfig();
  useIdleReset(config.reset_seconds);

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
    <>
    <div id="performance" className="screen">
      {status === "ready" && <AudioSourceCrate />}
      {/* Permanent 3-knob row above the drawer handle — performance
          palette (DENOISE / STRUCTURE / FEEDBACK / SEED). The component
          handles its own visibility internally; mount it unconditionally
          and let it decide. Stem overlay faders are mounted inline
          inside HeroMacros (next to the LoRA strength faders) when the
          active track has stems. */}
      <HeroMacros />
      {/* Top-right CTA pair (VST waitlist + Feedback), both Tally
          in-page modals. Self-positioning chrome — no host wiring. */}
      <FooterLinks />

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
      <DemonBrandMark />
      {isMobile && (
        <>
          <MobileRemixStepper />
          <MobileLoraBlendStepper />
        </>
      )}
      <AdvancedDrawer />
      <HudHelpReadout />
      <HeroMacrosTooltip />
      <ConfigModal />
      <ConfirmDialog />

      <StatusBar />

      <LiveIndicator />

      <WaveformScrubBox />

      <NetworkIndicator />

      <RecordingPreview />

      {/* Phone-only portrait gate. CSS-driven; renders behind the rest of
          the UI and only paints in (max-width: 768px) AND portrait. */}
      <PortraitLockOverlay />
    </div>
    {/* CustomCursor sits OUTSIDE #performance on purpose. #performance
        has `isolation: isolate`, so any descendant's z-index is scoped
        to that stacking context — modals are portaled to document.body
        and render in the root stacking context above all of #performance.
        Mounting the cursor canvas at the root puts its z-index (111) in
        the same stacking context as those modals so it draws on top of
        them. */}
    <CustomCursor />
    </>
  );
}
