import { usePerformanceStore } from "@/store/usePerformanceStore";
import { useSessionStore } from "@/store/useSessionStore";

// Flips the paused flag in the performance store and mirrors that into the
// AudioContext (suspend/resume). Used by the OperatorStrip button, the
// spacebar shortcut, and the click-to-pause graph overlay — keeping all
// three in lockstep so the UI flag and the audio context can't drift.
export function togglePauseAndAudio(): void {
  usePerformanceStore.getState().togglePause();
  const player = useSessionStore.getState().player;
  if (!player?.ctx) return;
  if (player.ctx.state === "running") void player.ctx.suspend();
  else void player.ctx.resume();
}
