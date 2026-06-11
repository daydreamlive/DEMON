import { SAMPLE_RATE } from "@demon/client";
import { usePerformanceStore } from "@/store/usePerformanceStore";
import { useSessionStore } from "@/store/useSessionStore";

// Flips the paused flag in the performance store and mirrors that into the
// AudioContext (suspend/resume). Used by the OperatorStrip button, the
// spacebar shortcut, and the click-to-pause graph overlay — keeping all
// three in lockstep so the UI flag and the audio context can't drift.
export function togglePauseAndAudio(): void {
  const perf = usePerformanceStore.getState();
  perf.togglePause();
  const player = useSessionStore.getState().player;
  if (!player?.ctx) return;
  if (player.ctx.state === "running") {
    void player.ctx.suspend();
    return;
  }
  // Un-pausing. If loop is off and the playhead is parked at end-of-
  // buffer, the worklet would immediately re-fire endOfBuffer and the
  // listener would auto-pause us right back. Seek to 0 first so the
  // user gets a real restart from one click.
  if (
    !perf.loopOn &&
    player.frameCount > 0 &&
    player.positionSec >= (player.frameCount - 1) / SAMPLE_RATE
  ) {
    player.seek(0);
  }
  void player.ctx.resume();
}
