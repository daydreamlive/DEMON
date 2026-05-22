"use client";

import { useEffect, type RefObject } from "react";

import { tickActiveCursor } from "@/engine/cursor";
import { EffectsRenderer } from "@/engine/render/EffectsRenderer";
import { frameScheduler } from "@/engine/scheduler/FrameScheduler";
import { GraphRenderer } from "@/engine/render/GraphRenderer";
import { HUD } from "@/engine/render/HUD";
import { getConfig, subscribeConfig } from "@/lib/config";
import {
  destroyHaloRibbon,
  destroyRibbons,
  initHaloRibbon,
  initRibbons,
  initStartMarkRibbon,
  tickHaloRibbon,
  tickRibbons,
  tickStartMarkRibbon,
  type HaloRibbon,
  type RibbonBar,
  type StartMarkRibbon,
} from "@/engine/render/ribbons";
import {
  destroyTurntable,
  initTurntable,
  tickTurntable,
  type Turntable,
} from "@/engine/render/turntable";
import { useLoraStore } from "@/store/useLoraStore";
import { usePerformanceStore } from "@/store/usePerformanceStore";
import { useSessionStore } from "@/store/useSessionStore";

// Single top-level render loop. Owns HUD + GraphRenderer + EffectsRenderer.
// RAF-driven; pauses when document is hidden (Phase 8 perf goal #8). Other
// renderers (Ambient, Ribbons) hook into the same loop in later phases.

interface Refs {
  hudCanvas: RefObject<HTMLCanvasElement | null>;
  graphCanvas: RefObject<HTMLCanvasElement | null>;
  effectsCanvas: RefObject<HTMLCanvasElement | null>;
  videoA: RefObject<HTMLVideoElement | null>;
  videoB: RefObject<HTMLVideoElement | null>;
}

export function useRenderLoop(refs: Refs) {
  useEffect(() => {
    const hudEl = refs.hudCanvas.current;
    const graphEl = refs.graphCanvas.current;
    const effectsEl = refs.effectsCanvas.current;
    if (!hudEl || !graphEl) return;

    let cancelled = false;

    const hud = new HUD(hudEl);
    const graph = new GraphRenderer(graphEl);

    // Initialise ribbons (one-time DOM mutation against the .install-edge-*
    // divs rendered by <HUDFrame />). Bars come back as { edge, ribbons[] }.
    const ribbons: RibbonBar[] = initRibbons();

    // Halo badge ribbons — the SVG itself is rendered by <HaloBadge />;
    // we just grab a handle to its 4 paths so we can tick them in the
    // same RAF as the perimeter ribbons.
    const haloHost = document.querySelector(".halo-badge") as HTMLElement | null;
    const halo: HaloRibbon | null = haloHost ? initHaloRibbon(haloHost) : null;

    // Now-playing dock collapse bubble — wears the same writhing ribbon
    // ring as the halo badge. AudioSourceCrate mounts/unmounts with
    // session state, so (like the turntable below) re-query each frame
    // and cache once it appears.
    let bubbleHalo: HaloRibbon | null = null;
    let lastBubbleHost: Element | null = null;

    // Turntable grooves — concentric polar paths driven by the live audio
    // mirror buffer (the music is literally etched into the disc surface).
    // The button mounts/unmounts with session state so re-query each frame
    // and cache once it appears.
    let turntable: Turntable | null = null;
    let lastTurntableHost: Element | null = null;

    // Start-mark ribbons — wreathe the title-screen logo with the same
    // ribbon language. The .start-cta doubles as the play button (icon +
    // whisper share one click target); the render loop ticks the ribbons
    // regardless so the writhe is already in motion the moment the
    // overlay is shown.
    const startMarkHost = document.querySelector(".start-cta") as HTMLElement | null;
    const startMark: StartMarkRibbon | null = startMarkHost
      ? initStartMarkRibbon(startMarkHost)
      : null;

    let effects: EffectsRenderer | null = null;
    let unsubEffectsConfig: (() => void) | null = null;
    if (effectsEl) {
      try {
        const e = new EffectsRenderer(effectsEl);
        effects = e;
        e.setDubstep(0);
        e.setDaftPunk(0);
        // Effects values come from web/public/config.json. Re-apply on
        // every applyConfig() so an async-arriving config or future
        // "Reload config" affordance lands without a page refresh.
        const applyFx = (c: ReturnType<typeof getConfig>) => {
          e.setParallaxStrength(c.effects.parallax_strength);
          e.setBloomOnKick(c.effects.bloom_on_kick);
          e.setBloomThreshold(c.effects.bloom_threshold);
          e.setWarpStrength(c.effects.warp_strength);
        };
        applyFx(getConfig());
        unsubEffectsConfig = subscribeConfig(applyFx);
      } catch (e) {
        // WebGL2 unavailable → fall back to the raw <video> the canvas
        // overlays. App still works without bloom/parallax.
        console.warn("[EffectsRenderer] init failed:", e);
        effects = null;
      }
    }

    let mirrorUnsub: (() => void) | null = null;
    const sessionUnsub = useSessionStore.subscribe((s) => {
      mirrorUnsub?.();
      mirrorUnsub = null;
      if (s.player) {
        const refreshHud = () => {
          const m = s.player!.getMirror();
          if (m) hud.updateWaveform(m, s.player!.channels);
        };
        refreshHud();
        mirrorUnsub = s.player.onMirrorChange(refreshHud);
      }
    });
    {
      const player = useSessionStore.getState().player;
      if (player) {
        const m = player.getMirror();
        if (m) hud.updateWaveform(m, player.channels);
      }
    }

    let lastSampleAt = 0;
    const SAMPLE_INTERVAL_MS = 50;
    const startedAt = performance.now();
    // Last value written to --bloom-amount. Skipping no-op writes prevents
    // the rest of the document (badge / perimeter / button drop-shadows
    // that read --bloom-amount) from re-rastering on frames where the
    // binned kick hasn't actually changed.
    let lastBloom = -1;
    // Bloom host: target the performance scene root so style recalc is
    // scoped to the subtree that actually consumes --bloom-amount, not
    // the whole document. Falls back to <html> if the element isn't
    // mounted yet (e.g. very first frames before the scene exists).
    // Cached per-frame: re-resolve when null so a delayed mount picks up,
    // but stop walking the DOM once we have it.
    let bloomHost: HTMLElement | null = null;

    function tick(now: number) {
      if (cancelled) return;
      const session = useSessionStore.getState();
      const perf = usePerformanceStore.getState();

      let frac = 0;
      let kick = 0;
      if (session.player) {
        const dur = session.player.duration;
        const pos = session.player.positionSec;
        frac = dur > 0 ? Math.max(0, Math.min(1, pos / dur)) : 0;
        // Kick is computed in the AudioWorklet on the audio thread and
        // posted with each position update (~21 ms cadence). Reading
        // it here is a single field load — no per-frame audio buffer
        // walk on the main thread. The ScriptProcessor fallback path
        // leaves player.kick = 0 (graceful degradation; non-secure
        // contexts only). See PERFORMANCE.md.
        kick = session.player.kick;
      }
      // CI / devtools synthetic-kick override: set window.__PERF_STRESS_KICK__
      // to a number in [0, 1] to drive the visuals as if a kick of that
      // strength were happening continuously. Used by tests/perf/smoke.mjs
      // to measure chorus-frame cost without needing real audio playback.
      if (
        typeof window !== "undefined" &&
        typeof (window as { __PERF_STRESS_KICK__?: number }).__PERF_STRESS_KICK__ === "number"
      ) {
        kick = (window as { __PERF_STRESS_KICK__?: number }).__PERF_STRESS_KICK__ ?? kick;
      }

      // Bloom CSS var, rounded to 0.1 bins so the document doesn't
      // re-paint every drop-shadow consumer on every frame. Skip the
      // write entirely when the binned value hasn't moved — most frames
      // are no-ops once audio settles. We also pass the binned value
      // into the canvas tickers (turntable / ribbons / halo / cursor)
      // so they don't have to call getComputedStyle every frame, which
      // would force a style recalc and jank the page.
      const bloom = Math.round(kick * 10) / 10;
      if (typeof document !== "undefined") {
        if (bloom !== lastBloom) {
          if (!bloomHost) {
            bloomHost =
              (document.getElementById("performance") as HTMLElement | null) ??
              document.documentElement;
          }
          bloomHost.style.setProperty("--bloom-amount", bloom.toString());
          lastBloom = bloom;
        }
      }

      // While paused: AudioContext is suspended, positionSec freezes, and
      // the playhead marker stops. Skip sampling so the parameter history
      // also stops advancing — otherwise polylines keep drifting left
      // behind a frozen playhead. Leaving lastSampleAt untouched means the
      // next tick after resume samples immediately, no warm-up gap.
      if (now - lastSampleAt >= SAMPLE_INTERVAL_MS && !perf.paused) {
        lastSampleAt = now;
        const lora = useLoraStore.getState();
        const sample: Record<string, number> = { ...perf.sliderValues };
        // Seed is a uint32 integer that snaps on randomize / manual edit
        // — graphing it would just spike to noise. Omitted intentionally.
        for (const id of lora.enabled) {
          const v = lora.strengths[id];
          if (typeof v === "number") sample[`lora_str_${id}`] = v;
        }
        graph.sample(sample);
      }

      hud.draw(frac, { transparentBg: false });
      graph.draw(kick, now);

      // Drive ribbons. The top-edge bar's --fill is the denoise slider; the
      // L/R bars track the first/second active LoRA. For Phase 8 just feed
      // the denoise value to the top bar so the ribbons animate
      // immediately; LoRA-driven --fill on the side bars lands in Phase 11.
      const denoiseFrac = (perf.sliderValues.denoise ?? 0) / 1.0;
      for (const bar of ribbons) {
        if (bar.edge.classList.contains("install-edge-top")) {
          bar.edge.style.setProperty("--fill", denoiseFrac.toString());
        }
      }
      const ribbonTime = (now - startedAt) / 1000;
      tickRibbons(ribbons, ribbonTime, kick, bloom);
      if (halo) tickHaloRibbon(halo, ribbonTime, kick, bloom);
      const bubbleHost = document.querySelector(".audio-source-bubble");
      if (bubbleHost !== lastBubbleHost) {
        if (bubbleHalo) destroyHaloRibbon(bubbleHalo);
        bubbleHalo = bubbleHost
          ? initHaloRibbon(bubbleHost as HTMLElement)
          : null;
        lastBubbleHost = bubbleHost;
      }
      if (bubbleHalo) tickHaloRibbon(bubbleHalo, ribbonTime, kick, bloom);
      if (startMark) tickStartMarkRibbon(startMark, ribbonTime);

      // Turntable: the grooves trace the actual audio waveform from the
      // worklet's mirror buffer. Each frame we walk a slowly-advancing
      // window over the buffer so the wave appears to "scroll" around the
      // disc, mimicking a record being played.
      const turntableHost = document.querySelector(".turntable");
      if (turntableHost !== lastTurntableHost) {
        if (turntable) destroyTurntable(turntable);
        turntable = turntableHost
          ? initTurntable(turntableHost as HTMLElement)
          : null;
        lastTurntableHost = turntableHost;
      }
      if (turntable) {
        const player = useSessionStore.getState().player;
        const mirror = player?.getMirror() ?? null;
        const channels = player?.channels ?? 2;
        tickTurntable(turntable, mirror, channels, ribbonTime, bloom);
      }

      if (effects) {
        // Pull whichever <video> is currently the displayed one. VideoLayer
        // swaps which element holds the active stream; reading both and
        // picking the one with non-zero opacity is more robust than
        // tracking who's "active" externally.
        const videoEl = pickActiveVideo(refs.videoA.current, refs.videoB.current);
        const elapsed = (now - startedAt) / 1000;
        effects.tick(videoEl, elapsed, kick);
      }

      // Cursor — driven from this loop instead of its own RAF (saves a
      // vsync wakeup and keeps draws in one batch per frame). Bloom is
      // the same binned kick value the ribbons receive, so the cursor's
      // glow halo pulses in lockstep with the perimeter ribbons without
      // either side needing to read a CSS variable.
      tickActiveCursor(now, bloom);
    }

    // Single-rAF policy: master render loop registers with FrameScheduler.
    // Compute-phase callbacks (tweens, scheduled curves) run before this
    // tick on the same frame, so the store state we read is up to date.
    // When document.hidden, skip the work entirely — Chrome throttles
    // rAF to 1 Hz on hidden tabs but we want zero work, not slow work.
    const unregister = frameScheduler.register(
      "render-loop",
      (now) => {
        if (cancelled) return;
        if (typeof document !== "undefined" && document.hidden) return;
        tick(now);
      },
      { phase: "render", budgetMs: 10 },
    );

    return () => {
      cancelled = true;
      unregister();
      sessionUnsub();
      mirrorUnsub?.();
      unsubEffectsConfig?.();
      hud.destroy();
      graph.destroy();
      effects?.destroy();
      if (turntable) destroyTurntable(turntable);
      destroyRibbons(ribbons);
      if (halo) destroyHaloRibbon(halo);
      if (bubbleHalo) destroyHaloRibbon(bubbleHalo);
    };
  }, [refs.hudCanvas, refs.graphCanvas, refs.effectsCanvas, refs.videoA, refs.videoB]);
}

function pickActiveVideo(
  a: HTMLVideoElement | null,
  b: HTMLVideoElement | null,
): HTMLVideoElement | null {
  if (a && a.style.opacity !== "0" && a.readyState >= 2) return a;
  if (b && b.style.opacity !== "0" && b.readyState >= 2) return b;
  return a ?? b;
}
