"use client";

import { SliderGroup } from "./SliderGroup";

// Generic mixer tile that wraps a row of sliders. Replaces the dynamic
// buildChannelTile() helper from app.js. `params` is the list of slider
// names (must exist in SLIDER_META).

interface Props {
  label: string;
  params: { param: string; label: string; max?: number }[];
}

// Display names — anchored in traditional audio vocabulary (synth /
// multi-FX / EQ heritage) so the labels read instantly to anyone who's
// touched a hardware unit or plugin. Where the underlying concept has
// no clean analog (shift, noise_share), the technical label stays.
// CSS uppercases these on render via .slider-label / .mixer-tile-label.
const DISPLAY_NAMES: Record<string, string> = {
  // Core (CORE tab) — the dial-it-and-go knobs.
  denoise: "denoise",        // engine-honest; the most-touched knob keeps its native name
  hint_strength: "structure",// "how much we follow the source's structure"
  timbre_strength: "timbre", // universal in synthesis
  feedback: "feedback",      // universal in delay/echo/mod — keep canonical
  dcw_scaler: "bass",        // universal EQ
  dcw_high_scaler: "treble",

  // Mod (MOD tab) — time-variant / model-internal expert knobs.
  ode_noise: "jitter",       // clocking / granular — stochastic variation
  noise_share: "n.share",    // no clean analog; keep technical
  shift: "shift",            // no clean analog; expert

  // Channels (CHANNELS tab) — the model's internal latent channels.
  ch_g0: "v1", ch_g1: "v2", ch_g2: "v3", ch_g3: "v4",
  ch_g4: "v5", ch_g5: "v6", ch_g6: "v7", ch_g7: "v8",
  ch13: "m1", ch14: "m2", ch19: "m3", ch23: "m4", ch29: "m5", ch56: "m6",
};

// Tooltip copy for each tweakable param, surfaced via the slider label's
// hover tooltip in SliderGroup. Aim: a 1–2 second read that tells the
// user WHEN to reach for this knob — what musical outcome it produces,
// not the diffusion-process plumbing underneath. Renders via
// data-dd-tooltip-wide (white-space: normal, max-width 280px).
const PARAM_TOOLTIPS: Record<string, string> = {
  // ── Main remix controls ──
  denoise:
    "How much the model reshapes the source audio. Keep it low for a subtle remix that stays close to the original; push it high to fully transform the track into something new. The most expressive knob — try sweeping it during playback.",
  hint_strength:
    "How closely the model follows the original song's structure — sections, rhythm, dynamics. Crank it up to keep the arrangement intact; drop it to let the model rearrange more freely.",
  timbre_strength:
    "How much of the source's instrument character (tone, color) carries into the output. High keeps the original instruments recognizable; low frees the model to swap them for whatever fits the prompt.",

  // ── Engine internals ──
  feedback:
    "How similar each new generation is to the previous one. Low values give you variety on every refresh; higher values give you a continuous evolution where each generation flows into the next. 0.3–0.5 is the sweet spot for smooth continuity without everything sounding the same.",
  shift:
    "Advanced: changes where the model concentrates its work across denoising. The default is tuned for the turbo engine and works well in most cases — leave it alone unless you're chasing a specific feel.",
  noise_share:
    "Fine-grained sibling of FEEDBACK. Where FEEDBACK is one number for the whole generation, this lets the noise-sharing vary across the timeline. Leave at 0 unless you want detailed control over how successive generations evolve.",
  ode_noise:
    "Adds a touch of randomness during generation. Bump it up if the model feels too deterministic — small values add subtle variation, higher values produce surprising bursts of creativity. Zero keeps generation fully predictable.",

  // ── DCW ──
  dcw_scaler:
    "Boost or attenuate the model's low end (bass, body). Push positive if the output feels thin; pull negative if the bass is overpowering. The range is small on purpose — these are fine adjustments.",
  dcw_high_scaler:
    "Boost or attenuate the model's high end (transients, brightness, air). Push up for crispness and snap; pull down to round off harsh tops.",
};

// Per-channel tooltips. The 64-channel latent space hasn't been fully
// mapped to perceptual qualities yet, so the copy frames each channel
// as something to discover by ear — not a labeled knob with a known
// purpose. Generated programmatically to avoid 14 near-identical
// hand-written strings.
const CHANNEL_GAINS = ["ch_g0", "ch_g1", "ch_g2", "ch_g3", "ch_g4", "ch_g5", "ch_g6", "ch_g7"] as const;
const NAMED_CHANNELS = ["ch13", "ch14", "ch19", "ch23", "ch29", "ch56"] as const;
for (const [i, p] of CHANNEL_GAINS.entries()) {
  PARAM_TOOLTIPS[p] =
    `Experimental — adjusts the strength of one of the model's internal audio channels (channel ${i}). Each channel encodes a different aspect of the sound (frequency band, dynamics, transients); the exact mapping is still being explored. Sweep it to discover what it does for your source.`;
}
for (const p of NAMED_CHANNELS) {
  const idx = p.slice(2);
  PARAM_TOOLTIPS[p] =
    `Experimental — a hand-picked internal audio channel (#${idx}) that produces a noticeable perceptual change. Sweep it to hear what this specific channel controls for your source.`;
}

export function tooltipFor(param: string): string | undefined {
  // LoRA strength sliders (param like `lora_str_<id>`) get a generic
  // tooltip rather than per-LoRA copy — the row already shows the
  // LoRA's name as its visible label.
  if (param.startsWith("lora_str_")) {
    return "How strongly this LoRA shapes the output. LoRAs are little style packs — set a low value for a subtle flavor, crank past 1.0 to make this LoRA dominate the sound. Multiple LoRAs stack — turn several on at once for combined styles.";
  }
  if (param === "lora_blend") {
    return "Crossfade between LoRA A and LoRA B. 0 = A only, 1 = B only, 0.5 = both at half strength. Use this to morph between two styles smoothly.";
  }
  return PARAM_TOOLTIPS[param];
}

// Map slider param → keyboard hint shown beneath the slider. Mirrors the
// chord layout in hooks/useKeyboardShortcuts.ts; if you change one, change
// the other.
const KBD_FOR_PARAM: Record<string, string> = {
  denoise: "A + ▲▼",
  hint_strength: "G + ▲▼",
  timbre_strength: "C + ▲▼",
  feedback: "E + ▲▼",
  shift: "H + ▲▼",
  noise_share: "N + ▲▼",
  ode_noise: "D + ▲▼",
  ch_g0: "0 + ▲▼",
  ch_g1: "1 + ▲▼",
  ch_g2: "2 + ▲▼",
  ch_g3: "3 + ▲▼",
  ch_g4: "4 + ▲▼",
  ch_g5: "5 + ▲▼",
  ch_g6: "6 + ▲▼",
  ch_g7: "7 + ▲▼",
  ch13: "⇧1 + ▲▼",
  ch14: "⇧2 + ▲▼",
  ch19: "⇧3 + ▲▼",
  ch23: "⇧4 + ▲▼",
  ch29: "⇧5 + ▲▼",
  ch56: "⇧6 + ▲▼",
  dcw_scaler: "W + ▲▼",
  dcw_high_scaler: "Y + ▲▼",
};

export function kbdHintFor(param: string): string | undefined {
  return KBD_FOR_PARAM[param];
}

export function SliderTile({ label, params }: Props) {
  return (
    <div className="mixer-tile" data-tile={label.toLowerCase().replace(/ /g, "-")}>
      <div className="mixer-tile-label">{label}</div>
      <div className="mixer-channels">
        {params.map(({ param, label: pLabel, max }) => (
          <SliderGroup
            key={param}
            param={param}
            label={pLabel}
            max={max}
            kbd={KBD_FOR_PARAM[param]}
          />
        ))}
      </div>
    </div>
  );
}

export function defaultLabelFor(param: string): string {
  return DISPLAY_NAMES[param] ?? param.replace(/_/g, " ");
}
