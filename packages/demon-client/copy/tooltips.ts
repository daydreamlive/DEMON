// Shared tooltip copy for the controls that exist in EVERY client — the
// three remix macros (strength / structure / timbre) and the Trained
// Style knobs (per-style strength + A/B blend). Each tooltip is a 1–2
// second read: when to reach for the knob and what musical outcome it
// produces, not the diffusion-process plumbing underneath.
//
// Client-specific controls (web's activation-steering, latent-channel,
// and DCW sliders; the VST's sequencer) keep their tooltips in their own
// codebase — only the cross-client controls live here. Render however
// the host does it: the web demo feeds these into its `data-dd-tooltip`
// system, the VST into plain `title=` attributes, the Radio into its
// info-bar.

export const TOOLTIPS = {
  // ── Remix macros ──
  strength:
    "How much the model reshapes the source audio. Keep it low for a subtle remix that stays close to the original; push it high to fully transform the track into something new. The most expressive knob — try sweeping it during playback.",
  structure:
    "How closely the model follows the original song's structure — sections, rhythm, dynamics. Crank it up to keep the arrangement intact; drop it to let the model rearrange more freely.",
  timbre:
    "How much of the source's instrument character (tone, color) carries into the output. High keeps the original instruments recognizable; low frees the model to swap them for whatever fits the prompt.",

  // ── Trained Styles ──
  trainedStyleStrength:
    "How strongly this Trained Style shapes the output. Trained Styles are little style packs — set a low value for a subtle flavor, crank past 1.0 to make this style dominate the sound. Multiple Trained Styles stack — turn several on at once for combined styles.",
  trainedStyleBlend:
    "Crossfade between Trained Style A and Trained Style B. 0 = A only, 1 = B only, 0.5 = both at half strength. Use this to morph between two styles smoothly.",
} as const;

// Mapping from engine param id → shared tooltip, for the cross-client
// controls. Per-style strength sliders are dynamic (`lora_str_<id>`) so
// they're matched by prefix rather than listed. Returns undefined for
// params a given client handles with its own copy.
const SHARED_PARAM_TOOLTIPS: Record<string, string> = {
  denoise: TOOLTIPS.strength,
  hint_strength: TOOLTIPS.structure,
  timbre_strength: TOOLTIPS.timbre,
  lora_blend: TOOLTIPS.trainedStyleBlend,
};

/** Shared tooltip for an engine param id, or undefined if not cross-client. */
export function sharedTooltipFor(param: string): string | undefined {
  if (param.startsWith("lora_str_")) return TOOLTIPS.trainedStyleStrength;
  return SHARED_PARAM_TOOLTIPS[param];
}
