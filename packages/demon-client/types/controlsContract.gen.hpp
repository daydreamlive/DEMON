// AUTO-GENERATED — do not edit by hand.
//
// Projected from the TypeScript controls/ copy layer
//   packages/demon-client/controls/{lexicon,copy}.ts
// by packages/demon-client/scripts/genConfigTypes.mjs.
//
// Regenerate after any controls/ change:
//   (cd packages/demon-client && npm run build && npm run gen:config)
// Drift-guarded by web/tests/unit/configContractDrift.test.ts
// (a stale copy fails CI).
//
// The FOURTH contract: user-facing control copy — the labels/tooltips a frontend
// renders ON its knobs/inputs, distinct from config + wire + the terse
// agent-facing /api/knobs descriptions. JS hosts (web demo, M4L generator, VST
// webview) import @demon/client/controls directly; the C++ VST cannot, so its
// DAW-facing APVTS param names key off these generated constants.
//
// terms::    — the product-noun lexicon (TERMS): flip a noun in ONE place.
// display::  — friendly per-param labels (CONTROL_DISPLAY_NAMES); only the
//              SDK's explicit overrides (others fall back to snake->space).
// describe:: — rich tooltip prose (CONTROL_DESCRIPTIONS) + the runtime-family
//              shared strings (LoRA / manual-steering slots).

#pragma once

namespace demon::controls {

namespace terms {

  // Product-facing nouns the team renames over time, keyed by a stable id.
  inline constexpr const char* kTags = "Tags";
  inline constexpr const char* kLora = "LoRA";
  inline constexpr const char* kLoraPlural = "LoRAs";

}  // namespace terms

namespace display {

  inline constexpr const char* kDenoise = "strength";
  inline constexpr const char* kHintStrength = "structure";
  inline constexpr const char* kTimbreStrength = "timbre";
  inline constexpr const char* kDcwScaler = "DCW low";
  inline constexpr const char* kDcwHighScaler = "DCW high";

}  // namespace display

namespace describe {

  // Per-param tooltip prose (CONTROL_DESCRIPTIONS).
  /** How much the model reshapes the source audio. Keep it low for a subtle remix that stays close to the original; push it high to fully transform the track into something new. The most expressive knob — try sweeping it during playback. */
  inline constexpr const char* kDenoise = "How much the model reshapes the source audio. Keep it low for a subtle remix that stays close to the original; push it high to fully transform the track into something new. The most expressive knob — try sweeping it during playback.";
  /** How closely the model follows the original song's structure — sections, rhythm, dynamics. Crank it up to keep the arrangement intact; drop it to let the model rearrange more freely. */
  inline constexpr const char* kHintStrength = "How closely the model follows the original song's structure — sections, rhythm, dynamics. Crank it up to keep the arrangement intact; drop it to let the model rearrange more freely.";
  /** How much of the source's instrument character (tone, color) carries into the output. High keeps the original instruments recognizable; low frees the model to swap them for whatever fits the prompt. */
  inline constexpr const char* kTimbreStrength = "How much of the source's instrument character (tone, color) carries into the output. High keeps the original instruments recognizable; low frees the model to swap them for whatever fits the prompt.";
  /** How similar each new generation is to the previous one. Low values give you variety on every refresh; higher values give you a continuous evolution where each generation flows into the next. 0.3–0.5 is the sweet spot for smooth continuity without everything sounding the same. */
  inline constexpr const char* kFeedback = "How similar each new generation is to the previous one. Low values give you variety on every refresh; higher values give you a continuous evolution where each generation flows into the next. 0.3–0.5 is the sweet spot for smooth continuity without everything sounding the same.";
  /** How far back in time the Feedback knob reaches. 1 (default) blends with the most recent generation. Higher values reach back several ticks for an echo / ghost effect — a faint repeat of an earlier moment surfaces in the current output. Lets you get distant feedback without cranking Feedback all the way up. */
  inline constexpr const char* kFeedbackDepth = "How far back in time the Feedback knob reaches. 1 (default) blends with the most recent generation. Higher values reach back several ticks for an echo / ghost effect — a faint repeat of an earlier moment surfaces in the current output. Lets you get distant feedback without cranking Feedback all the way up.";
  /** Advanced: changes where the model concentrates its work across denoising. The default is tuned for the turbo engine and works well in most cases — leave it alone unless you're chasing a specific feel. */
  inline constexpr const char* kShift = "Advanced: changes where the model concentrates its work across denoising. The default is tuned for the turbo engine and works well in most cases — leave it alone unless you're chasing a specific feel.";
  /** Diffusion step count. Lower steps = lower quality. Higher steps = more latency. Default 8 is the turbo balance. Changing this rebuilds the streaming pipeline, so expect a brief audio glitch when you move it. */
  inline constexpr const char* kStepsOverride = "Diffusion step count. Lower steps = lower quality. Higher steps = more latency. Default 8 is the turbo balance. Changing this rebuilds the streaming pipeline, so expect a brief audio glitch when you move it.";
  /** CFG strength. Only takes effect when the RCFG mode dropdown below is NOT 'off'. Higher values push the output further toward the prompt at the cost of more artifacts. Turbo is CFG-distilled, so the useful range is narrower than a base SD model — try 3–8. */
  inline constexpr const char* kGuidanceScale = "CFG strength. Only takes effect when the RCFG mode dropdown below is NOT 'off'. Higher values push the output further toward the prompt at the cost of more artifacts. Turbo is CFG-distilled, so the useful range is narrower than a base SD model — try 3–8.";
  /** After CFG, mix the guided velocity's magnitude back toward what the positive forward produced. 0 keeps raw CFG; 1 fully snaps the magnitude. Pair with high guidance_scale to keep the prompt-push without the harshness that high CFG causes on its own. */
  inline constexpr const char* kCfgRescale = "After CFG, mix the guided velocity's magnitude back toward what the positive forward produced. 0 keeps raw CFG; 1 fully snaps the magnitude. Pair with high guidance_scale to keep the prompt-push without the harshness that high CFG causes on its own.";
  /** Activation-steering: positive alpha shifts spectral centroid up (brighter, more highs). 0 = off; useful range 5-15 by ear. Recreate as a manual slot: vector brightness_l09_t3 at layer = 9, step = round(3/8 x steps_count). */
  inline constexpr const char* kSteerBright = "Activation-steering: positive alpha shifts spectral centroid up (brighter, more highs). 0 = off; useful range 5-15 by ear. Recreate as a manual slot: vector brightness_l09_t3 at layer = 9, step = round(3/8 x steps_count).";
  /** Activation-steering: positive alpha tilts the spectrum toward bass (warmer). The raw vector points the wrong way for this axis, so this knob folds in a -1 sign. 0 = off; useful range 5-15 by ear. Recreate as a manual slot: vector warmth_l15_t0 at layer = 15, step = 0, then INVERT alpha sign (manual mode is sign-agnostic). */
  inline constexpr const char* kSteerWarm = "Activation-steering: positive alpha tilts the spectrum toward bass (warmer). The raw vector points the wrong way for this axis, so this knob folds in a -1 sign. 0 = off; useful range 5-15 by ear. Recreate as a manual slot: vector warmth_l15_t0 at layer = 15, step = 0, then INVERT alpha sign (manual mode is sign-agnostic).";
  /** Activation-steering: positive alpha increases spectral flatness (grittier, noisier). Vector magnitude at this probe cell is small, so effect builds slowly. 0 = off; useful range 5-15 by ear. Recreate as a manual slot: vector roughness_l09_t3 at layer = 9, step = round(3/8 x steps_count). */
  inline constexpr const char* kSteerRough = "Activation-steering: positive alpha increases spectral flatness (grittier, noisier). Vector magnitude at this probe cell is small, so effect builds slowly. 0 = off; useful range 5-15 by ear. Recreate as a manual slot: vector roughness_l09_t3 at layer = 9, step = round(3/8 x steps_count).";
  /** Activation-steering: positive alpha thins the texture toward sparse/minimal. Inject layer is shifted 3 shallower than the probe layer (Phase-3 transfer finding). 0 = off; useful range 5-15 by ear. Recreate as a manual slot: vector density_l18_t3 at layer = 15 (probe 18 minus 3), step = round(3/8 x steps_count). */
  inline constexpr const char* kSteerDensity = "Activation-steering: positive alpha thins the texture toward sparse/minimal. Inject layer is shifted 3 shallower than the probe layer (Phase-3 transfer finding). 0 = off; useful range 5-15 by ear. Recreate as a manual slot: vector density_l18_t3 at layer = 15 (probe 18 minus 3), step = round(3/8 x steps_count).";
  /** Experimental — adjusts the low-band strength of an internal correction the model applies to itself during generation (DCW). This scaler is active in the early part of the run. The exact audio mapping is still being explored — sweep it to discover what it does for your source. Extreme values can be unpredictable but cool. */
  inline constexpr const char* kDcwScaler = "Experimental — adjusts the low-band strength of an internal correction the model applies to itself during generation (DCW). This scaler is active in the early part of the run. The exact audio mapping is still being explored — sweep it to discover what it does for your source. Extreme values can be unpredictable but cool.";
  /** Experimental — adjusts the high-band strength of an internal correction the model applies to itself during generation (DCW). This scaler is active in the later part of the run. The exact audio mapping is still being explored — sweep it to discover what it does for your source. Extreme values can be unpredictable but cool. */
  inline constexpr const char* kDcwHighScaler = "Experimental — adjusts the high-band strength of an internal correction the model applies to itself during generation (DCW). This scaler is active in the later part of the run. The exact audio mapping is still being explored — sweep it to discover what it does for your source. Extreme values can be unpredictable but cool.";
  /** Experimental — adjusts the strength of one of the model's internal latent channels (channel 0). Each channel encodes a different aspect of the sound (frequency band, dynamics, transients); the exact mapping is still being explored. Sweep it to discover what it does for your source. */
  inline constexpr const char* kChG0 = "Experimental — adjusts the strength of one of the model's internal latent channels (channel 0). Each channel encodes a different aspect of the sound (frequency band, dynamics, transients); the exact mapping is still being explored. Sweep it to discover what it does for your source.";
  /** Experimental — adjusts the strength of one of the model's internal latent channels (channel 1). Each channel encodes a different aspect of the sound (frequency band, dynamics, transients); the exact mapping is still being explored. Sweep it to discover what it does for your source. */
  inline constexpr const char* kChG1 = "Experimental — adjusts the strength of one of the model's internal latent channels (channel 1). Each channel encodes a different aspect of the sound (frequency band, dynamics, transients); the exact mapping is still being explored. Sweep it to discover what it does for your source.";
  /** Experimental — adjusts the strength of one of the model's internal latent channels (channel 2). Each channel encodes a different aspect of the sound (frequency band, dynamics, transients); the exact mapping is still being explored. Sweep it to discover what it does for your source. */
  inline constexpr const char* kChG2 = "Experimental — adjusts the strength of one of the model's internal latent channels (channel 2). Each channel encodes a different aspect of the sound (frequency band, dynamics, transients); the exact mapping is still being explored. Sweep it to discover what it does for your source.";
  /** Experimental — adjusts the strength of one of the model's internal latent channels (channel 3). Each channel encodes a different aspect of the sound (frequency band, dynamics, transients); the exact mapping is still being explored. Sweep it to discover what it does for your source. */
  inline constexpr const char* kChG3 = "Experimental — adjusts the strength of one of the model's internal latent channels (channel 3). Each channel encodes a different aspect of the sound (frequency band, dynamics, transients); the exact mapping is still being explored. Sweep it to discover what it does for your source.";
  /** Experimental — adjusts the strength of one of the model's internal latent channels (channel 4). Each channel encodes a different aspect of the sound (frequency band, dynamics, transients); the exact mapping is still being explored. Sweep it to discover what it does for your source. */
  inline constexpr const char* kChG4 = "Experimental — adjusts the strength of one of the model's internal latent channels (channel 4). Each channel encodes a different aspect of the sound (frequency band, dynamics, transients); the exact mapping is still being explored. Sweep it to discover what it does for your source.";
  /** Experimental — adjusts the strength of one of the model's internal latent channels (channel 5). Each channel encodes a different aspect of the sound (frequency band, dynamics, transients); the exact mapping is still being explored. Sweep it to discover what it does for your source. */
  inline constexpr const char* kChG5 = "Experimental — adjusts the strength of one of the model's internal latent channels (channel 5). Each channel encodes a different aspect of the sound (frequency band, dynamics, transients); the exact mapping is still being explored. Sweep it to discover what it does for your source.";
  /** Experimental — adjusts the strength of one of the model's internal latent channels (channel 6). Each channel encodes a different aspect of the sound (frequency band, dynamics, transients); the exact mapping is still being explored. Sweep it to discover what it does for your source. */
  inline constexpr const char* kChG6 = "Experimental — adjusts the strength of one of the model's internal latent channels (channel 6). Each channel encodes a different aspect of the sound (frequency band, dynamics, transients); the exact mapping is still being explored. Sweep it to discover what it does for your source.";
  /** Experimental — adjusts the strength of one of the model's internal latent channels (channel 7). Each channel encodes a different aspect of the sound (frequency band, dynamics, transients); the exact mapping is still being explored. Sweep it to discover what it does for your source. */
  inline constexpr const char* kChG7 = "Experimental — adjusts the strength of one of the model's internal latent channels (channel 7). Each channel encodes a different aspect of the sound (frequency band, dynamics, transients); the exact mapping is still being explored. Sweep it to discover what it does for your source.";
  /** Experimental — a hand-picked internal latent channel (#13) that produces a noticeable perceptual change. Sweep it to hear what this specific channel controls for your source. */
  inline constexpr const char* kCh13 = "Experimental — a hand-picked internal latent channel (#13) that produces a noticeable perceptual change. Sweep it to hear what this specific channel controls for your source.";
  /** Experimental — a hand-picked internal latent channel (#14) that produces a noticeable perceptual change. Sweep it to hear what this specific channel controls for your source. */
  inline constexpr const char* kCh14 = "Experimental — a hand-picked internal latent channel (#14) that produces a noticeable perceptual change. Sweep it to hear what this specific channel controls for your source.";
  /** Experimental — a hand-picked internal latent channel (#19) that produces a noticeable perceptual change. Sweep it to hear what this specific channel controls for your source. */
  inline constexpr const char* kCh19 = "Experimental — a hand-picked internal latent channel (#19) that produces a noticeable perceptual change. Sweep it to hear what this specific channel controls for your source.";
  /** Experimental — a hand-picked internal latent channel (#23) that produces a noticeable perceptual change. Sweep it to hear what this specific channel controls for your source. */
  inline constexpr const char* kCh23 = "Experimental — a hand-picked internal latent channel (#23) that produces a noticeable perceptual change. Sweep it to hear what this specific channel controls for your source.";
  /** Experimental — a hand-picked internal latent channel (#29) that produces a noticeable perceptual change. Sweep it to hear what this specific channel controls for your source. */
  inline constexpr const char* kCh29 = "Experimental — a hand-picked internal latent channel (#29) that produces a noticeable perceptual change. Sweep it to hear what this specific channel controls for your source.";
  /** Experimental — a hand-picked internal latent channel (#56) that produces a noticeable perceptual change. Sweep it to hear what this specific channel controls for your source. */
  inline constexpr const char* kCh56 = "Experimental — a hand-picked internal latent channel (#56) that produces a noticeable perceptual change. Sweep it to hear what this specific channel controls for your source.";

  // Runtime-family shared copy (LoRA strength / blend, manual slots).
  namespace family {

    /** How strongly this LoRA shapes the output. LoRAs are little style packs — set a low value for a subtle flavor, crank past 1.0 to make this LoRA dominate the sound. Multiple LoRAs stack — turn several on at once for combined styles. */
    inline constexpr const char* kLoraStrength = "How strongly this LoRA shapes the output. LoRAs are little style packs — set a low value for a subtle flavor, crank past 1.0 to make this LoRA dominate the sound. Multiple LoRAs stack — turn several on at once for combined styles.";

    /** Crossfade between LoRA A and LoRA B. 0 = A only, 1 = B only, 0.5 = both at half strength. Use this to morph between two styles smoothly. */
    inline constexpr const char* kLoraBlend = "Crossfade between LoRA A and LoRA B. 0 = A only, 1 = B only, 0.5 = both at half strength. Use this to morph between two styles smoothly.";

    /** Catalog index of the steering vector this slot fires. The catalog enumerates every pre-built (axis, build_layer, build_step) cell on disk in stable axis-major order. Double-click the readout to type an exact index; query the MCP list_manual_steering_vectors tool for the full table. Has no effect until α is non-zero. */
    inline constexpr const char* kManualSrc = "Catalog index of the steering vector this slot fires. The catalog enumerates every pre-built (axis, build_layer, build_step) cell on disk in stable axis-major order. Double-click the readout to type an exact index; query the MCP list_manual_steering_vectors tool for the full table. Has no effect until α is non-zero.";

    /** DiT inject layer (0-23). The vector is added to this layer's post-block residual. Bypasses the auto path's density layer offset — the value lands exactly where you point it. */
    inline constexpr const char* kManualLayer = "DiT inject layer (0-23). The vector is added to this layer's post-block residual. Bypasses the auto path's density layer offset — the value lands exactly where you point it.";

    /** Diffusion inject step (0-15). Bypasses the auto path's fractional step mapping; the engine fires the injection only on the step that matches this value. If you pick a step past the current steps count - 1, the slot stays silent until you raise the step count. */
    inline constexpr const char* kManualStep = "Diffusion inject step (0-15). Bypasses the auto path's fractional step mapping; the engine fires the injection only on the step that matches this value. If you pick a step past the current steps count - 1, the slot stays silent until you raise the step count.";

    /** Strength of this manual slot's injection. 0 disables the slot. Negative α inverts the vector's direction at injection time (no sign correction is applied; what you set is what the engine receives). Sweep range and breakage point mirror the perceptual steering knobs. */
    inline constexpr const char* kManualAlpha = "Strength of this manual slot's injection. 0 disables the slot. Negative α inverts the vector's direction at injection time (no sign correction is applied; what you set is what the engine receives). Sweep range and breakage point mirror the perceptual steering knobs.";

  }  // namespace family

}  // namespace describe

}  // namespace demon::controls
