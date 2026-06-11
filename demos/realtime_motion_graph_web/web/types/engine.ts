// Slider definitions and engine-side knobs.

export interface SliderMeta {
  /** Maximum value (slider top). */
  max: number;
  /** Minimum value (slider bottom). Defaults to 0 when omitted. Used
   *  for sliders whose engine value has a non-zero floor (e.g.
   *  feedback_depth, where 0 is meaningless — the smallest valid tap is
   *  1, "blend with the most recent finished latent"). */
  min?: number;
  /** Arrow-key step. */
  step: number;
  /** Hide from the simple Main tile; live in pro/advanced areas. */
  pro?: boolean;
}

/** Standard non-LoRA sliders. LoRA sliders (`lora_str_<id>`) get added at
 * runtime when the catalog arrives — they aren't listed here. */
export const SLIDER_META: Record<string, SliderMeta> = {
  denoise: { max: 1.0, step: 0.05 },
  hint_strength: { max: 1.0, step: 0.05 },
  timbre_strength: { max: 1.0, step: 0.05 },
  // 0 = LoRA A only, 1 = LoRA B only, 0.5 = both at half-max. UI-only knob —
  // useEdgeLoraBinding watches this and writes the paired lora_str_<id> values.
  lora_blend: { max: 1.0, step: 0.05 },
  // 0 = Tags A only, 1 = Tags B only. Lives in the slider system so the
  // Smooth toggle / tween machinery / graph / MIDI generic path all work
  // uniformly. Stripped out of the engine `params` dict by useParamSync
  // (engine doesn't read it from there); shipped via the dedicated
  // ``set_prompt_blend`` WS message by usePromptBlendSync.
  prompt_blend: { max: 1.0, step: 0.05 },

  feedback: { max: 1.0, step: 0.05, pro: true },
  // Delay-tap depth for `feedback`. 1 == blend with the most recent
  // finished latent (current behavior). N>1 reaches N ticks back, so
  // feedback can produce echo / ghost effects without needing to crank
  // the blend coefficient near 1.0 (which is the only way to reach
  // distant past via the implicit recursion).
  feedback_depth: { min: 1, max: 8, step: 1, pro: true },
  // ACE-Step flow shift. Directly the value the diffusion solver receives;
  // 3.0 is the upstream turbo default, 3.5 our packaged default. Useful
  // operator range is roughly [1, 6].
  shift: { min: 1.0, max: 6.0, step: 0.5, pro: true },
  // RCFG guidance scale. Only takes effect when rcfg_mode != "off". The
  // turbo model is CFG-distilled (trained to operate at scale=1 with
  // conditioning baked in); driving guidance past ~10 on turbo tends
  // to artifact. 7.0 is the SD-style default, useful starting point.
  guidance_scale: { max: 15.0, step: 0.5, pro: true },
  // Per-frame mix toward vt_pos's magnitude after APG. 0 = raw APG;
  // 1 = fully snap norm to vt_pos. Useful at high guidance_scale to
  // tame saturation.
  cfg_rescale: { max: 1.0, step: 0.05, pro: true },

  // Diffusion step count for the StreamPipeline. Default 8 (turbo
  // packaged default). Lower = faster but rougher; higher = smoother
  // but more latency. Changing this rebuilds the pipeline (steps is in
  // the rebuild-on-change set on StreamDenoise), so each move glitches
  // the audio for one ring's worth of ticks.
  steps_override: { min: 1, max: 16, step: 1, pro: true },

  ch_g0: { max: 3.0, step: 0.15, pro: true },
  ch_g1: { max: 3.0, step: 0.15, pro: true },
  ch_g2: { max: 3.0, step: 0.15, pro: true },
  ch_g3: { max: 3.0, step: 0.15, pro: true },
  ch_g4: { max: 3.0, step: 0.15, pro: true },
  ch_g5: { max: 3.0, step: 0.15, pro: true },
  ch_g6: { max: 3.0, step: 0.15, pro: true },
  ch_g7: { max: 3.0, step: 0.15, pro: true },

  ch13: { max: 3.0, step: 0.15, pro: true },
  ch14: { max: 3.0, step: 0.15, pro: true },
  ch19: { max: 3.0, step: 0.15, pro: true },
  ch23: { max: 3.0, step: 0.15, pro: true },
  ch29: { max: 3.0, step: 0.15, pro: true },
  ch56: { max: 3.0, step: 0.15, pro: true },

  // Activation-steering. ±30 range so the operator can invert the axis
  // without leaving the surface. Useful magnitude ~5..15 by ear.
  steer_bright:   { min: -30.0, max: 30.0, step: 0.5, pro: true },
  steer_warm:     { min: -30.0, max: 30.0, step: 0.5, pro: true },
  steer_rough:    { min: -30.0, max: 30.0, step: 0.5, pro: true },
  steer_density:  { min: -30.0, max: 30.0, step: 0.5, pro: true },

  // Manual steering slots (man_*_<N>) are pre-registered in the loop
  // below so a runtime slot add doesn't have to mutate SLIDER_META.

  // DCW (wavelet-domain post-step correction). Numeric knobs only; the
  // boolean ON/OFF + mode + wavelet choices live in their own panel state.
  //
  // Caps for dcw_scaler / dcw_high_scaler are bumped past the upstream
  // "usable" range (~0.1) so the operator can drive DCW into audible
  // artifact territory while A/B'ing the three advanced faders. The
  // defaults (0.05 / 0.02) still match upstream-v0.1.7.
  dcw_scaler: { max: 0.5, step: 0.02, pro: true },
  dcw_high_scaler: { max: 0.5, step: 0.02, pro: true },
  // Advanced surface — composes on top of the additive update. Field
  // names mirror DCWAdvanced in acestep/engine/dcw.py one-to-one, so
  // sliderValues spreads straight into the server params dict with
  // no remap layer in useParamSync.
  dcw_mult_blend: { max: 1.0, step: 0.05, pro: true },
  dcw_mag_phase: { max: 1.0, step: 0.05, pro: true },
  dcw_soft_thresh: { max: 0.3, step: 0.01, pro: true },
};

// SLIDER_META pre-reg ceiling. Authoritative cap is the server's
// ``manual_slot_cap``; raising the server cap above this means bumping
// this number too.
const MANUAL_SLOT_PREREG = 16;
for (let slot = 1; slot <= MANUAL_SLOT_PREREG; slot++) {
  SLIDER_META[`man_src_${slot}`]   = { min: 0, max: 143, step: 1, pro: true };
  SLIDER_META[`man_layer_${slot}`] = { min: 0, max: 23,  step: 1, pro: true };
  SLIDER_META[`man_step_${slot}`]  = { min: 0, max: 15,  step: 1, pro: true };
  SLIDER_META[`man_alpha_${slot}`] = {
    min: -30.0, max: 30.0, step: 0.5, pro: true,
  };
}

export const DCW_MODES = ["low", "high", "double", "pix"] as const;
export const DCW_WAVELETS = ["haar", "db4", "sym8", "db8"] as const;
export type DcwMode = (typeof DCW_MODES)[number];
export type DcwWavelet = (typeof DCW_WAVELETS)[number];
export function isDcwMode(v: unknown): v is DcwMode {
  return typeof v === "string" && (DCW_MODES as readonly string[]).includes(v);
}
export function isDcwWavelet(v: unknown): v is DcwWavelet {
  return (
    typeof v === "string" && (DCW_WAVELETS as readonly string[]).includes(v)
  );
}

// RCFG (Residual Classifier-Free Guidance) modes. "off" disables APG
// entirely on the wire (turbo default — no guidance, no extra forwards).
// "initialize" runs the uncond pass only at step 0 per slot, caches the
// velocity, reuses it for the slot's remaining steps. "self" skips the
// uncond forward entirely; virtual ``v_uncond ≈ initial_noise``
// (flow-matching identity with ``x0_uncond ≈ 0``). See
// acestep/engine/stream.py. The engine also supports "full" (standard
// two-pass CFG, 2x cost), but it's intentionally NOT in the demo
// dropdown — turbo is CFG-distilled and an externally-driven full CFG
// against an empty-prompt uncond doesn't produce the right perceptual
// direction. Test scripts can still set ``rcfg_mode="full"`` directly.
export const RCFG_MODES = ["off", "initialize", "self"] as const;
export type RcfgMode = (typeof RCFG_MODES)[number];
export function isRcfgMode(v: unknown): v is RcfgMode {
  return typeof v === "string" && (RCFG_MODES as readonly string[]).includes(v);
}

// Capped at 1.8 (was 2.0). Operator finding: most LoRAs we ship turn
// to noise above ~1.7 (e.g. v5/discofunk noise at 2.0, hardrock noise
// at 2.0). 1.8 still leaves room above the natural sweet spot for
// every shipped LoRA without giving users a slider position that
// reliably destroys the output.
export const LORA_SLIDER_MAX = 1.8;
export const LORA_SLIDER_STEP = 0.2;

/** Default LoRA strength as a fraction of LORA_SLIDER_MAX. Used in two
 * places that must agree: the catalog seeder (when the server doesn't
 * ship a per-LoRA strength) and the side-bar empty-state visual (no
 * LoRA bound yet). Both reading the same constant keeps the ribbon's
 * canvas fill, the hint's head position, and ARIA aria-valuenow in
 * lock-step. */
export const LORA_DEFAULT_STRENGTH_FRACTION = 0.7;

/** Visibility floor for the side-bar (LoRA) ribbon. Shared by:
 *   - the canvas in engine/render/ribbons.ts (so the writhe never
 *     fully disappears at strength=0)
 *   - the RemixHint head position in DesktopEdgeDrag.tsx (so the
 *     hint sits at the visible head, not below it)
 * Without sharing the same floor, drags below this threshold leave
 * the hint floating below the ribbon's visible end. */
export const LORA_SIDE_VISIBLE_FLOOR = 0.25;

/** Visibility floor for the top (Remix Strength) ribbon. Same role
 * as LORA_SIDE_VISIBLE_FLOOR but for the horizontal bar. Without it,
 * dragging denoise to 0 collapses the ribbon to nothing and the user
 * has no visual cue that the slider still exists along the top edge.
 * The drag overlay stays mounted at full width either way — the floor
 * only affects rendering, not the stored value (engine still receives
 * 0). Smaller than the side-bar floor because the top bar spans most
 * of the viewport: 4% is still ~50-80px of grabbable ribbon. */
export const REMIX_VISIBLE_FLOOR = 0.04;

export type DisplayMode = "graph" | "video";

/** Full keyscale set the model accepts (mirrors VALID_KEYSCALES in app.js). */
const NOTES = ["A", "B", "C", "D", "E", "F", "G"] as const;
const ACCIDENTALS = ["", "#", "b", "♯", "♭"] as const;
const MODES = ["major", "minor"] as const;

export const VALID_KEYSCALES: string[] = (() => {
  const out: string[] = [];
  for (const note of NOTES) {
    for (const acc of ACCIDENTALS) {
      for (const mode of MODES) {
        out.push(`${note}${acc} ${mode}`);
      }
    }
  }
  return out;
})();

/** Time signatures the model accepts. Mirrors
 *  ``acestep.constants.VALID_TIME_SIGNATURES`` (``[2, 3, 4, 6]``). The
 *  encoder takes the value as a string in
 *  ``Session.encode_text(time_signature=...)``, where it gets baked into
 *  the prompt as ``- timesignature: <value>``. We carry the strings
 *  end-to-end so there's no int/string round-tripping at the wire. */
export const VALID_TIME_SIGNATURES = ["2", "3", "4", "6"] as const;
export type TimeSignature = (typeof VALID_TIME_SIGNATURES)[number];

/** Display labels for the dropdowns. Numerators map to the conventional
 *  meter notation (4 → 4/4, 6 → 6/8). The wire value stays the bare
 *  numerator string the encoder expects. */
export const TIME_SIGNATURE_LABELS: Record<TimeSignature, string> = {
  "2": "2/4",
  "3": "3/4",
  "4": "4/4",
  "6": "6/8",
};

export const DEFAULT_TIME_SIGNATURE: TimeSignature = "4";

export function isTimeSignature(v: unknown): v is TimeSignature {
  return typeof v === "string"
    && (VALID_TIME_SIGNATURES as readonly string[]).includes(v);
}
