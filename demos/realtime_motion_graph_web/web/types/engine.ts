// Slider definitions and engine-side knobs. Mirrors the SLIDER_META and
// CONFIG.controls shape from DEMON's static/app.js.

export interface SliderMeta {
  /** Maximum value (slider top). */
  max: number;
  /** Arrow-key step. */
  step: number;
  /** Hide from the simple Main tile; live in pro/advanced areas. */
  pro?: boolean;
}

/** Standard non-LoRA sliders. LoRA sliders (`lora_str_<id>`) get added at
 * runtime when the catalog arrives — they aren't listed here. */
export const SLIDER_META: Record<string, SliderMeta> = {
  denoise: { max: 1.0, step: 0.1 },
  hint_strength: { max: 2.0, step: 0.2 },

  feedback: { max: 1.0, step: 0.1, pro: true },
  shift: { max: 1.0, step: 0.1, pro: true },
  noise_share: { max: 1.0, step: 0.1, pro: true },
  ode_noise: { max: 0.5, step: 0.05, pro: true },

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

  // DCW (wavelet-domain post-step correction). Numeric knobs only; the
  // boolean ON/OFF + mode + wavelet choices live in their own panel state.
  dcw_scaler: { max: 0.2, step: 0.02, pro: true },
  dcw_high_scaler: { max: 0.1, step: 0.01, pro: true },
};

export const DCW_MODES = ["low", "high", "double", "pix"] as const;
export const DCW_WAVELETS = ["haar", "db4", "sym8", "db8"] as const;
export type DcwMode = (typeof DCW_MODES)[number];
export type DcwWavelet = (typeof DCW_WAVELETS)[number];

export const LORA_SLIDER_MAX = 2.0;
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
