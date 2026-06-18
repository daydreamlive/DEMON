// Short user-facing labels derived from the canonical terminology in
// ./terms. Defined once so the web demo, public demo (Radio + vendored
// Performance UI), and the VST webui all render identical wording. Each
// client maps these onto its own widgets — there is no shared component
// layer, only this shared string surface.

import { TRAINED_STYLE, TRAINED_STYLES } from "./terms";

export const LABELS = {
  /** "Trained Styles" / "Trained Style" — re-exported for convenience. */
  plural: TRAINED_STYLES,
  singular: TRAINED_STYLE,

  /** Library panel / accordion header. */
  library: `${TRAINED_STYLES} Library`,
  /** Empty-state copy when no styles are available. */
  noneFound: `no ${TRAINED_STYLES} found`,
  /** Search box placeholder (lowercase — reads as a hint). */
  searchPlaceholder: `search ${TRAINED_STYLES.toLowerCase()}`,
  /** Search box accessible name. */
  searchAria: `Search ${TRAINED_STYLES} library`,

  /** Crossfade control label (slot A ↔ slot B). */
  blend: `${TRAINED_STYLE} Blend`,
  /** Status shown while the engine reloads adapters. */
  refitInProgress: `Applying… (${TRAINED_STYLE} refit in progress)`,
} as const;
