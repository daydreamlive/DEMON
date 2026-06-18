// Canonical user-facing terminology, shared across every DEMON frontend.
//
// The engine, wire protocol, config files, and MIDI map all call these
// style-adapter packs "LoRAs" — that's the honest technical name and it
// stays everywhere internal (param ids like `lora_blend`, config keys
// like `enabled_loras`, the admin training studio). But player-facing UI
// reads better with a plain-language term, so every client renders the
// concept as "Trained Styles". Keep that one decision here so the three
// clients never drift on wording.

/** Player-facing name for the LoRA concept (plural). */
export const TRAINED_STYLES = "Trained Styles";

/** Player-facing name for a single LoRA. */
export const TRAINED_STYLE = "Trained Style";
