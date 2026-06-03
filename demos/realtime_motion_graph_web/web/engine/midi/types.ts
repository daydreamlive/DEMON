// Static names for actions a Note button can trigger. Storing the action by
// name (instead of a function reference) lets localStorage round-trip a
// learned mapping cleanly.

export const LORA_SLOT_MARKER = ["lora_slot_0", "lora_slot_1"] as const;

export type LoraSlotMarker = (typeof LORA_SLOT_MARKER)[number];

export type NoteAction =
  | "seed"
  | "send_prompt"
  | "mode_toggle"
  | "pause"
  | "kiosk_toggle"
  | "schedule_curves_toggle"
  // Toggle / transport actions exposed in both drawers. Shape-identical
  // to the existing one-shot actions — a note-on flips the bound state.
  | "dcw_enabled"
  | "smooth"
  | "lufs"
  | "loop"
  | "seek_start"
  | "drawer_toggle"
  | "drawer_expand_toggle"
  | "stem_vocals_toggle"
  | "stem_instruments_toggle";

/** Per-CC controller type, chosen at MIDI-learn time. Governs how a raw
 *  CC value (0..127) is turned into a parameter change:
 *   - `absolute`  knobs/faders that hold position. The physical position
 *                 IS the value: CC 0 → bottom of rail, 127 → top, straight
 *                 through the control's own mapping. (Default for any CC
 *                 with no explicit type.)
 *   - `relative`  endless encoders. One 2's-complement tick moves the param
 *                 the SAME amount as one mouse-scroll-wheel notch (T-space
 *                 step), stepping from the param's current value.
 *   - `centered`  centre-resting springs (joystick, pitch). CC 64 (rest) →
 *                 default, 0 → one rail, 127 → the other. Each side is an
 *                 independent linear ramp from the default. Springs back to
 *                 the control's default on release.
 *   - `unipolar`  bottom-resting springs (mod / pitch wheel). CC 0 (rest)
 *                 → default, CC 127 → the farther rail end. Springs back to
 *                 default on release. */
export type CcMode = "absolute" | "relative" | "centered" | "unipolar";

export interface MidiMap {
  /** CC number → param name (or LORA_SLOT_MARKER). Continuous controllers
   *  driving slider params. */
  cc: Record<string, string>;
  /** Note number → action name. Pads / buttons that fire one-shot actions. */
  notes: Record<string, NoteAction>;
  /** CC number → action name. Some pad controllers send CC instead of
   *  NOTE on press; this lets those CCs trigger discrete actions like
   *  randomize-seed without forcing the user to reconfigure their hardware
   *  mode. Optional for backward compatibility with old localStorage maps. */
  ccActions?: Record<string, NoteAction>;
  /** CC number → binding mode. Absent entry defaults to `absolute`, so
   *  maps persisted before this field existed keep their old behaviour. */
  ccMode?: Record<string, CcMode>;
  /** CC number → enum target id (see engine/midi/enumTargets.ts). An
   *  absolute knob sweep quantizes across the target's options. Kept
   *  separate from `cc` because the value is an enum id, not a numeric
   *  slider param fed to setSlider. */
  ccEnum?: Record<string, string>;
  /** Note number → enum target id. A pad press cycles to the next option
   *  (wraps). Separate from `notes` because the value is an enum id, not
   *  a NoteAction. */
  noteEnum?: Record<string, string>;
}

export const DEFAULT_MIDI_MAP: MidiMap = {
  cc: {
    "70": "denoise",
    "71": LORA_SLOT_MARKER[0],
    "72": LORA_SLOT_MARKER[1],
    "73": "hint_strength",
    "74": "feedback",
    "75": "shift",
  },
  notes: {
    "36": "seed",
    "37": "send_prompt",
  },
  ccActions: {},
};

export const MIDI_STORAGE_KEY = "dd_music_midi_map_v1";
