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
  | "schedule_curves_toggle";

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
}

export const DEFAULT_MIDI_MAP: MidiMap = {
  cc: {
    "70": "denoise",
    "71": LORA_SLOT_MARKER[0],
    "72": LORA_SLOT_MARKER[1],
    "73": "hint_strength",
    "74": "feedback",
    "75": "shift",
    "76": "noise_share",
    "77": "ode_noise",
  },
  notes: {
    "36": "seed",
    "37": "send_prompt",
  },
  ccActions: {},
};

export const MIDI_STORAGE_KEY = "dd_music_midi_map_v1";
