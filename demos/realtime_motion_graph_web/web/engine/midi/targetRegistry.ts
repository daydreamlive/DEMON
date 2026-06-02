import { LORA_SLOT_MARKER, type CcMode, type NoteAction } from "./types";

export type MidiTargetKind = "cc" | "note" | "enum";

export interface MidiTargetRow {
  kind: MidiTargetKind;
  target: string;
  label: string;
}

export const MIDI_CC_ROWS: MidiTargetRow[] = [
  { kind: "cc", target: "denoise", label: "Remix strength" },
  { kind: "cc", target: "hint_strength", label: "Structure strength" },
  { kind: "cc", target: "feedback", label: "Feedback" },
  { kind: "cc", target: "shift", label: "Shift" },
  { kind: "cc", target: LORA_SLOT_MARKER[0], label: "LoRA slot 1 strength" },
  { kind: "cc", target: LORA_SLOT_MARKER[1], label: "LoRA slot 2 strength" },
  { kind: "cc", target: "stem_vocals", label: "Stem: vocals" },
  { kind: "cc", target: "stem_instruments", label: "Stem: instruments" },
];

export const MIDI_ENUM_ROWS: MidiTargetRow[] = [
  { kind: "enum", target: "dcw_mode", label: "DCW mode" },
  { kind: "enum", target: "dcw_wavelet", label: "DCW wavelet" },
  { kind: "enum", target: "rcfg_mode", label: "RCFG mode" },
  { kind: "enum", target: "pipeline_depth", label: "Pipeline depth" },
  { kind: "enum", target: "key", label: "Musical key" },
  { kind: "enum", target: "time_signature", label: "Time signature" },
];

export const MIDI_NOTE_ACTIONS: { target: NoteAction; label: string }[] = [
  { target: "seed", label: "Randomize seed" },
  { target: "send_prompt", label: "Send prompt" },
  { target: "pause", label: "Pause / resume" },
  { target: "seek_start", label: "Seek to start" },
  { target: "loop", label: "Toggle loop" },
  { target: "mode_toggle", label: "Toggle display mode" },
  { target: "kiosk_toggle", label: "Toggle kiosk" },
  { target: "schedule_curves_toggle", label: "Toggle curve editor" },
  { target: "drawer_toggle", label: "Toggle Full Controls" },
  { target: "drawer_expand_toggle", label: "Toggle expanded Full Controls" },
  { target: "dcw_enabled", label: "Toggle DCW" },
  { target: "smooth", label: "Toggle smoothing" },
  { target: "lufs", label: "Toggle loudness match" },
  { target: "stem_vocals_toggle", label: "Toggle vocals stem" },
  { target: "stem_instruments_toggle", label: "Toggle instruments stem" },
];

export const MIDI_NOTE_ROWS: MidiTargetRow[] = MIDI_NOTE_ACTIONS.map((n) => ({
  kind: "note",
  target: n.target,
  label: n.label,
}));

export const MIDI_TARGET_ROWS: MidiTargetRow[] = [
  ...MIDI_CC_ROWS,
  ...MIDI_ENUM_ROWS,
  ...MIDI_NOTE_ROWS,
];

export const MIDI_MODE_OPTIONS: { value: CcMode; label: string }[] = [
  { value: "relative", label: "Turn wheel: clicks nudge value" },
  { value: "centered", label: "Joystick/pitch: center returns to default" },
  { value: "unipolar", label: "Mod wheel: bottom returns to default" },
  { value: "absolute", label: "Knob/fader: position sets value" },
];

const LABELS = new Map(MIDI_TARGET_ROWS.map((r) => [r.target, r.label]));

export function midiTargetLabel(target: string): string {
  const known = LABELS.get(target);
  if (known) return known;
  if (target.startsWith("lora_str_")) {
    return `LoRA: ${target.slice("lora_str_".length)}`;
  }
  if (target === "prompt_blend") return "Prompt A/B blend";
  if (target === "stem_vocals") return "Vocals stem volume";
  if (target === "stem_instruments") return "Instruments stem volume";
  return target;
}
