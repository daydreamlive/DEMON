export const LEGO_TRACKS = [
  "woodwinds",
  "brass",
  "fx",
  "synth",
  "strings",
  "percussion",
  "keyboard",
  "guitar",
  "bass",
  "drums",
  "backing_vocals",
  "vocals",
] as const;

export type LegoTrack = (typeof LEGO_TRACKS)[number];

export interface LegoLayerConfig {
  track: LegoTrack;
  prompt: string;
}

export const DEFAULT_LEGO_PROMPTS: Record<LegoTrack, string> = {
  woodwinds: "expressive woodwind orchestral layer",
  brass: "cinematic brass stabs and swells",
  fx: "musical transition effects and risers",
  synth: "wide analog synth layer",
  strings: "violin orchestra hans zimmer",
  percussion: "cinematic percussion accents",
  keyboard: "funky electric keyboard groove",
  guitar: "tight electric rhythm guitar layer",
  bass: "deep bass groove locked to the song",
  drums: "punchy drum kit groove",
  backing_vocals: "soft backing vocal harmonies",
  vocals: "lead vocal melodic phrase",
};

export function labelForLegoTrack(track: string): string {
  return track
    .split("_")
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(" ");
}
