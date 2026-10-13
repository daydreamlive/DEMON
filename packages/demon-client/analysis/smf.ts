// Minimal Standard MIDI File writer for transcribed notes: format 1, one
// conductor track (tempo + 4/4) and one track per instrument. Drums go on
// channel 10; every note gets a fixed velocity (the wire carries none).

import type { NoteItem } from "./types";

export interface SmfOptions {
  /** Tempo written to the conductor track; notes keep their real time. */
  bpm?: number;
  /** Ticks per quarter note. */
  ppq?: number;
  /** Note-on velocity, 1..127. */
  velocity?: number;
  /** Canvas second that becomes tick 0; notes starting earlier are left
   *  out. */
  originS?: number;
}

/** Instrument name that maps to the GM percussion channel. */
export const DRUM_INSTRUMENT = "drums";
const DRUM_CHANNEL = 9;

/** GM program for each MuScriptor instrument group (the first program of
 *  the group in MuScriptor's MT3_FULL_PLUS vocabulary). `program_<n>`
 *  names map to n; anything else gets no program change. */
const GM_PROGRAM: Readonly<Record<string, number>> = {
  acoustic_piano: 0,
  electric_piano: 2,
  chromatic_percussion: 8,
  organ: 16,
  acoustic_guitar: 24,
  clean_electric_guitar: 26,
  distorted_electric_guitar: 29,
  acoustic_bass: 32,
  electric_bass: 33,
  violin: 40,
  viola: 41,
  cello: 42,
  contrabass: 43,
  orchestral_harp: 46,
  timpani: 47,
  string_ensemble: 48,
  synth_strings: 50,
  voice: 52,
  orchestra_hit: 55,
  trumpet: 56,
  trombone: 57,
  tuba: 58,
  french_horn: 60,
  brass_section: 61,
  soprano_and_alto_sax: 64,
  tenor_sax: 66,
  baritone_sax: 67,
  oboe: 68,
  english_horn: 69,
  bassoon: 70,
  clarinet: 71,
  flutes: 72,
  synth_lead: 80,
  synth_pad: 88,
};

export function gmProgramFor(instrument: string): number | null {
  if (instrument in GM_PROGRAM) return GM_PROGRAM[instrument];
  const m = /^program_(\d+)$/.exec(instrument);
  if (m) {
    const n = Number(m[1]);
    if (n >= 0 && n < 128) return n;
  }
  return null;
}

function vlq(value: number): number[] {
  let v = Math.max(0, Math.floor(value));
  const bytes = [v & 0x7f];
  v >>>= 7;
  while (v > 0) {
    bytes.unshift((v & 0x7f) | 0x80);
    v >>>= 7;
  }
  return bytes;
}

function textMeta(kind: number, text: string): number[] {
  const data = Array.from(new TextEncoder().encode(text));
  return [0xff, kind, ...vlq(data.length), ...data];
}

/** Append a chunk (4-char tag, big-endian length, body) to `out`. Loops
 *  rather than spreading: a long track would overflow the argument limit. */
function appendChunk(out: number[], tag: string, body: number[]): void {
  const n = body.length;
  for (let i = 0; i < 4; i++) out.push(tag.charCodeAt(i));
  out.push((n >>> 24) & 0xff, (n >>> 16) & 0xff, (n >>> 8) & 0xff, n & 0xff);
  for (const b of body) out.push(b);
}

interface TimedEvent {
  tick: number;
  /** Sort rank at equal ticks: note-offs (0) before note-ons (1). */
  rank: number;
  bytes: number[];
}

function trackBody(header: number[], events: TimedEvent[]): number[] {
  events.sort((a, b) => a.tick - b.tick || a.rank - b.rank);
  const body = [...header];
  let last = 0;
  for (const ev of events) {
    body.push(...vlq(ev.tick - last), ...ev.bytes);
    last = ev.tick;
  }
  body.push(0x00, 0xff, 0x2f, 0x00);
  return body;
}

/**
 * Encode `notes` (canvas seconds) as a format-1 SMF.
 *
 * Instruments are written in name order, one track each, on channels
 * 1..16 skipping 10 (wrapping past 15 melodic instruments); `drums` uses
 * channel 10. A note that overlaps the next onset of the same pitch on the
 * same track is cut at that onset so note-on/off pairs never interleave.
 */
export function notesToSmf(
  notes: readonly NoteItem[],
  options: SmfOptions = {},
): Uint8Array {
  const bpm = options.bpm && options.bpm > 0 ? options.bpm : 120;
  const ppq = Math.max(24, Math.round(options.ppq ?? 480));
  const velocity = Math.min(127, Math.max(1, Math.round(options.velocity ?? 100)));
  const origin = options.originS ?? 0;
  const ticksPerSec = (bpm / 60) * ppq;
  const toTick = (s: number) => Math.round((s - origin) * ticksPerSec);

  const byInstrument = new Map<string, NoteItem[]>();
  for (const note of notes) {
    if (note.start_s < origin || note.pitch < 0 || note.pitch > 127) continue;
    const list = byInstrument.get(note.instrument) ?? [];
    list.push(note);
    byInstrument.set(note.instrument, list);
  }
  const instruments = [...byInstrument.keys()].sort();

  const usPerQuarter = Math.round(60_000_000 / bpm);
  const conductor = trackBody(
    [
      0x00, ...textMeta(0x03, "tempo"),
      0x00, 0xff, 0x51, 0x03,
      (usPerQuarter >>> 16) & 0xff, (usPerQuarter >>> 8) & 0xff, usPerQuarter & 0xff,
      0x00, 0xff, 0x58, 0x04, 0x04, 0x02, 0x18, 0x08,
    ],
    [],
  );

  const tracks: number[][] = [conductor];
  let melodic = 0;
  for (const instrument of instruments) {
    let channel: number;
    if (instrument === DRUM_INSTRUMENT) {
      channel = DRUM_CHANNEL;
    } else {
      channel = melodic % 15;
      if (channel >= DRUM_CHANNEL) channel += 1;
      melodic += 1;
    }
    const header = [0x00, ...textMeta(0x03, instrument)];
    const program = gmProgramFor(instrument);
    if (program !== null && channel !== DRUM_CHANNEL) {
      header.push(0x00, 0xc0 | channel, program);
    }

    const sorted = [...(byInstrument.get(instrument) ?? [])].sort(
      (a, b) => a.start_s - b.start_s || a.pitch - b.pitch,
    );
    const nextOnset = new Map<number, number>();
    const events: TimedEvent[] = [];
    for (let i = sorted.length - 1; i >= 0; i--) {
      const note = sorted[i];
      const on = toTick(note.start_s);
      let off = Math.max(on + 1, toTick(note.end_s));
      const next = nextOnset.get(note.pitch);
      // An exact duplicate (same pitch, same onset tick) is written once.
      if (next === on) continue;
      if (next !== undefined && next < off) off = Math.max(on + 1, next);
      nextOnset.set(note.pitch, on);
      events.push(
        { tick: on, rank: 1, bytes: [0x90 | channel, note.pitch, velocity] },
        { tick: off, rank: 0, bytes: [0x80 | channel, note.pitch, 0] },
      );
    }
    tracks.push(trackBody(header, events));
  }

  const out: number[] = [];
  appendChunk(out, "MThd", [
    0x00, 0x01,
    (tracks.length >>> 8) & 0xff, tracks.length & 0xff,
    (ppq >>> 8) & 0xff, ppq & 0xff,
  ]);
  for (const body of tracks) appendChunk(out, "MTrk", body);
  return Uint8Array.from(out);
}
