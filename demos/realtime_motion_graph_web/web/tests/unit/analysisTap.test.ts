// Analysis seam client: the analysis_tap envelope, the typed
// analysis_state / analysis_update CustomEvents, the AnalysisStore
// range-replace rules (clear, epoch drop, per-range seq, ordering, loop
// wrap), the audible-playhead helper and the SMF writer.

import { afterEach, beforeEach, describe, expect, it } from "vitest";

import {
  AnalysisStore,
  RemoteBackend,
  analysisTapsAvailable,
  notesToSmf,
  playheadForNotes,
  type AnalysisUpdateEventPayload,
  type NoteItem,
} from "@demon/client";

import { FakeWebSocket, installFakeWebSocket } from "../replay/harness";

function note(
  start_s: number,
  pitch = 60,
  end_s = start_s + 0.25,
  instrument = "acoustic_piano",
): NoteItem {
  return { start_s, end_s, pitch, instrument };
}

function update(
  over: Partial<AnalysisUpdateEventPayload> & { seq: number },
): AnalysisUpdateEventPayload {
  return {
    type: "analysis_update",
    tap: "notes",
    source_epoch: 0,
    range_start_s: 0,
    span_s: 4,
    basis: "provisional",
    late: false,
    clear: false,
    kind: "note",
    items: [],
    playhead_s: 0,
    wall_s: 0.1,
    ...over,
  };
}

const starts = (store: AnalysisStore<NoteItem>) =>
  store.items().map((n) => n.start_s);

const placed = (rows: { item: NoteItem; shift_s: number }[]) =>
  rows.map((w) => [w.item.start_s, w.shift_s]);

async function openBackend(capabilities?: Record<string, unknown>) {
  const remote = new RemoteBackend(
    "ws://test.invalid/",
    new Float32Array(0),
    2,
    { use_server_fixture: true, fixture_name: "x.wav" },
  );
  const p = remote.connect();
  const ws = FakeWebSocket.last;
  ws.emitOpen();
  ws.emitMessage(
    JSON.stringify({
      type: "ready",
      duration: 0.02,
      sample_rate: 48000,
      channels: 2,
      ...(capabilities ? { capabilities } : {}),
    }),
  );
  ws.emitMessage(new ArrayBuffer(0));
  await p.catch(() => {});
  return { remote, ws };
}

describe("RemoteBackend analysis wire", () => {
  let restore: () => void;
  beforeEach(() => {
    restore = installFakeWebSocket();
  });
  afterEach(() => restore());

  it("sends the analysis_tap envelope with and without options", async () => {
    const { remote, ws } = await openBackend();
    remote.sendAnalysisTap("notes", true, {
      size: "small",
      instruments: ["acoustic_piano"],
      horizon_s: 8,
    });
    remote.sendAnalysisTap("notes", false);
    const frames = ws.sent
      .filter((d): d is string => typeof d === "string")
      .map((d) => JSON.parse(d))
      .filter((m) => m.type === "analysis_tap");
    expect(frames).toEqual([
      {
        type: "analysis_tap",
        tap: "notes",
        enabled: true,
        options: {
          size: "small",
          instruments: ["acoustic_piano"],
          horizon_s: 8,
        },
      },
      { type: "analysis_tap", tap: "notes", enabled: false },
    ]);
  });

  it("reads analysis_taps from the ready capabilities", async () => {
    const { remote } = await openBackend({
      swap: true,
      analysis_taps: ["notes"],
    });
    expect(remote.analysisTaps).toEqual(["notes"]);
    expect(analysisTapsAvailable({ analysis_taps: "notes" })).toEqual([]);
    expect(analysisTapsAvailable(null)).toEqual([]);
  });

  it("dispatches typed analysis events and feeds an attached store", async () => {
    const { remote, ws } = await openBackend();
    const states: unknown[] = [];
    const generic: unknown[] = [];
    remote.addEventListener("analysis_state", (e) =>
      states.push((e as CustomEvent).detail),
    );
    remote.addEventListener("json", (e) =>
      generic.push((e as CustomEvent).detail),
    );
    const store = new AnalysisStore<NoteItem>("notes");
    const detach = store.attach(remote);

    const state = {
      type: "analysis_state",
      tap: "notes",
      enabled: true,
      engine: "muscriptor",
      size: "small",
      licence: "terms",
      mode: "lookahead",
      error: null,
      dropped: 0,
    };
    ws.emitMessage(JSON.stringify(state));
    ws.emitMessage(
      JSON.stringify(
        update({ seq: 1, clear: true, source_epoch: 2, span_s: 0 }),
      ),
    );
    ws.emitMessage(
      JSON.stringify(update({ seq: 2, source_epoch: 2, items: [note(1)] })),
    );

    expect(states).toEqual([state]);
    expect(generic).toEqual([]);
    expect(starts(store)).toEqual([1]);
    detach();
    ws.emitMessage(
      JSON.stringify(update({ seq: 3, source_epoch: 2, items: [note(2)] })),
    );
    expect(starts(store)).toEqual([1]);
  });
});

describe("AnalysisStore", () => {
  it("applies updates sent without a server playhead", () => {
    const store = new AnalysisStore<NoteItem>("notes");
    expect(
      store.apply(update({ seq: 1, playhead_s: null, items: [note(1)] })),
    ).toBe(true);
    expect(starts(store)).toEqual([1]);
  });

  it("ignores other taps", () => {
    const store = new AnalysisStore<NoteItem>("notes");
    expect(
      store.apply(update({ seq: 1, tap: "beats", items: [note(1)] })),
    ).toBe(false);
    expect(store.size).toBe(0);
  });

  it("keeps items sorted by start then pitch across ranges", () => {
    const store = new AnalysisStore<NoteItem>("notes");
    store.apply(
      update({
        seq: 1,
        range_start_s: 4,
        items: [note(6.5), note(4.1, 64), note(4.1, 60)],
      }),
    );
    store.apply(
      update({ seq: 2, range_start_s: 0, items: [note(3.9), note(0.2)] }),
    );
    expect(store.items().map((n) => [n.start_s, n.pitch])).toEqual([
      [0.2, 60],
      [3.9, 60],
      [4.1, 60],
      [4.1, 64],
      [6.5, 60],
    ]);
  });

  it("replaces only items whose onset lies in the range", () => {
    const store = new AnalysisStore<NoteItem>("notes");
    // A note starting at 3.9 that rings into the next cell belongs to [0, 4).
    store.apply(
      update({
        seq: 1,
        range_start_s: 0,
        items: [note(1), note(3.9, 60, 4.6)],
      }),
    );
    store.apply(
      update({ seq: 2, range_start_s: 4, items: [note(4), note(7.99)] }),
    );
    store.apply(
      update({
        seq: 3,
        range_start_s: 4,
        basis: "settled",
        items: [note(5)],
      }),
    );
    expect(starts(store)).toEqual([1, 3.9, 5]);
    expect(store.items().map((n) => n.basis)).toEqual([
      "provisional",
      "provisional",
      "settled",
    ]);
    expect(store.settledItems().map((n) => n.start_s)).toEqual([5]);
    // An empty update empties its range.
    store.apply(update({ seq: 4, range_start_s: 0, items: [] }));
    expect(starts(store)).toEqual([5]);
  });

  it("drops a lower or repeated seq for the same range, not for another range", () => {
    const store = new AnalysisStore<NoteItem>("notes");
    store.apply(update({ seq: 5, range_start_s: 0, items: [note(1)] }));
    expect(
      store.apply(update({ seq: 4, range_start_s: 0, items: [note(2)] })),
    ).toBe(false);
    expect(
      store.apply(update({ seq: 5, range_start_s: 0, items: [note(2)] })),
    ).toBe(false);
    expect(
      store.apply(update({ seq: 3, range_start_s: 4, items: [note(4.5)] })),
    ).toBe(true);
    expect(starts(store)).toEqual([1, 4.5]);
  });

  it("clear drops everything, adopts the epoch, and older epochs are ignored", () => {
    const store = new AnalysisStore<NoteItem>("notes");
    store.apply(update({ seq: 1, source_epoch: 0, items: [note(1)] }));
    store.apply(update({ seq: 2, source_epoch: 1, clear: true, span_s: 0 }));
    expect(store.size).toBe(0);
    expect(store.epoch).toBe(1);
    // A late result from the old source is ignored.
    expect(
      store.apply(update({ seq: 3, source_epoch: 0, items: [note(2)] })),
    ).toBe(false);
    // Seq tracking restarts with the epoch.
    expect(
      store.apply(update({ seq: 4, source_epoch: 1, items: [note(3)] })),
    ).toBe(true);
    expect(starts(store)).toEqual([3]);
    // A clear from an older epoch is ignored too.
    expect(
      store.apply(update({ seq: 5, source_epoch: 0, clear: true })),
    ).toBe(false);
    expect(starts(store)).toEqual([3]);
  });

  it("notifies subscribers on accepted updates and reset only", () => {
    const store = new AnalysisStore<NoteItem>("notes");
    let calls = 0;
    const off = store.subscribe(() => calls++);
    store.apply(update({ seq: 2, items: [note(1)] }));
    store.apply(update({ seq: 1, items: [note(1)] }));
    store.reset();
    expect(calls).toBe(2);
    expect(store.epoch).toBe(null);
    off();
    store.apply(update({ seq: 3, items: [note(1)] }));
    expect(calls).toBe(2);
  });

  it("itemsIn returns overlapping items without wrap", () => {
    const store = new AnalysisStore<NoteItem>("notes");
    store.apply(
      update({
        seq: 1,
        span_s: 10,
        items: [note(1, 60, 2.5), note(3), note(9)],
      }),
    );
    expect(placed(store.itemsIn(2, 4))).toEqual([
      [1, 0],
      [3, 0],
    ]);
    expect(store.itemsIn(4, 2)).toEqual([]);
  });

  it("itemsIn unrolls the window across a loop band", () => {
    const store = new AnalysisStore<NoteItem>("notes");
    store.apply(
      update({
        seq: 1,
        span_s: 20,
        items: [note(2.5), note(4.2), note(7.8), note(8.5), note(12)],
      }),
    );
    const band = { start_s: 4, end_s: 8 };
    // Playhead at 7, 2 s ahead: [5, 9) reads [5, 8) then [4, 5) shifted +4.
    expect(placed(store.itemsIn(5, 9, band))).toEqual([
      [7.8, 0],
      [4.2, 4],
    ]);
    // Just past the band start, 0.5 s behind: [3.5, 6) reads [7.5, 8)
    // shifted -4, then [4, 6).
    expect(placed(store.itemsIn(3.5, 6, band))).toEqual([
      [7.8, -4],
      [4.2, 0],
    ]);
    // Items outside the band never show while it is active.
    const shown = store.itemsIn(0, 20, band).map((w) => w.item.start_s);
    expect(shown).not.toContain(2.5);
    expect(shown).not.toContain(12);
  });

  it("itemsIn wraps over the whole canvas", () => {
    const store = new AnalysisStore<NoteItem>("notes");
    store.apply(update({ seq: 1, span_s: 10, items: [note(0.5), note(9.5)] }));
    expect(placed(store.itemsIn(8, 12, { start_s: 0, end_s: 10 }))).toEqual([
      [9.5, 0],
      [0.5, 10],
    ]);
  });
});

describe("playheadForNotes", () => {
  it("subtracts the output latency", () => {
    expect(playheadForNotes(5, 0.05)).toBeCloseTo(4.95);
    expect(playheadForNotes(5, Number.NaN)).toBe(5);
    expect(playheadForNotes(0.01, 0.05)).toBe(0);
  });

  it("wraps across the loop band seam and the canvas end", () => {
    const band = { start_s: 4, end_s: 8 };
    expect(playheadForNotes(4.02, 0.05, band)).toBeCloseTo(7.97);
    expect(playheadForNotes(6, 0.05, band)).toBeCloseTo(5.95);
    // Read position outside the band (band just set): no band wrap.
    expect(playheadForNotes(9, 0.05, band, 10)).toBeCloseTo(8.95);
    expect(playheadForNotes(0.02, 0.05, null, 10)).toBeCloseTo(9.97);
  });
});

type ParsedTrack = {
  name?: string;
  tempo?: number;
  program?: number;
  /** [tick, status nibble, channel, key, velocity] */
  notes: number[][];
};

/** Parse an SMF far enough to check what notesToSmf writes. */
function parseSmf(bytes: Uint8Array) {
  const dv = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const tag = (o: number) => String.fromCharCode(...bytes.subarray(o, o + 4));
  expect(tag(0)).toBe("MThd");
  const format = dv.getUint16(8);
  const ntracks = dv.getUint16(10);
  const ppq = dv.getUint16(12);
  let o = 14;
  const tracks: ParsedTrack[] = [];
  for (let t = 0; t < ntracks; t++) {
    expect(tag(o)).toBe("MTrk");
    const len = dv.getUint32(o + 4);
    let p = o + 8;
    const end = p + len;
    let tick = 0;
    const track: ParsedTrack = { notes: [] };
    const vlq = () => {
      let v = 0;
      for (;;) {
        const b = bytes[p++];
        v = (v << 7) | (b & 0x7f);
        if (!(b & 0x80)) return v;
      }
    };
    while (p < end) {
      tick += vlq();
      const status = bytes[p++];
      if (status === 0xff) {
        const type = bytes[p++];
        const n = vlq();
        const data = bytes.subarray(p, p + n);
        p += n;
        if (type === 0x03) track.name = new TextDecoder().decode(data);
        if (type === 0x51) {
          track.tempo = (data[0] << 16) | (data[1] << 8) | data[2];
        }
      } else if ((status & 0xf0) === 0xc0) {
        track.program = bytes[p++];
      } else {
        const a = bytes[p++];
        const b = bytes[p++];
        track.notes.push([tick, status & 0xf0, status & 0x0f, a, b]);
      }
    }
    expect(p).toBe(end);
    tracks.push(track);
    o = end;
  }
  return { format, ntracks, ppq, tracks };
}

describe("notesToSmf", () => {
  it("writes a format-1 file: conductor + one track per instrument, drums on channel 10", () => {
    const smf = parseSmf(
      notesToSmf([
        note(0.5, 36, 0.6, "drums"),
        note(0, 60, 0.5, "acoustic_piano"),
        note(1, 40, 2, "electric_bass"),
      ]),
    );
    expect(smf.format).toBe(1);
    expect(smf.ppq).toBe(480);
    expect(smf.tracks.map((t) => t.name)).toEqual([
      "tempo",
      "acoustic_piano",
      "drums",
      "electric_bass",
    ]);
    expect(smf.tracks[0].tempo).toBe(500000);
    const [, piano, drums, bass] = smf.tracks;
    expect(piano.program).toBe(0);
    expect(drums.program).toBeUndefined();
    expect(bass.program).toBe(33);
    // 120 bpm, 480 ppq: 960 ticks per second; velocity fixed at 100.
    expect(piano.notes).toEqual([
      [0, 0x90, 0, 60, 100],
      [480, 0x80, 0, 60, 0],
    ]);
    expect(drums.notes).toEqual([
      [480, 0x90, 9, 36, 100],
      [576, 0x80, 9, 36, 0],
    ]);
    expect(bass.notes).toEqual([
      [960, 0x90, 1, 40, 100],
      [1920, 0x80, 1, 40, 0],
    ]);
  });

  it("cuts same-pitch overlaps, drops duplicates and notes before the origin", () => {
    const smf = parseSmf(
      notesToSmf(
        [note(1, 60, 3), note(2, 60, 2.5), note(2, 60, 2.5), note(0.5, 62, 1)],
        { originS: 1, bpm: 60, ppq: 100 },
      ),
    );
    // 60 bpm, 100 ppq: 100 ticks per second from the origin.
    expect(smf.tracks[1].notes).toEqual([
      [0, 0x90, 0, 60, 100],
      [100, 0x80, 0, 60, 0],
      [100, 0x90, 0, 60, 100],
      [150, 0x80, 0, 60, 0],
    ]);
  });
});
