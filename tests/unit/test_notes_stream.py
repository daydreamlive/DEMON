"""Streaming MuScriptor decoder over a fake LM: token encoder round trip,
ledger carry-over across hops, re-timing, tie prologue, guard discard,
token cap, wall budget, take reset, and the ``notes`` engine's streaming
path. No weights, no GPU."""

from __future__ import annotations

import importlib.util
import random

import numpy as np
import pytest

from acestep.analysis import notes_stream as ns
from acestep.analysis.engine import AnalysisRequest
from acestep.analysis.notes_engine import NOTES, NotesEngine
from acestep.analysis.notes_stream import (
    FRAME_RATE,
    StreamingNotesDecoder,
    WindowEvent,
    WindowNote,
    build_vocab,
    decode_window,
    encode_window,
    window_events,
)

VOCAB = build_vocab()
INDEX = {e: i for i, e in enumerate(VOCAB)}
EOS = 1
SR = 16000
HAS_MUSCRIPTOR = importlib.util.find_spec("muscriptor") is not None


def _inst(program):
    return "drums" if program == 128 else f"prog{program}"


# ---------------------------------------------------------------------------
# Truth notes in absolute ticks, and a fake LM that continues the canonical
# token stream of whatever the truth holds in the window.
# ---------------------------------------------------------------------------


class Truth:
    """``(onset, end, program, pitch, is_drum)`` in absolute canvas ticks."""

    def __init__(self, notes):
        self.notes = sorted(notes)

    def window_tokens(self, ws, frontier):
        """Canonical tokens for the window ``[ws, frontier)``: notes heard
        so far, ends at or past the frontier unknown (open)."""
        wnotes = []
        for onset, end, program, pitch, drum in self.notes:
            if onset >= frontier:
                continue
            if drum:
                if onset >= ws:
                    wnotes.append(WindowNote(onset - ws, onset - ws + 1, 128, pitch, True))
                continue
            if end <= ws:
                continue
            rel_end = end - ws if end < frontier else None
            rel_onset = onset - ws if onset >= ws else None
            wnotes.append(WindowNote(rel_onset, rel_end, program, pitch))
        ties, events = window_events(wnotes)
        return encode_window(ties, events, INDEX), ties


class TruthLM:
    """Fake backend: the test sets ``ws`` / ``frontier`` (ticks) before a
    step; generate asserts the prompt is a prefix of the truth's canonical
    stream for that window and yields the rest plus EOS."""

    vocab = VOCAB
    eos_id = EOS

    def __init__(self, truth, cap_extra=0):
        self.truth = truth
        self.ws = 0
        self.frontier = 0
        self.calls = []

    def instrument_for_program(self, program):
        return _inst(program)

    def generate(self, prompt_ids, audio, sr, instruments, max_new_tokens):
        full, ties = self.truth.window_tokens(self.ws, self.frontier)
        prompt_ids = list(prompt_ids)
        assert full[: len(prompt_ids)] == prompt_ids, "prompt is not the settled prefix"
        tail = full[len(prompt_ids):]
        self.calls.append({"prompt": prompt_ids, "ties": ties, "tail": len(tail),
                           "audio_len": len(audio), "sr": sr, "instruments": instruments})
        yield from tail[:max_new_tokens]
        if len(tail) <= max_new_tokens:
            yield EOS


def _random_truth(seed, dur_ticks, n=60, programs=(0, 33), drums=True):
    rng = random.Random(seed)
    notes = []
    last_end = {}
    for _ in range(n):
        program = rng.choice(programs)
        pitch = rng.randint(36, 84)
        onset = rng.randint(0, dur_ticks - 2)
        key = (program, pitch)
        # MuScriptor never holds two notes of one key at once.
        if any(o <= onset < e for o, e in last_end.get(key, [])):
            continue
        length = rng.choice([5, 20, 60, 180, 700])  # 700 ticks spans a window start
        end = min(onset + length, dur_ticks + 400)
        if any(onset < e and o < end for o, e in last_end.get(key, [])):
            continue
        last_end.setdefault(key, []).append((onset, end))
        notes.append((onset, end, program, pitch, False))
    if drums:
        for t in range(10, dur_ticks, 50):
            notes.append((t, t + 1, 128, 36, True))
    return Truth(notes)


def _stream(dec, lm, total_s, hop_s=0.2, window_s=5.0):
    """Drive ``dec`` like the tap's follow mode, every ``hop_s``."""
    results = []
    f = hop_s
    while f <= total_s + 1e-9:
        ws = max(0.0, f - window_s)
        lm.ws = ns._tick_ceil(ws)
        lm.frontier = ns._tick_floor(f)
        audio = np.zeros(int(round((f - ws) * SR)), dtype=np.float32)
        results.append(dec.step(audio, SR, ws, f))
        f = round(f + hop_s, 6)
    return results


# ---------------------------------------------------------------------------
# Vocabulary and encoder
# ---------------------------------------------------------------------------


def test_vocab_layout():
    assert len(VOCAB) == 1393  # the small model's card
    assert VOCAB[:3] == [("PAD", 0), ("EOS", 0), ("UNK", 0)]
    assert VOCAB[3] == ("shift", 0) and VOCAB[1003] == ("shift", 1000)
    assert VOCAB[1004] == ("pitch", 0)
    assert INDEX[("tie", 0)] == 1004 + 128 + 2


@pytest.mark.skipif(not HAS_MUSCRIPTOR, reason="muscriptor not installed")
def test_vocab_matches_muscriptor():
    from muscriptor.tokenizer.mt3 import MT3Tokenizer

    tok = MT3Tokenizer(instrument_vocabulary="MT3_FULL_PLUS", max_shift_steps=1001)
    assert [(e.type, e.value) for e in tok._vocab] == VOCAB
    assert tok.eos_id == EOS
    keys = [(0, 64), (0, 60), (33, 40)]
    assert tok.tie_section_token_ids(keys) == encode_window(keys, [], INDEX)


def test_encode_layout_by_hand():
    events = [
        WindowEvent(0, False, 0, 1, 60),
        WindowEvent(0, False, 0, 1, 64),
        WindowEvent(25, True, 128, 1, 36),
        WindowEvent(25, False, 0, 0, 60),
        WindowEvent(40, False, 33, 1, 40),
    ]
    got = encode_window([(0, 67)], events, INDEX)
    want = [("program", 0), ("pitch", 67), ("tie", 0),
            ("velocity", 1), ("pitch", 60), ("pitch", 64),  # program 0 carries from the prologue
            ("shift", 25), ("velocity", 0), ("pitch", 60), ("drum", 36),
            ("shift", 40), ("program", 33), ("velocity", 1), ("pitch", 40)]
    assert [VOCAB[i] for i in got] == want


@pytest.mark.parametrize("seed", range(8))
def test_encoder_round_trip(seed):
    truth = _random_truth(seed, 480, n=40)
    for ws, frontier in [(0, 480), (130, 480), (300, 470)]:
        tokens, ties = truth.window_tokens(ws, frontier)
        notes = decode_window(tokens, VOCAB, ties)
        t2, e2 = window_events(notes)
        assert sorted(t2) == sorted(ties)
        assert encode_window(t2, e2, INDEX) == tokens  # encode(decode(tokens)) == tokens


def test_retrigger_round_trip():
    notes = [WindowNote(0, 30, 0, 60), WindowNote(30, 50, 0, 60), WindowNote(None, 30, 0, 62)]
    ties, events = window_events(notes)
    tokens = encode_window(ties, events, INDEX)
    assert sorted(decode_window(tokens, VOCAB, ties), key=repr) == sorted(notes, key=repr)


@pytest.mark.skipif(not HAS_MUSCRIPTOR, reason="muscriptor not installed")
@pytest.mark.parametrize("seed", range(4))
def test_decoder_agrees_with_muscriptor_tracker(seed):
    from muscriptor.events import ChunkBoundary, OpenNoteTracker, _DrumHit, _EndNote, _StartNote
    from muscriptor.tokenizer.notes import build_event_vocab

    truth = _random_truth(seed, 480, n=40)
    tokens, _ = truth.window_tokens(0, 480)
    tracker = OpenNoteTracker(build_event_vocab(1001), 100)
    tracker.feed(ChunkBoundary(0.0, None))
    ref = []
    for t in tokens:
        for a in tracker.feed(t):
            kind = {_StartNote: "start", _EndNote: "end", _DrumHit: "drum"}[type(a)]
            program = 128 if kind == "drum" else a.program
            ref.append((kind, program, a.pitch, round(a.time * 100)))
    dec = ns.WindowDecoder(VOCAB)
    ours = [(a.kind, a.program, a.pitch, a.tick) for t in tokens for a in dec.feed(t)]
    assert ours == ref


# ---------------------------------------------------------------------------
# Ledger across hops
# ---------------------------------------------------------------------------


def _ledger_ticks(dec):
    return sorted(((n.onset, n.end, n.program, n.pitch, n.is_drum) for n in dec._ledger),
                  key=lambda r: (r[0], -1 if r[1] is None else r[1], r[2:]))


@pytest.mark.parametrize("seed", range(4))
def test_ledger_carries_over_and_matches_truth(seed):
    truth = _random_truth(seed, 1200, n=80)
    lm = TruthLM(truth)
    dec = StreamingNotesDecoder(lm, new_token_cap=10_000, hop_wall_budget_s=1e9)
    results = _stream(dec, lm, 12.0)
    assert not any(r.truncated for r in results)
    commit = dec._commit_tick
    assert commit == ns._tick_floor(12.0 - ns.GUARD_S)
    # Every truth note committed before the commit point is in the ledger,
    # with its end where the end was also before the commit point.
    last_ws = ns._tick_ceil(12.0 - 5.0)
    for onset, end, program, pitch, drum in truth.notes:
        if onset >= commit or (end <= last_ws):
            continue
        want_end = end if end < commit else None
        assert (onset, want_end, program, pitch, drum) in _ledger_ticks(dec)
    # Prompts carried the ledger: later hops decode only a short tail.
    tails = [c["tail"] for c in lm.calls[30:]]
    prompts = [len(c["prompt"]) for c in lm.calls[30:]]
    assert max(tails) < min(prompts)
    assert any(c["ties"] for c in lm.calls), "no window started inside a sounding note"


def test_prompt_retimes_to_window_start():
    # The sustained note keeps a sounding note at every window start, so
    # the ledger forces the prompt.
    truth = Truth([(620, 650, 0, 60, False), (100, 2000, 33, 40, False)])
    lm = TruthLM(truth)
    dec = StreamingNotesDecoder(lm, new_token_cap=10_000, hop_wall_budget_s=1e9)
    _stream(dec, lm, 7.5, hop_s=0.5)
    last = lm.calls[-1]
    # Window [2.5, 7.5): the note at canvas 6.20 s sits at window tick 370.
    toks = [VOCAB[i] for i in last["prompt"]]
    assert ("shift", 370) in toks and ("shift", 400) in toks


def test_tie_prologue_for_note_sounding_at_window_start():
    truth = Truth([(100, 700, 33, 40, False)])
    lm = TruthLM(truth)
    dec = StreamingNotesDecoder(lm, new_token_cap=10_000, hop_wall_budget_s=1e9)
    _stream(dec, lm, 8.0, hop_s=0.5)
    # Window [3.0, 8.0) starts inside the note: prologue program 33 pitch 40 tie,
    # then its offset at window tick 400.
    last = [VOCAB[i] for i in lm.calls[-1]["prompt"]]
    assert last[:3] == [("program", 33), ("pitch", 40), ("tie", 0)]
    assert ("shift", 400) in last
    assert _ledger_ticks(dec) == [(100, 700, 33, 40, False)]


def _hop(dec, lm, ws, f):
    lm.ws, lm.frontier = ns._tick_ceil(ws), ns._tick_floor(f)
    return dec.step(np.zeros(int(round((f - ws) * SR)), np.float32), SR, ws, f)


MID_MUSIC = [(100, 450, 26, 48, False), (200, 900, 33, 36, False), (350, 380, 0, 60, False)]


def test_first_hop_sends_no_prompt_and_adopts_model_ties():
    lm = TruthLM(Truth(MID_MUSIC))
    dec = StreamingNotesDecoder(lm, new_token_cap=10_000, hop_wall_budget_s=1e9)
    r = _hop(dec, lm, 3.0, 5.0)  # take starts mid-music
    assert lm.calls[0]["prompt"] == [] and r.prompt_tokens == 0
    tied = sorted((n.program, n.pitch, n.start_s, n.end_s) for n in dec._open_notes() + r.closed
                  if n.tied_in)
    assert tied == [(26, 48, 3.0, 4.5), (33, 36, 3.0, None)]
    # Tied-in notes are not onsets; their ends are reported.
    assert [(n.pitch, n.start_s) for n in r.new_notes] == [(60, 3.5)]
    assert sorted(n.pitch for n in r.closed) == [48, 60]
    closed, opened = dec.notes_in(0.0, 5.0)
    assert [n.pitch for n in closed] == [60] and opened == []


def test_second_hop_prompt_starts_with_adopted_tie_keys():
    lm = TruthLM(Truth(MID_MUSIC))
    dec = StreamingNotesDecoder(lm, new_token_cap=10_000, hop_wall_budget_s=1e9)
    _hop(dec, lm, 3.0, 5.0)
    r = _hop(dec, lm, 3.2, 5.2)  # TruthLM asserts the whole prompt is the settled prefix
    head = [VOCAB[i] for i in lm.calls[1]["prompt"][:5]]
    assert head == [("program", 26), ("pitch", 48), ("program", 33), ("pitch", 36), ("tie", 0)]
    assert r.prompt_tokens > 5 and r.new_notes == ()
    r = _hop(dec, lm, 4.5, 9.5)
    assert [(n.pitch, n.start_s, n.end_s, n.tied_in) for n in r.revised] == [(36, 3.0, 9.0, True)]


def test_window_start_in_covered_silence_forces_the_empty_tie():
    lm = TruthLM(Truth([(10, 40, 0, 60, False), (620, 650, 0, 62, False)]))
    dec = StreamingNotesDecoder(lm, new_token_cap=10_000, hop_wall_budget_s=1e9)
    _hop(dec, lm, 0.0, 1.0)
    _hop(dec, lm, 0.0, 2.0)   # covered through 1.85 s
    r = _hop(dec, lm, 1.5, 6.5)  # nothing sounds at 1.5 s and it is analysed: [tie] is the truth
    assert lm.calls[-1]["prompt"] == [INDEX[("tie", 0)]]
    assert r.prompt_tokens == 1 and r.new_tokens == lm.calls[-1]["tail"] <= 5  # only the tail
    assert [n.start_s for n in r.new_notes] == [6.2]
    closed, opened = dec.notes_in(0.0, 7.0)
    assert closed == [] and [n.start_s for n in opened] == [6.2]  # 0.1-0.4 pruned behind 1.5


def test_window_start_beyond_coverage_sends_empty_prompt():
    lm = TruthLM(Truth([(10, 40, 0, 60, False), (620, 650, 0, 62, False)]))
    dec = StreamingNotesDecoder(lm, new_token_cap=10_000, hop_wall_budget_s=1e9)
    _hop(dec, lm, 0.0, 1.0)      # covered through 0.85 s
    _hop(dec, lm, 1.5, 6.5)      # a stall gap: 1.5 s was never analysed
    assert lm.calls[-1]["prompt"] == []


def test_window_start_past_the_ledger_falls_back_to_empty_prompt():
    lm = TruthLM(Truth([(0, 2000, 33, 40, False)]))
    dec = StreamingNotesDecoder(lm, new_token_cap=10_000, hop_wall_budget_s=1e9)
    _hop(dec, lm, 0.0, 1.0)
    _hop(dec, lm, 0.5, 1.5)
    assert lm.calls[-1]["prompt"][:3] == [INDEX[("program", 33)], INDEX[("pitch", 40)], INDEX[("tie", 0)]]
    _hop(dec, lm, 4.0, 9.0)  # a stall: the window starts past the commit point
    assert lm.calls[-1]["prompt"] == []
    # The ledger keeps its own note (not a second, adopted one).
    assert [(n.start_s, n.tied_in) for n in dec._open_notes()] == [(0.0, False)]


def test_guard_holds_late_onsets_for_next_hop():
    truth = Truth([(195, 260, 0, 60, False)])  # onset at 1.95 s
    lm = TruthLM(truth)
    dec = StreamingNotesDecoder(lm, new_token_cap=10_000, hop_wall_budget_s=1e9)
    lm.ws, lm.frontier = 0, 200
    r = dec.step(np.zeros(2 * SR, np.float32), SR, 0.0, 2.0)
    assert r.new_notes == () and dec._ledger == []          # inside the 0.15 s guard
    lm.ws, lm.frontier = 0, 220
    r = dec.step(np.zeros(int(2.2 * SR), np.float32), SR, 0.0, 2.2)
    assert [n.start_s for n in r.new_notes] == [1.95] and r.open_notes
    lm.ws, lm.frontier = 0, 300
    r = dec.step(np.zeros(3 * SR, np.float32), SR, 0.0, 3.0)
    assert [(n.start_s, n.end_s) for n in r.revised] == [(1.95, 2.6)]


def test_audio_aligned_to_token_grid_and_capped_at_window():
    lm = TruthLM(Truth([]))
    dec = StreamingNotesDecoder(lm)
    lm.ws, lm.frontier = ns._tick_ceil(0.0033), 200
    dec.step(np.zeros(int(round((2.0 - 0.0033) * SR)), np.float32), SR, 0.0033, 2.0)
    assert lm.calls[-1]["audio_len"] == int(1.99 * SR)  # starts on the 10 ms grid
    dec.reset()
    lm.ws, lm.frontier = 400, 900
    dec.step(np.zeros(9 * SR, np.float32), SR, 0.0, 9.0)  # caller sent 9 s
    assert lm.calls[-1]["audio_len"] == 5 * SR


class EndlessLM(TruthLM):
    """Never emits EOS."""

    def generate(self, prompt_ids, audio, sr, instruments, max_new_tokens):
        self.calls.append({"prompt": list(prompt_ids)})
        i = 0
        while True:
            i += 1
            yield INDEX[("shift", min(i, 999))]


def test_token_cap_discards_hop_and_keeps_ledger(caplog):
    truth = Truth([(50, 80, 0, 60, False)])
    lm = TruthLM(truth)
    dec = StreamingNotesDecoder(lm, new_token_cap=10_000, hop_wall_budget_s=1e9)
    _stream(dec, lm, 1.0, hop_s=1.0)
    before = _ledger_ticks(dec)
    commit = dec._commit_tick
    dec.backend = EndlessLM(truth)
    dec.new_token_cap = dec.full_token_cap = 16
    r1 = dec.step(np.zeros(2 * SR, np.float32), SR, 0.0, 2.0)
    r2 = dec.step(np.zeros(3 * SR, np.float32), SR, 0.0, 3.0)
    assert r1.truncated and r1.reason == "cap" and r1.new_tokens == 16
    assert r2.truncated
    assert _ledger_ticks(dec) == before and dec._commit_tick == commit


def test_wall_budget_discards_hop():
    t = [0.0]

    def clock():
        t[0] += 0.1
        return t[0]

    lm = TruthLM(_random_truth(1, 300, n=40))
    dec = StreamingNotesDecoder(lm, new_token_cap=10_000, hop_wall_budget_s=0.35, clock=clock)
    lm.ws, lm.frontier = 0, 300
    r = dec.step(np.zeros(3 * SR, np.float32), SR, 0.0, 3.0)
    assert r.truncated and r.reason == "wall" and dec._ledger == []


def test_warning_once_per_take(monkeypatch):
    warnings = []
    monkeypatch.setattr(ns.logger, "warning", lambda *a, **k: warnings.append(a))
    dec = StreamingNotesDecoder(EndlessLM(Truth([])), new_token_cap=4)
    for f in (1.0, 2.0, 3.0):
        dec.step(np.zeros(int(f * SR), np.float32), SR, 0.0, f)
    assert len(warnings) == 1
    dec.reset()
    dec.step(np.zeros(SR, np.float32), SR, 0.0, 1.0)
    assert len(warnings) == 2


def test_frontier_going_back_resets():
    truth = Truth([(50, 80, 0, 60, False)])
    lm = TruthLM(truth)
    dec = StreamingNotesDecoder(lm, new_token_cap=10_000, hop_wall_budget_s=1e9)
    _stream(dec, lm, 2.0, hop_s=1.0)
    assert dec._ledger
    lm.truth = Truth([])
    lm.ws, lm.frontier = 0, 100
    dec.step(np.zeros(SR, np.float32), SR, 0.0, 1.0)
    assert dec._ledger == []


def test_stale_open_note_closed():
    truth = Truth([(10, 5000, 0, 60, False)])
    lm = TruthLM(truth)
    dec = StreamingNotesDecoder(lm, new_token_cap=10_000, hop_wall_budget_s=1e9)
    _stream(dec, lm, 10.2, hop_s=0.2)
    assert dec._ledger[0].end is None            # commit 10.05 s: 10.04 s old
    lm.ws, lm.frontier = ns._tick_ceil(5.4), ns._tick_floor(10.4)
    dec.step(np.zeros(5 * SR, np.float32), SR, 5.4, 10.4)
    assert dec._ledger[0].end == dec._commit_tick == 1025


# ---------------------------------------------------------------------------
# Engine integration
# ---------------------------------------------------------------------------


def _req(t0, span, lead, tail, epoch=1, stream=None, sr=SR):
    n = int(round((lead + span + tail) * sr))
    opts = {} if stream is None else {"stream": stream}
    return AnalysisRequest(audio=np.zeros(n, np.float32), sr=sr, t0_s=t0, span_s=span,
                           context_lead_s=lead, context_tail_s=tail, source_epoch=epoch,
                           revision=0, options=opts)


def test_spec_is_stateful():
    assert NOTES.stateful is True


def test_engine_streaming_items_and_open_items():
    truth = Truth([(30, 60, 0, 60, False), (120, 400, 33, 40, False), (150, 151, 128, 36, True)])
    lm = TruthLM(truth)
    eng = NotesEngine("small", model=object(), stream_backend=lm)
    lm.ws, lm.frontier = 0, 200
    # Follow window [0, 2.0), reported stretch [0, 1.7).
    res = eng.analyze(_req(0.0, 1.7, 0.0, 0.3, stream={"take": 1}))
    assert [(i["start_s"], i["end_s"], i["pitch"], i["instrument"]) for i in res.items] == [
        (0.3, 0.6, 60, "prog0"), (1.5, 1.51, 36, "drums")]
    assert [(i["start_s"], i["end_s"], i["instrument"]) for i in res.open_items] == [
        (1.2, 2.0, "prog33")]
    assert res.truncated is False and res.range_start_s == 0.0 and res.span_s == 1.7


def test_engine_take_change_resets_ledger():
    truth = Truth([(30, 60, 0, 60, False)])
    lm = TruthLM(truth)
    eng = NotesEngine("small", model=object(), stream_backend=lm)
    lm.ws, lm.frontier = 0, 200
    eng.analyze(_req(0.0, 1.7, 0.0, 0.3, stream={"take": 1}))
    lm.truth = Truth([])
    res = eng.analyze(_req(0.0, 1.7, 0.0, 0.3, stream={"take": 2}))
    assert res.items == [] and lm.calls[-1]["prompt"] == []  # new take: empty prompt


def test_engine_stream_ids_are_separate_and_options_pass_through():
    lm = TruthLM(Truth([(30, 60, 0, 60, False)]))
    eng = NotesEngine("small", model=object(), stream_backend=lm)
    lm.ws, lm.frontier = 0, 200
    req = _req(0.0, 1.7, 0.0, 0.3, stream={"take": 1, "id": "a"})
    req = AnalysisRequest(**{**req.__dict__, "options": {**req.options, "instruments": ["acoustic_piano"]}})
    eng.analyze(req)
    eng.analyze(_req(0.0, 1.7, 0.0, 0.3, stream={"take": 1, "id": "b"}))
    assert lm.calls[0]["instruments"] == ["acoustic_piano"]
    assert lm.calls[1]["prompt"] == []  # b starts its own ledger
    assert set(eng._streams) == {"a", "b"}


def test_engine_unload_drops_streams():
    lm = TruthLM(Truth([]))
    eng = NotesEngine("small", model=object(), stream_backend=lm)
    lm.ws, lm.frontier = 0, 100
    eng.analyze(_req(0.0, 0.7, 0.0, 0.3, stream={}))
    eng.unload()
    assert len(eng._streams) == 0


# ---------------------------------------------------------------------------
# The real backend over a tiny randomly initialised MuScriptor LM on CPU
# (no weights are loaded): prompt skipping, the token cap, and the
# conditioner's length mask for a window shorter than 5 s.
# ---------------------------------------------------------------------------


def _tiny_transcription_model():
    import torch
    from muscriptor.tokenizer.mt3 import MT3Tokenizer
    from muscriptor.transcription_model import TranscriptionModel, _build_model, _ModelConfig

    torch.manual_seed(0)
    device = torch.device("cpu")
    lm = _build_model(device, _ModelConfig(dim=32, num_heads=2, num_layers=1, card=1393)).eval()
    tok = MT3Tokenizer(instrument_vocabulary="MT3_FULL_PLUS", max_shift_steps=1001)
    return TranscriptionModel(model=lm, tokenizer=tok, device=device)


def _capture_mel(tm):
    cond = tm._model.condition_provider.conditioners["self_wav"]
    seen = []
    orig = cond.forward

    def forward(x):
        out = orig(x)
        seen.append((out[0].detach().clone(), int(x.length[0])))
        return out

    cond.forward = forward
    return seen


@pytest.mark.skipif(not HAS_MUSCRIPTOR, reason="muscriptor not installed")
@pytest.mark.parametrize("pad_mode,frames", [("mask", 501), ("short", 201)])
def test_real_backend_on_tiny_random_lm(pad_mode, frames):
    tm = _tiny_transcription_model()
    seen = _capture_mel(tm)
    backend = ns.MuScriptorStreamBackend(tm, pad_mode=pad_mode)
    assert backend.vocab == VOCAB and backend.eos_id == EOS
    audio = (0.1 * np.random.default_rng(0).standard_normal(2 * 48000)).astype(np.float32)
    prompt = encode_window([(0, 60)], [WindowEvent(10, False, 0, 0, 60)], INDEX)
    out = list(backend.generate(prompt, audio, 48000, ["acoustic_piano"], 8))
    assert 1 <= len(out) <= 8 and all(0 <= t < 1393 for t in out)
    emb, length = seen[-1]
    assert length == 2 * ns.MODEL_SR and emb.shape[1] == frames
    kept = int(np.ceil(length / 160))
    assert emb[0, :kept].abs().sum() > 0
    assert emb[0, kept:].abs().sum() == 0  # masked frames are exact zeros
    # Forbidden programs never appear.
    allowed = {INDEX[("program", 0)]}
    assert not any(VOCAB[t][0] == "program" and t not in allowed for t in out)
    # The decoder runs end to end on it.
    dec = StreamingNotesDecoder(backend, new_token_cap=8, full_token_cap=8,
                                hop_wall_budget_s=60.0, full_hop_wall_budget_s=60.0)
    r = dec.step(audio, 48000, 0.0, 2.0, ["acoustic_piano"])
    assert r.prompt_tokens == 0 and r.new_tokens <= 8  # first hop: empty prompt



@pytest.mark.parametrize("seed", range(4))
def test_take_starting_mid_music_matches_truth(seed):
    truth = _random_truth(seed, 1200, n=80)
    lm = TruthLM(truth)
    dec = StreamingNotesDecoder(lm, new_token_cap=10_000, hop_wall_budget_s=1e9)
    f = 3.2
    while f <= 12.0 + 1e-9:
        _hop(dec, lm, max(3.0, f - 5.0), f)  # TruthLM checks every forced prefix
        f = round(f + 0.2, 6)
    commit, last_ws = dec._commit_tick, ns._tick_ceil(7.0)
    got = _ledger_ticks(dec)
    for onset, end, program, pitch, drum in truth.notes:
        if not 300 <= onset < commit or end <= last_ws:
            continue
        assert (onset, end if end < commit else None, program, pitch, drum) in got
    assert all(n.onset >= 300 for n in dec._ledger)
