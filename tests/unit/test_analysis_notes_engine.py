"""The ``notes`` engine over a fake MuScriptor model: pairing, context
dropping, canvas-second conversion, option pass-through. No weights."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from acestep.analysis import notes_engine
from acestep.analysis.engine import AnalysisEngine, AnalysisRequest
from acestep.analysis.notes_engine import NotesEngine, PairedNote, pair_events


# Same class names as muscriptor.events: pair_events matches on name.
@dataclass
class NoteStartEvent:
    pitch: int
    start_time: float
    index: int
    instrument: str


@dataclass
class NoteEndEvent:
    end_time: float
    start_event: NoteStartEvent


@dataclass
class ProgressEvent:
    completed: int
    total: int


def _note(i, pitch, st, en, inst="acoustic_piano"):
    s = NoteStartEvent(pitch, st, i, inst)
    return s, NoteEndEvent(en, s)


class FakeModel:
    """Yields a fixed event stream; records what it was called with."""

    def __init__(self, events):
        self.events = events
        self.calls = []

    def transcribe(self, audio, instruments=None):
        wav, sr = audio
        self.calls.append({"shape": tuple(wav.shape), "sr": sr, "instruments": instruments,
                           "dtype": str(wav.dtype)})
        yield ProgressEvent(0, 1)
        yield from self.events
        yield ProgressEvent(1, 1)


def test_pair_events_clamps_and_closes():
    a_s, a_e = _note(0, 60, 0.10, 0.50)
    b_s, b_e = _note(1, 64, 0.30, 9.00)          # offset past the audio -> clamp
    c_s = NoteStartEvent(67, 1.90, 2, "drums")    # never closed -> 0.25 s, clamped
    d_s = NoteStartEvent(55, 0.20, 3, "acoustic_bass")  # never closed, room for 0.25 s
    e_s, e_e = _note(4, 48, 2.50, 3.00)          # start past the audio -> zero length
    events = [ProgressEvent(0, 1), a_s, b_s, d_s, a_e, c_s, e_s, b_e, e_e]
    notes = pair_events(events, duration_s=2.0)
    assert notes == [
        PairedNote(0.10, 0.50, 60, "acoustic_piano"),
        PairedNote(0.20, 0.45, 55, "acoustic_bass"),
        PairedNote(0.30, 2.00, 64, "acoustic_piano"),
        PairedNote(1.90, 2.00, 67, "drums"),
        PairedNote(2.50, 2.50, 48, "acoustic_piano"),
    ]
    assert all(n.end_s >= n.start_s for n in notes)


def test_pair_events_unclosed_start_past_audio_guarded():
    s = NoteStartEvent(60, 3.0, 0, "acoustic_piano")
    assert pair_events([s], duration_s=2.0) == [PairedNote(3.0, 3.0, 60, "acoustic_piano")]


def _engine(events):
    model = FakeModel(events)
    return NotesEngine("small", model), model


def test_engine_satisfies_protocol():
    eng, _ = _engine([])
    assert isinstance(eng, AnalysisEngine)
    assert eng.name == "notes" and eng.size == "small"


def test_analyze_drops_context_and_converts_to_canvas_seconds():
    sr = 16000
    # Window = 0.5 s lead + 4 s cell + 0.5 s tail, cell at canvas t0 = 12 s.
    events = []
    for i, (st, en) in enumerate([
        (0.20, 0.60),   # onset in lead context: dropped (even though it rings into the cell)
        (0.50, 0.90),   # onset exactly at cell start: kept
        (2.00, 2.40),   # inside: kept
        (4.499, 4.70),  # just before cell end: kept, offset runs into the tail
        (4.50, 4.80),   # onset exactly at cell end: dropped
        (4.80, 4.90),   # in tail context: dropped
    ]):
        events.extend(_note(i, 60 + i, st, en))
    eng, model = _engine(events)
    req = AnalysisRequest(
        audio=np.zeros(int(5.0 * sr), np.float32), sr=sr, t0_s=12.0, span_s=4.0,
        context_lead_s=0.5, context_tail_s=0.5, source_epoch=3, revision=41,
        options={"instruments": ["acoustic_piano"], "horizon_s": 8.0},
    )
    res = eng.analyze(req)
    assert res.kind == "note"
    assert (res.range_start_s, res.span_s) == (12.0, 4.0)
    assert (res.source_epoch, res.revision) == (3, 41)
    got = [(round(i["start_s"], 6), round(i["end_s"], 6), i["pitch"], i["instrument"]) for i in res.items]
    assert got == [
        (12.0, 12.4, 61, "acoustic_piano"),
        (13.5, 13.9, 62, "acoustic_piano"),
        (15.999, 16.2, 63, "acoustic_piano"),
    ]
    assert model.calls == [{"shape": (1, 80000), "sr": sr, "instruments": ["acoustic_piano"],
                            "dtype": "torch.float32"}]
    assert res.wall_s >= 0.0


def test_analyze_edge_without_context_keeps_edge_onsets():
    sr = 16000
    events = [*_note(0, 60, 0.0, 0.3), *_note(1, 62, 3.9, 4.2), *_note(2, 64, 4.1, 4.3)]
    eng, model = _engine(events)
    # Canvas start: no lead context; 0.3 s of tail context.
    req = AnalysisRequest(
        audio=np.zeros(int(4.3 * sr), np.float32), sr=sr, t0_s=0.0, span_s=4.0,
        context_lead_s=0.0, context_tail_s=0.3, source_epoch=0, revision=0,
    )
    res = eng.analyze(req)
    assert [(i["pitch"], i["start_s"]) for i in res.items] == [(60, 0.0), (62, 3.9)]
    assert model.calls[0]["instruments"] is None


def test_analyze_downmixes_stereo_and_checks_sr():
    eng, model = _engine([])
    req = AnalysisRequest(
        audio=np.ones((2, 1600), np.float32), sr=16000, t0_s=0.0, span_s=0.1,
        context_lead_s=0.0, context_tail_s=0.0, source_epoch=0, revision=0,
    )
    eng.analyze(req)
    assert model.calls[0]["shape"] == (1, 1600)
    low = AnalysisRequest(
        audio=np.zeros(800, np.float32), sr=8000, t0_s=0.0, span_s=0.1,
        context_lead_s=0.0, context_tail_s=0.0, source_epoch=0, revision=0,
    )
    with pytest.raises(ValueError, match="sr >= 16000"):
        eng.analyze(low)


def test_unload_is_idempotent_and_blocks_analyze():
    eng, _ = _engine([])
    eng.unload()
    eng.unload()
    with pytest.raises(RuntimeError, match="unloaded"):
        eng.analyze(AnalysisRequest(
            audio=np.zeros(16, np.float32), sr=16000, t0_s=0.0, span_s=0.001,
            context_lead_s=0.0, context_tail_s=0.0, source_epoch=0, revision=0,
        ))


def test_make_engine_uses_loader(monkeypatch):
    calls = []

    def fake_load(size, device):
        calls.append((size, device))
        return FakeModel([])

    monkeypatch.setattr(notes_engine, "_load_model", fake_load)
    eng = notes_engine.NOTES.make_engine("medium", "cuda")
    assert isinstance(eng, NotesEngine) and eng.size == "medium"
    assert calls == [("medium", "cuda")]


def test_module_is_torch_free_at_import():
    import subprocess
    import sys

    code = (
        "import sys; import acestep.analysis.notes_engine, acestep.analysis.host, "
        "acestep.analysis.midi_transcribe; "
        "sys.exit(1 if ('torch' in sys.modules or 'muscriptor' in sys.modules) else 0)"
    )
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    assert subprocess.run([sys.executable, "-c", code], cwd=root).returncode == 0


class RunawayModel:
    """A lazy stream of notes that never ends on its own; counts how many
    events were actually generated."""

    def __init__(self, sleep_s=0.0):
        self.generated = 0
        self.closed = False
        self.sleep_s = sleep_s

    def transcribe(self, audio, instruments=None):
        import time

        i = 0
        try:
            while True:
                s = NoteStartEvent(60, 0.6 + (i % 300) * 0.01, i, "drums")
                self.generated += 1
                yield s
                yield NoteEndEvent(s.start_time + 0.05, s)
                i += 1
                if self.sleep_s:
                    time.sleep(self.sleep_s)
        finally:
            self.closed = True


def _window_req(seconds=5.0):
    sr = 16000
    return AnalysisRequest(
        audio=np.zeros(int(seconds * sr), np.float32), sr=sr, t0_s=8.0, span_s=4.0,
        context_lead_s=0.5, context_tail_s=0.5, source_epoch=2, revision=7,
    )


def test_note_cap_stops_decoding_and_discards_window(monkeypatch):
    monkeypatch.setattr(notes_engine, "NOTE_CAP_PER_WINDOW", 50)
    model = RunawayModel()
    eng = NotesEngine("small", model)
    res = eng.analyze(_window_req())
    assert res.truncated is True and list(res.items) == []
    assert (res.range_start_s, res.span_s, res.source_epoch, res.revision) == (8.0, 4.0, 2, 7)
    assert model.generated == 51  # stopped on the first start past the cap
    assert model.closed


def test_note_cap_scales_with_clip_length(monkeypatch):
    monkeypatch.setattr(notes_engine, "NOTE_CAP_PER_WINDOW", 50)
    model = RunawayModel()
    NotesEngine("small", model).analyze(_window_req(seconds=12.0))  # 3 chunks
    assert model.generated == 151


def test_wall_budget_stops_decoding_and_discards_window(monkeypatch):
    monkeypatch.setattr(notes_engine, "WALL_BUDGET_S", 0.05)
    model = RunawayModel(sleep_s=0.01)
    res = NotesEngine("small", model).analyze(_window_req())
    assert res.truncated is True and list(res.items) == []
    assert model.generated < notes_engine.NOTE_CAP_PER_WINDOW
    assert model.closed


def test_guards_quiet_on_normal_window():
    events = [ev for i in range(10) for ev in _note(i, 60 + i, 1.0 + i * 0.1, 1.05 + i * 0.1)]
    res = NotesEngine("small", FakeModel(events)).analyze(_window_req())
    assert res.truncated is False and len(res.items) == 10
