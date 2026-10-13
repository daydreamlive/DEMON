"""The plugin's one-shot ``midi_transcribe`` served through the analysis
host: same notes, same surface as the standalone transcriber it replaced.

The expectation below was recorded from the pre-host implementation
(base d38fa48c) fed the same fake event stream."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import numpy as np
import pytest

from acestep.analysis import host as host_mod
from acestep.analysis import midi_transcribe, notes_engine
from acestep.analysis.host import AnalysisHost
from acestep.analysis.midi_transcribe import MidiNote, MidiTranscriber


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


def _stream():
    a = NoteStartEvent(60, 0.12, 0, "acoustic_piano")
    b = NoteStartEvent(43, 0.12, 1, "acoustic_bass")
    c = NoteStartEvent(38, 0.5, 2, "drums")
    d = NoteStartEvent(72, 1.25, 3, "acoustic_piano")
    e = NoteStartEvent(36, 1.75, 4, "drums")  # never closed
    f = NoteStartEvent(60, 0.12, 5, "electric_piano")
    return [
        ProgressEvent(0, 1), a, b, f, c, NoteEndEvent(0.55, c), NoteEndEvent(0.8, a), d, e,
        NoteEndEvent(1.6, b), NoteEndEvent(9.0, d), NoteEndEvent(0.9, f), ProgressEvent(1, 1),
    ]


RECORDED = [
    {"start_s": 0.12, "end_s": 1.6, "pitch": 43, "instrument": "acoustic_bass"},
    {"start_s": 0.12, "end_s": 0.8, "pitch": 60, "instrument": "acoustic_piano"},
    {"start_s": 0.12, "end_s": 0.9, "pitch": 60, "instrument": "electric_piano"},
    {"start_s": 0.5, "end_s": 0.55, "pitch": 38, "instrument": "drums"},
    {"start_s": 1.25, "end_s": 2.0, "pitch": 72, "instrument": "acoustic_piano"},
    {"start_s": 1.75, "end_s": 2.0, "pitch": 36, "instrument": "drums"},
]


class FakeModel:
    def __init__(self):
        self.calls = []

    def transcribe(self, audio, instruments=None):
        wav, sr = audio
        self.calls.append((tuple(wav.shape), sr, instruments, wav.numpy().copy()))
        yield from _stream()


@pytest.fixture
def fake_host(monkeypatch):
    """A fresh process host with no VRAM telemetry (never touches CUDA),
    the notes probe forced available, and the MuScriptor loader stubbed."""
    loads = []
    models = []

    def fake_load(size, device):
        loads.append((size, device))
        models.append(FakeModel())
        return models[-1]

    monkeypatch.setattr(notes_engine, "_load_model", fake_load)
    spec = dataclasses.replace(notes_engine.NOTES, probe=lambda: None)
    host = AnalysisHost(spec_lookup=lambda name: spec, vram_probe=lambda: None)
    monkeypatch.setattr(host_mod, "_HOST", host)
    monkeypatch.setattr(MidiTranscriber, "_instance", None)
    yield host, loads, models
    host.shutdown()


def test_transcribe_parity_with_recorded(fake_host, monkeypatch):
    host, loads, models = fake_host
    monkeypatch.delenv("DEMON_MIDI_MODEL", raising=False)
    wf = np.zeros((2, 96000), np.float32)
    wf[0, :10] = 1.0
    result = MidiTranscriber.get().transcribe(wf)
    assert [n.to_wire() for n in result.notes] == RECORDED
    assert all(isinstance(n, MidiNote) for n in result.notes)
    assert result.model == "small"
    assert result.duration_s == 2.0
    assert result.wall_s >= 0.0
    assert loads == [("small", "cuda")]
    shape, sr, instruments, audio = models[0].calls[0]
    assert (shape, sr, instruments) == ((1, 96000), 48000, None)
    assert audio[0, 0] == pytest.approx(0.5) and audio[0, 10] == 0.0  # mono downmix


def test_transcribe_accepts_tensor_and_1d(fake_host):
    import torch

    t = MidiTranscriber("small")
    r1 = t.transcribe(torch.zeros(96000))
    r2 = t.transcribe(torch.zeros(1, 96000), sample_rate=48000)
    assert [n.to_wire() for n in r1.notes] == RECORDED == [n.to_wire() for n in r2.notes]


def test_size_from_env_and_shared_instance(fake_host, monkeypatch):
    host, loads, _ = fake_host
    monkeypatch.setenv("DEMON_MIDI_MODEL", "Medium")
    t = MidiTranscriber.get()
    assert t.size == "medium"
    t.transcribe(np.zeros((1, 4800), np.float32))
    t.transcribe(np.zeros((1, 4800), np.float32))
    assert loads == [("medium", "cuda")]  # one engine, reused
    assert host.loaded() == [("notes", "medium")]
    monkeypatch.setenv("DEMON_MIDI_MODEL", "bogus")
    assert midi_transcribe._model_size() == "small"


def test_transcriber_available(monkeypatch):
    import importlib.util

    monkeypatch.setenv("DEMON_MIDI_DISABLE", "1")
    assert midi_transcribe.transcriber_available() is False
    monkeypatch.delenv("DEMON_MIDI_DISABLE")
    real = importlib.util.find_spec
    monkeypatch.setattr(
        importlib.util, "find_spec",
        lambda name, *a: None if name == "muscriptor" else real(name, *a),
    )
    assert midi_transcribe.transcriber_available() is False


def test_to_wire_rounds():
    n = MidiNote(0.123456, 0.987654, 60.0, "drums")
    assert n.to_wire() == {"start_s": 0.1235, "end_s": 0.9877, "pitch": 60, "instrument": "drums"}


def test_one_shot_is_unbounded(fake_host, monkeypatch):
    """The one-shot asks the engine for every note: the runaway guards
    must not apply even when the clip would trip them."""
    host, _loads, _models = fake_host
    monkeypatch.setattr(notes_engine, "NOTE_CAP_PER_WINDOW", 1)
    monkeypatch.setattr(notes_engine, "WALL_BUDGET_S", 0.0)
    result = MidiTranscriber("small").transcribe(np.zeros((1, 96000), np.float32))
    assert [n.to_wire() for n in result.notes] == RECORDED


def test_truncated_one_shot_raises(fake_host, monkeypatch):
    """A truncated result never reaches the client as an empty list."""
    from acestep.analysis.engine import AnalysisResult

    host, _loads, _models = fake_host
    seen = []

    def fake_run(name, size, request):
        seen.append(dict(request.options))
        return AnalysisResult(
            kind="note", range_start_s=0.0, span_s=2.0, items=[],
            source_epoch=0, revision=0, wall_s=0.1, truncated=True,
        )

    monkeypatch.setattr(host, "run_one_shot", fake_run)
    with pytest.raises(RuntimeError, match="transcription abandoned"):
        MidiTranscriber("small").transcribe(np.zeros((1, 96000), np.float32))
    assert seen == [{"bounded": False}]
