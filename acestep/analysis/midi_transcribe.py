"""On-demand audio -> MIDI notes via MuScriptor, for the ``midi_transcribe``
wire command (the plugin's "drag MIDI out").

Shape: the client uploads the exact audio it is about to drag out (the same
PCM frame format every other upload uses), the server transcribes it once
and answers ``midi_notes`` with absolute-seconds note events. Nothing is
kept per session; the model is a process-wide singleton loaded on first
use and serialized behind a lock (one GPU model, one transcription at a
time — a second request queues behind the first).

Why on demand rather than a live per-session transcriber (the spike in
PR #345): the drag needs the notes for the clip the user hears in the
plugin, which may be a frozen result rather than the live canvas, and a
one-shot batch transcription of a 30 s clip costs a few seconds on a
5090 — no continuous GPU tax on every session.

Dependencies are optional (``muscriptor`` is not in pyproject): a pod
without them reports ``midi_transcribe: false`` in ``ready.capabilities``
and rejects the command with ``midi_failed``. The MuScriptor weights are
gated on Hugging Face (CC BY-NC 4.0) — the pod's HF token must have
accepted the terms for the configured size.

Env:
  DEMON_MIDI_MODEL   small | medium | large   (default small)
  DEMON_MIDI_DISABLE set to anything to report the capability as absent
"""

from __future__ import annotations

import importlib.util
import os
import threading
import time
from dataclasses import dataclass

import numpy as np
from loguru import logger

SAMPLE_RATE = 48000
_VALID_SIZES = ("small", "medium", "large")


def _model_size() -> str:
    size = os.environ.get("DEMON_MIDI_MODEL", "small").strip().lower()
    return size if size in _VALID_SIZES else "small"


def transcriber_available() -> bool:
    """Cheap capability probe: deps importable and not disabled. Does NOT
    load weights (that happens on the first request)."""
    if os.environ.get("DEMON_MIDI_DISABLE"):
        return False
    return importlib.util.find_spec("muscriptor") is not None


@dataclass(frozen=True)
class MidiNote:
    start_s: float
    end_s: float
    pitch: int
    instrument: str

    def to_wire(self) -> dict:
        return {
            "start_s": round(self.start_s, 4),
            "end_s": round(self.end_s, 4),
            "pitch": int(self.pitch),
            "instrument": self.instrument,
        }


@dataclass(frozen=True)
class MidiTranscription:
    notes: list
    model: str
    duration_s: float
    wall_s: float


class MidiTranscriber:
    """Process-wide MuScriptor holder. ``get()`` for the singleton."""

    _instance = None
    _instance_lock = threading.Lock()

    def __init__(self, size: str):
        self.size = size
        self._model = None
        self._lock = threading.Lock()

    @classmethod
    def get(cls) -> "MidiTranscriber":
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = cls(_model_size())
            return cls._instance

    def _ensure_model(self):
        if self._model is not None:
            return self._model
        from muscriptor.transcription_model import TranscriptionModel

        t0 = time.perf_counter()
        self._model = TranscriptionModel.load_model(
            self.size, device="cuda", dtype="float16"
        )
        logger.info(
            "midi_transcriber_loaded size={} wall_s={:.1f}",
            self.size, time.perf_counter() - t0,
        )
        return self._model

    def transcribe(self, waveform, sample_rate: int = SAMPLE_RATE) -> MidiTranscription:
        """``waveform``: float32 array/tensor shaped [channels, samples]
        (what ``_decode_audio_msg`` returns). Returns paired notes in
        absolute seconds from the start of the buffer."""
        import torch

        wf = waveform if isinstance(waveform, torch.Tensor) else torch.from_numpy(np.asarray(waveform))
        if wf.ndim == 1:
            wf = wf[None, :]
        mono = wf.float().mean(dim=0, keepdim=True).cpu()
        duration_s = mono.shape[1] / float(sample_rate)

        with self._lock:
            model = self._ensure_model()
            t0 = time.perf_counter()
            starts: dict = {}
            notes: list = []
            for ev in model.transcribe((mono, sample_rate)):
                name = type(ev).__name__
                if name == "NoteStartEvent":
                    starts[ev.index] = ev
                elif name == "NoteEndEvent":
                    s = ev.start_event
                    starts.pop(getattr(s, "index", None), None)
                    # The decoder can place an offset past the audio it saw
                    # (last chunk padding); the clip ends where the clip ends.
                    st = float(s.start_time)
                    notes.append(MidiNote(
                        st, max(st, min(float(ev.end_time), duration_s)),
                        int(s.pitch), str(s.instrument),
                    ))
            # Starts the model never closed: clamp to the buffer end.
            for s in starts.values():
                st = float(s.start_time)
                notes.append(MidiNote(
                    st, min(duration_s, st + 0.25), int(s.pitch), str(s.instrument),
                ))
            wall_s = time.perf_counter() - t0

        notes.sort(key=lambda n: (n.start_s, n.pitch))
        logger.info(
            "midi_transcribed size={} audio_s={:.1f} wall_s={:.2f} notes={}",
            self.size, duration_s, wall_s, len(notes),
        )
        return MidiTranscription(
            notes=notes, model=self.size, duration_s=duration_s, wall_s=wall_s,
        )
