"""On-demand audio -> MIDI notes via MuScriptor, for the ``midi_transcribe``
wire command (the plugin's "drag MIDI out").

Shape: the client uploads the exact audio it is about to drag out (the same
PCM frame format every other upload uses), the server transcribes it once
and answers ``midi_notes`` with absolute-seconds note events. Nothing is
kept per session; the model is the ``notes`` engine of the process-wide
analysis host (:mod:`acestep.analysis.host`), loaded on first use. The
host runs one inference at a time and puts a one-shot ahead of any queued
live analysis; a second one-shot queues behind the first.

Why on demand rather than a live per-session transcriber (the spike in
PR #345): the drag needs the notes for the clip the user hears in the
plugin, which may be a frozen result rather than the live canvas, and a
one-shot batch transcription of a 30 s clip costs a few seconds on a
5090 — no continuous GPU tax on every session.

Dependencies are optional (the ``analysis-notes`` extra): a pod
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
    """The one-shot's front door. ``get()`` for the singleton.

    The model itself lives in the process-wide analysis host
    (:mod:`acestep.analysis.host`) as the ``notes`` engine of size
    ``DEMON_MIDI_MODEL``; when that equals the live tap's size the two
    share one loaded instance. The one-shot runs ahead of queued live
    analysis and blocks until done.
    """

    _instance = None
    _instance_lock = threading.Lock()

    def __init__(self, size: str):
        self.size = size

    @classmethod
    def get(cls) -> "MidiTranscriber":
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = cls(_model_size())
            return cls._instance

    def transcribe(self, waveform, sample_rate: int = SAMPLE_RATE) -> MidiTranscription:
        """``waveform``: float32 array/tensor shaped [channels, samples]
        (what ``_decode_audio_msg`` returns). Returns paired notes in
        absolute seconds from the start of the buffer.

        The request is unbounded (no runaway guards): the drag wants every
        note. Raises ``RuntimeError`` if a result still comes back
        truncated, so the adapter answers ``midi_failed`` rather than an
        empty clip."""
        import torch

        from acestep.analysis.engine import AnalysisRequest
        from acestep.analysis.host import get_host

        wf = waveform if isinstance(waveform, torch.Tensor) else torch.from_numpy(np.asarray(waveform))
        if wf.ndim == 1:
            wf = wf[None, :]
        mono = wf.float().mean(dim=0).cpu().numpy()
        duration_s = mono.shape[0] / float(sample_rate)

        request = AnalysisRequest(
            audio=mono,
            sr=int(sample_rate),
            t0_s=0.0,
            span_s=duration_s,
            context_lead_s=0.0,
            context_tail_s=0.0,
            source_epoch=0,
            revision=0,
            options={"bounded": False},
        )
        result = get_host().run_one_shot("notes", self.size, request)
        if result.truncated:
            # An empty list would read as "no notes"; fail the command instead.
            raise RuntimeError("transcription abandoned")
        notes = [
            MidiNote(float(i["start_s"]), float(i["end_s"]), int(i["pitch"]), str(i["instrument"]))
            for i in result.items
        ]
        logger.info(
            "midi_transcribed size={} audio_s={:.1f} wall_s={:.2f} notes={}",
            self.size, duration_s, result.wall_s, len(notes),
        )
        return MidiTranscription(
            notes=notes, model=self.size, duration_s=duration_s, wall_s=result.wall_s,
        )
