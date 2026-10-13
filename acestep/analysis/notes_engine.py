"""The ``notes`` analyzer: audio -> MIDI notes via MuScriptor.

Registers :data:`NOTES` at import. Torch-free at import: MuScriptor and
torch load inside :func:`make_notes_engine` and :meth:`NotesEngine.analyze`,
so the wire layer can read the spec on a pod without the optional
dependency (``pip install demon[analysis-notes]``).

One window is the cell plus ``context_s`` on each side, one MuScriptor
5 s chunk at the defaults. The context is transcribed but only notes whose
onset lies inside the cell are reported, in canvas seconds. The same
engine serves the plugin's one-shot (:mod:`acestep.analysis.midi_transcribe`),
which sends the whole clip as one cell with no context.

Env:
  DEMON_ANALYSIS_NOTES_MODEL  small | medium | large   (default small)
  DEMON_MIDI_DISABLE          set to anything to report the analyzer absent
"""

from __future__ import annotations

import importlib.util
import os
import time
from typing import Any, Iterable, NamedTuple, Optional

import numpy as np
from loguru import logger

from acestep.analysis.engine import AnalysisRequest, AnalysisResult
from acestep.analysis.registry import AnalyzerOption, AnalyzerSpec, register

#: Duration given to a note the model opened and never closed, clamped
#: to the audio it saw.
UNCLOSED_NOTE_S = 0.25

LICENCE = "MuScriptor weights are gated on Hugging Face (CC BY-NC 4.0)."


class PairedNote(NamedTuple):
    """One note in seconds from the start of the analysed audio."""

    start_s: float
    end_s: float
    pitch: int
    instrument: str


def pair_events(events: Iterable[Any], duration_s: float) -> list[PairedNote]:
    """Pair MuScriptor ``NoteStartEvent`` / ``NoteEndEvent`` by index into
    notes sorted by ``(start_s, pitch)``.

    The decoder can place an offset past the audio it saw (last chunk
    padding), so offsets clamp to ``duration_s``; a start the model never
    closed ends at ``min(duration_s, start + UNCLOSED_NOTE_S)``. Both are
    guarded with ``max(start, ...)`` so a note never ends before it
    starts. Other event types (progress anchors) are ignored. Events are
    matched by class name, so no MuScriptor import is needed.
    """
    starts: dict = {}
    notes: list[PairedNote] = []
    for ev in events:
        name = type(ev).__name__
        if name == "NoteStartEvent":
            starts[ev.index] = ev
        elif name == "NoteEndEvent":
            s = ev.start_event
            starts.pop(getattr(s, "index", None), None)
            st = float(s.start_time)
            notes.append(PairedNote(
                st, max(st, min(float(ev.end_time), duration_s)), int(s.pitch), str(s.instrument),
            ))
    for s in starts.values():
        st = float(s.start_time)
        notes.append(PairedNote(
            st, max(st, min(duration_s, st + UNCLOSED_NOTE_S)), int(s.pitch), str(s.instrument),
        ))
    notes.sort(key=lambda n: (n.start_s, n.pitch))
    return notes


def _load_model(size: str, device: str):
    """Load the MuScriptor model (the only place weights are touched)."""
    from muscriptor.transcription_model import TranscriptionModel

    return TranscriptionModel.load_model(size, device=device, dtype="float16")


class NotesEngine:
    """A loaded MuScriptor model serving the ``notes`` analyzer."""

    name = "notes"

    def __init__(self, size: str, model: Any):
        self.size = size
        self._model = model

    def analyze(self, request: AnalysisRequest) -> AnalysisResult:
        """Transcribe the request's window and return the notes whose onset
        lies inside the cell, in canvas seconds.

        A side with no context (``context_lead_s`` / ``context_tail_s`` of
        0: a canvas edge, or the one-shot's whole clip) is not filtered,
        so nothing at the very edge of the audio is lost.
        """
        if self._model is None:
            raise RuntimeError("notes engine is unloaded")
        import torch

        audio = np.asarray(request.audio, dtype=np.float32)
        if audio.ndim == 2:
            audio = audio.mean(axis=0, dtype=np.float32)
        if audio.ndim != 1:
            raise ValueError(f"notes engine wants mono audio, got shape {audio.shape}")
        if request.sr < NOTES.min_sr:
            raise ValueError(f"notes engine needs sr >= {NOTES.min_sr}, got {request.sr}")
        duration_s = audio.shape[0] / float(request.sr)
        instruments = request.options.get("instruments") or None

        t0 = time.perf_counter()
        wav = torch.from_numpy(np.ascontiguousarray(audio))[None, :]
        events = self._model.transcribe((wav, int(request.sr)), instruments=instruments)
        notes = pair_events(events, duration_s)
        wall_s = time.perf_counter() - t0

        lo = request.context_lead_s
        hi = request.context_lead_s + request.span_s
        offset = request.t0_s - request.context_lead_s
        items = [
            {
                "start_s": n.start_s + offset,
                "end_s": n.end_s + offset,
                "pitch": n.pitch,
                "instrument": n.instrument,
            }
            for n in notes
            if (request.context_lead_s <= 0 or n.start_s >= lo)
            and (request.context_tail_s <= 0 or n.start_s < hi)
        ]
        return AnalysisResult(
            kind="note",
            range_start_s=request.t0_s,
            span_s=request.span_s,
            items=items,
            source_epoch=request.source_epoch,
            revision=request.revision,
            wall_s=wall_s,
        )

    def unload(self) -> None:
        """Drop the model and return its cached VRAM. Safe to call twice."""
        if self._model is None:
            return
        self._model = None
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as exc:  # noqa: BLE001 - unload is best effort
            logger.debug("notes_engine_empty_cache_failed error={}", exc)


def make_notes_engine(size: str, device: Any) -> NotesEngine:
    """``AnalyzerSpec.make_engine`` for ``notes``."""
    t0 = time.perf_counter()
    model = _load_model(size, str(device))
    logger.info("notes_engine_loaded size={} wall_s={:.1f}", size, time.perf_counter() - t0)
    return NotesEngine(size, model)


def probe_notes() -> Optional[str]:
    """``None`` when MuScriptor is importable, not disabled and the
    configured size is valid; else a user-facing reason. Never loads
    weights."""
    if os.environ.get("DEMON_MIDI_DISABLE"):
        return "notes analysis is disabled on this server (DEMON_MIDI_DISABLE)"
    if importlib.util.find_spec("muscriptor") is None:
        return "MuScriptor is not installed on this server (extra: analysis-notes)"
    try:
        NOTES.resolve_size()
    except ValueError as exc:
        return str(exc)
    return None


def _validate_instruments(names: list) -> Optional[list]:
    """Exact MuScriptor instrument group names; an empty list means no
    constraint. Raises ``ValueError`` listing the valid names."""
    if not names:
        return None
    try:
        from muscriptor.tokenizer.mt3 import instrument_group_from_names
    except ImportError:
        raise ValueError("MuScriptor is not installed on this server") from None
    instrument_group_from_names(names)
    return list(names)


NOTES = register(AnalyzerSpec(
    name="notes",
    display_name="Notes (MuScriptor)",
    engine_name="muscriptor",
    item_kind="note",
    make_engine=make_notes_engine,
    probe=probe_notes,
    licence=LICENCE,
    sizes=("small", "medium", "large"),
    default_size="small",
    size_env="DEMON_ANALYSIS_NOTES_MODEL",
    window_s=4.0,
    context_s=0.5,
    min_sr=16000,
    placement="gpu",
    resident_gb={"small": 0.3, "medium": 0.8, "large": 3.0},
    options=(
        AnalyzerOption(
            "size", "str", default=None, bounds=("small", "medium", "large"),
            help="Model size; default from DEMON_ANALYSIS_NOTES_MODEL, then small.",
        ),
        AnalyzerOption(
            "instruments", "str_list", default=None,
            help="Restrict output to these MuScriptor instrument names.",
            validate=_validate_instruments,
        ),
        AnalyzerOption(
            "horizon_s", "float", default=8.0, bounds=(4.0, 30.0),
            help="How far ahead of the playhead ring canvases are analysed (s).",
        ),
    ),
))
