"""The analysis seam's engine contract: models that LISTEN to the canvas.

The generator seam (:mod:`acestep.streaming.generator_backend`) owns the
canvas, the knobs and the LoRAs. An analysis engine never writes audio;
it is handed a snapshot of canvas audio and returns items (notes, beats,
chords) for a time range. Everything here is torch-free and importable
by the wire layer.

Vocabulary:

- **item kind**: the wire shape of one item. Kinds are a closed, versioned
  set registered once in the protocol (``note`` first). A new kind is a
  deliberate protocol + codegen edit, never a schema projected from an
  engine.
- **window**: the audio an engine wants per analysis, ``window_s`` of
  cell plus ``context_s`` on each side (the context is analysed but only
  onsets inside the cell are reported).
- **result**: the items plus the output range they replace. An engine
  whose corrections span more than its cell (beat grids, chord changes)
  returns the wider range; the tap applies it as a range replace.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Protocol, Sequence, runtime_checkable

import numpy as np

#: Registered wire item kinds. Keep in sync with the protocol registry's
#: ``analysis_update.items`` tagged union.
ITEM_KINDS = ("note",)

#: ``note`` item: canvas seconds, MIDI pitch, MuScriptor instrument name.
#: No velocity (the first engine has none; a later kind may add it).
NOTE_FIELDS = ("start_s", "end_s", "pitch", "instrument")


class EngineBusy(Exception):
    """The engine's model is held by a higher-priority caller (an
    interactive one-shot). The cell stays dirty and is retried later."""


class EngineUnavailable(Exception):
    """The engine cannot run here: dependency missing, weights gated,
    size not allowed, VRAM admission refused. Message is user-facing."""


@dataclass(frozen=True)
class AnalysisRequest:
    """One analysis of one window.

    ``audio`` is mono float32 at ``sr`` covering
    ``[t0_s - context_s, t0_s + span_s + context_s)`` clipped to the canvas;
    ``context_lead_s`` / ``context_tail_s`` say how much of that is context
    (less than ``context_s`` at a canvas edge). ``t0_s`` and ``span_s`` are
    the cell in canvas seconds. ``source_epoch`` and ``revision`` are
    opaque to the engine and echoed in the result so the tap can discard
    stale work.
    """

    audio: np.ndarray
    sr: int
    t0_s: float
    span_s: float
    context_lead_s: float
    context_tail_s: float
    source_epoch: int
    revision: int
    options: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AnalysisResult:
    """Items for ``[range_start_s, range_start_s + span_s)`` in canvas
    seconds, replacing whatever the tap previously held for that range.
    ``kind`` names the item kind of every entry in ``items``. ``wall_s``
    is the engine's own wall time for telemetry. ``truncated`` marks a
    window the engine abandoned (runaway decode); ``items`` is then empty
    and the range is still replaced, so nothing stale survives. A
    streaming engine (``options["stream"]``) keeps a ledger across hops:
    on a truncated hop its ``items`` are the notes its ledger already
    held for the range (possibly none), never anything from the discarded
    hop.

    ``open_items`` (additive, streaming engines only) are items that have
    started inside the range but not ended yet, with ``end_s`` set to the
    frontier; the tap may publish them as provisional. ``items`` then
    holds only closed ones."""

    kind: str
    range_start_s: float
    span_s: float
    items: Sequence[Mapping[str, Any]]
    source_epoch: int
    revision: int
    wall_s: float = 0.0
    truncated: bool = False
    open_items: Sequence[Mapping[str, Any]] = ()


@runtime_checkable
class AnalysisEngine(Protocol):
    """A loaded listening model. Instances are owned by the process-wide
    host (:mod:`acestep.analysis.host`), never by a session.

    ``analyze`` is called on the host's worker thread, one call in flight
    per process. It raises :class:`EngineBusy` when its model is held by
    a one-shot caller, and any other exception is logged by the host and
    turns into an ``analysis_state`` error for the tap that asked.
    """

    #: Analyzer name this engine serves (``AnalyzerSpec.name``).
    name: str
    #: Model size actually loaded (``small`` ...), for ``analysis_state``.
    size: str

    def analyze(self, request: AnalysisRequest) -> AnalysisResult: ...

    def unload(self) -> None:
        """Release the model. The host calls it at shutdown (and later,
        on idle). Safe to call twice."""
        ...


class AnalysisSubmitter(Protocol):
    """What a session-side tap needs from the process-wide host: submit a
    request at live priority and get the result through a callback on the
    host's worker thread. ``cancel_token`` lets the tap withdraw queued
    work (on disable, swap, close) without waiting for it.

    The host keeps ONE bounded queue per process, one inference in flight,
    live work below one-shot work; a full queue drops the oldest live
    request and reports it through ``on_dropped``.
    """

    def submit(
        self,
        analyzer: str,
        size: Optional[str],
        request: AnalysisRequest,
        on_result: "callable[[AnalysisResult], None]",
        on_error: "callable[[BaseException], None]",
        cancel_token: object,
    ) -> None: ...

    def cancel(self, cancel_token: object) -> int:
        """Withdraw every queued request carrying ``cancel_token``;
        returns how many were withdrawn. Work already in flight finishes
        and its result is still delivered (the tap checks epoch/revision)."""
        ...

    def engine_info(self, analyzer: str, size: Optional[str]) -> Mapping[str, Any]:
        """``{"name", "size", "loaded": bool, "licence", "engine",
        "display_name"}`` without loading anything (``engine`` is
        ``AnalyzerSpec.engine_id``, for ``analysis_state.engine``). Raises :class:`EngineUnavailable` when the engine cannot
        run here (message user-facing)."""
        ...
