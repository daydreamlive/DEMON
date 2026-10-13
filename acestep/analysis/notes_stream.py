"""Streaming MuScriptor inference: decode only the new tail of each hop.

MuScriptor's batch API re-decodes every note of a 5 s window on every
call, at about 3 ms per token. This module keeps a per-take **note
ledger** in canvas time and, on each hop, teacher-forces the ledger's
notes inside the window as the token prompt (re-timed to the window
start, with the tie prologue for notes sounding at the window start,
the same thing ``TranscriptionModel.transcribe`` does across chunks), so
the model only generates the tokens past what is already settled.

The ledger may force a window's prologue only once it has heard the
model there: the model opens every window cut mid-music with a tie
section naming the notes already sounding, and forcing an empty one
contradicts the audio (the model answers EOS at once). So the first hop
of a take and a window start beyond what the ledger has analysed (a
stall gap) send an EMPTY prompt and decode the whole window. A window
start inside analysed audio where nothing was sounding forces the empty
tie section: there it is the truth; the model's own tie section is adopted into
the ledger as notes sounding since the window start (``tied_in``: never
reported as onsets, but they get their ends and become the next hop's
tie keys).

Each hop stops decoding at the first shift past the commit point (nothing
later can commit) or at a backwards shift (the model re-transcribing);
an onset within :data:`DEDUPE_TICKS` of a ledger note of the same key is
the model restating a settled note across the boundary and is dropped;
a hop that hits its wall budget or token cap keeps what it decoded up to
the last complete tick. A loop detector stops a hop whose new onsets
repeat one key at near-equal spacing or exceed a rate per second of new
audio, and keeps nothing from where the loop started.

Pure python and torch-free at import. The model sits behind a narrow
backend (:class:`MuScriptorStreamBackend` wraps a loaded
``TranscriptionModel``; tests use a fake that yields scripted tokens),
so the ledger, the prompt builder, the token encoder and the hop guards
are all proven without a GPU.

Token layout (``muscriptor.tokenizer.notes.build_event_vocab``, rebuilt
here by :func:`build_vocab`): ``PAD EOS UNK``, ``shift 0..1000`` (an
absolute tick from the chunk start at 100 frames/s, not a delta),
``pitch 0..127``, ``velocity 0..1`` (1 onset, 0 offset), ``tie``,
``program 0..129``, ``drum 0..127``. A chunk is a tie prologue
(``program pitch... tie``, programs sorted, each program once per run)
then note events sorted by ``(tick, is_drum, program, velocity, pitch)``;
``shift`` only when the tick advances, ``program`` / ``velocity`` only
when they change (the program state carries over from the prologue, the
velocity state resets at ``tie``), a drum hit is a lone ``drum`` token.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Iterator, Optional, Protocol, Sequence

import numpy as np
from loguru import logger

#: MuScriptor's token frame rate (ticks per second).
FRAME_RATE = 100
#: Longest window the model is trained on (``_SEGMENT_DURATION``).
WINDOW_S = 5.0
#: Model sample rate.
MODEL_SR = 16000
#: Most tokens generated past the prompt on one hop. Set from the live run.
NEW_TOKEN_CAP = 256
#: Same, for a hop that decodes the whole window from an empty prompt (the
#: first hop of a take, or a window start the ledger cannot speak for). A
#: dense 5 s lofi window is ~330 tokens. Set from the live run.
FULL_TOKEN_CAP = 1024
#: Onsets later than ``frontier_s - GUARD_S`` are re-decoded next hop
#: (the model has too little audio after them). Set from the live run.
GUARD_S = 0.15
#: Most wall time one hop may take; past it the hop's tail is discarded
#: and the ledger kept. Set from the live run.
HOP_WALL_BUDGET_S = 0.5
#: Wall budget of a whole-window hop (a dense 5 s window took 1.06 s on
#: the small model, eager). Set from the live run.
FULL_HOP_WALL_BUDGET_S = 2.0
#: An open note older than this is closed at the commit point (MuScriptor
#: training shortens notes over 10 s; keeps the tie prologue bounded).
MAX_OPEN_NOTE_S = 10.0
#: Loop detector: this many onsets of one (program, pitch) in a hop's
#: tail at near-equal spacing is the model looping. Set from the live run.
LOOP_REPEATS = 4
#: Spacing tolerance of those repeats, in ticks.
LOOP_SPACING_TOL = 2
#: Loop detector: more onsets than this per second of new audio in a hop's
#: tail is a runaway (a dense 5 s lofi window is ~24/s). Set from the live run.
LOOP_ONSETS_PER_S = 25.0
#: The rate rule's floor is :data:`LOOP_REPEATS`, deliberately low: on a
#: short hop it also defers a burst of onsets at the frontier (a chord, a
#: fill, or the model guessing with little audio after the onset) until a
#: later hop has heard more. Measured on the small model: a floor of 24
#: committed those bursts at once and cut stream-vs-batch F1 from 0.41 to
#: 0.27 (techno) and 0.30 to 0.25 (lofi), because a wrong commit feeds the
#: next hop's forced prefix. A deferred burst is not lost, it waits.
#: Two onsets of one (program, pitch) this close (ticks, 30 ms) are one
#: note restated across the commit boundary, never a real retrigger.
DEDUPE_TICKS = 3
#: Duration of a drum hit (``MINIMUM_NOTE_DURATION_SEC``), in ticks.
DRUM_TICKS = 1
#: MIDI program the tokenizer uses for drums.
DRUM_PROGRAM = 128

_SPECIAL = ("PAD", "EOS", "UNK")
_MAX_SHIFT_STEPS = 1001


def build_vocab(max_shift_steps: int = _MAX_SHIFT_STEPS) -> list[tuple[str, int]]:
    """MuScriptor's token-index -> ``(type, value)`` table (mirrors
    ``muscriptor.tokenizer.notes.build_event_vocab``; a test pins the two
    together where MuScriptor is installed)."""
    ranges = [(t, 0, 0) for t in _SPECIAL] + [
        ("shift", 0, max_shift_steps - 1),
        ("pitch", 0, 127),
        ("velocity", 0, 1),
        ("tie", 0, 0),
        ("program", 0, 129),
        ("drum", 0, 127),
    ]
    return [(t, v) for t, lo, hi in ranges for v in range(lo, hi + 1)]


# ---------------------------------------------------------------------------
# Encoder (note events -> tokens) and decoder (tokens -> note actions)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WindowEvent:
    """One note event in window ticks: ``velocity`` 1 onset, 0 offset."""

    tick: int
    is_drum: bool
    program: int
    velocity: int
    pitch: int

    def sort_key(self):
        return (self.tick, self.is_drum, self.program, self.velocity, self.pitch)


def encode_window(tie_keys: Iterable[tuple[int, int]], events: Iterable[WindowEvent],
                  index: dict) -> list[int]:
    """Token ids for one chunk: the tie prologue for ``tie_keys``
    (``(program, pitch)`` sounding at the window start), then ``events``
    in the model's order. ``index`` maps ``(type, value)`` to a token id."""
    out: list[int] = []
    program_state: Optional[int] = None
    for program, pitch in sorted(set(tie_keys)):
        if program != program_state:
            out.append(index[("program", program)])
            program_state = program
        out.append(index[("pitch", pitch)])
    out.append(index[("tie", 0)])
    tick_state = 0
    velocity_state: Optional[int] = None
    for ev in sorted(events, key=WindowEvent.sort_key):
        if ev.tick < tick_state:
            raise ValueError(f"event tick {ev.tick} before {tick_state}")
        if ev.tick > tick_state:
            if ev.tick >= _MAX_SHIFT_STEPS:
                raise ValueError(f"tick {ev.tick} past the shift range")
            out.append(index[("shift", ev.tick)])
            tick_state = ev.tick
        if ev.is_drum:
            if ev.velocity:
                out.append(index[("drum", ev.pitch)])
            continue
        if ev.program != program_state:
            out.append(index[("program", ev.program)])
            program_state = ev.program
        if ev.velocity != velocity_state:
            out.append(index[("velocity", ev.velocity)])
            velocity_state = ev.velocity
        out.append(index[("pitch", ev.pitch)])
    return out


@dataclass(frozen=True)
class Action:
    """A decoded note action in window ticks: ``kind`` is ``start``,
    ``end`` or ``drum``."""

    kind: str
    program: int
    pitch: int
    tick: int


class WindowDecoder:
    """One chunk's decode state machine, the rules of MuScriptor's
    ``OpenNoteTracker`` for a single chunk (seek 0, no next-seek limit),
    seeded with the notes open before the chunk (``open_keys``) so the
    offsets of tied notes decode. Feed token ids one at a time."""

    def __init__(self, vocab: Sequence[tuple[str, int]], open_keys: Iterable[tuple[int, int]] = ()):
        self._vocab = vocab
        self._open: set[tuple[int, int]] = set(open_keys)
        self._tick = 0
        self._program: Optional[int] = None
        self._velocity: Optional[int] = None
        self._in_prologue = True
        self._skip_rest = False
        self._tie_set: set[tuple[int, int]] = set()

    @property
    def open_keys(self) -> list[tuple[int, int]]:
        return sorted(self._open)

    @property
    def tick(self) -> int:
        """Current window tick (the last shift)."""
        return self._tick

    @property
    def in_prologue(self) -> bool:
        return self._in_prologue

    @property
    def tie_keys(self) -> list[tuple[int, int]]:
        """``(program, pitch)`` the chunk's tie prologue declared so far."""
        return sorted(self._tie_set)

    def feed(self, token: int) -> list[Action]:
        etype, value = self._vocab[token]
        if self._in_prologue:
            if etype == "tie":
                self._in_prologue = False
                self._velocity = None
                ended = sorted(k for k in self._open if k not in self._tie_set)
                # Every declared tie is open from here on, including keys
                # not seeded (a model-stated prologue the caller adopts), so
                # their offsets decode.
                self._open = set(self._tie_set)
                return [Action("end", p, q, 0) for p, q in ended]
            if etype == "shift":
                self._in_prologue = False
                self._skip_rest = True
                ended = sorted(self._open)
                self._open.clear()
                return [Action("end", p, q, 0) for p, q in ended]
            if etype == "program":
                self._program = value
            elif etype == "pitch" and self._program is not None:
                self._tie_set.add((self._program, value))
            return []
        if self._skip_rest:
            return []
        if etype == "shift":
            if value > 0:
                self._tick = value
        elif etype == "program":
            self._program = value
        elif etype == "velocity":
            self._velocity = value
        elif etype == "drum":
            return [Action("drum", DRUM_PROGRAM, value, self._tick)]
        elif etype == "pitch":
            if self._program is None or self._velocity is None:
                return []
            key = (self._program, value)
            actions = []
            if key in self._open:
                self._open.discard(key)
                actions.append(Action("end", key[0], key[1], self._tick))
            if self._velocity > 0:
                self._open.add(key)
                actions.append(Action("start", key[0], key[1], self._tick))
            return actions
        return []


@dataclass(frozen=True)
class WindowNote:
    """A note in window ticks; ``onset`` None for a note tied in from
    before the window, ``end`` None for one still open."""

    onset: Optional[int]
    end: Optional[int]
    program: int
    pitch: int
    is_drum: bool = False


def decode_window(tokens: Iterable[int], vocab: Sequence[tuple[str, int]],
                  tie_keys: Iterable[tuple[int, int]] = ()) -> list[WindowNote]:
    """Every note a chunk's tokens imply, sorted (round-trip helper)."""
    dec = WindowDecoder(vocab, tie_keys)
    started: dict[tuple[int, int], Optional[int]] = {k: None for k in tie_keys}
    notes: list[WindowNote] = []
    for tok in tokens:
        for a in dec.feed(tok):
            key = (a.program, a.pitch)
            if a.kind == "drum":
                notes.append(WindowNote(a.tick, a.tick + DRUM_TICKS, DRUM_PROGRAM, a.pitch, True))
            elif a.kind == "end":
                notes.append(WindowNote(started.pop(key, None), a.tick, *key))
            else:
                started[key] = a.tick
    for key in dec.open_keys:
        notes.append(WindowNote(started.get(key), None, *key))
    return sorted(notes, key=lambda n: (-1 if n.onset is None else n.onset, n.program, n.pitch))


def window_events(notes: Iterable[WindowNote]) -> tuple[list[tuple[int, int]], list[WindowEvent]]:
    """Inverse of :func:`decode_window`: tie keys and events for notes."""
    ties: list[tuple[int, int]] = []
    events: list[WindowEvent] = []
    for n in notes:
        if n.is_drum:
            events.append(WindowEvent(n.onset, True, DRUM_PROGRAM, 1, n.pitch))
            continue
        if n.onset is None:
            ties.append((n.program, n.pitch))
        else:
            events.append(WindowEvent(n.onset, False, n.program, 1, n.pitch))
        if n.end is not None:
            events.append(WindowEvent(n.end, False, n.program, 0, n.pitch))
    return ties, events


# ---------------------------------------------------------------------------
# Backend seam
# ---------------------------------------------------------------------------


class StreamBackend(Protocol):
    """What the decoder needs from a loaded model."""

    #: token id -> ``(type, value)``.
    vocab: Sequence[tuple[str, int]]
    eos_id: int

    def instrument_for_program(self, program: int) -> str: ...

    def generate(self, prompt_ids: Sequence[int], audio: np.ndarray, sr: int,
                 instruments: Optional[Sequence[str]], max_new_tokens: int) -> Iterator[int]:
        """Teacher-force ``prompt_ids`` on the window ``audio`` and yield
        only the generated token ids (greedy), EOS included, at most
        ``max_new_tokens``. Closing the iterator stops decoding."""
        ...


class MuScriptorStreamBackend:
    """:class:`StreamBackend` over a loaded ``TranscriptionModel``. Uses
    its private ``_model`` (LMModel), ``_tokenizer`` and
    ``_instrument_for_program``; no change inside the package.

    ``pad_mode`` decides how a window shorter than 5 s is conditioned:
    ``"mask"`` pads the waveform to 5 s and sets the condition length to
    the real samples, so the mel frames past it are exact zero embeddings
    (the conditioner's length mask; MuScriptor's own last chunk instead
    pads with zeros at full length, which decodes the padding as audio);
    ``"short"`` passes the short waveform so fewer mel frames are
    prepended. Neither is the zero-padded batch behaviour.

    Decoding mode, passed straight to ``LMModel.generate``: greedy by
    default; ``use_sampling`` with ``temperature`` / ``top_k`` / ``top_p``;
    ``cfg_coef`` (1.0 = no guidance, else a doubled CFG batch).

    ``repeat_breaker`` forbids, for the next step only, the exact token
    that would restate the onset just emitted at the same tick (the same
    ``pitch`` with velocity 1, or the same ``drum``). ``generate`` takes
    one ``forbidden_tokens`` tensor for every step but reads it at each
    step, so the backend appends one slot to it and rewrites that slot in
    place between steps (``PAD`` when nothing is forbidden; the model
    never emits ``PAD``).
    """

    def __init__(self, model: Any, pad_mode: str = "mask", *, use_sampling: bool = False,
                 temperature: float = 1.0, top_k: int = 0, top_p: float = 0.0,
                 cfg_coef: float = 1.0, repeat_breaker: bool = False):
        if pad_mode not in ("mask", "short"):
            raise ValueError(f"pad_mode must be mask or short, got {pad_mode!r}")
        self._tm = model
        self.pad_mode = pad_mode
        self.use_sampling = bool(use_sampling)
        self.temperature = float(temperature)
        self.top_k = int(top_k)
        self.top_p = float(top_p)
        self.cfg_coef = float(cfg_coef)
        self.repeat_breaker = bool(repeat_breaker)
        tok = model._tokenizer
        self.vocab = [(e.type, e.value) for e in tok._vocab]
        self.eos_id = int(tok.eos_id)
        self._forbidden_cache: dict[tuple, Any] = {}

    def instrument_for_program(self, program: int) -> str:
        return self._tm._instrument_for_program(program)

    def generate(self, prompt_ids, audio, sr, instruments, max_new_tokens):
        import torch
        import torch.nn.functional as F
        from muscriptor.modules.conditioners import ConditioningAttributes, WavCondition
        from muscriptor.tokenizer.mt3 import instrument_group_from_names
        from muscriptor.utils.audio import resample

        device = self._tm._device
        wav = torch.from_numpy(np.ascontiguousarray(audio, dtype=np.float32))[None, :]
        if int(sr) != MODEL_SR:
            wav = resample(wav, int(sr), MODEL_SR)
        wav = wav.to(device)
        seg = int(WINDOW_S * MODEL_SR)
        if wav.shape[-1] > seg:
            wav = wav[:, -seg:]
        n = int(wav.shape[-1])
        if self.pad_mode == "mask" and n < seg:
            wav = F.pad(wav, (0, seg - n))
        cond = WavCondition(
            wav=wav.unsqueeze(0), length=torch.tensor([n], device=device),
            sample_rate=[MODEL_SR], path=[None], seek_time=[0.0],
        )
        group = instrument_group_from_names(instruments) if instruments else None
        attrs = ConditioningAttributes(
            wav={"self_wav": cond},
            text={"instrument_group": group, "dataset_name": None},
        )
        forbidden = None
        if instruments:
            key = tuple(instruments)
            forbidden = self._forbidden_cache.get(key)
            if forbidden is None:
                forbidden = torch.tensor(self._tm._tokenizer.forbidden_token_ids(instruments),
                                         device=device, dtype=torch.long)
                self._forbidden_cache[key] = forbidden
        breaker = None
        if self.repeat_breaker:
            pad = torch.zeros(1, device=device, dtype=torch.long)  # PAD id 0
            forbidden = pad if forbidden is None else torch.cat([forbidden, pad])
            breaker = _RepeatBreaker(self.vocab)
        n_prompt = len(prompt_ids)
        prompt = (torch.tensor([list(prompt_ids)], device=device, dtype=torch.long)
                  if n_prompt else None)
        gen = self._tm._model.generate(
            prompt=prompt, conditions=[attrs], max_gen_len=n_prompt + int(max_new_tokens),
            use_sampling=self.use_sampling, temp=self.temperature, top_k=self.top_k,
            top_p=self.top_p, cfg_coef=self.cfg_coef,
            early_stop_on_token=self.eos_id, beam_size=1, forbidden_tokens=forbidden,
        )
        try:
            # LMModel.generate yields the prompt's own steps first.
            for i, step in enumerate(gen):
                tok = int(step[0])
                if breaker is not None:
                    # Read by the next step's logits mask only.
                    forbidden[-1].fill_(breaker.next_forbidden(tok))
                if i >= n_prompt:
                    yield tok
        finally:
            gen.close()


class _RepeatBreaker:
    """Tracks the token stream and names the one token to forbid next:
    the onset just emitted, restated at the same tick (``PAD`` = none)."""

    def __init__(self, vocab: Sequence[tuple[str, int]]):
        self._vocab = vocab
        self._velocity: Optional[int] = None
        self._in_prologue = True

    def next_forbidden(self, token: int) -> int:
        etype, value = self._vocab[token]
        if etype == "tie":
            self._in_prologue = False
            self._velocity = None
            return 0
        if self._in_prologue:
            return 0
        if etype == "velocity":
            self._velocity = value
        elif etype == "drum" or (etype == "pitch" and self._velocity == 1):
            return token
        return 0


# ---------------------------------------------------------------------------
# Ledger and decoder
# ---------------------------------------------------------------------------


@dataclass
class _LedgerNote:
    onset: int  # absolute canvas tick
    end: Optional[int]
    program: int
    pitch: int
    is_drum: bool
    instrument: str
    hop: int  # hop that committed the onset
    tied_in: bool = False  # adopted from a model tie section; onset = window start


@dataclass(frozen=True)
class StreamNote:
    """A ledger note in canvas seconds; ``end_s`` None while open."""

    start_s: float
    end_s: Optional[float]
    pitch: int
    instrument: str
    program: int
    #: Sounding since before ``start_s`` (adopted from a tie section); not an onset.
    tied_in: bool = False

    def item(self, open_end_s: Optional[float] = None) -> dict:
        """The wire ``note`` item; an open note ends at ``open_end_s``."""
        end = self.end_s if self.end_s is not None else max(self.start_s, open_end_s or self.start_s)
        return {"start_s": self.start_s, "end_s": end, "pitch": self.pitch,
                "instrument": self.instrument}


@dataclass(frozen=True)
class StepResult:
    """What one hop changed. ``new_notes`` committed their onset this hop;
    ``closed`` got their end this hop (drums included); ``revised`` are
    the closed ones whose onset an earlier hop committed; ``open_notes``
    is every note open after the hop. A ``truncated`` hop (``reason``
    ``cap``, ``wall`` or ``loop``) kept what it decoded up to the last
    complete tick before the cut and left the rest for the next hop."""

    new_notes: tuple = ()
    closed: tuple = ()
    revised: tuple = ()
    open_notes: tuple = ()
    wall_s: float = 0.0
    prompt_tokens: int = 0
    new_tokens: int = 0
    commit_s: float = 0.0
    truncated: bool = False
    reason: Optional[str] = None


def _tick_ceil(t: float) -> int:
    return int(math.ceil(t * FRAME_RATE - 1e-6))


def _tick_floor(t: float) -> int:
    return int(math.floor(t * FRAME_RATE + 1e-6))


class _LoopDetector:
    """Watches one hop's generated actions for a runaway. Only onsets at
    or after ``c_old_rel`` (window ticks: the audio not settled yet)
    count. Trips when one key repeats :data:`LOOP_REPEATS` times at
    near-equal spacing (only with ``repeats``: a whole-window hop holds
    legitimately repeating hats and bass lines) or when the onsets exceed
    :data:`LOOP_ONSETS_PER_S` per second of ``new_audio_s`` (floor
    :data:`LOOP_REPEATS`, see there). ``feed``
    returns the window tick where the loop started, or ``None``."""

    def __init__(self, c_old_rel: int, new_audio_s: float, repeats: bool = True):
        self.c_old_rel = c_old_rel
        self.limit = max(LOOP_REPEATS, LOOP_ONSETS_PER_S * max(new_audio_s, 0.0))
        self.repeats = repeats
        self.onsets: list[int] = []
        self.by_key: dict[tuple[int, int], list[int]] = {}

    def feed(self, actions: Iterable[Action]) -> Optional[int]:
        for a in actions:
            if a.kind == "end" or a.tick < self.c_old_rel:
                continue
            self.onsets.append(a.tick)
            if len(self.onsets) > self.limit:
                return self.onsets[0]
            if not self.repeats:
                continue
            ticks = self.by_key.setdefault((a.program, a.pitch), [])
            ticks.append(a.tick)
            if len(ticks) >= LOOP_REPEATS:
                run = ticks[-LOOP_REPEATS:]
                gaps = [b - a_ for a_, b in zip(run, run[1:])]
                if min(gaps) > 0 and max(gaps) - min(gaps) <= LOOP_SPACING_TOL:
                    return run[0]
        return None


class StreamingNotesDecoder:
    """Per-take streaming transcription over a :class:`StreamBackend`.

    Not thread-safe: one caller (the analysis host's worker thread).
    """

    def __init__(self, backend: StreamBackend, *, new_token_cap: int = NEW_TOKEN_CAP,
                 guard_s: float = GUARD_S, hop_wall_budget_s: float = HOP_WALL_BUDGET_S,
                 full_token_cap: int = FULL_TOKEN_CAP,
                 full_hop_wall_budget_s: float = FULL_HOP_WALL_BUDGET_S,
                 loop_detect: bool = True,
                 clock: Callable[[], float] = time.perf_counter):
        self.backend = backend
        self.new_token_cap = int(new_token_cap)
        self.guard_s = float(guard_s)
        self.hop_wall_budget_s = float(hop_wall_budget_s)
        self.full_token_cap = int(full_token_cap)
        self.full_hop_wall_budget_s = float(full_hop_wall_budget_s)
        self.loop_detect = bool(loop_detect)
        self._clock = clock
        self._vocab = list(backend.vocab)
        self._index = {e: i for i, e in enumerate(self._vocab)}
        self.reset()

    def reset(self) -> None:
        """Forget the take: a new take, epoch or a backwards frontier."""
        self._ledger: list[_LedgerNote] = []
        self._commit_tick = 0
        # First canvas tick the ledger is complete from (None: nothing heard).
        self._known_from: Optional[int] = None
        self._frontier_s: Optional[float] = None
        self._hop = 0
        self._warned = False
        self._loop_warned = False

    @property
    def commit_s(self) -> float:
        """Onsets before this canvas time are settled in the ledger."""
        return self._commit_tick / FRAME_RATE

    # -- prompt --------------------------------------------------------

    def build_prompt(self, ws_tick: int, c_old: int) -> tuple[list[int], list[tuple[int, int]]]:
        """Prompt ids for a window starting at canvas tick ``ws_tick``:
        the ledger notes inside ``[ws_tick, c_old)`` re-timed to the
        window, with the tie prologue for notes sounding at ``ws_tick``.
        Returns ``(ids, tie_keys)``."""
        ties: list[tuple[int, int]] = []
        events: list[WindowEvent] = []
        for n in self._ledger:
            if n.is_drum:
                if ws_tick <= n.onset < c_old:
                    events.append(WindowEvent(n.onset - ws_tick, True, DRUM_PROGRAM, 1, n.pitch))
                continue
            key = (n.program, n.pitch)
            started = n.onset < ws_tick or (n.tied_in and n.onset <= ws_tick)
            tied = started and (n.end is None or n.end > ws_tick)
            if tied:
                ties.append(key)
            elif not n.tied_in and ws_tick <= n.onset < c_old:
                events.append(WindowEvent(n.onset - ws_tick, False, n.program, 1, n.pitch))
            else:
                continue
            if n.end is not None and n.end < c_old:
                events.append(WindowEvent(n.end - ws_tick, False, n.program, 0, n.pitch))
        return encode_window(ties, events, self._index), sorted(set(ties))

    def _ledger_speaks_for(self, ws_tick: int) -> bool:
        """Whether the ledger may force the window's prologue: it has
        analysed ``ws_tick`` (complete from ``known_from`` up to the last
        successful hop's commit point). An empty tie set is then the truth
        (silence at the window start) and is forced too. Otherwise the
        model states its own tie section (an empty prompt)."""
        return (self._known_from is not None
                and self._known_from <= ws_tick <= self._commit_tick)

    # -- hop -----------------------------------------------------------

    def step(self, audio: np.ndarray, sr: int, window_start_s: float, frontier_s: float,
             instruments: Optional[Sequence[str]] = None) -> StepResult:
        """Decode one hop. ``audio`` is mono covering ``[window_start_s,
        frontier_s)`` at ``sr`` (shorter than 5 s early in a take; never
        zero-padded at the end by the caller)."""
        audio = np.asarray(audio, dtype=np.float32)
        if audio.ndim != 1:
            raise ValueError(f"streaming decoder wants mono audio, got {audio.shape}")
        frontier_s = float(frontier_s)
        if self._frontier_s is not None and frontier_s < self._frontier_s - 1e-6:
            logger.info("notes_stream_reset reason=frontier_back frontier_s={:.2f} prev_s={:.2f}",
                        frontier_s, self._frontier_s)
            self.reset()
        # The window starts on the 10 ms token grid, at most WINDOW_S long.
        ws = max(float(window_start_s), frontier_s - WINDOW_S)
        ws_tick = _tick_ceil(ws)
        drop = int(round((ws_tick / FRAME_RATE - float(window_start_s)) * sr))
        audio = audio[max(0, drop):]
        c_new = _tick_floor(frontier_s - self.guard_s)
        c_old = max(self._commit_tick, ws_tick)
        self._frontier_s = frontier_s
        self._hop += 1

        t0 = self._clock()
        prompt, ties = self.build_prompt(ws_tick, c_old)
        full = not self._ledger_speaks_for(ws_tick)
        if full:
            # Seed the decoder with the ledger's notes sounding before the
            # window, so the model's tie section closes the ones it omits.
            prompt = []
            ties = sorted({(n.program, n.pitch) for n in self._ledger
                           if n.end is None and not n.is_drum
                           and (n.onset < ws_tick or (n.tied_in and n.onset <= ws_tick))})
        cap = self.full_token_cap if full else self.new_token_cap
        budget = self.full_hop_wall_budget_s if full else self.hop_wall_budget_s
        dec = WindowDecoder(self._vocab, ties)
        for tok in prompt:
            dec.feed(tok)
        actions: list[Action] = []
        n_new = 0
        reason: Optional[str] = "cap"
        eos = self.backend.eos_id
        c_rel = c_new - ws_tick
        loop = _LoopDetector(c_old - ws_tick, frontier_s - c_old / FRAME_RATE,
                             repeats=not full) if self.loop_detect else None
        loop_at: Optional[int] = None
        gen = self.backend.generate(prompt, audio, int(sr), instruments, cap)
        try:
            for tok in gen:
                if tok == eos:
                    reason = None
                    break
                n_new += 1
                etype, value = self._vocab[int(tok)]
                if etype == "shift" and value > 0 and not dec.in_prologue:
                    if value < dec.tick:
                        # Time went backwards: the model is re-transcribing
                        # what it already said. Keep what came before.
                        reason = None
                        break
                    if value >= c_rel:
                        # Nothing at or past the commit point can commit
                        # this hop: stop decoding (lossless, saves the wall).
                        reason = None
                        break
                fed = dec.feed(int(tok))
                actions.extend(fed)
                if loop is not None:
                    loop_at = loop.feed(fed)
                    if loop_at is not None:
                        reason = "loop"
                        break
                if self._clock() - t0 > budget:
                    reason = "wall"
                    break
                if n_new >= cap:
                    break  # no EOS within the cap
        finally:
            close = getattr(gen, "close", None)
            if close is not None:
                close()
        wall_s = self._clock() - t0

        c_eff = c_new
        if reason is not None:
            if reason == "loop":
                if not self._loop_warned:
                    self._loop_warned = True
                    logger.warning(
                        "notes_stream_loop reason=loop new_tokens={} loop_s={:.2f} "
                        "frontier_s={:.2f} (once per take)",
                        n_new, (ws_tick + loop_at) / FRAME_RATE, frontier_s,
                    )
            elif not self._warned:
                self._warned = True
                logger.warning(
                    "notes_stream_hop_truncated reason={} new_tokens={} wall_s={:.3f} "
                    "frontier_s={:.2f} (once per take)", reason, n_new, wall_s, frontier_s,
                )
            if dec.in_prologue:
                return StepResult(open_notes=self._open_notes(), wall_s=wall_s,
                                  prompt_tokens=len(prompt), new_tokens=n_new,
                                  commit_s=self.commit_s, truncated=True, reason=reason)
            # Keep the decoded tail up to the last complete tick: events at
            # the current tick may still have company the model never got
            # to emit, everything before it is whole.
            c_eff = min(c_new, ws_tick + dec.tick)
            if loop_at is not None:
                # Nothing from where the loop started.
                c_eff = min(c_eff, ws_tick + loop_at)

        adopt = [k for k in dec.tie_keys if k not in set(ties)] if full else []
        new, closed = self._apply(actions, ws_tick, c_old, c_eff, adopt)
        self._commit_tick = max(c_old, c_eff)
        if full:
            self._known_from = ws_tick
        self._close_stale()
        self._prune(ws_tick)
        revised = tuple(self._out(n) for n in closed if n.hop < self._hop)
        return StepResult(
            new_notes=tuple(self._out(n) for n in new),
            closed=tuple(self._out(n) for n in closed),
            revised=revised,
            open_notes=self._open_notes(),
            wall_s=wall_s, prompt_tokens=len(prompt), new_tokens=n_new,
            commit_s=self.commit_s, truncated=reason is not None, reason=reason,
        )

    def _apply(self, actions: list[Action], ws_tick: int, c_old: int, c_new: int,
               adopt: Sequence[tuple[int, int]] = ()):
        """Fold one hop's generated actions into the ledger: ``adopt``
        (tie keys the model named that the ledger lacks) become ``tied_in``
        notes from the window start; onsets in ``[c_old, c_new)`` commit,
        ends before ``c_new`` close open notes, everything later is left
        for the next hop."""
        open_by_key = {(n.program, n.pitch): n for n in self._ledger
                       if n.end is None and not n.is_drum}
        new: list[_LedgerNote] = []
        closed: list[_LedgerNote] = []
        for program, pitch in adopt:
            if (program, pitch) in open_by_key:
                continue
            n = _LedgerNote(ws_tick, None, program, pitch, False,
                            self.backend.instrument_for_program(program), self._hop, tied_in=True)
            open_by_key[(program, pitch)] = n
            self._ledger.append(n)
        skip_next = False
        for i, a in enumerate(actions):
            if skip_next:
                skip_next = False
                continue
            t = a.tick + ws_tick
            key = (a.program, a.pitch)
            if a.kind == "end":
                n = open_by_key.get(key)
                nxt = actions[i + 1] if i + 1 < len(actions) else None
                if (n is not None and nxt is not None and nxt.kind == "start"
                        and (nxt.program, nxt.pitch) == key and nxt.tick == a.tick
                        and not n.tied_in and abs(n.onset - t) <= DEDUPE_TICKS):
                    # The model restated an open ledger note a tick or two
                    # later (end + start of its key): not a retrigger.
                    skip_next = True
                    continue
                # An end at or before a ledger note's onset belongs to an
                # earlier note of that key re-decoded from an empty prompt.
                if n is not None and t < c_new and (t > n.onset or n.tied_in):
                    n.end = max(t, n.onset + 1)
                    del open_by_key[key]
                    closed.append(n)
            elif c_old <= t < c_new:
                if self._is_restated(key, t):
                    continue
                if a.kind == "drum":
                    n = _LedgerNote(t, t + DRUM_TICKS, DRUM_PROGRAM, a.pitch, True, "drums", self._hop)
                    closed.append(n)
                else:
                    prev = open_by_key.get(key)
                    if prev is not None:
                        # A re-decoded window restarted a key the ledger
                        # holds open: the old note ends here.
                        prev.end = max(t, prev.onset + 1)
                        closed.append(prev)
                    n = _LedgerNote(t, None, a.program, a.pitch, False,
                                    self.backend.instrument_for_program(a.program), self._hop)
                    open_by_key[key] = n
                self._ledger.append(n)
                new.append(n)
        return new, closed

    def _is_restated(self, key: tuple[int, int], t: int) -> bool:
        """A ledger note of ``key`` already has its onset within
        :data:`DEDUPE_TICKS` of ``t``: the model re-emitting a settled
        note just past the commit boundary (tick jitter), not a new one."""
        return any((n.program, n.pitch) == key and not n.tied_in
                   and abs(n.onset - t) <= DEDUPE_TICKS for n in self._ledger)

    def _close_stale(self) -> None:
        limit = int(MAX_OPEN_NOTE_S * FRAME_RATE)
        for n in self._ledger:
            if n.end is None and self._commit_tick - n.onset > limit:
                n.end = self._commit_tick

    def _prune(self, ws_tick: int) -> None:
        self._ledger = [n for n in self._ledger if n.end is None or n.end > ws_tick]

    def _out(self, n: _LedgerNote) -> StreamNote:
        return StreamNote(n.onset / FRAME_RATE, None if n.end is None else n.end / FRAME_RATE,
                          n.pitch, n.instrument, n.program, n.tied_in)

    def _open_notes(self) -> tuple:
        return tuple(self._out(n) for n in self._ledger if n.end is None)

    def notes_in(self, lo_s: float, hi_s: float) -> tuple[list[StreamNote], list[StreamNote]]:
        """Ledger notes with onset in ``[lo_s, hi_s)``: ``(closed, open)``,
        each sorted by ``(start_s, pitch)``. ``tied_in`` notes have no
        heard onset and are never reported here."""
        lo, hi = _tick_ceil(lo_s), _tick_ceil(hi_s)
        picked = sorted((n for n in self._ledger if not n.tied_in and lo <= n.onset < hi),
                        key=lambda n: (n.onset, n.pitch))
        closed = [self._out(n) for n in picked if n.end is not None]
        opened = [self._out(n) for n in picked if n.end is None]
        return closed, opened
