"""SPIKE: live MIDI transcription of the streaming session's canvas.

Attaches a MuScriptor transcriber to a :class:`StreamingSession` as an
``EventBus`` subscriber. The bus contract does the isolation work:
``publish`` is non-blocking and a slow subscriber cannot stall GPU
ticks — the open question this spike measures is GPU *contention*
(the transcriber's forward passes vs. tick_ms), not thread safety.

Shape:

* ``_CanvasMirror`` — a numpy mirror of the playback buffer, written
  from the subscription's drainer thread (cheap memcpy only). Tracks
  the write frontier through wraparound. Seeded from the session's
  ``initial_buffer`` so the canvas is complete from t=0.
* ``LiveMidiTranscriber`` — a worker thread that, each time the
  frontier advances by ``stride_s``, transcribes the 5 s window ending
  at the frontier and emits the note onsets that fall inside the fresh
  stride region (interior of earlier windows, so each note is emitted
  exactly once, with real right-context). Onsets are mapped back to
  absolute canvas seconds.
* Optional ``mido`` output: each note is scheduled to sound when the
  *playhead* reaches its canvas position, i.e. ear-synced. A note whose
  canvas position the playhead has already passed is counted late and
  dropped — the late-rate IS the "does this feel real-time" metric.

Requires ``muscriptor`` (and ``python-rtmidi`` for port output) in the
environment; both are spike-only deps installed with ``uv pip install``,
not declared in pyproject.

Weights note: MuScriptor weights are CC BY-NC 4.0 (gated on HF). Fine
for this local research spike; NOT shippable in the product without a
license conversation.
"""

from __future__ import annotations

import heapq
import threading
import time
from dataclasses import dataclass, field

import numpy as np

from acestep.streaming.events import AudioReady

SAMPLE_RATE = 48000
# MuScriptor's training segment length. The model sees exactly this much
# context; the window we cut from the canvas matches it.
WINDOW_S = 5.0


@dataclass
class LiveMidiStats:
    """Rolling counters, printed by the driver. All wall seconds."""

    windows: int = 0
    notes_emitted: int = 0
    notes_late: int = 0
    window_wall_s: list = field(default_factory=list)   # per-window transcribe cost
    first_event_s: list = field(default_factory=list)   # window start -> first note event
    tick_ms_seen: list = field(default_factory=list)    # tick_ms sampled off AudioReady

    def summary(self) -> str:
        w = np.array(self.window_wall_s[1:] or [0.0])   # [0] carries CUDA warmup
        f = np.array(self.first_event_s or [0.0])
        t = np.array(self.tick_ms_seen or [0.0])
        return (
            f"windows={self.windows} notes={self.notes_emitted} "
            f"late={self.notes_late} "
            f"| window wall s: mean={w.mean():.3f} p95={np.percentile(w, 95):.3f} "
            f"max={w.max():.3f} "
            f"| first-event s: mean={f.mean():.3f} "
            f"| tick_ms: mean={t.mean():.1f} p95={np.percentile(t, 95):.1f}"
        )


class _CanvasMirror:
    """Numpy mirror of the wrap-around playback canvas.

    Written from the bus drainer thread (slice memcpy under a lock),
    read by the transcriber worker. Frontier updates follow the runner's
    semantics: writes land at/ahead of the playhead and sweep forward;
    a write that would move the frontier *backward* by more than half
    the buffer is a wrap, a small retreat is the runner re-refining and
    leaves the frontier alone (see pipeline_runner's frontier notes).
    """

    def __init__(self, initial: np.ndarray):
        if initial.ndim == 1:
            initial = initial[:, None]
        self._buf = initial.astype(np.float32, copy=True)
        self._n = self._buf.shape[0]
        self._lock = threading.Lock()
        # Unwrapped monotonic frontier in samples; starts one full canvas
        # in so the first window has real audio behind it.
        self._frontier_total = self._n
        self._frontier_pos = 0  # == _frontier_total % _n

    @property
    def duration_s(self) -> float:
        return self._n / SAMPLE_RATE

    def write(self, ev: AudioReady) -> None:
        audio = ev.audio if ev.audio.ndim == 2 else ev.audio[:, None]
        start = int(ev.start_sample) % self._n
        n = audio.shape[0]
        with self._lock:
            end = start + n
            if end <= self._n:
                self._buf[start:end] = audio
            else:  # wrap-around write
                k = self._n - start
                self._buf[start:] = audio[:k]
                self._buf[: end - self._n] = audio[k:]
            wrapped_end = end % self._n
            advance = (wrapped_end - self._frontier_pos) % self._n
            # Forward moves are small (slices arrive ~0.04s apart); a huge
            # "advance" is really a retreat/backfill — ignore it.
            if 0 < advance < self._n // 2:
                self._frontier_total += advance
                self._frontier_pos = wrapped_end

    def frontier_total(self) -> int:
        with self._lock:
            return self._frontier_total

    def read_window(self, end_total: int, num: int) -> tuple[np.ndarray, float]:
        """Copy ``num`` samples ending at unwrapped position ``end_total``.

        Returns ``(mono float32 [num], canvas_start_s)`` where
        ``canvas_start_s`` is the window start in wrapped canvas seconds
        (what playhead positions are measured in).
        """
        start_pos = (end_total - num) % self._n
        with self._lock:
            if start_pos + num <= self._n:
                out = self._buf[start_pos : start_pos + num].copy()
            else:
                k = self._n - start_pos
                out = np.concatenate((self._buf[start_pos:], self._buf[: num - k]))
        mono = out.mean(axis=1)
        return mono, start_pos / SAMPLE_RATE


class LiveMidiTranscriber:
    """Owns the model, the worker thread, and (optionally) a MIDI port."""

    def __init__(
        self,
        session,
        *,
        model_size: str = "small",
        stride_s: float = 1.0,
        guard_s: float = 0.3,
        midi_port: str | None = None,
        on_note=None,
        log=print,
    ):
        self._session = session
        self._stride = int(stride_s * SAMPLE_RATE)
        # Onsets within ``guard_s`` of the frontier have no right-context
        # yet; they are deferred to the next window (where they are
        # interior). Costs ``guard_s`` of latency, buys stable onsets.
        self._guard = int(guard_s * SAMPLE_RATE)
        self._log = log
        self._on_note = on_note
        self.stats = LiveMidiStats()

        self._mirror = _CanvasMirror(session.initial_buffer)
        self._running = False
        self._worker: threading.Thread | None = None
        self._scheduler: threading.Thread | None = None
        self._sub = None

        # ---- model (lazy import: spike-only dependency) ----
        from muscriptor.transcription_model import TranscriptionModel

        t0 = time.perf_counter()
        self._model = TranscriptionModel.load_model(
            model_size, device="cuda", dtype="float16"
        )
        self._log(f"[midi_live] {model_size} loaded in {time.perf_counter()-t0:.1f}s")

        # ---- optional MIDI out (ear-synced scheduling) ----
        self._port = None
        self._heap: list = []  # (due_wall_s, seq, mido.Message)
        self._heap_lock = threading.Lock()
        self._heap_seq = 0
        self._chan_by_instrument: dict[str, int] = {}
        if midi_port is not None:
            import mido

            self._port = mido.open_output(midi_port)
            self._log(f"[midi_live] MIDI out -> {midi_port!r}")

    # ------------------------------------------------------------------
    def attach(self) -> None:
        self._running = True
        self._sub = self._session.bus.subscribe(self._on_event, name="midi_live")
        self._worker = threading.Thread(
            target=self._run, name="midi_live_worker", daemon=True
        )
        self._worker.start()
        if self._port is not None:
            self._scheduler = threading.Thread(
                target=self._run_scheduler, name="midi_live_sched", daemon=True
            )
            self._scheduler.start()

    def detach(self) -> None:
        self._running = False
        if self._sub is not None:
            self._session.bus.unsubscribe(self._sub)
        if self._worker is not None:
            self._worker.join(timeout=10.0)
        if self._port is not None:
            self._all_notes_off()
            self._port.close()

    # ------------------------------------------------------------------
    def _on_event(self, ev) -> None:
        # Drainer thread: memcpy only, never model work.
        if isinstance(ev, AudioReady):
            self._mirror.write(ev)
            self.stats.tick_ms_seen.append(float(ev.tick_ms))

    # ------------------------------------------------------------------
    def _run(self) -> None:
        window_n = int(WINDOW_S * SAMPLE_RATE)
        next_at = self._mirror.frontier_total() + self._stride
        import torch

        while self._running:
            ft = self._mirror.frontier_total()
            if ft < next_at:
                time.sleep(0.02)
                continue
            end_total = ft
            mono, canvas_start_s = self._mirror.read_window(end_total, window_n)
            emit_lo = end_total - self._stride - self._guard  # unwrapped samples
            emit_hi = end_total - self._guard
            next_at = end_total + self._stride

            t0 = time.perf_counter()
            first_event_s = None
            n_emitted = 0
            wav = torch.from_numpy(mono)[None, :]
            try:
                for ev in self._model.transcribe((wav, SAMPLE_RATE)):
                    name = type(ev).__name__
                    if name == "NoteStartEvent":
                        if first_event_s is None:
                            first_event_s = time.perf_counter() - t0
                        onset_total = (
                            end_total - window_n + int(ev.start_time * SAMPLE_RATE)
                        )
                        if emit_lo <= onset_total < emit_hi:
                            self._emit(ev, canvas_start_s)
                            n_emitted += 1
            except Exception as exc:  # spike: log, keep streaming
                self._log(f"[midi_live] transcribe failed: {exc!r}")
                continue
            dt = time.perf_counter() - t0

            self.stats.windows += 1
            self.stats.window_wall_s.append(dt)
            if first_event_s is not None:
                self.stats.first_event_s.append(first_event_s)
            self.stats.notes_emitted += n_emitted
            self._log(
                f"[midi_live] window @{canvas_start_s:6.1f}s "
                f"wall={dt:5.3f}s notes={n_emitted:3d} "
                f"late={self.stats.notes_late}"
            )

    # ------------------------------------------------------------------
    def _emit(self, ev, canvas_start_s: float) -> None:
        onset_canvas_s = (canvas_start_s + ev.start_time) % self._mirror.duration_s
        if self._on_note is not None:
            self._on_note(ev, onset_canvas_s)
        if self._port is None:
            return

        # Ear-sync: schedule against the live playhead. Lead is circular;
        # a "lead" in the back half of the canvas means the playhead
        # already passed this note -> late, drop, count.
        playhead_s = self._session.audio_eng.position / SAMPLE_RATE
        dur = self._mirror.duration_s
        lead = (onset_canvas_s - playhead_s) % dur
        if lead > dur / 2:
            self.stats.notes_late += 1
            return

        import mido

        chan = self._channel_for(ev.instrument)
        now = time.monotonic()
        self._push(now + lead, mido.Message(
            "note_on", note=ev.pitch, velocity=96, channel=chan))
        # No offsets from NoteStartEvent alone; fixed length keeps the
        # spike simple (NoteEndEvent pairing is a follow-up).
        self._push(now + lead + 0.25, mido.Message(
            "note_off", note=ev.pitch, velocity=0, channel=chan))

    def _channel_for(self, instrument: str) -> int:
        if "drum" in instrument.lower():
            return 9
        if instrument not in self._chan_by_instrument:
            used = len(self._chan_by_instrument)
            chans = [c for c in range(16) if c != 9]
            self._chan_by_instrument[instrument] = chans[used % len(chans)]
        return self._chan_by_instrument[instrument]

    def _push(self, due: float, msg) -> None:
        with self._heap_lock:
            heapq.heappush(self._heap, (due, self._heap_seq, msg))
            self._heap_seq += 1

    def _run_scheduler(self) -> None:
        while self._running:
            now = time.monotonic()
            msg = None
            with self._heap_lock:
                if self._heap and self._heap[0][0] <= now:
                    _, _, msg = heapq.heappop(self._heap)
            if msg is not None:
                self._port.send(msg)
            else:
                time.sleep(0.002)

    def _all_notes_off(self) -> None:
        import mido

        for chan in range(16):
            self._port.send(mido.Message("control_change", control=123,
                                         value=0, channel=chan))
