"""LiveInputBackend: a canvas the client feeds instead of a generator.

The ``live_input`` family (display "Audio input") runs no model at all.
Its canvas is either

* **a clip**: the session's source upload (or a later ``swap_source``)
  is the canvas, the client loops it, and nothing ever rewrites it; or
* **a capture**: a silent canvas of ``live_input_capture_s`` seconds that
  the client fills with ``audio_in`` frames (microphone PCM) appended at
  the write frontier.

Everything that listens to a session (the analysis taps, the wire's
slice stream) works on it unchanged: the canvas is an ordinary
:class:`~acestep.streaming.audio_engine.AudioEngine`, appends publish
:class:`~acestep.streaming.events.AudioReady` like any append-only
family, and a new take is a :class:`~acestep.streaming.events.SwapReady`
with a new ``source_epoch``, so taps clear exactly as they do on a swap.

Threading. The runner still drives this backend (that keeps the session
lifecycle, ``before_tick`` swaps and teardown identical to every other
family), but there is nothing to generate: :meth:`LiveInputBackend.produce`
only paces the loop and flushes a due ``audio_in_ack``. The append path,
:meth:`LiveInputBackend.append_input`, is the one deliberate exception to
the seam's runner-thread-only rule: it runs on the transport's dispatch
thread so a frame lands in the canvas the moment it arrives instead of a
runner tick later. It touches only the audio engine (which locks its own
buffer), the session state under ``state._lock`` and the event bus (which
is thread-safe), all under this backend's own lock, so it never races
the runner.

Lock order: backend lock, then ``state._lock``, then the audio engine's
lock (inside its write calls). :meth:`LiveInputBackend.analysis_frontier_s`
takes no lock at all, because the analysis observer calls it from inside
an audio-engine write notification.

Transport-agnostic: numpy in, events out. Torch-free.
"""

from __future__ import annotations

import threading
import time
from typing import Callable, Optional

import numpy as np

from acestep.engine.obs import logger
from acestep.streaming.events import AudioInAck, AudioReady, CommandFailed, SwapReady
from acestep.streaming.generator_backend import (
    AudioChunk,
    AudioGeometry,
    Capabilities,
    LeadProfile,
    ProduceMode,
    TickContext,
    UnsupportedOperation,
)

#: Canvas sample rate. Client frames use the same 48 kHz upload format as
#: every other PCM command; there is no resampling on this path.
SAMPLE_RATE = 48000

#: Capture canvas length when the client does not choose one, and its
#: bounds (the ``live_input_capture_s`` family config field).
DEFAULT_CAPTURE_S = 600.0
MIN_CAPTURE_S = 30.0
MAX_CAPTURE_S = 1800.0

#: The server playhead sits this far behind the write frontier during a
#: capture, so lookahead math and ``analysis_update.late`` stay meaningful
#: without the client sending ``params.playback_pos``.
PLAYHEAD_BEHIND_S = 0.25

#: A gap in the stream (missing frames, or a ``start_s`` ahead of the
#: frontier) up to this long is filled with silence; a longer one, or any
#: step backwards, starts a new take.
MAX_GAP_FILL_S = 2.0

#: Stream positions this close to the frontier count as contiguous (the
#: client's float ``start_s`` rounds to samples).
CONTIGUOUS_TOLERANCE_S = 0.001

#: ``audio_in_ack`` is coalesced to at most one per this many seconds.
ACK_INTERVAL_S = 0.2

#: Nominal client frame cadence (100 ms frames); the geometry's
#: ``chunk_rate_hz``, a pacing prior only.
FRAME_RATE_HZ = 10.0

#: How long one runner iteration waits before it re-checks the ack.
IDLE_WAIT_S = 0.05

MODE_CAPTURE = "capture"
MODE_CLIP = "clip"


def clamp_capture_s(value) -> float:
    """``live_input_capture_s`` defaulted and clamped to its bounds."""
    try:
        seconds = float(value) if value is not None else 0.0
    except (TypeError, ValueError):
        seconds = 0.0
    if not seconds or seconds <= 0.0:
        seconds = DEFAULT_CAPTURE_S
    return max(MIN_CAPTURE_S, min(seconds, MAX_CAPTURE_S))


class LiveInputBackend:
    """The ``live_input`` family's GeneratorBackend. See module docstring.

    ``mode`` is :data:`MODE_CAPTURE` (silent canvas, filled by
    :meth:`append_input`) or :data:`MODE_CLIP` (the canvas is a static
    upload). ``capture_s`` is the length of every capture take.
    """

    name = "live_input"

    def __init__(
        self,
        *,
        audio_eng,
        state,
        bus,
        mode: str,
        capture_s: float,
        clock: Callable[[], float] = time.monotonic,
    ):
        if mode not in (MODE_CAPTURE, MODE_CLIP):
            raise ValueError(f"live_input mode {mode!r} is not capture or clip")
        self.audio_eng = audio_eng
        self.state = state
        self.bus = bus
        self._clock = clock
        self.capture_samples = int(round(clamp_capture_s(capture_s) * SAMPLE_RATE))

        # Runner bookkeeping (PipelineRunner reads these; nothing renders).
        self.vae_window = 1.0
        self.decode_span_s = 0.0
        self.last_tick_ms = 0.0
        self.last_dec_ms = 0.0

        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._closed = False

        self._mode = mode
        # Write frontier in canvas samples. A clip is final end to end.
        self._frontier = len(audio_eng.current) if mode == MODE_CLIP else 0
        # Lock-free mirror of the frontier for analysis_frontier_s.
        self._frontier_s_view = self._frontier / SAMPLE_RATE
        # Stream position (client sample clock) of the frontier; None until
        # the take's first frame anchors it.
        self._stream_frontier: Optional[int] = None
        self._last_seq: Optional[int] = None

        self._pending_ack: Optional[tuple] = None
        self._last_ack_wall = -float("inf")

        #: Counters for logs and tests.
        self.takes = 1
        self.gap_fills = 0

    # ---- contract ----------------------------------------------------------

    def capabilities(self) -> Capabilities:
        # Append-only (nothing ever re-renders), a swap is a new file, and
        # the canvas is fed by the client: no generator to protect.
        return Capabilities(
            refines_audio=False,
            swap=True,
            swap_resize=True,
            generates_audio=False,
            audio_in=True,
        )

    def geometry(self) -> AudioGeometry:
        return AudioGeometry(
            sample_rate=SAMPLE_RATE,
            channels=int(self.audio_eng.channels),
            chunk_rate_hz=FRAME_RATE_HZ,
            duration_s=len(self.audio_eng.current) / SAMPLE_RATE,
        )

    def lead_profile(self) -> LeadProfile:
        return LeadProfile()

    def knob_specs(self, lora_ids=()) -> list:
        return []

    # ---- LoRA facade: no model, so no LoRA surface --------------------------
    # Capabilities.lora is False, so the session rejects the wire commands
    # before they get here; these exist for protocol conformance and raise
    # rather than silently no-op.

    def lora_compatible(self, metadata: dict) -> bool:
        return False

    def lora_available(self) -> bool:
        return False

    def register_lora(self, path: str) -> str:
        raise UnsupportedOperation("lora")

    def prewarm_lora(self, lora_id: str):
        raise UnsupportedOperation("lora")

    def enable_lora(self, lora_id: str, strength=None) -> None:
        raise UnsupportedOperation("lora")

    def disable_lora(self, lora_id: str) -> None:
        raise UnsupportedOperation("lora")

    def set_lora_strength(self, lora_id: str, strength: float) -> None:
        raise UnsupportedOperation("lora")

    def list_loras(self) -> list:
        return []

    # ---- session control hooks ----------------------------------------------

    def handle_set_prompt(self, tags: str, *, tags_b: Optional[str] = None) -> None:
        """No model reads a prompt; accepted and ignored so the universal
        ``set_prompt`` op does not fail on this family."""

    def handle_set_prompt_blend(self, value: float) -> None:
        """No model reads a prompt blend; accepted and ignored."""

    def handle_swap_source(self, waveform, sample_rate: int, duration_s=None) -> int:
        """Backend half of ``swap_source``: the new file becomes the canvas.

        The session has already truncated ``waveform`` (``[C, N]``) to its
        ceiling; the canvas is exactly the file (``swap_resize``), so the
        returned playback length is its sample count. Runs on the runner
        thread before the session swaps the audio engine, which is what
        lets the analysis observer read the clip's frontier on that swap.
        """
        del sample_rate, duration_s  # the canvas is the file, at 48 kHz
        n = int(waveform.shape[-1])
        with self._lock:
            self._mode = MODE_CLIP
            self._set_frontier(n)
            self._stream_frontier = None
            self._last_seq = None
            self._pending_ack = None
        logger.info("live_input_clip duration_s={:.1f}", n / SAMPLE_RATE)
        return n

    def close(self) -> None:
        with self._lock:
            self._closed = True
        self._wake.set()

    # ---- analysis hook -------------------------------------------------------

    def analysis_frontier_s(self) -> Optional[float]:
        """Seconds of the canvas that are final right now: the clip's
        length, or a capture's write frontier.

        Read by the analysis observer whenever its cell grid resets (tap
        enable, swap, new take), because a static clip produces no write
        from which the observer could learn its frontier. Lock-free (a
        single attribute read): the observer calls it from inside an
        audio-engine write notification, possibly while this backend's
        lock is held by the writer."""
        return self._frontier_s_view

    # ---- the append path (dispatch thread) -----------------------------------

    def append_input(self, pcm: np.ndarray, *, seq: int, start_s: Optional[float] = None) -> None:
        """Append one client frame at the write frontier.

        ``pcm`` is ``[N, C]`` float32 at 48 kHz (mono or stereo; coerced
        to the canvas channel count). ``seq`` is the client's frame
        counter; ``start_s`` the frame's position on the client's stream
        clock, when it sends one. Placement:

        * contiguous with the frontier (or the take's first frame): written
          at the frontier;
        * ahead of it by at most :data:`MAX_GAP_FILL_S` (missing frames, or
          a ``start_s`` jump): the gap is written as silence first;
        * further ahead, behind it, or a ``seq`` that did not advance (the
          client restarted its stream): a new take.

        A write that reaches the canvas end fills it; the remainder (and
        every later frame) starts a new take. Each write publishes
        ``AudioReady``, moves the server playhead to
        :data:`PLAYHEAD_BEHIND_S` behind the frontier, and queues an
        ``audio_in_ack`` (coalesced to :data:`ACK_INTERVAL_S`).
        """
        seq = int(seq)
        with self._lock:
            if self._closed:
                return
            if self._mode == MODE_CLIP:
                self.bus.publish(CommandFailed(
                    command="audio_in",
                    requires="",
                    error=(
                        "this session's canvas is an uploaded clip; open a "
                        "session without a source to stream audio in"
                    ),
                ))
                return
            frame = self._coerce(pcm)
            n = int(frame.shape[0])
            stream_pos = self._stream_position(seq, start_s, n)
            gap = self._place(seq, stream_pos)
            self._last_seq = seq
            if gap > 0:
                self.gap_fills += 1
                frame = np.concatenate(
                    [np.zeros((gap, frame.shape[1]), dtype=np.float32), frame],
                )
            self._write(frame)
            self._pending_ack = (seq, self._frontier / SAMPLE_RATE)
            self._flush_ack(self._clock())

    def _coerce(self, pcm) -> np.ndarray:
        """``[N, C]`` float32 in the canvas channel count."""
        arr = np.asarray(pcm, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        channels = int(self.audio_eng.channels)
        if arr.shape[1] != channels:
            if channels == 1:
                arr = arr.mean(axis=1, keepdims=True)
            else:
                arr = np.repeat(arr[:, :1], channels, axis=1)
        return np.ascontiguousarray(arr, dtype=np.float32)

    def _stream_position(self, seq: int, start_s: Optional[float], n: int) -> int:
        """Where this frame begins on the client's stream clock."""
        if start_s is not None:
            return int(round(float(start_s) * SAMPLE_RATE))
        if self._stream_frontier is None:
            return 0
        missing = 0 if self._last_seq is None else seq - self._last_seq - 1
        return self._stream_frontier + max(0, missing) * n

    def _place(self, seq: int, stream_pos: int) -> int:
        """Anchor the frame against the frontier; returns the silence gap
        (samples) to write before it. Starts a new take when the frame
        cannot continue this one."""
        if self._stream_frontier is None:
            self._stream_frontier = stream_pos
            return 0
        restarted = self._last_seq is not None and seq <= self._last_seq
        gap = stream_pos - self._stream_frontier
        tolerance = int(CONTIGUOUS_TOLERANCE_S * SAMPLE_RATE)
        if not restarted and abs(gap) <= tolerance:
            return 0
        if restarted or gap < 0 or gap > int(MAX_GAP_FILL_S * SAMPLE_RATE):
            reason = "stream_restart" if restarted else ("rewind" if gap < 0 else "gap")
            self._new_take(reason, gap_s=gap / SAMPLE_RATE)
            self._stream_frontier = stream_pos
            return 0
        return gap

    def _write(self, block: np.ndarray) -> None:
        """Write ``block`` at the frontier, rolling over into new takes."""
        while block.shape[0] > 0:
            room = len(self.audio_eng.current) - self._frontier
            if room <= 0:
                self._new_take("canvas_full", gap_s=0.0)
                room = len(self.audio_eng.current)
            part, block = block[:room], block[room:]
            start = self._frontier
            # patch_window, not patch: patch fades each write in from the
            # buffer's old content, which on an append is silence, so every
            # frame boundary would dip. Contiguous final audio needs no fade.
            self.audio_eng.patch_window(part, start)
            self._set_frontier(start + part.shape[0])
            self._stream_frontier += part.shape[0]
            now = self._clock()
            with self.state._lock:
                eng = self.audio_eng
                eng.position = max(0, self._frontier - int(PLAYHEAD_BEHIND_S * SAMPLE_RATE))
                eng.position_report_wall_s = now
                eng.position_staleness_s = 0.0
                params = dict(self.state.params)
            self.bus.publish(AudioReady(
                audio=part,
                start_sample=start,
                num_samples=int(part.shape[0]),
                channels=int(part.shape[1]),
                tick_ms=0.0,
                dec_ms=0.0,
                num_gens=0,
                params=params,
                published_wall_s=now,
            ))

    def _new_take(self, reason: str, *, gap_s: float) -> None:
        """Replace the canvas with a fresh silent capture of
        ``capture_samples`` and publish ``SwapReady`` with a new epoch.
        The stream clock is the caller's: a rollover keeps counting, a
        restart re-anchors it. Caller holds the backend lock."""
        channels = int(self.audio_eng.channels)
        canvas = np.zeros((self.capture_samples, channels), dtype=np.float32)
        state = self.state
        with state._lock:
            # Frontier first: the analysis observer reads it during the
            # audio-engine swap below.
            self._mode = MODE_CAPTURE
            self._set_frontier(0)
            state.source_epoch += 1
            state.playback_samples = self.capture_samples
            state.duration = self.capture_samples / SAMPLE_RATE
            state.n_channels = channels
            self.audio_eng.swap(canvas)
            self.audio_eng.position = 0
            self.audio_eng.loop_band = None
            epoch = state.source_epoch
        self.takes += 1
        logger.info(
            "live_input_new_take reason={} take={} epoch={} gap_s={:.3f}",
            reason, self.takes, epoch, gap_s,
        )
        self.bus.publish(SwapReady(
            duration=self.capture_samples / SAMPLE_RATE,
            sample_rate=SAMPLE_RATE,
            channels=channels,
            bpm=None,
            key=None,
            time_signature=None,
            fixture_name=None,
            initial_buffer=canvas,
            source_epoch=epoch,
        ))

    def _set_frontier(self, samples: int) -> None:
        self._frontier = int(samples)
        self._frontier_s_view = self._frontier / SAMPLE_RATE

    def _flush_ack(self, now: float) -> None:
        """Publish the pending ack if one is due. Caller holds the lock."""
        if self._pending_ack is None or now - self._last_ack_wall < ACK_INTERVAL_S:
            return
        seq, frontier_s = self._pending_ack
        self._pending_ack = None
        self._last_ack_wall = now
        self.bus.publish(AudioInAck(seq=seq, frontier_s=float(frontier_s)))

    # ---- hot loop (runner thread) ----------------------------------------------

    def sync_source(self, ctx: TickContext) -> None:
        return None

    def read_knobs(self) -> dict:
        return {}

    def has_pending_refit(self) -> bool:
        return False

    def rebuild_imminent(self, knobs: dict) -> bool:
        return False

    def produce(self, knobs: dict, ctx: TickContext, mode: ProduceMode) -> bool:
        """Nothing to generate. Paces the runner loop and publishes the
        coalesced ack a burst of frames left pending, so the last frame
        of a burst is acknowledged within :data:`ACK_INTERVAL_S`."""
        self._wake.wait(IDLE_WAIT_S)
        with self._lock:
            if not self._closed:
                self._flush_ack(self._clock())
        return False

    def render_window(self, t_start_s: float) -> Optional[AudioChunk]:
        return None

    def render_full(self) -> Optional[AudioChunk]:
        return None

    def has_renderable_state(self) -> bool:
        # Never: the runner must not gap-fill or re-render this canvas.
        return False

    def playable_duration_s(self) -> Optional[float]:
        return len(self.audio_eng.current) / SAMPLE_RATE

    def max_duration_s(self) -> Optional[float]:
        return MAX_CAPTURE_S

    def on_fresh_generation(self, knobs: dict) -> None:
        return None
