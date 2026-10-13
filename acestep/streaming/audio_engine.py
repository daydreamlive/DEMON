"""Lock-protected audio buffer with optional sounddevice playback.

Torch-free. sounddevice is imported lazily inside ``start`` so headless
test harnesses and websocket backends can use the buffer without PortAudio.

The buffer is versioned at the write point: every write (``swap``,
``patch_window``, ``patch``) bumps :attr:`AudioEngine.revision` under the
lock and, after the lock is released, tells each registered write
listener which samples changed. Readers that must not trust the bus as a
record of writes (the analysis observer) use the listener plus
:meth:`AudioEngine.snapshot`.
"""

import threading

import numpy as np
from loguru import logger

#: ``kind`` values passed to write listeners.
WRITE_SWAP = "swap"
WRITE_PATCH_WINDOW = "patch_window"
WRITE_PATCH = "patch"


# Server-side crossfade length used when ``swap`` is called between
# buffers (e.g. on source swap). Mirrors the TS/audio-worklet constant
# of the same name; kept here as a private duplicate so this module
# stays free of demo / wire-protocol imports.
CROSSFADE_SECONDS = 0.025


class AudioEngine:
    """Lock-protected audio buffer with sounddevice playback."""

    def __init__(self, data, sr, *, crossfade_seconds: float = CROSSFADE_SECONDS):
        self._sd = None
        self._stream = None
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        self.sr = sr
        self.channels = data.shape[1]
        self.current = data.copy()
        self.position = 0
        # Estimated queueing delay (seconds) of the report that last set
        # ``position``, written by the session alongside it (see
        # StreamingSession.set_knobs / pipeline_runner.
        # ReportStalenessEstimator). The runner's playhead clock advances
        # its anchor by this much so delayed reports don't drag the
        # estimate into the past. 0.0 when reports carry no send stamp.
        self.position_staleness_s = 0.0
        # ``time.monotonic()`` of the last client playhead report that set
        # ``position`` (0.0 = never). Lets readers tell a live reported
        # playhead from a stale one when no local output stream runs.
        self.position_report_wall_s = 0.0
        # Explicit stationary render placement in absolute buffer seconds.
        # None means normal transport-driven placement. The params command
        # owns its sticky absent/value/null lifetime; source swaps hard-clear it.
        self.render_anchor_s = None
        # Batch stationary placement: a FIFO of anchor seconds the runner
        # renders back-to-back (one window per tick, popped after its window
        # is emitted) whenever no scalar ``render_anchor_s`` is set — the
        # scalar always preempts, the queue resumes when it clears. Set
        # cross-thread by the params command (replace semantics; empty/null
        # clears); source swaps hard-clear it alongside the scalar anchor.
        # Guarded by its own lock: the buffer ``_lock`` sits on the audio
        # callback hot path and queue ops must never contend with it.
        self._render_anchor_queue: list = []
        self._render_anchor_queue_gen = 0
        self._render_anchor_queue_lock = threading.Lock()
        # Client-observed slice landing lead (seconds; negative = the
        # slice patched audio the listener already played) and the wall
        # time the report arrived. None until the client sends one. The
        # runner's transport-lead controller folds each new report into
        # its playback lead so slices land ahead of the ear even with
        # network transit and client main-thread scheduling in the path.
        self.observed_slice_lead_s = None
        self.observed_slice_lead_wall_s = 0.0
        self.swap_count = 0
        # Active client loop band as ``(start_sec, end_sec)`` or ``None``.
        # Set cross-thread by the WS recv thread (backend ``loop_band``
        # handler) and read by the PipelineRunner to wrap its decode target
        # inside the band. A single immutable-tuple attribute so the read is
        # one atomic reference load under the GIL — no torn start/end.
        self.loop_band = None
        self.crossfade_len = max(1, int(sr * crossfade_seconds))
        self._old = None
        self._fading = False
        self._fade_pos = 0
        self._lock = threading.Lock()
        # Monotonic canvas revision, bumped under ``_lock`` on every write.
        self.revision = 0
        # Write listeners. Replaced (copy-on-write), never mutated in place,
        # so a writer reads the tuple once without taking a second lock.
        self._write_listeners: tuple = ()
        self._listeners_lock = threading.Lock()

    # ---- write listeners + snapshot ----------------------------------------

    def add_write_listener(self, fn) -> None:
        """Register ``fn(start_sample, end_sample, revision, kind)``.

        Called on the writer's thread AFTER the buffer lock is released,
        once per write; ``kind`` is ``"swap"``, ``"patch_window"`` or
        ``"patch"``. A swap reports the whole new buffer. A listener that
        raises is logged and removed, so a broken observer can never stall
        the runner."""
        with self._listeners_lock:
            if fn not in self._write_listeners:
                self._write_listeners = self._write_listeners + (fn,)

    def remove_write_listener(self, fn) -> None:
        """Unregister ``fn``; a no-op when it is not registered."""
        with self._listeners_lock:
            self._write_listeners = tuple(f for f in self._write_listeners if f != fn)

    def _notify_write(self, listeners, start, end, revision, kind) -> None:
        for fn in listeners:
            try:
                fn(start, end, revision, kind)
            except Exception:  # noqa: BLE001 - a listener must never break a write
                logger.opt(exception=True).error(
                    "audio_write_listener_failed listener={!r}; removing it", fn,
                )
                self.remove_write_listener(fn)

    def snapshot(self, start_sample, end_sample):
        """``(copy of current[start:end], revision)`` read under the lock.

        Bounds are clipped to the buffer. The copy and the revision are
        consistent: no write lands between them."""
        with self._lock:
            n = len(self.current)
            s0 = max(0, min(int(start_sample), n))
            s1 = max(s0, min(int(end_sample), n))
            return self.current[s0:s1].copy(), self.revision

    # ---- render-anchor queue (see __init__) --------------------------------

    def set_render_anchor_queue(self, anchors) -> None:
        """Replace the queued stationary anchors (seconds). Empty clears.

        Bumps the queue GENERATION: a tick that peeked the old queue must
        not pop from the replacement. A same-value head is not enough to
        detect that (ABA: replacing ``[1, 2]`` with ``[1, 3]`` while ``1``
        renders would let the old tick pop the NEW queue's ``1``)."""
        anchors = [float(a) for a in (anchors or [])]
        with self._render_anchor_queue_lock:
            self._render_anchor_queue = anchors
            self._render_anchor_queue_gen += 1

    def peek_render_anchor(self):
        """``(head_anchor_s, generation)`` of the queue, or ``None``."""
        with self._render_anchor_queue_lock:
            q = self._render_anchor_queue
            if not q:
                return None
            return q[0], self._render_anchor_queue_gen

    def pop_render_anchor(self, expected, generation) -> None:
        """Pop the queue head iff it still equals ``expected`` AND the queue
        has not been replaced since the peek (``generation`` matches).

        The runner peeks at tick start and pops only after the window is
        emitted; the guards make a client replace mid-tick safe."""
        with self._render_anchor_queue_lock:
            if self._render_anchor_queue_gen != generation:
                return
            q = self._render_anchor_queue
            if q and q[0] == expected:
                q.pop(0)

    @property
    def duration(self):
        return len(self.current) / self.sr

    @property
    def playback_position(self):
        return self.position / self.sr

    def swap(self, new_data):
        if new_data.ndim == 1:
            new_data = new_data.reshape(-1, 1)
        if new_data.shape[1] != self.channels:
            if self.channels == 2 and new_data.shape[1] == 1:
                new_data = np.column_stack([new_data, new_data])
            elif self.channels == 1 and new_data.shape[1] == 2:
                new_data = new_data.mean(axis=1, keepdims=True)
        listeners = self._write_listeners
        with self._lock:
            self._old = self.current.copy()
            self.current = new_data
            self.swap_count += 1
            self._fading = True
            self._fade_pos = 0
            self.revision += 1
            revision = self.revision
        if listeners:
            self._notify_write(listeners, 0, len(new_data), revision, WRITE_SWAP)

    def patch_window(self, window, start_sample):
        """Write a window of audio into ``self.current`` in place.

        Hot-path replacement for the runner's old
        ``buf = audio_eng.current.copy(); ...; audio_eng.swap(buf)``
        idiom, which moved ~46 MB of host RAM per windowed decode for a
        60 s buffer (one copy in the runner, one in ``swap``). Holds the
        lock only for the slice-assign so the audio callback's reads stay
        consistent; no global crossfade is triggered — the runner is
        expected to have already crossfaded the window's leading and
        trailing edges against ``self.current`` before calling.

        Multi-channel coercion mirrors :meth:`swap` so callers that
        produce mono or shape-mismatched windows behave identically.
        """
        if window.ndim == 1:
            window = window.reshape(-1, 1)
        if window.shape[1] != self.channels:
            if self.channels == 2 and window.shape[1] == 1:
                window = np.column_stack([window, window])
            elif self.channels == 1 and window.shape[1] == 2:
                window = window.mean(axis=1, keepdims=True)
        buf_len = len(self.current)
        end = min(start_sample + len(window), buf_len)
        n = end - start_sample
        if n <= 0:
            return
        listeners = self._write_listeners
        with self._lock:
            self.current[start_sample:end] = window[:n]
            self.revision += 1
            revision = self.revision
        if listeners:
            self._notify_write(listeners, start_sample, end, revision, WRITE_PATCH_WINDOW)

    def patch(self, data, start_sample):
        """Write audio into the buffer at *start_sample* in-place."""
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        if data.shape[1] != self.channels:
            if self.channels == 2 and data.shape[1] == 1:
                data = np.column_stack([data, data])
            elif self.channels == 1 and data.shape[1] == 2:
                data = data.mean(axis=1, keepdims=True)
        buf_len = len(self.current)
        end = min(start_sample + len(data), buf_len)
        actual = end - start_sample
        if actual <= 0:
            return
        fragment = data[:actual].copy()
        xfade = min(int(self.sr * 0.001), actual // 4)
        if xfade > 0 and start_sample > 0:
            t = np.linspace(0.0, 1.0, xfade).reshape(-1, 1)
            old = self.current[start_sample:start_sample + xfade]
            fragment[:xfade] = old * (1.0 - t) + fragment[:xfade] * t
        if xfade > 0 and end < buf_len:
            t = np.linspace(1.0, 0.0, xfade).reshape(-1, 1)
            old = self.current[end - xfade:end]
            fragment[-xfade:] = fragment[-xfade:] * t + old * (1.0 - t)
        listeners = self._write_listeners
        with self._lock:
            self.current[start_sample:end] = fragment
            self.revision += 1
            revision = self.revision
        if listeners:
            self._notify_write(listeners, start_sample, end, revision, WRITE_PATCH)

    def _fill(self, buf, src, pos, frames):
        n = len(src)
        written = 0
        p = pos % n
        while written < frames:
            chunk = min(frames - written, n - p)
            buf[written:written + chunk] = src[p:p + chunk]
            written += chunk
            p = (p + chunk) % n

    def _callback(self, outdata, frames, _time_info, _status):
        n = len(self.current)
        if n == 0:
            outdata[:] = 0
            return
        with self._lock:
            out = np.zeros((frames, self.channels), dtype="float32")
            self._fill(out, self.current, self.position, frames)
            if self._fading and self._old is not None:
                old_out = np.zeros((frames, self.channels), dtype="float32")
                self._fill(old_out, self._old, self.position, frames)
                fade_frames = min(frames, self.crossfade_len - self._fade_pos)
                t = np.linspace(
                    self._fade_pos / self.crossfade_len,
                    (self._fade_pos + fade_frames) / self.crossfade_len,
                    fade_frames,
                ).reshape(-1, 1)
                out[:fade_frames] = old_out[:fade_frames] * (1 - t) + out[:fade_frames] * t
                self._fade_pos += fade_frames
                if self._fade_pos >= self.crossfade_len:
                    self._fading = False
                    self._old = None
            self.position = (self.position + frames) % n
        outdata[:] = out

    def start(self):
        if self._sd is None:
            import sounddevice as sd
            self._sd = sd
        self._stream = self._sd.OutputStream(
            samplerate=self.sr,
            channels=self.channels,
            callback=self._callback,
            blocksize=2048,
        )
        self._stream.start()

    def stop(self):
        if self._stream is None:
            return
        self._stream.stop()
        self._stream.close()
        self._stream = None
