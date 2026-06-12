"""Lock-protected audio buffer with optional sounddevice playback.

Torch-free. sounddevice is imported lazily inside ``start`` so headless
test harnesses and websocket backends can use the buffer without PortAudio.
"""

import threading

import numpy as np


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
        with self._lock:
            self._old = self.current.copy()
            self.current = new_data
            self.swap_count += 1
            self._fading = True
            self._fade_pos = 0

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
        with self._lock:
            self.current[start_sample:end] = window[:n]

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
        with self._lock:
            self.current[start_sample:end] = fragment

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
