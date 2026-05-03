"""Lock-protected audio buffer with sounddevice playback.

Torch-free. sounddevice is imported lazily inside ``__init__`` so headless
test harnesses can import this module without an audio device present.
"""

import threading

import numpy as np

from .protocol import CROSSFADE_SECONDS


class AudioEngine:
    """Lock-protected audio buffer with sounddevice playback."""

    def __init__(self, data, sr):
        import sounddevice as sd
        self._sd = sd
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        self.sr = sr
        self.channels = data.shape[1]
        self.current = data.copy()
        self.position = 0
        self.swap_count = 0
        self.crossfade_len = max(1, int(sr * CROSSFADE_SECONDS))
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
        self._stream = self._sd.OutputStream(
            samplerate=self.sr,
            channels=self.channels,
            callback=self._callback,
            blocksize=2048,
        )
        self._stream.start()

    def stop(self):
        self._stream.stop()
        self._stream.close()
