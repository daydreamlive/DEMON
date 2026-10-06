"""Window head crossfade near the loop end (short SA3 SFX loops hit it
every lap): the ramp must clip to the samples left in the buffer."""

import numpy as np

from acestep.streaming.pipeline_runner import _fade_in_over


def test_fade_in_clips_at_buffer_end():
    current = np.ones((192000, 2), np.float32)
    win = np.zeros((48000, 2), np.float32)
    start = current.shape[0] - 1108          # 1108 samples left < 1200 ramp
    _fade_in_over(win, current, start, 1200)
    assert win[0, 0] == 1.0                  # starts on the live buffer
    assert win[1107, 0] == 0.0               # ramp ends on the window
    assert np.all(win[1108:] == 0.0)


def test_fade_in_full_ramp_mid_buffer_and_none_at_zero():
    current = np.ones((192000, 2), np.float32)
    win = np.zeros((48000, 2), np.float32)
    _fade_in_over(win, current, 96000, 1200)
    assert win[0, 0] == 1.0 and win[1199, 0] == 0.0 and win[600, 0] > 0.0
    win0 = np.zeros((48000, 2), np.float32)
    _fade_in_over(win0, current, 0, 1200)
    assert np.all(win0 == 0.0)
