"""Lean uploads and replies for wide sources. Tier 1, no GPU.

A ``swap_source`` frame described by ``pcm`` carries 16-bit samples of only
the channels with audio; the adapter expands it to the full float32 tensor
the plain frame decodes to. A ``derived_windows`` client builds the windows
it can derive itself: ``ready`` and ``swap_ready`` leave out a silent buffer,
and ``swap_ready`` a kept one.
"""

from __future__ import annotations

import struct
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from acestep.streaming.config import SessionConfig
from acestep.streaming.source import _decode_audio_msg, _decode_compact_pcm
from demos.realtime_motion_graph_web.ws_adapter import _derived_window, _server_capabilities


def _wide(n: int = 480, ch: int = 10, live=(4, 5)) -> np.ndarray:
    """[n, ch] float32, silent but for the `live` channels."""
    out = np.zeros((n, ch), dtype=np.float32)
    out[:, list(live)] = np.random.default_rng(1).uniform(-0.9, 0.9, (n, len(live)))
    return out


def _plain(a: np.ndarray) -> bytes:
    return struct.pack("<II", a.shape[1], a.shape[0]) + a.tobytes()


def _compact(a: np.ndarray, present, enc="s16le") -> bytes:
    part = a[:, present]
    data = np.round(part * 32768).clip(-32768, 32767).astype(np.int16) if enc == "s16le" else part
    return struct.pack("<II", len(present), a.shape[0]) + data.tobytes()


def test_compact_frame_expands_to_the_plain_one():
    a = _wide()
    plain = _decode_audio_msg(_plain(a), max_channels=None)
    s16 = _decode_compact_pcm(_compact(a, [4, 5]), {"encoding": "s16le", "channels": 10, "present": [4, 5]})
    assert tuple(s16.shape) == (10, 480)
    assert float((s16 - plain).abs().max()) <= 0.5 / 32768 + 1e-7
    assert not s16[[0, 1, 2, 3, 6, 7, 8, 9]].any()
    f32 = _decode_compact_pcm(_compact(a, [4, 5], "f32le"), {"encoding": "f32le", "channels": 10, "present": [4, 5]})
    assert np.array_equal(f32.numpy(), plain.numpy())
    # present in any order lands each channel where it belongs; default present = all
    swapped = _decode_compact_pcm(_compact(a, [5, 4]), {"encoding": "s16le", "channels": 10, "present": [5, 4]})
    assert np.array_equal(swapped.numpy(), s16.numpy())
    whole = _decode_compact_pcm(_compact(a, list(range(10)), "f32le"), {"channels": 10})
    assert np.array_equal(whole.numpy(), plain.numpy())


@pytest.mark.parametrize("pcm, frame_ch", [
    ({"encoding": "s24le", "channels": 10, "present": [4, 5]}, 2),
    ({"encoding": "s16le", "channels": 0, "present": [0]}, 1),
    ({"encoding": "s16le", "channels": 17, "present": [0]}, 1),
    ({"encoding": "s16le", "channels": True, "present": [0]}, 1),
    ({"encoding": "s16le", "channels": 10, "present": []}, 0),
    ({"encoding": "s16le", "channels": 10, "present": [4, 10]}, 2),
    ({"encoding": "s16le", "channels": 10, "present": [4, 4]}, 2),
    ({"encoding": "s16le", "channels": 10, "present": [4, 5]}, 3),  # header disagrees with present
    ("s16le", 2),
])
def test_compact_frame_rejects_a_bad_description(pcm, frame_ch):
    msg = struct.pack("<II", frame_ch, 8) + np.zeros((8, frame_ch), np.int16).tobytes()
    with pytest.raises(ValueError):
        _decode_compact_pcm(msg, pcm)


def test_compact_frame_rejects_a_short_body_and_an_oversized_expansion():
    pcm = {"encoding": "s16le", "channels": 10, "present": [4, 5]}
    msg = _compact(_wide(), [4, 5])
    with pytest.raises(ValueError):
        _decode_compact_pcm(msg[:-2], pcm)
    huge = struct.pack("<II", 1, 1 << 26)  # 67M samples x 16 channels: 4 GiB, refused before the body is read
    with pytest.raises(ValueError, match="expands"):
        _decode_compact_pcm(huge, {"encoding": "s16le", "channels": 16, "present": [0]})


def test_derived_windows_is_opt_in_and_names_what_the_client_builds():
    silent, loud = np.zeros((48, 10), np.float32), _wide(n=48)
    assert SessionConfig().derived_windows is False
    for buf, kept in ((silent, False), (loud, True)):
        assert _derived_window({}, buf, kept=kept) is None, "without the flag every frame goes"
    on = {"derived_windows": True}
    assert _derived_window(on, silent) == "silent"
    assert _derived_window(on, loud) is None, "audio the client can't derive goes"
    assert _derived_window(on, loud, kept=True) == "kept"


def test_server_capabilities_advertise_compact_pcm():
    caps = _server_capabilities({"swap": True})
    assert caps["compact_pcm"] is True and caps["swap"] is True
    assert "midi_transcribe" in caps
