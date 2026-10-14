"""A swap_source upload keeps every channel up to a stream layout's
``encode_source``, and nowhere else. Tier 1, no GPU.

The wire decode keeps all channels for ``swap_source`` only; the session
hands them to a backend-owned swap but keeps the client buffer stereo;
SA3Backend passes them to the layout's own ``encode_source`` and clips to
stereo for the stock encode (no layout, or a layout without its own
``encode_source``).
"""

from __future__ import annotations

import struct

import numpy as np
import torch

from acestep.streaming.source import _decode_audio_msg
from tests.unit.test_sa3_stream_layout import (
    C, L, N_STREAMS, _LayoutRuntime, _knobs, _layout_backend, _run_until_fresh,
    _stock_backend,
)
from tests.unit.test_swap_resize_session import (
    OLD_PLAY, SAMPLE_RATE, _StubBackend, _SwapSession,
)


def _frame(ch: int, n: int) -> bytes:
    pcm = np.arange(ch * n, dtype=np.float32).reshape(n, ch)
    return struct.pack("<II", ch, n) + pcm.tobytes()


def test_decode_clips_to_stereo_by_default_and_keeps_all_on_request():
    msg = _frame(10, 16)
    assert tuple(_decode_audio_msg(msg).shape) == (2, 16)
    full = _decode_audio_msg(msg, max_channels=None)
    assert tuple(full.shape) == (10, 16)
    assert torch.equal(full[:2], _decode_audio_msg(msg))
    assert float(full[7, 3]) == 3 * 10 + 7     # channel 7 of frame 3


def test_session_hands_the_backend_every_channel_and_the_client_stereo():
    backend = _StubBackend(swap_resize=False)
    sess = _SwapSession(backend)
    wf = sess.swap(2.0, channels=10)
    assert backend.calls[-1]["channels"] == 10
    assert sess.state.n_channels == 2
    assert sess.audio_eng.current.shape == (OLD_PLAY, 2)
    head = sess.audio_eng.current[: wf.shape[-1]]
    assert np.array_equal(head, wf[:2].numpy().T)
    assert sess.bus.events[-1].channels == 2


def _record_encoder(b):
    seen = []
    shape = b._source_latent_btc.movedim(1, 2).shape

    def rec(waveform, sample_rate, sample_size):
        seen.append(int(waveform.shape[0]))
        return torch.zeros(shape)

    b._source_encoder = rec
    return seen


def test_layout_encode_source_receives_the_raw_channels():
    runtime = _LayoutRuntime({
        "encode_source": lambda ctx, audio, n: torch.zeros(1, C, N_STREAMS * L),
    })
    b, _context, _ = _layout_backend(runtime=runtime)
    _run_until_fresh(b, _knobs(b, sa3_denoise=0.7))
    seen = _record_encoder(b)
    b.handle_swap_source(torch.randn(10, 1000), SAMPLE_RATE)
    assert seen == [10]
    assert b._source_waveform.shape[0] == 10


def test_stock_encodes_stay_stereo():
    for b in (_layout_backend()[0], _stock_backend(None)):
        seen = _record_encoder(b)
        b.handle_swap_source(torch.randn(10, 1000), SAMPLE_RATE)
        assert seen == [2]
