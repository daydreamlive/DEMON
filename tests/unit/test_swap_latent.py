"""A latent in a swap, a latent out. Tier 1, no GPU.

* ``swap_source`` may give streams as latents: a ``latent`` header field
  and ONE more binary frame after the PCM, read by the one-shot's reader.
  The adapter reads every frame before it checks anything, refuses a bad
  header or frame with ``swap_failed``, and hands ``{stream: [C, L]}`` to
  the session, which hands it to the backend's swap hook, which hands it to
  the layout's ``encode_source`` as ``latent=``.
* A backend or model that does not take a latent refuses the swap
  (``swap_latent`` off: the session; a layout without a ``latent`` keyword:
  the backend; no layout encode: the context).
* ``result_latent`` answers with the latent being rendered now, every
  stream, float16 ``[S, C, L]``, or ``ok`` false when there is none.
"""

from __future__ import annotations

import json
import struct
import threading
import types

import numpy as np
import pytest
import torch

from acestep.nodes.types import Audio
from acestep.streaming.events import ResultLatent, SwapFailed
from acestep.streaming.generator_backend import Capabilities
from acestep.streaming.oneshot import OneshotError
from acestep.streaming.session import StreamingSession
from acestep.streaming.source import swap_latent_streams
from demos.realtime_motion_graph_web.audio_codec import send_result_latent
from demos.realtime_motion_graph_web.ws_adapter import _dispatch_swap_source
from tests.unit.test_sa3_stream_layout import C, L, N_STREAMS, _LayoutRuntime, _layout_backend
from tests.unit.test_swap_resize_session import SAMPLE_RATE, _StubBackend, _SwapSession


def _latent(p, c=C, length=L, seed=0):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((p, c, length)).astype("<f2")


def _spec(arr, streams):
    return {"dtype": "float16", "shape": list(arr.shape), "streams": list(streams)}


def _pcm(ch=10, n=16):
    return struct.pack("<II", ch, n) + np.zeros((n, ch), np.float32).tobytes()


# ---- the header and the frame -------------------------------------------------


def test_streams_come_back_bit_for_bit_at_their_indices():
    arr = _latent(2)
    out = swap_latent_streams(_spec(arr, [1, 3]), arr.tobytes())
    assert sorted(out) == [1, 3]
    assert out[1].dtype == torch.float16 and tuple(out[1].shape) == (C, L)
    assert np.array_equal(out[1].numpy(), arr[0])
    assert np.array_equal(out[3].numpy(), arr[1])


@pytest.mark.parametrize("spec, cut, needle", [
    ({"shape": [2, C, L], "streams": [0, 1]}, 2, "bytes, expected"),
    ({"shape": [2, C], "streams": [0, 1]}, 0, "shape must be"),
    ({"shape": [2, C, 0], "streams": [0, 1]}, 0, "positive"),
    ({"dtype": "float32", "shape": [2, C, L], "streams": [0, 1]}, 0, "float16"),
    ({"shape": [2, C, L], "streams": [1, 1]}, 0, "no repeats"),
    ({"shape": [2, C, L], "streams": [3, 1]}, 0, "ascending"),
    ({"shape": [2, C, L], "streams": [0]}, 0, "must list 2"),
    ({"shape": [2, C, L], "streams": [0, -1]}, 0, "must list 2"),
    ({"shape": [2, C, L]}, 0, "must list 2"),
    ("not an object", 0, "must be an object"),
])
def test_bad_headers_and_frames_are_refused(spec, cut, needle):
    blob = _latent(2).tobytes()
    blob = blob[:-cut] if cut else blob
    with pytest.raises(OneshotError) as exc:
        swap_latent_streams(spec, blob)
    assert needle in str(exc.value)
    assert "swap_source.latent" in str(exc.value)


class _Streaming:
    def __init__(self):
        self.swaps: list = []

    def swap_source(self, audio, **kw):
        self.swaps.append((audio, kw))


def _dispatch(head, frames):
    reads, sent, rec = [], [], _Streaming()

    def recv(fail_type, **extra):
        reads.append(fail_type)
        return frames.pop(0) if frames else None

    _dispatch_swap_source(head, rec, recv, sent.append, "ws")
    return reads, sent, rec


def test_adapter_reads_the_pcm_then_the_latent_and_stages_both():
    arr = _latent(2)
    head = {"type": "swap_source", "keep": {"drums": [0, 1]},
            "latent": _spec(arr, [0, 2])}
    reads, sent, rec = _dispatch(head, [_pcm(), arr.tobytes()])
    assert reads == ["swap_failed", "swap_failed"] and sent == []
    audio, kw = rec.swaps[0]
    assert isinstance(audio, Audio) and tuple(audio.waveform.shape) == (10, 16)
    assert kw["keep"] == {"drums": [0, 1]}
    assert sorted(kw["latent"]) == [0, 2]
    assert np.array_equal(kw["latent"][2].numpy(), arr[1])


def test_adapter_without_a_latent_reads_one_frame_as_before():
    reads, sent, rec = _dispatch({"type": "swap_source"}, [_pcm()])
    assert reads == ["swap_failed"] and sent == []
    assert rec.swaps[0][1]["latent"] is None


def test_a_bad_latent_frame_is_answered_and_nothing_is_staged():
    arr = _latent(2)
    head = {"type": "swap_source", "latent": _spec(arr, [0, 1])}
    frames = [_pcm(), arr.tobytes()[:-2], b"next"]
    reads, sent, rec = _dispatch(head, frames)
    assert len(reads) == 2, "both frames read, the stream stays in step"
    assert frames == [b"next"]
    assert sent[0]["type"] == "swap_failed" and "bytes, expected" in sent[0]["error"]
    assert rec.swaps == []


def test_a_bad_pcm_frame_still_consumes_the_latent_frame():
    arr = _latent(1)
    head = {"type": "swap_source", "latent": _spec(arr, [0])}
    frames = [b"\0\0", arr.tobytes(), b"next"]
    reads, sent, rec = _dispatch(head, frames)
    assert frames == [b"next"] and len(reads) == 2
    assert sent[0]["type"] == "swap_failed" and rec.swaps == []


def test_a_missing_latent_frame_stops_the_swap():
    arr = _latent(1)
    head = {"type": "swap_source", "latent": _spec(arr, [0])}
    reads, sent, rec = _dispatch(head, [_pcm()])     # recv answers itself
    assert len(reads) == 2 and rec.swaps == []


# ---- session -> backend -> layout ---------------------------------------------


def _staging_session(caps):
    state = types.SimpleNamespace(
        _lock=threading.Lock(), swap_pending={}, prompt_text="p",
        last_activity_ts=0.0)
    eng = types.SimpleNamespace(render_anchor_s=1.0, set_render_anchor_queue=lambda q: None)
    bus = types.SimpleNamespace(events=[])
    bus.publish = bus.events.append
    backend = types.SimpleNamespace(name="stub", capabilities=lambda: caps)
    return types.SimpleNamespace(state=state, audio_eng=eng, bus=bus, backend=backend)


def test_session_stages_the_latent_when_the_backend_takes_it():
    fake = _staging_session(Capabilities(swap=True, swap_latent=True))
    lat = {0: torch.zeros(C, L)}
    swap = StreamingSession.swap_source.__wrapped__
    swap(fake, Audio(waveform=torch.zeros(10, 8), sample_rate=SAMPLE_RATE), latent=lat)
    assert fake.state.swap_pending["latent"] == lat
    assert fake.bus.events == []
    swap(fake, Audio(waveform=torch.zeros(10, 8), sample_rate=SAMPLE_RATE))
    assert fake.state.swap_pending["latent"] is None


def test_session_refuses_a_latent_the_backend_does_not_take():
    fake = _staging_session(Capabilities(swap=True))
    swap = StreamingSession.swap_source.__wrapped__
    swap(fake, Audio(waveform=torch.zeros(10, 8), sample_rate=SAMPLE_RATE),
         latent={0: torch.zeros(C, L)})
    assert "waveform" not in fake.state.swap_pending, "nothing staged"
    assert isinstance(fake.bus.events[-1], SwapFailed)
    assert "no latent" in fake.bus.events[-1].error


class _LatentStub(_StubBackend):
    def handle_swap_source(self, waveform, sample_rate, duration_s=None, **kw):
        self.kw = kw
        return super().handle_swap_source(waveform, sample_rate, duration_s)


def test_backend_owned_swap_hands_the_latent_to_the_hook():
    b = _LatentStub(swap_resize=False)
    sess = _SwapSession(b)
    lat = {1: torch.zeros(C, L)}
    sess._apply_swap_backend_owned(
        b.handle_swap_source, torch.zeros(10, 1000), None, None, None, None, lat,
    )
    assert b.kw == {"latent": lat}
    assert type(sess.bus.events[-1]).__name__ == "SwapReady"
    sess._apply_swap_backend_owned(b.handle_swap_source, torch.zeros(10, 1000), None)
    assert b.kw == {}


class _LatentRuntime(_LayoutRuntime):
    """A layout whose encode_source takes ``latent`` and records it."""

    def __init__(self):
        self.seen: list = []
        super().__init__({"encode_source": self._enc})

    def _enc(self, ctx, audio, n, keep=None, grow=None, latent=None):
        self.seen.append(latent)
        return torch.zeros(1, C, N_STREAMS * L)


def _recording_encoder(b):
    seen = []

    def enc(waveform, sample_rate, sample_size, **kw):
        seen.append(kw)
        return torch.zeros(1, C, N_STREAMS * L)

    b._source_encoder = enc
    return seen


def test_a_latent_reaches_the_encoder_on_a_layout_that_takes_it():
    b, _context, _ = _layout_backend(runtime=_LatentRuntime())
    assert b.capabilities().swap_latent is True
    seen = _recording_encoder(b)
    lat = {0: torch.ones(C, L, dtype=torch.float16)}
    b.handle_swap_source(torch.zeros(10, 1000), SAMPLE_RATE, latent=lat)
    assert seen == [{"latent": lat}]
    b.handle_swap_source(torch.zeros(10, 1000), SAMPLE_RATE)
    assert seen[-1] == {}


def test_a_latent_for_a_stream_the_session_lacks_is_refused():
    b, _context, _ = _layout_backend(runtime=_LatentRuntime())
    seen = _recording_encoder(b)
    with pytest.raises(ValueError, match="streams 0..2"):
        b.handle_swap_source(torch.zeros(10, 1000), SAMPLE_RATE,
                             latent={N_STREAMS: torch.zeros(C, L)})
    assert seen == []


def test_a_model_without_latent_support_refuses():
    plain = _LayoutRuntime({
        "encode_source": lambda ctx, a, n, **kw: torch.zeros(1, C, N_STREAMS * L),
    })
    for runtime in (plain, None):
        b, _context, _ = _layout_backend(runtime=runtime)
        assert b.capabilities().swap_latent is False
        seen = _recording_encoder(b)
        with pytest.raises(RuntimeError, match="no latent"):
            b.handle_swap_source(torch.zeros(10, 1000), SAMPLE_RATE,
                                 latent={0: torch.zeros(C, L)})
        assert seen == [], "refused before any encode"


def test_session_turns_a_refusing_backend_into_swap_failed():
    b, _context, _ = _layout_backend()
    sess = _SwapSession(b)
    sess._apply_swap_backend_owned(
        b.handle_swap_source, torch.zeros(10, 1000), None, None, None, None,
        {0: torch.zeros(C, L)},
    )
    assert isinstance(sess.bus.events[-1], SwapFailed)
    assert "no latent" in sess.bus.events[-1].error


def test_context_passes_the_latent_to_the_layout_and_refuses_without_one():
    from acestep.engine.sa3_context import SA3Context
    from acestep.plugins.model_extensions import StreamLayout

    runtime = _LatentRuntime()
    ctx = types.SimpleNamespace(stream_layout=lambda: runtime.layout)
    lat = {2: torch.zeros(C, L)}
    SA3Context.encode_source(ctx, (SAMPLE_RATE, torch.zeros(2, 8)), 4, latent=lat)
    assert runtime.seen == [lat]
    for layout in (StreamLayout(n_streams=N_STREAMS), None):
        ctx = types.SimpleNamespace(stream_layout=lambda _l=layout: _l)
        with pytest.raises(RuntimeError, match="no latent"):
            SA3Context.encode_source(
                ctx, (SAMPLE_RATE, torch.zeros(2, 8)), 4, latent=lat,
            )


# ---- result_latent --------------------------------------------------------------


def test_backend_result_latent_is_every_stream_side_by_side():
    b, _context, _ = _layout_backend()
    assert b.capabilities().result_latent is True
    b._current_result = None
    b._last_result_latent = None
    assert b.result_latent() is None
    torch.manual_seed(1)
    native = torch.randn(1, C, N_STREAMS * L)
    b._last_result_latent = native.movedim(1, 2).contiguous()
    out = b.result_latent()
    assert out.dtype == np.float16 and out.shape == (N_STREAMS, C, L)
    for k in range(N_STREAMS):
        want = native[0, :, k * L:(k + 1) * L].half().numpy()
        assert np.array_equal(out[k], want)
    # this tick's result wins over the cached one
    b._current_result = torch.zeros_like(b._last_result_latent)
    assert not b.result_latent().any()


def test_result_latent_answers_the_streams_the_layout_holds():
    torch.manual_seed(2)
    take = torch.randn(C, L, dtype=torch.float32)
    held = {1: take, 2: torch.ones(C, L + 1), 0: "not a tensor"}
    b, _context, _ = _layout_backend(
        runtime=_LayoutRuntime({"held_latent": lambda: held}),
    )
    b._current_result = None
    native = torch.randn(1, C, N_STREAMS * L)
    b._last_result_latent = native.movedim(1, 2).contiguous()
    out = b.result_latent()
    assert out.dtype == np.float16 and out.shape == (N_STREAMS, C, L)
    assert np.array_equal(out[1], take.half().numpy()), "a held stream is its take"
    for k in (0, 2):  # another shape, or not a tensor: the rendered stream stays
        assert np.array_equal(out[k], native[0, :, k * L:(k + 1) * L].half().numpy())
    held.clear()
    assert np.array_equal(
        b.result_latent()[1], native[0, :, L:2 * L].half().numpy(),
    ), "nothing held: the rendered latent only"


class _ResultSession:
    result_latent = StreamingSession.result_latent
    _service_result_latents = StreamingSession._service_result_latents

    def __init__(self, backend):
        self.backend = backend
        self.state = types.SimpleNamespace(
            _lock=threading.Lock(), last_activity_ts=0.0, source_epoch=5)
        self.bus = types.SimpleNamespace(events=[])
        self.bus.publish = self.bus.events.append
        self._result_latents: list = []


class _ResultBackend:
    name = "stub"

    def __init__(self, latent, caps=True):
        self.latent = latent
        self.copies = 0
        self._caps = Capabilities(result_latent=caps)

    def capabilities(self):
        return self._caps

    def result_latent(self):
        self.copies += 1
        return self.latent


def test_session_answers_queued_requests_with_one_copy():
    lat = _latent(N_STREAMS)
    sess = _ResultSession(_ResultBackend(lat))
    sess.result_latent("a")
    sess.result_latent("b")
    assert sess.bus.events == [], "answered on the session thread"
    sess._service_result_latents()
    assert sess.backend.copies == 1
    assert [e.token for e in sess.bus.events] == ["a", "b"]
    for e in sess.bus.events:
        assert isinstance(e, ResultLatent) and e.ok and e.source_epoch == 5
        assert e.latent is lat and e.message == ""
    sess._service_result_latents()
    assert sess.backend.copies == 1, "nothing queued, nothing copied"


def test_session_answers_not_ok_without_a_result():
    sess = _ResultSession(_ResultBackend(None))
    sess.result_latent("a")
    sess._service_result_latents()
    e = sess.bus.events[-1]
    assert not e.ok and e.latent is None and "no latent" in e.message


def test_session_answers_at_once_when_the_backend_has_none():
    sess = _ResultSession(_ResultBackend(None, caps=False))
    sess.result_latent("a")
    e = sess.bus.events[-1]
    assert not e.ok and e.token == "a" and "has no result latent" in e.message
    assert sess._result_latents == []


class _Ws:
    def __init__(self):
        self.sent: list = []

    def send(self, data):
        if not isinstance(data, (str, bytes)):
            data = b"".join(bytes(p) for p in data)     # a fragmented message
        self.sent.append(data)


def test_reply_carries_the_header_and_one_frame():
    lat = _latent(N_STREAMS, length=400)                 # > 256 KiB: fragmented
    ws = _Ws()
    send_result_latent(ws, token="t", ok=True, message="", latent=lat, source_epoch=7)
    head = json.loads(ws.sent[0])
    assert head == {
        "type": "result_latent", "token": "t", "ok": True, "message": "",
        "source_epoch": 7,
        "latent": {"dtype": "float16", "shape": [N_STREAMS, C, 400],
                   "bytes": lat.nbytes},
    }
    assert len(ws.sent) == 2
    back = np.frombuffer(ws.sent[1], dtype="<f2").reshape(head["latent"]["shape"])
    assert np.array_equal(back, lat)


def test_reply_without_a_result_has_no_frame():
    ws = _Ws()
    send_result_latent(ws, token="t", ok=False, message="none yet", latent=None,
                       source_epoch=7)
    assert len(ws.sent) == 1
    head = json.loads(ws.sent[0])
    assert head["ok"] is False and "latent" not in head and head["message"] == "none yet"
