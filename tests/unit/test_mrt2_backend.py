"""MRT2 backend family: frame protocol + append-only frontier semantics.

Covers the pieces that don't need the sidecar (or a GPU):

* protocol framing round-trips over a real socketpair,
* the rolling-window emission contract (overlap head, wrap-seam clamp,
  start_sample placement) with an injected fake sidecar client,
* credit pacing and knob forwarding,
* the family's declared contract surface (capabilities / geometry /
  knob universe registration — the homonym guard in
  test_knob_homonyms.py picks the universe up automatically).

The live path (real sidecar in the MRT2 venv, WS session end-to-end)
is exercised manually; see scripts/mrt2_sidecar.py.
"""

import socket
import time

import numpy as np
import pytest

from acestep.streaming.generator_backend import Capabilities, TickContext
from acestep.streaming.knobs import KnobState
from acestep.streaming.mrt2 import protocol as mp
from acestep.streaming.mrt2.backend import (
    WINDOW_S,
    WINDOW_SAMPLES,
    XFADE,
    MRT2Backend,
    mrt2_knob_specs,
)


# ---------------------------------------------------------------------------
# Protocol framing
# ---------------------------------------------------------------------------


def test_protocol_json_roundtrip():
    a, b = socket.socketpair()
    try:
        msg = {"type": "knobs", "temperature": 1.7, "top_k": 64}
        a.sendall(mp.pack_json(msg))
        kind, payload = mp.recv_msg(b)
        assert kind == mp.MSG_JSON
        assert mp.unpack_json(payload) == msg
    finally:
        a.close()
        b.close()


def test_protocol_audio_roundtrip():
    a, b = socket.socketpair()
    try:
        pcm = np.arange(2 * mp.FRAME_SAMPLES * mp.CHANNELS, dtype=np.float32)
        a.sendall(mp.pack_audio(123, 2, pcm.tobytes()))
        kind, payload = mp.recv_msg(b)
        assert kind == mp.MSG_AUDIO
        idx, n, raw = mp.unpack_audio(payload)
        assert (idx, n) == (123, 2)
        out = np.frombuffer(raw, dtype=np.float32)
        np.testing.assert_array_equal(out, pcm)
    finally:
        a.close()
        b.close()


def test_protocol_rejects_corrupt_length():
    a, b = socket.socketpair()
    try:
        a.sendall(b"\xff\xff\xff\xff")
        with pytest.raises(ConnectionError):
            mp.recv_msg(b)
    finally:
        a.close()
        b.close()


# ---------------------------------------------------------------------------
# Backend with a fake sidecar client
# ---------------------------------------------------------------------------


class FakeClient:
    """Stands in for SidecarClient: records control sends, lets tests
    feed audio frames directly."""

    def __init__(self):
        self.sent: list = []
        self.lost = False
        self.last_rx_ts = time.monotonic()
        self.frames_received = 0
        self.meta = {"type": "meta", "model": "fake"}
        self.closed = False
        self._audio: list = []

    def send_json(self, obj):
        self.sent.append(obj)

    def pop_audio(self):
        out, self._audio = self._audio, []
        return out

    def close(self):
        self.closed = True

    # test helper
    def feed_frames(self, num_frames: int, value: float = 0.5):
        arr = np.full(
            (num_frames * mp.FRAME_SAMPLES, mp.CHANNELS), value, np.float32,
        )
        self._audio.append(arr)
        self.frames_received += num_frames
        self.last_rx_ts = time.monotonic()


class _State:
    prompt_text = "test prompt"
    prompt_text_b = "test prompt b"

    def __init__(self):
        self.params = {}


def make_backend():
    client = FakeClient()
    backend = MRT2Backend(
        config=None,
        state=_State(),
        midi_knobs=KnobState(mrt2_knob_specs()),
        client=client,
    )
    return backend, client


def _ctx(playhead_s=0.0):
    return TickContext(playhead_s=playhead_s, buffer_duration_s=WINDOW_S)


def test_constructor_sends_initial_prompt():
    backend, client = make_backend()
    prompts = [m for m in client.sent if m["type"] == "prompt"]
    assert prompts and prompts[0]["tags"] == "test prompt"
    assert prompts[0]["tags_b"] == "test prompt b"


def test_contract_surface():
    backend, _ = make_backend()
    assert backend.capabilities() == Capabilities()  # everything False
    g = backend.geometry()
    assert (g.sample_rate, g.channels) == (mp.SAMPLE_RATE, mp.CHANNELS)
    assert g.chunk_rate_hz == pytest.approx(25.0)
    assert g.duration_s == WINDOW_S
    assert backend.playable_duration_s() == WINDOW_S
    assert backend.vae_window > 0
    assert not backend.has_renderable_state()
    assert backend.render_full() is None
    names = {s.name for s in backend.knob_specs()}
    assert all(n.startswith("mrt2_") for n in names)


def test_produce_grants_credit_for_the_lead():
    backend, client = make_backend()
    knobs = backend.read_knobs()
    backend.sync_source(_ctx(0.0))
    fresh = backend.produce(knobs, _ctx(0.0), "generate")
    assert fresh is False  # nothing fed yet
    credits = [m for m in client.sent if m["type"] == "credit"]
    assert len(credits) == 1
    # default mrt2_lead = 0.75 s -> ceil(0.75 * 48000 / 1920) = 19 frames
    assert credits[0]["frames"] == 19
    # No double-grant while the first is outstanding.
    backend.produce(knobs, _ctx(0.0), "generate")
    assert len([m for m in client.sent if m["type"] == "credit"]) == 1


def test_emission_overlap_and_placement():
    backend, client = make_backend()
    knobs = backend.read_knobs()

    client.feed_frames(5, value=0.25)
    backend.sync_source(_ctx(0.0))
    assert backend.produce(knobs, _ctx(0.0), "generate") is True

    chunk1 = backend.render_window(0.0)
    assert chunk1 is not None
    assert chunk1.start_sample == 0  # first chunk: no overlap head
    assert chunk1.pcm.shape == (5 * mp.FRAME_SAMPLES, mp.CHANNELS)
    assert backend.has_renderable_state()

    # Second batch: chunk must re-emit the previous XFADE samples at
    # its head so the runner's leading-edge crossfade blends identical
    # audio.
    client.feed_frames(3, value=0.75)
    assert backend.produce(knobs, _ctx(0.0), "generate") is True
    chunk2 = backend.render_window(0.0)
    assert chunk2.start_sample == 5 * mp.FRAME_SAMPLES - XFADE
    assert chunk2.pcm.shape[0] == 3 * mp.FRAME_SAMPLES + XFADE
    np.testing.assert_array_equal(
        chunk2.pcm[:XFADE],
        np.full((XFADE, mp.CHANNELS), 0.25, np.float32),
    )
    np.testing.assert_array_equal(
        chunk2.pcm[XFADE:],
        np.full((3 * mp.FRAME_SAMPLES, mp.CHANNELS), 0.75, np.float32),
    )

    # Runner mutates chunks in place (crossfade); the backend's tail
    # copy must be unaffected.
    chunk2.pcm[:] = -1.0
    client.feed_frames(1, value=0.5)
    backend.produce(knobs, _ctx(0.0), "generate")
    chunk3 = backend.render_window(0.0)
    np.testing.assert_array_equal(
        chunk3.pcm[:XFADE],
        np.full((XFADE, mp.CHANNELS), 0.75, np.float32),
    )


def test_emission_clamps_at_window_wrap():
    backend, client = make_backend()
    knobs = backend.read_knobs()

    # Park the frontier just shy of the window seam.
    short = 1000
    backend._abs_written = WINDOW_SAMPLES - short
    backend._tail = np.full((XFADE, mp.CHANNELS), 0.1, np.float32)

    client.feed_frames(2, value=0.9)  # 3840 samples > room
    backend.sync_source(_ctx(0.0))
    backend.produce(knobs, _ctx(0.0), "generate")

    pre = backend.render_window(0.0)
    # Clamped at the seam: overlap head + exactly `short` new samples.
    assert pre.start_sample == WINDOW_SAMPLES - short - XFADE
    assert pre.pcm.shape[0] == short + XFADE
    assert pre.start_sample + pre.pcm.shape[0] == WINDOW_SAMPLES

    post = backend.render_window(0.0)
    # Remainder lands at the window start, overlap skipped across the
    # seam (it would wrap backwards).
    assert post.start_sample == 0
    assert post.pcm.shape[0] == 2 * mp.FRAME_SAMPLES - short
    assert backend.render_window(0.0) is None  # drained


def test_knob_changes_forward_once():
    backend, client = make_backend()
    state = backend.midi_knobs
    backend.sync_source(_ctx(0.0))
    backend.produce(backend.read_knobs(), _ctx(0.0), "generate")
    baseline = [m for m in client.sent if m["type"] == "knobs"]
    assert len(baseline) == 1  # initial defaults forwarded once

    state.update({"mrt2_temperature": 2.0})
    backend.produce(backend.read_knobs(), _ctx(0.0), "generate")
    backend.produce(backend.read_knobs(), _ctx(0.0), "generate")
    updates = [m for m in client.sent if m["type"] == "knobs"][1:]
    assert updates == [{"type": "knobs", "temperature": 2.0}]


def test_prompt_hooks_forward_to_sidecar():
    backend, client = make_backend()
    backend.handle_set_prompt("acid techno", tags_b="lofi house")
    backend.handle_set_prompt_blend(0.4)
    assert {"type": "prompt", "tags": "acid techno", "tags_b": "lofi house"} in client.sent
    assert {"type": "blend", "value": 0.4} in client.sent


def test_family_is_registered():
    from acestep.streaming.families import (
        FAMILIES,
        FAMILY_KNOB_UNIVERSES,
        SESSION_CREATORS,
    )

    assert "mrt2" in FAMILIES
    assert "mrt2" in FAMILY_KNOB_UNIVERSES
    assert "mrt2" in SESSION_CREATORS
    universe = FAMILY_KNOB_UNIVERSES["mrt2"]()
    assert {s.name for s in universe} == {s.name for s in mrt2_knob_specs()}
