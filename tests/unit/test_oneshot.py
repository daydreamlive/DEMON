"""``oneshot``: whole renders of N versions outside the ring, kept latents, redo.

CPU-only. The real SA3Backend (mock DiTs, fake codecs, the
test_sa3_backend / test_sa3_stream_layout helpers) for the render; a bare
StreamingSession for the request lifecycle; the wire framing by bytes.

The load-bearing property: a one-shot version IS the ring's generation for
the same seed and knobs (depth 1, the same step math and noise hooks), so
the one-shot path changes no sound, only the cost.
"""

from __future__ import annotations

import json
import threading
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from acestep.engine.sa3_adapter import SA3Adapter
from acestep.plugins import StreamLayout
from acestep.streaming.config import SessionConfig
from acestep.streaming.events import OneshotFailed, OneshotReady
from acestep.streaming.generator_backend import Capabilities
from acestep.streaming.knobs import KnobState
from acestep.streaming.oneshot import (
    KeptLatents,
    OneshotError,
    OneshotResult,
    parse_request,
    to_int16,
)
from acestep.streaming.sa3_backend import (
    DELIVERY_SAMPLE_RATE,
    SA3Backend,
    sa3_knob_specs,
)
from acestep.streaming.session import StreamingSession
from demos.realtime_motion_graph_web import protocol
from demos.realtime_motion_graph_web.audio_codec import send_oneshot_payload

from tests.unit.test_sa3_backend import (
    CTX,
    C,
    T,
    _FakeCodec,
    _FakeCond,
    _schedule_builder_factory,
)
from tests.unit import test_sa3_stream_layout as LT


# ---- fakes ---------------------------------------------------------------------


class _XDit(torch.nn.Module):
    """A velocity that depends on x and t (so the trajectory, the renoise
    order and the step math all show in the result); records batch sizes."""

    def __init__(self):
        super().__init__()
        self.batches: list = []

    def forward(self, x, t, **kwargs):
        self.batches.append(int(x.shape[0]))
        return 0.3 * x + 0.1 * t.reshape(-1, 1, 1).to(x.dtype)


def _stock(depth=1, steps=4, dit=None):
    dit = dit if dit is not None else _XDit()
    adapter = SA3Adapter(
        dit, schedule_builder=_schedule_builder_factory(steps),
        device="cpu", dtype=torch.float32,
    )
    b = SA3Backend(
        adapter=adapter, codec=_FakeCodec(), cond=_FakeCond(),
        schedule_builder_factory=_schedule_builder_factory,
        knob_state=KnobState(sa3_knob_specs()), steps=steps, depth=depth,
        vae_window_s=0.1,
    )
    return b, dit


def _ring_latent(b, knobs, ticks=12):
    """The ring's first fresh latent for ``knobs`` (depth 1)."""
    for _ in range(ticks):
        if b.produce(knobs, CTX, "generate"):
            return b._current_result.clone()
    pytest.fail("the ring produced nothing")


def _req(seeds=(3,), knobs=None, redo=None):
    data = {"request_id": "r", "seeds": list(seeds), "knobs": knobs or {}}
    if redo is not None:
        data["redo"] = redo
    return parse_request(data)


# ---- the render is the ring's generation ------------------------------------------


def test_stock_oneshot_is_the_rings_generation():
    ring, _ = _stock()
    knobs = {**ring.read_knobs(), "steps_override": 4, "seed": 11}
    want = _ring_latent(ring, knobs)
    one, _ = _stock()
    got = one._oneshot_denoise((11,), {**one.read_knobs(), "steps_override": 4})
    assert torch.equal(got, want)


def test_stock_versions_batch_on_dim0_and_match_single_renders():
    b, dit = _stock()
    store = KeptLatents()
    res = b.render_oneshot(_req(seeds=(1, 2, 3)), store=store)
    assert max(dit.batches) == 3                      # one call per step for all three
    singles = []
    for s in (1, 2, 3):
        b1, _ = _stock()
        singles.append(b1._oneshot_denoise((s,), b1.read_knobs()))
    for i, key in enumerate(res.ids):
        kept = store.get(key).latent
        np.testing.assert_allclose(
            kept.numpy(), singles[i].movedim(1, 2).numpy(), rtol=1e-5, atol=1e-6,
        )


def test_layout_oneshot_is_the_rings_generation():
    """Through a stream layout's make_noise (and the stock seeded renoise):
    the same latent the depth-1 ring produces for that seed."""
    ring, ctx_ring, _ = LT._layout_backend()
    ring.adapter.dit = _XDit()
    want = _ring_latent(ring, LT._knobs(ring, seed=5, sa3_denoise=1.0))
    one, _ctx, _ = LT._layout_backend()
    one.adapter.dit = _XDit()
    got = one._oneshot_denoise((5,), LT._knobs(one, sa3_denoise=1.0))
    assert torch.equal(got, want)


def test_layout_renoise_hook_drives_every_step():
    calls = []

    def renoise(request, step):
        calls.append((int(request.seed), int(step)))
        g = torch.Generator().manual_seed(1000 * int(request.seed) + int(step))
        return torch.randn(1, LT.N_STREAMS * LT.L, LT.C, generator=g)

    rt = LT._LayoutRuntime({"make_renoise": renoise})
    ring, _, _ = LT._layout_backend(runtime=rt)
    want = _ring_latent(ring, LT._knobs(ring, seed=4, sa3_denoise=1.0))
    ring_calls = list(calls)
    calls.clear()
    rt2 = LT._LayoutRuntime({"make_renoise": renoise})
    one, _, _ = LT._layout_backend(runtime=rt2)
    got = one._oneshot_denoise((4,), LT._knobs(one, sa3_denoise=1.0))
    assert torch.equal(got, want)
    assert calls == ring_calls == [(4, 0)]           # steps=2: one re-noise


def test_on_slot_done_sees_each_version():
    seen = []
    rt = LT._LayoutRuntime({"on_slot_done": lambda req, lat: seen.append(
        (int(req.seed), tuple(lat.shape)))})
    b, _, _ = LT._layout_backend(runtime=rt)
    b.render_oneshot(_req(seeds=(1, 2)), store=KeptLatents())
    assert seen == [(1, (1, LT.N_STREAMS * LT.L, LT.C)), (2, (1, LT.N_STREAMS * LT.L, LT.C))]


# ---- result shape, decode, kept ids ------------------------------------------------


def test_result_is_int16_versions_by_streams_trimmed_to_the_playable_window():
    b, ctx, _ = LT._layout_backend()
    store = KeptLatents()
    res = b.render_oneshot(_req(seeds=(1, 2)), store=store)
    assert isinstance(res, OneshotResult)
    total = round(b.playable_duration_s() * DELIVERY_SAMPLE_RATE)
    assert res.pcm.dtype == np.int16
    assert res.pcm.shape == (2, LT.N_STREAMS, total, 2)
    assert res.streams == (0, 1, 2) and res.seeds == (1, 2)
    assert ctx.codec.decodes == 2 * LT.N_STREAMS       # every stream of every version
    assert set(res.ms) == {"generate", "decode", "pack"}
    # Stream k of version v is the codec's decode of that slice.
    lat = store.get(res.ids[1]).latent
    mean = float(lat[..., 2 * LT.L:3 * LT.L].mean()) * 0.1
    assert res.pcm[1, 2, 100, 0] == to_int16(np.float32(mean))
    assert len(set(res.ids)) == 2 and len(store) == 2


def test_batch_decode_is_used_when_the_codec_has_it():
    class _Batch(LT._SliceCodec):
        def __init__(self):
            super().__init__()
            self.batches = []

        def decode_full_batch(self, latents, *, decode_seed=None):
            self.batches.append((tuple(latents.shape), decode_seed))
            return torch.stack([
                torch.full((2, LT.N44), float(latents[i].mean()) * 0.1)
                for i in range(latents.shape[0])
            ])

    codec = _Batch()
    b, _, _ = LT._layout_backend(codec=codec)
    res = b.render_oneshot(_req(seeds=(8, 9)), store=KeptLatents())
    assert codec.batches == [((3, LT.C, LT.L), 8), ((3, LT.C, LT.L), 9)]
    assert codec.decodes == 0
    assert res.pcm.shape[:2] == (2, 3)


def test_version_controls_set_each_versions_knobs():
    seen = []
    rt = LT._LayoutRuntime({"version_controls": lambda s: {LT.GAINS[0]: float(s) / 10}})
    orig = rt.apply_controls
    rt.apply_controls = lambda values: (seen.append(dict(values)), orig(values))
    b, _, _ = LT._layout_backend(runtime=rt)
    b.render_oneshot(_req(seeds=(3, 4), knobs={LT.GAINS[1]: 0.5}), store=KeptLatents())
    assert [v[LT.GAINS[0]] for v in seen] == [0.3, 0.4]
    assert all(v[LT.GAINS[1]] == 0.5 for v in seen)


# ---- redo -----------------------------------------------------------------------


def _hold_runtime():
    calls = []
    rt = LT._LayoutRuntime({"hold_streams": lambda lat, held: calls.append(
        (None if lat is None else lat.clone(), held))})
    return rt, calls


def test_redo_holds_the_kept_version_and_returns_one_stream():
    rt, calls = _hold_runtime()
    b, ctx, _ = LT._layout_backend(runtime=rt)
    store = KeptLatents()
    first = b.render_oneshot(_req(seeds=(1,)), store=store)
    kept = store.get(first.ids[0]).latent
    ctx.codec.decodes = 0
    redo = b.render_oneshot(
        _req(redo={"id": first.ids[0], "stream": 1, "seed": 42}), store=store,
    )
    assert len(calls) == 2
    assert torch.equal(calls[0][0], kept) and calls[0][1] == (1, 0, 1)
    assert calls[1] == (None, None)                   # released after the denoise
    assert redo.streams == (1,) and redo.seeds == (42,)
    assert redo.pcm.shape[:2] == (1, 1)
    assert ctx.codec.decodes == 1                      # the redone stream only
    assert redo.redo_of == first.ids[0] and redo.ids[0] != first.ids[0]
    assert len(store) == 2


def test_redo_releases_the_hold_when_the_denoise_fails():
    rt, calls = _hold_runtime()
    b, _, _ = LT._layout_backend(runtime=rt)
    store = KeptLatents()
    first = b.render_oneshot(_req(seeds=(1,)), store=store)
    b.adapter.dit = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom"))
    with pytest.raises(RuntimeError):
        b.render_oneshot(_req(redo={"id": first.ids[0], "stream": 0, "seed": 2}), store=store)
    assert calls[-1] == (None, None)


def test_redo_errors():
    b, _, _ = LT._layout_backend()                      # no hold_streams
    store = KeptLatents()
    first = b.render_oneshot(_req(seeds=(1,)), store=store)
    with pytest.raises(OneshotError, match="cannot redo"):
        b.render_oneshot(_req(redo={"id": first.ids[0], "stream": 0, "seed": 2}), store=store)
    rt, _ = _hold_runtime()
    b2, _, _ = LT._layout_backend(runtime=rt)
    with pytest.raises(OneshotError, match="no kept version"):
        b2.render_oneshot(_req(redo={"id": "nope", "stream": 0, "seed": 2}), store=store)
    with pytest.raises(OneshotError, match="redo.stream"):
        b2.render_oneshot(_req(redo={"id": first.ids[0], "stream": 3, "seed": 2}), store=store)
    other = store.put(torch.zeros(1, LT.C, 2 * LT.N_STREAMS * LT.L), n_streams=3, seed=0)
    with pytest.raises(OneshotError, match="another geometry"):
        b2.render_oneshot(_req(redo={"id": other, "stream": 0, "seed": 2}), store=store)


def test_stock_backend_cannot_redo():
    b, _ = _stock()
    store = KeptLatents()
    first = b.render_oneshot(_req(seeds=(1,)), store=store)
    with pytest.raises(OneshotError, match="cannot redo"):
        b.render_oneshot(_req(redo={"id": first.ids[0], "stream": 0, "seed": 2}), store=store)


# ---- request parse, store ------------------------------------------------------------


def test_parse_request():
    r = parse_request({"request_id": "a", "seeds": [1, 2.0], "knobs": {"seed": 3}})
    assert r.seeds == (1, 2) and r.knobs == {"seed": 3.0} and r.redo is None
    r = parse_request({"request_id": "b", "redo": {"id": "k", "stream": 2, "seed": 9}})
    assert r.redo.id == "k" and r.redo.stream == 2 and r.seeds == (9,) and r.versions == 1
    for bad in (
        {"seeds": []}, {"seeds": [1.5]}, {"seeds": [-1]}, {"seeds": [True]},
        {"seeds": list(range(17))}, {"seeds": [1], "knobs": {"a": "x"}},
        {"redo": {"stream": 0, "seed": 1}}, {"redo": {"id": "k", "stream": 0}},
        {"seeds": [1], "knobs": [1]},
    ):
        with pytest.raises(OneshotError):
            parse_request(bad)


def test_kept_latents_expire_by_age_and_count():
    now = [0.0]
    store = KeptLatents(ttl_s=10.0, max_items=2, clock=lambda: now[0])
    a = store.put(torch.ones(1, 2, 3), n_streams=1, seed=1)
    b = store.put(torch.ones(1, 2, 3), n_streams=1, seed=2)
    c = store.put(torch.ones(1, 2, 3), n_streams=1, seed=3)
    with pytest.raises(OneshotError):
        store.get(a)                                   # dropped: over the count
    assert store.get(b).seed == 2
    now[0] = 11.0
    with pytest.raises(OneshotError):
        store.get(c)                                   # dropped: too old


# ---- session lifecycle ----------------------------------------------------------------


class _FakeBackend:
    name = "fake"

    def __init__(self, oneshot=True, error=None):
        self._caps = Capabilities(oneshot=oneshot)
        self.error = error
        self.seen = []

    def capabilities(self):
        return self._caps

    def render_oneshot(self, req):
        self.seen.append(req)
        if self.error is not None:
            raise self.error
        pcm = np.zeros((len(req.seeds), 2, 10, 2), np.int16)
        return OneshotResult(pcm, 48000, tuple(f"id{s}" for s in req.seeds),
                             req.seeds, (0, 1), {"generate": 1.0, "decode": 2.0,
                                                 "pack": 0.1}, 10 / 48000)


def _session(backend, render_only=False):
    s = StreamingSession.__new__(StreamingSession)
    s.backend = backend
    s.config = SessionConfig(render_only=render_only)
    s._bounces = []
    s._oneshots = []
    s._oneshot_wake = threading.Event()
    s.state = SimpleNamespace(
        _lock=threading.RLock(), last_activity_ts=0.0, running=True,
        pending_register=[], pending_enable=[], pending_disable=[],
        pending_depth=None, swap_pending={"waveform": None}, params={},
    )
    events: list = []
    s.bus = SimpleNamespace(publish=events.append)
    return s, events


def test_session_serves_queued_requests_in_order():
    backend = _FakeBackend()
    s, events = _session(backend)
    s.oneshot({"request_id": "a", "seeds": [1, 2]})
    s.oneshot({"request_id": "b", "seeds": [3]})
    assert events == [] and len(s._oneshots) == 2
    s._service_oneshots()
    assert [e.request_id for e in events] == ["a", "b"]
    assert all(isinstance(e, OneshotReady) for e in events)
    assert events[0].ids == ("id1", "id2") and events[0].streams == (0, 1)
    assert set(events[0].ms) == {"generate", "decode", "pack", "total"}
    assert s._oneshots == []


def test_session_rejects_without_the_capability_and_on_bad_fields():
    s, events = _session(_FakeBackend(oneshot=False))
    s.oneshot({"request_id": "a", "seeds": [1]})
    assert isinstance(events[0], OneshotFailed) and "cannot render" in events[0].error
    s2, events2 = _session(_FakeBackend())
    s2.oneshot({"request_id": "b", "seeds": []})
    assert isinstance(events2[0], OneshotFailed) and s2._oneshots == []


def test_a_failed_render_is_answered_and_the_next_still_runs():
    backend = _FakeBackend(error=RuntimeError("cuda"))
    s, events = _session(backend)
    s.oneshot({"request_id": "a", "seeds": [1]})
    s._service_oneshots()
    assert isinstance(events[0], OneshotFailed) and "cuda" in events[0].error
    backend.error = OneshotError("no kept version 'x'")
    s.oneshot({"request_id": "b", "redo": {"id": "x", "stream": 0, "seed": 1}})
    s._service_oneshots()
    assert events[1].error == "no kept version 'x'"


def test_render_only_loop_serves_requests_without_a_runner():
    backend = _FakeBackend()
    s, events = _session(backend, render_only=True)
    assert s.render_only()
    s.apply_pending = lambda: s._service_oneshots()
    t = threading.Thread(target=s._run_render_only, daemon=True)
    t.start()
    s.oneshot({"request_id": "a", "seeds": [1]})
    for _ in range(100):
        if events:
            break
        threading.Event().wait(0.01)
    s.state.running = False
    t.join(1.0)
    assert not t.is_alive()
    assert isinstance(events[0], OneshotReady)


def test_render_only_needs_the_capability():
    s, _ = _session(_FakeBackend(oneshot=False), render_only=True)
    assert not s.render_only()
    s2, _ = _session(_FakeBackend(), render_only=False)
    assert not s2.render_only()


# ---- wire --------------------------------------------------------------------------


def test_oneshot_payload_framing():
    sent: list = []
    ws = SimpleNamespace(
        send=lambda m: sent.append(m if isinstance(m, (str, bytes)) else b"".join(m)),
    )
    pcm = np.arange(2 * 3 * 4 * 2, dtype=np.int16).reshape(2, 3, 4, 2)
    send_oneshot_payload(
        ws, request_id="r", pcm=pcm, sample_rate=48000, ids=("a", "b"),
        seeds=(1, 2), streams=(0, 1, 2), ms={"generate": 1.0}, redo_of=None,
    )
    head = json.loads(sent[0])
    assert head == {
        "type": "oneshot_ready", "request_id": "r", "ids": ["a", "b"],
        "seeds": [1, 2], "streams": [0, 1, 2], "sample_rate": 48000,
        "channels": 2, "frames": 4, "encoding": "pcm_s16le",
        "ms": {"generate": 1.0}, "redo_of": None,
    }
    body = np.frombuffer(sent[1], dtype="<i2")
    assert np.array_equal(body, pcm.reshape(-1))
    # Block (version 1, stream 2) starts at (1 * 3 + 2) * frames * channels.
    assert body[(1 * 3 + 2) * 8] == pcm[1, 2, 0, 0]
    # Every header key is a registered field of the event.
    spec = next(e for e in protocol.EVENTS if e.name == "oneshot_ready")
    assert set(head) - {"type"} == {f.name for f in spec.fields}


def test_command_and_config_are_registered():
    cmd = next(c for c in protocol.COMMANDS if c.name == "oneshot")
    assert {f.name for f in cmd.fields} == {"request_id", "seeds", "knobs", "redo"}
    assert "render_only" in protocol.config_catalog()


def test_sa3_declares_oneshot():
    b, _ = _stock()
    assert b.capabilities().oneshot is True


def test_batch_decode_out_of_memory_falls_back_to_one_stream_at_a_time(monkeypatch):
    class _Oom(LT._SliceCodec):
        def decode_full_batch(self, latents, *, decode_seed=None):
            raise torch.cuda.OutOfMemoryError("too big")

    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    codec = _Oom()
    b, _, _ = LT._layout_backend(codec=codec)
    res = b.render_oneshot(_req(seeds=(1, 2)), store=KeptLatents())
    assert codec.decodes == 2 * LT.N_STREAMS
    assert res.pcm.shape[:2] == (2, LT.N_STREAMS)
