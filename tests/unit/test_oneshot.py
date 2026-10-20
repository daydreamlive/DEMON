"""``oneshot``: whole renders of N versions outside the ring, kept latents, redo.

CPU-only. The real SA3Backend (mock DiTs, fake codecs, the
test_sa3_backend / test_sa3_stream_layout helpers) for the render; a bare
StreamingSession for the request lifecycle; the wire framing by bytes.

The load-bearing property: a one-shot version IS the ring's generation for
the same seed and knobs: the same SlotRequest on the same StreamPipeline at
depth 1, so the one-shot path changes no sound, only the cost. The deleted
hand-written denoise loop is kept below as the oracle the pipeline-driven
render must match bit for bit.
"""

from __future__ import annotations

import json
import threading
from dataclasses import replace
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
from acestep.engine import ode_steps
from acestep.engine.stream import SlotRequest
from acestep.streaming.oneshot import (
    QUEUE_MAX,
    KeptLatents,
    OneshotError,
    OneshotResult,
    latent_from_bytes,
    latent_nbytes,
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
from demos.realtime_motion_graph_web.ws_adapter import _dispatch_oneshot

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


@torch.no_grad()
def _old_oneshot_denoise(b, seeds, knobs):
    """ORACLE: the hand-written one-shot denoise loop (and its knob prep,
    ``_oneshot_controls``) as they stood at ff880d47, before the one-shot
    became the ring's own pipeline. Kept verbatim (``self`` -> ``b``; the
    LoRA branches dropped: these fakes carry no LoRA manager)."""
    shift = float(knobs.get("sa3_shift", 1.0))
    if abs(shift - float(b.adapter.shift_alpha)) > 1e-3:
        with b._control_lock:
            b.adapter.shift_alpha = shift
            b.pipeline.invalidate_schedule_cache()
    b._apply_extension_controls(knobs)

    steps = int(knobs.get("steps_override", b._steps))
    if steps != b._steps:
        b._steps = steps
        b.pipeline = b._build_pipeline(steps)
    denoise = float(knobs.get("sa3_denoise", 1.0))
    with b._control_lock:
        aux = b._active_bundle
        frames = int(b._cond.latent_frames)
    source = b._source_latent_btc
    reqs = [
        SlotRequest(
            seed=int(s), denoise=denoise, source_latents=source,
            aux_cond=aux, latent_frames=frames, sde_noise_seeded=True,
        )
        for s in seeds
    ]
    device, dtype = b.adapter.request_device_dtype(reqs[0])
    pipe = b.pipeline
    pipe._ensure_device(device, dtype)
    sched = b.adapter.build_schedule(pipe.config, denoise, device, dtype).cpu()
    noise = torch.cat([pipe._make_noise(r) for r in reqs], dim=0)
    t_start = sched[0].item()
    if source is not None and denoise < 1.0:
        xt = t_start * noise + (1.0 - t_start) * source
    else:
        xt = noise.clone()
    renoise = getattr(b.adapter, "make_renoise", None)
    gens = []
    for s in seeds:
        g = torch.Generator(device=device)
        g.manual_seed(int(s))
        gens.append(g)
    sde = b._sampler in ("sde", "pingpong")
    B = len(reqs)
    for i in range(sched.numel() - 1):
        t = sched[i].item()
        t_next = sched[i + 1].item()
        vt = b.adapter.batched_forward(
            xt, [t] * B, [None] * B, [None] * B, [None] * B, [aux] * B,
        )
        x0 = ode_steps.x0_from_vel(xt, vt, t)
        if t_next <= 0:
            xt = x0
            break
        if not sde:
            xt = xt + (t_next - t) * vt
            continue
        rows = []
        for bi, r in enumerate(reqs):
            nz = renoise(r, i) if renoise is not None else None
            if nz is None:
                nz = torch.randn(
                    (1,) + tuple(xt.shape[1:]), device=xt.device,
                    dtype=xt.dtype, generator=gens[bi],
                )
            rows.append(nz.to(device=xt.device, dtype=xt.dtype))
        xt = ode_steps.step_sde_renoise(xt, x0, t_next, torch.cat(rows, dim=0))
    done = getattr(b.adapter, "on_slot_done", None)
    if done is not None:
        for bi, r in enumerate(reqs):
            done(r, xt[bi:bi + 1])
    return xt


def _one(b, seed, knobs):
    """One one-shot version through the production path (engine layout)."""
    b._oneshot_pipeline_ready()
    return b._oneshot_latent({**knobs, "seed": float(seed)})


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
    got = _one(one, 11, {**one.read_knobs(), "steps_override": 4})
    assert torch.equal(got, want)


def test_pipeline_oneshot_is_the_deleted_loop_bit_for_bit():
    """The oracle: the pipeline-driven version equals the old hand-written
    loop exactly, stock and under a stream layout (its noise hooks), at
    several seeds and denoise levels (a partial denoise starts from the
    source)."""
    for seed in (0, 5, 123):
        for denoise in (1.0, 0.6):
            old, _ = _stock()
            want = _old_oneshot_denoise(old, (seed,), {**old.read_knobs(), "sa3_denoise": denoise})
            new, _ = _stock()
            got = _one(new, seed, {**new.read_knobs(), "sa3_denoise": denoise})
            assert torch.equal(got, want), (seed, denoise)
            torch.manual_seed(99)                     # the fixture's random source
            lo, _c, _r = LT._layout_backend()
            lo.adapter.dit = _XDit()
            want = _old_oneshot_denoise(lo, (seed,), LT._knobs(lo, sa3_denoise=denoise))
            torch.manual_seed(99)
            ln, _c, _r = LT._layout_backend()
            ln.adapter.dit = _XDit()
            got = _one(ln, seed, LT._knobs(ln, sa3_denoise=denoise))
            assert torch.equal(got, want), ("layout", seed, denoise)


def test_render_oneshot_keeps_the_oracle_latent_per_version():
    b, dit = _stock()
    store = KeptLatents()
    res = b.render_oneshot(_req(seeds=(1, 2, 3)), store=store)
    assert set(dit.batches) == {1}                    # one slot: depth 1, batch 1
    assert b.pipeline.depth == 1
    for i, key in enumerate(res.ids):
        ref, _ = _stock()
        want = _old_oneshot_denoise(ref, (res.seeds[i],), ref.read_knobs())
        assert torch.equal(store.get(key).latent, want.movedim(1, 2))


def test_oneshot_runs_at_depth_one_and_a_steps_change_rebuilds_the_pipeline():
    b, _ = _stock(depth=3)
    before = b.pipeline
    b.render_oneshot(_req(seeds=(1,), knobs={"steps_override": 4}), store=KeptLatents())
    assert b.pipeline is before and b.pipeline.depth == 1
    b.render_oneshot(_req(seeds=(1,), knobs={"steps_override": 2}), store=KeptLatents())
    assert b._steps == 2 and b.pipeline is not before and b.pipeline.depth == 1


def test_layout_oneshot_is_the_rings_generation():
    """Through a stream layout's make_noise (and the stock seeded renoise):
    the same latent the depth-1 ring produces for that seed."""
    ring, ctx_ring, _ = LT._layout_backend()
    ring.adapter.dit = _XDit()
    want = _ring_latent(ring, LT._knobs(ring, seed=5, sa3_denoise=1.0))
    one, _ctx, _ = LT._layout_backend()
    one.adapter.dit = _XDit()
    got = _one(one, 5, LT._knobs(one, sa3_denoise=1.0))
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
    got = _one(one, 4, LT._knobs(one, sa3_denoise=1.0))
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


class _WindowSliceCodec(LT._SliceCodec):
    """A windowed codec (SAME-L): records every decode_window call; its
    full decodes must never be reached from the one-shot."""

    def __init__(self):
        super().__init__()
        self.windows: list = []

    def decode_window(self, latent_bct, start, n):
        self.windows.append((int(start), int(n)))
        return torch.full((2, int(n)), float(latent_bct.mean()) * 0.1)

    def decode_full(self, latent_bct, *, decode_seed=None):
        raise AssertionError("one-shot called decode_full")

    def decode_full_batch(self, latents, *, decode_seed=None):
        raise AssertionError("one-shot called decode_full_batch")


def test_oneshot_decodes_through_decode_window_over_the_whole_song():
    codec = _WindowSliceCodec()
    b, _, _ = LT._layout_backend(codec=codec)
    store = KeptLatents()
    res = b.render_oneshot(_req(seeds=(8, 9)), store=store)
    assert codec.windows == [(0, LT.L * 4096)] * (2 * LT.N_STREAMS)
    total = round(b.playable_duration_s() * DELIVERY_SAMPLE_RATE)
    assert res.pcm.shape == (2, LT.N_STREAMS, total, 2)
    lat = store.get(res.ids[0]).latent
    mean = float(lat[..., LT.L:2 * LT.L].mean()) * 0.1
    # a constant through the 44.1 -> 48 kHz filter: within one LSB
    assert abs(int(res.pcm[0, 1, 100, 0]) - int(to_int16(np.float32(mean)))) <= 1


def test_held_streams_come_from_the_layouts_render_stream_in_both_paths():
    """A stream the layout renders itself (a held take) is that render, in
    a ring window and in a one-shot alike: the codec decodes the others."""
    take = np.full((10 ** 6, 2), 0.25, np.float32)
    rt = LT._LayoutRuntime({"render_stream": lambda k, part: take if k == 0 else None})
    codec = _WindowSliceCodec()
    b, _, _ = LT._layout_backend(runtime=rt, codec=codec)
    res = b.render_oneshot(_req(seeds=(1,)), store=KeptLatents())
    assert len(codec.windows) == LT.N_STREAMS - 1
    assert (res.pcm[0, 0] == to_int16(np.float32(0.25))).all()
    codec.windows.clear()
    audio = b._render_stream_audio(
        torch.zeros(1, LT.N_STREAMS * LT.L, LT.C), 588, 4096,
        start48=0, n48=480, lead48=640,
    )
    assert codec.windows == [(588, 4096)] * (LT.N_STREAMS - 1)
    assert audio.shape == (LT.N_STREAMS, 480, 2)
    assert torch.equal(audio[0], torch.full((480, 2), 0.25))


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


def _session(backend, render_only=True):
    s = StreamingSession.__new__(StreamingSession)
    s.backend = backend
    s.config = SessionConfig(render_only=render_only)
    s._knob_specs_by_name = {k.name: k for k in sa3_knob_specs()}
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
    # Every header key is a registered field of the event; the new optional
    # ones only when given.
    spec = next(e for e in protocol.EVENTS if e.name == "oneshot_ready")
    assert set(head) - {"type"} == (
        {f.name for f in spec.fields} - {"latent_header", "latents"}
    )


def test_command_and_config_are_registered():
    cmd = next(c for c in protocol.COMMANDS if c.name == "oneshot")
    assert {f.name for f in cmd.fields} == {
        "request_id", "seeds", "knobs", "redo", "return_latents",
    }
    assert cmd.binary and cmd.binary_optional         # only a redo with latent
    assert "render_only" in protocol.config_catalog()


def test_sa3_declares_oneshot():
    b, _ = _stock()
    assert b.capabilities().oneshot is True
    assert b.capabilities().oneshot_redo is False      # no layout, no hold_streams
    rt, _ = _hold_runtime()
    lb, _, _ = LT._layout_backend(runtime=rt)
    assert lb.capabilities().oneshot_redo is True
    lb2, _, _ = LT._layout_backend()
    assert lb2.capabilities().oneshot_redo is False


# ---- render-only only, knob contract, bounded queue -----------------------------------


def test_a_ring_session_refuses_oneshot_and_does_not_advertise_it():
    backend = _FakeBackend()
    backend._caps = Capabilities(oneshot=True, oneshot_redo=True)
    ring, events = _session(backend, render_only=False)
    ring.oneshot({"request_id": "a", "seeds": [1]})
    assert isinstance(events[0], OneshotFailed) and "render_only" in events[0].error
    assert ring._oneshots == [] and backend.seen == []
    caps = ring.capabilities_payload()
    assert caps["oneshot"] is False and caps["oneshot_redo"] is False
    ro, _ = _session(backend, render_only=True)
    caps = ro.capabilities_payload()
    assert caps["oneshot"] is True and caps["oneshot_redo"] is True


def test_unclamped_knobs_are_clamped_by_the_rings_contract():
    backend = _FakeBackend()
    s, events = _session(backend)
    s.oneshot({"request_id": "a", "seeds": [1],
               "knobs": {"steps_override": 99, "sa3_shift": 0.01, "sa3_denoise": 3.0}})
    s._service_oneshots()
    assert isinstance(events[0], OneshotReady)
    knobs = backend.seen[0].knobs
    assert knobs["steps_override"] == 16.0              # the ring's steps_override max
    assert knobs["sa3_shift"] == 0.25 and knobs["sa3_denoise"] == 1.0
    s.oneshot({"request_id": "b", "seeds": [1], "knobs": {"steps_override": 0.2}})
    s._service_oneshots()
    assert backend.seen[1].knobs["steps_override"] == 1.0


def test_unknown_knobs_are_refused():
    backend = _FakeBackend()
    s, events = _session(backend)
    s.oneshot({"request_id": "a", "seeds": [1], "knobs": {"no_such_knob": 1}})
    assert isinstance(events[0], OneshotFailed) and "no_such_knob" in events[0].error
    assert s._oneshots == []


def test_a_full_queue_refuses_with_queue_full_never_a_silent_drop():
    backend = _FakeBackend()
    s, events = _session(backend)
    for i in range(QUEUE_MAX):
        s.oneshot({"request_id": f"r{i}", "seeds": [1]})
    assert events == [] and len(s._oneshots) == QUEUE_MAX == 4
    s.oneshot({"request_id": "late", "seeds": [1]})
    assert len(events) == 1 and isinstance(events[0], OneshotFailed)
    assert events[0].request_id == "late" and events[0].reason == "queue_full"
    assert [r.request_id for r in s._oneshots] == ["r0", "r1", "r2", "r3"]
    s._service_oneshots()
    assert [e.request_id for e in events[1:]] == ["r0", "r1", "r2", "r3"]
    s.oneshot({"request_id": "again", "seeds": [1]})   # room again after answers
    assert s._oneshots[0].request_id == "again"


def test_close_clears_the_queue():
    backend = _FakeBackend()
    s, _ = _session(backend)
    s.oneshot({"request_id": "a", "seeds": [1]})
    assert len(s._oneshots) == 1
    s._drop_fetched_loras = lambda: None
    s._close_analysis_taps = lambda: None
    s.bus.close = lambda: None
    s.audio_eng = SimpleNamespace(stop=lambda: None)
    s.stream = None
    s.session = None
    s.closed = threading.Event()
    s.close()
    assert s._oneshots == []


# ---- which DiT served each request (song_gen_20261019 lane D) ---------------------------


class _DescribedDit(_XDit):
    """A layout step that says what it runs on (the multistem EngineStep /
    EagerStep ``describe()``)."""

    def __init__(self, label):
        super().__init__()
        self.label = label

    def describe(self):
        return self.label


def test_result_names_the_dit_that_ran():
    b, _ = _stock()
    assert b.render_oneshot(_req(seeds=(1,)), store=KeptLatents()).dit == "_XDit"
    b2, _ = _stock(dit=_DescribedDit("tensorrt:song.trt calls=8"))
    assert b2.render_oneshot(_req(seeds=(1,)), store=KeptLatents()).dit == "tensorrt:song.trt calls=8"


def test_session_logs_the_dit_per_request():
    from loguru import logger

    backend = _FakeBackend()
    real = backend.render_oneshot

    def described(req):
        import dataclasses

        return dataclasses.replace(real(req), dit="eager:plugin_multistem_engine off")

    backend.render_oneshot = described
    s, events = _session(backend)
    lines: list = []
    sink = logger.add(lambda m: lines.append(str(m)), level="INFO", format="{message}")
    try:
        s.oneshot({"request_id": "a", "seeds": [1]})
        s._service_oneshots()
    finally:
        logger.remove(sink)
    gen = [ln for ln in lines if ln.startswith("oneshot_generate")]
    assert gen and "dit=eager:plugin_multistem_engine off" in gen[0]
    assert isinstance(events[0], OneshotReady)


# ---- slow DEMON: several streams, adjust at a denoise, client-held latents ---------


def _denoise_runtime():
    calls = []

    def hold(lat, held, denoise=None):
        calls.append((None if lat is None else lat.clone(), held, denoise))

    return LT._LayoutRuntime({"hold_streams": hold}), calls


def _client_latent(store, key):
    """A kept version as the client holds it: float16 [n_streams, C, L]."""
    kept = store.get(key).latent                       # [1, C, n * L]
    return torch.stack(kept[0].split(LT.L, dim=-1)).to(torch.float16).numpy()


def test_parse_streams_denoise_and_latent():
    r = parse_request({"redo": {"id": "k", "streams": [2, 0], "seed": 1}})
    assert r.redo.streams == (0, 2) and r.redo.denoise == (1.0, 1.0)
    assert r.redo.stream == 0 and r.versions == 1 and r.return_latents is False
    # stream and streams are the same thing when they agree
    a = parse_request({"redo": {"id": "k", "stream": 1, "seed": 1}}).redo
    b = parse_request({"redo": {"id": "k", "streams": [1], "stream": 1, "seed": 1}}).redo
    assert a == b and a.streams == (1,)
    # scalar denoise: every regenerated stream; list: per stream, kept paired
    r = parse_request({"redo": {"id": "k", "streams": [0, 2], "seed": 1, "denoise": 0.4}})
    assert r.redo.denoise == (0.4, 0.4)
    r = parse_request({"redo": {"id": "k", "streams": [2, 0], "seed": 1, "denoise": [0.3, 1]}})
    assert r.redo.streams == (0, 2) and r.redo.denoise == (1.0, 0.3)
    # client-held latent in place of id
    r = parse_request({
        "return_latents": True,
        "redo": {"streams": [1], "seed": 1,
                 "latent": {"dtype": "float16", "shape": [3, 4, 5]}},
    })
    assert r.return_latents and r.redo.id is None and r.redo.latent_shape == (3, 4, 5)
    assert latent_nbytes({"dtype": "float16", "shape": [3, 4, 5]}) == 120
    lat = {"dtype": "float16", "shape": [3, 4, 5]}
    for bad in (
        {"redo": {"id": "k", "stream": 0, "streams": [1], "seed": 1}},
        {"redo": {"id": "k", "streams": [], "seed": 1}},
        {"redo": {"id": "k", "streams": [1, 1], "seed": 1}},
        {"redo": {"id": "k", "seed": 1}},
        {"redo": {"id": "k", "streams": [0, 1], "seed": 1, "denoise": [0.5]}},
        {"redo": {"id": "k", "stream": 0, "seed": 1, "denoise": 0}},
        {"redo": {"id": "k", "stream": 0, "seed": 1, "denoise": 1.5}},
        {"redo": {"id": "k", "stream": 0, "seed": 1, "denoise": True}},
        {"redo": {"id": "k", "stream": 0, "seed": 1, "latent": lat}},
        {"redo": {"stream": 0, "seed": 1, "latent": {"dtype": "float32", "shape": [3, 4, 5]}}},
        {"redo": {"stream": 0, "seed": 1, "latent": {"dtype": "float16", "shape": [3, 4]}}},
        {"redo": {"stream": 0, "seed": 1, "latent": {"dtype": "float16", "shape": [3, 0, 5]}}},
        {"seeds": [1], "return_latents": "yes"},
    ):
        with pytest.raises(OneshotError):
            parse_request(bad)


def test_latent_from_bytes_checks_the_length():
    r = parse_request({"redo": {"stream": 0, "seed": 1,
                                "latent": {"dtype": "float16", "shape": [2, 3, 4]}}})
    data = np.arange(24, dtype="<f2").reshape(2, 3, 4)
    out = latent_from_bytes(r.redo, data.tobytes())
    assert out.latent.dtype == np.float16 and np.array_equal(out.latent, data)
    with pytest.raises(OneshotError, match="bytes"):
        latent_from_bytes(r.redo, data.tobytes()[:-2])


def test_redo_several_streams_returns_them_ascending():
    rt, calls = _denoise_runtime()
    b, ctx, _ = LT._layout_backend(runtime=rt)
    store = KeptLatents()
    first = b.render_oneshot(_req(seeds=(1,)), store=store)
    ctx.codec.decodes = 0
    redo = b.render_oneshot(
        _req(redo={"id": first.ids[0], "streams": [2, 0], "seed": 5}), store=store,
    )
    # every regenerated d == 1.0: called exactly as before (no denoise kwarg)
    assert calls[0][1] == (0, 1, 0) and calls[0][2] is None
    assert calls[1] == (None, None, None)
    assert redo.streams == (0, 2) and redo.pcm.shape[:2] == (1, 2)
    assert ctx.codec.decodes == 2
    assert redo.latent_header == {
        "n_streams": LT.N_STREAMS, "channels": LT.C, "frames": LT.L,
        "seconds": pytest.approx(first.seconds),
    }
    assert redo.latents is None


def test_adjust_passes_the_per_stream_denoise():
    rt, calls = _denoise_runtime()
    b, _, _ = LT._layout_backend(runtime=rt)
    store = KeptLatents()
    first = b.render_oneshot(_req(seeds=(1,)), store=store)
    kept = store.get(first.ids[0]).latent
    b.render_oneshot(
        _req(redo={"id": first.ids[0], "streams": [0, 1], "seed": 5,
                   "denoise": [0.4, 1.0]}), store=store,
    )
    lat, held, denoise = calls[0]
    assert torch.equal(lat, kept)
    assert denoise == (0.4, 1.0, 0.0)
    assert held == (1, 0, 1)                           # derived for old callers
    assert calls[1] == (None, None, None)


def test_a_layout_without_denoise_cannot_adjust():
    rt, calls = _hold_runtime()                        # hold_streams(lat, held)
    b, _, _ = LT._layout_backend(runtime=rt)
    store = KeptLatents()
    first = b.render_oneshot(_req(seeds=(1,)), store=store)
    with pytest.raises(OneshotError, match="cannot adjust a stem"):
        b.render_oneshot(
            _req(redo={"id": first.ids[0], "stream": 0, "seed": 2, "denoise": 0.5}),
            store=store,
        )
    assert calls == []                                 # refused before any work
    # a re-roll (d == 1.0) still works on it
    b.render_oneshot(_req(redo={"id": first.ids[0], "stream": 0, "seed": 2}), store=store)
    assert calls[0][1] == (0, 1, 1)


def _client_redo(latent_np, **redo):
    shape = list(latent_np.shape)
    req = parse_request({"request_id": "c", "redo": {
        "seed": 3, "latent": {"dtype": "float16", "shape": shape}, **redo}})
    return replace(req, redo=latent_from_bytes(req.redo, latent_np.astype("<f2").tobytes()))


def test_client_latent_stands_in_for_a_kept_id():
    rt, calls = _denoise_runtime()
    b, _, _ = LT._layout_backend(runtime=rt)
    store = KeptLatents()
    first = b.render_oneshot(_req(seeds=(1,)), store=store)
    held_np = _client_latent(store, first.ids[0])
    res = b.render_oneshot(_client_redo(held_np, streams=[1]), store=store)
    want = torch.from_numpy(held_np).float()
    assert torch.equal(calls[0][0], torch.cat(list(want.unbind(0)), dim=-1).unsqueeze(0))
    assert calls[0][1] == (1, 0, 1)
    assert res.streams == (1,) and res.redo_of is None


def test_client_latent_geometry_and_finiteness_are_checked_first():
    rt, calls = _denoise_runtime()
    b, _, _ = LT._layout_backend(runtime=rt)
    store = KeptLatents()
    good = np.zeros((LT.N_STREAMS, LT.C, LT.L), np.float16)
    for bad in (
        np.zeros((LT.N_STREAMS - 1, LT.C, LT.L), np.float16),
        np.zeros((LT.N_STREAMS, LT.C - 1, LT.L), np.float16),
        np.zeros((LT.N_STREAMS, LT.C, LT.L + 1), np.float16),
    ):
        with pytest.raises(OneshotError, match="this session needs"):
            b.render_oneshot(_client_redo(bad, stream=0), store=store)
    nan = good.copy()
    nan[1, 2, 3] = np.inf
    with pytest.raises(OneshotError, match="non-finite"):
        b.render_oneshot(_client_redo(nan, stream=0), store=store)
    assert calls == [] and len(store) == 0


def test_return_latents_stacks_every_version_and_stream():
    b, _, _ = LT._layout_backend()
    store = KeptLatents()
    res = b.render_oneshot(
        parse_request({"seeds": [1, 2], "return_latents": True}), store=store,
    )
    assert res.latents.dtype == np.float16
    assert res.latents.shape == (2, LT.N_STREAMS, LT.C, LT.L)
    for v, key in enumerate(res.ids):
        assert np.array_equal(res.latents[v], _client_latent(store, key))
    # a redo returns the full latent of its one version, every stream
    rt, _ = _denoise_runtime()
    b2, _, _ = LT._layout_backend(runtime=rt)
    redo = b2.render_oneshot(parse_request({
        "return_latents": True, "redo": {"id": res.ids[0], "stream": 1, "seed": 4}}),
        store=store)
    assert redo.latents.shape == (1, LT.N_STREAMS, LT.C, LT.L)
    assert redo.pcm.shape[1] == 1


def test_stock_result_carries_the_header_and_latents_on_request():
    b, _ = _stock()
    plain = b.render_oneshot(_req(seeds=(1,)), store=KeptLatents())
    assert plain.latents is None
    assert plain.latent_header["n_streams"] == 1 and plain.latent_header["channels"] == C
    asked = b.render_oneshot(parse_request({"seeds": [1], "return_latents": True}),
                             store=KeptLatents())
    assert asked.latents.shape == (1, 1, C, plain.latent_header["frames"])
    assert np.array_equal(asked.pcm, plain.pcm)        # asking changes no sound


def test_session_hands_the_latent_frame_to_the_redo():
    backend = _FakeBackend()
    s, events = _session(backend)
    blob = np.ones((2, 3, 4), "<f2")
    head = {"request_id": "a", "redo": {"stream": 0, "seed": 1,
            "latent": {"dtype": "float16", "shape": [2, 3, 4]}}}
    s.oneshot(head, latent=blob.tobytes())
    assert np.array_equal(s._oneshots[0].redo.latent, blob)
    s.oneshot({**head, "request_id": "b"}, latent=blob.tobytes()[:4])
    assert isinstance(events[-1], OneshotFailed) and events[-1].request_id == "b"
    assert len(s._oneshots) == 1


def test_latents_frame_follows_the_pcm():
    sent: list = []
    ws = SimpleNamespace(
        send=lambda m: sent.append(m if isinstance(m, (str, bytes)) else b"".join(m)),
    )
    pcm = np.zeros((1, 1, 4, 2), np.int16)
    lat = np.arange(1 * 2 * 3 * 4, dtype=np.float16).reshape(1, 2, 3, 4)
    header = {"n_streams": 2, "channels": 3, "frames": 4, "seconds": 1.5}
    send_oneshot_payload(
        ws, request_id="r", pcm=pcm, sample_rate=48000, ids=("a",), seeds=(1,),
        streams=(0,), ms={}, redo_of=None, latent_header=header, latents=lat,
    )
    head = json.loads(sent[0])
    assert head["latent_header"] == header
    assert head["latents"] == {"dtype": "float16", "shape": [1, 2, 3, 4], "bytes": 48}
    assert len(sent) == 3 and len(sent[1]) == pcm.nbytes
    assert np.array_equal(np.frombuffer(sent[2], "<f2"), lat.reshape(-1))
    spec = next(e for e in protocol.EVENTS if e.name == "oneshot_ready")
    assert set(head) - {"type"} == {f.name for f in spec.fields}


class _Recorder:
    def __init__(self):
        self.calls = []

    def oneshot(self, data, *, origin, latent=None):
        self.calls.append((data, latent))


def test_adapter_reads_the_latent_frame_and_checks_its_length():
    head = {"type": "oneshot", "request_id": "q", "redo": {
        "stream": 0, "seed": 1, "latent": {"dtype": "float16", "shape": [2, 3, 4]}}}
    frames = [b"\0" * 48]
    sent = []
    rec = _Recorder()
    reads = []

    def recv(fail_type, **extra):
        reads.append((fail_type, extra))
        return frames.pop(0)

    _dispatch_oneshot(head, rec, recv, sent.append, "ws")
    assert reads == [("oneshot_failed", {"request_id": "q"})]
    assert rec.calls[0][1] == b"\0" * 48 and sent == []
    # a frame of the wrong length: answered, never queued
    frames.append(b"\0" * 46)
    _dispatch_oneshot(head, rec, recv, sent.append, "ws")
    assert len(rec.calls) == 1
    assert sent[0]["type"] == "oneshot_failed" and sent[0]["request_id"] == "q"
    assert "46 bytes" in sent[0]["error"]
    # no latent: no frame read, straight to the session
    _dispatch_oneshot({"type": "oneshot", "request_id": "p", "seeds": [1]},
                      rec, recv, sent.append, "ws")
    assert len(reads) == 2 and rec.calls[-1][1] is None
    # the recv failed (timeout / not binary): it answered; nothing queued
    _dispatch_oneshot(head, rec, lambda *a, **k: None, sent.append, "ws")
    assert len(rec.calls) == 2


def test_oneshot_applies_a_lora_strength_under_the_refit_mirror():
    """A one-shot's ``lora_str_<id>`` reaches the engine under the refit
    mirror, where the ring stashes a change from ``rebuild_imminent`` (a
    runner call a render-only session never makes): the one-shot detects
    it itself and the strength lands, mirror synced, before the DiT runs."""
    from types import SimpleNamespace

    b, dit = _stock()
    order: list = []
    desc = SimpleNamespace(id="x", state="enabled", strength=0.5)

    def set_strength(lora_id, s):
        desc.strength = s
        order.append(("strength", lora_id, s))

    b._use_lora = True
    b._lora_mgr = SimpleNamespace(
        list_loras=lambda: [desc], set_lora_strength=set_strength,
        touches_conditioner=lambda lora_id: False, has_active_loras=False,
    )
    b._refit_mirror = SimpleNamespace(sync=lambda reason: order.append(("sync", reason)))
    forward = dit.forward
    dit.forward = lambda x, t, **kw: (order.append(("dit",)), forward(x, t, **kw))[1]
    b.render_oneshot(_req(seeds=(1,), knobs={"lora_str_x": 1.0}), store=KeptLatents())
    assert order[:2] == [("strength", "x", 1.0), ("sync", "strength")]
    assert order[2] == ("dit",) and desc.strength == 1.0
    # unchanged strength: nothing stashed, no refit
    order.clear()
    b.render_oneshot(_req(seeds=(2,), knobs={"lora_str_x": 1.0}), store=KeptLatents())
    assert order and all(o == ("dit",) for o in order)

