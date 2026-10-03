"""YuE2Backend through the REAL StreamPipeline, plus the family registration.

The adapter is the production :class:`YuE2Adapter` over a fake velocity;
the codec is a recording fake. Pinned here:

* contract surface (capabilities, geometry, manifest, no LoRA);
* a prompt change re-composes in the background, publishes atomically,
  drops latents still in flight for the old song, and the ring renders
  the new song's anchor with one full solve before applying the knobs;
* the settled short-circuit stops ticking only when nothing changed;
* ``set_prompt_blend`` is a hard switch at 0.5 between two songs;
* window rendering clamps at the song edges (and the VAE window plan);
* registration, config fields and the preflight verdicts.

CPU only, no weights.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from acestep.engine.yue2_adapter import YuE2Adapter
from acestep.engine.yue2_trt import window_plan
from acestep.streaming.generator_backend import GeneratorBackend, TickContext
from acestep.streaming.knobs import KnobState
from acestep.streaming.yue2_backend import (
    YUE2_MAX_SONG_S,
    YuE2Backend,
    playable_seconds,
    yue2_knob_specs,
)
from acestep.streaming.yue2_recompose import Song

T = 40
STEPS = 4
CTX = TickContext(playhead_s=0.0, buffer_duration_s=2.0)


class _Bundle:
    def __init__(self, offset: float, frames: int = T, cond_tokens: int = 2000, seed: int = 0):
        self.offset, self.frames, self.cond_tokens, self.seed = offset, frames, cond_tokens, seed


def _song(offset=0.0, tags="pop", **kw):
    return Song(bundle=_Bundle(offset, **kw), anchor=torch.zeros(1, T, 64), tags=tags)


def _velocity(bundle, state, raw_times):
    return -0.5 * state + bundle.offset


class _Codec:
    def __init__(self):
        self.windows: list = []
        self.released: list = []

    def decode_window(self, latent, start, n):
        self.windows.append((start, n))
        return torch.zeros(2, n * 1920)

    def decode_full(self, latent):
        return torch.zeros(2, latent.shape[1] * 1920 - 64)

    def release_bundle(self, bundle):
        self.released.append(bundle)


def _backend(*, depth=1, recompose=None, song=None, song_b=None, codec=None, submit=None, **kw):
    extra = {"submit": submit} if submit is not None else {}
    kw.setdefault("settled_nap_s", 0.0)
    return YuE2Backend(
        adapter=YuE2Adapter(_velocity, steps=STEPS),
        codec=codec or _Codec(),
        song=song or _song(),
        song_b=song_b,
        knob_state=KnobState(yue2_knob_specs()),
        recompose=recompose,
        steps=STEPS,
        depth=depth,
        **extra,
        **kw,
    )


def _recompose_to(offset, frames=T):
    calls = []

    def recompose(tags, epoch):
        calls.append((tags, epoch))
        return Song(bundle=_Bundle(offset, frames=frames), anchor=None, tags=tags, epoch=epoch)

    recompose.calls = calls
    return recompose


def _knobs(**over):
    knobs = {"yue2_denoise": 1.0, "seed": 1, "x0_target": 0.0, "feedback": 0.0,
             "feedback_depth": 1}
    knobs.update(over)
    return knobs


def _produce_until_fresh(backend, knobs, limit=50):
    for _ in range(limit):
        if backend.produce(knobs, CTX, "generate"):
            return backend.pipeline.last_finished_request
    raise AssertionError("no fresh latent")


def test_contract_surface():
    backend = _backend()
    assert isinstance(backend, GeneratorBackend)
    caps = backend.capabilities()
    assert caps.refines_audio and caps.loop_band and caps.render_anchor_queue
    assert not (caps.swap or caps.lora or caps.timbre or caps.write_audio or caps.curves)
    geo = backend.geometry()
    assert (geo.sample_rate, geo.channels, geo.chunk_rate_hz) == (48000, 2, 25.0)
    assert geo.duration_s == pytest.approx((T * 1920 - 64) / 48000)
    assert backend.max_duration_s() == YUE2_MAX_SONG_S
    names = [s.name for s in backend.knob_specs()]
    assert names == ["yue2_denoise", "x0_target", "feedback", "feedback_depth", "seed"]
    assert backend.rebuild_imminent({"steps_override": 8}) is False
    assert backend.lora_available() is False and backend.list_loras() == []
    with pytest.raises(RuntimeError):
        backend.register_lora("x.safetensors")


def test_ring_emits_after_the_schedule_and_attributes_the_bundle():
    song = _song(0.25)
    backend = _backend(song=song)
    req = _produce_until_fresh(backend, _knobs())
    assert req.aux_cond is song.bundle and req.latent_frames == T
    assert backend.pipeline.ticks == STEPS


def test_prompt_change_recomposes_and_the_ring_anchors_the_new_song():
    old = _song(0.0)
    recompose = _recompose_to(1.0)
    backend = _backend(song=old, recompose=recompose)
    knobs = _knobs(seed=0)  # the song seed: the knobs ask for the anchor
    _produce_until_fresh(backend, knobs)
    backend.handle_set_prompt("rock")
    assert recompose.calls == [("rock", 1)]
    new = backend._active
    assert new is not old and new.anchor is None and new.tags == "rock"
    req = _produce_until_fresh(backend, knobs)
    # The anchor request: full solve from the song seed, no source/lock.
    assert req.aux_cond is new.bundle and req.denoise == 1.0
    assert req.source_latents is None and req.x0_target is None
    assert new.anchor is not None
    # Knobs equal to "the anchor": settled straight away.
    ticks = backend.pipeline.ticks
    assert backend.produce(knobs, CTX, "generate") is False
    assert backend.pipeline.ticks == ticks
    # Any other knob renders over the NEW anchor.
    req = _produce_until_fresh(backend, _knobs(seed=0, yue2_denoise=0.5))
    assert req.x0_target is new.anchor and req.source_latents is new.anchor


def test_latents_in_flight_for_the_old_song_are_dropped():
    backend = _backend(depth=1, recompose=_recompose_to(1.0))
    knobs = _knobs(seed=2)
    backend.produce(knobs, CTX, "generate")  # a slot starts on the old song
    backend.handle_set_prompt("rock")
    emerged = []
    for _ in range(3 * STEPS):
        if backend.produce(knobs, CTX, "generate"):
            emerged.append(backend.pipeline.last_finished_request)
    assert emerged and all(r.aux_cond is backend._active.bundle for r in emerged)


def test_unchanged_tags_do_not_recompose_and_b_follows_a():
    recompose = _recompose_to(1.0)
    backend = _backend(song=_song(tags="pop"), recompose=recompose)
    backend.handle_set_prompt("pop")
    assert recompose.calls == []
    backend.handle_set_prompt("rock", tags_b="jazz")
    assert [c[0] for c in recompose.calls] == ["rock", "jazz"]
    assert backend._song_a.tags == "rock" and backend._song_b.tags == "jazz"
    backend.handle_set_prompt("rock")  # B now follows A again
    assert backend._song_b is backend._song_a


def test_a_recompose_of_the_wrong_length_is_rejected():
    codec = _Codec()
    backend = _backend(codec=codec, recompose=_recompose_to(0.0, frames=T + 5))
    before = backend._active
    backend.handle_set_prompt("rock")
    assert backend._active is before
    assert len(codec.released) == 1
    with pytest.raises(RuntimeError, match="recompose"):
        _backend().handle_set_prompt("rock")


def test_a_superseded_recompose_never_publishes():
    from concurrent.futures import Future

    jobs = []

    def deferred(fn, *args):
        jobs.append((fn, args))
        return Future()

    codec = _Codec()
    backend = _backend(codec=codec, recompose=_recompose_to(1.0), submit=deferred)
    before = backend._active
    backend.handle_set_prompt("rock")
    backend.handle_set_prompt("jazz")
    fn, args = jobs[0]
    assert fn(*args) is None  # superseded before it started: skipped
    fn, args = jobs[1]
    song = fn(*args)
    assert backend._active is song and song.tags == "jazz" and before is not song


def test_settled_short_circuit_only_when_nothing_changed():
    backend = _backend()
    knobs = _knobs()
    _produce_until_fresh(backend, knobs)
    ticks = backend.pipeline.ticks
    assert backend.produce(knobs, CTX, "generate") is False
    assert backend.pipeline.ticks == ticks  # no GPU work while settled
    moved = _knobs(seed=2)
    assert backend.produce(moved, CTX, "generate") is False
    assert backend.pipeline.ticks == ticks + 1
    _produce_until_fresh(backend, moved)
    # Feedback depends on history: it never settles.
    fed = _knobs(seed=2, feedback=0.5)
    ticks = backend.pipeline.ticks
    for _ in range(2 * STEPS):
        backend.produce(fed, CTX, "generate")
    assert backend.pipeline.ticks == ticks + 2 * STEPS


def test_prompt_blend_is_a_hard_switch_at_half():
    a, b = _song(0.0), _song(1.0, tags="jazz")
    backend = _backend(song=a, song_b=b)
    backend.handle_set_prompt_blend(0.49)
    assert backend._active is a
    backend.handle_set_prompt_blend(0.5)
    assert backend._active is b
    backend.handle_set_prompt_blend(0.0)
    assert backend._active is a


def test_steps_stay_on_the_released_grid_whatever_the_knobs_say():
    backend = _backend()
    backend.produce(_knobs(steps_override=8), CTX, "generate")
    assert backend.adapter.steps == STEPS and backend.pipeline.config.infer_steps == STEPS


def test_render_window_clamps_at_the_song_edges():
    codec = _Codec()
    backend = _backend(codec=codec)
    _produce_until_fresh(backend, _knobs())
    n = backend.window_frames()
    assert n == 10  # 0.4 s = two 5-frame window cores
    chunk = backend.render_window(0.0)
    assert codec.windows[-1] == (0, n) and chunk.start_sample == 0
    chunk = backend.render_window(10.0)  # past the end
    assert codec.windows[-1] == (T - n, n)
    assert chunk.start_sample == (T - n) * 1920
    # The last window stops at the full decode's length (1920*T - 64).
    assert chunk.pcm.shape == (n * 1920 - 64, 2)
    full = backend.render_full()
    assert full.pcm.shape == (T * 1920 - 64, 2)


def test_window_plan_interior_and_edges():
    assert window_plan(100, 10, 400) == [(84, 16, 5), (89, 16, 5)]
    assert window_plan(0, 10, 400) == [(0, 0, 5), (0, 5, 5)]
    assert window_plan(395, 5, 400) == [(363, 32, 5)]
    assert window_plan(398, 10, 400) == [(363, 35, 2)]
    with pytest.raises(ValueError):
        window_plan(0, 5, 20)


def test_close_releases_every_bundle():
    codec = _Codec()
    a, b = _song(0.0), _song(1.0, tags="jazz")
    backend = _backend(codec=codec, song=a, song_b=b)
    backend.close()
    assert set(map(id, codec.released)) == {id(a.bundle), id(b.bundle)}


def test_emerged_params_are_stamped():
    class _State:
        params: dict = {}
        prompt_text = "pop"
        interp_feedback = "slerp"

    state = _State()
    state.params = {}
    backend = _backend(state=state)
    _produce_until_fresh(backend, _knobs(yue2_denoise=0.5, seed=9))
    backend.on_fresh_generation({})
    assert state.params["gen_yue2_denoise"] == 0.5
    assert state.params["gen_seed"] == 9 and state.params["gen_cond_epoch"] == 0
    assert state.params["yue2_denoise"] == 0.5 and state.params["num_gens"] == 1


def test_playable_seconds():
    assert playable_seconds(25) == pytest.approx((25 * 1920 - 64) / 48000)


# ---------------------------------------------------------------------------
# Family registration
# ---------------------------------------------------------------------------


def test_family_is_registered():
    from acestep.streaming.families import (
        FAMILY_SPECS,
        YUE2_TEXT_ONLY_MAX_DURATION_S,
        get_family,
        resolve_checkpoint,
    )

    spec = get_family("yue2")
    assert FAMILY_SPECS["yue2"] is spec
    assert [s.name for s in spec.knob_universe()] == [s.name for s in yue2_knob_specs()]
    assert resolve_checkpoint("yue2-3b") == ("yue2", "YuE2-3B")
    assert spec.warmup_policy == "none" and spec.prompt_policy == "acestep"
    assert spec.supports_extensions is False and spec.accepts_checkpoint_dir is False
    assert YUE2_TEXT_ONLY_MAX_DURATION_S == YUE2_MAX_SONG_S
    assert spec.text_only.max_duration_s == YUE2_MAX_SONG_S
    assert spec.text_only.default_duration_s == 60.0
    assert spec.text_only.duration_field == "yue2_duration_s"
    assert [cf.name for cf in spec.config_fields] == ["yue2_lyrics", "yue2_duration_s"]
    assert spec.shutdown is not None


def test_config_fields_parse():
    from acestep.streaming.config import SessionConfig

    cfg = SessionConfig.from_dict({"yue2_lyrics": "[Verse]\nhi\n", "yue2_duration_s": "45"})
    assert cfg.family_config["yue2_lyrics"] == "[Verse]\nhi\n"
    assert cfg.family_config["yue2_duration_s"] == 45.0
    empty = SessionConfig.from_dict({}).family_config
    assert empty["yue2_lyrics"] is None and empty["yue2_duration_s"] is None


def test_make_backend_requires_the_create_path():
    from acestep.streaming.families import make_backend

    class _Session:
        backend_init = None

    with pytest.raises(ValueError, match="create path"):
        make_backend("yue2", _Session())


def _fake_weights(root, *, size=4):
    for name in ("YuE2-3B", "YuE2-Vae"):
        d = root / name
        d.mkdir(parents=True)
        (d / "config.json").write_text("{}")
        (d / "model.safetensors").write_bytes(b"x" * size)
        (d / "weights_manifest.json").write_text(json.dumps(
            {"files": {"model.safetensors": {"sha256": "0", "bytes": 4}}}))


def _preflight():
    from acestep.streaming.families import get_family
    from acestep.streaming.preflight import PreflightRequest

    return get_family("yue2").preflight(PreflightRequest(model_id="YuE2-3B"))


def test_preflight_verdicts(tmp_path, monkeypatch):
    from acestep.engine import yue2_runtime as rt

    monkeypatch.delenv(rt.ROOT_ENV, raising=False)
    monkeypatch.delenv(rt.TRT_DIR_ENV, raising=False)
    res = _preflight()
    assert not res.ok and "weights" in res.title

    _fake_weights(tmp_path / "w", size=5)
    monkeypatch.setenv(rt.ROOT_ENV, str(tmp_path / "w"))
    res = _preflight()
    assert not res.ok and "manifest says 4" in res.lines[0]

    (tmp_path / "w2").mkdir()
    _fake_weights(tmp_path / "w2" / "r")
    monkeypatch.setenv(rt.ROOT_ENV, str(tmp_path / "w2" / "r"))
    monkeypatch.setattr(rt, "missing_modules", lambda: ["yue2"])
    res = _preflight()
    assert not res.ok and "importable" in res.title

    monkeypatch.setattr(rt, "missing_modules", lambda: [])
    assert _preflight().ok

    trt = tmp_path / "trt"
    (trt / "flexible_song").mkdir(parents=True)
    (trt / "flexible_song" / "velocity.trt").write_bytes(b"")
    monkeypatch.setenv(rt.TRT_DIR_ENV, str(trt))
    res = _preflight()
    assert not res.ok and "vae_fp32_t37.trt" in res.lines[0]
    (trt / "vae_fp32_t37.trt").write_bytes(b"")
    assert _preflight().ok


@pytest.mark.parametrize("depth", [1, 2])
@pytest.mark.parametrize("denoise", [1.0, 0.5])
def test_settles_at_the_released_step_count(depth, denoise):
    """A 32-step solve outlives a short submission history; the ring must
    still recognise the emerged latent and stop ticking."""
    backend = YuE2Backend(
        adapter=YuE2Adapter(_velocity, steps=32), codec=_Codec(), song=_song(),
        knob_state=KnobState(yue2_knob_specs()), steps=32, depth=depth, settled_nap_s=0.0,
    )
    knobs = _knobs(yue2_denoise=denoise)
    for _ in range(4 * 32):
        backend.produce(knobs, CTX, "generate")
    ticks = backend.pipeline.ticks
    assert ticks < 4 * 32
    assert backend.produce(knobs, CTX, "generate") is False
    assert backend.pipeline.ticks == ticks


class _FakeContext:
    def __init__(self):
        self.shapes = {}

    def set_input_shape(self, name, shape):
        self.shapes[name] = shape
        return True

    def set_tensor_address(self, name, ptr):
        pass

    def get_tensor_shape(self, name):
        return self.shapes["state"]

    def execute_async_v3(self, stream):
        return True


class _FakeEngine:
    def create_execution_context(self):
        return _FakeContext()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="binds on the current CUDA stream")
def test_trt_binding_made_under_inference_mode_runs_outside_it():
    """The anchor solve binds under inference_mode; the runner then calls
    the same binding outside it and must be able to write the staging
    buffers."""
    from types import SimpleNamespace

    from acestep.engine.yue2_trt import _BoundExecution

    z = torch.zeros(1, device="cuda")
    bundle = SimpleNamespace(frames=8, keys=z, values=z, nar=SimpleNamespace(cos=z, sin=z, pos_emb=z))
    with torch.inference_mode():
        bound = _BoundExecution(_FakeEngine(), bundle, 1, "cuda")
    out = bound(torch.ones(1, 8, 64, device="cuda", dtype=torch.bfloat16), [0.5])
    assert out.shape == (1, 8, 64) and not out.is_inference()


def test_ring_gpu_work_waits_for_a_held_gate():
    """A conditioning-worker graph capture holds the gate; the ring's
    produce and render must not run GPU work meanwhile."""
    import threading

    gate = threading.Lock()
    backend = _backend(gpu_gate=gate)
    done = threading.Event()
    gate.acquire()
    worker = threading.Thread(target=lambda: (backend.produce(_knobs(), CTX, "generate"), done.set()))
    worker.start()
    assert not done.wait(0.2)
    gate.release()
    assert done.wait(5.0)
    worker.join()


def test_repeat_window_renders_of_one_latent_decode_once():
    codec = _Codec()
    backend = _backend(codec=codec)
    _produce_until_fresh(backend, _knobs())
    first = backend.render_window(0.4)
    first.pcm[:] = 1.0  # the runner crossfades in place
    again = backend.render_window(0.4)
    assert len(codec.windows) == 1
    assert not np.any(again.pcm == 1.0)
    _produce_until_fresh(backend, _knobs(seed=5))  # a new latent: decode again
    backend.render_window(0.4)
    assert len(codec.windows) == 2


def test_a_published_song_wakes_the_idle_runner():
    class _State:
        last_activity_ts = 0.0
        params: dict = {}

    state = _State()
    backend = _backend(state=state, recompose=_recompose_to(1.0))
    backend.handle_set_prompt("rock")
    assert state.last_activity_ts > 0.0


def test_a_settled_ring_paces_the_runner():
    import time as _time

    backend = _backend(settled_nap_s=0.05)
    knobs = _knobs()
    _produce_until_fresh(backend, knobs)
    t0 = _time.perf_counter()
    assert backend.produce(knobs, CTX, "generate") is False
    assert _time.perf_counter() - t0 >= 0.05


# ---- review fixes: latest-wins on a revert, close race, failures, depth ----


def _deferred():
    """A ``submit`` that queues jobs; ``run(i)`` runs job i on the caller."""
    from concurrent.futures import Future

    jobs = []

    def submit(fn, *args):
        future = Future()
        jobs.append((fn, args, future))
        return future

    def run(i):
        fn, args, future = jobs[i]
        try:
            future.set_result(fn(*args))
        except BaseException as exc:  # noqa: BLE001
            future.set_exception(exc)
        return future

    submit.jobs, submit.run = jobs, run
    return submit


def test_going_back_to_the_playing_prompt_keeps_it_published():
    """A -> X -> A: the X job must not publish (it used to: the revert
    made no new request, so X's token stayed current)."""
    a = _song(0.0, tags="pop")
    codec = _Codec()
    submit = _deferred()
    backend = _backend(song=a, codec=codec, recompose=_recompose_to(1.0), submit=submit)
    backend.handle_set_prompt("rock")
    backend.handle_set_prompt("pop")
    assert len(submit.jobs) == 1  # the revert queues nothing
    assert submit.run(0).result() is None
    assert backend._song_a is a and backend._active is a


def test_a_revert_during_the_build_releases_the_abandoned_song():
    a = _song(0.0, tags="pop")
    codec = _Codec()
    holder = {}

    def recompose(tags, epoch):
        holder["backend"].handle_set_prompt("pop")  # the user goes back mid-build
        return Song(bundle=_Bundle(1.0), anchor=None, tags=tags, epoch=epoch)

    backend = _backend(song=a, codec=codec, recompose=recompose)
    holder["backend"] = backend
    backend.handle_set_prompt("rock")
    assert backend._song_a is a and backend._active is a
    assert len(codec.released) == 1 and codec.released[0].offset == 1.0


def test_b_going_back_to_following_a_drops_the_b_job():
    a = _song(0.0, tags="pop")
    codec = _Codec()
    submit = _deferred()
    backend = _backend(song=a, codec=codec, recompose=_recompose_to(1.0), submit=submit)
    backend.handle_set_prompt("pop", tags_b="jazz")
    backend.handle_set_prompt("pop")  # B follows A again
    assert submit.run(0).result() is None
    assert backend._song_b is a
    # A B song that slipped past the job's check is released at publish.
    late = Song(bundle=_Bundle(2.0), anchor=None, tags="jazz")
    backend._publish_song("b", late, 0.0)
    assert backend._song_b is a and codec.released[-1] is late.bundle


def test_close_during_a_recompose_releases_its_song():
    import threading

    codec = _Codec()
    started, finish = threading.Event(), threading.Event()

    def recompose(tags, epoch):
        started.set()
        finish.wait(5.0)
        return Song(bundle=_Bundle(1.0), anchor=None, tags=tags, epoch=epoch)

    def threaded(fn, *args):
        from concurrent.futures import Future

        future = Future()
        future.set_running_or_notify_cancel()  # as a pool worker picks the job up
        threading.Thread(target=lambda: future.set_result(fn(*args))).start()
        return future

    a = _song(0.0)
    backend = _backend(song=a, codec=codec, recompose=recompose, submit=threaded)
    backend.handle_set_prompt("rock")
    assert started.wait(5.0)
    backend.close()
    finish.set()
    for _ in range(100):
        if len(codec.released) == 2:
            break
        import time as _time

        _time.sleep(0.02)
    assert {b.offset for b in codec.released} == {0.0, 1.0}
    assert backend._songs == []
    # A publish that passed the job's check before close also releases.
    late = Song(bundle=_Bundle(3.0), anchor=None, tags="jazz")
    backend._publish_song("a", late, 0.0)
    assert codec.released[-1] is late.bundle and backend._song_a is a


def test_close_unqueues_a_recompose_that_has_not_started():
    submit = _deferred()
    backend = _backend(recompose=_recompose_to(1.0), submit=submit)
    backend.handle_set_prompt("rock")
    backend.close()
    assert submit.jobs[0][2].cancelled()


def test_a_failed_recompose_is_reported_and_the_song_keeps_playing():
    errors = []

    def recompose(tags, epoch):
        raise RuntimeError("semantic stage returned no tokens")

    a = _song(0.0)
    backend = _backend(song=a, recompose=recompose, on_error=lambda c, m: errors.append((c, m)))
    backend.handle_set_prompt("rock")
    assert backend._active is a
    assert len(errors) == 1 and errors[0][0] == "yue2_recompose_failed"
    assert "rock" in errors[0][1] and "no tokens" in errors[0][1]


def test_a_failed_recompose_publishes_the_session_error_event():
    from types import SimpleNamespace

    from acestep.streaming.events import SessionError
    from acestep.streaming.families import get_family

    published = []

    class _Ctx:
        velocity = staticmethod(_velocity)
        device = torch.device("cpu")
        gpu_gate = None

        def compose(self, **kw):
            raise RuntimeError("boom")

        def submit(self, fn, *args):
            from acestep.streaming.yue2_recompose import run_inline

            return run_inline(fn, *args)

    import threading

    ctx = _Ctx()
    ctx.gpu_gate = threading.Lock()
    ss = SimpleNamespace(
        backend_init={"context": ctx, "composition": SimpleNamespace(lyrics="", seed=0),
                      "song": _song(0.0)},
        virtual_knobs=KnobState(yue2_knob_specs()), state=SimpleNamespace(current_depth=1, params={}),
        vae_window=0.4, bus=SimpleNamespace(publish=published.append),
    )
    backend = get_family("yue2").make_backend(ss)
    backend.handle_set_prompt("rock")
    assert len(published) == 1 and isinstance(published[0], SessionError)
    assert published[0].code == "yue2_recompose_failed"
