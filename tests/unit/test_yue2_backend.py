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
import math

import numpy as np
import pytest
import torch

from acestep.engine.yue2_adapter import YuE2Adapter
from acestep.engine.yue2_trt import window_plan
from acestep.streaming.generator_backend import GeneratorBackend, TickContext
from acestep.streaming.knobs import KnobState
from acestep.engine.yue2_velocity import truncated_grid, upstream_noise
from acestep.streaming.yue2_backend import (
    YUE2_MAX_SONG_S,
    YUE2_STEP_CHOICES,
    YuE2Backend,
    playable_seconds,
    snap_steps,
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
    assert names == ["yue2_denoise", "yue2_steps", "x0_target", "feedback", "feedback_depth",
                     "seed"]
    steps_spec = next(s for s in backend.knob_specs() if s.name == "yue2_steps")
    assert steps_spec.default == 32 and steps_spec.type == "int"
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


def test_shared_steps_override_is_ignored():
    backend = _backend()
    backend.produce(_knobs(steps_override=8), CTX, "generate")
    assert backend.adapter.steps == STEPS and backend.pipeline.config.infer_steps == STEPS


@pytest.mark.parametrize("raw, snapped", [(32, 32), (8, 8), (9, 8), (10, 12), (14, 16),
                                          (20, 24), (28, 32), (1, 4), (99, 32), ("x", 32)])
def test_steps_snap_to_the_validated_grids(raw, snapped):
    assert snap_steps(raw) == snapped
    assert snapped in YUE2_STEP_CHOICES


def test_yue2_steps_default_is_the_released_grid():
    backend = YuE2Backend(
        adapter=YuE2Adapter(_velocity, steps=32), codec=_Codec(), song=_song(),
        knob_state=KnobState(yue2_knob_specs()), settled_nap_s=0.0,
    )
    knobs = {**backend.read_knobs(), "seed": 1}
    assert knobs["yue2_steps"] == 32
    _produce_until_fresh(backend, knobs, limit=80)
    assert backend.adapter.steps == 32 and backend.pipeline.ticks == 32


@pytest.mark.parametrize("steps", YUE2_STEP_CHOICES)
@pytest.mark.parametrize("denoise", [1.0, 0.5])
def test_yue2_steps_switches_the_grid_and_matches_an_explicit_midpoint_solve(steps, denoise):
    """Through the backend: after a settled 32-step render, ``yue2_steps``
    restarts the ring on the chosen grid and the emerged latent equals an
    explicit midpoint solve on that grid (re-noised anchor at ``denoise``)."""
    anchor = torch.linspace(-1, 1, T * 64).view(1, T, 64)
    song = Song(bundle=_Bundle(0.1), anchor=anchor, tags="pop")
    backend = YuE2Backend(
        adapter=YuE2Adapter(_velocity, steps=32), codec=_Codec(), song=song,
        knob_state=KnobState(yue2_knob_specs()), settled_nap_s=0.0,
    )
    _produce_until_fresh(backend, _knobs(yue2_steps=32), limit=80)
    while backend.produce(_knobs(yue2_steps=32), CTX, "generate"):
        pass
    before = backend.pipeline
    ticks_before = before.ticks
    knobs = _knobs(yue2_steps=steps, yue2_denoise=denoise, seed=5)
    req = _produce_until_fresh(backend, knobs, limit=80)
    assert backend.adapter.steps == steps and backend.pipeline.config.infer_steps == steps
    assert (backend.pipeline is before) == (steps == 32)  # a new grid restarts the ring
    ran = backend.pipeline.ticks - (ticks_before if backend.pipeline is before else 0)
    assert ran == math.ceil(steps * denoise)
    schedule = truncated_grid(steps, denoise)
    s0 = schedule[0].item()
    x = s0 * upstream_noise(5, T)[None] + (1.0 - s0) * anchor if denoise < 1 else upstream_noise(5, T)[None]
    h = 1.0 / steps
    for s in schedule[:-1].tolist():
        first = _velocity(song.bundle, x, None)
        x = x - _velocity(song.bundle, x - first * (h / 2), None) * h
    assert req.aux_cond is song.bundle
    assert torch.equal(backend._last_result_latent, x)
    assert backend.state is None or backend.state.params["gen_yue2_steps"] == steps
    # Settled on the new grid: no more ticks.
    ticks = backend.pipeline.ticks
    assert backend.produce(knobs, CTX, "generate") is False
    assert backend.pipeline.ticks == ticks


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


def test_depth_two_settles_without_a_frozen_slot():
    """At depth >= 2 the other slot used to freeze mid-solve at settle and
    emerge first on the next change, carrying the old request (with the
    new shared x0 strength). Now it is dropped at settle."""
    backend = _backend(depth=2)
    knobs = _knobs()
    _produce_until_fresh(backend, knobs)
    assert backend.produce(knobs, CTX, "generate") is False
    assert backend.pipeline.active_slots == 0
    moved = _knobs(x0_target=0.5)
    req = _produce_until_fresh(backend, moved)
    assert req.x0_target_strength == 0.5


def test_a_published_song_restarts_the_ring_instead_of_finishing_the_old_solve():
    backend = _backend(recompose=_recompose_to(1.0))
    knobs = _knobs(seed=2)
    backend.produce(knobs, CTX, "generate")  # a slot starts on the old song
    backend.handle_set_prompt("rock")
    for n in range(1, 3 * STEPS):
        if backend.produce(knobs, CTX, "generate"):
            break
    # The new song's anchor emerged after one full solve, not after the
    # old slot's remaining steps plus one.
    assert n == STEPS
    assert backend.pipeline.last_finished_request.aux_cond is backend._active.bundle


# ---- review v2 fixes ----


def test_a_repeated_prompt_during_the_build_keeps_the_job_in_flight():
    """A second identical ``set_prompt`` while A re-composes used to
    supersede the running job and build the same song again."""
    codec = _Codec()
    submit = _deferred()
    recompose = _recompose_to(1.0)
    backend = _backend(codec=codec, recompose=recompose, submit=submit)
    backend.handle_set_prompt("rock")
    backend.handle_set_prompt("rock")
    assert len(submit.jobs) == 1
    song = submit.run(0).result()
    assert song is not None and backend._active is song and song.tags == "rock"
    assert recompose.calls == [("rock", 1)] and codec.released == []


def test_a_b_only_change_during_an_a_build_leaves_a_alone():
    submit = _deferred()
    backend = _backend(song=_song(tags="pop"), recompose=_recompose_to(1.0), submit=submit)
    backend.handle_set_prompt("rock")
    backend.handle_set_prompt("rock", tags_b="jazz")  # the page sends A with B
    assert [args[1] for _, args, _ in submit.jobs] == ["rock", "jazz"]
    song_a = submit.run(0).result()
    song_b = submit.run(1).result()
    assert backend._song_a is song_a and backend._song_b is song_b
    assert (song_a.tags, song_b.tags) == ("rock", "jazz")


def test_resending_a_prompt_after_its_build_failed_retries_it():
    calls = []

    def recompose(tags, epoch):
        calls.append(tags)
        if len(calls) == 1:
            raise RuntimeError("semantic stage returned no tokens")
        return Song(bundle=_Bundle(1.0), anchor=None, tags=tags, epoch=epoch)

    backend = _backend(recompose=recompose, on_error=lambda c, m: None)
    backend.handle_set_prompt("rock")
    backend.handle_set_prompt("rock")
    assert calls == ["rock", "rock"] and backend._active.tags == "rock"


def test_repeated_recomposes_keep_no_dead_bundles():
    """Each re-composed song's bundle is freed once nothing can use it;
    only the last six used to be checked, so up to five dead ones stayed."""
    codec = _Codec()
    first = _song(0.0, tags="pop")
    made = [first.bundle]

    def recompose(tags, epoch):
        song = Song(bundle=_Bundle(float(epoch)), anchor=None, tags=tags, epoch=epoch)
        made.append(song.bundle)
        return song

    backend = _backend(song=first, codec=codec, recompose=recompose)
    knobs = _knobs(seed=0)
    _produce_until_fresh(backend, knobs)
    for i in range(10):
        backend.handle_set_prompt(f"style {i}")
        _produce_until_fresh(backend, knobs)
        live = [b for b in made if all(b is not r for r in codec.released)]
        assert live == [backend._active.bundle], f"after re-compose {i}: {len(live)} live"
    assert len(codec.released) == len(set(map(id, codec.released)))  # each freed once
    backend.close()
    assert len(codec.released) == len(made)


def test_a_replaced_song_is_freed_on_the_runner_not_at_publish():
    """The ring may still be mid-solve on the old song when the worker
    publishes; its bundle is freed by the runner after it moves on."""
    codec = _Codec()
    old = _song(0.0)
    backend = _backend(song=old, codec=codec, recompose=_recompose_to(1.0))
    backend.produce(_knobs(seed=2), CTX, "generate")  # a slot on the old song
    backend.handle_set_prompt("rock")
    assert codec.released == []
    backend.produce(_knobs(seed=2), CTX, "generate")  # restarts on the new song
    backend.produce(_knobs(seed=2), CTX, "generate")
    assert codec.released == [old.bundle]


# ---- review v2: engine profile limits come from the engine ----


class _ProfileContext(_FakeContext):
    def __init__(self, limits):
        super().__init__()
        self.limits = limits

    def set_input_shape(self, name, shape):
        lo, hi = self.limits.get(name, (None, None))
        if lo is not None and not all(a <= s <= b for a, s, b in zip(lo, shape, hi)):
            return False
        return super().set_input_shape(name, shape)


class _ProfileEngine:
    """Reports ``(min, opt, max)`` per input like TensorRT; its contexts
    refuse shapes outside ``limits`` (which may be narrower than the
    reported profile, to model a bind that fails anyway)."""

    def __init__(self, profile, limits=None):
        self.profile = profile
        self.limits = limits if limits is not None else {k: (v[0], v[2]) for k, v in profile.items()}

    def get_tensor_profile_shape(self, name, index):
        return self.profile[name]

    def create_execution_context(self):
        return _ProfileContext(self.limits)


class _NarBundle:
    def __init__(self, frames, cond_tokens):
        self.frames, self.seed = frames, 0
        self.keys = self.values = torch.zeros(2, cond_tokens, 1)
        pad = torch.zeros(1, frames + 2, 1)
        from types import SimpleNamespace

        self.nar = SimpleNamespace(cos=pad, sin=pad, pos_emb=pad)

    @property
    def cond_tokens(self):
        return int(self.keys.shape[1])


def _narrow_profile():
    return {
        "state": ([1, 1000, 64], [1, 1000, 64], [2, 2000, 64]),
        "raw_time": ([1], [1], [2]),
        "keys": ([2, 1001, 1], [2, 1001, 1], [2, 3000, 1]),
        "values": ([2, 1001, 1], [2, 1001, 1], [2, 3000, 1]),
        "cos": ([1, 1002, 1], [1, 1002, 1], [1, 2002, 1]),
        "sin": ([1, 1002, 1], [1, 1002, 1], [1, 2002, 1]),
        "position": ([1, 1002, 1], [1, 1002, 1], [1, 2002, 1]),
    }


def _trt_velocity(monkeypatch, engine):
    from acestep.engine import yue2_trt

    monkeypatch.setattr(yue2_trt, "_deserialize", lambda path: engine)
    eager_calls = []

    def eager(bundle, state, raw_times):
        eager_calls.append(bundle)
        return torch.zeros_like(state)

    velocity = yue2_trt.TRTVelocity("unused", eager, device="cpu")
    return velocity, eager_calls


def test_trt_velocity_reads_its_profile_from_the_engine(monkeypatch):
    """A song inside the code's constants (<= 2500 frames, <= 4000
    tokens) but outside a narrower engine runs eager instead of failing."""
    velocity, eager_calls = _trt_velocity(monkeypatch, _ProfileEngine(_narrow_profile()))
    inside, long_song, long_cond = _NarBundle(1500, 2000), _NarBundle(2200, 2000), _NarBundle(1500, 3500)
    assert velocity.covers(inside)
    assert not velocity.covers(long_song) and not velocity.covers(long_cond)
    for bundle in (long_song, long_cond):
        out = velocity(bundle, torch.zeros(1, bundle.frames, 64), [0.5])
        assert out.shape == (1, bundle.frames, 64)
    assert eager_calls == [long_song, long_cond]
    # Batch over the engine's max batch (2 here, not the constant 4).
    velocity(inside, torch.zeros(3, 1500, 64), [0.5] * 3)
    assert eager_calls[-1] is inside
    assert velocity.path_for(inside) == "trt" and velocity.path_for(long_song) == "eager"


def test_a_bind_the_engine_refuses_falls_back_to_eager_for_that_bundle(monkeypatch):
    profile = _narrow_profile()
    limits = {k: (v[0], v[2]) for k, v in profile.items()}
    limits["cos"] = ([1, 1002, 1], [1, 1200, 1])  # narrower than reported
    velocity, eager_calls = _trt_velocity(monkeypatch, _ProfileEngine(profile, limits))
    bundle = _NarBundle(1500, 2000)
    assert velocity.covers(bundle)
    for _ in range(3):
        velocity(bundle, torch.zeros(1, 1500, 64), [0.5])
    assert eager_calls == [bundle] * 3
    assert velocity.path_for(bundle) == "eager"


def test_telemetry_reports_the_nar_path_of_the_song_that_plays():
    """A re-composed song can fall off the TRT profile (conditioning past
    its token bound); ``yue2_nar`` used to keep saying ``trt``."""

    class _PathVelocity:
        def __call__(self, bundle, state, raw_times):
            return _velocity(bundle, state, raw_times)

        def path_for(self, bundle):
            return "eager" if bundle.offset == 1.0 else "trt"

    class _State:
        prompt_text = "pop"
        interp_feedback = "slerp"
        last_activity_ts = 0.0

    state = _State()
    state.params = {"yue2_nar": "trt"}
    backend = YuE2Backend(
        adapter=YuE2Adapter(_PathVelocity(), steps=STEPS), codec=_Codec(), song=_song(0.0),
        knob_state=KnobState(yue2_knob_specs()), recompose=_recompose_to(1.0), steps=STEPS,
        state=state, settled_nap_s=0.0,
    )
    knobs = _knobs(seed=0)
    _produce_until_fresh(backend, knobs)
    assert state.params["yue2_nar"] == "trt"
    backend.handle_set_prompt("rock")
    _produce_until_fresh(backend, knobs)
    assert state.params["yue2_nar"] == "eager"


def test_telemetry_says_which_slots_are_recomposing():
    """``yue2_recomposing`` rides every params_update, so the page can
    show the composing state (it used to show nothing for ~30 s)."""

    class _State:
        last_activity_ts = 0.0

    state = _State()
    state.params = {}
    submit = _deferred()
    backend = _backend(state=state, recompose=_recompose_to(1.0), submit=submit)
    assert state.params["yue2_recomposing"] == []
    backend.handle_set_prompt("rock", tags_b="jazz")
    assert state.params["yue2_recomposing"] == ["a", "b"]
    submit.run(0)
    assert state.params["yue2_recomposing"] == ["b"]
    backend.handle_set_prompt("rock")  # B follows A again: its job is dropped
    assert state.params["yue2_recomposing"] == []
    backend.handle_set_prompt("punk")
    assert state.params["yue2_recomposing"] == ["a"]
    backend.close()
    assert state.params["yue2_recomposing"] == []


def test_a_failed_recompose_clears_the_recomposing_flag():
    class _State:
        last_activity_ts = 0.0

    def recompose(tags, epoch):
        raise RuntimeError("boom")

    state = _State()
    state.params = {}
    backend = _backend(state=state, recompose=recompose, on_error=lambda c, m: None)
    backend.handle_set_prompt("rock")
    assert state.params["yue2_recomposing"] == []
