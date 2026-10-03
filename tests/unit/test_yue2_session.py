"""The yue2 create path with a fake YuE2Context (no GPU, no weights).

``create_yue2_session`` must build a real StreamingSession whose geometry
comes from the composed song (not from the synthesised text-only
anchor), pick the NAR path the TRT velocity will use, flag truncated
songs, honour ``yue2_duration_s`` as the semantic budget, and compose
``prompt_b`` as a second song at song A's length.
"""

from __future__ import annotations

import threading
from types import SimpleNamespace

import pytest
import torch

from acestep.streaming import yue2_session
from acestep.streaming.yue2_backend import YuE2Backend, playable_seconds

T = 1200


class _FakeVelocity:
    """The TRT velocity's path decision on the builder's profile."""

    def __init__(self, has_trt):
        self.has_trt = has_trt

    def __call__(self, bundle, state, raw):
        return torch.zeros_like(state)

    def path_for(self, bundle):
        from acestep.engine.yue2_trt import flexible_profile_fits

        fits = flexible_profile_fits(bundle.frames, bundle.cond_tokens)
        return "trt" if self.has_trt and fits else "eager"


class _FakeContext:
    def __init__(self, *, frames=T, cond_tokens=2000, truncated=False, has_trt=True):
        self.frames, self.cond_tokens, self.truncated = frames, cond_tokens, truncated
        self.has_trt_nar = has_trt
        self.gpu_gate = threading.Lock()
        self.device = torch.device("cpu")
        self.velocity = _FakeVelocity(has_trt)
        self.compose_calls: list = []
        self.bundle_styles: list = []
        self.released: list = []

    def compose(self, *, style, lyrics, seed, max_frames, exact_frames=None):
        self.compose_calls.append(dict(style=style, lyrics=lyrics, seed=seed, max_frames=max_frames,
                                       exact_frames=exact_frames))
        return SimpleNamespace(style=style, lyrics=lyrics, seed=seed,
                               timings_ms={"plan_ms": 1.0, "semantic_ms": 2.0})

    def bundle(self, composition, *, style=None, epoch=0):
        self.bundle_styles.append(composition.style)
        return SimpleNamespace(frames=self.frames, cond_tokens=self.cond_tokens, seed=composition.seed,
                               truncated=self.truncated, tags=style or composition.style)

    def solve(self, bundle, **kw):
        return torch.zeros(1, bundle.frames, 64)

    def decode_full(self, latent):
        return torch.zeros(2, latent.shape[1] * 1920 - 64)

    def decode_window(self, latent, start, n):
        return torch.zeros(2, n * 1920)

    def release_bundle(self, bundle):
        self.released.append(bundle)

    def submit(self, fn, *args):
        from acestep.streaming.yue2_recompose import run_inline

        return run_inline(fn, *args)


def _create(monkeypatch, context, **config):
    from acestep.streaming.config import SessionConfig
    from acestep.streaming.families import get_family
    from acestep.streaming.session import StreamingSession

    monkeypatch.setattr(yue2_session, "get_yue2_context", lambda: context)
    cfg = SessionConfig.from_dict({"backend": "yue2", "prompt": "city pop", **config})
    return get_family("yue2").create_session(
        StreamingSession, audio=None, config=cfg, checkpoint="YuE2-3B", session_id="t",
        decoder_backend="tensorrt", vae_backend="tensorrt", offload_text_encoder=False,
        checkpoint_dir=None, model_extension=None,
    )


def _close(ss):
    ss.backend.close()
    ss.audio_eng.stop()


def test_create_builds_a_session_from_the_composed_song(monkeypatch):
    context = _FakeContext()
    ss = _create(monkeypatch, context, yue2_lyrics="[Verse]\nla\n", yue2_duration_s=60)
    try:
        assert isinstance(ss.backend, YuE2Backend)
        assert ss.canvas is None and ss.session is None
        assert ss.state.duration == pytest.approx(playable_seconds(T))
        assert ss.initial_buffer.shape == (T * 1920 - 64, 2)
        assert ss.backend.geometry().duration_s == pytest.approx(playable_seconds(T))
        call = context.compose_calls[0]
        assert call["lyrics"] == "[Verse]\nla\n" and call["max_frames"] == 1500
        assert call["style"] == "city pop"
        assert ss.state.params["yue2_nar"] == "trt"
        assert ss.state.params["yue2_truncated"] is False
        assert context.bundle_styles == ["city pop"]  # no prompt_b: one song
    finally:
        _close(ss)


def test_defaults_instrumental_and_full_budget(monkeypatch):
    context = _FakeContext()
    ss = _create(monkeypatch, context)
    try:
        call = context.compose_calls[0]
        assert call["lyrics"] == "[Verse]\n\n[Outro]\n"
        assert call["max_frames"] == 2500
    finally:
        _close(ss)


def test_budget_is_capped_at_the_engine_cover(monkeypatch):
    context = _FakeContext()
    ss = _create(monkeypatch, context, yue2_duration_s=500)
    try:
        assert context.compose_calls[0]["max_frames"] == 2500
    finally:
        _close(ss)


def test_conditioning_past_the_profile_selects_eager_and_truncation_is_flagged(monkeypatch):
    context = _FakeContext(cond_tokens=4500, truncated=True)
    ss = _create(monkeypatch, context)
    try:
        assert ss.state.params["yue2_nar"] == "eager"
        assert ss.state.params["yue2_truncated"] is True
    finally:
        _close(ss)


def test_short_song_without_engines_is_eager(monkeypatch):
    context = _FakeContext(frames=700, has_trt=False)
    ss = _create(monkeypatch, context)
    try:
        assert ss.state.params["yue2_nar"] == "eager"
        assert ss.state.duration == pytest.approx(playable_seconds(700))
    finally:
        _close(ss)


def test_prompt_b_is_a_second_song_at_the_same_length(monkeypatch):
    context = _FakeContext()
    ss = _create(monkeypatch, context, prompt_b="dark techno", yue2_lyrics="[Verse] la")
    try:
        a_call, b_call = context.compose_calls
        assert b_call["style"] == "dark techno" and b_call["lyrics"] == a_call["lyrics"]
        assert b_call["exact_frames"] == T
        ss.backend.handle_set_prompt_blend(1.0)
        song_b = ss.backend._active
        assert song_b.tags == "dark techno" and song_b.anchor is not None
    finally:
        _close(ss)


def test_ring_depth_is_one_whatever_the_client_asks(monkeypatch):
    context = _FakeContext()
    ss = _create(monkeypatch, context, depth=4)
    try:
        assert ss.state.current_depth == 1
        assert ss.max_pipeline_depth == 1
    finally:
        _close(ss)


def test_a_failed_second_song_releases_the_first_bundle(monkeypatch):
    context = _FakeContext()
    compose = context.compose

    def failing_b(**kw):
        if kw.get("exact_frames") is not None:
            raise RuntimeError("song B semantic stage failed")
        return compose(**kw)

    context.compose = failing_b
    with pytest.raises(RuntimeError, match="song B"):
        _create(monkeypatch, context, prompt_b="dark techno")
    assert [b.tags for b in context.released] == ["city pop"]


def test_a_failed_session_construction_releases_every_bundle(monkeypatch):
    from acestep.streaming import yue2_session as mod

    context = _FakeContext()

    def broken(*args, **kwargs):
        raise RuntimeError("session ctor failed")

    monkeypatch.setattr(mod, "_assemble_session", broken)
    with pytest.raises(RuntimeError, match="ctor"):
        _create(monkeypatch, context, prompt_b="dark techno")
    assert sorted(b.tags for b in context.released) == ["city pop", "dark techno"]


def test_create_runs_the_anchor_solve_and_decode_under_the_gpu_gate(monkeypatch):
    context = _FakeContext()
    held = []
    solve, decode = context.solve, context.decode_full
    context.solve = lambda bundle, **kw: (held.append(context.gpu_gate.locked()), solve(bundle))[1]
    context.decode_full = lambda latent: (held.append(context.gpu_gate.locked()), decode(latent))[1]
    ss = _create(monkeypatch, context)
    try:
        assert held == [True, True]
        assert context.released == []  # the session owns the bundle now
    finally:
        _close(ss)


# ---- review v2: songs shorter than the 37-frame VAE window ----


def test_budget_has_a_floor_above_the_vae_window(monkeypatch):
    from acestep.engine.yue2_trt import VAE_CTX_FRAMES

    context = _FakeContext()
    ss = _create(monkeypatch, context, yue2_duration_s=1)
    try:
        assert context.compose_calls[0]["max_frames"] == 50  # 2 s
        assert context.compose_calls[0]["max_frames"] >= VAE_CTX_FRAMES
    finally:
        _close(ss)


def test_a_song_shorter_than_the_vae_window_fails_create_clearly(monkeypatch):
    context = _FakeContext(frames=30, has_trt=False)
    with pytest.raises(ValueError, match="at least 37 frames"):
        _create(monkeypatch, context, yue2_duration_s=2)
    assert len(context.released) == 1  # the bundle made before the check


def test_semantic_budget_never_allows_a_song_under_the_vae_window():
    from acestep.engine.yue2_context import YuE2Context
    from acestep.engine.yue2_trt import VAE_CTX_FRAMES

    eager = SimpleNamespace(has_trt_nar=False)
    assert YuE2Context.semantic_budget(eager, 50)["min_tokens"] == VAE_CTX_FRAMES
    trt = SimpleNamespace(has_trt_nar=True)
    assert YuE2Context.semantic_budget(trt, 500)["min_tokens"] == VAE_CTX_FRAMES
    assert YuE2Context.semantic_budget(trt, 2500)["min_tokens"] == 1000
