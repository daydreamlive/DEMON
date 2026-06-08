"""SA3Backend contract + produce/render mechanics (Tier 1).

Drives the real :class:`~acestep.streaming.sa3_backend.SA3Backend`
(and through it the shared StreamPipeline + SA3Adapter) with a mock
DiT and a fake codec, validating the GeneratorBackend contract the
runner consumes: capability mask, delivery geometry (44.1 kHz native →
48 kHz delivered, decision 2), the sa3_* knob manifest, produce-mode
behavior, the steps_override rebuild path, and the cached full-decode
window rendering with the resample at the decode boundary.
"""

from __future__ import annotations

import numpy as np
import torch

from acestep.engine.sa3_adapter import SA3Adapter
from acestep.streaming.generator_backend import TickContext
from acestep.streaming.knobs import KnobState
from acestep.streaming.sa3_backend import (
    DELIVERY_SAMPLE_RATE,
    SA3_SAMPLE_RATE,
    SA3Backend,
    sa3_knob_specs,
)

C = 256
T = 16                      # latent frames
N44 = T * 4096              # 44.1k-domain samples for T frames
CTX = TickContext(playhead_s=0.0, buffer_duration_s=10.0)


class _ZeroDit(torch.nn.Module):
    def forward(self, x, t, **kwargs):
        return torch.zeros_like(x)


class _FakeCond:
    def __init__(self):
        self.cond_bundle = {
            "cross_attn_cond": torch.ones(1, 3, 4),
            "cross_attn_mask": torch.ones(1, 3),
            "cfg_scale": 1.0,
        }
        self.latent_frames = T
        self.audio_sample_size = N44


class _FakeCodec:
    """Deterministic ramp 'decode' so resample/crop math is checkable."""

    def __init__(self):
        self.decodes = 0

    def decode_full(self, latent_bct: torch.Tensor) -> torch.Tensor:
        self.decodes += 1
        assert latent_bct.shape == (1, C, T), latent_bct.shape
        ramp = torch.linspace(-0.5, 0.5, N44)
        return torch.stack([ramp, -ramp])  # [2, N44]


def _schedule_builder_factory(steps: int):
    def _build(denoise: float) -> torch.Tensor:
        return torch.linspace(float(denoise), 0.0, steps + 1)
    return _build


def _backend(**kw):
    steps = kw.pop("steps", 3)
    adapter = SA3Adapter(
        _ZeroDit(),
        schedule_builder=_schedule_builder_factory(steps),
        device="cpu",
        dtype=torch.float32,
    )
    return SA3Backend(
        adapter=adapter,
        codec=_FakeCodec(),
        cond=kw.pop("cond", _FakeCond()),
        schedule_builder_factory=_schedule_builder_factory,
        knob_state=KnobState(sa3_knob_specs()),
        steps=steps,
        depth=2,
        vae_window_s=0.1,
        **kw,
    )


def _knobs(b, **over):
    """Registry defaults with steps_override pinned to the ctor steps,
    so produce() doesn't swap in a fresh pipeline (which would discard
    the submit wrapper / shared curves the wiring tests inspect)."""
    return {**b.read_knobs(), "steps_override": b._steps, **over}


def _capture_submits(b):
    """Wrap the live pipeline's submit so tests can inspect the exact
    SlotRequest the backend builds (knob wiring lands there)."""
    submitted: list = []
    orig = b.pipeline.submit

    def _wrapped(req):
        submitted.append(req)
        return orig(req)

    b.pipeline.submit = _wrapped
    return submitted


def test_contract_surface():
    b = _backend()

    caps = b.capabilities()
    assert caps.refines_audio is True
    for field in ("swap", "timbre", "structure", "lora", "stems",
                  "loop_band", "depth", "curves", "notes_conditioning"):
        assert getattr(caps, field) is False, field

    g = b.geometry()
    assert g.sample_rate == DELIVERY_SAMPLE_RATE  # delivered rate, v1
    assert g.channels == 2
    assert abs(g.chunk_rate_hz - 44100.0 / 4096.0) < 1e-9
    assert abs(g.duration_s - N44 / SA3_SAMPLE_RATE) < 1e-9

    names = [s.name for s in b.knob_specs()]
    assert names == [
        "sa3_denoise", "sa3_shift", "x0_target",
        "feedback", "feedback_depth", "seed", "steps_override",
    ]


def test_knob_defaults_flow_through_read_knobs():
    b = _backend()
    raw = b.read_knobs()
    assert raw["sa3_denoise"] == 1.0
    assert raw["steps_override"] == 8.0  # registry default; ctor steps differ
    assert raw["sa3_shift"] == 1.0      # stock checkpoint schedule
    assert raw["x0_target"] == 0.0
    assert raw["feedback"] == 0.0
    assert raw["feedback_depth"] == 1.0


def test_produce_emits_and_renders_windows():
    b = _backend()
    knobs = {"sa3_denoise": 0.7, "seed": 42, "steps_override": 3}

    fresh = 0
    for _ in range(10):
        if b.produce(knobs, CTX, "generate"):
            fresh += 1
    assert fresh >= 1
    assert b.has_renderable_state()

    chunk = b.render_window(0.5)
    n = round(0.1 * DELIVERY_SAMPLE_RATE)
    assert chunk.pcm.shape == (n, 2)
    assert chunk.pcm.dtype == np.float32
    assert chunk.start_sample == round(0.5 * DELIVERY_SAMPLE_RATE)

    # Same latent -> same cached render, bit-stable across gap-fills.
    again = b.render_window(0.5)
    assert np.array_equal(chunk.pcm, again.pcm)
    assert b.codec.decodes == 1  # one full decode per fresh latent

    full = b.render_full()
    expect_len = round(N44 * DELIVERY_SAMPLE_RATE / SA3_SAMPLE_RATE)
    assert abs(full.pcm.shape[0] - expect_len) <= 2  # resampler rounding
    assert full.start_sample == 0


def test_render_window_clamps_to_song_end():
    b = _backend()
    knobs = b.read_knobs()
    for _ in range(10):
        b.produce(knobs, CTX, "generate")
    dur = b.playable_duration_s()
    chunk = b.render_window(dur + 5.0)
    n = round(0.1 * DELIVERY_SAMPLE_RATE)
    assert chunk.pcm.shape[0] == n
    assert chunk.start_sample <= round(dur * DELIVERY_SAMPLE_RATE)


def test_reuse_and_skip_modes():
    b = _backend()
    knobs = b.read_knobs()
    for _ in range(10):
        b.produce(knobs, CTX, "generate")
    assert b.has_renderable_state()

    assert b.produce(knobs, CTX, "reuse") is True     # DiT-pause re-adopt
    assert b.produce(knobs, CTX, "skip") is False     # gap-fill tick


def test_steps_override_rebuilds_pipeline():
    b = _backend(steps=3)
    assert b.rebuild_imminent({"steps_override": 3}) is False
    assert b.rebuild_imminent({"steps_override": 5}) is True

    b.produce({"sa3_denoise": 1.0, "seed": 1, "steps_override": 5},
              CTX, "generate")
    assert b.pipeline.config.infer_steps == 5
    # The fresh pipeline's schedules match the new step count: drain on.
    fresh = 0
    for _ in range(12):
        if b.produce({"sa3_denoise": 1.0, "seed": 1, "steps_override": 5},
                     CTX, "generate"):
            fresh += 1
    assert fresh >= 1


def test_sa3_shift_applies_and_invalidates_schedule_cache():
    b = _backend()
    knobs = {**_knobs(b), "sa3_denoise": 0.7}
    b.produce(knobs, CTX, "generate")
    assert 0.7 in b.pipeline._schedule_cache
    stock = b.pipeline._schedule_cache[0.7].clone()

    b.produce({**knobs, "sa3_shift": 2.0}, CTX, "generate")
    assert b.adapter.shift_alpha == 2.0
    warped = b.pipeline._schedule_cache[0.7]
    # Same cache key, different schedule — exactly the staleness the
    # invalidation exists to prevent.
    assert not torch.allclose(warped, stock)
    assert warped[0].item() == stock[0].item()   # sigma_max pinned
    assert warped[-1].item() == 0.0

    # No-op change (same alpha) must not clear the cache every tick.
    cache_id = id(b.pipeline._schedule_cache)
    b.produce({**knobs, "sa3_shift": 2.0}, CTX, "generate")
    assert 0.7 in b.pipeline._schedule_cache
    assert id(b.pipeline._schedule_cache) == cache_id


def test_x0_target_rides_shared_curve_and_request():
    src = torch.randn(1, C, T)
    b = _backend(source_latent_bct=src)
    submitted = _capture_submits(b)

    b.produce({**_knobs(b), "x0_target": 0.5}, CTX, "generate")

    shared = b.pipeline._shared_curves["x0_target_strength"]
    assert float(shared.flatten()[0]) == 0.5
    req = submitted[-1]
    assert req.x0_target_strength == 0.5
    # The morph target is the clean anchor in engine layout.
    assert torch.equal(req.x0_target, src.movedim(1, 2))

    # Dropping the knob back to 0 releases in-flight slots too.
    b.produce(_knobs(b), CTX, "generate")
    shared = b.pipeline._shared_curves["x0_target_strength"]
    assert float(shared.flatten()[0]) == 0.0


def test_feedback_taps_latent_history():
    src = torch.randn(1, C, T)
    b = _backend(source_latent_bct=src)
    knobs = _knobs(b)
    for _ in range(10):
        b.produce(knobs, CTX, "generate")
    assert len(b._latent_history) >= 1
    tap = b._latent_history[0].clone()

    submitted = _capture_submits(b)
    b.produce({**knobs, "feedback": 1.0, "feedback_depth": 1},
              CTX, "generate")
    # feedback=1.0 fully replaces the anchor with the tapped latent
    # (slerp endpoint); the request's init source must be the tap, and
    # the morph target must stay the clean anchor.
    req = submitted[-1]
    assert torch.allclose(req.source_latents, tap, atol=1e-6)
    assert torch.equal(req.x0_target, src.movedim(1, 2))

    # Depth beyond available history falls back to the oldest tap
    # rather than disabling feedback (the operator said "feedback").
    oldest = b._latent_history[-1].clone()
    b.produce({**knobs, "feedback": 1.0, "feedback_depth": 8},
              CTX, "generate")
    assert torch.allclose(submitted[-1].source_latents, oldest, atol=1e-6)


def test_feedback_without_history_or_source_is_inert():
    b = _backend()  # no source anchor
    submitted = _capture_submits(b)
    b.produce({**_knobs(b), "feedback": 1.0}, CTX, "generate")
    assert submitted[-1].source_latents is None


def _cond_ab():
    """A/B captures with orthogonal unit token embeddings (so the
    slerp midpoint is checkable in closed form) and disjoint mask
    tails (so the union is observable)."""
    a, b = _FakeCond(), _FakeCond()
    a.cond_bundle["cross_attn_cond"] = (
        torch.tensor([1.0, 0.0, 0.0, 0.0]).repeat(1, 3, 1)
    )
    a.cond_bundle["cross_attn_mask"] = torch.tensor([[1.0, 1.0, 0.0]])
    b.cond_bundle["cross_attn_cond"] = (
        torch.tensor([0.0, 1.0, 0.0, 0.0]).repeat(1, 3, 1)
    )
    b.cond_bundle["cross_attn_mask"] = torch.tensor([[1.0, 0.0, 1.0]])
    return a, b


def test_prompt_blend_endpoints_midpoint_and_submit():
    ca, cb = _cond_ab()
    b = _backend(cond=ca, cond_b=cb)

    # Endpoints return the captures verbatim (identity keeps the TRT
    # wrapper's id()-keyed staging warm).
    assert b._active_bundle is ca.cond_bundle
    b.handle_set_prompt_blend(1.0)
    assert b._active_bundle is cb.cond_bundle

    b.handle_set_prompt_blend(0.5)
    mid = b._active_bundle
    assert mid is not ca.cond_bundle and mid is not cb.cond_bundle
    cc = mid["cross_attn_cond"]
    # Slerp, not lerp: unit vectors stay unit at the midpoint (a linear
    # blend of orthogonal units collapses to norm ~0.707).
    assert torch.allclose(cc.norm(dim=-1), torch.ones(1, 3), atol=1e-5)
    r = 2.0 ** -0.5
    assert torch.allclose(
        cc, torch.tensor([r, r, 0.0, 0.0]).repeat(1, 3, 1), atol=1e-5,
    )
    # Token masks union so either side's tokens stay attended.
    assert torch.equal(mid["cross_attn_mask"], torch.tensor([[1.0, 1.0, 1.0]]))
    # The captures themselves are never mutated.
    assert torch.equal(
        ca.cond_bundle["cross_attn_cond"],
        torch.tensor([1.0, 0.0, 0.0, 0.0]).repeat(1, 3, 1),
    )

    # The next submit carries the blended bundle.
    submitted = _capture_submits(b)
    b.produce(_knobs(b), CTX, "generate")
    assert submitted[-1].aux_cond is mid


def test_prompt_blend_without_b_is_a_noop():
    b = _backend()  # cond_b defaults to cond: blend A against A
    b.handle_set_prompt_blend(0.7)
    assert b._active_bundle is b._cond.cond_bundle


def test_set_prompt_recaptures_b_and_invalidates_schedules():
    captures = []

    def rebuilder(tags, steps):
        captures.append(tags)
        cond = _FakeCond()
        return cond, lambda s: _schedule_builder_factory(s)

    b = _backend(prompt_rebuilder=rebuilder)
    b.pipeline._get_schedule(0.9)
    assert b.pipeline._schedule_cache

    b.handle_set_prompt("tags a", tags_b="tags b")
    assert captures == ["tags a", "tags b"]
    # The builder swap changes what the same denoise key would build —
    # stale schedules must not survive the prompt change.
    assert not b.pipeline._schedule_cache
    assert b._cond_b is not b._cond
    assert b._active_bundle is b._cond.cond_bundle  # blend still 0

    # An absent B resets B to A (the ACE set_prompt convention).
    b.handle_set_prompt("tags solo")
    assert b._cond_b is b._cond
