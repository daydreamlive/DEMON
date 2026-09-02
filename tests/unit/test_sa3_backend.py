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
    delivered_samples,
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
        self.decode_seeds: list = []

    def decode_full(
        self, latent_bct: torch.Tensor, *, decode_seed: int | None = None,
    ) -> torch.Tensor:
        self.decodes += 1
        self.decode_seeds.append(decode_seed)
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
    # loop_band is armed for SA3 (windowed renderer pre-fills the seam at
    # the loop point — see SA3Backend.capabilities).
    assert caps.loop_band is True
    # swap is backend-owned: handle_swap_source re-anchors in place at
    # fixed geometry (the session dispatches there, not the ACE body).
    assert caps.swap is True
    for field in ("timbre", "structure", "lora", "stems",
                  "depth", "curves", "notes_conditioning"):
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


def test_playable_duration_excludes_duration_padding():
    # prepare_sa3_conditioning pads the render window
    # (cond.audio_sample_size) past seconds_total by duration_padding_sec
    # of outro headroom, which the model fades to silence — upstream
    # generate() trims it off (truncate_output_to_duration). The playable
    # surface must stop at the conditioned duration, or every loop ends
    # in the padded fade/silence (the "silence from 9:30 to midnight"
    # DreamSampler report).
    window_s = N44 / SA3_SAMPLE_RATE
    song_s = window_s / 2
    b = _backend(playable_duration_s=song_s)
    assert abs(b.playable_duration_s() - song_s) < 1e-9
    assert abs(b.geometry().duration_s - song_s) < 1e-9

    # A caller value past the window clamps back inside it.
    b2 = _backend(playable_duration_s=window_s + 10.0)
    assert abs(b2.playable_duration_s() - window_s) < 1e-9

    # Direct construction without the arg keeps the full window
    # (pre-existing test-backend behavior).
    b3 = _backend()
    assert abs(b3.playable_duration_s() - window_s) < 1e-9


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


def test_handle_swap_source_reanchors_and_clears_history():
    old_src = torch.randn(1, C, T)
    new_latent_bct = torch.randn(1, C, T)
    encodes = []

    def encoder(waveform, sample_rate, sample_size):
        encodes.append((int(waveform.shape[-1]), int(sample_rate),
                        int(sample_size)))
        return new_latent_bct

    b = _backend(source_latent_bct=old_src, source_encoder=encoder)
    # Build up feedback history on the OLD anchor.
    knobs = _knobs(b)
    for _ in range(10):
        b.produce(knobs, CTX, "generate")
    assert len(b._latent_history) >= 1

    b.handle_swap_source(torch.zeros(2, 48000), 48000)
    # Encoded at the session's FIXED cond geometry, not the upload's.
    assert encodes == [(48000, 48000, N44)]
    # The anchor is the new latent in engine layout [1, T, C]...
    assert torch.equal(b._source_latent_btc, new_latent_bct.movedim(1, 2))
    # ...and the feedback ring is dropped: its taps cover the OLD
    # source and must not smear into the new anchor.
    assert len(b._latent_history) == 0

    # The next submit inits from (and morph-targets) the new anchor.
    submitted = _capture_submits(b)
    b.produce({**knobs, "sa3_denoise": 0.5}, CTX, "generate")
    assert torch.equal(
        submitted[-1].source_latents, new_latent_bct.movedim(1, 2),
    )
    assert torch.equal(
        submitted[-1].x0_target, new_latent_bct.movedim(1, 2),
    )


def test_handle_swap_source_discards_covers_of_the_old_source():
    """Slots submitted before the swap were initialised from the old
    anchor, so what emerges from them is the previous source. Neither
    those latents nor the cached one may be rendered into the new
    buffer; the first adopted latent after a swap is a new-anchor one."""
    old_src = torch.randn(1, C, T)
    new_latent_bct = torch.randn(1, C, T)
    b = _backend(
        source_latent_bct=old_src,
        source_encoder=lambda waveform, sample_rate, sample_size: new_latent_bct,
    )
    knobs = _knobs(b)
    for _ in range(10):
        b.produce(knobs, CTX, "generate")
    assert b.has_renderable_state()

    b.handle_swap_source(torch.zeros(2, 48000), 48000)
    # Nothing to render until a new-anchor slot completes.
    assert not b.has_renderable_state()
    assert b.render_window(0.0) is None

    fresh = []
    for _ in range(10):
        fresh.append(b.produce(knobs, CTX, "generate"))
        if fresh[-1]:
            break
    # The in-flight old-anchor slots emerged first and were discarded...
    assert fresh[:-1] and not any(fresh[:-1])
    assert fresh[-1] is True
    # ...and the adopted latent came from a request built on the new anchor.
    assert b._emerged_request.x0_target is b._source_latent_btc
    assert b.has_renderable_state()


def test_handle_swap_source_without_encoder_fails_loudly():
    b = _backend()  # direct construction: no source_encoder
    try:
        b.handle_swap_source(torch.zeros(2, 48000), 48000)
    except RuntimeError as exc:
        assert "source_encoder" in str(exc)
    else:
        raise AssertionError("expected RuntimeError without source_encoder")


class _FakeCondSized:
    """A _FakeCond at an arbitrary latent-frame count, for resize tests."""

    def __init__(self, frames: int):
        self.cond_bundle = {
            "cross_attn_cond": torch.ones(1, 3, 4),
            "cross_attn_mask": torch.ones(1, 3),
            "cfg_scale": 1.0,
        }
        self.latent_frames = frames
        self.audio_sample_size = frames * 4096


def _resize_backend(new_frames: int, **kw):
    """Backend wired with a fake resizer that lands on ``new_frames``
    (the create-time geometry is T frames / N44 samples) plus an
    encoder that records the sample_size it was asked for."""
    encodes: list = []
    resizes: list = []

    def encoder(waveform, sample_rate, sample_size):
        encodes.append(int(sample_size))
        frames = int(sample_size) // 4096
        return torch.randn(1, C, frames)

    def resizer(new_duration_s, tags_a, tags_b, steps):
        resizes.append((float(new_duration_s), tags_a, tags_b, int(steps)))
        cond = _FakeCondSized(new_frames)
        return (
            cond.audio_sample_size / SA3_SAMPLE_RATE, cond, None,
            _ZeroDit(), _schedule_builder_factory,
        )

    b = _backend(
        source_latent_bct=torch.randn(1, C, T),
        source_encoder=encoder,
        resizer=resizer,
        prompt_tags="warm tags",
        **kw,
    )
    return b, encodes, resizes


def test_handle_swap_source_resize_rederives_geometry():
    """duration_s re-derives the render geometry: new cond captures at
    the live prompt, a rebuilt pipeline, the anchor encoded at the NEW
    sample size, and the hook returns the new delivery-rate playback
    length for the session to adopt."""
    new_frames = 2 * T
    b, encodes, resizes = _resize_backend(new_frames)
    assert b.capabilities().swap_resize is True
    old_cond = b._cond
    old_pipeline = b.pipeline
    old_playable = b.playable_duration_s()

    requested_s = new_frames * 4096 / SA3_SAMPLE_RATE
    got = b.handle_swap_source(
        torch.zeros(2, 96000), 48000, duration_s=requested_s,
    )

    # The resizer saw the request against the live prompt pair.
    assert resizes == [(requested_s, "warm tags", None, b._steps)]
    # The anchor was encoded at the NEW window, not the old one.
    assert encodes == [new_frames * 4096]
    assert b._source_latent_btc.shape == (1, new_frames, C)
    # Geometry followed: cond, playable duration, pipeline all new.
    assert b._cond is not old_cond
    assert int(b._cond.latent_frames) == new_frames
    assert b.pipeline is not old_pipeline
    assert b.playable_duration_s() > old_playable
    assert abs(b.geometry().duration_s - requested_s) < 1e-6
    # The session resizes its buffer from the returned length.
    expected_44k = min(
        int(round(b.playable_duration_s() * SA3_SAMPLE_RATE)),
        new_frames * 4096,
    )
    assert got == delivered_samples(expected_44k)
    # History/caches of the old geometry are gone.
    assert len(b._latent_history) == 0
    assert b._last_result_latent is None
    assert not b.has_renderable_state()


def test_handle_swap_source_resize_noop_on_same_geometry():
    """A duration_s that clamps back onto the current window (same
    latent_frames) must not churn the session: same cond, same
    pipeline, plain re-anchor, None returned (buffer length holds)."""
    b, encodes, resizes = _resize_backend(T)
    old_cond = b._cond
    old_pipeline = b.pipeline

    got = b.handle_swap_source(
        torch.zeros(2, 48000), 48000, duration_s=N44 / SA3_SAMPLE_RATE,
    )

    assert len(resizes) == 1          # the captures ran...
    assert got is None                # ...but nothing was adopted
    assert b._cond is old_cond
    assert b.pipeline is old_pipeline
    assert encodes == [N44]           # legacy re-anchor at the old window


def test_handle_swap_source_resize_without_resizer_fails_loudly():
    b = _backend(
        source_latent_bct=torch.randn(1, C, T),
        source_encoder=lambda waveform, sample_rate, sample_size: (
            torch.randn(1, C, T)
        ),
    )  # direct construction: encoder but no resizer
    assert b.capabilities().swap_resize is False
    try:
        b.handle_swap_source(torch.zeros(2, 48000), 48000, duration_s=30.0)
    except RuntimeError as exc:
        assert "resizer" in str(exc)
    else:
        raise AssertionError("expected RuntimeError without a resizer")


def test_plain_swap_with_resizer_keeps_geometry():
    """No duration_s = the legacy fixed-geometry swap, byte-identical:
    old clients keep exactly the behavior they shipped against."""
    b, encodes, resizes = _resize_backend(2 * T)
    old_cond = b._cond
    old_pipeline = b.pipeline

    got = b.handle_swap_source(torch.zeros(2, 48000), 48000)

    assert got is None
    assert resizes == []
    assert encodes == [N44]
    assert b._cond is old_cond
    assert b.pipeline is old_pipeline


def test_decode_seed_follows_the_emerged_request_seed():
    """The render decode is pinned to the seed of the request that
    produced the decoded latent, so identical session inputs replay to
    bit-identical audio (the SAME decoder draws noise at inference)."""
    b = _backend()

    for _ in range(10):
        b.produce(_knobs(b, seed=42), CTX, "generate")
    b.render_window(0.0)
    assert b.codec.decode_seeds == [42]

    # A seed change pins subsequent fresh latents to the new seed.
    for _ in range(10):
        b.produce(_knobs(b, seed=99), CTX, "generate")
    b.render_window(0.0)
    assert b.codec.decode_seeds[-1] == 99

    # DiT-pause "reuse" keeps the pairing: the re-adopted latent is the
    # one seed 99 produced, and its render decode stays seed-99-pinned.
    b._rendered_for = None  # drop the render cache to force a decode
    b.produce(_knobs(b, seed=1234), CTX, "reuse")
    b.render_window(0.0)
    assert b.codec.decode_seeds[-1] == 99


# ---------------------------------------------------------------------------
# LoRA surface (notes/SA3_LORA_PLAN.md phase 1) — fake manager, real wiring
# ---------------------------------------------------------------------------

from acestep.streaming.sa3_backend import sa3_lora_compatible


class _FakeLoraDesc:
    def __init__(self, lora_id, state="registered", strength=0.0):
        self.id = lora_id
        self.name = lora_id
        self.path = f"/nonexistent/{lora_id}.safetensors"
        self.state = state
        self.strength = strength
        self.materialized_bytes = 0


class _FakeLoraManager:
    def __init__(self, ids=(), touches=()):
        self._descs = {i: _FakeLoraDesc(i) for i in ids}
        self._touches = set(touches)
        self.calls: list = []
        self.closed = False

    def list_loras(self):
        return list(self._descs.values())

    @property
    def has_active_loras(self):
        return any(d.state == "enabled" for d in self._descs.values())

    def enable_lora(self, lora_id, strength=None):
        self.calls.append(("enable", lora_id, strength))
        d = self._descs.setdefault(lora_id, _FakeLoraDesc(lora_id))
        d.state = "enabled"
        if strength is not None:
            d.strength = float(strength)

    def disable_lora(self, lora_id):
        self.calls.append(("disable", lora_id))
        if lora_id in self._descs:
            self._descs[lora_id].state = "registered"

    def set_lora_strength(self, lora_id, strength):
        self.calls.append(("strength", lora_id, strength))
        self._descs[lora_id].strength = float(strength)

    def touches_conditioner(self, lora_id):
        return lora_id in self._touches

    def close(self):
        self.closed = True


def test_lora_capability_bit_requires_manager_and_toggle():
    assert _backend().capabilities().lora is False
    assert _backend(
        lora_manager=_FakeLoraManager(), use_lora=False,
    ).capabilities().lora is False
    assert _backend(use_lora=True).capabilities().lora is False  # no manager
    assert _backend(
        lora_manager=_FakeLoraManager(), use_lora=True,
    ).capabilities().lora is True


def test_lora_knob_specs_expand_per_id():
    b = _backend(lora_manager=_FakeLoraManager(), use_lora=True)
    names = [s.name for s in b.knob_specs(["ambient", "phonk"])]
    assert "lora_str_ambient" in names
    assert "lora_str_phonk" in names
    # Base manifest unchanged and first (display order).
    assert names[0] == "sa3_denoise"


def test_sa3_lora_compatible_family_and_lineage():
    assert sa3_lora_compatible({"lora_family": "ace"}, "medium") is False
    assert sa3_lora_compatible(
        {"lora_family": "sa3", "base_model": "medium-base"}, "medium",
    ) is True
    assert sa3_lora_compatible(
        {"lora_family": "sa3", "base_model": "small-music-base"}, "medium",
    ) is False
    # Unknown on either side stays permissive.
    assert sa3_lora_compatible({}, "medium") is True
    assert sa3_lora_compatible(
        {"lora_family": "sa3", "base_model": "mystery"}, "medium",
    ) is True
    assert sa3_lora_compatible(
        {"lora_family": "sa3", "base_model": "medium-base"}, "",
    ) is True


def test_prepare_tick_drives_strength_from_knob():
    mgr = _FakeLoraManager(ids=["x"])
    mgr.enable_lora("x", strength=1.0)
    b = _backend(lora_manager=mgr, use_lora=True)

    b.produce(_knobs(b, **{"lora_str_x": 0.5}), CTX, "skip")
    assert ("strength", "x", 0.5) in mgr.calls

    # Within the 0.02 slider-delta gate: no engine call.
    n_calls = len(mgr.calls)
    b.produce(_knobs(b, **{"lora_str_x": 0.51}), CTX, "skip")
    assert len(mgr.calls) == n_calls

    # Non-enabled entries are ignored even if a stray knob rides in.
    mgr.disable_lora("x")
    n_calls = len(mgr.calls)
    b.produce(_knobs(b, **{"lora_str_x": 1.7}), CTX, "skip")
    assert not any(c[0] == "strength" for c in mgr.calls[n_calls:])


def test_dit_swaps_to_eager_while_lora_active():
    mgr = _FakeLoraManager(ids=["x"])
    b = _backend(lora_manager=mgr, use_lora=True, eager_dit=_ZeroDit())
    accel = b._dit_accel
    eager = b._dit_eager
    assert b.adapter.dit is accel

    b.enable_lora("x", strength=1.0)
    assert b.adapter.dit is eager

    b.disable_lora("x")
    assert b.adapter.dit is accel


def test_eager_session_dit_swap_is_noop():
    mgr = _FakeLoraManager(ids=["x"])
    b = _backend(lora_manager=mgr, use_lora=True)  # no separate eager dit
    dit = b.adapter.dit
    b.enable_lora("x", strength=1.0)
    assert b.adapter.dit is dit
    b.disable_lora("x")
    assert b.adapter.dit is dit


def test_conditioner_lora_triggers_cond_rebuild(monkeypatch):
    mgr = _FakeLoraManager(ids=["c"], touches=["c"])
    b = _backend(lora_manager=mgr, use_lora=True)
    rebuilds: list = []
    monkeypatch.setattr(
        b, "_rebuild_conditioning_after_lora",
        lambda op, lid: rebuilds.append((op, lid)),
    )

    b.enable_lora("c", strength=1.0)
    b.set_lora_strength("c", 0.5)
    b.disable_lora("c")

    assert rebuilds == [
        ("enable_lora", "c"),
        ("set_lora_strength", "c"),
        ("disable_lora", "c"),
    ]


def test_dit_only_lora_skips_cond_rebuild(monkeypatch):
    mgr = _FakeLoraManager(ids=["d"])
    b = _backend(lora_manager=mgr, use_lora=True)
    rebuilds: list = []
    monkeypatch.setattr(
        b, "_rebuild_conditioning_after_lora",
        lambda op, lid: rebuilds.append((op, lid)),
    )
    b.enable_lora("d", strength=1.0)
    b.disable_lora("d")
    assert rebuilds == []


def test_cond_rebuild_goes_through_prompt_swap_path(monkeypatch):
    import types

    mgr = _FakeLoraManager(ids=["c"], touches=["c"])
    b = _backend(lora_manager=mgr, use_lora=True)
    b._prompt_rebuilder = object()  # present: rebuild not skipped
    b.state = types.SimpleNamespace(
        prompt_text="warm tags", prompt_text_b="other tags",
    )
    swaps: list = []
    monkeypatch.setattr(
        b, "handle_set_prompt",
        lambda tags, tags_b=None: swaps.append((tags, tags_b)),
    )

    b._rebuild_conditioning_after_lora("enable_lora", "c")
    assert swaps == [("warm tags", "other tags")]


def test_lora_pending_gates_has_pending_refit():
    import threading
    import types

    mgr = _FakeLoraManager(ids=["x"])
    b = _backend(lora_manager=mgr, use_lora=True)
    b.state = types.SimpleNamespace(
        _lock=threading.Lock(), pending_enable=[], pending_disable=[],
        interp_feedback="slerp",
    )
    assert b.has_pending_refit() is False
    b.state.pending_enable.append(("x", 1.0))
    assert b.has_pending_refit() is True
    b.state.pending_enable.clear()
    b.state.pending_disable.append("x")
    assert b.has_pending_refit() is True


def test_close_tears_down_lora_manager():
    mgr = _FakeLoraManager(ids=["x"])
    b = _backend(lora_manager=mgr, use_lora=True)
    b.close()
    assert mgr.closed is True


# ---------------------------------------------------------------------------
# Phase-2 refit-mirror routing (fake mirror; the real one is GPU-gated)
# ---------------------------------------------------------------------------


class _FakeMirror:
    def __init__(self):
        self.syncs: list = []

    def sync(self, *, reason=""):
        self.syncs.append(reason)
        return 1


def test_mirror_mode_never_swaps_dit_and_syncs_on_enable_disable():
    mgr = _FakeLoraManager(ids=["x"])
    mirror = _FakeMirror()
    b = _backend(
        lora_manager=mgr, use_lora=True, eager_dit=_ZeroDit(),
        refit_mirror=mirror,
    )
    accel = b._dit_accel
    assert accel is not b._dit_eager  # distinct objects: swap WOULD fire

    b.enable_lora("x", strength=1.0)
    assert b.adapter.dit is accel          # no eager swap in mirror mode
    assert mirror.syncs == ["enable_lora"]

    b.disable_lora("x")
    assert b.adapter.dit is accel
    assert mirror.syncs == ["enable_lora", "disable_lora"]


def test_mirror_strength_routes_through_pending_stash():
    mgr = _FakeLoraManager(ids=["x"])
    mgr.enable_lora("x", strength=1.0)
    mirror = _FakeMirror()
    b = _backend(lora_manager=mgr, use_lora=True, refit_mirror=mirror)
    mirror.syncs.clear()

    knobs = _knobs(b, **{"lora_str_x": 0.4})
    # The announcement point: rebuild_imminent stashes and reports.
    assert b.rebuild_imminent(knobs) is True
    assert b._pending_lora_strengths == {"x": 0.4}
    assert b.has_pending_refit() is True

    # The same tick's produce applies the stash: one manager write, one
    # batched mirror sync, stash drained.
    b.produce(knobs, CTX, "skip")
    assert ("strength", "x", 0.4) in mgr.calls
    assert mirror.syncs == ["strength"]
    assert b._pending_lora_strengths == {}
    assert b.has_pending_refit() is False

    # Within the 0.02 gate: no announcement, no stash.
    assert b.rebuild_imminent(_knobs(b, **{"lora_str_x": 0.41})) is False
    assert b._pending_lora_strengths == {}


def test_eager_mode_strength_is_not_stashed():
    mgr = _FakeLoraManager(ids=["x"])
    mgr.enable_lora("x", strength=1.0)
    b = _backend(lora_manager=mgr, use_lora=True)  # no mirror

    knobs = _knobs(b, **{"lora_str_x": 0.4})
    assert b.rebuild_imminent(knobs) is False  # buffer write, no stall
    b.produce(knobs, CTX, "skip")
    assert ("strength", "x", 0.4) in mgr.calls  # applied inline


def test_close_drops_mirror_and_pending():
    mgr = _FakeLoraManager(ids=["x"])
    mirror = _FakeMirror()
    b = _backend(lora_manager=mgr, use_lora=True, refit_mirror=mirror)
    b._pending_lora_strengths["x"] = 0.5
    b.close()
    assert b._refit_mirror is None
    assert b._pending_lora_strengths == {}
    assert mgr.closed is True
