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
        cond=_FakeCond(),
        schedule_builder_factory=_schedule_builder_factory,
        knob_state=KnobState(sa3_knob_specs()),
        steps=steps,
        depth=2,
        vae_window_s=0.1,
        **kw,
    )


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
    assert names == ["sa3_denoise", "seed", "steps_override"]


def test_knob_defaults_flow_through_read_knobs():
    b = _backend()
    raw = b.read_knobs()
    assert raw["sa3_denoise"] == 1.0
    assert raw["steps_override"] == 8.0  # registry default; ctor steps differ


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
