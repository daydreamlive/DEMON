"""Regression fixtures for ``session.generate()`` output.

Captures a byte-level baseline of ``generate()`` output for every feature
configuration that production workflows exercise (mirroring
``workflows/covers/*.py``). Purpose: guard against regressions during the
diffusion-primitive unification refactor.

Design — why we cache the inputs, not just outputs
--------------------------------------------------
``VAEEncodeAudio`` samples from the VAE's posterior distribution
(``latent = mean + std * randn_like``) and ``TextEncode`` depends on the
sampled ``refer_latent``. That makes ``prepared_source.latent``,
``prepared_source.context_latent``, and any conditioning built with
``refer_latent=`` non-deterministic **across processes** (they are fine
within one process because pytest session-scoped fixtures share state).
Baseline fixtures must compare *outputs of generate() given identical
inputs*, so we serialize the stochastic upstream outputs once to
``stable_inputs.pt`` and reuse them on every subsequent run. Curves,
masks, and seeds are deterministic — they are recomputed inline per
test.

Usage::

    # First run auto-captures stable_inputs + baseline outputs, skips assertions:
    uv run pytest tests/test_generate_parity.py -v

    # Explicit (re)capture — overwrites all fixtures:
    CAPTURE_FIXTURES=1 uv run pytest tests/test_generate_parity.py -v

    # Regression mode (assert against baselines):
    uv run pytest tests/test_generate_parity.py -v

Fixtures live under ``tests/fixtures/generate_parity/{backend}/``.
``{backend}`` is ``trt`` when TRT engines are loaded, else ``pt``;
fixtures are not portable across backends.

Note on CFG / ``guidance_curve``:
    The ``cover_guidance_curve`` test exercises CFG with negative
    conditioning. CFG was restored in Phase 2 via APG-based per-slot
    momentum blending inside ``StreamPipeline._tick_complex_pt``; the
    test asserts byte-equivalence against the pre-Phase-1 baseline.
"""

import os
from pathlib import Path

import pytest
import soundfile as sf
import torch

from acestep.constants import TASK_INSTRUCTIONS
from acestep.nodes import Mask
from acestep.nodes.cond_nodes import ConditioningCombine, ConditioningZeroOut
from acestep.nodes.curve_nodes import CurveRamp, CurveWave
from acestep.nodes.mask_nodes import TemporalMask, SetLatentNoiseMask
from acestep.nodes.types import (
    Audio,
    Conditioning,
    ConditioningEntry,
    Latent,
)


FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "generate_parity"
CAPTURE = os.environ.get("CAPTURE_FIXTURES", "0") == "1"
TOL = 1e-4

# Keep knobs in sync with workflows/covers/*.py for parity with production.
SEED = 1528
STEPS = 8
SHIFT = 3.0

# Audio clip used for all tests (same file existing tests use).
SAMPLE_RATE = 48000
TEST_DURATION = 30.0  # shorter than workflows' 60s — faster tests, same code paths.
from acestep.fixtures import audio_fixture
TEST_AUDIO = audio_fixture("inside_confusion_loop_60s_gsm.wav")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _backend_tag(session) -> str:
    """``trt`` if the DiT decoder TRT engine is loaded, else ``pt``."""
    engine_obj = getattr(session.handler, "_diffusion_engine", None)
    if engine_obj is not None and getattr(engine_obj, "_trt_engine", None) is not None:
        return "trt"
    return "pt"


def _backend_dir(session) -> Path:
    return FIXTURE_ROOT / _backend_tag(session)


def _load_source_audio() -> Audio:
    """Load the fixture WAV into an Audio payload (deterministic from disk)."""
    data, sr = sf.read(str(TEST_AUDIO), dtype="float32")
    waveform = torch.from_numpy(data.T if data.ndim > 1 else data.reshape(1, -1))
    if sr != SAMPLE_RATE:
        import torchaudio
        waveform = torchaudio.transforms.Resample(sr, SAMPLE_RATE)(waveform)
    waveform = waveform[:2, :int(TEST_DURATION * SAMPLE_RATE)]
    pool = 1920 * 5
    rem = waveform.shape[-1] % pool
    if rem:
        waveform = waveform[:, :waveform.shape[-1] - rem]
    return Audio(waveform=waveform, sample_rate=SAMPLE_RATE)


def _serialize_conditioning(cond: Conditioning) -> list:
    """Capture each entry's tensors; other fields recreated on load."""
    return [
        {
            "encoder_hidden_states": e.encoder_hidden_states.detach().cpu().contiguous(),
            "encoder_attention_mask": e.encoder_attention_mask.detach().cpu().contiguous(),
        }
        for e in cond.to_entries()
    ]


def _deserialize_conditioning(entries_blob: list, device, dtype) -> Conditioning:
    return Conditioning(entries=[
        ConditioningEntry(
            encoder_hidden_states=e["encoder_hidden_states"].to(device=device, dtype=dtype),
            encoder_attention_mask=e["encoder_attention_mask"].to(device=device),
        )
        for e in entries_blob
    ])


def _capture_or_assert_output(session, name: str, actual: torch.Tensor) -> None:
    fixture_path = _backend_dir(session) / f"{name}.pt"

    if CAPTURE or not fixture_path.exists():
        fixture_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(actual.detach().cpu().contiguous(), fixture_path)
        if not CAPTURE:
            pytest.skip(
                f"Captured new baseline: {fixture_path.relative_to(FIXTURE_ROOT.parent.parent)}. "
                f"Re-run to assert against it."
            )
        return

    expected = torch.load(fixture_path, map_location="cpu")
    actual_cpu = actual.detach().cpu()

    assert actual_cpu.shape == expected.shape, (
        f"{name}: shape mismatch actual={list(actual_cpu.shape)} "
        f"expected={list(expected.shape)}"
    )
    diff = (actual_cpu.float() - expected.float()).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    assert torch.allclose(actual_cpu.float(), expected.float(), atol=TOL), (
        f"{name} diverged from baseline "
        f"({fixture_path.relative_to(FIXTURE_ROOT.parent.parent)}): "
        f"max abs diff {max_diff:.3e}, mean abs diff {mean_diff:.3e}, "
        f"tolerance atol={TOL:.0e}. "
        f"If intentional, re-capture with CAPTURE_FIXTURES=1."
    )


# ---------------------------------------------------------------------------
# The stable-inputs fixture — captures VAE / text-encode outputs once
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def stable_inputs(session):
    """Return a dict of process-stable inputs to ``generate()``.

    Cached under ``{backend}/stable_inputs.pt``. First call computes
    them (via VAE encode + text encode) and saves; subsequent calls
    load from disk so VAE-sampling stochasticity doesn't invalidate
    output baselines.

    Keys:
      - ``source_latent``: Latent
      - ``context_latent``: Latent
      - ``cover_cond_a``: Conditioning ("deathstep ...", cover instruction)
      - ``cover_cond_b``: Conditioning ("daft punk ...", cover instruction)
      - ``T``: int, frame count of source_latent
    """
    inputs_path = _backend_dir(session) / "stable_inputs.pt"
    device = session.handler.device
    dtype = session.handler.dtype

    if inputs_path.exists() and not CAPTURE:
        blob = torch.load(inputs_path, map_location="cpu")
    else:
        audio = _load_source_audio()
        prep = session.prepare_source(audio)
        T_frames = prep.latent.tensor.shape[1]
        duration = T_frames / 25.0

        cover_cond_a = session.encode_text(
            tags="deathstep death deaht deaht",
            instruction=TASK_INSTRUCTIONS["cover"],
            refer_latent=prep.latent,
            bpm=136, duration=duration, key="G# minor",
        )
        cover_cond_b = session.encode_text(
            tags="daft punk style electronic french house",
            instruction=TASK_INSTRUCTIONS["cover"],
            refer_latent=prep.latent,
            bpm=136, duration=duration, key="G# minor",
        )

        blob = {
            "source_latent": prep.latent.tensor.detach().cpu().contiguous(),
            "context_latent": prep.context_latent.tensor.detach().cpu().contiguous(),
            "cover_cond_a": _serialize_conditioning(cover_cond_a),
            "cover_cond_b": _serialize_conditioning(cover_cond_b),
        }
        inputs_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(blob, inputs_path)

    return {
        "source_latent": Latent(
            tensor=blob["source_latent"].to(device=device, dtype=dtype),
        ),
        "context_latent": Latent(
            tensor=blob["context_latent"].to(device=device, dtype=dtype),
        ),
        "cover_cond_a": _deserialize_conditioning(blob["cover_cond_a"], device, dtype),
        "cover_cond_b": _deserialize_conditioning(blob["cover_cond_b"], device, dtype),
        "T": blob["source_latent"].shape[1],
    }


# ---------------------------------------------------------------------------
# Parity tests — one per feature combination exercised by workflows/covers/*
# ---------------------------------------------------------------------------


class TestGenerateParity:
    """Byte-level regression baseline for every generate() feature path.

    Every test here captures or asserts against a .pt tensor under
    tests/fixtures/generate_parity/{backend}/. The tolerance is atol=1e-4
    (same as existing test_session.py deterministic-seed assertion).
    """

    def test_cover_denoise_50(self, session, stable_inputs):
        """Partial cover denoise at 0.5 — common production config."""
        result = session.generate(
            conditioning=stable_inputs["cover_cond_a"],
            context_latent=stable_inputs["context_latent"],
            source_latent=stable_inputs["source_latent"],
            seed=SEED, steps=STEPS, shift=SHIFT,
            denoise=0.5,
        )
        _capture_or_assert_output(session, "cover_denoise_50", result.tensor)

    def test_cover_denoise_75(self, session, stable_inputs):
        """Partial cover denoise at 0.75."""
        result = session.generate(
            conditioning=stable_inputs["cover_cond_a"],
            context_latent=stable_inputs["context_latent"],
            source_latent=stable_inputs["source_latent"],
            seed=SEED, steps=STEPS, shift=SHIFT,
            denoise=0.75,
        )
        _capture_or_assert_output(session, "cover_denoise_75", result.tensor)

    def test_cover_denoise_100(self, session, stable_inputs):
        """Full cover denoise (pure-noise init)."""
        result = session.generate(
            conditioning=stable_inputs["cover_cond_a"],
            context_latent=stable_inputs["context_latent"],
            source_latent=stable_inputs["source_latent"],
            seed=SEED, steps=STEPS, shift=SHIFT,
            denoise=1.0,
        )
        _capture_or_assert_output(session, "cover_denoise_100", result.tensor)

    def test_cover_sde_ramp(self, session, stable_inputs):
        """SDE method + sde_denoise_curve (ramp 0.3 -> 1.0)."""
        T = stable_inputs["T"]
        sde_curve = CurveRamp().execute(start=0.3, end=1.0, length=T)["curve"]
        result = session.generate(
            conditioning=stable_inputs["cover_cond_a"],
            context_latent=stable_inputs["context_latent"],
            source_latent=stable_inputs["source_latent"],
            seed=SEED, steps=STEPS, shift=SHIFT,
            method="sde",
            sde_denoise_curve=sde_curve,
        )
        _capture_or_assert_output(session, "cover_sde_ramp", result.tensor)

    def test_cover_velocity_scale(self, session, stable_inputs):
        """velocity_scale curve (ramp 0.2 -> 1.5)."""
        T = stable_inputs["T"]
        vel_curve = CurveRamp().execute(start=0.2, end=1.5, length=T)["curve"]
        result = session.generate(
            conditioning=stable_inputs["cover_cond_a"],
            context_latent=stable_inputs["context_latent"],
            source_latent=stable_inputs["source_latent"],
            seed=SEED, steps=STEPS, shift=SHIFT,
            denoise=1.0,
            velocity_scale=vel_curve,
        )
        _capture_or_assert_output(session, "cover_velocity_scale", result.tensor)

    def test_cover_ode_noise_sine(self, session, stable_inputs):
        """ode_noise_curve injection (1 Hz sine, range 0.0 -> 0.5)."""
        T = stable_inputs["T"]
        inject_curve = CurveWave().execute(
            wave_type="sine", frames_per_cycle=25,
            amplitude=0.25, offset=0.25, length=T,
        )["curve"]
        result = session.generate(
            conditioning=stable_inputs["cover_cond_a"],
            context_latent=stable_inputs["context_latent"],
            source_latent=stable_inputs["source_latent"],
            seed=SEED, steps=STEPS, shift=SHIFT,
            denoise=1.0,
            ode_noise_curve=inject_curve,
        )
        _capture_or_assert_output(session, "cover_ode_noise_sine", result.tensor)

    def test_cover_initial_noise_ramp(self, session, stable_inputs):
        """initial_noise_curve (ramp 0.3 -> 1.0) — per-frame init mix."""
        T = stable_inputs["T"]
        noise_curve = CurveRamp().execute(start=0.3, end=1.0, length=T)["curve"]
        result = session.generate(
            conditioning=stable_inputs["cover_cond_a"],
            context_latent=stable_inputs["context_latent"],
            source_latent=stable_inputs["source_latent"],
            seed=SEED, steps=STEPS, shift=SHIFT,
            denoise=1.0,
            initial_noise_curve=noise_curve,
        )
        _capture_or_assert_output(session, "cover_initial_noise_ramp", result.tensor)

    def test_cover_x0_target_blend(self, session, stable_inputs):
        """x0_target blending (two-pass: generate target then blend).

        The target pass uses the same seed/steps/shift as the blend pass
        so the target latent is itself stable. We re-generate it each
        test run (cheap compared to its role as a baseline input).
        """
        T = stable_inputs["T"]
        target = session.generate(
            conditioning=stable_inputs["cover_cond_b"],
            context_latent=stable_inputs["context_latent"],
            source_latent=stable_inputs["source_latent"],
            seed=SEED, steps=STEPS, shift=SHIFT,
            denoise=1.0,
        )
        blend_curve = CurveRamp().execute(start=0.0, end=0.8, length=T)["curve"]
        result = session.generate(
            conditioning=stable_inputs["cover_cond_a"],
            context_latent=stable_inputs["context_latent"],
            source_latent=stable_inputs["source_latent"],
            seed=SEED, steps=STEPS, shift=SHIFT,
            denoise=1.0,
            x0_target=target,
            x0_target_curve=blend_curve,
        )
        _capture_or_assert_output(session, "cover_x0_target_blend", result.tensor)

    def test_cover_latent_mask(self, session, stable_inputs):
        """latent_mask (selective denoising, pulse curve, denoise=0.85)."""
        T = stable_inputs["T"]
        mask_curve = CurveWave().execute(
            wave_type="pulse", frames_per_cycle=300,
            amplitude=0.5, offset=0.5, length=T,
        )["curve"]
        mask = TemporalMask().execute(
            latent=stable_inputs["source_latent"], curve=mask_curve,
        )["mask"]
        masked_latent = SetLatentNoiseMask().execute(
            latent=stable_inputs["source_latent"], mask=mask,
        )["latent"]
        result = session.generate(
            conditioning=stable_inputs["cover_cond_a"],
            context_latent=stable_inputs["context_latent"],
            source_latent=masked_latent,
            seed=SEED, steps=STEPS, shift=SHIFT,
            denoise=0.85,
        )
        _capture_or_assert_output(session, "cover_latent_mask", result.tensor)

    def test_cover_prompt_blend(self, session, stable_inputs):
        """Multi-condition temporal_weight (pulse crossfade, 2 prompts)."""
        T = stable_inputs["T"]
        blend_curve = CurveWave().execute(
            wave_type="pulse", frames_per_cycle=151,
            amplitude=0.5, offset=0.5, length=T,
        )["curve"]
        temporal_mask = Mask(tensor=blend_curve.tensor.clamp(0.0, 1.0))
        combined = ConditioningCombine().execute(
            conditioning_a=stable_inputs["cover_cond_a"],
            conditioning_b=stable_inputs["cover_cond_b"],
            temporal_weight_b=temporal_mask,
        )["conditioning"]
        result = session.generate(
            conditioning=combined,
            context_latent=stable_inputs["context_latent"],
            source_latent=stable_inputs["source_latent"],
            seed=SEED, steps=STEPS, shift=SHIFT,
            denoise=1.0,
        )
        _capture_or_assert_output(session, "cover_prompt_blend", result.tensor)

    def test_cover_guidance_curve(self, session, stable_inputs):
        """CFG with negative conditioning + per-frame guidance_curve.

        Baseline captured pre-Phase-1; Phase 2 re-ports CFG via APG into
        the streaming primitive (``_tick_complex_pt`` runs pos+neg rows
        in one batched forward pass and blends with a per-slot momentum
        buffer). The test asserts byte-equivalence vs that baseline.
        """
        T = stable_inputs["T"]
        negative = ConditioningZeroOut().execute(
            conditioning=stable_inputs["cover_cond_a"],
        )["conditioning"]
        cfg_curve = CurveRamp().execute(start=1.0, end=2.0, length=T)["curve"]
        result = session.generate(
            conditioning=stable_inputs["cover_cond_a"],
            negative=negative,
            context_latent=stable_inputs["context_latent"],
            source_latent=stable_inputs["source_latent"],
            seed=SEED, steps=STEPS, shift=SHIFT,
            denoise=1.0,
            guidance_curve=cfg_curve,
        )
        _capture_or_assert_output(session, "cover_guidance_curve", result.tensor)
