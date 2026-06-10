"""Tests for the multi-profile TRT engine picker.

Covers:
- ``select_trt_engines`` (pure: smallest-that-fits, no IO)
- ``available_trt_engines`` (existence-aware: smallest fitting profile
  whose required engines are on disk, falls back to next-larger)
- ``EngineNotBuiltError`` (carries the build command for the smallest
  fitting profile)
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from acestep.paths import (
    EngineNotBuiltError,
    available_trt_engines,
    max_profile_duration_s,
    select_trt_engines,
    smallest_fitting_profile_duration_s,
    trt_engine_path,
    trt_engine_profiles,
)


@pytest.fixture
def tmp_models_dir(monkeypatch, tmp_path):
    """Point ACESTEP_MODELS_DIR at a clean tmp dir so tests can create
    engine files without touching the real cache."""
    monkeypatch.setenv("ACESTEP_MODELS_DIR", str(tmp_path))
    (tmp_path / "trt_engines").mkdir()
    return tmp_path


def _create_engine_files(
    tmp_path: Path,
    max_dur: float,
    keys: tuple[str, ...],
    checkpoint: str = "acestep-v15-turbo",
) -> None:
    """Create empty engine files for one profile so existence checks pass."""
    profile = trt_engine_profiles(checkpoint)[max_dur]
    for k in keys:
        engine_name = profile[k]
        engine_path = trt_engine_path(engine_name)
        engine_path.parent.mkdir(parents=True, exist_ok=True)
        engine_path.write_bytes(b"")


# -----------------------------------------------------------------------
# select_trt_engines: pure picker (no IO)
# -----------------------------------------------------------------------

class TestSelectTrtEngines:
    def test_picks_60s_for_short_audio(self):
        paths = select_trt_engines(duration_s=30.0)
        assert "_60s" in paths["decoder"]
        assert "_60s" in paths["vae_encode"]
        assert "_60s" in paths["vae_decode"]

    def test_picks_60s_at_exact_60s_boundary(self):
        paths = select_trt_engines(duration_s=60.0)
        assert "_60s" in paths["decoder"]

    def test_picks_120s_just_over_60s(self):
        paths = select_trt_engines(duration_s=60.001)
        assert "_120s" in paths["decoder"]

    def test_picks_120s_at_exact_120s_boundary(self):
        paths = select_trt_engines(duration_s=120.0)
        assert "_120s" in paths["decoder"]

    def test_picks_240s_just_over_120s(self):
        paths = select_trt_engines(duration_s=120.001)
        assert "_240s" in paths["decoder"]

    def test_picks_240s_for_audio_within_240s(self):
        paths = select_trt_engines(duration_s=200.0)
        assert "_240s" in paths["decoder"]

    def test_falls_back_to_largest_when_audio_exceeds_all_profiles(self):
        """Pure picker has no other option; the largest profile is the
        last reasonable thing to return. Caller will fail at engine load."""
        paths = select_trt_engines(duration_s=10_000.0)
        assert "_240s" in paths["decoder"]

    def test_picks_xl_decoder_for_xl_checkpoint(self):
        paths = select_trt_engines(
            duration_s=30.0,
            checkpoint="acestep-v15-xl-turbo",
        )
        assert "decoder_xl-turbo_fp8_refit_b4_60s" in paths["decoder"]
        assert "vae_encode_fp16_60s" in paths["vae_encode"]

    def test_xl_checkpoint_has_240s_max_profile(self):
        assert max_profile_duration_s(checkpoint="acestep-v15-xl-turbo") == 240.0
        assert smallest_fitting_profile_duration_s(
            80.0,
            checkpoint="acestep-v15-xl-turbo",
        ) == 120.0
        assert smallest_fitting_profile_duration_s(
            150.0,
            checkpoint="acestep-v15-xl-turbo",
        ) == 240.0

    def test_unknown_checkpoint_is_rejected(self):
        with pytest.raises(ValueError, match="No canonical TRT engine profiles"):
            select_trt_engines(
                duration_s=30.0,
                checkpoint="acestep-v15-xl-base",
            )


# -----------------------------------------------------------------------
# available_trt_engines: existence-aware picker (IO)
# -----------------------------------------------------------------------

class TestAvailableTrtEnginesHappyPath:
    def test_picks_smallest_when_all_built(self, tmp_models_dir):
        # Build all three profiles; 30s audio should land on 60s.
        for d in (60.0, 120.0, 240.0):
            _create_engine_files(tmp_models_dir, d, keys=("decoder", "vae_encode", "vae_decode"))
        paths, picked = available_trt_engines(duration_s=30.0)
        assert picked == 60.0
        assert "_60s" in paths["decoder"]

    def test_returns_picked_duration(self, tmp_models_dir):
        for d in (60.0, 120.0, 240.0):
            _create_engine_files(tmp_models_dir, d, keys=("decoder", "vae_encode", "vae_decode"))
        _, picked = available_trt_engines(duration_s=119.0)
        assert picked == 120.0

    def test_xl_checkpoint_uses_xl_decoder_profiles(self, tmp_models_dir):
        _create_engine_files(
            tmp_models_dir,
            120.0,
            keys=("decoder", "vae_encode", "vae_decode"),
            checkpoint="acestep-v15-xl-turbo",
        )
        paths, picked = available_trt_engines(
            duration_s=80.0,
            checkpoint="acestep-v15-xl-turbo",
        )
        assert picked == 120.0
        assert "decoder_xl-turbo_fp8_refit_b4_120s" in paths["decoder"]


class TestAvailableTrtEnginesFallback:
    def test_falls_back_to_240s_when_60s_missing(self, tmp_models_dir):
        # 30s audio normally picks 60s, but only 240s is built.
        _create_engine_files(tmp_models_dir, 240.0, keys=("decoder", "vae_encode", "vae_decode"))
        paths, picked = available_trt_engines(duration_s=30.0)
        assert picked == 240.0
        assert "_240s" in paths["decoder"]

    def test_falls_back_to_240s_when_120s_missing(self, tmp_models_dir):
        # 80s audio normally picks 120s, but only 240s is built.
        _create_engine_files(tmp_models_dir, 240.0, keys=("decoder", "vae_encode", "vae_decode"))
        paths, picked = available_trt_engines(duration_s=80.0)
        assert picked == 240.0

    def test_does_not_pick_smaller_than_needed(self, tmp_models_dir):
        # 80s audio cannot use the 60s engine even if it's the only one built.
        _create_engine_files(tmp_models_dir, 60.0, keys=("decoder", "vae_encode", "vae_decode"))
        with pytest.raises(EngineNotBuiltError):
            available_trt_engines(duration_s=80.0)


class TestAvailableTrtEnginesNeedsParameter:
    def test_decoder_only_session_does_not_require_vae_engines(self, tmp_models_dir):
        # Mixed-backend setup (TRT decoder, eager VAE) only needs the
        # decoder engine — missing VAE engines should not disqualify.
        _create_engine_files(tmp_models_dir, 60.0, keys=("decoder",))
        paths, picked = available_trt_engines(
            duration_s=30.0, needs=("decoder",),
        )
        assert picked == 60.0
        assert "_60s" in paths["decoder"]

    def test_vae_only_session_does_not_require_decoder_engine(self, tmp_models_dir):
        _create_engine_files(tmp_models_dir, 60.0, keys=("vae_encode", "vae_decode"))
        paths, picked = available_trt_engines(
            duration_s=30.0, needs=("vae_encode", "vae_decode"),
        )
        assert picked == 60.0

    def test_vae_only_session_uses_shared_profiles_for_unregistered_checkpoint(
        self,
        tmp_models_dir,
    ):
        _create_engine_files(tmp_models_dir, 60.0, keys=("vae_encode", "vae_decode"))
        paths, picked = available_trt_engines(
            duration_s=30.0,
            needs=("vae_encode", "vae_decode"),
            checkpoint="acestep-v15-xl-base",
        )
        assert picked == 60.0
        assert "vae_encode_fp16_60s" in paths["vae_encode"]


# -----------------------------------------------------------------------
# vae_encode pinning: smallest built encode engine serves every duration
# (the TRT encode path chunks longer inputs — see vae_nodes)
# -----------------------------------------------------------------------

class TestVaeEncodePinnedToSmallestBuilt:
    def test_long_audio_keeps_smallest_built_vae_encode(self, tmp_models_dir):
        for d in (60.0, 120.0, 240.0):
            _create_engine_files(tmp_models_dir, d, keys=("decoder", "vae_encode", "vae_decode"))
        paths, picked = available_trt_engines(duration_s=200.0)
        assert picked == 240.0
        assert "_240s" in paths["decoder"]
        assert "_240s" in paths["vae_decode"]
        assert "vae_encode_fp16_60s" in paths["vae_encode"]

    def test_swap_between_profiles_shares_vae_encode(self, tmp_models_dir):
        for d in (60.0, 240.0):
            _create_engine_files(tmp_models_dir, d, keys=("decoder", "vae_encode", "vae_decode"))
        short, _ = available_trt_engines(duration_s=30.0)
        long, _ = available_trt_engines(duration_s=200.0)
        assert short["vae_encode"] == long["vae_encode"]

    def test_falls_back_to_larger_encode_when_small_not_built(self, tmp_models_dir):
        # A pod that only built the 240s profile keeps using it.
        _create_engine_files(tmp_models_dir, 240.0, keys=("decoder", "vae_encode", "vae_decode"))
        paths, picked = available_trt_engines(duration_s=200.0)
        assert picked == 240.0
        assert "vae_encode_fp16_240s" in paths["vae_encode"]

    def test_no_encode_built_still_raises_when_needed(self, tmp_models_dir):
        _create_engine_files(tmp_models_dir, 60.0, keys=("decoder", "vae_decode"))
        with pytest.raises(EngineNotBuiltError):
            available_trt_engines(duration_s=30.0)

    def test_decoder_only_needs_ignores_vae_encode_state(self, tmp_models_dir):
        _create_engine_files(tmp_models_dir, 60.0, keys=("decoder",))
        paths, picked = available_trt_engines(duration_s=30.0, needs=("decoder",))
        assert picked == 60.0


# -----------------------------------------------------------------------
# Chunked VAE encode planning (pure; mirrors vae_nodes chunk geometry)
# -----------------------------------------------------------------------

class TestPlanEncodeChunks:
    SPF = 1920  # samples per latent frame

    def _check_plan(self, n_samples, max_samples, min_samples):
        from acestep.nodes.vae_nodes import (
            _VAE_ENCODE_CHUNK_MARGIN_FRAMES,
            _plan_encode_chunks,
        )

        plan = _plan_encode_chunks(n_samples, max_samples, min_samples)
        total_frames = (n_samples + self.SPF - 1) // self.SPF
        # Cores tile [0, total_frames) exactly, in order.
        assert plan[0][2] == 0
        assert plan[-1][3] == total_frames
        for (a, b), (c, d) in zip(
            [(p[2], p[3]) for p in plan], [(p[2], p[3]) for p in plan[1:]],
        ):
            assert b == c
        for ctx_start, ctx_end, core_start, core_end in plan:
            # Context spans fit the engine profile.
            ctx_samples = (ctx_end - ctx_start) * self.SPF
            assert ctx_samples <= max_samples
            assert ctx_samples >= min(min_samples, n_samples)
            # Core sits inside the context with the required margin
            # (or extends to the input edge).
            assert ctx_start <= core_start <= core_end <= ctx_end
            if core_start > 0:
                assert core_start - ctx_start >= _VAE_ENCODE_CHUNK_MARGIN_FRAMES
            if core_end < total_frames:
                assert ctx_end - core_end >= _VAE_ENCODE_CHUNK_MARGIN_FRAMES
        return plan

    def test_240s_input_through_60s_engine(self):
        plan = self._check_plan(240 * 48000, 2_880_000, 240_000)
        assert len(plan) > 1

    def test_61s_input_through_60s_engine(self):
        self._check_plan(61 * 48000, 2_880_000, 240_000)

    def test_tail_chunk_extends_left_context_to_engine_min(self):
        # Craft a tail chunk shorter than the engine min: the plan must
        # widen its context to min_samples rather than emit an
        # out-of-profile shape.
        plan = self._check_plan(2_880_000 + 5 * self.SPF, 2_880_000, 240_000)
        ctx_start, ctx_end, _, core_end = plan[-1]
        assert (ctx_end - ctx_start) * self.SPF >= 240_000

    def test_non_frame_aligned_input(self):
        self._check_plan(240 * 48000 + 777, 2_880_000, 240_000)

    def test_engine_smaller_than_margins_raises(self):
        from acestep.nodes.vae_nodes import _plan_encode_chunks

        with pytest.raises(RuntimeError, match="too small"):
            _plan_encode_chunks(10_000_000, 100 * self.SPF, 1)


# -----------------------------------------------------------------------
# EngineNotBuiltError: actionable build command
# -----------------------------------------------------------------------

class TestEngineNotBuiltError:
    def test_recommends_smallest_fitting_profile(self, tmp_models_dir):
        # Nothing built; 30s audio fits in the 60s profile.
        with pytest.raises(EngineNotBuiltError) as excinfo:
            available_trt_engines(duration_s=30.0)
        assert excinfo.value.build_command is not None
        assert "--duration 60" in excinfo.value.build_command

    def test_recommends_120s_for_audio_in_that_range(self, tmp_models_dir):
        with pytest.raises(EngineNotBuiltError) as excinfo:
            available_trt_engines(duration_s=80.0)
        assert "--duration 120" in excinfo.value.build_command

    def test_no_build_command_when_audio_exceeds_all_profiles(self, tmp_models_dir):
        with pytest.raises(EngineNotBuiltError) as excinfo:
            available_trt_engines(duration_s=10_000.0)
        assert excinfo.value.build_command is None
        assert "exceeds" in str(excinfo.value).lower()

    def test_carries_duration_and_needs(self, tmp_models_dir):
        with pytest.raises(EngineNotBuiltError) as excinfo:
            available_trt_engines(duration_s=80.0, needs=("decoder",))
        assert excinfo.value.duration_s == 80.0
        assert excinfo.value.needs == ("decoder",)

    def test_xl_missing_engine_recommends_xl_build_command(self, tmp_models_dir):
        with pytest.raises(EngineNotBuiltError) as excinfo:
            available_trt_engines(
                duration_s=30.0,
                needs=("decoder",),
                checkpoint="acestep-v15-xl-turbo",
            )
        command = excinfo.value.build_command
        assert command is not None
        assert "--checkpoint acestep-v15-xl-turbo" in command
        assert "--decoder-only" in command
        assert "--duration 60" in command
        assert "--batch-max 4" in command
        assert "--batch-opt 4" in command
        assert "--builder-optimization-level 5" in command
        assert "--workspace-gb 20" in command
        assert "--decoder-precision fp8_mixed" in command
        assert "--activation-absmax-json" in command
        assert "decoder_xl_fp8/60s/activation_absmax.json" in command
