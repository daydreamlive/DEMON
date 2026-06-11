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
    WINDOWED_VAE_DECODE_NAME,
    EngineNotBuiltError,
    available_trt_engines,
    max_built_profile_duration_s,
    max_profile_duration_s,
    select_trt_engines,
    smallest_fitting_profile_duration_s,
    trt_engine_needs,
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


def _create_windowed_decode_engine() -> None:
    """Create the fixed 1 s windowed VAE decode engine file."""
    engine_path = trt_engine_path(WINDOWED_VAE_DECODE_NAME)
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


# -----------------------------------------------------------------------
# trt_engine_needs: which engine keys a session actually consumes
# -----------------------------------------------------------------------

class TestTrtEngineNeeds:
    def test_non_trt_session_needs_nothing(self):
        assert trt_engine_needs(
            decoder_tensorrt=False, vae_tensorrt=False,
        ) == ()

    def test_decoder_only(self):
        assert trt_engine_needs(
            decoder_tensorrt=True, vae_tensorrt=False,
        ) == ("decoder",)

    def test_full_trt_without_windowing_needs_all_three(self, tmp_models_dir):
        assert trt_engine_needs(
            decoder_tensorrt=True, vae_tensorrt=True,
        ) == ("decoder", "vae_encode", "vae_decode")

    def test_windowed_decode_without_built_engine_still_needs_vae_decode(
        self, tmp_models_dir,
    ):
        # vae_window > 0 but the 1s fixed engine isn't built: the
        # session would fall back to the full-length engine, so it
        # stays in the needs.
        assert trt_engine_needs(
            decoder_tensorrt=True, vae_tensorrt=True, windowed_decode=True,
        ) == ("decoder", "vae_encode", "vae_decode")

    def test_windowed_decode_with_built_engine_drops_vae_decode(
        self, tmp_models_dir,
    ):
        _create_windowed_decode_engine()
        assert trt_engine_needs(
            decoder_tensorrt=True, vae_tensorrt=True, windowed_decode=True,
        ) == ("decoder", "vae_encode")


# -----------------------------------------------------------------------
# max_built_profile_duration_s: existence-aware source-duration cap
# -----------------------------------------------------------------------

class TestMaxBuiltProfileDuration:
    def test_falls_back_to_registry_max_when_nothing_built(self, tmp_models_dir):
        # Non-TRT sessions / truly-unbuilt installs keep the registered
        # ceiling; EngineNotBuiltError fires later with the fix command.
        assert max_built_profile_duration_s() == max_profile_duration_s()

    def test_only_60s_built_caps_at_60(self, tmp_models_dir):
        _create_engine_files(
            tmp_models_dir, 60.0, keys=("decoder", "vae_encode", "vae_decode"),
        )
        assert max_built_profile_duration_s() == 60.0

    def test_60_and_120_built_caps_at_120(self, tmp_models_dir):
        for d in (60.0, 120.0):
            _create_engine_files(
                tmp_models_dir, d, keys=("decoder", "vae_encode", "vae_decode"),
            )
        assert max_built_profile_duration_s() == 120.0

    def test_respects_needs(self, tmp_models_dir):
        # Decoder-only session: 120s decoder exists but its VAE engines
        # don't — with needs=("decoder",) the 120s profile still counts.
        _create_engine_files(tmp_models_dir, 60.0, keys=("decoder", "vae_encode", "vae_decode"))
        _create_engine_files(tmp_models_dir, 120.0, keys=("decoder",))
        assert max_built_profile_duration_s(needs=("decoder",)) == 120.0
        assert max_built_profile_duration_s() == 60.0


# -----------------------------------------------------------------------
# Minimal preset end-to-end: the engine set `demon-setup` builds must
# resolve cleanly for the demo's windowed-decode session shape.
# -----------------------------------------------------------------------

class TestMinimalPresetResolution:
    def _build_minimal_set(self, tmp_models_dir):
        # What `acestep.engine.trt.build --preset minimal` produces:
        # 60s decoder + 60s vae_encode + fixed 1s windowed vae_decode.
        # No full-length vae_decode engine at any duration.
        _create_engine_files(
            tmp_models_dir, 60.0, keys=("decoder", "vae_encode"),
        )
        _create_windowed_decode_engine()

    def test_windowed_session_resolves_on_minimal_set(self, tmp_models_dir):
        self._build_minimal_set(tmp_models_dir)
        needs = trt_engine_needs(
            decoder_tensorrt=True, vae_tensorrt=True, windowed_decode=True,
        )
        paths, picked = available_trt_engines(duration_s=45.0, needs=needs)
        assert picked == 60.0
        assert "_60s" in paths["decoder"]

    def test_minimal_set_caps_sources_at_60s(self, tmp_models_dir):
        self._build_minimal_set(tmp_models_dir)
        needs = trt_engine_needs(
            decoder_tensorrt=True, vae_tensorrt=True, windowed_decode=True,
        )
        assert max_built_profile_duration_s(needs=needs) == 60.0

    def test_full_length_decode_session_fails_on_minimal_set(
        self, tmp_models_dir,
    ):
        # vae_window=0 sessions still need a full-length decode engine,
        # which the minimal preset deliberately does not build.
        self._build_minimal_set(tmp_models_dir)
        needs = trt_engine_needs(
            decoder_tensorrt=True, vae_tensorrt=True, windowed_decode=False,
        )
        with pytest.raises(EngineNotBuiltError):
            available_trt_engines(duration_s=45.0, needs=needs)
