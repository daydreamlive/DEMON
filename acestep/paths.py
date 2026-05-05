"""Central path resolution for ACE-Step models and engines.

All model/checkpoint/engine paths should be resolved through this module.
Nothing should hardcode paths or use relative symlinks.

Directory layout under MODELS_DIR:
    checkpoints/          Model weights (acestep-v15-turbo, etc.)
    trt_engines/          TensorRT engines and ONNX exports
    loras/                LoRA .safetensors files (flat — id is filename stem)

Resolution order for MODELS_DIR:
    1. ACESTEP_MODELS_DIR environment variable
    2. ~/.daydream-scope/models/demon
"""

from __future__ import annotations

import os
from pathlib import Path

_ENV_MODELS_DIR = "ACESTEP_MODELS_DIR"
_DEFAULT_MODELS_DIR = os.path.join(os.path.expanduser("~"), ".daydream-scope", "models", "demon")


def models_dir() -> Path:
    """Root directory for all ACEStep models and engines."""
    return Path(os.environ.get(_ENV_MODELS_DIR, _DEFAULT_MODELS_DIR))


def checkpoints_dir() -> Path:
    """Directory containing model checkpoints (acestep-v15-turbo, etc.)."""
    return models_dir() / "checkpoints"


def trt_engines_dir() -> Path:
    """Directory containing TensorRT engines and ONNX exports."""
    return models_dir() / "trt_engines"


def loras_dir() -> Path:
    """Directory containing LoRA .safetensors files.

    Flat layout: each ``*.safetensors`` becomes one library entry whose
    id is the filename stem. Subdirectories are not scanned.
    """
    return models_dir() / "loras"


def discover_loras(directory: Path | None = None) -> list[Path]:
    """List ``*.safetensors`` files in ``directory`` (default: ``loras_dir()``).

    Returns an empty list if the directory does not exist; callers should
    treat that as "no library", not as an error. Hidden files
    (``.gitignore``, etc.) and subdirectories are ignored.
    """
    d = Path(directory) if directory is not None else loras_dir()
    if not d.is_dir():
        return []
    return sorted(p for p in d.glob("*.safetensors") if p.is_file())


def trt_engine_path(engine_name: str) -> Path:
    """Full path to a specific TRT engine file.

    Args:
        engine_name: Engine directory name, e.g. "decoder_mixed_refit_b8_240s"

    Returns:
        Path like ~/.daydream-scope/models/demon/trt_engines/decoder_mixed_refit_b8_240s/decoder_mixed_refit_b8_240s.engine
    """
    return trt_engines_dir() / engine_name / f"{engine_name}.engine"


# Canonical engine profiles. Key is the maximum audio duration in seconds
# the engine context will accept. Engines are named by duration so the
# build script (`acestep.engine.trt.build --all --duration N`) can drive
# both halves from a single integer.
#
# Larger profiles reserve more workspace at TRT context-creation time and
# sit on more VRAM regardless of the actual input — see
# tests/benchmarks/vram_60s_vs_240s_results.md. Pick the smallest profile
# that fits the audio (see `select_trt_engines` and `available_trt_engines`).
_TRT_ENGINE_PROFILES: dict[float, dict[str, str]] = {
    60.0: {
        "decoder": "decoder_mixed_refit_b8_60s",
        "vae_encode": "vae_encode_fp16_60s",
        "vae_decode": "vae_decode_fp16_60s",
    },
    120.0: {
        "decoder": "decoder_mixed_refit_b8_120s",
        "vae_encode": "vae_encode_fp16_120s",
        "vae_decode": "vae_decode_fp16_120s",
    },
    240.0: {
        "decoder": "decoder_mixed_refit_b8_240s",
        "vae_encode": "vae_encode_fp16_240s",
        "vae_decode": "vae_decode_fp16_240s",
    },
}

_DEFAULT_TRT_NEEDS: tuple[str, ...] = ("decoder", "vae_encode", "vae_decode")


def default_trt_engines(
    decoder: str = "decoder_mixed_refit_b8_60s",
    vae_encode: str = "vae_encode_fp16_60s",
    vae_decode: str = "vae_decode_fp16_60s",
) -> dict[str, str]:
    """Return a trt_engines dict ready to pass to Session().

    Args:
        decoder: Decoder engine directory name.
        vae_encode: VAE encode engine directory name.
        vae_decode: VAE decode engine directory name.

    Returns:
        Dict with "decoder", "vae_encode", "vae_decode" keys mapping to
        absolute engine file paths as strings.
    """
    return {
        "decoder": str(trt_engine_path(decoder)),
        "vae_encode": str(trt_engine_path(vae_encode)),
        "vae_decode": str(trt_engine_path(vae_decode)),
    }


def max_profile_duration_s() -> float:
    """Largest registered TRT engine duration profile, in seconds.

    Useful as the upper bound on user-supplied audio: anything longer
    than this can't be handled by any built engine and would fail at
    inference time anyway. Demos cap at this value rather than
    hardcoding a single duration.
    """
    return max(_TRT_ENGINE_PROFILES.keys())


def select_trt_engines(duration_s: float = 60.0) -> dict[str, str]:
    """Pick the smallest engine profile that can handle ``duration_s``.

    Pure: returns paths without checking the filesystem. Use
    :func:`available_trt_engines` when you want existence-aware picking
    that falls back to the next-larger profile if the smallest fitting
    one isn't built. If ``duration_s`` exceeds every registered profile,
    the largest profile is returned (the caller then fails at engine
    load with a TRT-side error, same as before).

    Args:
        duration_s: Generation duration in seconds.

    Returns:
        Dict with ``decoder`` / ``vae_encode`` / ``vae_decode`` keys
        mapping to absolute engine file paths as strings.
    """
    for max_dur in sorted(_TRT_ENGINE_PROFILES.keys()):
        if max_dur >= duration_s:
            return default_trt_engines(**_TRT_ENGINE_PROFILES[max_dur])
    largest = max(_TRT_ENGINE_PROFILES.keys())
    return default_trt_engines(**_TRT_ENGINE_PROFILES[largest])


class EngineNotBuiltError(RuntimeError):
    """Raised when no built TRT engine profile satisfies a request.

    Carries enough context for callers (the demo server, primarily) to
    surface an actionable error to the operator: which duration was
    asked for, which engine keys were needed, what was checked, and the
    exact build command that would fix it.
    """

    def __init__(
        self,
        duration_s: float,
        needs: tuple[str, ...],
        missing: dict[float, list[str]],
    ) -> None:
        self.duration_s = float(duration_s)
        self.needs = tuple(needs)
        # Map of profile_max_dur -> list of missing engine paths for that
        # profile. Empty if no profile could even fit the duration.
        self.missing = dict(missing)

        fitting = sorted(d for d in _TRT_ENGINE_PROFILES if d >= duration_s)
        if fitting:
            recommended = int(fitting[0])
            self.build_command = (
                f"python -m acestep.engine.trt.build --all --duration {recommended}"
            )
            msg = (
                f"No TRT engine profile is built that can handle "
                f"{self.duration_s:.1f}s of audio. To build the smallest "
                f"fitting profile, run: {self.build_command}"
            )
        else:
            largest = max(_TRT_ENGINE_PROFILES.keys())
            self.build_command = None
            msg = (
                f"Audio duration {self.duration_s:.1f}s exceeds the largest "
                f"registered profile ({largest:.0f}s). Either use shorter "
                f"audio or add a larger profile to acestep/paths.py and "
                f"build it."
            )
        super().__init__(msg)


def available_trt_engines(
    duration_s: float = 60.0,
    *,
    needs: tuple[str, ...] = _DEFAULT_TRT_NEEDS,
) -> tuple[dict[str, str], float]:
    """Pick the smallest profile that fits ``duration_s`` AND is built.

    Walks profiles in ascending order. Returns the first one whose
    requested ``needs`` keys all exist on disk. Falls back to the
    next-larger profile (with the VRAM cost that implies) when the
    smallest fitting profile isn't built.

    Args:
        duration_s: Audio duration the engines must handle.
        needs: Which engine keys must be present on disk. Pass only the
            keys the caller will actually use; for a mixed-backend
            session that runs only the decoder on TRT, pass
            ``("decoder",)`` so missing VAE engines don't disqualify
            an otherwise-usable profile.

    Returns:
        ``(paths, max_dur)`` — ``paths`` is the dict of engine paths
        (with all keys, not just ``needs``), ``max_dur`` is the chosen
        profile's max duration. Caller can compare ``max_dur`` against
        ``duration_s`` to decide whether to log a "using larger profile"
        warning.

    Raises:
        EngineNotBuiltError: No profile can handle ``duration_s`` with
            the requested ``needs`` keys present on disk.
    """
    missing: dict[float, list[str]] = {}
    for max_dur in sorted(_TRT_ENGINE_PROFILES.keys()):
        if max_dur < duration_s:
            continue
        profile = _TRT_ENGINE_PROFILES[max_dur]
        paths = default_trt_engines(**profile)
        absent = [paths[k] for k in needs if not Path(paths[k]).exists()]
        if not absent:
            return paths, max_dur
        missing[max_dur] = absent
    raise EngineNotBuiltError(duration_s=duration_s, needs=needs, missing=missing)


# ------------------------------------------------------------------
# DreamVAE (distilled student decoder, drop-in for vae_decode)
# ------------------------------------------------------------------
#
# The dreamvae engines are NOT in ``_TRT_ENGINE_PROFILES`` because they
# don't replace the standard profile triple — they ride alongside it,
# selected per-session by the demo's ``fast_vae`` flag. Naming follows
# the same ``<component>_fp16_<dur>s`` convention as the teacher
# engines so the duration sweep stays consistent.

def dreamvae_decode_engine_name(duration_s: int) -> str:
    """Engine directory/file stem for a dreamvae decoder at duration_s."""
    return f"dreamvae_decode_fp16_{int(duration_s)}s"


def dreamvae_decode_engine_path(duration_s: int) -> Path:
    """Path to a dreamvae decode engine for a specific duration.

    Pure: does not check existence. Use
    :func:`available_dreamvae_decode_engine` for existence-aware lookup
    that mirrors the standard ``available_trt_engines`` fallback.
    """
    return trt_engine_path(dreamvae_decode_engine_name(duration_s))


def available_dreamvae_decode_engine(duration_s: float) -> Path | None:
    """Pick the smallest *built* dreamvae engine that fits ``duration_s``.

    Returns ``None`` if no fitting dreamvae engine is built, so callers
    (the demo's ``fast_vae`` path) can fall back to the teacher decoder
    without raising.
    """
    candidates = sorted(d for d in _TRT_ENGINE_PROFILES if d >= duration_s)
    if not candidates:
        candidates = [max(_TRT_ENGINE_PROFILES.keys())]
    for dur in candidates:
        path = dreamvae_decode_engine_path(int(dur))
        if path.exists():
            return path
    return None


def project_root() -> Path:
    """ACEStep source/project root (for non-model resources like test fixtures).

    Resolution order:
        1. ACESTEP_ROOT environment variable
        2. Walk up from this file to find the repo root
    """
    env_root = os.environ.get("ACESTEP_ROOT")
    if env_root:
        return Path(env_root)
    # Walk up from acestep/paths.py -> repo root
    d = Path(__file__).parent.parent
    return d
