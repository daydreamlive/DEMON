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


_TRT_ENGINE_PROFILES: dict[float, dict[str, str]] = {
    60.0: {
        "decoder": "decoder_mixed_refit_b8_60s",
        "vae_encode": "vae_encode_fp16_60s",
        "vae_decode": "vae_decode_fp16_60s",
    },
    240.0: {
        "decoder": "decoder_mixed_refit_b8_240s",
        "vae_encode": "vae_encode_fp16_240s",
        "vae_decode": "vae_decode_fp16_240s",
    },
}


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


def select_trt_engines(duration_s: float = 60.0) -> dict[str, str]:
    """Pick the smallest engine set that can handle ``duration_s`` of audio.

    The 240s engines reserve workspace sized for their max profile at TRT
    context-creation time, so they sit on ~9 GB more VRAM than the 60s
    engines even when fed identical 60-second input (decoder +2.4 GB,
    vae_encode +6.4 GB, vae_decode +0.3 GB; see
    ``tests/benchmarks/vram_60s_vs_240s_results.md``).

    Default to the 60s set; only escalate to 240s when the requested
    duration would exceed the 60s engines' max profile (1500 latent
    frames at 25 fps).

    Args:
        duration_s: Generation duration in seconds.

    Returns:
        Dict suitable for passing to ``Session(trt_engines=...)``.
    """
    profile = _TRT_ENGINE_PROFILES[60.0] if duration_s <= 60.0 else _TRT_ENGINE_PROFILES[240.0]
    return default_trt_engines(**profile)


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
