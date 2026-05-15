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


def lora_trigger(lora_path: Path | str) -> str:
    """Read the optional trigger-word sidecar for a LoRA.

    The sidecar is a plain-text ``<stem>.trigger.txt`` file living next to
    the ``.safetensors``. It holds a single activation word (the token
    the LoRA was trained against) — when present, the engine prepends it
    to the user's caption before passing to the text encoder so the LoRA
    style actually fires at inference. The sidecar is OPTIONAL; LoRAs
    trained without a documented trigger (or pulled in via a manifest
    line without the ``|<TRIGGER>`` field) just have no file and the
    engine treats them as no-trigger styles.

    Returns the trigger string with whitespace stripped, or ``""`` when:
    - the sidecar doesn't exist
    - the sidecar is empty after stripping
    - the file can't be read (permissions, IO error)

    Empty-string return is a deliberate signal — callers can do
    ``if trigger: ...`` to decide whether to inject. No exceptions
    escape; this is read every catalog broadcast and shouldn't crash
    the WS loop on a malformed sidecar.
    """
    p = Path(lora_path)
    # Strip the .safetensors suffix to land on the stem, then add .trigger.txt
    # so we resolve siblings like:
    #   /…/loras/bptkno.safetensors        → /…/loras/bptkno.trigger.txt
    sidecar = p.with_suffix("").with_suffix(".trigger.txt")
    try:
        return sidecar.read_text(encoding="utf-8").strip()
    except (OSError, UnicodeDecodeError):
        return ""


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

_XL_TURBO_TRT_ENGINE_PROFILES: dict[float, dict[str, str]] = {
    60.0: {
        "decoder": "decoder_xl-turbo_fp8_refit_b4_60s",
        "vae_encode": "vae_encode_fp16_60s",
        "vae_decode": "vae_decode_fp16_60s",
    },
    120.0: {
        "decoder": "decoder_xl-turbo_fp8_refit_b4_120s",
        "vae_encode": "vae_encode_fp16_120s",
        "vae_decode": "vae_decode_fp16_120s",
    },
    240.0: {
        "decoder": "decoder_xl-turbo_fp8_refit_b4_240s",
        "vae_encode": "vae_encode_fp16_240s",
        "vae_decode": "vae_decode_fp16_240s",
    },
}

_DEFAULT_TRT_CHECKPOINT = "acestep-v15-turbo"
_TRT_ENGINE_PROFILES_BY_CHECKPOINT: dict[str, dict[float, dict[str, str]]] = {
    _DEFAULT_TRT_CHECKPOINT: _TRT_ENGINE_PROFILES,
    "acestep-v15-xl-turbo": _XL_TURBO_TRT_ENGINE_PROFILES,
}

_DEFAULT_TRT_NEEDS: tuple[str, ...] = ("decoder", "vae_encode", "vae_decode")


def _checkpoint_name(checkpoint: str | Path | None) -> str:
    if checkpoint is None:
        return _DEFAULT_TRT_CHECKPOINT
    raw = str(checkpoint).replace("\\", "/").rstrip("/")
    return raw.rsplit("/", 1)[-1] or _DEFAULT_TRT_CHECKPOINT


def trt_engine_profiles(
    checkpoint: str | Path = _DEFAULT_TRT_CHECKPOINT,
) -> dict[float, dict[str, str]]:
    """Canonical TRT engine profiles for a checkpoint.

    The VAE engine names are shared across ACE-Step 1.5 checkpoints, but
    decoder engines are checkpoint-specific. Raising for unknown checkpoints
    prevents demos from accidentally loading a 2B decoder engine for a
    different DiT variant.
    """
    name = _checkpoint_name(checkpoint)
    try:
        return _TRT_ENGINE_PROFILES_BY_CHECKPOINT[name]
    except KeyError as exc:
        supported = ", ".join(sorted(_TRT_ENGINE_PROFILES_BY_CHECKPOINT))
        raise ValueError(
            f"No canonical TRT engine profiles registered for checkpoint "
            f"{name!r}. Supported checkpoints: {supported}."
        ) from exc


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


def max_profile_duration_s(
    checkpoint: str | Path = _DEFAULT_TRT_CHECKPOINT,
) -> float:
    """Largest registered TRT engine duration profile, in seconds.

    Useful as the upper bound on user-supplied audio: anything longer
    than this can't be handled by any built engine and would fail at
    inference time anyway. Demos cap at this value rather than
    hardcoding a single duration.
    """
    profiles = trt_engine_profiles(checkpoint)
    return max(profiles.keys())


def smallest_fitting_profile_duration_s(
    duration_s: float,
    *,
    checkpoint: str | Path = _DEFAULT_TRT_CHECKPOINT,
) -> float:
    """Smallest registered profile duration that can hold ``duration_s``.

    Pure: ignores filesystem state. Returns the registered profile,
    not whichever was *built* — so callers can compare against the
    actually-loaded profile to decide whether a fallback happened.
    Falls back to ``max_profile_duration_s()`` when no registered
    profile is large enough (matches ``select_trt_engines``).
    """
    profiles = trt_engine_profiles(checkpoint)
    for max_dur in sorted(profiles.keys()):
        if max_dur >= duration_s:
            return max_dur
    return max(profiles.keys())


def select_trt_engines(
    duration_s: float = 60.0,
    *,
    checkpoint: str | Path = _DEFAULT_TRT_CHECKPOINT,
) -> dict[str, str]:
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
    profiles = trt_engine_profiles(checkpoint)
    for max_dur in sorted(profiles.keys()):
        if max_dur >= duration_s:
            return default_trt_engines(**profiles[max_dur])
    largest = max(profiles.keys())
    return default_trt_engines(**profiles[largest])


def _trt_build_command(
    *,
    checkpoint: str,
    duration_s: float,
    needs: tuple[str, ...],
) -> str:
    duration = int(duration_s)
    parts = ["python", "-m", "acestep.engine.trt.build", "--all"]
    if checkpoint != _DEFAULT_TRT_CHECKPOINT:
        parts.extend(["--checkpoint", checkpoint])
    if "decoder" not in needs:
        parts.append("--vae-only")
    elif "vae_encode" not in needs and "vae_decode" not in needs:
        parts.append("--decoder-only")
    parts.extend(["--duration", str(duration)])
    if checkpoint == "acestep-v15-xl-turbo" and "decoder" in needs:
        # XL canonical engines are FP8 W8A8, which requires a per-profile
        # activation absmax JSON captured against the matching bf16 engine.
        # The path encodes the duration so the hint matches the storage
        # layout used by scripts/collect_activation_absmax.py --output-dir.
        absmax_json = (
            f"<MODELS_DIR>/calibration/decoder_xl_fp8/{duration}s/"
            "activation_absmax.json"
        )
        parts.extend([
            "--batch-max", "4",
            "--batch-opt", "4",
            "--builder-optimization-level", "5",
            "--workspace-gb", "20",
            "--export-locally",
            "--decoder-precision", "fp8_mixed",
            "--activation-absmax-json", absmax_json,
        ])
    return " ".join(parts)


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
        checkpoint: str | Path = _DEFAULT_TRT_CHECKPOINT,
    ) -> None:
        self.duration_s = float(duration_s)
        self.needs = tuple(needs)
        self.checkpoint = _checkpoint_name(checkpoint)
        # Map of profile_max_dur -> list of missing engine paths for that
        # profile. Empty if no profile could even fit the duration.
        self.missing = dict(missing)

        profiles = trt_engine_profiles(self.checkpoint)
        fitting = sorted(d for d in profiles if d >= duration_s)
        if fitting:
            recommended = int(fitting[0])
            self.build_command = _trt_build_command(
                checkpoint=self.checkpoint,
                duration_s=recommended,
                needs=self.needs,
            )
            msg = (
                f"No TRT engine profile is built that can handle "
                f"{self.duration_s:.1f}s of audio for {self.checkpoint}. "
                f"To build the smallest fitting profile, run: "
                f"{self.build_command}"
            )
        else:
            largest = max(profiles.keys())
            self.build_command = None
            msg = (
                f"Audio duration {self.duration_s:.1f}s exceeds the largest "
                f"registered TRT profile for {self.checkpoint} "
                f"({largest:.0f}s). Either use shorter audio or add a larger "
                f"profile to acestep/paths.py and build it."
            )
        super().__init__(msg)


def available_trt_engines(
    duration_s: float = 60.0,
    *,
    needs: tuple[str, ...] = _DEFAULT_TRT_NEEDS,
    checkpoint: str | Path = _DEFAULT_TRT_CHECKPOINT,
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
    profile_checkpoint = (
        checkpoint if "decoder" in needs else _DEFAULT_TRT_CHECKPOINT
    )
    profiles = trt_engine_profiles(profile_checkpoint)
    missing: dict[float, list[str]] = {}
    for max_dur in sorted(profiles.keys()):
        if max_dur < duration_s:
            continue
        profile = profiles[max_dur]
        paths = default_trt_engines(**profile)
        absent = [paths[k] for k in needs if not Path(paths[k]).exists()]
        if not absent:
            return paths, max_dur
        missing[max_dur] = absent
    raise EngineNotBuiltError(
        duration_s=duration_s,
        needs=needs,
        missing=missing,
        checkpoint=profile_checkpoint,
    )


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


# ------------------------------------------------------------------
# Windowed VAE decode (single shared profile across both decoder
# variants). The profile is small enough that it costs ~1.5 GB of
# workspace at TRT context-creation time vs ~9 GB for the canonical
# 240 s engine — see tests/benchmarks/bench_vae_decode_profiles.py.
#
# Profile shape (in latent frames at 25 fps):
#     min = 75   (3 s)
#     opt = 125  (5 s)
#     max = 750  (30 s)
#
# The ``StreamVAEDecode`` window+overlap chunks fit comfortably inside
# this range for any user-facing window in [3, 30] seconds, which is
# the range Session enforces when ``vae_window > 0``. The lower bound
# matches the engine profile's ``min_frames=75`` (3 s at 25 fps); the
# previous defensive clamp at 5.0 silently rounded smaller user-set
# windows up to 5 s and inflated every wire slice by ~67 % for nothing.
# ------------------------------------------------------------------

WINDOWED_VAE_DECODE_NAME = "vae_decode_fp16_3to30s"
WINDOWED_DREAMVAE_DECODE_NAME = "dreamvae_decode_fp16_3to30s"
WINDOWED_VAE_PROFILE_FRAMES: tuple[int, int, int] = (75, 125, 750)
WINDOWED_VAE_WINDOW_RANGE_S: tuple[float, float] = (3.0, 30.0)


def windowed_vae_decode_engine_name(*, dreamvae: bool = False) -> str:
    """Engine directory/file stem for the windowed VAE decode engine.

    Args:
        dreamvae: Pick the distilled student engine instead of the
            standard teacher engine. The two share the same profile
            shape so they're interchangeable from the runtime's POV.
    """
    return WINDOWED_DREAMVAE_DECODE_NAME if dreamvae else WINDOWED_VAE_DECODE_NAME


def windowed_vae_decode_engine_path(*, dreamvae: bool = False) -> Path:
    """Path to the windowed VAE decode engine. Pure: does not check
    existence. Use :func:`available_windowed_vae_decode_engine` for
    existence-aware lookup."""
    return trt_engine_path(windowed_vae_decode_engine_name(dreamvae=dreamvae))


def available_windowed_vae_decode_engine(*, dreamvae: bool = False) -> Path | None:
    """Return the windowed VAE decode engine path if it is built, else None.

    Callers (Session, demo backends) use this to opportunistically swap
    in the small-profile engine when ``vae_window > 0``, falling back
    silently to whatever the caller originally configured.
    """
    p = windowed_vae_decode_engine_path(dreamvae=dreamvae)
    return p if p.exists() else None


def looks_like_dreamvae_engine(path: str | Path) -> bool:
    """True when ``path`` points at a dreamvae (distilled) engine.

    The runtime distinguishes the two variants only by name; both share
    the same I/O contract (latents [B,64,T] -> audio [B,2,1920*T]).
    """
    return Path(path).name.startswith("dreamvae_decode_")


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
