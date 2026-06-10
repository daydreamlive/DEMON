"""Central path resolution for ACE-Step models and engines.

All model/checkpoint/engine paths should be resolved through this module.
Nothing should hardcode paths or use relative symlinks.

Directory layout under MODELS_DIR:
    checkpoints/          Model weights (acestep-v15-turbo, etc.)
    trt_engines/          TensorRT engines and ONNX exports
    loras/                LoRA .safetensors files (flat — id is filename stem)
    MelBandRoFormer/      Mel-Band RoFormer stem-separation checkpoint
    fixtures/             Test fixtures and their precomputed sidecars
    user_uploads/         User-uploaded audio and their precomputed sidecars

Resolution order for MODELS_DIR:
    1. ACESTEP_MODELS_DIR environment variable
    2. ~/.daydream-scope/models/demon
"""

from __future__ import annotations

import json
import os
from pathlib import Path

_ENV_MODELS_DIR = "ACESTEP_MODELS_DIR"
_DEFAULT_MODELS_DIR = os.path.join(os.path.expanduser("~"), ".daydream-scope", "models", "demon")
_LOCAL_CONFIG_FILENAME = "acestep.local.json"


def models_dir() -> Path:
    """Root directory for all ACEStep models and engines."""
    return Path(os.environ.get(_ENV_MODELS_DIR, _DEFAULT_MODELS_DIR))


# Module-level memoization for the local config. The file is read once
# on first access — the engine's LoRA library is registered at boot and
# can't be hot-swapped after, so a process restart is already required
# to pick up config edits. ``_LOCAL_CONFIG_UNLOADED`` is a sentinel so
# we can distinguish "not yet read" from "read and got empty dict".
_LOCAL_CONFIG_UNLOADED: object = object()
_local_config_cache: object = _LOCAL_CONFIG_UNLOADED


def local_config_path() -> Path:
    """Path to the operator's local config file (``acestep.local.json``
    at the project root).

    The file is gitignored so each clone can carry its own local paths
    without leaking them. See ``acestep.local.example.json`` (committed
    sibling) for the schema and field documentation.
    """
    return project_root() / _LOCAL_CONFIG_FILENAME


def load_local_config() -> dict:
    """Read ``acestep.local.json`` from the project root.

    Memoized on first call; restart the process to pick up edits.
    Returns ``{}`` on missing file or any read/parse error — a broken
    local config must never block engine boot. Errors are printed to
    stdout so a typo doesn't fail silently.
    """
    global _local_config_cache
    if _local_config_cache is not _LOCAL_CONFIG_UNLOADED:
        return _local_config_cache  # type: ignore[return-value]
    p = local_config_path()
    try:
        loaded = json.loads(p.read_text(encoding="utf-8"))
    except FileNotFoundError:
        _local_config_cache = {}
        return {}
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as e:
        print(f"[paths] failed to read {p}: {e}; ignoring")
        _local_config_cache = {}
        return {}
    if not isinstance(loaded, dict):
        print(
            f"[paths] {p} must be a JSON object, got "
            f"{type(loaded).__name__}; ignoring"
        )
        _local_config_cache = {}
        return {}
    _local_config_cache = loaded
    return loaded


def clear_local_config_cache() -> None:
    """Drop the memoized local config. Tests + manual reloads only."""
    global _local_config_cache
    _local_config_cache = _LOCAL_CONFIG_UNLOADED


def extra_lora_dirs() -> list[Path]:
    """Additional LoRA root directories from the local config.

    Reads the ``lora_extra_dirs`` array from ``acestep.local.json``.
    Relative paths are resolved against the config file's directory
    (the project root) so an operator can check a config into a
    sibling repo and still have the paths work. ``~`` is expanded.

    Used by the LoRA discovery path so operators can point at training
    output directories or alternate libraries for testing without
    moving files into ``loras_dir()``. Missing dirs are silently
    skipped at scan time; we don't validate here so a stale path in
    the config doesn't crash the boot.
    """
    cfg = load_local_config()
    raw = cfg.get("lora_extra_dirs")
    if not isinstance(raw, list):
        return []
    base = local_config_path().parent
    out: list[Path] = []
    for entry in raw:
        if not isinstance(entry, str):
            continue
        s = entry.strip()
        if not s:
            continue
        p = Path(os.path.expanduser(s))
        if not p.is_absolute():
            p = (base / p).resolve()
        out.append(p)
    return out


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


def melband_roformer_dir() -> Path:
    """Directory containing the Mel-Band RoFormer stem-separation checkpoint."""
    return models_dir() / "MelBandRoFormer"


def melband_roformer_model_path(
    filename: str = "MelBandRoformer_fp16.safetensors",
) -> Path:
    """Full path to the Mel-Band RoFormer checkpoint file."""
    return melband_roformer_dir() / filename


def fixtures_dir() -> Path:
    """Directory containing test fixture audio and precomputed sidecars."""
    return models_dir() / "fixtures"


def user_uploads_dir() -> Path:
    """Directory containing user-uploaded audio and precomputed sidecars."""
    return models_dir() / "user_uploads"


# Per-checkpoint probe-bundle subpath; shared between HF fetch
# (``steering_vectors/<subpath>``) and the local cache. XL is absent
# because the 2B vectors don't transfer to a different hidden_size /
# layer count.
_STEERING_VECTORS_BY_CHECKPOINT: dict[str, str] = {
    "acestep-v15-turbo": "v15-turbo/shift3.5_n8_seed1528",
}


def steering_bundle_subpath(
    checkpoint: str | Path | None = None,
) -> str | None:
    """Bundle subpath registered for ``checkpoint`` (pure lookup)."""
    name = _checkpoint_name(checkpoint)
    return _STEERING_VECTORS_BY_CHECKPOINT.get(name)


def steering_vectors_dir() -> Path:
    """Root directory for cached steering-vector bundles."""
    return models_dir() / "steering_vectors"


def steering_vector_dir(
    checkpoint: str | Path | None = None,
) -> Path | None:
    """Local cache directory for a checkpoint's steering bundle.

    Pure: returns where vectors WOULD live; ``None`` when no bundle
    is registered. Callers needing the dir populated should go through
    ``acestep.steering.hub.ensure_steering_vectors`` first.
    """
    subpath = steering_bundle_subpath(checkpoint)
    if subpath is None:
        return None
    return steering_vectors_dir() / subpath


def discover_loras(directory: Path | None = None) -> list[Path]:
    """List ``*.safetensors`` files recursively under ``directory``
    (default: ``loras_dir()``).

    Subdirectories are scanned so an operator can group LoRAs by
    artist / training run / source without flattening into one folder.
    Returns an empty list if the directory does not exist; callers
    should treat that as "no library", not as an error. Hidden files
    (``.gitignore``, etc.) are still skipped via the ``*.safetensors``
    pattern.
    """
    d = Path(directory) if directory is not None else loras_dir()
    if not d.is_dir():
        return []
    return sorted(p for p in d.rglob("*.safetensors") if p.is_file())


def discover_all_loras() -> list[Path]:
    """Scan ``loras_dir()`` plus every directory in ``extra_lora_dirs()``.

    Recursive in each root. Deduplicated by absolute path so a LoRA
    that lives under both the primary library and an extra dir (e.g.
    a symlink) is registered once. Order is: primary library first
    (sorted by path), then each extra dir in env-declaration order
    (each sorted by path). This makes the catalog deterministic
    across boots.
    """
    roots: list[Path] = [loras_dir(), *extra_lora_dirs()]
    seen: set[str] = set()
    out: list[Path] = []
    for root in roots:
        for p in discover_loras(root):
            key = str(p.resolve())
            if key in seen:
                continue
            seen.add(key)
            out.append(p)
    return out


def lora_sidecar(lora_path: Path | str, suffix: str) -> Path:
    """Resolve a LoRA weight file to a sibling sidecar with ``suffix``.

    ``…/loras/bptkno.safetensors`` + ``.metadata.json`` →
    ``…/loras/bptkno.metadata.json``.

    The naive ``with_suffix("").with_suffix(suffix)`` trick is wrong: it
    strips a *second* suffix, so any dot inside the stem corrupts the
    result (``acestep1.5-dora-v2.safetensors`` → ``acestep1.metadata.json``
    instead of ``acestep1.5-dora-v2.metadata.json``). ``Path.stem`` strips
    only the final extension, which is exactly the one real weight suffix,
    so it handles arbitrary dotted stems.

    ``suffix`` is appended verbatim and should include its leading dot
    (e.g. ``".metadata.json"``, ``".trigger.txt"``).
    """
    p = Path(lora_path)
    return p.with_name(p.stem + suffix)


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
    sidecar = lora_sidecar(lora_path, ".trigger.txt")
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
# scripts/benchmarks/vram_60s_vs_240s_results.md. Pick the smallest profile
# that fits the audio (see `select_trt_engines` and `available_trt_engines`).
_TRT_ENGINE_PROFILES: dict[float, dict[str, str]] = {
    60.0: {
        "decoder": "spectral_decoder_mixed_refit_b8_60s",
        "vae_encode": "vae_encode_fp16_60s",
        "vae_decode": "vae_decode_fp16_60s",
    },
    120.0: {
        "decoder": "spectral_decoder_mixed_refit_b8_120s",
        "vae_encode": "vae_encode_fp16_120s",
        "vae_decode": "vae_decode_fp16_120s",
    },
    240.0: {
        "decoder": "spectral_decoder_mixed_refit_b8_240s",
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


# Maps internal checkpoint identifiers to the model-scale label used by
# LoRA sidecar metadata (``model.base_model_scale`` — "2B" or "5B").
# Checkpoints not in this map are scale-unknown; the runtime falls back
# to "don't filter" so an undocumented checkpoint doesn't accidentally
# hide every LoRA.
_CHECKPOINT_SCALES: dict[str, str] = {
    "acestep-v15-turbo": "2B",
    "acestep-v15-xl-turbo": "5B",
}


def checkpoint_scale(checkpoint: str | Path | None) -> str | None:
    """Map a checkpoint identifier to its model-scale label ("2B" / "5B").

    Compares against the LoRA sidecar's ``model.base_model_scale`` so
    the UI can hide incompatible LoRAs. Returns ``None`` for unknown
    checkpoints — callers should treat this as "don't filter" rather
    than "incompatible with everything".
    """
    if checkpoint is None:
        return None
    return _CHECKPOINT_SCALES.get(_checkpoint_name(checkpoint))


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
    decoder: str = "spectral_decoder_mixed_refit_b8_60s",
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
        # layout used by scripts/calibration/collect_activation_absmax.py --output-dir.
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

    # vae_encode is duration-independent: the TRT encode path chunks
    # inputs longer than the engine's max profile shape (see
    # acestep/nodes/vae_nodes.py), so the smallest BUILT encode engine
    # serves every source duration. Larger encode engines exist only
    # for pods that never built the small one; preferring the smallest
    # saves multi-GB of context-creation workspace (the 240 s engine
    # reserves ~6.4 GB more than the 60 s one — see
    # scripts/benchmarks/vram_60s_vs_240s_results.md) and means profile
    # swaps never reload vae_encode, removing the highest-pressure
    # engine load from the swap path entirely.
    smallest_built_enc: str | None = None
    for max_dur in sorted(profiles.keys()):
        enc_path = trt_engine_path(profiles[max_dur]["vae_encode"])
        if enc_path.exists():
            smallest_built_enc = str(enc_path)
            break

    missing: dict[float, list[str]] = {}
    for max_dur in sorted(profiles.keys()):
        if max_dur < duration_s:
            continue
        profile = profiles[max_dur]
        paths = default_trt_engines(**profile)
        if smallest_built_enc is not None:
            paths["vae_encode"] = smallest_built_enc
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
# Windowed VAE decode — a FIXED 1-second engine (min == opt == max ==
# 25 frames). Selected by the runtime whenever ``vae_window > 0``.
#
# Every streaming decode feeds this engine exactly 25 latent frames
# (1.0 s): the kept "wire slice" is the middle ``vae_window`` seconds,
# and the leftover frames on each side are the receptive-field margin
# that StreamVAEDecode trims off. Because the profile is fixed, the
# node ALWAYS hands it a 25-frame span and derives the margin from
# ``25 - keep`` (see StreamVAEDecode) — the ``vae_overlap`` knob is
# ignored in this mode so live demo controls can never feed an
# out-of-profile shape.
#
# Why fixed 1 s rather than a 0.32-30 s range:
#   * Speed is identical at the 1 s operating point (~2.2 ms either way).
#   * A ranged (…, max=750) engine reserves TRT activation workspace for
#     its 30 s max — ~1.58 GB committed at context creation — even though
#     it only ever decodes 1 s. The fixed engine reserves ~81 MB. That's
#     ~1.5 GB of VRAM saved for zero speed cost. See
#     scripts/benchmarks/bench_vae_decode_profiles.py.
#
# Margin sizing: 8 frames (0.32 s) of context each side already puts the
# kept center below the fp16 decode noise; 10 frames (0.4 s) is bit-
# identical to a full-context decode. With a 25-frame decode, a 9-frame
# (0.36 s) keep leaves 8 frames of margin each side — safely converged.
# See scripts/benchmarks/vae_window_convergence.py and the project memory
# "VAE decoder receptive field".
#
# The profile is recorded in each engine's .metadata.json (PR #152), so
# changing these frames invalidates the sidecar and forces a rebuild.
# ------------------------------------------------------------------

WINDOWED_VAE_DECODE_NAME = "vae_decode_fp16_1s_fixed"
WINDOWED_DREAMVAE_DECODE_NAME = "dreamvae_decode_fp16_1s_fixed"
WINDOWED_VAE_PROFILE_FRAMES: tuple[int, int, int] = (25, 25, 25)
# Keep ("wire slice") range in seconds. The decode is always 1 s; this
# bounds how much of the middle we emit. Upper bound 0.36 s (9 frames)
# keeps the per-side margin at >= 8 frames (0.32 s), the converged floor.
WINDOWED_VAE_WINDOW_RANGE_S: tuple[float, float] = (0.04, 0.36)


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


# ------------------------------------------------------------------
# Windowed VAE encode — a FIXED 5-second engine (min == opt == max ==
# 240000 samples = 5 s @ 48 kHz). The runtime-input ("play into the
# model") path selects it opportunistically, mirroring how the windowed
# decode engine above is swapped in when ``vae_window > 0``.
#
# Each live re-encode feeds this engine exactly 240000 samples (5 s) and
# keeps the middle frames; the leftover frames on each side are the
# encoder's receptive-field margin (~7 frames measured; 10 frames /
# 0.4 s each side is floor-exact — see
# scripts/experiments/realtime_input/encoder_receptive_field.py). A 5 s
# window therefore yields 105 usable center frames (4.2 s) per call.
#
# Why fixed 5 s rather than a 1 s engine like the decode side:
#   * sub-5 s encode shapes are a Myelin trap on TRT 10.16 / RTX 5090.
#     At builder opt level 1 (the documented workaround level for the
#     ranged 5-60 s encoder) they emit a broken fusion that access-
#     violates (0xC0000005, hard process crash) at context creation or
#     the first execute. Opt level 0 builds run but its kernels are
#     ~7x slower per sample and batching does not amortize it.
#   * A 5 s window sits inside the shape range where the level-1 build
#     is proven (the ranged 5-60 s engine runs it daily), so it is the
#     smallest safe fixed encode profile. Do NOT shrink it below 5 s.
# See scripts/experiments/realtime_input/build_windowed_encoder.py.
#
# The profile is recorded in the engine's .metadata.json, so changing
# these samples invalidates the sidecar and forces a rebuild.
# ------------------------------------------------------------------

WINDOWED_VAE_ENCODE_NAME = "vae_encode_fp16_5s_fixed"
# Fixed audio-sample profile (min == opt == max): 5 s @ 48 kHz.
WINDOWED_VAE_ENCODE_PROFILE_SAMPLES: tuple[int, int, int] = (240000, 240000, 240000)
# Encoder receptive-field margin trimmed off each side of a re-encoded
# window before splicing its center (measured ~7 frames; 10 frames /
# 0.4 s is floor-exact). A 125-frame (5 s) window therefore yields
# 125 - 2*10 = 105 usable center frames (4.2 s) per call.
WINDOWED_VAE_ENCODE_MARGIN_FRAMES: int = 10


def windowed_vae_encode_engine_name() -> str:
    """Engine directory/file stem for the windowed VAE encode engine."""
    return WINDOWED_VAE_ENCODE_NAME


def windowed_vae_encode_engine_path() -> Path:
    """Path to the windowed VAE encode engine. Pure: does not check
    existence. Use :func:`available_windowed_vae_encode_engine` for
    existence-aware lookup."""
    return trt_engine_path(windowed_vae_encode_engine_name())


def available_windowed_vae_encode_engine() -> Path | None:
    """Return the windowed VAE encode engine path if it is built, else None.

    The real-time-input splice path uses this to opportunistically swap
    in the small fixed-profile encode engine, falling back silently to
    whatever ranged ``vae_encode`` engine the caller already loaded.
    """
    p = windowed_vae_encode_engine_path()
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
