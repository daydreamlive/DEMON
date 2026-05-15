#!/usr/bin/env python3
"""Build TensorRT engines for the ACE-Step decoder and VAE.

Single entry point for all TRT engine creation.  Supports building individual
engines (fine-grained control) or the full matrix across durations.

ONNX exports are duration-agnostic and stored in a shared trt_engines/_onnx/
directory.  Existing ONNX files are auto-detected and reused; the model is
only loaded when an ONNX export is actually needed.

Usage:
    # Build the canonical engine matrix (60s + 120s + 240s, VAE + decoder
    # refit-only). Matches acestep.paths._TRT_ENGINE_PROFILES.
    python -m acestep.engine.trt.build --all

    # Build a single duration (e.g. just 120s):
    python -m acestep.engine.trt.build --all --duration 120

    # Build a custom subset:
    python -m acestep.engine.trt.build --all --duration 60 240

    # Build only decoders (skip VAE):
    python -m acestep.engine.trt.build --all --decoder-only

    # Preview what will be built:
    python -m acestep.engine.trt.build --all --dry-run

    # Force rebuild (existing engines are skipped by default):
    python -m acestep.engine.trt.build --all --force-rebuild

    # Single engine (fine-grained control):
    python -m acestep.engine.trt.build --max-duration 60
    python -m acestep.engine.trt.build --skip-vae --decoder --decoder-mixed --decoder-refit --max-duration 240

Requirements:
    - tensorrt (uv pip install tensorrt)
    - ACE-Step model checkpoint at checkpoints/acestep-v15-turbo
"""

import argparse
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
import time
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from importlib import metadata as importlib_metadata
from pathlib import Path

# Suppress flash_attn import (not needed for export)
import importlib, importlib.util
_orig = importlib.util.find_spec
def _patch(name, *a, **k):
    if "flash_attn" in str(name):
        return None
    return _orig(name, *a, **k)
importlib.util.find_spec = _patch

from loguru import logger
import torch

_SUPPORTED_TRT_MIN = (10, 16)
_SUPPORTED_TRT_MAX = (10, 17)
_ENGINE_METADATA_SCHEMA = 1
_DECODER_PRECISION_CHOICES = (
    "auto",
    "fp32",
    "fp16_mixed",
    "bf16_mixed",
    "fp8_mixed",
)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _default_trt_dir() -> str:
    """Default TRT engine directory from acestep.paths."""
    # Import lazily to avoid circular deps at module level
    from acestep.paths import trt_engines_dir
    return str(trt_engines_dir())


def _default_checkpoints_dir() -> str:
    """Default checkpoints directory from acestep.paths."""
    from acestep.paths import checkpoints_dir
    return str(checkpoints_dir())


def _parse_version_tuple(version: str) -> tuple[int, ...]:
    """Extract a comparable numeric prefix from versions like 10.16.1.11."""
    return tuple(int(p) for p in re.findall(r"\d+", version)[:3])


def _dist_version(name: str) -> str | None:
    try:
        return importlib_metadata.version(name)
    except importlib_metadata.PackageNotFoundError:
        return None


def _nvidia_smi_summary() -> dict:
    """Best-effort driver/GPU snapshot for engine metadata."""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,compute_cap",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
    except Exception as exc:
        return {"available": False, "error": str(exc)}

    gpus = []
    for line in result.stdout.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 3:
            gpus.append({
                "name": parts[0],
                "driver_version": parts[1],
                "compute_capability": parts[2],
            })
    return {"available": True, "gpus": gpus}


def _active_gpu_summary(device: str) -> dict:
    if not torch.cuda.is_available():
        return {"available": False}

    torch_device = torch.device(device)
    index = torch_device.index if torch_device.index is not None else torch.cuda.current_device()
    props = torch.cuda.get_device_properties(index)
    return {
        "available": True,
        "index": index,
        "name": props.name,
        "compute_capability": f"{props.major}.{props.minor}",
        "total_memory_bytes": props.total_memory,
    }


def _preflight(device: str) -> dict:
    """Validate and log the TensorRT/CUDA stack before building engines."""
    import tensorrt as trt

    trt_version = trt.__version__
    parsed = _parse_version_tuple(trt_version)
    if parsed < _SUPPORTED_TRT_MIN or parsed >= _SUPPORTED_TRT_MAX:
        raise RuntimeError(
            "DEMON TensorRT builds target TensorRT >=10.16,<10.17; "
            f"found {trt_version}. Run `uv sync --upgrade-package tensorrt`."
        )

    env = {
        "schema_version": _ENGINE_METADATA_SCHEMA,
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "python": platform.python_version(),
        },
        "packages": {
            "tensorrt": trt_version,
            "tensorrt_cu13": _dist_version("tensorrt-cu13"),
            "tensorrt_cu13_bindings": _dist_version("tensorrt-cu13-bindings"),
            "tensorrt_cu13_libs": _dist_version("tensorrt-cu13-libs"),
            "cuda_python": _dist_version("cuda-python"),
            "cuda_toolkit": _dist_version("cuda-toolkit"),
            "polygraphy": _dist_version("polygraphy"),
            "onnx": _dist_version("onnx"),
            "torch": torch.__version__,
        },
        "torch_cuda": torch.version.cuda,
        "onnx_parser_version": getattr(trt, "get_nv_onnx_parser_version", lambda: None)(),
        "active_gpu": _active_gpu_summary(device),
        "nvidia_smi": _nvidia_smi_summary(),
    }

    logger.info("=" * 60)
    logger.info("TensorRT build preflight")
    logger.info("=" * 60)
    logger.info("TensorRT: {}", env["packages"]["tensorrt"])
    logger.info("TensorRT cu13: {}", env["packages"]["tensorrt_cu13"])
    logger.info("CUDA Python: {}", env["packages"]["cuda_python"])
    logger.info("CUDA toolkit wheel: {}", env["packages"]["cuda_toolkit"])
    logger.info("Polygraphy: {}", env["packages"]["polygraphy"])
    logger.info("ONNX: {}", env["packages"]["onnx"])
    logger.info("Torch: {} (CUDA {})", env["packages"]["torch"], env["torch_cuda"])
    gpu = env["active_gpu"]
    if gpu.get("available"):
        logger.info(
            "Active GPU: cuda:{} {} (SM {})",
            gpu["index"], gpu["name"], gpu["compute_capability"],
        )
    else:
        logger.warning("No active CUDA GPU detected in torch")
    return env


def _sha256_file(path: str | os.PathLike[str]) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _config_dict(config) -> dict:
    if is_dataclass(config):
        return asdict(config)
    return dict(vars(config))


def _looks_like_xl_checkpoint(checkpoint: str) -> bool:
    return "xl" in os.path.basename(checkpoint).lower()


def _resolve_decoder_precision(
    *,
    checkpoint: str,
    requested: str,
    decoder_mixed: bool,
) -> str:
    """Resolve the decoder ONNX precision recipe for this checkpoint.

    ``decoder_mixed`` preserves the legacy 2B behavior: when callers ask
    for the old mixed path, ``auto`` maps to the fp16 mixed export recipe.
    XL checkpoints need bf16 range, so their ``auto`` default is bf16_mixed.
    """
    if requested != "auto":
        return requested
    if _looks_like_xl_checkpoint(checkpoint):
        return "bf16_mixed"
    if decoder_mixed:
        return "fp16_mixed"
    return "fp32"


def _decoder_precision_is_strongly_typed(decoder_precision: str, decoder_mixed: bool) -> bool:
    if decoder_precision == "fp16_mixed":
        return True
    if decoder_precision == "bf16_mixed":
        return True
    if decoder_precision == "fp8_mixed":
        return True
    return decoder_mixed


def _decoder_onnx_needs_dynbatch_patch(
    *,
    checkpoint: str,
    decoder_precision: str,
    batch_max: int,
) -> bool:
    # FP8 reuses the bf16 ONNX as its base; same dynbatch concern applies.
    needs_bf16_base = decoder_precision in ("bf16_mixed", "fp8_mixed")
    return (
        batch_max > 1
        and _looks_like_xl_checkpoint(checkpoint)
        and needs_bf16_base
    )


def _patch_decoder_onnx_for_dynamic_batch(
    onnx_paths: dict[str, str],
    *,
    need_decoder_std: bool,
    need_decoder_refit: bool,
    checkpoint: str,
    decoder_precision: str,
    batch_max: int,
    force: bool,
) -> dict[str, str]:
    """Return ONNX paths, replacing XL decoder ONNX with dynbatch-patched copies."""
    if not _decoder_onnx_needs_dynbatch_patch(
        checkpoint=checkpoint,
        decoder_precision=decoder_precision,
        batch_max=batch_max,
    ):
        return onnx_paths

    from .export import patch_decoder_onnx_dynamic_batch_reshapes

    patched = dict(onnx_paths)
    for key, needed in (
        ("decoder", need_decoder_std),
        ("decoder_refit", need_decoder_refit),
    ):
        if not needed:
            continue
        patched[key] = str(
            patch_decoder_onnx_dynamic_batch_reshapes(
                patched[key],
                force=force,
            )
        )
    return patched


def _patch_decoder_onnx_for_fp8(
    onnx_paths: dict[str, str],
    *,
    need_decoder_std: bool,
    need_decoder_refit: bool,
    decoder_precision: str,
    calibration_npz: str,
    activation_absmax_json: str | None,
    activation_percentile: str,
    smoothquant_alpha: float,
    force: bool,
) -> dict[str, str]:
    """Insert FP8 QDQ into decoder ONNX(es) when decoder_precision='fp8_mixed'.

    Runs after the dynbatch reshape patch so the FP8 graph inherits the
    dynamic-batch-safe reshapes. The patched ONNX is a sibling of the
    source; this function rewrites the paths dict to point at it.

    When ``activation_absmax_json`` is provided, the patch runs in W8A8
    mode (activation Q->DQ inserted alongside weight DQ); otherwise it
    runs in weight-only W8A16 mode.
    """
    if decoder_precision != "fp8_mixed":
        return onnx_paths

    cal_path = None
    if calibration_npz:
        cal_path = Path(calibration_npz)
        if not cal_path.exists():
            raise FileNotFoundError(f"Calibration .npz not found: {cal_path}")

    amax_path = None
    if activation_absmax_json:
        amax_path = Path(activation_absmax_json)
        if not amax_path.exists():
            raise FileNotFoundError(f"Activation absmax JSON not found: {amax_path}")

    from .fp8_onnx import patch_bf16_onnx_to_fp8

    patched = dict(onnx_paths)
    for key, needed in (
        ("decoder", need_decoder_std),
        ("decoder_refit", need_decoder_refit),
    ):
        if not needed:
            continue
        patched[key] = str(
            patch_bf16_onnx_to_fp8(
                bf16_onnx_path=patched[key],
                calibration_npz_path=cal_path,
                activation_absmax_json_path=amax_path,
                activation_percentile=activation_percentile,
                smoothquant_alpha=smoothquant_alpha,
                force=force,
            )
        )
    return patched


def _metadata_path(engine_path: str | os.PathLike[str]) -> Path:
    return Path(str(engine_path) + ".metadata.json")


def _expected_metadata(
    *,
    component: str,
    onnx_path: str,
    config,
    env: dict,
) -> dict:
    gpu = env.get("active_gpu", {})
    return {
        "schema_version": _ENGINE_METADATA_SCHEMA,
        "component": component,
        "tensorrt_version": env["packages"]["tensorrt"],
        "gpu_compute_capability": gpu.get("compute_capability"),
        "gpu_name": gpu.get("name"),
        "config": _config_dict(config),
        "onnx_path": str(Path(onnx_path).resolve()),
        "onnx_sha256": _sha256_file(onnx_path),
    }


def _write_metadata(
    *,
    engine_path: str,
    expected: dict,
    env: dict,
) -> None:
    payload = dict(expected)
    payload["built_at"] = datetime.now(timezone.utc).isoformat()
    payload["environment"] = env
    path = _metadata_path(engine_path)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    logger.info("Engine metadata saved to {}", path)


def _metadata_matches(engine_path: str, expected: dict) -> tuple[bool, str]:
    path = _metadata_path(engine_path)
    if not path.exists():
        return False, "missing metadata"
    try:
        actual = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return False, f"metadata unreadable: {exc}"

    for key in (
        "schema_version",
        "component",
        "tensorrt_version",
        "gpu_compute_capability",
        "config",
        "onnx_sha256",
    ):
        if actual.get(key) != expected.get(key):
            return False, f"metadata mismatch: {key}"
    return True, "metadata match"


def _verify_engines(engine_paths: list[tuple[str, str]]):
    """Load and print I/O info for each engine."""
    import tensorrt as trt

    rt = trt.Runtime(trt.Logger(trt.Logger.WARNING))
    for name, path in engine_paths:
        if not os.path.exists(path):
            logger.error("  {}: MISSING ({})", name, path)
            continue
        with open(path, "rb") as f:
            engine = rt.deserialize_cuda_engine(f.read())
        if engine is None:
            logger.error("  {}: FAILED to load", name)
            continue

        io_info = []
        for i in range(engine.num_io_tensors):
            tname = engine.get_tensor_name(i)
            mode = engine.get_tensor_mode(tname)
            shape = engine.get_tensor_shape(tname)
            label = "IN" if mode == trt.TensorIOMode.INPUT else "OUT"
            io_info.append(f"{label}: {tname} {shape}")

        profiles = []
        for i in range(engine.num_io_tensors):
            tname = engine.get_tensor_name(i)
            if engine.get_tensor_mode(tname) == trt.TensorIOMode.INPUT:
                shapes = engine.get_tensor_profile_shape(tname, 0)
                profiles.append(f"{tname}: min={shapes[0]} opt={shapes[1]} max={shapes[2]}")

        size_mb = os.path.getsize(path) / 1e6
        logger.info("  {}: OK ({:.1f} MB)", name, size_mb)
        for s in io_info:
            logger.info("    {}", s)
        for s in profiles:
            logger.info("    Profile: {}", s)


def _engine_path(output_dir: str, engine_filename: str) -> str:
    """Resolve engine path: trt_engines/<name>/<name>.engine."""
    name = engine_filename.replace(".engine", "")
    return os.path.join(output_dir, name, engine_filename)


# ------------------------------------------------------------------
# ONNX setup
# ------------------------------------------------------------------


def _ensure_onnx(
    *,
    onnx_dir: str,
    project_root: str,
    checkpoint: str,
    device: str,
    need_vae: bool,
    need_decoder_std: bool,
    need_decoder_refit: bool,
    decoder_precision: str,
    skip_onnx: bool,
    force_onnx: bool = False,
    export_locally: bool = False,
) -> dict[str, str]:
    """Resolve ONNX paths for the build, fetching from HF when missing.

    Resolution order for each needed component:
      1. Local cache present -> reuse, unless ``force_onnx`` is set.
      2. ``skip_onnx``       -> error (no fetch, no export).
      3. ``force_onnx``      -> re-export from the model checkpoint.
      4. ``export_locally``  -> export missing files from the checkpoint.
      5. Default             -> download missing files from HF via ``onnx_hub``.

    HF-first is the clean default: machines that don't have the model
    checkpoint can build engines without ever touching torch's model
    loader, and CI / cloud runs skip a multi-minute model load. Pass
    ``--export-locally`` to recover the old behavior, e.g. when iterating
    on the export code itself.

    VAE ONNX is stored in a shared ``_onnx_vae/`` directory (sibling to
    ``onnx_dir``) since all DiT variants share the same VAE. Decoder
    ONNX lives in ``onnx_dir`` (checkpoint-specific).
    """
    from .onnx_hub import fetch_onnx

    # VAE is shared across checkpoints; decoder is checkpoint-specific
    vae_onnx_dir = os.path.join(os.path.dirname(onnx_dir), "_onnx_vae")
    os.makedirs(vae_onnx_dir, exist_ok=True)

    paths = {
        "vae_encode": os.path.join(vae_onnx_dir, "vae_encode", "vae_encode.onnx"),
        "vae_decode": os.path.join(vae_onnx_dir, "vae_decode", "vae_decode.onnx"),
        "decoder": os.path.join(onnx_dir, "decoder", "decoder.onnx"),
        "decoder_refit": os.path.join(onnx_dir, "decoder_refit", "decoder_refit.onnx"),
    }

    # Also check old _onnx/ location for VAE (backward compat).
    old_onnx_dir = os.path.join(os.path.dirname(onnx_dir), "_onnx")
    for key in ("vae_encode", "vae_decode"):
        if not os.path.exists(paths[key]):
            old_path = os.path.join(old_onnx_dir, key, f"{key}.onnx")
            if os.path.exists(old_path):
                logger.info("Found VAE ONNX at old location: {}", old_path)
                paths[key] = old_path

    # Build the list of (component, fetch_kwargs) pairs that the caller
    # actually needs and that aren't yet on disk. With --force-onnx, all
    # requested components are included so the checkpoint exporter overwrites
    # stale or suspect ONNX files.
    requested: list[tuple[str, dict]] = []
    if need_vae:
        if force_onnx or not os.path.exists(paths["vae_encode"]):
            requested.append(("vae_encode", {}))
        else:
            logger.info("Reusing existing VAE encoder ONNX: {}", paths["vae_encode"])
        if force_onnx or not os.path.exists(paths["vae_decode"]):
            requested.append(("vae_decode", {}))
        else:
            logger.info("Reusing existing VAE decoder ONNX: {}", paths["vae_decode"])
    if need_decoder_std and (force_onnx or not os.path.exists(paths["decoder"])):
        requested.append(("decoder", {"checkpoint": checkpoint}))
    elif need_decoder_std:
        logger.info("Reusing existing decoder ONNX: {}", paths["decoder"])
    if need_decoder_refit and (force_onnx or not os.path.exists(paths["decoder_refit"])):
        requested.append(("decoder_refit", {"checkpoint": checkpoint}))
    elif need_decoder_refit:
        logger.info("Reusing existing decoder ONNX (refit): {}", paths["decoder_refit"])

    # --skip-onnx: no resolution path. Error if anything's missing.
    if skip_onnx:
        if requested:
            for comp, _ in requested:
                logger.error(
                    "Missing ONNX file (refusing to fetch/export with --skip-onnx): {}",
                    paths[comp],
                )
            sys.exit(1)
        logger.info("All ONNX exports found, --skip-onnx satisfied.")
        return paths

    if not requested:
        logger.info("All ONNX exports already present, nothing to fetch or export.")
        return paths

    # HF-first path. Each fetch lands the file at the same local path
    # the local exporter would write, so the rest of the pipeline
    # downstream doesn't care about the source.
    if not export_locally and not force_onnx:
        local_root = os.path.dirname(onnx_dir)  # the trt_engines dir
        try:
            for comp, kw in requested:
                fetched = fetch_onnx(comp, local_root=local_root, **kw)
                # fetch_onnx returns the canonical local path; the
                # ``paths`` dict already points at the same location, but
                # update it in case the registry's local_subdir ever
                # diverges from this function's hardcoded layout.
                paths[comp] = str(fetched)
            return paths
        except Exception as exc:
            logger.error(
                "ONNX fetch from HuggingFace failed: {}. "
                "Re-run with --export-locally to export from the model "
                "checkpoint instead, or with --skip-onnx if you have the "
                "files in a non-standard location.",
                exc,
            )
            sys.exit(1)

    # --export-locally / --force-onnx path: load the model and export from weights.
    export_vae = any(c.startswith("vae_") for c, _ in requested)
    export_decoder_refit = any(c == "decoder_refit" for c, _ in requested)
    export_decoder_std = any(c == "decoder" for c, _ in requested)

    mode = "--force-onnx" if force_onnx else "--export-locally"
    logger.info("Loading model from checkpoints/{} ({})...", checkpoint, mode)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    from acestep.engine.model_context import ModelContext
    handler = ModelContext(
        project_root=project_root,
        config_path=checkpoint,
        device=device,
        use_flash_attention=False,
        compile_decoder=False,
        compile_vae=False,
        skip_vae=not export_vae,
    )
    logger.info("Model loaded.")

    if export_vae:
        from .vae_export import (
            export_vae_encoder_onnx, export_vae_decoder_onnx, VAEExportConfig,
        )
        logger.info("=" * 60)
        logger.info("VAE ONNX EXPORT")
        logger.info("=" * 60)
        with handler._load_model_context("vae"):
            t0 = time.time()
            export_vae_encoder_onnx(
                handler.vae, paths["vae_encode"], device=device,
                config=VAEExportConfig(trace_audio_samples=48000 * 30),
            )
            logger.info("VAE encoder exported in {:.1f}s", time.time() - t0)

            t0 = time.time()
            export_vae_decoder_onnx(
                handler.vae, paths["vae_decode"], device=device,
                config=VAEExportConfig(trace_latent_frames=750),
            )
            logger.info("VAE decoder exported in {:.1f}s", time.time() - t0)

    if export_decoder_refit or export_decoder_std:
        from .export import OnnxExportConfig, export_decoder_onnx

        # fp8_mixed reuses the bf16_mixed ONNX as its export base; the
        # FP8 QDQ patch runs afterwards in a separate step.
        underlying_precision = (
            "bf16_mixed" if decoder_precision == "fp8_mixed"
            else decoder_precision
        )

        def decoder_export_config(*, for_refit: bool) -> OnnxExportConfig:
            if underlying_precision == "fp16_mixed":
                return OnnxExportConfig(mixed_precision=True, for_refit=for_refit)
            return OnnxExportConfig(
                precision=underlying_precision,
                mixed_precision=False,
                for_refit=for_refit,
            )

        with handler._load_model_context("model"):
            if export_decoder_refit:
                logger.info("=" * 60)
                logger.info(
                    "DECODER ONNX EXPORT (refit-enabled, precision={})",
                    decoder_precision,
                )
                logger.info("=" * 60)
                t0 = time.time()
                export_decoder_onnx(
                    handler.model, paths["decoder_refit"], device=device,
                    config=decoder_export_config(for_refit=True),
                )
                logger.info("Decoder ONNX (refit) exported in {:.1f}s", time.time() - t0)

            if export_decoder_std:
                logger.info("=" * 60)
                logger.info(
                    "DECODER ONNX EXPORT (standard, precision={})",
                    decoder_precision,
                )
                logger.info("=" * 60)
                t0 = time.time()
                export_decoder_onnx(
                    handler.model, paths["decoder"], device=device,
                    config=decoder_export_config(for_refit=False),
                )
                logger.info("Decoder ONNX (standard) exported in {:.1f}s", time.time() - t0)

    # Free model memory before TRT builds
    del handler
    torch.cuda.empty_cache()

    return paths


# ------------------------------------------------------------------
# Engine builders
# ------------------------------------------------------------------

def _build_vae_engines(
    *,
    output_dir: str,
    onnx_paths: dict[str, str],
    duration: int,
    workspace_gb: float,
    env: dict,
    force_rebuild: bool = False,
) -> list[tuple[str, str, float, str]]:
    """Build VAE encode + decode TRT engines for one duration.

    Returns list of (label, engine_path, elapsed_seconds, status).
    Existing engines are skipped unless force_rebuild is True.
    """
    from .vae_export import (
        build_vae_decode_engine, build_vae_encode_engine, VAETRTBuildConfig,
    )

    config = VAETRTBuildConfig(
        workspace_gb=workspace_gb,
        decode_max_frames=duration * 25,
        encode_max_samples=duration * 48000,
    )

    results = []
    for component, builder in [
        ("vae_decode", build_vae_decode_engine),
        ("vae_encode", build_vae_encode_engine),
    ]:
        name = config.engine_filename(component).replace(".engine", "")
        engine_dir = os.path.join(output_dir, name)
        engine_path = os.path.join(engine_dir, f"{name}.engine")

        label = f"VAE {component.split('_')[1]} {duration}s"
        expected_metadata = _expected_metadata(
            component=component,
            onnx_path=onnx_paths[component],
            config=config,
            env=env,
        )

        if not force_rebuild and os.path.exists(engine_path):
            matches, reason = _metadata_matches(engine_path, expected_metadata)
            if matches:
                size_mb = os.path.getsize(engine_path) / 1e6
                logger.info("SKIP {} ({:.0f} MB, {})", name, size_mb, reason)
                results.append((label, engine_path, 0.0, "SKIPPED"))
                continue
            logger.info("REBUILD {} ({})", name, reason)

        logger.info("=" * 60)
        logger.info("VAE TRT BUILD: {} (max_duration={}s)", name, duration)
        logger.info("=" * 60)

        t0 = time.time()
        builder(onnx_paths[component], engine_path, config=config)
        _write_metadata(engine_path=engine_path, expected=expected_metadata, env=env)
        elapsed = time.time() - t0
        logger.info("Built in {:.0f}s", elapsed)
        results.append((label, engine_path, elapsed, "OK"))

    return results


def _build_windowed_vae_decode_engine(
    *,
    output_dir: str,
    onnx_paths: dict[str, str],
    workspace_gb: float,
    force_rebuild: bool = False,
) -> tuple[str, str, float, str]:
    """Build the single windowed (3-30 s) VAE decode engine.

    This profile is shared across all duration tiers — it's selected
    by the runtime when ``vae_window > 0`` regardless of the song
    length, because the chunk size fed to the engine is bounded by
    the user-facing window (5-30 s) plus overlap, never by the full
    song duration. Building it costs ~75 s and saves ~7.7 GB of TRT
    workspace at session-creation time vs the canonical 240 s engine.
    """
    from .vae_export import build_vae_decode_engine, VAETRTBuildConfig
    from acestep.paths import (
        WINDOWED_VAE_DECODE_NAME,
        WINDOWED_VAE_PROFILE_FRAMES,
    )

    name = WINDOWED_VAE_DECODE_NAME
    engine_dir = os.path.join(output_dir, name)
    engine_path = os.path.join(engine_dir, f"{name}.engine")
    label = "VAE decode windowed (3-30s)"

    if not force_rebuild and os.path.exists(engine_path):
        size_mb = os.path.getsize(engine_path) / 1e6
        logger.info("SKIP {} ({:.0f} MB)", name, size_mb)
        return (label, engine_path, 0.0, "SKIPPED")

    min_f, opt_f, max_f = WINDOWED_VAE_PROFILE_FRAMES
    config = VAETRTBuildConfig(
        workspace_gb=workspace_gb,
        decode_min_frames=min_f,
        decode_opt_frames=opt_f,
        decode_max_frames=max_f,
    )

    logger.info("=" * 60)
    logger.info("VAE TRT BUILD (windowed): {} (min={} opt={} max={})",
                name, min_f, opt_f, max_f)
    logger.info("=" * 60)

    t0 = time.time()
    build_vae_decode_engine(onnx_paths["vae_decode"], engine_path, config=config)
    elapsed = time.time() - t0
    logger.info("Built in {:.0f}s", elapsed)
    return (label, engine_path, elapsed, "OK")


def _checkpoint_to_variant(checkpoint: str) -> str:
    """Extract short variant name from checkpoint path.

    'acestep-v15-turbo' -> 'turbo'
    'acestep-v15-base'  -> 'base'
    'acestep-v15-sft'   -> 'sft'
    """
    name = os.path.basename(checkpoint)
    # Strip the common 'acestep-v15-' prefix
    prefix = "acestep-v15-"
    if name.startswith(prefix):
        return name[len(prefix):]
    return name


def _build_decoder_engine(
    *,
    output_dir: str,
    onnx_paths: dict[str, str],
    duration: int,
    mixed: bool,
    refit: bool,
    workspace_gb: float,
    batch_max: int,
    env: dict,
    batch_opt: int | None = None,
    builder_optimization_level: int | None = None,
    force_rebuild: bool = False,
    checkpoint: str = "acestep-v15-turbo",
    decoder_precision: str = "fp16_mixed",
    strongly_typed: bool = True,
) -> tuple[str, str, float, str]:
    """Build one decoder TRT engine.

    Returns (label, engine_path, elapsed_seconds, status).
    Existing engines are skipped unless force_rebuild is True.
    """
    from .export import build_trt_engine, TRTBuildConfig

    variant = _checkpoint_to_variant(checkpoint)
    config = TRTBuildConfig(
        fp16=True,
        strongly_typed=strongly_typed,
        refit=refit,
        workspace_gb=workspace_gb,
        batch_max=batch_max,
        seq_opt=min(duration * 25, 1500),
        seq_max=duration * 25,
        variant=variant,
        onnx_precision=decoder_precision,
    )
    if batch_opt is not None:
        config.batch_opt = batch_opt
    if builder_optimization_level is not None:
        config.builder_optimization_level = builder_optimization_level

    name = config.engine_filename().replace(".engine", "")
    engine_dir = os.path.join(output_dir, name)
    engine_path = os.path.join(engine_dir, f"{name}.engine")

    onnx_key = "decoder_refit" if refit else "decoder"
    refit_label = "refit" if refit else "no-refit"
    label = f"Decoder {variant} {duration}s, {refit_label}"
    expected_metadata = _expected_metadata(
        component=onnx_key,
        onnx_path=onnx_paths[onnx_key],
        config=config,
        env=env,
    )

    if not force_rebuild and os.path.exists(engine_path):
        matches, reason = _metadata_matches(engine_path, expected_metadata)
        if matches:
            size_mb = os.path.getsize(engine_path) / 1e6
            logger.info("SKIP {} ({:.0f} MB, {})", name, size_mb, reason)
            return (label, engine_path, 0.0, "SKIPPED")
        logger.info("REBUILD {} ({})", name, reason)

    logger.info("=" * 60)
    logger.info(
        "DECODER TRT BUILD (refit={}, mixed={}, precision={}) -> {}",
        refit, mixed, decoder_precision, engine_path,
    )
    logger.info("=" * 60)

    t0 = time.time()
    build_trt_engine(onnx_paths[onnx_key], engine_path, config=config)
    _write_metadata(engine_path=engine_path, expected=expected_metadata, env=env)
    elapsed = time.time() - t0
    logger.info("Built in {:.0f}s", elapsed)

    return (label, engine_path, elapsed, "OK")


# ------------------------------------------------------------------
# Batch mode (--all)
# ------------------------------------------------------------------

def _print_matrix(durations, build_vae, build_decoder, output_dir, batch_max,
                   checkpoint="acestep-v15-turbo", build_dreamvae=False):
    """Print the build matrix for --all mode, showing existing vs new."""
    variant = _checkpoint_to_variant(checkpoint)
    vtag = f"_{variant}" if variant != "turbo" else ""

    from acestep.paths import (
        WINDOWED_VAE_DECODE_NAME,
        WINDOWED_DREAMVAE_DECODE_NAME,
    )

    # (label, engine_dir_name) pairs
    jobs = []
    for dur in durations:
        if build_vae:
            jobs.append((f"VAE decode {dur}s", f"vae_decode_fp16_{dur}s"))
            jobs.append((f"VAE encode {dur}s", f"vae_encode_fp16_{dur}s"))
        if build_decoder:
            jobs.append((f"Decoder {variant} {dur}s, refit", f"decoder{vtag}_mixed_refit_b{batch_max}_{dur}s"))
        if build_dreamvae:
            jobs.append((f"DreamVAE decode {dur}s", f"dreamvae_decode_fp16_{dur}s"))

    # Windowed engines are duration-independent (single shared 3-30s
    # profile) so they're appended once, outside the duration loop.
    if build_vae:
        jobs.append(("VAE decode windowed (3-30s)", WINDOWED_VAE_DECODE_NAME))
    if build_dreamvae:
        jobs.append(("DreamVAE decode windowed (3-30s)", WINDOWED_DREAMVAE_DECODE_NAME))

    to_build = 0
    to_skip = 0
    lines = []
    for label, dir_name in jobs:
        engine_file = os.path.join(output_dir, dir_name, f"{dir_name}.engine")
        if os.path.exists(engine_file):
            size_mb = os.path.getsize(engine_file) / 1e6
            lines.append(f"  [exists]  {label}  ({size_mb:.0f} MB)")
            to_skip += 1
        else:
            lines.append(f"  [build]   {label}")
            to_build += 1

    print(f"\nBuild matrix: {to_build} to build, {to_skip} existing (skipped)")
    for line in lines:
        print(line)
    print()
    return jobs


def _print_summary(results, output_dir):
    """Print build summary and list engines on disk."""
    print(f"\n{'=' * 60}")
    print("BUILD SUMMARY")
    print(f"{'=' * 60}")
    for label, path, elapsed, status in results:
        print(f"  {status:7s} {elapsed:6.0f}s  {label}")

    failures = sum(1 for _, _, _, s in results if s == "FAILED")
    if failures:
        print(f"\n{failures} build(s) FAILED")
    else:
        active = sum(1 for _, _, _, s in results if s != "SKIPPED")
        skipped = sum(1 for _, _, _, s in results if s == "SKIPPED")
        parts = [f"{active} built"]
        if skipped:
            parts.append(f"{skipped} skipped")
        print(f"\nAll done ({', '.join(parts)}).")

    # List engine files on disk
    from pathlib import Path
    trt_dir = Path(output_dir)
    print(f"\nEngines in {trt_dir}:")
    for d in sorted(trt_dir.iterdir()):
        if not d.is_dir() or d.name.startswith("_"):
            continue
        engine_file = d / f"{d.name}.engine"
        if engine_file.exists():
            size_mb = engine_file.stat().st_size / 1e6
            print(f"  {d.name + '/':50s} {size_mb:8.1f} MB")

    return failures


def _save_build_report(results, output_dir):
    """Append CSV build report to trt_engines/build_report.csv."""
    import csv
    from datetime import datetime

    report_path = os.path.join(output_dir, "build_report.csv")
    write_header = not os.path.exists(report_path)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with open(report_path, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["timestamp", "engine", "status", "build_time_s", "size_mb"])
        for label, path, elapsed, status in results:
            size_mb = os.path.getsize(path) / 1e6 if os.path.exists(path) else -1
            writer.writerow([timestamp, label, status, f"{elapsed:.1f}", f"{size_mb:.1f}"])

    print(f"Build report appended to: {report_path}")


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Build ACE-Step TRT engines",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Batch mode
    batch = parser.add_argument_group("batch mode (--all)")
    batch.add_argument("--all", action="store_true",
                       help="Build full engine matrix (VAE + refit-only "
                            "decoder, across durations)")
    batch.add_argument("--duration", nargs="*", type=int, default=None,
                       help="Duration(s) in seconds for --all mode "
                            "(default: 60 120 240 — the canonical profile set "
                            "registered in acestep.paths._TRT_ENGINE_PROFILES)")
    batch.add_argument("--force-rebuild", action="store_true",
                       help="Rebuild engines even if they already exist "
                            "(default: skip existing engines)")
    batch.add_argument("--dry-run", action="store_true",
                       help="Print build matrix without building")
    batch.add_argument("--decoder-only", action="store_true",
                       help="Only build decoder engines (skip VAE)")
    batch.add_argument("--vae-only", action="store_true",
                       help="Only build VAE engines (skip decoder)")
    batch.add_argument("--with-dreamvae", action="store_true",
                       help="Also build dreamvae (distilled decoder) engines "
                            "for each duration. ONNX is fetched from "
                            "huggingface.co/daydreamlive/DreamVAE on first use.")
    batch.add_argument("--dreamvae-only", action="store_true",
                       help="Build ONLY dreamvae engines (skip standard "
                            "VAE/decoder builds). Implies --with-dreamvae.")

    # Shared / single mode
    single = parser.add_argument_group("single mode / shared options")
    single.add_argument("--output-dir",
                        default=_default_trt_dir(),
                        help="Directory for ONNX and engine files "
                             "(default: ~/.daydream-scope/models/demon/trt_engines)")
    single.add_argument("--checkpoint", default="acestep-v15-turbo",
                        help="Model checkpoint directory name")
    single.add_argument("--skip-onnx", action="store_true",
                        help="Don't fetch or export ONNX. Error if any "
                             "needed ONNX file is missing locally.")
    single.add_argument("--force-onnx", action="store_true",
                        help="Re-export ONNX files from the model checkpoint "
                             "even if matching files already exist. Implies "
                             "--export-locally.")
    single.add_argument("--export-locally", action="store_true",
                        help="Re-export ONNX from the model checkpoint "
                             "instead of fetching from HuggingFace. The "
                             "default is HF-first; use this when iterating "
                             "on the export code or when offline.")
    single.add_argument("--max-duration", type=int, default=240,
                        help="Max audio duration in seconds for single mode "
                             "(default: 240 = 4min)")
    single.add_argument("--device", default="cuda")
    single.add_argument("--workspace-gb", type=float, default=16.0,
                        help="TRT builder workspace in GB (default: 16)")
    single.add_argument("--decoder", action="store_true",
                        help="Build decoder engine(s)")
    single.add_argument("--decoder-mixed", action="store_true",
                        help="Build the bf16-hybrid decoder recipe "
                             "(bf16 trunk + fp32 islands + per-Linear fp16 "
                             "wrappers + fp32 proj_out deconv, "
                             "strongly_typed=True)")
    single.add_argument("--decoder-precision",
                        choices=_DECODER_PRECISION_CHOICES,
                        default="auto",
                        help="Decoder ONNX export precision recipe. "
                             "'auto' keeps the legacy fp16_mixed recipe for "
                             "2B mixed builds and uses bf16_mixed for XL "
                             "checkpoints.")
    single.add_argument("--decoder-refit",
                        action=argparse.BooleanOptionalAction, default=True,
                        help="Build refit-enabled decoder for LoRA "
                             "(default: True, use --no-decoder-refit)")
    single.add_argument("--batch-max", type=int, default=8,
                        help="Max batch size for decoder (default: 8)")
    single.add_argument("--batch-opt", type=int, default=None,
                        help="Optimal batch size for decoder TRT profile "
                             "(default: TRTBuildConfig default)")
    single.add_argument("--builder-optimization-level", type=int, default=None,
                        help="TensorRT builder optimization level, 0-5 "
                             "(default: TRTBuildConfig default)")
    single.add_argument("--skip-vae", action="store_true",
                        help="Skip VAE engine build")

    fp8 = parser.add_argument_group("fp8_mixed options")
    fp8.add_argument("--calibration-npz", type=str, default=None,
                     help="Path to calibration .npz (currently informational; "
                          "the FP8 patcher infers stats from "
                          "--activation-absmax-json).")
    fp8.add_argument("--activation-absmax-json", type=str, default=None,
                     help="Per-Linear activation absmax JSON from "
                          "scripts/collect_activation_absmax.py. When set, "
                          "fp8_mixed runs in W8A8 mode (activation Q->DQ + "
                          "weight DQ); when omitted, fp8_mixed stays in "
                          "weight-only W8A16.")
    fp8.add_argument("--activation-percentile",
                     choices=("absmax", "p99", "p99_9", "p99_99"),
                     default="absmax",
                     help="Which activation amax field drives the FP8 scale "
                          "(default: absmax).")
    fp8.add_argument("--smoothquant-alpha", type=float, default=0.0,
                     help="SmoothQuant migration strength (0.0 = off, 0.5 = "
                          "standard).")

    args = parser.parse_args()
    if args.skip_onnx and args.force_onnx:
        parser.error("--skip-onnx and --force-onnx are mutually exclusive")
    if args.batch_opt is not None and args.batch_opt < 1:
        parser.error("--batch-opt must be >= 1")
    if args.batch_opt is not None and args.batch_opt > args.batch_max:
        parser.error("--batch-opt must be <= --batch-max")
    if (
        args.builder_optimization_level is not None
        and not 0 <= args.builder_optimization_level <= 5
    ):
        parser.error("--builder-optimization-level must be between 0 and 5")
    if args.force_onnx:
        args.export_locally = True

    checkpoints_root = _default_checkpoints_dir()
    env = None if args.dry_run else _preflight(args.device)

    os.makedirs(args.output_dir, exist_ok=True)
    # ONNX directory is checkpoint-specific for decoder (different weights)
    # but shared for VAE (same weights across all DiT variants).
    onnx_dir = os.path.join(args.output_dir, f"_onnx_{args.checkpoint}")
    os.makedirs(onnx_dir, exist_ok=True)

    if args.all:
        _run_all(args, checkpoints_root, onnx_dir, env)
    else:
        _run_single(args, checkpoints_root, onnx_dir, env)


def _run_all(args, project_root, onnx_dir, env):
    """Build the full engine matrix."""
    durations = tuple(args.duration) if args.duration else (60, 120, 240)
    decoder_precision = _resolve_decoder_precision(
        checkpoint=args.checkpoint,
        requested=args.decoder_precision,
        decoder_mixed=True,
    )
    decoder_strongly_typed = _decoder_precision_is_strongly_typed(
        decoder_precision, decoder_mixed=True,
    )
    # --dreamvae-only is shorthand for "skip standard VAE/decoder, only
    # build dreamvae". --with-dreamvae adds dreamvae on top of the
    # standard build. Both forms enable the dreamvae build.
    build_dreamvae = args.with_dreamvae or args.dreamvae_only
    if args.dreamvae_only:
        build_vae = False
        build_decoder = False
    else:
        build_vae = not args.decoder_only
        build_decoder = not args.vae_only

    # Print matrix
    _print_matrix(durations, build_vae, build_decoder,
                  args.output_dir, args.batch_max, args.checkpoint,
                  build_dreamvae=build_dreamvae)

    if args.dry_run:
        return

    # ONNX phase (once, shared across all durations).  Only refit-enabled
    # decoder ONNX is needed for the standard build; dreamvae has its
    # own ONNX, fetched lazily from HF inside _build_dreamvae_engines so
    # we don't pay network cost when --dreamvae isn't requested.
    if build_vae or build_decoder:
        onnx_paths = _ensure_onnx(
            onnx_dir=onnx_dir,
            project_root=project_root,
            checkpoint=args.checkpoint,
            device=args.device,
            need_vae=build_vae,
            need_decoder_std=False,
            need_decoder_refit=build_decoder,
            decoder_precision=decoder_precision,
            skip_onnx=args.skip_onnx,
            force_onnx=args.force_onnx,
            export_locally=args.export_locally,
        )
        onnx_paths = _patch_decoder_onnx_for_dynamic_batch(
            onnx_paths,
            need_decoder_std=False,
            need_decoder_refit=build_decoder,
            checkpoint=args.checkpoint,
            decoder_precision=decoder_precision,
            batch_max=args.batch_max,
            force=args.force_onnx,
        )
        onnx_paths = _patch_decoder_onnx_for_fp8(
            onnx_paths,
            need_decoder_std=False,
            need_decoder_refit=build_decoder,
            decoder_precision=decoder_precision,
            calibration_npz=args.calibration_npz,
            activation_absmax_json=args.activation_absmax_json,
            activation_percentile=args.activation_percentile,
            smoothquant_alpha=args.smoothquant_alpha,
            force=args.force_onnx,
        )
    else:
        onnx_paths = {}

    # Engine phase
    results = []
    for dur in durations:
        if build_vae:
            results.extend(_build_vae_engines(
                output_dir=args.output_dir,
                onnx_paths=onnx_paths,
                duration=dur,
                workspace_gb=args.workspace_gb,
                env=env,
                force_rebuild=args.force_rebuild,
            ))
        if build_decoder:
            results.append(_build_decoder_engine(
                output_dir=args.output_dir,
                onnx_paths=onnx_paths,
                duration=dur,
                mixed=True,
                refit=True,
                workspace_gb=args.workspace_gb,
                batch_max=args.batch_max,
                batch_opt=args.batch_opt,
                builder_optimization_level=args.builder_optimization_level,
                env=env,
                force_rebuild=args.force_rebuild,
                checkpoint=args.checkpoint,
                decoder_precision=decoder_precision,
                strongly_typed=decoder_strongly_typed,
            ))

    # Windowed VAE decode (single 3-30s profile, duration-independent).
    # Auto-selected by Session when vae_window > 0.
    if build_vae:
        results.append(_build_windowed_vae_decode_engine(
            output_dir=args.output_dir,
            onnx_paths=onnx_paths,
            workspace_gb=args.workspace_gb,
            force_rebuild=args.force_rebuild,
        ))

    if build_dreamvae:
        # dreamvae fetches its own ONNX from HF on first use; no shared
        # ONNX state with the loop above. Built last so a missing HF
        # token / network error doesn't tank an otherwise-successful
        # standard build.
        from .dreamvae_export import (
            build_dreamvae_engines,
            build_windowed_dreamvae_engine,
        )
        results.extend(build_dreamvae_engines(
            output_dir=args.output_dir,
            durations=durations,
            workspace_gb=args.workspace_gb,
            force_rebuild=args.force_rebuild,
        ))
        results.append(build_windowed_dreamvae_engine(
            output_dir=args.output_dir,
            workspace_gb=args.workspace_gb,
            force_rebuild=args.force_rebuild,
        ))

    # Summary
    failures = _print_summary(results, args.output_dir)
    _save_build_report(results, args.output_dir)

    if failures:
        sys.exit(1)


def _run_single(args, project_root, onnx_dir, env):
    """Build a single engine configuration."""
    build_vae = not args.skip_vae
    build_decoder = args.decoder
    decoder_precision = _resolve_decoder_precision(
        checkpoint=args.checkpoint,
        requested=args.decoder_precision,
        decoder_mixed=args.decoder_mixed,
    )
    decoder_strongly_typed = _decoder_precision_is_strongly_typed(
        decoder_precision, decoder_mixed=args.decoder_mixed,
    )

    # ONNX phase
    onnx_paths = _ensure_onnx(
        onnx_dir=onnx_dir,
        project_root=project_root,
        checkpoint=args.checkpoint,
        device=args.device,
        need_vae=build_vae,
        need_decoder_std=build_decoder and not args.decoder_refit,
        need_decoder_refit=build_decoder and args.decoder_refit,
        decoder_precision=decoder_precision,
        skip_onnx=args.skip_onnx,
        force_onnx=args.force_onnx,
        export_locally=args.export_locally,
    )
    onnx_paths = _patch_decoder_onnx_for_dynamic_batch(
        onnx_paths,
        need_decoder_std=build_decoder and not args.decoder_refit,
        need_decoder_refit=build_decoder and args.decoder_refit,
        checkpoint=args.checkpoint,
        decoder_precision=decoder_precision,
        batch_max=args.batch_max,
        force=args.force_onnx,
    )
    onnx_paths = _patch_decoder_onnx_for_fp8(
        onnx_paths,
        need_decoder_std=build_decoder and not args.decoder_refit,
        need_decoder_refit=build_decoder and args.decoder_refit,
        decoder_precision=decoder_precision,
        calibration_npz=args.calibration_npz,
        activation_absmax_json=args.activation_absmax_json,
        activation_percentile=args.activation_percentile,
        smoothquant_alpha=args.smoothquant_alpha,
        force=args.force_onnx,
    )

    # Engine phase
    built_engines = []

    if build_vae:
        results = _build_vae_engines(
            output_dir=args.output_dir,
            onnx_paths=onnx_paths,
            duration=args.max_duration,
            workspace_gb=args.workspace_gb,
            env=env,
            force_rebuild=args.force_rebuild,
        )
        for label, path, elapsed, status in results:
            if status == "OK":
                built_engines.append((label, path))

    if build_decoder:
        result = _build_decoder_engine(
            output_dir=args.output_dir,
            onnx_paths=onnx_paths,
            duration=args.max_duration,
            mixed=args.decoder_mixed,
            refit=args.decoder_refit,
            workspace_gb=args.workspace_gb,
            batch_max=args.batch_max,
            batch_opt=args.batch_opt,
            builder_optimization_level=args.builder_optimization_level,
            env=env,
            force_rebuild=args.force_rebuild,
            checkpoint=args.checkpoint,
            decoder_precision=decoder_precision,
            strongly_typed=decoder_strongly_typed,
        )
        label, path, elapsed, status = result
        if status == "OK":
            built_engines.append((label, path))

    # Verify
    if built_engines:
        logger.info("=" * 60)
        logger.info("VERIFICATION")
        logger.info("=" * 60)
        _verify_engines(built_engines)

    logger.info("=" * 60)
    logger.info("Built {} engine(s):", len(built_engines))
    for name, path in built_engines:
        logger.info("  {} -> {}", name, path)
    logger.info("Output directory: {}", args.output_dir)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
