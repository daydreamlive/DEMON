"""Shared engine-metadata helpers for the TRT engine builders.

Every freshly-built TRT engine writes a sidecar ``<engine>.metadata.json``
that records the build environment (TRT version, GPU compute capability,
ONNX file hash, build config). On rebuild, the existing engine is
compared against the *expected* metadata for the current build env;
mismatch (e.g. TRT version bump via ``uv sync``) triggers a rebuild.

This module exists so every engine builder — standard VAE/decoder
(:func:`acestep.engine.trt.build._build_vae_engines`), windowed VAE
(:func:`acestep.engine.trt.build._build_windowed_vae_decode_engine`),
and the DreamVAE variants in
:mod:`acestep.engine.trt.dreamvae_export` — share the same metadata
contract. Previously the dreamvae/windowed builders skipped engines on
file existence alone (no metadata read, no metadata write), so when
TRT was bumped underneath them the stale engines silently rode into
:warm bakes and FATAL'd at session load:

    IRuntime::deserializeCudaEngine: API Usage Error
      (The engine plan file is not compatible with this version of
       TensorRT, expecting library version <new> got <old>)

Sharing the helpers makes the freshness check trivially correct
everywhere. Avoid circular imports — this module imports nothing
from ``build.py`` or ``dreamvae_export.py``.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path

from loguru import logger


_ENGINE_METADATA_SCHEMA = 1


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


def metadata_path(engine_path: str | os.PathLike[str]) -> Path:
    return Path(str(engine_path) + ".metadata.json")


def expected_metadata(
    *,
    component: str,
    onnx_path: str | os.PathLike[str],
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


def write_metadata(
    *,
    engine_path: str | os.PathLike[str],
    expected: dict,
    env: dict,
) -> None:
    payload = dict(expected)
    payload["built_at"] = datetime.now(timezone.utc).isoformat()
    payload["environment"] = env
    path = metadata_path(engine_path)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    logger.info("Engine metadata saved to {}", path)


def metadata_matches(
    engine_path: str | os.PathLike[str],
    expected: dict,
) -> tuple[bool, str]:
    """Compare on-disk metadata sidecar against ``expected``.

    Returns ``(matches, reason)``. A missing or unreadable sidecar is
    treated as a non-match so the engine is rebuilt and a fresh
    sidecar is written. Compares the keys that govern engine
    compatibility (TRT version, GPU compute capability, build config,
    ONNX content hash); ignores metadata-only fields like ``built_at``.
    """
    path = metadata_path(engine_path)
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
