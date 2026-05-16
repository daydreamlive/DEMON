"""Hugging Face source for FP8 activation calibration JSONs.

The XL FP8 decoder build is gated on a per-Linear activation absmax JSON
(``--activation-absmax-json``). Capturing one is a separate offline step
(``scripts/collect_activation_absmax.py``) that needs a calibrated bf16
engine and ~10 minutes of GPU. We mirror the JSON on HF so a fresh box
can build the canonical XL FP8 engine without first running calibration
locally.

Layout mirrors :mod:`onnx_hub`: the demon-onnx repo gets a
``calibrations/`` subtree parallel to its existing ``decoders/`` and
``vae/`` subtrees. The local on-disk layout matches what
``collect_activation_absmax.py --output-dir`` writes, so a build
invocation against an already-staged calibration is a no-op fetch.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

from loguru import logger


DEMON_ONNX_REPO = "daydreamlive/demon-onnx"

# Files within a calibration directory. activation_absmax.json is the
# build input; manifest.json records the calibration provenance (model,
# duration, sample count) and is fetched alongside for traceability.
_CALIBRATION_FILES = ("activation_absmax.json", "manifest.json")


@dataclass(frozen=True)
class _CalibrationSource:
    """How to fetch one calibration component from HF.

    repo:               HF repo id.
    hf_subdir:          Repo subtree, e.g. ``calibrations/decoder_xl_fp8``.
                        Supports ``{profile}`` substitution.
    local_subdir:       Target subdirectory under ``local_root``, mirrors
                        what ``collect_activation_absmax.py`` writes.
                        Supports ``{profile}`` substitution.
    """
    repo: str
    hf_subdir: str
    local_subdir: str


_REGISTRY: dict[str, _CalibrationSource] = {
    "decoder_xl_fp8": _CalibrationSource(
        repo=DEMON_ONNX_REPO,
        hf_subdir="calibrations/decoder_xl_fp8/{profile}",
        local_subdir="decoder_xl_fp8/{profile}",
    ),
}


def known_components() -> tuple[str, ...]:
    return tuple(_REGISTRY.keys())


def fetch_calibration(
    component: str,
    *,
    profile: str,
    local_root: Union[str, Path],
    force_download: bool = False,
) -> Path:
    """Ensure ``component``/``profile`` calibration is present locally.

    Args:
        component: One of :func:`known_components`. Currently only
            ``"decoder_xl_fp8"``.
        profile: Per-profile subdir, e.g. ``"60s"`` / ``"120s"`` /
            ``"240s"``. Matches what
            ``scripts/collect_activation_absmax.py`` writes.
        local_root: Calibration root, typically
            ``~/.daydream-scope/models/demon/calibration``.
        force_download: Re-fetch even when local files exist.

    Returns the path to ``activation_absmax.json``. ``manifest.json``
    lands as a sibling.
    """
    if component not in _REGISTRY:
        raise ValueError(
            f"Unknown calibration component: {component!r}. "
            f"Known: {known_components()}"
        )
    source = _REGISTRY[component]
    ctx = {"profile": profile}

    local_root = Path(local_root)
    target_dir = local_root / source.local_subdir.format(**ctx)
    target_main = target_dir / "activation_absmax.json"

    if target_main.exists() and not force_download:
        logger.info("Reusing local calibration at {}", target_main)
        return target_main

    target_dir.mkdir(parents=True, exist_ok=True)

    from huggingface_hub import hf_hub_download

    hf_dir = source.hf_subdir.format(**ctx)
    logger.info(
        "Fetching calibration {!r}/{!r} from HF: {}/{}",
        component, profile, source.repo, hf_dir,
    )
    import shutil
    for fname in _CALIBRATION_FILES:
        cached = Path(hf_hub_download(
            repo_id=source.repo,
            filename=f"{hf_dir}/{fname}",
            force_download=force_download,
        ))
        shutil.copy2(cached, target_dir / fname)

    size_mb = target_main.stat().st_size / (1 << 20)
    logger.info(
        "Calibration {!r}/{!r} ready at {} ({:.1f} MB)",
        component, profile, target_main, size_mb,
    )
    return target_main


def resolve_or_fetch(
    requested_path: Union[str, Path],
    *,
    force_download: bool = False,
) -> Optional[Path]:
    """If ``requested_path`` is missing on disk, fetch from HF by parsing
    its structure: ``.../<component>/<profile>/activation_absmax.json``.

    Returns the resolved path (existing or newly fetched), or ``None``
    if the path doesn't match the canonical calibration layout (in
    which case the caller should treat it as a normal missing file).
    """
    p = Path(requested_path)
    if p.exists():
        return p

    # Canonical structure: <local_root>/<component>/<profile>/activation_absmax.json
    if p.name != "activation_absmax.json":
        return None
    profile = p.parent.name
    component = p.parent.parent.name
    if component not in _REGISTRY:
        return None

    # ``local_root`` is everything above <component>/<profile>/file.
    local_root = p.parent.parent.parent
    return fetch_calibration(
        component,
        profile=profile,
        local_root=local_root,
        force_download=force_download,
    )
