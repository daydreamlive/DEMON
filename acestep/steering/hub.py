"""HF source for activation-steering vector bundles.

Mirrors :mod:`acestep.engine.trt.onnx_hub`. Local layout matches
:func:`acestep.paths.steering_vector_dir` so a fresh fetch and a
warm cache resolve to the same path.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Optional

from loguru import logger

from acestep.engine.trt.onnx_hub import DEMON_ONNX_REPO
from acestep.paths import steering_bundle_subpath, steering_vectors_dir


# Steering bundles share the demon-onnx repo today; alias kept so a
# future split to ``daydreamlive/demon-steering`` is a one-line change.
DEMON_STEERING_REPO = DEMON_ONNX_REPO
_HF_PREFIX = "steering_vectors"


def hf_bundle_dir(subpath: str) -> str:
    """The HF in-repo path for a bundle subpath (no trailing slash)."""
    return f"{_HF_PREFIX}/{subpath}"


def ensure_steering_vectors(
    checkpoint: Optional[str] = None,
    *,
    force_download: bool = False,
) -> Optional[Path]:
    """Ensure the checkpoint's bundle is on disk; return its cache dir.

    Returns ``None`` when no bundle is registered for the checkpoint
    (XL today). HF errors are best-effort — they log a warning and
    return the (possibly empty) cache dir so the session keeps booting.
    """
    subpath = steering_bundle_subpath(checkpoint)
    if subpath is None:
        return None

    target_dir = steering_vectors_dir() / subpath
    # Cache-hit heuristic: directory has at least one .pt file. We
    # don't compare against an HF manifest because the bundle is
    # small and the operator can force_download=True to refresh.
    if not force_download and target_dir.is_dir():
        if any(target_dir.glob("*.pt")):
            return target_dir

    target_dir.mkdir(parents=True, exist_ok=True)
    hf_dir = hf_bundle_dir(subpath)

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        logger.warning(
            "steering: huggingface_hub not available ({}); "
            "steering knobs will no-op until vectors are installed at {}.",
            exc, target_dir,
        )
        return target_dir

    logger.info(
        "Fetching steering bundle {!r} from HF: {}/{}",
        subpath, DEMON_STEERING_REPO, hf_dir,
    )
    try:
        snap_dir = Path(snapshot_download(
            repo_id=DEMON_STEERING_REPO,
            allow_patterns=[f"{hf_dir}/*"],
            force_download=force_download,
        ))
    except Exception as exc:
        logger.warning(
            "steering: HF fetch failed ({}); steering knobs will no-op. "
            "Cache dir: {}",
            exc, target_dir,
        )
        return target_dir

    src_dir = snap_dir / hf_dir
    if not src_dir.is_dir():
        logger.warning(
            "steering: HF bundle dir {} not present in snapshot at {}; "
            "steering knobs will no-op.",
            hf_dir, snap_dir,
        )
        return target_dir

    copied = 0
    for f in src_dir.iterdir():
        if f.is_file():
            shutil.copy2(f, target_dir / f.name)
            copied += 1
    logger.info(
        "Steering bundle ready at {} ({} files)",
        target_dir, copied,
    )
    return target_dir


def upload_steering_vectors(
    checkpoint: str,
    *,
    local_dir: Optional[Path] = None,
    repo: str = DEMON_STEERING_REPO,
    commit_message: Optional[str] = None,
    dry_run: bool = False,
) -> None:
    """Upload a local steering bundle to HF (operator-facing)."""
    subpath = steering_bundle_subpath(checkpoint)
    if subpath is None:
        raise ValueError(
            f"No steering bundle registered for checkpoint {checkpoint!r}. "
            f"Add it to _STEERING_VECTORS_BY_CHECKPOINT in acestep/paths.py."
        )

    src = Path(local_dir) if local_dir is not None else steering_vectors_dir() / subpath
    if not src.is_dir():
        raise FileNotFoundError(f"Local steering dir not found: {src}")

    files = sorted(p for p in src.iterdir() if p.is_file())
    if not files:
        raise FileNotFoundError(f"Local steering dir is empty: {src}")

    total_mb = sum(p.stat().st_size for p in files) / (1 << 20)
    hf_dir = hf_bundle_dir(subpath)
    logger.info(
        "Upload plan: {} -> {}/{} ({} files, {:.1f} MB)",
        subpath, repo, hf_dir, len(files), total_mb,
    )
    for f in files[:10]:
        logger.info("  {}", f.name)
    if len(files) > 10:
        logger.info("  ... {} more", len(files) - 10)

    if dry_run:
        logger.info("--dry-run: not uploading")
        return

    from huggingface_hub import HfApi
    msg = commit_message or (
        f"Upload steering bundle {subpath} ({len(files)} files, {total_mb:.0f} MB)"
    )
    HfApi().upload_folder(
        repo_id=repo,
        folder_path=str(src),
        path_in_repo=hf_dir,
        commit_message=msg,
    )
    logger.info("Uploaded steering bundle {} to {}/{}", subpath, repo, hf_dir)
