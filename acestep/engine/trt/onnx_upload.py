"""Upload locally-exported ONNX trees to Hugging Face.

Companion to :mod:`acestep.engine.trt.onnx_hub` (which fetches): this
module pushes the local ``trt_engines/_onnx_*`` layout into the HF
repo(s) the registry points at, so other machines can fetch instead of
re-exporting.

CLI:
    # Preview which files would be uploaded for one component
    python -m acestep.engine.trt.onnx_upload --component vae_decode --dry-run

    # Upload the standard ONNX bundle for a checkpoint
    python -m acestep.engine.trt.onnx_upload \\
        --components vae_encode vae_decode decoder decoder_refit \\
        --checkpoint acestep-v15-turbo

The CLI does NOT default to "upload everything" — every invocation must
name the components explicitly. Uploads are public-facing and slow
(the decoder ONNX is ~3 GB across hundreds of files), and we don't want
a stray invocation pushing a half-baked export.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

from loguru import logger

from .onnx_hub import _ONNX_REGISTRY, known_components


def upload_onnx(
    component: str,
    *,
    local_root: Path,
    checkpoint: str | None = None,
    repo_id: str | None = None,
    commit_message: str | None = None,
    dry_run: bool = False,
) -> None:
    """Upload one component's ONNX dir to HF.

    Args:
        component: One of :func:`known_components`.
        local_root: TRT engines root (``trt_engines/``) — same root the
            local exporter writes to.
        checkpoint: Required for ``decoder*`` components.
        repo_id: Override the registry's repo for this upload (rare;
            useful for staging to a fork before the public repo exists).
        commit_message: HF commit message; auto-generated if omitted.
        dry_run: Print the plan, don't actually upload.
    """
    if component not in _ONNX_REGISTRY:
        raise ValueError(
            f"Unknown component: {component!r}. Known: {known_components()}"
        )
    source = _ONNX_REGISTRY[component]
    needs_checkpoint = "{checkpoint}" in (
        source.local_subdir + source.hf_main_file + source.hf_glob
    )
    if needs_checkpoint and not checkpoint:
        raise ValueError(
            f"component={component!r} is checkpoint-specific; pass checkpoint=..."
        )
    ctx = {"checkpoint": checkpoint} if needs_checkpoint else {}

    repo = repo_id or source.repo
    local_dir = Path(local_root) / source.local_subdir.format(**ctx)
    main_file = local_dir / source.local_basename(ctx)
    if not main_file.exists():
        raise FileNotFoundError(
            f"Missing local ONNX for {component!r}: {main_file}"
        )

    hf_main = source.hf_main_file.format(**ctx)
    is_single_file = "*" not in source.hf_glob

    if is_single_file:
        # One file, possibly renamed across local/HF (e.g. dreamvae's
        # local `dreamvae_decode.onnx` -> HF `onnx/model.onnx`). Use
        # upload_file with explicit path_in_repo so the rename sticks.
        size_mb = main_file.stat().st_size / (1 << 20)
        logger.info(
            "Upload plan: {} -> {}/{} ({:.1f} MB)",
            component, repo, hf_main, size_mb,
        )
        logger.info("  {} -> {}", main_file.name, Path(hf_main).name)
        if dry_run:
            logger.info("--dry-run: not uploading")
            return
        from huggingface_hub import HfApi
        api = HfApi()
        msg = commit_message or f"Upload {component} ONNX ({size_mb:.0f} MB)"
        api.upload_file(
            repo_id=repo,
            path_or_fileobj=str(main_file),
            path_in_repo=hf_main,
            commit_message=msg,
        )
    else:
        # External-data layout: graph + many sibling weight files. The
        # local dir's contents mirror the HF subtree 1:1, so upload_folder
        # is the right call. ``path_in_repo`` is the parent of hf_main.
        repo_dir = str(Path(hf_main).parent).replace("\\", "/")
        files = sorted(local_dir.iterdir())
        total_mb = sum(f.stat().st_size for f in files if f.is_file()) / (1 << 20)
        logger.info(
            "Upload plan: {} -> {}/{} ({} files, {:.1f} MB)",
            component, repo, repo_dir, len(files), total_mb,
        )
        for f in files[:10]:
            logger.info("  {}", f.name)
        if len(files) > 10:
            logger.info("  ... {} more", len(files) - 10)
        if dry_run:
            logger.info("--dry-run: not uploading")
            return
        from huggingface_hub import HfApi
        api = HfApi()
        msg = commit_message or f"Upload {component} ONNX ({len(files)} files, {total_mb:.0f} MB)"
        api.upload_folder(
            repo_id=repo,
            folder_path=str(local_dir),
            path_in_repo=repo_dir,
            commit_message=msg,
        )

    logger.info("Uploaded {} to {}", component, repo)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Upload local ONNX exports to Hugging Face",
    )
    parser.add_argument(
        "--components", nargs="+", required=True,
        choices=list(known_components()),
        help="Which components to upload.",
    )
    parser.add_argument(
        "--checkpoint", default=None,
        help="Required for decoder/decoder_refit (ignored for shared "
             "components like vae_encode / vae_decode / dreamvae).",
    )
    parser.add_argument(
        "--local-root", default=None,
        help="TRT engines directory (default: from acestep.paths.trt_engines_dir()).",
    )
    parser.add_argument(
        "--repo", default=None,
        help="Override the registry repo (rare; for staging).",
    )
    parser.add_argument(
        "--commit-message", default=None,
        help="HF commit message (auto-generated if omitted).",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print the plan without uploading.",
    )
    args = parser.parse_args()

    if args.local_root is None:
        from acestep.paths import trt_engines_dir
        local_root = trt_engines_dir()
    else:
        local_root = Path(args.local_root)

    for component in args.components:
        upload_onnx(
            component,
            local_root=local_root,
            checkpoint=args.checkpoint,
            repo_id=args.repo,
            commit_message=args.commit_message,
            dry_run=args.dry_run,
        )


if __name__ == "__main__":
    main()
