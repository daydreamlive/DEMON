"""Hugging Face source for ONNX exports consumed by the TRT build.

The local ``trt_engines/_onnx_*`` layout is expensive to recreate — the
decoder export alone takes minutes and pulls a full model load — so we
mirror it on Hugging Face and let the build pipeline opt into fetching
instead of re-exporting on machines that don't have the model
checkpoint or the patience.

Two repos are involved:
* ``daydreamlive/demon-onnx`` — VAE + decoder ONNX, per-checkpoint
  subdirectories. Decoder ONNX uses external data (graph + many weight
  files) so we use :func:`huggingface_hub.snapshot_download` with
  allow_patterns to grab whole subtrees atomically.
* ``daydreamlive/DreamVAE`` — distilled student decoder. Single ONNX
  file; we keep this fetch pointed at the public dreamvae release so
  end-users using DreamVAE outside this repo see one canonical home.

Local layout produced by this module is identical to what the local
ONNX exporter writes, so existing :func:`_ensure_onnx` callers don't
need to know whether the file came from HF or a local export.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

from loguru import logger


# ------------------------------------------------------------------
# Repo configuration
# ------------------------------------------------------------------

DEMON_ONNX_REPO = "daydreamlive/demon-onnx"
DREAMVAE_ONNX_REPO = "daydreamlive/DreamVAE"


@dataclass(frozen=True)
class _OnnxSource:
    """How to fetch one ONNX component from HF.

    repo:           HF repo id.
    hf_glob:        Pattern passed to ``snapshot_download(allow_patterns=...)``.
                    Supports ``{checkpoint}`` substitution.
    hf_main_file:   The .onnx file's path inside the repo (also supports
                    ``{checkpoint}``); used for fetch + upload routing.
    local_subdir:   Target subdirectory under the local ONNX root, also
                    supports ``{checkpoint}``. Layout mirrors what the
                    local exporter would produce.
    local_filename: Optional override for the local filename. Defaults
                    to the basename of ``hf_main_file``. Set this when
                    the HF file name differs from what the local
                    exporter writes — e.g. dreamvae's ``model.onnx``
                    on HF maps to ``dreamvae_decode.onnx`` locally to
                    match the rest of the build pipeline's naming.
    """
    repo: str
    hf_glob: str
    hf_main_file: str
    local_subdir: str
    local_filename: str | None = None

    def local_basename(self, ctx: dict) -> str:
        if self.local_filename:
            return self.local_filename.format(**ctx)
        return Path(self.hf_main_file.format(**ctx)).name


# Components map to where their ONNX lives on HF and on disk. ``decoder``
# and ``decoder_refit`` are checkpoint-specific (the DiT weights differ
# across acestep-v15-base / sft / turbo / xl-turbo); the VAE and
# DreamVAE are shared across DiT variants.
_ONNX_REGISTRY: dict[str, _OnnxSource] = {
    "vae_encode": _OnnxSource(
        repo=DEMON_ONNX_REPO,
        hf_glob="vae/vae_encode/*",
        hf_main_file="vae/vae_encode/vae_encode.onnx",
        local_subdir="_onnx_vae/vae_encode",
    ),
    "vae_decode": _OnnxSource(
        repo=DEMON_ONNX_REPO,
        hf_glob="vae/vae_decode/*",
        hf_main_file="vae/vae_decode/vae_decode.onnx",
        local_subdir="_onnx_vae/vae_decode",
    ),
    "decoder": _OnnxSource(
        repo=DEMON_ONNX_REPO,
        hf_glob="decoders/{checkpoint}/decoder/*",
        hf_main_file="decoders/{checkpoint}/decoder/decoder.onnx",
        local_subdir="_onnx_{checkpoint}/decoder",
    ),
    "decoder_refit": _OnnxSource(
        repo=DEMON_ONNX_REPO,
        hf_glob="decoders/{checkpoint}/decoder_refit/*",
        hf_main_file="decoders/{checkpoint}/decoder_refit/decoder_refit.onnx",
        local_subdir="_onnx_{checkpoint}/decoder_refit",
    ),
    "dreamvae": _OnnxSource(
        repo=DREAMVAE_ONNX_REPO,
        hf_glob="onnx/model.onnx",
        hf_main_file="onnx/model.onnx",
        # DreamVAE keeps its own ONNX cache name so the build pipeline's
        # registry doesn't collide with the standard vae_decode dir.
        local_subdir="_onnx_dreamvae",
        # HF's generic ``model.onnx`` is renamed locally to match the
        # rest of the build pipeline's naming (vae_decode.onnx etc.).
        local_filename="dreamvae_decode.onnx",
    ),
}


def known_components() -> tuple[str, ...]:
    return tuple(_ONNX_REGISTRY.keys())


# ------------------------------------------------------------------
# Staleness inspection
# ------------------------------------------------------------------

def decoder_onnx_has_steering(path: Union[str, Path]) -> bool:
    """Whether a decoder ONNX carries the 'steering' graph input.

    Spectral steering added this input; export.py's engine build hard-
    fails on ONNX exported before it. Checking the graph proto here
    (external weight data is not loaded; ~ms on the 2 MB proto) lets
    the build's ONNX resolver treat a pre-steering file as stale and
    replace it, instead of failing deep in the TRT build with no
    recovery path. Unreadable files count as stale for the same reason.
    """
    import onnx

    try:
        model = onnx.load(str(path), load_external_data=False)
    except Exception as exc:
        logger.warning("Could not inspect decoder ONNX at {}: {}", path, exc)
        return False
    return any(i.name == "steering" for i in model.graph.input)


def probe_onnx_main_file(
    component: str,
    *,
    checkpoint: Optional[str] = None,
) -> Path:
    """Fetch ONLY the component's main ``.onnx`` into the HF cache.

    Cheap freshness probe for the multi-file decoder components: the
    graph proto is ~2 MB while the external-weight siblings are
    multi-GB. hf_hub_download is etag-aware, so a repeat probe against
    an unchanged repo costs a couple of metadata requests, not a
    download. Returns the cached file path; writes nothing into the
    local ONNX tree.
    """
    if component not in _ONNX_REGISTRY:
        raise ValueError(
            f"Unknown ONNX component: {component!r}. "
            f"Known: {known_components()}"
        )
    source = _ONNX_REGISTRY[component]
    ctx = {"checkpoint": checkpoint} if checkpoint else {}
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(
        repo_id=source.repo,
        filename=source.hf_main_file.format(**ctx),
    ))


# ------------------------------------------------------------------
# Fetch
# ------------------------------------------------------------------

def fetch_onnx(
    component: str,
    *,
    local_root: Union[str, Path],
    checkpoint: Optional[str] = None,
    force_download: bool = False,
) -> Path:
    """Ensure ``component``'s ONNX is present locally; download from HF if not.

    Args:
        component: One of :func:`known_components` (e.g. ``"vae_decode"``,
            ``"decoder_refit"``, ``"dreamvae"``).
        local_root: TRT engines root (``trt_engines/``) — the ONNX is
            placed at ``local_root / <component-specific subdir>``.
        checkpoint: Required for ``decoder*`` components, ignored for
            shared components (VAE, DreamVAE).
        force_download: If True, re-fetch even when local files exist.

    Returns the path to the main ``.onnx`` file. External-data files
    (the decoder's many ``.weight`` / ``.bias`` siblings) are placed
    next to it so the ONNX parser resolves them without further work.
    """
    if component not in _ONNX_REGISTRY:
        raise ValueError(
            f"Unknown ONNX component: {component!r}. "
            f"Known: {known_components()}"
        )
    source = _ONNX_REGISTRY[component]
    if "{checkpoint}" in (source.hf_glob + source.hf_main_file + source.local_subdir):
        if not checkpoint:
            raise ValueError(
                f"component={component!r} is checkpoint-specific; "
                f"pass checkpoint=..."
            )
        ctx = {"checkpoint": checkpoint}
    else:
        ctx = {}

    local_root = Path(local_root)
    target_dir = local_root / source.local_subdir.format(**ctx)
    main_filename = source.local_basename(ctx)
    target_main = target_dir / main_filename

    if target_main.exists() and not force_download:
        logger.info("Reusing local ONNX at {}", target_main)
        return target_main

    target_dir.mkdir(parents=True, exist_ok=True)

    # Single-file fetches (VAE, DreamVAE) → hf_hub_download is faster
    # and less chatty than snapshot_download. Multi-file (decoder*) →
    # snapshot_download with allow_patterns grabs the .onnx + every
    # external-data sibling in one go.
    is_single_file = "*" not in source.hf_glob
    from huggingface_hub import hf_hub_download, snapshot_download

    logger.info(
        "Fetching ONNX {!r} from HF: {}/{}",
        component, source.repo, source.hf_main_file.format(**ctx),
    )
    if is_single_file:
        cached_main = Path(hf_hub_download(
            repo_id=source.repo,
            filename=source.hf_main_file.format(**ctx),
            force_download=force_download,
        ))
        shutil.copy2(cached_main, target_main)
    else:
        snap_dir = Path(snapshot_download(
            repo_id=source.repo,
            allow_patterns=[source.hf_glob.format(**ctx)],
            force_download=force_download,
        ))
        cached_main = snap_dir / source.hf_main_file.format(**ctx)
        # Copy the entire external-data dir, not just the .onnx — the
        # ONNX parser resolves external weights relative to the .onnx
        # path so siblings must travel together.
        cached_dir = cached_main.parent
        for f in cached_dir.iterdir():
            if f.is_file():
                shutil.copy2(f, target_dir / f.name)

    size_mb = target_main.stat().st_size / (1 << 20)
    logger.info("ONNX {!r} ready at {} ({:.1f} MB)", component, target_main, size_mb)
    return target_main
