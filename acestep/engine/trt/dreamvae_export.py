"""TensorRT engine builds for the DreamVAE distilled decoder.

DreamVAE is a distilled student of the ACE-Step Oobleck VAE decoder. It
keeps the same I/O contract — ``latents [B, 64, T] -> audio [B, 2,
1920*T]`` at 48 kHz — so the resulting engines are drop-in replacements
for ``vae_decode_fp16_*`` whenever the demo's ``fast_vae`` path is
enabled.

Weights and pre-exported ONNX live on Hugging Face at
``daydreamlive/DreamVAE``. Engine builds reuse the standard VAE
optimization-profile knobs from :class:`VAETRTBuildConfig`, so the
duration sweep matches ``acestep.paths._TRT_ENGINE_PROFILES`` exactly.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Union

from loguru import logger

from .onnx_hub import fetch_onnx
from .vae_export import VAETRTBuildConfig, build_vae_decode_engine


def dreamvae_engine_filename(duration_s: int) -> str:
    """Engine filename stem for a given duration. Stable, consumed by
    the demo via ``acestep.paths.dreamvae_decode_engine_path``.
    """
    return f"dreamvae_decode_fp16_{int(duration_s)}s"


# ------------------------------------------------------------------
# ONNX acquisition
# ------------------------------------------------------------------

def fetch_dreamvae_onnx(
    output_dir: Union[str, Path],
    *,
    force_download: bool = False,
) -> Path:
    """Ensure the dreamvae ONNX is present in the local trt_engines tree.

    Thin wrapper over :func:`acestep.engine.trt.onnx_hub.fetch_onnx` so
    the dreamvae build path uses the same HF-fetch infrastructure as
    every other ONNX in the build matrix.
    """
    return fetch_onnx(
        "dreamvae", local_root=output_dir, force_download=force_download,
    )


# ------------------------------------------------------------------
# ONNX export (from weights -> ONNX file on disk)
# ------------------------------------------------------------------
#
# This mirrors the historical export at
# ``research_program/vae_distillation/hf_upload/scripts/export_trt.py``
# (commit fa93128) so the source-of-truth for the ONNX export of
# DreamVAE lives in this repo for posterity. The released
# ``daydreamlive/DreamVAE/onnx/model.onnx`` file was produced by that
# same script — re-running this exporter against the released weights
# should produce a byte-identical file under the same torch / opset.

# Trace shape and opset are pinned to match the released ONNX. Changing
# either would change the resulting graph and break the byte-identity
# property: 30 s of audio at 25 latent fps = 750 frames; opset 18 was
# what the original export used, so we keep it.
_DREAMVAE_TRACE_LATENT_FRAMES = 750
_DREAMVAE_ONNX_OPSET = 18


def export_dreamvae_onnx(
    onnx_path: Union[str, Path],
    *,
    repo_id: str = "daydreamlive/DreamVAE",
    device: str = "cuda",
) -> Path:
    """Export FastOobleckDecoder weights to ONNX.

    Loads weights from the public ``daydreamlive/DreamVAE`` HF release
    and runs ``torch.onnx.export`` with the same call shape the
    historical script used. The result should be byte-identical to
    ``daydreamlive/DreamVAE/onnx/model.onnx`` on the same torch +
    opset; the verifier in :func:`verify_dreamvae_onnx_matches_hf`
    can confirm.

    Returns the path the ONNX was written to.
    """
    import torch

    from acestep.models.dreamvae import load_dreamvae_from_hf

    onnx_path = Path(onnx_path)
    onnx_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Loading DreamVAE weights from %s ...", repo_id)
    model = load_dreamvae_from_hf(repo_id=repo_id, device=device, dtype=torch.float32)

    example = torch.randn(
        1, 64, _DREAMVAE_TRACE_LATENT_FRAMES,
        device=device, dtype=torch.float32,
    )

    logger.info("Exporting DreamVAE to ONNX (opset %d, trace_T=%d) -> %s",
                _DREAMVAE_ONNX_OPSET, _DREAMVAE_TRACE_LATENT_FRAMES, onnx_path)
    with torch.no_grad():
        torch.onnx.export(
            model,
            (example,),
            str(onnx_path),
            input_names=["latents"],
            output_names=["audio"],
            dynamic_axes={
                "latents": {0: "batch", 2: "latent_frames"},
                "audio": {0: "batch", 2: "samples"},
            },
            opset_version=_DREAMVAE_ONNX_OPSET,
            do_constant_folding=True,
            dynamo=False,
        )
    size_mb = onnx_path.stat().st_size / (1 << 20)
    logger.info("DreamVAE ONNX written: %s (%.1f MB)", onnx_path, size_mb)
    return onnx_path


def verify_dreamvae_onnx_matches_hf(local_onnx: Union[str, Path]) -> bool:
    """SHA-compare a locally-exported ONNX to the public release.

    Returns True when the local file is byte-identical to
    ``daydreamlive/DreamVAE/onnx/model.onnx``. Logs both hashes either
    way so the caller can read the divergence if there is one.
    """
    import hashlib
    from huggingface_hub import hf_hub_download

    def sha256(path: Path) -> str:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()

    local_onnx = Path(local_onnx)
    remote = Path(hf_hub_download("daydreamlive/DreamVAE", "onnx/model.onnx"))
    a = sha256(local_onnx)
    b = sha256(remote)
    logger.info("Local  : %s  %s", local_onnx, a)
    logger.info("HF     : %s  %s", remote, b)
    return a == b


# ------------------------------------------------------------------
# Engine build
# ------------------------------------------------------------------

def build_dreamvae_engine(
    *,
    output_dir: Union[str, Path],
    duration_s: int,
    onnx_path: Optional[Union[str, Path]] = None,
    workspace_gb: float = 8.0,
    force_rebuild: bool = False,
) -> tuple[Path, str]:
    """Build a single dreamvae TRT engine for one duration.

    Returns ``(engine_path, status)`` where status is ``"OK"`` if the
    engine was built this call, or ``"SKIPPED"`` if it already existed.
    """
    output_dir = Path(output_dir)
    name = dreamvae_engine_filename(duration_s)
    engine_path = output_dir / name / f"{name}.engine"

    if engine_path.exists() and not force_rebuild:
        size_mb = engine_path.stat().st_size / (1 << 20)
        logger.info("SKIP %s (%.0f MB)", name, size_mb)
        return engine_path, "SKIPPED"

    if onnx_path is None:
        onnx_path = fetch_dreamvae_onnx(output_dir)
    onnx_path = Path(onnx_path)

    config = VAETRTBuildConfig(
        workspace_gb=workspace_gb,
        decode_max_frames=int(duration_s) * 25,  # 25 latent frames/sec
    )

    logger.info("=" * 60)
    logger.info("DREAMVAE TRT BUILD: %s (max_duration=%ds)", name, duration_s)
    logger.info("=" * 60)

    engine_path.parent.mkdir(parents=True, exist_ok=True)
    build_vae_decode_engine(onnx_path, engine_path, config=config)
    return engine_path, "OK"


def build_dreamvae_engines(
    *,
    output_dir: Union[str, Path],
    durations: Iterable[int] = (60, 120, 240),
    workspace_gb: float = 8.0,
    force_rebuild: bool = False,
) -> list[tuple[str, str, float, str]]:
    """Build the canonical dreamvae engine matrix.

    Returns a list of ``(label, engine_path, elapsed_seconds, status)``
    matching the shape used by the standard VAE/decoder builders in
    :mod:`acestep.engine.trt.build`, so callers can splice the result
    into the same summary/report path.
    """
    import time

    output_dir = Path(output_dir)
    # Fetch once up front so the per-duration loop doesn't pay HF
    # network latency three times.
    onnx_path = fetch_dreamvae_onnx(output_dir)

    results: list[tuple[str, str, float, str]] = []
    for dur in durations:
        label = f"DreamVAE decode {dur}s"
        t0 = time.time()
        try:
            engine_path, status = build_dreamvae_engine(
                output_dir=output_dir,
                duration_s=dur,
                onnx_path=onnx_path,
                workspace_gb=workspace_gb,
                force_rebuild=force_rebuild,
            )
            elapsed = time.time() - t0
            if status == "OK":
                logger.info("Built in %.0fs", elapsed)
            results.append((label, str(engine_path), elapsed, status))
        except Exception as e:  # build failures shouldn't kill the matrix
            logger.error("DreamVAE %ds build failed: %s", dur, e)
            results.append((label, "", time.time() - t0, "FAILED"))
    return results
