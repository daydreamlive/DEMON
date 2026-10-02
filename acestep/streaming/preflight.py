"""Boot preflight, per family: pure checks that return a verdict.

The server runs the active family's :attr:`FamilySpec.preflight` before it
accepts a connection, prints the banner a failing result carries, and
exits. Nothing here prints or exits itself, so a family's check can be
unit-tested and reused by pod tooling.

Each check imports what it needs lazily; importing this module pulls
nothing GPU-heavy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from acestep.engine.obs import logger


@dataclass(frozen=True)
class PreflightRequest:
    """What the operator asked the server to serve."""

    #: The resolved model id for the family (an ACE checkpoint directory
    #: name, or an SA3 catalog id such as ``"medium"``).
    model_id: str
    decoder_accel: str = "tensorrt"
    vae_accel: str = "tensorrt"
    #: Operator-supplied checkpoint directory that overrides the catalog
    #: location for ``model_id`` (``--sa3-base-checkpoint``). Only families
    #: with ``FamilySpec.accepts_checkpoint_dir`` receive one.
    checkpoint_dir: Optional[str] = None


@dataclass(frozen=True)
class PreflightResult:
    """``ok`` or a banner: a title and the lines that explain the fix."""

    ok: bool
    title: str = ""
    lines: tuple = ()

    @classmethod
    def passed(cls) -> "PreflightResult":
        return cls(True)

    @classmethod
    def failed(cls, title: str, *lines: str) -> "PreflightResult":
        return cls(False, title, tuple(lines))


def acestep_preflight(req: PreflightRequest) -> PreflightResult:
    """The ACE-Step family's boot check.

    Two checks, mirroring what the first WebSocket session would hit:

    1. Checkpoints — downloads the ACE-Step weights NOW, in the terminal
       where the operator can see progress, rather than stalling the
       first silent WS connect for 5-15 minutes.
    2. TRT engines — when either component runs TensorRT, verify a 60 s
       profile is built (decoder and/or VAE engines, matching the
       backends in use).
    """
    from acestep.model_downloader import ensure_dit_model, ensure_main_model
    from acestep.paths import (
        EngineNotBuiltError,
        available_trt_engines,
        checkpoints_dir,
    )
    from acestep.setup import DEMO_COMMAND, SETUP_COMMAND

    checkpoint = req.model_id
    ok, msg = ensure_main_model()
    if ok and checkpoint != "acestep-v15-turbo":
        ok, msg = ensure_dit_model(checkpoint)
    if not ok:
        return PreflightResult.failed(
            "Model checkpoints unavailable",
            str(msg),
            f"expected location: {checkpoints_dir()}",
            f"fix: run `{SETUP_COMMAND}` (or `uv run acestep-download`)",
        )
    logger.info("preflight_checkpoints_ok checkpoint={}", checkpoint)

    needs: tuple = ()
    if req.decoder_accel == "tensorrt":
        needs += ("decoder",)
    if req.vae_accel == "tensorrt":
        needs += ("vae_encode", "vae_decode")
    if not needs:
        return PreflightResult.passed()
    trt_profile_checkpoint = (
        checkpoint if req.decoder_accel == "tensorrt" else "acestep-v15-turbo"
    )
    try:
        available_trt_engines(
            duration_s=60.0, needs=needs, checkpoint=trt_profile_checkpoint,
        )
    except EngineNotBuiltError as exc:
        lines = [str(exc), "", f"fix (recommended): {SETUP_COMMAND}"]
        if getattr(exc, "build_command", None):
            lines.append(f"fix (manual):      {exc.build_command}")
        lines += [
            "",
            "Or run without TensorRT (slower, long first-tick warmup):",
            f"  {DEMO_COMMAND} -- --accel compile",
        ]
        return PreflightResult.failed("TensorRT engines not built", *lines)
    logger.info(
        "preflight_engines_ok needs={} checkpoint={}",
        ",".join(needs), trt_profile_checkpoint,
    )
    return PreflightResult.passed()


def sa3_preflight(req: PreflightRequest) -> PreflightResult:
    """The Stable Audio 3 family's boot check.

    Light path-existence only: the model id's safetensors (or the
    operator-supplied directory) plus the vendored ``stable_audio_3``
    source. A missing TRT engine is NOT fatal — SA3 degrades to the eager
    DiT at session create — so this never gates on engines. The status
    message carries the failure-specific remedy.
    """
    from acestep.engine.sa3_helpers import (
        sa3_checkpoint_status,
        sa3_custom_checkpoint_status,
    )

    if req.checkpoint_dir:
        ok, msg = sa3_custom_checkpoint_status(req.checkpoint_dir)
    else:
        ok, msg = sa3_checkpoint_status(req.model_id)
    if not ok:
        return PreflightResult.failed("SA3 model unavailable", str(msg))
    logger.info(
        "preflight_sa3_ok model_id={} base_dir={}", req.model_id, req.checkpoint_dir,
    )
    return PreflightResult.passed()


def mrt2_preflight(req: PreflightRequest) -> PreflightResult:
    """The Magenta RealTime 2 family's boot check.

    The model runs in an out-of-process sidecar (``scripts/mrt2_sidecar.py``
    in a Linux/WSL venv with magenta_rt + JAX). The sidecar only listens
    once its JIT warmup is done, so a plain TCP connect proves it is up
    and warm. Nothing is sent; the sidecar treats a bare connect/close as
    a dropped connection and keeps serving.
    """
    import socket

    from acestep.streaming.mrt2.protocol import sidecar_address

    try:
        host, port = sidecar_address()
    except ValueError as exc:
        return PreflightResult.failed("MRT2 sidecar address invalid", str(exc))
    try:
        with socket.create_connection((host, port), timeout=1.0):
            pass
    except OSError as exc:
        return PreflightResult.failed(
            "MRT2 sidecar not running",
            f"nothing is listening at {host}:{port} ({exc})",
            "fix: in the MRT2 Linux/WSL venv (magenta_rt + JAX) run",
            "  python scripts/mrt2_sidecar.py --model mrt2_small",
            "and wait for 'serving on'; or point DEMON_MRT2_SIDECAR=host:port at it",
        )
    logger.info("preflight_mrt2_ok sidecar={}:{}", host, port)
    return PreflightResult.passed()


def minimax_preflight(req: PreflightRequest) -> PreflightResult:
    """The MiniMax-Music3 family's boot check.

    Offline path-existence only (never downloads, never imports torch):
    the diffusers-layout checkpoint under ``DEMON_MINIMAX_DIR``, the
    models directory, or the local Hugging Face cache, with the renderer
    components present. An absent autoregressive stage is not fatal (a
    saved capture still streams), and neither is a missing DiT engine
    (the renderer degrades to eager). An explicitly set
    ``DEMON_MINIMAX_TRT_DIR`` that does not exist IS fatal: the operator
    asked for engines in a place that is not there.
    """
    import os

    from acestep.engine.minimax_helpers import minimax_checkpoint_status

    ok, msg = minimax_checkpoint_status()
    if not ok:
        return PreflightResult.failed("MiniMax-Music3 weights unavailable", str(msg))
    trt_dir = os.environ.get("DEMON_MINIMAX_TRT_DIR")
    if trt_dir and not os.path.isdir(trt_dir):
        return PreflightResult.failed(
            "MiniMax-Music3 TRT engine directory missing",
            f"DEMON_MINIMAX_TRT_DIR={trt_dir} is not a directory.",
            "Unset it to run the eager renderer, or point it at the engines "
            "built by acestep/engine/trt/minimax_build.py.",
        )
    logger.info("preflight_minimax_ok model_id={} detail={}", req.model_id, msg)
    return PreflightResult.passed()
