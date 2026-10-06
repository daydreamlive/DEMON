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
    _sa3_engine_hint(req)
    logger.info(
        "preflight_sa3_ok model_id={} base_dir={}", req.model_id, req.checkpoint_dir,
    )
    return PreflightResult.passed()


#: ``sa3_build`` invocation per catalog id with TRT DiT support.
SA3_ENGINE_BUILD_COMMANDS = {
    "medium": "python -m acestep.engine.trt.sa3_build --all",
    "small-music": "python -m acestep.engine.trt.sa3_build --model small-music --all",
}


def _sa3_engine_hint(req: PreflightRequest) -> Optional[str]:
    """Warn (never fail) when TensorRT was asked for but no DiT engine
    exists for the model: sessions then run the eager DiT, several times
    slower, and the operator should know why and how to fix it. Returns
    the warning text (None when engines exist, TRT is off, or the model
    has no engine support)."""
    from acestep.engine.sa3_trt import max_dit_engine_latents

    cmd = SA3_ENGINE_BUILD_COMMANDS.get(req.model_id)
    if req.decoder_accel != "tensorrt" or cmd is None:
        return None
    if max_dit_engine_latents(req.model_id) is not None:
        return None
    text = (
        f"no SA3 TensorRT DiT engine for {req.model_id}: sessions will run "
        f"the eager DiT. Build them with: {cmd}"
    )
    logger.warning("preflight_sa3_no_trt_engine model_id={} hint={!r}", req.model_id, text)
    return text


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
    components present. An absent autoregressive stage is fatal unless
    ``DEMON_MINIMAX_CAPTURE`` names a saved capture to stream; a missing
    DiT engine is not (the renderer degrades to eager). An explicitly set
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
    # Capture mode is a dev/test mode that serves one fixed composition
    # to every user: say so at warning level.
    log = logger.warning if os.environ.get("DEMON_MINIMAX_CAPTURE") else logger.info
    log("preflight_minimax_ok model_id={} detail={}", req.model_id, msg)
    return PreflightResult.passed()


def yue2_preflight(req: PreflightRequest) -> PreflightResult:
    """The YuE2 family's boot check.

    Offline and cheap: both checkpoint dirs under ``DEMON_YUE2_ROOT`` with
    manifest-matching sizes (the SHA256 check runs at model load), the
    upstream ``yue2`` package and ``tiktoken`` importable from
    ``DEMON_YUE2_YUE_SRC`` / ``DEMON_YUE2_EXTRA_PATH``, and, when
    ``DEMON_YUE2_TRT_DIR`` is set, both engines present in it (a set but
    incomplete engine dir is an operator mistake; an unset one means the
    eager NAR, which is several times slower per tick).
    """
    from acestep.engine import yue2_runtime as rt
    from acestep.engine.yue2_trt import find_flexible_engine, find_vae_window_engine

    ok, msg = rt.weights_status(rt.weights_root())
    if not ok:
        return PreflightResult.failed(
            "YuE2 weights unavailable", msg,
            f"set {rt.ROOT_ENV} to a directory holding {rt.MODEL_DIR}/ and {rt.VAE_DIR}/ "
            f"(m-a-p/YuE2-3B @ {rt.MODEL_REVISION[:8]}, m-a-p/YuE2-Vae @ {rt.VAE_REVISION[:8]})",
        )
    missing = rt.missing_modules()
    if missing:
        return PreflightResult.failed(
            "YuE2 runtime not importable", f"cannot import: {', '.join(missing)}",
            f"set {rt.YUE_SRC_ENV} to upstream YuE @ {rt.CODE_REVISION[:8]} src/ "
            f"and {rt.EXTRA_PATH_ENV} to a directory holding tiktoken",
        )
    trt_dir = rt.trt_dir()
    if trt_dir is not None:
        absent = [name for name, found in (
            ("flexible_song/velocity.trt", find_flexible_engine(trt_dir)),
            ("vae_fp32_t37.trt", find_vae_window_engine(trt_dir)),
        ) if found is None]
        if absent:
            return PreflightResult.failed(
                "YuE2 TensorRT engines missing", f"{rt.TRT_DIR_ENV}={trt_dir} lacks {', '.join(absent)}",
                f"build: python -m acestep.engine.trt.yue2_build nar --out {trt_dir}",
                f"       python -m acestep.engine.trt.yue2_build vae --out {trt_dir}",
                f"or unset {rt.TRT_DIR_ENV} to run the eager NAR",
            )
    logger.info("preflight_yue2_ok {} trt_dir={}", msg, trt_dir)
    return PreflightResult.passed()
