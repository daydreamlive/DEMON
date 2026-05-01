"""ONNX export and TensorRT engine build for the ACE-Step VAE.

Export flow:
  1. Export VAE encoder to ONNX: audio [B, 2, samples] -> moments [B, 128, T]
  2. Export VAE decoder to ONNX: latents [B, 64, T] -> audio [B, 2, samples]
  3. Build TRT engines with FP16 and dynamic-length optimization profiles

The VAE is a 1D convolutional network with no attention, so export is
straightforward (no Lambda replacements or attention config needed).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

from loguru import logger
import torch
import torch.nn as nn


# ------------------------------------------------------------------
# Export wrappers
# ------------------------------------------------------------------

class VAEEncoderForExport(nn.Module):
    """Wraps vae.encode() to return raw moments (mean + logvar concat)."""

    def __init__(self, vae: nn.Module):
        super().__init__()
        self.vae = vae

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        # vae.encode returns AutoencoderOobleckOutput with .latent_dist
        out = self.vae.encode(audio)
        return out.latent_dist.parameters  # [B, 128, T] (mean ++ logvar)


class VAEDecoderForExport(nn.Module):
    """Wraps vae.decode() to return the audio sample directly."""

    def __init__(self, vae: nn.Module):
        super().__init__()
        self.vae = vae

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        return self.vae.decode(latents).sample


# ------------------------------------------------------------------
# ONNX export
# ------------------------------------------------------------------

@dataclass
class VAEExportConfig:
    """Configuration for VAE ONNX export."""
    # Trace input sizes
    batch_size: int = 1
    # Encoder: audio samples for tracing (30s at 48kHz)
    trace_audio_samples: int = 48000 * 30
    # Decoder: latent frames for tracing (30s at 25Hz)
    trace_latent_frames: int = 750

    opset_version: int = 18
    do_constant_folding: bool = True


def export_vae_encoder_onnx(
    vae: nn.Module,
    onnx_path: Union[str, Path],
    device: str = "cuda",
    config: Optional[VAEExportConfig] = None,
) -> Path:
    """Export VAE encoder to ONNX.

    Input:  audio [B, 2, samples]
    Output: moments [B, 128, T]
    """
    if config is None:
        config = VAEExportConfig()

    onnx_path = Path(onnx_path)
    onnx_path.parent.mkdir(parents=True, exist_ok=True)

    wrapper = VAEEncoderForExport(vae).float().eval().to(device)

    example = torch.randn(
        config.batch_size, 2, config.trace_audio_samples,
        device=device, dtype=torch.float32,
    )

    logger.info("Tracing VAE encoder for ONNX export (samples=%d)...", config.trace_audio_samples)

    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            (example,),
            str(onnx_path),
            input_names=["audio"],
            output_names=["moments"],
            dynamic_axes={
                "audio": {0: "batch", 2: "samples"},
                "moments": {0: "batch", 2: "latent_frames"},
            },
            opset_version=config.opset_version,
            do_constant_folding=config.do_constant_folding,
            dynamo=False,
        )

    logger.info("VAE encoder ONNX saved to %s", onnx_path)
    return onnx_path


def export_vae_decoder_onnx(
    vae: nn.Module,
    onnx_path: Union[str, Path],
    device: str = "cuda",
    config: Optional[VAEExportConfig] = None,
) -> Path:
    """Export VAE decoder to ONNX.

    Input:  latents [B, 64, T]
    Output: audio [B, 2, samples]
    """
    if config is None:
        config = VAEExportConfig()

    onnx_path = Path(onnx_path)
    onnx_path.parent.mkdir(parents=True, exist_ok=True)

    wrapper = VAEDecoderForExport(vae).float().eval().to(device)

    example = torch.randn(
        config.batch_size, 64, config.trace_latent_frames,
        device=device, dtype=torch.float32,
    )

    logger.info("Tracing VAE decoder for ONNX export (T=%d)...", config.trace_latent_frames)

    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            (example,),
            str(onnx_path),
            input_names=["latents"],
            output_names=["audio"],
            dynamic_axes={
                "latents": {0: "batch", 2: "latent_frames"},
                "audio": {0: "batch", 2: "samples"},
            },
            opset_version=config.opset_version,
            do_constant_folding=config.do_constant_folding,
            dynamo=False,
        )

    logger.info("VAE decoder ONNX saved to %s", onnx_path)
    return onnx_path


# ------------------------------------------------------------------
# TensorRT engine build
# ------------------------------------------------------------------

@dataclass
class VAETRTBuildConfig:
    """Configuration for VAE TensorRT engine build."""
    fp16: bool = True
    workspace_gb: float = 16.0
    # See note in build_vae_trt_engine: must stay at 1 to avoid TRT 10.15+ Myelin
    # codegen segfault on the oobleck VAE graph.
    builder_optimization_level: int = 1

    # VAE decoder profile (latent frames)
    decode_min_frames: int = 125      # ~5s
    decode_opt_frames: int = 1500     # 60s
    decode_max_frames: int = 6000     # 4min

    # VAE encoder profile (audio samples at 48kHz)
    encode_min_samples: int = 240000      # 5s
    encode_opt_samples: int = 2880000     # 60s
    encode_max_samples: int = 11520000    # 4min

    @property
    def max_duration_s(self) -> int:
        """Max duration in seconds, derived from decode_max_frames at 25Hz."""
        return self.decode_max_frames // 25

    def engine_filename(self, component: str) -> str:
        """Generate a standardized engine filename.

        Args:
            component: "vae_encode" or "vae_decode"

        Format: {component}_{precision}_{duration}s.engine
        Uses seconds so naming is stable across sample rates and frame rates.
        """
        prec = "fp16" if self.fp16 else "fp32"
        dur = self.max_duration_s
        return f"{component}_{prec}_{dur}s.engine"


def build_vae_trt_engine(
    onnx_path: Union[str, Path],
    engine_path: Union[str, Path],
    input_name: str,
    input_dims: int,
    min_dynamic: int,
    opt_dynamic: int,
    max_dynamic: int,
    config: Optional[VAETRTBuildConfig] = None,
) -> Path:
    """Build a TRT engine from a VAE ONNX file.

    Args:
        onnx_path: Path to ONNX file.
        engine_path: Where to write the engine.
        input_name: Name of the input tensor ("audio" or "latents").
        input_dims: Number of channels (2 for audio, 64 for latents).
        min_dynamic: Minimum dynamic axis size.
        opt_dynamic: Optimal dynamic axis size.
        max_dynamic: Maximum dynamic axis size.
        config: Build configuration.
    """
    import tensorrt as trt

    if config is None:
        config = VAETRTBuildConfig()

    onnx_path = Path(onnx_path)
    engine_path = Path(engine_path)
    engine_path.parent.mkdir(parents=True, exist_ok=True)

    trt_logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(trt_logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, trt_logger)

    # parse_from_file resolves external data relative to the ONNX path
    onnx_abs = str(onnx_path.resolve())
    if not parser.parse_from_file(onnx_abs):
        for i in range(parser.num_errors):
            logger.error("ONNX parse error: %s", parser.get_error(i))
        raise RuntimeError("ONNX parsing failed")

    build_config = builder.create_builder_config()
    build_config.set_memory_pool_limit(
        trt.MemoryPoolType.WORKSPACE,
        int(config.workspace_gb * (1 << 30)),
    )
    if config.fp16:
        build_config.set_flag(trt.BuilderFlag.FP16)

    # TensorRT 10.15+ Myelin codegen bug: at builder_optimization_level >= 2 the
    # oobleck VAE graph (Snake activation + WeightNorm convs + ConvTranspose1d)
    # produces a fused kernel that segfaults inside execute_async_v3 on RTX 5090.
    # Bisected: optlvl 0 and 1 produce a working ~360 MB engine using the same
    # kernel set; optlvl 2 introduces the broken fusion. Tactic source bisection
    # (no JIT_CONVOLUTIONS, no EDGE_MASK_CONVOLUTIONS) does not rescue optlvl >= 2,
    # so the bug is in core Myelin and not a single tactic source. Pin to 1.
    build_config.builder_optimization_level = config.builder_optimization_level

    profile = builder.create_optimization_profile()
    profile.set_shape(
        input_name,
        min=(1, input_dims, min_dynamic),
        opt=(1, input_dims, opt_dynamic),
        max=(1, input_dims, max_dynamic),
    )
    build_config.add_optimization_profile(profile)

    logger.info(
        "Building TRT engine: %s [%d, %d, %d]",
        input_name, min_dynamic, opt_dynamic, max_dynamic,
    )

    serialized = builder.build_serialized_network(network, build_config)
    if serialized is None:
        raise RuntimeError("TRT engine build failed")

    with open(engine_path, "wb") as f:
        f.write(serialized)

    size_mb = engine_path.stat().st_size / (1 << 20)
    logger.info("Engine saved to %s (%.1f MB)", engine_path, size_mb)
    return engine_path


def build_vae_decode_engine(
    onnx_path: Union[str, Path],
    engine_path: Union[str, Path],
    config: Optional[VAETRTBuildConfig] = None,
) -> Path:
    """Build TRT engine for VAE decoder."""
    if config is None:
        config = VAETRTBuildConfig()
    return build_vae_trt_engine(
        onnx_path, engine_path,
        input_name="latents", input_dims=64,
        min_dynamic=config.decode_min_frames,
        opt_dynamic=config.decode_opt_frames,
        max_dynamic=config.decode_max_frames,
        config=config,
    )


def build_vae_encode_engine(
    onnx_path: Union[str, Path],
    engine_path: Union[str, Path],
    config: Optional[VAETRTBuildConfig] = None,
) -> Path:
    """Build TRT engine for VAE encoder."""
    if config is None:
        config = VAETRTBuildConfig()
    return build_vae_trt_engine(
        onnx_path, engine_path,
        input_name="audio", input_dims=2,
        min_dynamic=config.encode_min_samples,
        opt_dynamic=config.encode_opt_samples,
        max_dynamic=config.encode_max_samples,
        config=config,
    )
