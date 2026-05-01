"""INT8 TensorRT engine build for the VAE decoder.

Strategy
--------
- Pure 1D conv network → TRT's mature INT8 CNN path is a natural fit.
- Snake activation (sin/exp/reciprocal/pow) isn't int8-friendly → enable FP16
  fallback so TRT auto-chooses per-layer precision.
- Narrow dynamic profile matching ``vae_window`` in production.
- PTQ via IInt8EntropyCalibrator2 (default) or IInt8MinMaxCalibrator.
- Optional fp16 pinning for the first / last conv layers, which sit closest
  to the audio output and dominate quantization-induced reconstruction error.

The engine's I/O binding names (``latents``, ``audio``) match
:func:`acestep.nodes.vae_nodes._trt_vae_decode` so no wiring changes are
needed on the runtime side.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Union

from loguru import logger
import torch


# ------------------------------------------------------------------
# Calibrator
# ------------------------------------------------------------------

def _make_calibrator(
    latent_dir: Path,
    cache_path: Path,
    *,
    batch_size: int = 1,
    kind: str = "entropy",
):
    """Construct an INT8 calibrator over .pt latents in ``latent_dir``.

    Each .pt must be a tensor of shape ``[1, 64, opt_frames]`` (matching
    the calibration profile's pinned shape). The calibrator iterates
    them once, feeding one batch at a time from a persistent GPU buffer.

    Args:
        kind: "entropy" → IInt8EntropyCalibrator2 (KL divergence; default,
              tuned for image classifiers but works fine on most CNNs).
              "minmax" → IInt8MinMaxCalibrator (uses observed min/max,
              clipping nothing). Better when activations have heavy
              tails the entropy method clips off; worse when there are
              outlier activations that should be clipped.
    """
    import tensorrt as trt

    latents: List[Path] = sorted(latent_dir.glob("latent_*.pt"))
    if not latents:
        raise FileNotFoundError(
            f"No calibration latents found in {latent_dir}. "
            f"Run scripts/collect_vae_calibration.py first."
        )
    logger.info(f"Calibrator[{kind}]: {len(latents)} latent samples in {latent_dir}")

    if kind == "entropy":
        Base = trt.IInt8EntropyCalibrator2
    elif kind == "minmax":
        Base = trt.IInt8MinMaxCalibrator
    else:
        raise ValueError(f"Unknown calibrator kind: {kind}")

    class LatentCalibrator(Base):
        def __init__(self):
            super().__init__()
            self._idx = 0
            self._cache_path = cache_path
            self._batch_size = batch_size
            self._gpu_buf: Optional[torch.Tensor] = None

        def get_batch_size(self):
            return self._batch_size

        def get_batch(self, names):
            if self._idx >= len(latents):
                return None
            path = latents[self._idx]
            self._idx += 1
            try:
                t = torch.load(path, weights_only=True, map_location="cuda")
            except Exception as e:
                logger.warning("Skip %s: %s", path, e)
                return None
            t = t.to(device="cuda", dtype=torch.float32).contiguous()
            if self._gpu_buf is None or self._gpu_buf.shape != t.shape:
                self._gpu_buf = t.clone()
            else:
                self._gpu_buf.copy_(t)
            if self._idx % 8 == 0 or self._idx == len(latents):
                logger.info(f"Calibrator: {self._idx}/{len(latents)}")
            return [int(self._gpu_buf.data_ptr())]

        def read_calibration_cache(self):
            if self._cache_path.exists():
                logger.info("Reusing calibration cache: %s", self._cache_path)
                return self._cache_path.read_bytes()
            return None

        def write_calibration_cache(self, cache):
            self._cache_path.parent.mkdir(parents=True, exist_ok=True)
            self._cache_path.write_bytes(bytes(cache))
            logger.info(f"Wrote calibration cache: {self._cache_path} ({len(cache)/1024:.1f} KB)")

    return LatentCalibrator()


# ------------------------------------------------------------------
# Layer pinning
# ------------------------------------------------------------------

def _pin_first_last_conv(network, *, n_first: int = 1, n_last: int = 1) -> List[str]:
    """Force the first ``n_first`` and last ``n_last`` Conv layers (in
    topological order) to fp16. These sit closest to the latent input
    and audio output respectively, where quantization error is most
    audible.

    Returns the list of pinned layer names.
    """
    import tensorrt as trt

    convs = [
        i for i in range(network.num_layers)
        if network.get_layer(i).type == trt.LayerType.CONVOLUTION
    ]
    if not convs:
        logger.warning("No Conv layers found to pin")
        return []

    indices = list(dict.fromkeys(convs[:n_first] + convs[-n_last:]))
    pinned: List[str] = []
    for i in indices:
        layer = network.get_layer(i)
        layer.precision = trt.float16
        for j in range(layer.num_outputs):
            layer.set_output_type(j, trt.float16)
        pinned.append(f"#{i} {layer.name}")
    logger.info(f"Pinned to fp16: {pinned}")
    return pinned


# ------------------------------------------------------------------
# Build
# ------------------------------------------------------------------

@dataclass
class VAEDecodeInt8Config:
    """Configuration for INT8 VAE decoder engine build.

    Default dynamic profile (5s/10s/15s) matches typical short ``vae_window``
    usage. For 60s engines pass min=125, opt=1500, max=1500 to match the
    fp16 60s engine profile.
    """
    workspace_gb: float = 6.0
    # latent frames @ 25 Hz
    min_frames: int = 125   # 5s
    opt_frames: int = 250   # 10s (== default vae_window)
    max_frames: int = 375   # 15s

    calibrator_kind: str = "entropy"  # entropy | minmax
    pin_first_conv: int = 0           # how many leading Conv layers to fp16-pin
    pin_last_conv: int = 0            # how many trailing Conv layers to fp16-pin


def build_vae_decode_int8_engine(
    onnx_path: Union[str, Path],
    engine_path: Union[str, Path],
    calibration_dir: Union[str, Path],
    *,
    config: Optional[VAEDecodeInt8Config] = None,
    cache_path: Optional[Union[str, Path]] = None,
) -> Path:
    """Build an INT8-quantized TRT engine for the VAE decoder."""
    import tensorrt as trt

    if config is None:
        config = VAEDecodeInt8Config()
    onnx_path = Path(onnx_path)
    engine_path = Path(engine_path)
    calibration_dir = Path(calibration_dir)
    engine_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_path is None:
        cache_path = engine_path.with_suffix(".calib")
    cache_path = Path(cache_path)

    trt_logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(trt_logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )

    parser = trt.OnnxParser(network, trt_logger)
    if not parser.parse_from_file(str(onnx_path.resolve())):
        for i in range(parser.num_errors):
            logger.error("ONNX parse error: %s", parser.get_error(i))
        raise RuntimeError(f"ONNX parsing failed: {onnx_path}")

    build_config = builder.create_builder_config()
    build_config.set_memory_pool_limit(
        trt.MemoryPoolType.WORKSPACE, int(config.workspace_gb * (1 << 30))
    )
    # INT8 with FP16 fallback: TRT picks per-layer precision automatically,
    # keeping Snake ops in fp16 while running conv/conv_transpose in int8.
    build_config.set_flag(trt.BuilderFlag.INT8)
    build_config.set_flag(trt.BuilderFlag.FP16)

    # Optional layer pinning.
    if config.pin_first_conv or config.pin_last_conv:
        _pin_first_last_conv(
            network,
            n_first=config.pin_first_conv,
            n_last=config.pin_last_conv,
        )
        # OBEY = TRT must honour layer.precision (vs PREFER which is advisory).
        build_config.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)

    # Optimization profile + calibration profile (both required for dynamic
    # INT8 on TRT 10+).
    profile = builder.create_optimization_profile()
    profile.set_shape(
        "latents",
        min=(1, 64, config.min_frames),
        opt=(1, 64, config.opt_frames),
        max=(1, 64, config.max_frames),
    )
    build_config.add_optimization_profile(profile)

    calib_profile = builder.create_optimization_profile()
    calib_profile.set_shape(
        "latents",
        min=(1, 64, config.opt_frames),
        opt=(1, 64, config.opt_frames),
        max=(1, 64, config.opt_frames),
    )
    build_config.set_calibration_profile(calib_profile)

    calibrator = _make_calibrator(
        calibration_dir, cache_path, kind=config.calibrator_kind,
    )
    build_config.int8_calibrator = calibrator

    logger.info(
        f"Building INT8 VAE decode engine: frames min={config.min_frames} "
        f"opt={config.opt_frames} max={config.max_frames}  "
        f"calibrator={config.calibrator_kind}  "
        f"pin_first={config.pin_first_conv} pin_last={config.pin_last_conv}  "
        f"calib_dir={calibration_dir}"
    )
    serialized = builder.build_serialized_network(network, build_config)
    if serialized is None:
        raise RuntimeError("INT8 engine build failed (see TRT logger output)")
    engine_path.write_bytes(serialized)

    size_mb = engine_path.stat().st_size / (1 << 20)
    logger.info(f"Engine written: {engine_path} ({size_mb:.1f} MB)")
    return engine_path
