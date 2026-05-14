"""Standalone NVFP4 engine build driver.

Mirrors what `acestep.engine.trt.build` does for FP8 but for NVFP4, without
touching the FP8 code path. Specifically:

  1. Loads the NVFP4Linear plugin DLL (must happen before TRT's ONNX parser
     sees the patched ONNX).
  2. Patches the bf16 decoder ONNX via `nvfp4_onnx.patch_bf16_onnx_to_nvfp4`.
  3. Builds a TRT engine from the patched ONNX, reusing the existing
     `build_trt_engine` from `export.py`.
  4. Writes the engine + sibling metadata.json next to it.

Usage:
    python -m acestep.engine.trt.nvfp4_build \\
        --activation-percentile absmax \\
        --outlier-skip-ratio 0
"""
from __future__ import annotations

import argparse
import ctypes
import os
import sys
import time
from pathlib import Path

from loguru import logger


# ------------------------------------------------------------------
# Defaults (production XL turbo paths)
# ------------------------------------------------------------------

BF16_ONNX = Path(os.path.expanduser(
    "~/.daydream-scope/models/demon/trt_engines/"
    "_onnx_acestep-v15-xl-turbo/decoder_refit/decoder_refit_dynbatch.onnx"
))
ABSMAX_JSON = Path(os.path.expanduser(
    "~/.daydream-scope/models/demon/calibration/"
    "decoder_xl_fp8/activation_absmax.json"
))

PLUGIN_DLL = Path(__file__).resolve().parent / "plugins" / "nvfp4_linear" / "nvfp4_linear_plugin.dll"

ENGINE_DIR = Path(os.path.expanduser(
    "~/.daydream-scope/models/demon/trt_engines/"
    "decoder_xl-turbo_nvfp4_refit_b4_60s"
))


def _load_plugin_dll():
    """Load the NVFP4Linear plugin DLL into the process address space.

    The plugin's static `REGISTER_TENSORRT_PLUGIN` macro fires during DLL
    load and registers the creator with TRT's plugin registry.
    """
    if not PLUGIN_DLL.exists():
        raise FileNotFoundError(
            f"Plugin DLL not found at {PLUGIN_DLL}. Build with "
            f"acestep/engine/trt/plugins/nvfp4_linear/build.bat first."
        )
    # Add dependent-DLL directories to the search path so cublasLt64_12.dll,
    # cudart64_12.dll and nvinfer_10.dll all resolve.
    venv_root = Path(sys.executable).resolve().parent.parent
    site_packages = venv_root / "Lib" / "site-packages"
    for d in (
        site_packages / "tensorrt_libs",
        site_packages / "torch" / "lib",
        Path(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin"),
    ):
        if d.exists():
            os.add_dll_directory(str(d))
    # Touch torch to make sure CUDA runtime is initialized
    import torch  # noqa: F401
    torch.cuda.init()
    _ = torch.empty(1, device="cuda")
    dll = ctypes.CDLL(str(PLUGIN_DLL))
    logger.info("Loaded NVFP4 plugin DLL: {}", PLUGIN_DLL)
    return dll


def build_nvfp4_engine(
    *,
    bf16_onnx_path: Path = BF16_ONNX,
    absmax_json_path: Path = ABSMAX_JSON,
    engine_dir: Path = ENGINE_DIR,
    activation_percentile: str = "absmax",
    activation_outlier_skip_ratio: float = 0.0,
    duration_s: int = 60,
    batch_max: int = 4,
    workspace_gb: float = 8.0,
    force: bool = False,
) -> Path:
    if not bf16_onnx_path.exists():
        raise FileNotFoundError(f"bf16 ONNX not found: {bf16_onnx_path}")
    if not absmax_json_path.exists():
        raise FileNotFoundError(f"absmax JSON not found: {absmax_json_path}")

    engine_dir = Path(engine_dir)
    engine_dir.mkdir(parents=True, exist_ok=True)
    engine_path = engine_dir / f"{engine_dir.name}.engine"

    logger.info("=" * 60)
    logger.info("NVFP4 ENGINE BUILD")
    logger.info("=" * 60)
    logger.info("  bf16 ONNX: {}", bf16_onnx_path)
    logger.info("  absmax JSON: {}", absmax_json_path)
    logger.info("  engine dir: {}", engine_dir)
    logger.info("  duration={}s batch_max={} workspace={:.1f}GB",
                duration_s, batch_max, workspace_gb)

    # Step 1: load plugin DLL.
    _load_plugin_dll()

    # Step 2: patch the bf16 ONNX -> NVFP4 ONNX.
    from .nvfp4_onnx import patch_bf16_onnx_to_nvfp4
    t0 = time.time()
    patched_onnx = patch_bf16_onnx_to_nvfp4(
        bf16_onnx_path=bf16_onnx_path,
        activation_absmax_json_path=absmax_json_path,
        force=force,
        activation_percentile=activation_percentile,
        activation_outlier_skip_ratio=activation_outlier_skip_ratio,
    )
    logger.info("Patcher done in {:.1f}s -> {}", time.time() - t0, patched_onnx)

    # Step 3: build the TRT engine.
    import tensorrt as trt
    trt_logger = trt.Logger(trt.Logger.INFO)
    trt.init_libnvinfer_plugins(trt_logger, "")

    builder = trt.Builder(trt_logger)
    net_flags = 1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
    network = builder.create_network(net_flags)
    parser = trt.OnnxParser(network, trt_logger)

    logger.info("Parsing patched ONNX (with plugin nodes)...")
    if not parser.parse_from_file(str(patched_onnx)):
        for i in range(parser.num_errors):
            logger.error("Parser error: {}", parser.get_error(i))
        raise RuntimeError("Failed to parse NVFP4-patched ONNX")
    logger.info("Parsed: {} inputs, {} outputs, {} layers",
                network.num_inputs, network.num_outputs, network.num_layers)

    # Build config.
    bc = builder.create_builder_config()
    bc.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, int(workspace_gb * (1 << 30)))
    bc.set_flag(trt.BuilderFlag.TF32)
    # REFIT is not enabled: plugin INT8 inputs (FP4 weights) aren't refittable
    # and TRT will fail "no tactic" when forced to build a refittable engine.
    if hasattr(bc, "builder_optimization_level"):
        bc.builder_optimization_level = 3

    # Optimization profile (matches the existing decoder profile).
    seq_max = duration_s * 25
    seq_opt = min(seq_max, 1500)
    profile = builder.create_optimization_profile()
    # Fixed seq dim (avoids Myelin tactic dropouts on the embedding conv1d).
    # Batch is dynamic 1..batch_max so the runtime can issue unconditional B=1
    # passes alongside cond B>=2 batches.
    profile.set_shape(
        "hidden_states",
        min=(1, seq_opt, 64), opt=(batch_max, seq_opt, 64), max=(batch_max, seq_max, 64),
    )
    profile.set_shape(
        "timestep",
        min=(1,), opt=(batch_max,), max=(batch_max,),
    )
    profile.set_shape(
        "encoder_hidden_states",
        min=(1, 32, 2048), opt=(batch_max, 200, 2048), max=(batch_max, 512, 2048),
    )
    profile.set_shape(
        "context_latents",
        min=(1, seq_opt, 128), opt=(batch_max, seq_opt, 128), max=(batch_max, seq_max, 128),
    )
    pidx = bc.add_optimization_profile(profile)
    if pidx < 0:
        raise RuntimeError("Failed to add optimization profile")

    logger.info("Building TRT engine ...")
    t0 = time.time()
    serialized = builder.build_serialized_network(network, bc)
    if serialized is None:
        raise RuntimeError("Engine build returned None")
    elapsed = time.time() - t0
    logger.info("Engine built in {:.1f}s ({} bytes)", elapsed, serialized.nbytes)

    with open(engine_path, "wb") as f:
        f.write(serialized)
    logger.info("Engine written: {}", engine_path)

    return engine_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bf16-onnx", type=Path, default=BF16_ONNX)
    ap.add_argument("--absmax-json", type=Path, default=ABSMAX_JSON)
    ap.add_argument("--engine-dir", type=Path, default=ENGINE_DIR)
    ap.add_argument("--activation-percentile", default="absmax")
    ap.add_argument("--outlier-skip-ratio", type=float, default=0.0)
    ap.add_argument("--duration", type=int, default=60)
    ap.add_argument("--batch-max", type=int, default=4)
    ap.add_argument("--workspace-gb", type=float, default=8.0)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    build_nvfp4_engine(
        bf16_onnx_path=args.bf16_onnx,
        absmax_json_path=args.absmax_json,
        engine_dir=args.engine_dir,
        activation_percentile=args.activation_percentile,
        activation_outlier_skip_ratio=args.outlier_skip_ratio,
        duration_s=args.duration,
        batch_max=args.batch_max,
        workspace_gb=args.workspace_gb,
        force=args.force,
    )


if __name__ == "__main__":
    main()
