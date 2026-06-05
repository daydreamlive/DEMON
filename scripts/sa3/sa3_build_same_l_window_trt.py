"""Build a window-profile TensorRT engine for SA3 SAME-L decode.

This uses Stability-AI/stable-audio-3-optimized's SAME-L decoder ONNX, but
builds a small latent-length profile sized for DEMON's SAME window decode.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path


HF_REPO = "stabilityai/stable-audio-3-optimized"
HF_ONNX = "onnx/same-l/dec_dynamic_triton_swa.onnx"


def _repo_root() -> Path:
    here = Path(__file__).resolve()
    return next(p for p in (here.parent, *here.parents) if (p / "pyproject.toml").exists())


def _default_plugin_dir() -> Path:
    return _repo_root() / "notes" / "SA3" / "stable-audio-3" / "optimized" / "tensorRT" / "scripts"


def _default_out(min_t: int, opt_t: int, max_t: int) -> Path:
    return (
        Path.home()
        / ".daydream-scope"
        / "models"
        / "demon"
        / "sa3"
        / "trt_engines"
        / f"same_l_decode_window_t{min_t}_{opt_t}_{max_t}"
        / f"same_l_decode_window_t{min_t}_{opt_t}_{max_t}.trt"
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-latents", type=int, default=32)
    ap.add_argument("--opt-latents", type=int, default=56)
    ap.add_argument("--max-latents", type=int, default=96)
    ap.add_argument("--workspace-gb", type=int, default=16)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--plugin-dir", type=Path, default=_default_plugin_dir())
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    if not (0 < args.min_latents <= args.opt_latents <= args.max_latents):
        raise ValueError("Require 0 < min <= opt <= max latent frames")

    out_path = args.out or _default_out(args.min_latents, args.opt_latents, args.max_latents)
    if out_path.exists() and not args.force:
        print(f"[skip] {out_path}")
        return 0

    sys.path.insert(0, str(args.plugin_dir))

    from huggingface_hub import hf_hub_download
    import tensorrt as trt

    print(f"[download] {HF_REPO}/{HF_ONNX}", flush=True)
    onnx_path = hf_hub_download(repo_id=HF_REPO, filename=HF_ONNX)
    print(f"[onnx] {onnx_path}", flush=True)

    print("[plugin] registering samel::diff_attn_swa", flush=True)
    import diff_attn_nocast_plugin  # noqa: F401

    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
    network = builder.create_network(network_flags)
    parser = trt.OnnxParser(network, logger)
    if not parser.parse_from_file(onnx_path):
        for i in range(parser.num_errors):
            print(f"[parse-error] {parser.get_error(i)}", flush=True)
        raise RuntimeError("ONNX parse failed")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(
        trt.MemoryPoolType.WORKSPACE,
        int(args.workspace_gb * (1 << 30)),
    )
    profile = builder.create_optimization_profile()
    profile.set_shape(
        "latent",
        (1, 256, args.min_latents),
        (1, 256, args.opt_latents),
        (1, 256, args.max_latents),
    )
    if config.add_optimization_profile(profile) < 0:
        raise RuntimeError("Failed to add optimization profile")

    print(
        f"[build] latent=[1,256,T] min={args.min_latents} "
        f"opt={args.opt_latents} max={args.max_latents} workspace={args.workspace_gb}GB",
        flush=True,
    )
    t0 = time.perf_counter()
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("TensorRT build failed")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as f:
        f.write(serialized)
    print(
        f"[write] {out_path} size={out_path.stat().st_size / 1e6:.1f}MB "
        f"build_s={time.perf_counter() - t0:.1f}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
