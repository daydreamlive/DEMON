"""Build a custom-profile TensorRT DiT engine for SA3 medium."""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path


HF_REPO = "stabilityai/stable-audio-3-optimized"
HF_FILES = ("onnx/sa3-m/dit.onnx", "onnx/sa3-m/dit.onnx.data")
SAMPLE_RATE = 44100
SAMPLES_PER_LATENT = 4096


def _default_out(min_l: int, opt_l: int, max_l: int) -> Path:
    return (
        Path.home()
        / ".daydream-scope"
        / "models"
        / "demon"
        / "sa3"
        / "trt_engines"
        / f"sa3_m_dit_l{min_l}_{opt_l}_{max_l}"
        / f"sa3_m_dit_l{min_l}_{opt_l}_{max_l}.trt"
    )


def _latents_for_seconds(seconds: float) -> int:
    return max(1, int(math.ceil(seconds * SAMPLE_RATE / SAMPLES_PER_LATENT)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=60.0)
    ap.add_argument("--min-latents", type=int, default=1)
    ap.add_argument("--opt-latents", type=int, default=None)
    ap.add_argument("--max-latents", type=int, default=None)
    ap.add_argument("--workspace-gb", type=int, default=16)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    profile_l = _latents_for_seconds(args.seconds)
    opt_l = int(args.opt_latents or profile_l)
    max_l = int(args.max_latents or profile_l)
    min_l = int(args.min_latents)
    if not (0 < min_l <= opt_l <= max_l):
        raise ValueError("Require 0 < min <= opt <= max latent frames")

    out_path = args.out or _default_out(min_l, opt_l, max_l)
    if out_path.exists() and not args.force:
        print(f"[skip] {out_path}")
        return 0

    from huggingface_hub import hf_hub_download
    import tensorrt as trt

    local_paths = []
    for hf_file in HF_FILES:
        print(f"[download] {HF_REPO}/{hf_file}", flush=True)
        local_paths.append(hf_hub_download(repo_id=HF_REPO, filename=hf_file))
    onnx_path = local_paths[0]
    print(f"[onnx] {onnx_path}", flush=True)

    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
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
    config.set_flag(trt.BuilderFlag.BF16)

    profile = builder.create_optimization_profile()
    profile.set_shape("x", (1, 256, min_l), (1, 256, opt_l), (1, 256, max_l))
    profile.set_shape("t", (1,), (1,), (1,))
    profile.set_shape("t5_hidden", (1, 256, 768), (1, 256, 768), (1, 256, 768))
    profile.set_shape("t5_mask", (1, 256), (1, 256), (1, 256))
    profile.set_shape("seconds_total", (1,), (1,), (1,))
    profile.set_shape("local_add_cond", (1, 257, min_l), (1, 257, opt_l), (1, 257, max_l))
    if config.add_optimization_profile(profile) < 0:
        raise RuntimeError("Failed to add optimization profile")

    print(
        f"[build] sa3-m DiT L=({min_l},{opt_l},{max_l}) "
        f"seconds~={max_l * SAMPLES_PER_LATENT / SAMPLE_RATE:.2f} "
        f"workspace={args.workspace_gb}GB BF16",
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
        f"[write] {out_path} size={out_path.stat().st_size / 1e9:.2f}GB "
        f"build_s={time.perf_counter() - t0:.1f}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
