"""Build small-profile VAE decode TRT engines for windowed-decode experiments.

Two engines are produced from the existing ``vae_decode.onnx``:

  * ``vae_decode_fp16_5s_fixed``  (min=opt=max=125 frames)  — specialised
    to a single shape; should produce the smallest workspace TRT can
    plausibly reserve for this graph.
  * ``vae_decode_fp16_3to30s``    (min=75, opt=125, max=750)  — a small
    dynamic range that comfortably covers any reasonable streaming
    window (3-30 s) while staying optimised at the typical 5 s chunk.

Both engines share the standard teacher VAE ONNX, fp16, and the same
build path as the canonical ``vae_decode_fp16_*`` engines. They land in
the standard trt_engines directory so the benchmark can discover them
the same way the runtime would.

Usage::

    uv run python scripts/benchmarks/build_vae_decode_test_engines.py
    uv run python scripts/benchmarks/build_vae_decode_test_engines.py --force
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from acestep.engine.trt.vae_export import (
    VAETRTBuildConfig,
    build_vae_decode_engine,
)
from acestep.paths import trt_engines_dir


VAE_DECODE_ONNX_CANDIDATES = (
    "_onnx_vae/vae_decode/vae_decode.onnx",
    "_onnx/vae_decode/vae_decode.onnx",
)


def find_vae_decode_onnx(trt_dir: Path) -> Path:
    for rel in VAE_DECODE_ONNX_CANDIDATES:
        p = trt_dir / rel
        if p.exists():
            return p
    raise FileNotFoundError(
        "vae_decode.onnx not found in any known location under "
        f"{trt_dir}. Run `python -m acestep.engine.trt.build --all "
        "--vae-only --duration 60` first to produce the ONNX export."
    )


# (engine_dir_name, min_frames, opt_frames, max_frames, description)
TEST_ENGINES = (
    (
        "vae_decode_fp16_5s_fixed",
        125, 125, 125,
        "fixed 5s (min=opt=max=125)",
    ),
    (
        "vae_decode_fp16_3to30s",
        75, 125, 750,
        "dynamic 3-30s (min=75, opt=125, max=750)",
    ),
)


def build_one(
    onnx_path: Path,
    out_root: Path,
    name: str,
    min_f: int,
    opt_f: int,
    max_f: int,
    workspace_gb: float,
    force: bool,
) -> tuple[str, Path, float, str]:
    engine_dir = out_root / name
    engine_path = engine_dir / f"{name}.engine"

    if engine_path.exists() and not force:
        size_mb = engine_path.stat().st_size / 1e6
        print(f"[skip] {name} already exists ({size_mb:.1f} MB)")
        return (name, engine_path, 0.0, "SKIPPED")

    engine_dir.mkdir(parents=True, exist_ok=True)
    config = VAETRTBuildConfig(
        fp16=True,
        workspace_gb=workspace_gb,
        decode_min_frames=min_f,
        decode_opt_frames=opt_f,
        decode_max_frames=max_f,
    )

    print(f"[build] {name}: min={min_f} opt={opt_f} max={max_f} frames")
    t0 = time.time()
    build_vae_decode_engine(onnx_path, engine_path, config=config)
    elapsed = time.time() - t0
    size_mb = engine_path.stat().st_size / 1e6
    print(f"[done]  {name}: {elapsed:.0f}s, {size_mb:.1f} MB")
    return (name, engine_path, elapsed, "OK")


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--force", action="store_true",
                   help="Rebuild even if engine already exists.")
    p.add_argument("--workspace-gb", type=float, default=8.0,
                   help="TRT builder workspace in GB (default: 8).")
    args = p.parse_args()

    trt_dir = trt_engines_dir()
    onnx_path = find_vae_decode_onnx(trt_dir)
    print(f"ONNX: {onnx_path}")
    print(f"Output dir: {trt_dir}\n")

    results = []
    for name, min_f, opt_f, max_f, _desc in TEST_ENGINES:
        results.append(build_one(
            onnx_path=onnx_path,
            out_root=trt_dir,
            name=name,
            min_f=min_f,
            opt_f=opt_f,
            max_f=max_f,
            workspace_gb=args.workspace_gb,
            force=args.force,
        ))

    print("\n" + "=" * 60)
    print("BUILD SUMMARY")
    print("=" * 60)
    for name, path, elapsed, status in results:
        size_mb = path.stat().st_size / 1e6 if path.exists() else -1
        print(f"  {status:7s} {elapsed:6.0f}s  {size_mb:7.1f} MB  {name}")


if __name__ == "__main__":
    main()
