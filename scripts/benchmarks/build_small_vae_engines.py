"""Build sub-second real (teacher) VAE decode TRT engines.

For the windowed-streaming experiment we want a decode engine sized to a
~1 s span instead of the 3-30 s windowed engine. Two engines, both from
the standard teacher ``vae_decode.onnx`` (NOT DreamVAE), fp16:

  * ``vae_decode_fp16_1s_fixed``  (min=opt=max=25)  — fixed 1.0 s span;
    smallest workspace / fastest kernel selection for the streaming chunk.
  * ``vae_decode_fp16_sub1s_dyn`` (min=10, opt=24, max=40)  — small
    dynamic range so the overlap margin can be tuned per-session
    (keep 8 fr + margin 1..16 fr) on one engine, and so fp16 convergence
    can be swept without a rebuild.

Usage::

    uv run python scripts/benchmarks/build_small_vae_engines.py
    uv run python scripts/benchmarks/build_small_vae_engines.py --force
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Force the DEMON repo to the FRONT of sys.path (sibling ACE-Step shadows
# ``acestep`` and lacks the modules we need).
PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
while PROJECT_ROOT in sys.path:
    sys.path.remove(PROJECT_ROOT)
sys.path.insert(0, PROJECT_ROOT)

from acestep.engine.trt.vae_export import VAETRTBuildConfig, build_vae_decode_engine
from acestep.paths import trt_engines_dir

VAE_DECODE_ONNX_CANDIDATES = (
    "_onnx_vae/vae_decode/vae_decode.onnx",
    "_onnx/vae_decode/vae_decode.onnx",
)

# (engine_dir_name, min_frames, opt_frames, max_frames, description)
ENGINES = (
    ("vae_decode_fp16_1s_fixed", 25, 25, 25, "fixed 1.0s (min=opt=max=25)"),
    ("vae_decode_fp16_sub1s_dyn", 10, 24, 40, "dynamic 0.4-1.6s (min=10 opt=24 max=40)"),
)


def find_onnx(trt_dir: Path) -> Path:
    for rel in VAE_DECODE_ONNX_CANDIDATES:
        p = trt_dir / rel
        if p.exists():
            return p
    raise FileNotFoundError(f"vae_decode.onnx not found under {trt_dir}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--workspace-gb", type=float, default=8.0)
    args = ap.parse_args()

    trt_dir = trt_engines_dir()
    onnx_path = find_onnx(trt_dir)
    print(f"ONNX: {onnx_path}\nOutput dir: {trt_dir}\n")

    for name, mn, op, mx, desc in ENGINES:
        engine_dir = trt_dir / name
        engine_path = engine_dir / f"{name}.engine"
        if engine_path.exists() and not args.force:
            print(f"[skip] {name} exists ({engine_path.stat().st_size/1e6:.1f} MB)")
            continue
        engine_dir.mkdir(parents=True, exist_ok=True)
        cfg = VAETRTBuildConfig(
            fp16=True, workspace_gb=args.workspace_gb,
            decode_min_frames=mn, decode_opt_frames=op, decode_max_frames=mx,
        )
        print(f"[build] {name}: {desc}")
        t0 = time.time()
        build_vae_decode_engine(onnx_path, engine_path, config=cfg)
        print(f"[done]  {name}: {time.time()-t0:.0f}s, "
              f"{engine_path.stat().st_size/1e6:.1f} MB\n")


if __name__ == "__main__":
    main()
