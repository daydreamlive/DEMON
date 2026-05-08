"""Probe TRT engine optimization profiles for the 60s 2B engines.

Prints the min/opt/max shapes per IO tensor so the isolation benchmark
can pick shapes that are inside every engine's allowed range.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from acestep.paths import trt_engine_path

ENGINES = {
    "decoder": "decoder_mixed_refit_b8_60s",
    "vae_encode": "vae_encode_fp16_60s",
    "vae_decode": "vae_decode_fp16_60s",
}


def main() -> None:
    import tensorrt as trt
    from polygraphy.backend.common import bytes_from_path
    from polygraphy.backend.trt import engine_from_bytes

    print(f"TensorRT runtime: {trt.__version__}")
    for label, name in ENGINES.items():
        path = trt_engine_path(name)
        print(f"\n[{label}] {name}")
        print(f"  path: {path}")
        if not path.exists():
            print("  MISSING")
            continue
        engine = engine_from_bytes(bytes_from_path(str(path)))
        for i in range(engine.num_io_tensors):
            tname = engine.get_tensor_name(i)
            mode = engine.get_tensor_mode(tname)
            dtype = engine.get_tensor_dtype(tname)
            shape = tuple(engine.get_tensor_shape(tname))
            io = "in " if mode == trt.TensorIOMode.INPUT else "out"
            print(f"  {io} {tname:32s} dtype={dtype} shape={shape}")
            if mode == trt.TensorIOMode.INPUT and engine.num_optimization_profiles > 0:
                try:
                    mn, op, mx = engine.get_tensor_profile_shape(tname, 0)
                    print(f"      profile0: min={tuple(mn)} opt={tuple(op)} max={tuple(mx)}")
                except Exception as e:
                    print(f"      profile probe failed: {e}")


if __name__ == "__main__":
    main()
