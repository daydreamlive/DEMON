"""Dump per-layer info for VAE engines using IEngineInspector.

Compares the broken 10.15 engine and the working 10.13 engine.
"""
import json
import sys
from pathlib import Path

import tensorrt as trt

ROOT = Path("C:/Users/ryanf/.daydream-scope/models/rtmg/trt_engines")

ENGINES = {
    "10.15_decode": ROOT / "vae_decode_fp16_240s/vae_decode_fp16_240s.engine",
    "10.13_decode": ROOT / "vae_decode_fp16_240s/vae_decode_fp16_240s.engine.trt10.13.bak",
    "10.15_encode": ROOT / "vae_encode_fp16_240s/vae_encode_fp16_240s.engine",
    "10.13_encode": ROOT / "vae_encode_fp16_240s/vae_encode_fp16_240s.engine.trt10.13.bak",
}

print(f"TRT version: {trt.__version__}", flush=True)

logger = trt.Logger(trt.Logger.WARNING)

OUT_DIR = Path("C:/_dev/projects/ACE-Step-1.5_alt/_remote_scripts/_inspect_out")
OUT_DIR.mkdir(parents=True, exist_ok=True)

for tag, eng_path in ENGINES.items():
    if not eng_path.exists():
        print(f"  MISSING: {tag} -> {eng_path}", flush=True)
        continue
    print(f"\n=== {tag}: {eng_path.name} ({eng_path.stat().st_size/(1<<20):.1f} MB) ===", flush=True)
    runtime = trt.Runtime(logger)
    with open(eng_path, "rb") as f:
        engine = runtime.deserialize_cuda_engine(f.read())
    if engine is None:
        print(f"  FAILED TO DESERIALIZE", flush=True)
        continue
    print(f"  num_layers={engine.num_layers}  num_io={engine.num_io_tensors}", flush=True)

    # Try to print I/O dtypes
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        mode = engine.get_tensor_mode(name)
        dtype = engine.get_tensor_dtype(name)
        shape = engine.get_tensor_shape(name)
        print(f"    [{mode.name}] {name}: dtype={dtype} shape={tuple(shape)}", flush=True)

    inspector = engine.create_engine_inspector()
    inspector.execution_context = engine.create_execution_context()
    info = inspector.get_engine_information(trt.LayerInformationFormat.JSON)
    out_path = OUT_DIR / f"{tag}.json"
    out_path.write_text(info)
    print(f"  saved layer info -> {out_path}", flush=True)

    # Try to count layer types for a quick comparison
    try:
        data = json.loads(info)
        layers = data.get("Layers", [])
        kinds = {}
        for L in layers:
            k = L.get("LayerType", "?")
            kinds[k] = kinds.get(k, 0) + 1
        print(f"  layer kinds:", flush=True)
        for k, n in sorted(kinds.items(), key=lambda x: -x[1]):
            print(f"    {n:5d} {k}", flush=True)
    except Exception as e:
        print(f"  json parse failed: {e}", flush=True)

print("\nDONE", flush=True)
