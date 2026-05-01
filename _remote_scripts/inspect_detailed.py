"""Detailed engine inspection with execution context (so shapes/dtypes are filled in).

Args: <engine_path> <input_name> <T_value> <out_json>
e.g. inspect_detailed.py vae_decode_fp16_240s.engine latents 1500 out.json

This will set up the context but NOT execute, so it's safe.
"""
import json
import sys
from pathlib import Path

import tensorrt as trt

eng_path = Path(sys.argv[1])
in_name = sys.argv[2]
T = int(sys.argv[3])
out_path = Path(sys.argv[4])
input_dim = int(sys.argv[5]) if len(sys.argv) > 5 else 64

print(f"TRT version: {trt.__version__}", flush=True)
print(f"engine: {eng_path}  ({eng_path.stat().st_size/(1<<20):.1f} MB)", flush=True)

logger = trt.Logger(trt.Logger.ERROR)
runtime = trt.Runtime(logger)

print("STAGE 1: deserialize", flush=True)
engine = runtime.deserialize_cuda_engine(eng_path.read_bytes())
print(f"  num_layers={engine.num_layers}", flush=True)

print("STAGE 2: create context", flush=True)
ctx = engine.create_execution_context()

print("STAGE 3: set input shape", flush=True)
ctx.set_input_shape(in_name, (1, input_dim, T))

print("STAGE 4: create_engine_inspector + bind context", flush=True)
inspector = engine.create_engine_inspector()
inspector.execution_context = ctx
print("    bound", flush=True)

print("STAGE 5: get_engine_information(JSON) DETAILED", flush=True)
info = inspector.get_engine_information(trt.LayerInformationFormat.JSON)
print(f"    bytes={len(info)}", flush=True)
out_path.write_text(info)
print(f"    saved -> {out_path}", flush=True)

# Count by LayerType
try:
    obj = json.loads(info)
    if isinstance(obj, dict):
        layers = obj.get("Layers", [])
    else:
        layers = obj
    print(f"  layers count: {len(layers)}", flush=True)
    kinds = {}
    types_per_name = []
    for L in layers:
        if isinstance(L, dict):
            k = L.get("LayerType", "?")
            n = L.get("Name", "?")
            kinds[k] = kinds.get(k, 0) + 1
            types_per_name.append((k, n))
    for k, n in sorted(kinds.items(), key=lambda x: -x[1]):
        print(f"    {n:5d} {k}", flush=True)
except Exception as e:
    print(f"  parse failed: {e!r}", flush=True)

print("DONE", flush=True)
