"""Dump per-layer info for ONE VAE engine. Args: <engine_path> <out_json>."""
import json
import sys
from pathlib import Path

import tensorrt as trt

eng_path = Path(sys.argv[1])
out_path = Path(sys.argv[2])

print(f"TRT version: {trt.__version__}", flush=True)
print(f"engine: {eng_path}  ({eng_path.stat().st_size/(1<<20):.1f} MB)", flush=True)

logger = trt.Logger(trt.Logger.ERROR)
runtime = trt.Runtime(logger)

print("STAGE 1: read", flush=True)
data = eng_path.read_bytes()

print("STAGE 2: deserialize", flush=True)
engine = runtime.deserialize_cuda_engine(data)
assert engine is not None
print(f"  num_layers={engine.num_layers}  num_io={engine.num_io_tensors}", flush=True)

print("STAGE 3: io tensors", flush=True)
for i in range(engine.num_io_tensors):
    name = engine.get_tensor_name(i)
    mode = engine.get_tensor_mode(name)
    dtype = engine.get_tensor_dtype(name)
    shape = engine.get_tensor_shape(name)
    print(f"    [{mode.name}] {name}: dtype={dtype} shape={tuple(shape)}", flush=True)

print("STAGE 4: create_engine_inspector", flush=True)
inspector = engine.create_engine_inspector()
print("    inspector OK", flush=True)

print("STAGE 5: get_engine_information(JSON) without execution context", flush=True)
try:
    info = inspector.get_engine_information(trt.LayerInformationFormat.JSON)
    out_path.write_text(info)
    print(f"  saved -> {out_path} ({len(info)} bytes)", flush=True)
except Exception as e:
    print(f"  failed (no ctx): {e!r}", flush=True)
    sys.exit(1)

print("STAGE 6: parse and histogram", flush=True)
try:
    obj = json.loads(info)
    if isinstance(obj, dict) and "Layers" in obj:
        layers = obj["Layers"]
    elif isinstance(obj, list):
        layers = obj
    else:
        layers = []
    print(f"  total layers in JSON: {len(layers)}", flush=True)
    kinds = {}
    for L in layers:
        k = L.get("LayerType", "?") if isinstance(L, dict) else "?"
        kinds[k] = kinds.get(k, 0) + 1
    for k, n in sorted(kinds.items(), key=lambda x: -x[1]):
        print(f"    {n:5d} {k}", flush=True)
except Exception as e:
    print(f"  parse failed: {e!r}", flush=True)

print("DONE", flush=True)
