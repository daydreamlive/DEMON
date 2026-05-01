"""Print decoder engine profile + IO info."""
import sys
from pathlib import Path
import tensorrt as trt

eng_path = Path(sys.argv[1])

print(f"TRT version: {trt.__version__}", flush=True)
print(f"engine: {eng_path}  ({eng_path.stat().st_size/(1<<20):.1f} MB)", flush=True)

logger = trt.Logger(trt.Logger.WARNING)
runtime = trt.Runtime(logger)
engine = runtime.deserialize_cuda_engine(eng_path.read_bytes())
print(f"num_layers={engine.num_layers}", flush=True)
print(f"num_io_tensors={engine.num_io_tensors}", flush=True)
print(f"num_optimization_profiles={engine.num_optimization_profiles}", flush=True)

for i in range(engine.num_io_tensors):
    name = engine.get_tensor_name(i)
    mode = engine.get_tensor_mode(name)
    dtype = engine.get_tensor_dtype(name)
    shape = engine.get_tensor_shape(name)
    print(f"  [{mode.name}] {name}: dtype={dtype} shape={tuple(shape)}", flush=True)

for p in range(engine.num_optimization_profiles):
    print(f"  profile[{p}]:", flush=True)
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
            mn, op, mx = engine.get_tensor_profile_shape(name, p)
            print(f"    {name}: min={tuple(mn)} opt={tuple(op)} max={tuple(mx)}", flush=True)
