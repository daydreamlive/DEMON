"""Print the input/output dtypes declared by an existing engine."""
import sys
from pathlib import Path
import tensorrt as trt

ENG = Path(r"C:\Users\ryanf\.daydream-scope\models\rtmg\trt_engines\decoder_mixed_refit_b8_240s\decoder_mixed_refit_b8_240s.engine")

if not ENG.exists():
    print(f"NOT FOUND: {ENG}")
    sys.exit(1)

print(f"engine: {ENG}")
print(f"size:   {ENG.stat().st_size/1e9:.2f} GB")

logger = trt.Logger(trt.Logger.WARNING)
runtime = trt.Runtime(logger)
engine = runtime.deserialize_cuda_engine(ENG.read_bytes())
print(f"num_io: {engine.num_io_tensors}")
for i in range(engine.num_io_tensors):
    name = engine.get_tensor_name(i)
    dtype = engine.get_tensor_dtype(name)
    mode = engine.get_tensor_mode(name)
    print(f"  {name:30s} {str(mode):20s} {dtype}")
