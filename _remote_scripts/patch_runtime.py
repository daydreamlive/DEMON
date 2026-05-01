"""Revert the export patch and fix _trt_decoder_step to read timestep dtype from engine."""
import re

# 1. Revert the bf16_mixed time_embed.float() insertion in export.py
ep = '/workspace/acestep/acestep/engine/trt/export.py'
src = open(ep).read()

bad = (
    '\n        # Force timestep embedding to fp32 (matches local working engine\n'
    '        # pattern; also lets DiffusionEngine._trt_decoder_step use its\n'
    '        # hardcoded fp32 timestep buffer).\n'
    '        decoder.time_embed.float()\n'
    '        decoder.time_embed_r.float()\n'
)
if bad in src:
    src = src.replace(bad, '')
    open(ep, 'w').write(src)
    print("export.py: reverted bf16_mixed timestep patch")
else:
    print("export.py: nothing to revert")

# 2. Patch diffusion.py: derive a per-input dtype map from the engine and use it
dp = '/workspace/acestep/acestep/engine/diffusion.py'
src = open(dp).read()

# Find the section where _trt_io_dtype is set; add a per-input dtype map alongside
old_init = (
    '        hs_trt_dtype = self._trt_engine.get_tensor_dtype("hidden_states")\n'
    '        self._trt_io_dtype = _trt_dtype_map.get(hs_trt_dtype, torch.float32)\n'
)
new_init = (
    '        hs_trt_dtype = self._trt_engine.get_tensor_dtype("hidden_states")\n'
    '        self._trt_io_dtype = _trt_dtype_map.get(hs_trt_dtype, torch.float32)\n'
    '        # Per-input dtype map (timestep may be fp32 or bf16 depending on export)\n'
    '        self._trt_input_dtypes = {\n'
    '            n: _trt_dtype_map.get(self._trt_engine.get_tensor_dtype(n), torch.float32)\n'
    '            for n in ("hidden_states", "timestep", "encoder_hidden_states", "context_latents")\n'
    '        }\n'
)
assert old_init in src, "_trt_io_dtype init marker not found in diffusion.py"
src = src.replace(old_init, new_init)

# Now replace the hardcoded fp32 timestep buffer in _trt_decoder_step
old_buf = (
    '            bufs = {\n'
    '                "hidden_states": torch.empty(hs_shape, dtype=io_dtype, device=dev),\n'
    '                "timestep": torch.empty(ts_shape, dtype=torch.float32, device=dev),\n'
    '                "encoder_hidden_states": torch.empty(enc_shape, dtype=io_dtype, device=dev),\n'
    '                "context_latents": torch.empty(cl_shape, dtype=io_dtype, device=dev),\n'
    '            }\n'
)
new_buf = (
    '            in_dt = self._trt_input_dtypes\n'
    '            bufs = {\n'
    '                "hidden_states": torch.empty(hs_shape, dtype=in_dt["hidden_states"], device=dev),\n'
    '                "timestep": torch.empty(ts_shape, dtype=in_dt["timestep"], device=dev),\n'
    '                "encoder_hidden_states": torch.empty(enc_shape, dtype=in_dt["encoder_hidden_states"], device=dev),\n'
    '                "context_latents": torch.empty(cl_shape, dtype=in_dt["context_latents"], device=dev),\n'
    '            }\n'
)
assert old_buf in src, "_trt_decoder_step buffer marker not found in diffusion.py"
src = src.replace(old_buf, new_buf)

open(dp, 'w').write(src)
print("diffusion.py: patched per-input timestep dtype")
