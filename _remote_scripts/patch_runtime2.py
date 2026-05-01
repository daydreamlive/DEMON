"""Fix the SECOND hardcoded fp32 timestep in diffusion.py fast path (line 1195)."""
dp = '/workspace/acestep/acestep/engine/diffusion.py'
src = open(dp).read()

old = (
            '            bufs = {\n'
            '                "hidden_states": torch.empty(cfg_bsz, eff_T, 64, dtype=io_dtype, device=device),\n'
            '                "timestep": torch.empty(cfg_bsz, dtype=torch.float32, device=device),\n'
            '                "encoder_hidden_states": torch.empty(cfg_bsz, L, 2048, dtype=io_dtype, device=device),\n'
            '                "context_latents": torch.empty(cfg_bsz, eff_T, 128, dtype=io_dtype, device=device),\n'
            '            }\n'
)
new = (
            '            in_dt = self._trt_input_dtypes\n'
            '            bufs = {\n'
            '                "hidden_states": torch.empty(cfg_bsz, eff_T, 64, dtype=in_dt["hidden_states"], device=device),\n'
            '                "timestep": torch.empty(cfg_bsz, dtype=in_dt["timestep"], device=device),\n'
            '                "encoder_hidden_states": torch.empty(cfg_bsz, L, 2048, dtype=in_dt["encoder_hidden_states"], device=device),\n'
            '                "context_latents": torch.empty(cfg_bsz, eff_T, 128, dtype=in_dt["context_latents"], device=device),\n'
            '            }\n'
)
assert old in src, "fast path buffer marker not found"
src = src.replace(old, new)
open(dp, 'w').write(src)
print("fast path patched")
