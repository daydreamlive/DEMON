"""Patch export.py on the remote: revert bad sed, add proper time_embed.float() to bf16_mixed."""
import re
p = '/workspace/acestep/acestep/engine/trt/export.py'
src = open(p).read()

# revert previous bad sed (if present): aggressive regex
src = re.sub(
    r' {4}decoder\.time_embed\.float\(\)\n {8}decoder\.time_embed_r\.float\(\)\n',
    '',
    src,
)

old = (
    '        # Make sure everything is bf16 to start (no-op if the model was\n'
    '        # loaded with dtype="bfloat16", but defensive).\n'
    '        decoder.to(torch.bfloat16)\n'
    '\n'
    '        # Patch unembedding'
)
new = (
    '        # Make sure everything is bf16 to start (no-op if the model was\n'
    '        # loaded with dtype="bfloat16", but defensive).\n'
    '        decoder.to(torch.bfloat16)\n'
    '\n'
    '        # Force timestep embedding to fp32 (matches local working engine\n'
    '        # pattern; also lets DiffusionEngine._trt_decoder_step use its\n'
    '        # hardcoded fp32 timestep buffer).\n'
    '        decoder.time_embed.float()\n'
    '        decoder.time_embed_r.float()\n'
    '\n'
    '        # Patch unembedding'
)
if old not in src:
    print("ERROR: old marker not found")
    raise SystemExit(1)
src = src.replace(old, new)
open(p, 'w').write(src)
print("patched OK")
