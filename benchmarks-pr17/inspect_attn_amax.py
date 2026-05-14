import json
from pathlib import Path
p = Path.home() / '.daydream-scope/models/demon/calibration/decoder_xl_fp8/activation_absmax.json'
raw = json.loads(p.read_text(encoding='utf-8'))
linears = raw['linears']
qkv = []
for name, rec in linears.items():
    if 'q_proj' in name or 'k_proj' in name or 'v_proj' in name:
        out_amax = rec.get('output_absmax', 0.0)
        qkv.append((name, out_amax, rec.get('absmax', 0)))
qkv.sort(key=lambda x: -x[1])
print('Top 15 attn projection outputs by output_absmax:')
for name, out_amax, in_amax in qkv[:15]:
    print(f'  out_amax={out_amax:>9.2f}  in_amax={in_amax:>8.2f}  {name}')
print()
ams = [x[1] for x in qkv]
print(f'Range over all QKV proj outputs: min={min(ams):.2f}  median={sorted(ams)[len(ams)//2]:.2f}  max={max(ams):.2f}')
print()
print('After scale by 1/sqrt(64)=0.125 (assumed scaled-Q amax):')
scaled = [x*0.125 for x in ams]
print(f'  min={min(scaled):.2f}  median={sorted(scaled)[len(scaled)//2]:.2f}  max={max(scaled):.2f}')
print()
# Per-layer breakdown
by_layer = {}
for name, out_amax, _ in qkv:
    # extract layer index
    import re
    m = re.search(r'layers\.(\d+)\.', name)
    if not m: continue
    layer = int(m.group(1))
    typ = 'q' if 'q_proj' in name else 'k' if 'k_proj' in name else 'v'
    attn = 'self' if 'self_attn' in name else 'cross'
    by_layer.setdefault((layer, attn), {})[typ] = out_amax
print('Per-(layer,attn-type) max QKV output absmax:')
for k in sorted(by_layer):
    layer, attn = k
    vals = by_layer[k]
    m = max(vals.values())
    print(f'  layer={layer:2d} {attn:5s}  q={vals.get("q",0):>7.2f}  k={vals.get("k",0):>7.2f}  v={vals.get("v",0):>7.2f}  max={m:>7.2f}')
