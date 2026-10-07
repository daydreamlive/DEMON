"""E2: which conditioning inputs reach the SA3 medium DiT (runtime check)."""
import sys
sys.argv = ["x"]
sys.path.insert(0, "scripts/tada")
import torch
import sa3_tada_run as R
from acestep.engine import sa3_tada

sam = R._load_sam()
dit = sam.model.model.model
print("dit:", type(dit).__name__, "global_cond_type", dit.global_cond_type, "timestep_cond_type", dit.timestep_cond_type,
      "patch", dit.patch_size, "input_concat_dim", dit.input_concat_dim)
print("has to_global_embed", hasattr(dit, "to_global_embed"), "has to_prepend_embed", hasattr(dit, "to_prepend_embed"),
      "to_cond_embed", hasattr(dit, "to_cond_embed"))
tr = dit.transformer
print("memory tokens", tr.num_memory_tokens, "global_cond_embedder", tr.global_cond_embedder is not None,
      "cross_attn_rope", tr.cross_attn_rotary_pos_emb is not None)
b0 = tr.layers[0]
print("block global_cond_dim", b0.global_cond_dim, "local_add", b0.to_local_embed is not None,
      "modular", list(b0.modular_local_embeds.keys()), "conformer", b0.conformer is not None)
print("wrapper:", type(sam.model.model).__name__)
cond = getattr(sam.model, "conditioner", None)
print("conditioner:", type(cond).__name__ if cond is not None else None)
if cond is not None:
    for k, v in getattr(cond, "conditioners", {}).items():
        print("  cond", k, type(v).__name__)
for a in ("cross_attn_cond_ids", "global_cond_ids", "prepend_cond_ids", "input_concat_ids"):
    print(a, getattr(sam.model, a, None))

def pre(m, args, kw):
    print("DiT.forward kwargs:")
    for k, v in kw.items():
        if torch.is_tensor(v):
            print(f"  {k}: {tuple(v.shape)} {v.dtype}")
        elif v is not None and not isinstance(v, (bool, int, float, str, tuple)):
            print(f"  {k}: {type(v).__name__}")
    for i, a in enumerate(args):
        if torch.is_tensor(a):
            print(f"  arg{i}: {tuple(a.shape)}")
    raise KeyboardInterrupt
h = dit.register_forward_pre_hook(pre, with_kwargs=True)
try:
    sa3_tada.generate(sam, ["fast upbeat drum and bass", "slow ambient pad"], seed=0, duration=10.0, steps=1)
except KeyboardInterrupt:
    pass
h.remove()
# does the prompt change any non-cross-attn input?
seen = {}
def rec(name):
    def pre(m, args, kw):
        seen.setdefault(name, []).append({k: v.detach().float().cpu().clone() for k, v in kw.items() if torch.is_tensor(v)})
        raise KeyboardInterrupt
    return pre
h = dit.register_forward_pre_hook(rec("dit"), with_kwargs=True)
for p in (["fast upbeat drum and bass"], ["slow ambient pad"]):
    try:
        sa3_tada.generate(sam, p, seed=0, duration=10.0, steps=1)
    except KeyboardInterrupt:
        pass
h.remove()
a, b = seen["dit"]
for k in a:
    same = a[k].shape == b[k].shape and torch.equal(a[k], b[k])
    print(f"prompt-dependent? {k}: {not same}")
