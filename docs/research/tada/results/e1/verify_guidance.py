"""E1 offline guidance checks on SA3 medium (DEMON venv, worktree on PYTHONPATH)."""
import sys, types
from pathlib import Path
import numpy as np, torch
sys.argv = ["x"]
sys.path.insert(0, "scripts/tada")
import sa3_tada_run as R
from acestep.engine import sa3_tada

torch.backends.cuda.matmul.allow_tf32 = True
sam = R._load_sam()
args = types.SimpleNamespace(out=R.TADA_ROOT / "sa3", method="caa", holdout=False, n_prompts=50)
tests = R._eval_prompts(args)[:25]
vec = R._load_vectors(args, "tempo", "loc")
blocks = [3, 6, 7]

def q(a):
    a = a.mean(dim=1, keepdim=True)
    return (a.float() * 32768.0).round().clamp(-32768, 32767).to(torch.int16).numpy()

# 1) guidance 1 reproduces the existing eval50 render
ref = np.load(args.out / "eval50/caa_loc_tempo/alpha_16.0/audios.npz")["audio"][:25]
g1 = q(R._render_alpha(sam, tests, vec, blocks, 16.0, 25))
print("g1 vs eval50 identical:", np.array_equal(ref, g1), "maxdiff", int(np.abs(ref.astype(int) - g1.astype(int)).max()))

# 2) guidance 3: wrapper called twice per step; alpha 0 untouched
calls = []
h = sam.model.model.register_forward_pre_hook(lambda m, a: calls.append(1))
R._GUIDANCE = 3.0
g3 = q(R._render_alpha(sam, tests[:2], vec, blocks, 16.0, 2))
n3 = len(calls); calls.clear()
z3 = q(R._render_alpha(sam, tests[:2], vec, blocks, 0.0, 2))
nz = len(calls); h.remove()
R._GUIDANCE = 1.0
z1 = q(R._render_alpha(sam, tests[:2], vec, blocks, 0.0, 2))
s1 = q(R._render_alpha(sam, tests[:2], vec, blocks, 16.0, 2))
print("wrapper calls g3 alpha16:", n3, "alpha0:", nz)
print("alpha0 g3 == alpha0 g1:", np.array_equal(z3, z1))
print("g3 differs from g1:", not np.array_equal(g3, s1),
      "rms g1-base", float(np.sqrt(((s1.astype(float) - z1) ** 2).mean())),
      "rms g3-base", float(np.sqrt(((g3.astype(float) - z1) ** 2).mean())))
