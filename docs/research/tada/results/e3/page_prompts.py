"""Ear renders for two free prompts with the eval_e5 ear-package setup."""
import sys, json
from pathlib import Path
import numpy as np, soundfile as sf, torch
sys.path.insert(0, "scripts/tada")
import sa3_tada_run as R
from acestep.engine import sa3_tada
dst = Path("E:/Projects/tada-replication/listen_sa3/page_prompts"); dst.mkdir(parents=True, exist_ok=True)
jobs = [("piano", "Slow tempo, peaceful meditation soundscape with a soft violin",
         [("1_zero", 0.0), ("2_pos_admitted", 5.7143), ("3_pos_max", 8.0), ("4_neg_admitted", -5.7143)]),
        ("tempo", "Cinematic soundtrack with dramatic tension",
         [("1_zero", 0.0), ("2_pos_admitted_fast", 9.1429), ("3_pos_max_fast", 16.0), ("4_neg_admitted_slow", -9.1429)])]
torch.backends.cuda.matmul.allow_tf32 = True
sam = R._load_sam()
blocks = [3, 5, 6, 7]
args = type("A", (), {"method": "caa", "out": R.DEFAULT_OUT, "vec_dir": "caa_e5"})()
out = []
for concept, prompt, variants in jobs:
    vec = R._load_vectors(args, concept, "loc")
    for tag, a in variants:
        audio = R._render_alpha(sam, [prompt], vec, blocks, a, 1)
        x = audio[0].float().mean(dim=0).numpy()
        name = f"{concept}_{tag}_alpha{a:+g}.wav"
        sf.write(dst / name, x, R.SA3_SR)
        out.append(name); print(name, flush=True)
(dst / "files.json").write_text(json.dumps(out, indent=1))
