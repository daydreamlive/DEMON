"""E3: pairwise cosine of the 9 CAA vectors (per block, mean over steps; and mean over blocks)."""
import json, sys, torch
from pathlib import Path
O = Path("E:/Projects/tada-replication/sa3")
vdir = sys.argv[1]; blocks = [int(x) for x in sys.argv[2:]]
cs = ["tempo", "piano", "mood", "vocal_gender", "vocal_style", "violin", "guitar_electronic", "electronic_music", "rock_genre"]
V = {c: torch.load(O / vdir / f"{c}.pt", weights_only=False)["vectors"] for c in cs}
steps = sorted(V[cs[0]])
def cos(a, b, blk):
    return sum(float(torch.nn.functional.cosine_similarity(V[a][s][blk].float(), V[b][s][blk].float(), dim=0)) for s in steps) / len(steps)
out = {"blocks": {}, "vdir": vdir}
for blk in blocks + ["mean"]:
    M = [[(sum(cos(a, b, x) for x in blocks) / len(blocks)) if blk == "mean" else cos(a, b, blk) for b in cs] for a in cs]
    out["blocks"][str(blk)] = M
    print(f"\n{vdir} block {blk} (mean over 8 steps)")
    print(" " * 18 + " ".join(f"{c[:6]:>6s}" for c in cs))
    for a, row in zip(cs, M):
        print(f"{a:18s}" + " ".join(f"{x:6.2f}" for x in row))
out["concepts"] = cs
(O / vdir / "cosine_matrix.json").write_text(json.dumps(out, indent=1))
