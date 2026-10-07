"""E1 per-guidance table: MuQ / CLAP AUC averaged over directions."""
import json
from pathlib import Path
O = Path("E:/Projects/tada-replication/sa3")
rows = [("g1 (eval50)", "eval50"), ("g3", "e1_g3"), ("g5", "e1_g5"), ("g7", "e1_g7"),
        ("g1 30 steps", "e1_s30"), ("g1 renorm x4", "e1_renorm")]
cs = ["tempo", "piano", "mood"]
print("| run | site | " + " | ".join(f"{c} MuQ / CLAP" for c in cs) + " | mean MuQ |")
print("|---|---|" + "---|" * (len(cs) + 1))
for name, sub in rows:
    f = O / sub / "auc.json"
    if not f.exists():
        print(f"| {name} | (pending) |"); continue
    d = json.loads(f.read_text())
    for site in ("caa_loc", "caa_all"):
        cells, mq = [], []
        for c in cs:
            r = d.get(c, {}).get(site)
            if not r:
                cells.append("-"); continue
            m = [r[k].get("muqt", float("nan")) for k in ("pos", "neg") if k in r]
            cl = [r[k].get("clap", float("nan")) for k in ("pos", "neg") if k in r]
            m, cl = sum(m) / len(m), sum(cl) / len(cl)
            mq.append(m); cells.append(f"{m:.3f} / {cl:.3f}")
        if mq:
            print(f"| {name} | {site[4:]} | " + " | ".join(cells) + f" | {sum(mq)/len(mq):.3f} |")
