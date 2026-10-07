"""E3: cosine of audio-token CAA vectors (caa_e3) to the old all-row vectors (caa), and K/V-site vectors."""
import json, sys, torch
from pathlib import Path
O = Path("E:/Projects/tada-replication/sa3")
blocks = [int(x) for x in sys.argv[1:]] or list(range(24))
res = {}
for f in sorted((O / "caa_e3").glob("*.pt")):
    if f.name.count(".") != 1:
        continue
    c = f.stem
    new = torch.load(f, weights_only=False)["vectors"]; old = torch.load(O / "caa" / f.name, weights_only=False)["vectors"]
    cs = [float(torch.nn.functional.cosine_similarity(new[s][b].float(), old[s][b].float(), dim=0)) for s in new for b in blocks]
    kv = {}
    if (O / "caa_e3" / f"{c}.kv.pt").exists():
        a = torch.load(O / "caa_e3" / f"{c}.kv.pt", weights_only=False)["vector"]; b = torch.load(O / "caa" / f"{c}.kv.pt", weights_only=False)["vector"]
        kv = float(torch.nn.functional.cosine_similarity(a.float(), b.float(), dim=0))
    res[c] = {"caa_mean": sum(cs) / len(cs), "caa_min": min(cs), "kv": kv}
    print(f"{c:18s} CAA cos mean {res[c]['caa_mean']:.3f} min {res[c]['caa_min']:.3f}  K/V cos {kv if kv == {} else round(kv, 3)}")
(O / "caa_e3" / "cosine_to_old.json").write_text(json.dumps({"blocks": blocks, "per_concept": res}, indent=1))
