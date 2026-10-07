"""E3 tables: AUC (avg of pos/neg) per variant, PCI rows, ratios to PCI-all; per-concept table.
usage: e3_table.py [sub=eval_e3]"""
import json, sys
from pathlib import Path
O = Path("E:/Projects/tada-replication/sa3")
sub = sys.argv[1] if len(sys.argv) > 1 else "eval_e3"
A = json.loads((O / sub / "auc.json").read_text())
CS = ["tempo", "piano", "mood", "violin", "guitar_electronic", "rock_genre", "electronic_music"]
CS = [c for c in CS if c in A]
labels = ["pci_all", "pci_loc", "caa_loc", "caa_all", "austeer_loc", "caakv_loc"]
NAMES = {"pci_all": "PCI-all (real prompt)", "pci_loc": "PCI-loc (real prompt, blocks only)", "caa_loc": "CAA localised",
         "caa_all": "CAA all blocks", "austeer_loc": "AUSteer localised", "caakv_loc": "CAA at the K/V site, localised"}
def avg(c, lab, m):
    r = A.get(c, {}).get(lab)
    if not r or "pos" not in r: return None
    return (r["pos"][m] + r["neg"][m]) / 2
def mean(xs):
    xs = [x for x in xs if x is not None and x == x]; return sum(xs) / len(xs) if xs else None
out = {}
for lab in labels:
    mu = mean([avg(c, lab, "muqt") for c in CS]); cl = mean([avg(c, lab, "clap") for c in CS])
    pq = mean([A[c][lab]["quality"]["PQ"] for c in CS if lab in A.get(c, {}) and "quality" in A[c][lab]])
    sm = mean([(A[c][lab]["pos"]["csm_muqt"] + A[c][lab]["neg"]["csm_muqt"]) / 2 for c in CS if lab in A.get(c, {})])
    out[lab] = {"muq": mu, "clap": cl, "pq": pq, "smooth_muq": sm, "n": sum(1 for c in CS if lab in A.get(c, {}))}
p_mu, p_cl = out["pci_all"]["muq"], out["pci_all"]["clap"]
f = lambda x: "n/a" if x is None else f"{x:.3f}"
r = lambda x, p: "n/a" if x is None or not p else f"{x / p:.2f}"
print(f"| Variant | concepts | AUC MuQ | AUC CLAP | ratio to PCI-all (MuQ) | ratio to PCI-all (CLAP) | smoothness MuQ | PQ |")
print("| --- | --- | --- | --- | --- | --- | --- | --- |")
for lab in labels:
    o = out[lab]
    if o["n"] == 0: continue
    print(f"| {NAMES[lab]} | {o['n']} | {f(o['muq'])} | {f(o['clap'])} | {r(o['muq'], p_mu)} | {r(o['clap'], p_cl)} | {f(o['smooth_muq'])} | {f(o['pq'])} |")
print()
print("| Concept | PCI-all MuQ | CAA loc MuQ | CAA all MuQ | AUSteer loc MuQ | K/V loc MuQ | CAA loc / PCI | CAA loc CLAP | PCI-all CLAP |")
print("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
for c in CS:
    pa = avg(c, "pci_all", "muqt"); cl = avg(c, "caa_loc", "muqt")
    print(f"| {c} | {f(pa)} | {f(cl)} | {f(avg(c, 'caa_all', 'muqt'))} | {f(avg(c, 'austeer_loc', 'muqt'))} | {f(avg(c, 'caakv_loc', 'muqt'))} | {r(cl, pa)} | {f(avg(c, 'caa_loc', 'clap'))} | {f(avg(c, 'pci_all', 'clap'))} |")
(O / sub / "e3_table.json").write_text(json.dumps(out, indent=1))
