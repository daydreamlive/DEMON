"""E5 final SA3 table (MuQ AUC avg over directions; ratio to PCI-all on the same 50 prompts)."""
import json
from pathlib import Path
O = Path("E:/Projects/tada-replication/sa3")
C7 = ["tempo", "piano", "mood", "violin", "guitar_electronic", "rock_genre", "electronic_music"]
L = lambda s: json.loads((O / s / "auc.json").read_text())
E5, E5P, E3 = L("eval_e5"), L("eval_e5p"), L("eval_e3")
def av(A, c, lab, m="muqt"):
    r = A.get(c, {}).get(lab)
    return None if not r or "pos" not in r else (r["pos"][m] + r["neg"][m]) / 2
f = lambda x: "n/a" if x is None else f"{x:.3f}"
q = lambda x, p: "n/a" if x is None or not p else f"{x / p:.2f}"
rows, sums = [], {}
print("| Concept | PCI-all | e5 loc | e5 loc/PCI | e5 all | e5 all/PCI | e3 loc/PCI | e3 all/PCI | e5p loc/PCI |")
print("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
for c in C7:
    p = av(E3, c, "pci_all")
    v = {"pci": p, "e5l": av(E5, c, "caa_loc"), "e5a": av(E5, c, "caa_all"), "e3l": av(E3, c, "caa_loc"),
         "e3a": av(E3, c, "caa_all"), "e5p": av(E5P, c, "caa_loc")}
    for k, x in v.items():
        if x is not None: sums.setdefault(k, []).append(x)
    print(f"| {c} | {f(p)} | {f(v['e5l'])} | {q(v['e5l'], p)} | {f(v['e5a'])} | {q(v['e5a'], p)} | {q(v['e3l'], p)} | {q(v['e3a'], p)} | {q(v['e5p'], p)} |")
m = {k: sum(x) / len(x) for k, x in sums.items()}
P = m["pci"]
print(f"| mean (7) | {f(P)} | {f(m['e5l'])} | {q(m['e5l'], P)} | {f(m['e5a'])} | {q(m['e5a'], P)} | {q(m.get('e3l'), P)} | {q(m.get('e3a'), P)} | n/a |")
p3 = sum(av(E3, c, "pci_all") for c in C7[:3]) / 3
print(f"3-concept (tempo, piano, mood) ratio of means: e5 loc {sum(av(E5,c,'caa_loc') for c in C7[:3])/3/p3:.2f}, "
      f"e5p loc {sum(av(E5P,c,'caa_loc') for c in C7[:3])/3/p3:.2f}, e3 loc {sum(av(E3,c,'caa_loc') for c in C7[:3])/3/p3:.2f}")
print(f"loc vs all (e5): {100 * (m['e5l'] / m['e5a'] - 1):+.0f}%  (paper ACE +39%)")
print("CLAP mean (7): PCI-all", f(sum(av(E3, c, "pci_all", "clap") for c in C7) / 7), "e5 loc", f(sum(av(E5, c, "caa_loc", "clap") for c in C7) / 7),
      "e5 all", f(sum(av(E5, c, "caa_all", "clap") for c in C7) / 7))
(O / "eval_e5" / "e5_table.json").write_text(json.dumps({"means": m}, indent=1))
