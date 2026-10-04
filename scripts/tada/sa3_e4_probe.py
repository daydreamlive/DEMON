"""E4 probe: is a text concept linearly present at SA3's cross-attention
output, at the mean level or only per token?

CPU only, on the SAE activation cache (``sa3_sae.py cache``: MusicCaps
captions, served SA3 medium, every step, audio tokens only). Captions are
labelled by keyword (piano vs no piano, fast vs slow, happy vs sad) and
balanced. Per block x step:

* mean level: L2 logistic probe, 5-fold accuracy, on the token-averaged
  activation; effect size along the mean difference (difference of class
  means of the projection over the pooled within-class std);
* token level: the same probe on individual tokens (folds grouped by
  caption so tokens of one clip never sit on both sides);
* cosine between the mean-difference direction and the CAA vector of the
  same concept, block and step (``caa_e3/<concept>.pt``).
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch

LABELS = {
    "piano": (r"\bpian(o|os|ist)\b", r"\b(keyboard|keys|synth)"),
    "tempo": (r"\b(fast|up-?tempo|quick)\b", r"\b(slow|down-?tempo)\b"),
    "mood": (r"\b(happy|joyful|cheerful|uplifting|joyous)\b",
             r"\b(sad|melanchol\w*|sorrowful|depress\w*|mournful)\b"),
}


def label_captions(caps, concept, cap_n, rng):
    pos_re, neg_re = LABELS[concept]
    low = [c.lower() for c in caps]
    if concept == "piano":
        pos = [i for i, c in enumerate(low) if re.search(pos_re, c)]
        neg = [i for i, c in enumerate(low) if not re.search(pos_re, c) and not re.search(neg_re, c)]
    else:
        p = [bool(re.search(pos_re, c)) for c in low]
        n = [bool(re.search(neg_re, c)) for c in low]
        pos = [i for i in range(len(low)) if p[i] and not n[i]]
        neg = [i for i in range(len(low)) if n[i] and not p[i]]
    m = min(len(pos), len(neg), cap_n)
    return (sorted(rng.choice(pos, m, replace=False).tolist()),
            sorted(rng.choice(neg, m, replace=False).tolist()), len(pos), len(neg))


def logreg_cv(x, y, groups, folds=5, l2=1e-3, iters=60, seed=0):
    """Mean 5-fold accuracy of an L2 logistic probe (standardised inputs,
    LBFGS), folds split by ``groups``."""
    x = torch.as_tensor(x, dtype=torch.float32)
    y = torch.as_tensor(y, dtype=torch.float32)
    ug = np.unique(groups)
    perm = np.random.default_rng(seed).permutation(ug)
    fold_of = {g: i % folds for i, g in enumerate(perm)}
    fid = np.array([fold_of[g] for g in groups])
    accs = []
    for f in range(folds):
        te = torch.from_numpy(fid == f)
        tr = ~te
        mu, sd = x[tr].mean(0), x[tr].std(0) + 1e-6
        xtr, xte = (x[tr] - mu) / sd, (x[te] - mu) / sd
        w = torch.zeros(x.shape[1], requires_grad=True)
        b = torch.zeros(1, requires_grad=True)
        opt = torch.optim.LBFGS([w, b], max_iter=iters, line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad()
            loss = torch.nn.functional.binary_cross_entropy_with_logits(xtr @ w + b, y[tr]) \
                + l2 * w.pow(2).sum()
            loss.backward()
            return loss
        opt.step(closure)
        with torch.no_grad():
            accs.append(float(((xte @ w + b > 0).float() == y[te]).float().mean()))
    return float(np.mean(accs))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="D:/tada-replication/sae_cache/sa3")
    ap.add_argument("--captions", default="E:/Projects/tada-replication/data/musiccaps-public.csv")
    ap.add_argument("--caa", default="E:/Projects/tada-replication/sa3/caa_e3")
    ap.add_argument("--blocks", type=int, nargs="+", default=[3, 5, 6, 7])
    ap.add_argument("--cap", type=int, default=1000, help="captions per class at most")
    ap.add_argument("--tokens", type=int, default=20000, help="tokens per class per step")
    ap.add_argument("--out", default="D:/tada-replication/e4_probe.json")
    args = ap.parse_args()
    torch.set_num_threads(max(1, torch.get_num_threads()))
    rng = np.random.default_rng(0)
    cache = Path(args.cache)
    cfg = json.loads((cache / "config.json").read_text())
    caps_all = pd.read_csv(args.captions)["caption"].astype(str).tolist()
    order = cfg["caption_index"]
    caps = [caps_all[i] for i in order]  # cached prompt index -> caption
    lab = {}
    for c in LABELS:
        pos, neg, npos, nneg = label_captions(caps, c, args.cap, rng)
        lab[c] = {"pos": set(pos), "neg": set(neg), "n_pos_all": npos, "n_neg_all": nneg, "n": len(pos)}
        print(f"{c}: {npos} pos / {nneg} neg captions, balanced {len(pos)} per class", flush=True)
    caa = {c: torch.load(Path(args.caa) / f"{c}.pt", weights_only=False)["vectors"] for c in LABELS}
    results = {"labels": {c: {k: v for k, v in d.items() if k.startswith("n_") or k == "n"} for c, d in lab.items()},
               "rows": []}
    want = set().union(*[d["pos"] | d["neg"] for d in lab.values()])
    for block in args.blocks:
        means, toks = {}, {}  # (prompt, step) -> mean [d]; (prompt, step) -> sampled tokens
        shards = sorted(p for p in (cache / f"block_{block}").glob("shard_*.npy")
                        if not p.name.endswith(".tmp.npy"))
        need = {c: int(np.ceil(args.tokens / d["n"])) for c, d in lab.items()}
        per_prompt = {}
        for c, d in lab.items():
            for p in d["pos"] | d["neg"]:
                per_prompt[p] = max(per_prompt.get(p, 0), need[c])
        for path in shards:
            m = json.loads(path.with_suffix(".json").read_text())
            idx = [i for i, p in enumerate(m["prompt"]) if p in want]
            if not idx:
                continue
            a = np.load(path, mmap_mode="r")
            sub = np.asarray(a[idx], dtype=np.float32)
            for j, i in enumerate(idx):
                key = (m["prompt"][i], m["step"][i])
                means[key] = sub[j].mean(0)
                sel = rng.choice(sub.shape[1], min(per_prompt[key[0]], sub.shape[1]), replace=False)
                toks[key] = sub[j][sel].astype(np.float16)
        steps = sorted({s for _, s in means})
        for c, d in lab.items():
            for s in steps:
                pp = [p for p in sorted(d["pos"]) if (p, s) in means]
                nn = [p for p in sorted(d["neg"]) if (p, s) in means]
                xm = np.stack([means[(p, s)] for p in pp + nn])
                ym = np.array([1] * len(pp) + [0] * len(nn))
                acc_mean = logreg_cv(xm, ym, np.array(pp + nn))
                acc_shuf = logreg_cv(xm, np.random.default_rng(1).permutation(ym), np.array(pp + nn))
                mu_p, mu_n = xm[ym == 1].mean(0), xm[ym == 0].mean(0)
                diff = mu_p - mu_n
                u = diff / (np.linalg.norm(diff) + 1e-12)
                proj = xm @ u
                sd = np.sqrt(0.5 * (proj[ym == 1].var() + proj[ym == 0].var()))
                effect = float((proj[ym == 1].mean() - proj[ym == 0].mean()) / (sd + 1e-12))
                q = need[c]
                xt = np.concatenate([toks[(p, s)][:q] for p in pp + nn]).astype(np.float32)
                gt = np.concatenate([[p] * len(toks[(p, s)][:q]) for p in pp + nn])
                yt = np.concatenate([[lab_] * len(toks[(p, s)][:q]) for p, lab_ in
                                     [(p, 1) for p in pp] + [(p, 0) for p in nn]])
                acc_tok = logreg_cv(xt, yt, gt)
                v = caa[c].get(s, {}).get(block) if isinstance(caa[c].get(s), dict) else None
                cos = float(np.dot(u, v.float().numpy() / (v.float().norm().item() + 1e-12))) \
                    if v is not None else float("nan")
                row = {"concept": c, "block": block, "step": s, "n_per_class": len(pp),
                       "mean_acc": acc_mean, "mean_acc_shuffled": acc_shuf, "token_acc": acc_tok, "effect": effect,
                       "diff_norm": float(np.linalg.norm(diff)), "cos_caa": cos,
                       "tokens_per_class": int(len(xt) // 2)}
                results["rows"].append(row)
                print(json.dumps(row), flush=True)
        del means, toks
        Path(args.out).write_text(json.dumps(results, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
