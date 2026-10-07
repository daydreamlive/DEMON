"""E5: large-N CAA (``caa_e5``) at all 24 blocks for the 7 SA3 concepts,
and probe-direction vectors (``caa_e5p``) for tempo/piano/mood at the
localised blocks, from the 24-block per-step means (``sa3_e5_capture.py``).

Classes: balanced caption keyword groups, poles as in ``caa_e3``
(positive first): piano / no piano, fast / slow, happy / sad, violin /
no violin, acoustic / electric guitar, classical / electronic, jazz /
rock. Two-sided concepts keep captions matching exactly one pole; presence
concepts take as negatives captions without the instrument word.
``caa_e5``: unit-norm mean difference per step x block. ``caa_e5p``:
unit-norm weight of an L2 logistic probe on the token means (inputs
standardised; the weight is mapped back to activation space by dividing
by the per-dimension std), sign so that the positive class scores higher.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PRESENCE = {  # concept: (positive regex, words that exclude a caption from the negative class)
    "piano": (r"\bpian(o|os|ist)\b", r"\b(keyboard|keys|synth)"),
    "violin": (r"\b(violin|violins|violinist|fiddle)\b", r"\b(string|strings|viola|cello)\b"),
}
TWO_SIDED = {
    "tempo": (r"\b(fast|up-?tempo|quick)\b", r"\b(slow|down-?tempo)\b"),
    "mood": (r"\b(happy|joyful|cheerful|uplifting|joyous)\b",
             r"\b(sad|melanchol\w*|sorrowful|depress\w*|mournful)\b"),
    "guitar_electronic": (r"\bacoustic guitar", r"\belectric guitar"),
    "electronic_music": (r"\b(classical|orchestra\w*|symphon\w*)\b",
                         r"\b(electronic|edm|techno|electro)\b"),
    "rock_genre": (r"\bjazz\w*\b", r"\brock\b"),
}
CONCEPTS = ["tempo", "piano", "mood", "violin", "guitar_electronic", "electronic_music", "rock_genre"]


def classes(caps, concept, cap_n, rng):
    low = [c.lower() for c in caps]
    if concept in PRESENCE:
        p_re, x_re = PRESENCE[concept]
        pos = [i for i, c in enumerate(low) if re.search(p_re, c)]
        neg = [i for i, c in enumerate(low) if not re.search(p_re, c) and not re.search(x_re, c)]
    else:
        p_re, n_re = TWO_SIDED[concept]
        p = [bool(re.search(p_re, c)) for c in low]
        n = [bool(re.search(n_re, c)) for c in low]
        pos = [i for i in range(len(low)) if p[i] and not n[i]]
        neg = [i for i in range(len(low)) if n[i] and not p[i]]
    m = min(len(pos), len(neg), cap_n)
    return (np.sort(rng.choice(pos, m, replace=False)), np.sort(rng.choice(neg, m, replace=False)),
            len(pos), len(neg))


def probe_direction(xp, xn, l2=1e-3, iters=100):
    x = torch.from_numpy(np.concatenate([xp, xn])).float()
    y = torch.cat([torch.ones(len(xp)), torch.zeros(len(xn))])
    mu, sd = x.mean(0), x.std(0) + 1e-6
    xs = (x - mu) / sd
    w = torch.zeros(x.shape[1], requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([w, b], max_iter=iters, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(xs @ w + b, y) + l2 * w.pow(2).sum()
        loss.backward()
        return loss
    opt.step(closure)
    v = (w.detach() / sd)
    acc = float((((xs @ w + b) > 0).float() == y).float().mean())
    return v / v.norm(), acc


def save(path, vecs, raw, concept, meta):
    torch.save({"vectors": vecs, "raw_norm": raw, "concept": concept, "pairs": None, **meta}, path)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--means", default="D:/tada-replication/e5_means")
    ap.add_argument("--captions", default="E:/Projects/tada-replication/data/musiccaps-public.csv")
    ap.add_argument("--caa", default="E:/Projects/tada-replication/sa3/caa_e3")
    ap.add_argument("--out", default="E:/Projects/tada-replication/sa3")
    ap.add_argument("--loc", type=int, nargs="+", default=[3, 5, 6, 7])
    ap.add_argument("--cap", type=int, default=1000)
    args = ap.parse_args()
    rng = np.random.default_rng(0)
    mdir = Path(args.means)
    info = json.loads((mdir / "meta.json").read_text())
    means = np.load(mdir / "means.npy", mmap_mode="r")
    n, steps, nb, _ = means.shape
    caps_all = pd.read_csv(args.captions)["caption"].astype(str).tolist()
    caps = [caps_all[i] for i in info["caption_index"]][:n]
    e5, e5p = Path(args.out) / "caa_e5", Path(args.out) / "caa_e5p"
    e5.mkdir(parents=True, exist_ok=True)
    e5p.mkdir(parents=True, exist_ok=True)
    meta5, meta5p = {}, {}
    src = ("SAE-lane capture: 5521 MusicCaps captions, ARC medium, 8 steps, cond pass, "
           "cross_attn_output audio-token means; balanced keyword classes")
    for c in CONCEPTS:
        pos, neg, npos, nneg = classes(caps, c, args.cap, rng)
        xp = np.asarray(means[pos], dtype=np.float64)  # [P, steps, blocks, d]
        xn = np.asarray(means[neg], dtype=np.float64)
        old = torch.load(Path(args.caa) / f"{c}.pt", weights_only=False)["vectors"]
        vecs, raw, cos = {}, {}, {}
        for s in range(steps):
            vecs[s], raw[s], cos[s] = {}, {}, {}
            for b in range(nb):
                diff = torch.from_numpy(xp[:, s, b].mean(0) - xn[:, s, b].mean(0)).float()
                nrm = float(diff.norm())
                vecs[s][b] = diff / nrm
                raw[s][b] = nrm
                ov = old[s][b].float()
                cos[s][b] = round(float(torch.dot(vecs[s][b], ov / ov.norm())), 4)
        m = {"classes": [len(pos), len(neg)], "source": src, "steps": steps, "duration": 10.0,
             "audio_from": 64, "blocks": list(range(nb))}
        save(e5 / f"{c}.pt", vecs, raw, c, m)
        allc = [x for d in cos.values() for x in d.values()]
        locc = [cos[s][b] for s in cos for b in args.loc]
        meta5[c] = {"n_pos_all": npos, "n_neg_all": nneg, "n_per_class": len(pos),
                    "cos_mean_all_blocks": float(np.mean(allc)), "cos_mean_loc": float(np.mean(locc)),
                    "cos_to_caa_e3": {str(s): {str(b): x for b, x in d.items()} for s, d in cos.items()}}
        print(f"caa_e5 {c}: {len(pos)}/class ({npos} pos, {nneg} neg), cos e3 loc {np.mean(locc):.3f} "
              f"all {np.mean(allc):.3f}", flush=True)
        if c in ("tempo", "piano", "mood"):
            pv, praw, pacc, pcos, pcos5 = {}, {}, {}, {}, {}
            for s in range(steps):
                pv[s], praw[s], pacc[s], pcos[s], pcos5[s] = {}, {}, {}, {}, {}
                for b in args.loc:
                    v, acc = probe_direction(xp[:, s, b], xn[:, s, b])
                    pv[s][b] = v
                    praw[s][b] = 1.0
                    pacc[s][b] = round(acc, 4)
                    ov = old[s][b].float()
                    pcos[s][b] = round(float(torch.dot(v, ov / ov.norm())), 4)
                    pcos5[s][b] = round(float(torch.dot(v, vecs[s][b])), 4)
            save(e5p / f"{c}.pt", pv, praw, c, {**m, "blocks": args.loc,
                                                 "method": "unit-norm L2 logistic probe weight"})
            meta5p[c] = {"n_per_class": len(pos), "train_acc": {str(s): {str(b): a for b, a in d.items()}
                                                                for s, d in pacc.items()},
                         "cos_to_caa_e3": {str(s): {str(b): x for b, x in d.items()} for s, d in pcos.items()},
                         "cos_to_caa_e5": {str(s): {str(b): x for b, x in d.items()} for s, d in pcos5.items()},
                         "cos_mean_e3": float(np.mean([x for d in pcos.values() for x in d.values()])),
                         "cos_mean_e5": float(np.mean([x for d in pcos5.values() for x in d.values()]))}
            print(f"caa_e5p {c}: cos e3 {meta5p[c]['cos_mean_e3']:.3f} cos e5 {meta5p[c]['cos_mean_e5']:.3f}",
                  flush=True)
    sig = Path(args.caa) / "sigmas.json"
    for d, mt in ((e5, meta5), (e5p, meta5p)):
        (d / ("e5_meta.json" if d == e5 else "e5p_meta.json")).write_text(json.dumps(mt, indent=1))
        if sig.exists():
            (d / "sigmas.json").write_text(sig.read_text())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
