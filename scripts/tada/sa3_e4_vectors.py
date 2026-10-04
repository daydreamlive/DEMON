"""E4b: large-N CAA vectors from the SAE activation cache.

Same balanced keyword classes as the E4 probe (``sa3_e4_probe.py``, rng
seed 0): per step x block, unit-norm (mean over positive-class clips of
the audio-token mean) minus (same for the negative class); positive pole
piano / fast / happy as in ``caa_e3``. Written in the ``caa_e3`` layout
(``{"vectors": {step: {block: [H]}}, "raw_norm": ...}``) so the SA3
lane's ``--vec-dir caa_e4`` loads it; only the cached blocks exist, so
use it with the ``loc`` site. ``e4_meta.json`` holds class counts and
the cosine to ``caa_e3`` per step and block.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sa3_e4_probe import LABELS, label_captions  # noqa: E402

CONCEPT_FILE = {"piano": "piano", "tempo": "tempo", "mood": "mood"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="D:/tada-replication/sae_cache/sa3")
    ap.add_argument("--captions", default="E:/Projects/tada-replication/data/musiccaps-public.csv")
    ap.add_argument("--caa", default="E:/Projects/tada-replication/sa3/caa_e3")
    ap.add_argument("--out", default="E:/Projects/tada-replication/sa3/caa_e4")
    ap.add_argument("--blocks", type=int, nargs="+", default=[3, 5, 6, 7])
    ap.add_argument("--cap", type=int, default=1000)
    args = ap.parse_args()
    rng = np.random.default_rng(0)
    cache = Path(args.cache)
    cfg = json.loads((cache / "config.json").read_text())
    caps_all = pd.read_csv(args.captions)["caption"].astype(str).tolist()
    caps = [caps_all[i] for i in cfg["caption_index"]]
    lab = {}
    for c in LABELS:
        pos, neg, npos, nneg = label_captions(caps, c, args.cap, rng)
        lab[c] = {"pos": set(pos), "neg": set(neg), "n_pos_all": npos, "n_neg_all": nneg}
    # sums[c][pole][step][block] -> (sum of clip means, count)
    sums = {c: {p: {} for p in ("pos", "neg")} for c in LABELS}
    for block in args.blocks:
        for path in sorted((cache / f"block_{block}").glob("shard_*.npy")):
            if path.name.endswith(".tmp.npy"):
                continue
            m = json.loads(path.with_suffix(".json").read_text())
            a = None
            for i, (p, s) in enumerate(zip(m["prompt"], m["step"])):
                for c, d in lab.items():
                    pole = "pos" if p in d["pos"] else "neg" if p in d["neg"] else None
                    if pole is None:
                        continue
                    if a is None:
                        a = np.load(path, mmap_mode="r")
                    x = np.asarray(a[i], dtype=np.float64).mean(0)
                    cur = sums[c][pole].setdefault(int(s), {}).get(block)
                    sums[c][pole][int(s)][block] = (x, 1) if cur is None else (cur[0] + x, cur[1] + 1)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    meta = {}
    for c, fname in CONCEPT_FILE.items():
        old = torch.load(Path(args.caa) / f"{fname}.pt", weights_only=False)
        vecs, raw, cos, counts = {}, {}, {}, {}
        for s in sorted(sums[c]["pos"]):
            vecs[s], raw[s], cos[s], counts[s] = {}, {}, {}, {}
            for b in args.blocks:
                (sp, np_), (sn, nn) = sums[c]["pos"][s][b], sums[c]["neg"][s][b]
                diff = torch.from_numpy(sp / np_ - sn / nn).float()
                n = float(diff.norm())
                v = diff / n
                vecs[s][b] = v
                raw[s][b] = n
                ov = old["vectors"][s][b].float()
                cos[s][b] = float(torch.dot(v, ov / ov.norm()))
                counts[s][b] = [np_, nn]
        torch.save({"vectors": vecs, "raw_norm": raw, "concept": c,
                    "pairs": None, "classes": counts[0][args.blocks[0]],
                    "source": "SAE cache: MusicCaps captions, balanced keyword classes, ARC medium, "
                              "8 steps, cross_attn_output, audio tokens only",
                    "blocks": args.blocks, "steps": len(vecs), "duration": 10.0, "audio_from": 64},
                   out / f"{fname}.pt")
        meta[c] = {"n_pos_all": lab[c]["n_pos_all"], "n_neg_all": lab[c]["n_neg_all"],
                   "n_per_class": counts[0][args.blocks[0]],
                   "cos_to_caa_e3": {str(s): {str(b): round(x, 4) for b, x in d.items()}
                                     for s, d in cos.items()},
                   "cos_mean": float(np.mean([x for d in cos.values() for x in d.values()]))}
        print(c, meta[c]["n_per_class"], round(meta[c]["cos_mean"], 3), flush=True)
    (out / "e4_meta.json").write_text(json.dumps(meta, indent=1))
    sig = Path(args.caa) / "sigmas.json"
    if sig.exists():
        (out / "sigmas.json").write_text(sig.read_text())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
