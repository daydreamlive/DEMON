"""Render / score a named prompt set (many-knobs v2 queue items P0 and P4).

render (demonenv): one batched SA3 render of ``--set`` (a key of ``--sets`` json:
list of prompt strings) at ``--seed`` and ``--duration``, optionally steered by
several packs at once (``--steer PACK:GAIN``, alpha = GAIN x pack magnitude, every
pack's step/block vectors scaled by its own alpha and summed; ``--scale`` multiplies
every alpha: the stacking rules). Audio int16 mono -> ``--out`` npz, overwritten in
place.

score (evalenv): label columns (``--cols`` scorer.label ..., the scorers loaded once
through label_corpus), CLAP + MuQ cosine to ``--anchors`` texts, LPAPS vs ``--ref``
npz; -> ``--json`` (per-clip lists). With ``--windows 0:10 20:30`` every column is
computed per time window too (30 s hold check).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
SR = 44100


def _save_inplace(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "r+b" if path.exists() else "wb") as f:
        f.truncate(0)
        np.savez(f, **arrays)


def load_set(sets: str, name: str) -> list:
    d = json.loads(Path(sets).read_text(encoding="utf-8"))
    v = d[name]
    return [x["prompt"] if isinstance(x, dict) else str(x) for x in v]


def render(args) -> int:
    sys.path.insert(0, str(REPO))
    sys.path.insert(0, str(REPO / "scripts" / "tada"))
    import torch

    import sa3_tada_run as R
    from acestep.engine import sa3_tada

    torch.backends.cuda.matmul.allow_tf32 = True
    prompts = load_set(args.sets, args.set)
    sam = R._load_sam()
    vectors, used = {s: {} for s in range(8)}, []
    for spec in args.steer:
        path, gain = spec.rsplit(":", 1)
        pv = R._pack_vectors(path, 8)
        alpha = float(gain) * float(R._PACK["magnitude"]) * args.scale
        used.append({"pack": path, "gain": float(gain), "alpha": alpha})
        for s, d in pv.items():
            for b, v in d.items():
                vectors[s][b] = vectors[s].get(b, 0) + v * alpha
    out = Path(args.out)
    if used:
        with sa3_tada.steer_offline(sam, vectors, 1.0, hook="post_block_residual"):
            audio = sa3_tada.generate(sam, prompts, seed=args.seed, duration=args.duration, steps=8)
    else:
        audio = sa3_tada.generate(sam, prompts, seed=args.seed, duration=args.duration, steps=8)
    wav = (audio.mean(dim=1) * 32768.0).round().clamp(-32768, 32767).to(torch.int16).numpy()
    _save_inplace(out, wav=wav, sr=np.int64(SR), seed=np.int64(args.seed), set=np.array(args.set),
                  steer=np.array(json.dumps(used)), scale=np.float64(args.scale))
    print(f"rendered {args.set} x{len(prompts)} seed {args.seed} {args.duration}s steer {len(used)} -> {out}")
    return 0


def score(args) -> int:
    sys.path.insert(0, str(HERE))
    from screen_knobs import RealBackend

    be = RealBackend(None, {"label_workers": args.workers})
    z = np.load(args.npz)
    wav, sr = z["wav"], int(z["sr"])
    ref = np.load(args.ref)["wav"] if args.ref else None
    wins = [tuple(float(x) for x in w.split(":")) for w in args.windows] or [None]
    res = {"npz": args.npz, "ref": args.ref, "windows": args.windows}
    for w in wins:
        key = "all" if w is None else f"{w[0]:g}-{w[1]:g}"
        a = wav if w is None else wav[:, int(w[0] * sr):int(w[1] * sr)]
        r = {"cols": be.cols(args.cols, a, sr) if args.cols else {},
             "muq": be.anchors("muq", args.anchors, a, sr) if args.anchors else {},
             "clap": be.anchors("clap", args.anchors, a, sr) if args.anchors else {}}
        if ref is not None:
            rr = ref if w is None else ref[:, int(w[0] * sr):int(w[1] * sr)]
            r["lpaps"] = be.lpaps(rr, a, sr)
        res[key] = r
    Path(args.json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.json).write_text(json.dumps(res, default=float))
    print(f"scored {args.npz} -> {args.json}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("render")
    r.add_argument("--sets", required=True)
    r.add_argument("--set", required=True)
    r.add_argument("--seed", type=int, default=2115)
    r.add_argument("--duration", type=float, default=10.0)
    r.add_argument("--steer", nargs="*", default=[])
    r.add_argument("--scale", type=float, default=1.0)
    r.add_argument("--out", required=True)
    s = sub.add_parser("score")
    s.add_argument("--npz", required=True)
    s.add_argument("--ref")
    s.add_argument("--cols", nargs="*", default=[])
    s.add_argument("--anchors", nargs="*", default=[])
    s.add_argument("--windows", nargs="*", default=[])
    s.add_argument("--workers", type=int, default=12)
    s.add_argument("--json", required=True)
    args = ap.parse_args()
    return render(args) if args.cmd == "render" else score(args)


if __name__ == "__main__":
    raise SystemExit(main())
