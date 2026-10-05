"""Descriptor scoring of bench sweep directories (CPU, DEMON venv) and one
summary row per directory.

For every ``<eval_sub>/<method>_<site>_<concept>[_bNN]`` dir with
``alpha_*/audios.npz``: the five descriptors (:mod:`proxies`) on every clip
-> ``descriptors.csv`` (alpha, clip, descriptor values). Per alpha:
mean LPAPS (``protocol_results/lpaps.csv`` from ``sa3_tada_score.py
protocol --lpaps-only``, if present) and the mean paired delta of every
descriptor vs alpha 0 (same prompt, same seed). Summary row (appended to
``--summary``):

* ``target_delta_<a>``: knob-signed delta of the concept's own descriptor
  (density: sign -1, so positive = fewer onsets = sparser)
* ``gain_per_lpaps``: mean over nonzero alphas of
  ``sign(alpha) * target_delta / lpaps`` (descriptor gain per unit distortion,
  both directions; needs LPAPS)
* ``x_<descriptor>_pos/neg``: every descriptor's delta at the largest +/- alpha
  (selectivity / cross-effects)

    python scripts/steering_bench/score_descriptors.py --dirs <out>/<eval_sub>/caa_loc_bright_b*
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from concepts import DESCRIPTORS, SPEC  # noqa: E402


def _alpha(p: Path) -> float:
    return float(p.name[len("alpha_"):])


def measure_dir(d: Path, force: bool = False) -> dict:
    """``{alpha: [n_clips, 5]}``, cached in ``descriptors.csv``."""
    from desc_pool import measure_clips

    cache = d / "descriptors.csv"
    alphas = sorted(_alpha(p) for p in d.glob("alpha_*") if (p / "audios.npz").exists())
    out = {}
    if cache.exists() and not force:
        rows = list(csv.DictReader(cache.open()))
        for a in alphas:
            r = [x for x in rows if float(x["alpha"]) == a]
            if r:
                out[a] = np.array([[float(x[k]) for k in DESCRIPTORS] for x in r])
    todo = [a for a in alphas if a not in out]
    for a in todo:
        z = np.load(d / f"alpha_{a}" / "audios.npz")
        sr = int(z["sr"])
        out[a] = measure_clips(list(z["audio"]), sr)
    if todo:
        with cache.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["alpha", "clip", *DESCRIPTORS])
            for a in sorted(out):
                for i, v in enumerate(out[a]):
                    w.writerow([a, i, *[f"{x:.6g}" for x in v]])
    return out


def lpaps_of(d: Path) -> dict:
    f = d / "protocol_results" / "lpaps.csv"
    if not f.exists():
        return {}
    return {float(r["alpha"]): float(r["mean"]) for r in csv.DictReader(f.open())}


def summarize(d: Path, force: bool = False) -> dict:
    name = d.name
    m = re.match(r"^(?P<method>[^_]+)_(?P<site>[^_]+)_(?P<concept>[a-z]+)(?:_b(?P<block>\d+))?", name)
    concept = m["concept"]
    sp = SPEC[concept]
    ti = DESCRIPTORS.index(sp["descriptor"])
    vals = measure_dir(d, force)
    lp = lpaps_of(d)
    if 0.0 not in vals:
        raise SystemExit(f"{d}: no alpha_0.0")
    base = vals[0.0]
    sw = json.loads((d / "sweep.json").read_text()) if (d / "sweep.json").exists() else {}
    row = {"dir": name, "concept": concept, "site": m["site"],
           "block": m["block"] if m["block"] is not None else ",".join(map(str, sw.get("blocks", []))),
           "descriptor": sp["descriptor"], "n_clips": int(len(base)),
           "base_" + sp["descriptor"]: round(float(base[:, ti].mean()), 6)}
    gains = []
    for a in sorted(vals):
        if a == 0.0:
            continue
        delta = vals[a] - base                                   # paired per clip
        td = sp["sign"] * float(delta[:, ti].mean())
        row[f"target_delta_{a:g}"] = round(td, 6)
        row[f"target_delta_sd_{a:g}"] = round(float(delta[:, ti].std(ddof=1)) if len(delta) > 1 else 0.0, 6)
        if a in lp:
            row[f"lpaps_{a:g}"] = round(lp[a], 4)
            if lp[a] > 0:
                gains.append(np.sign(a) * td / lp[a])
    row["gain_per_lpaps"] = round(float(np.mean(gains)), 6) if gains else ""
    for tagname, a in (("pos", max(vals)), ("neg", min(vals))):
        if a == 0.0:
            continue
        delta = (vals[a] - base).mean(0)
        for k, dk in zip(DESCRIPTORS, delta):
            row[f"x_{k}_{tagname}"] = round(float(dk), 6)
    return row


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dirs", nargs="+", required=True, type=Path)
    ap.add_argument("--summary", type=Path, default=None, help="CSV to append rows to")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    rows = []
    for d in args.dirs:
        if not d.is_dir() or not any(d.glob("alpha_*")):
            continue
        r = summarize(d, args.force)
        rows.append(r)
        print(json.dumps(r), flush=True)
    if args.summary and rows:
        keys = []
        for r in rows:
            keys += [k for k in r if k not in keys]
        old = list(csv.DictReader(args.summary.open())) if args.summary.exists() else []
        done = {r["dir"] for r in rows}
        old = [r for r in old if r["dir"] not in done]
        for r in old:
            keys += [k for k in r if k not in keys]
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        with args.summary.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for r in old + rows:
                w.writerow(r)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
