"""Descriptor-ridge steering directions (v3 smoke, deliverable A; CPU only).

For each descriptor (onset_rate, centroid, lra) a ridge regression from the
step-mean per-block audio-token mean (post_block_residual, the same feature
make_packs.py variant a reduces to) to the standardised descriptor, over the
population the v2 descriptor concept used. The coefficient vector, unit
normalised, becomes a format-1 production pack (``acestep.steering.packs``,
the same writer and reader as make_packs.py) with ``provenance.method =
"ridge"``. A second vector per descriptor is fit to the descriptor
residualised (OLS) against the catalogue_v2_notes.md confound set
(centroid, bass_db, flux_mean, lufs, perc_ratio, minus itself).

Block: the v2 variant a best block of the corpus concept with the same
label column (``dirs/index.csv``: bright = desc.centroid, onset_rate =
desc.onset_rate, dynamic = dyn.lra). ``--variant-b`` also fits the summed
top-3 block variant (one joint ridge over the concept's top-3 v2 blocks,
rows scaled relative to the best block, as build_directions.py variant b)
into ``<out>/variant_b/``.

Ridge alpha: 5-fold CV (GroupKFold on prompts.json ``base_id``, so the 4
seeds of a prompt never straddle folds) over a log grid, solved per fold by
eigendecomposition of the centred Gram matrix (exactly sklearn Ridge with an
intercept); the final vector is ``sklearn.linear_model.Ridge(alpha)`` on all
rows. Sign: positive knob = raises the descriptor.

Pack scale (knob semantics as make_packs.py): ``norm`` = projected gap
between the top and bottom quartile means of the (raw, not residualised)
descriptor along the unit vector, ``magnitude`` = norm x 0.1, so knob 10 =
one quartile gap, comparable with the v2 class-mean packs.

Fit report (``fit_report.json`` / ``.md``): CV R^2 (+ per-alpha curve, CV
R^2 at every stored block), n, block(s), cos(ridge, ridge-residualised),
cos(ridge, shipped control pack) (control blocks differ; also the ridge
refit at the control's block when it was captured), cos(ridge, v2 class-mean
direction of the same descriptor), and for lra the shipped packs whose label
column correlates best with dyn.lra (the dense_arrangement proxy check).

    python scripts/steering_bench/ridge_directions.py \\
        --cap E:/Projects/DEMON/steering-bench/many_knobs_v2/corpus \\
        --labels E:/Projects/DEMON/steering-bench/v3_smoke/labels_v2_subset.csv \\
        --controls E:/Projects/DEMON/steering-bench/many_knobs_v2/packs_final/sa3/medium \\
        --out E:/Projects/DEMON/steering-bench/v3_smoke/packs --variant-b
"""

from __future__ import annotations

import argparse
import csv
import datetime as _dt
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))
from build_directions import block_ids, load_corpus, load_labels, population_mask  # noqa: E402

KNOB_UNIT = 0.1
CONFOUNDS = ("desc.centroid", "desc.bass_db", "dyn.flux_mean", "dyn.lufs", "desc.perc_ratio")
#: descriptor -> (label column as score_descriptors.py / label_corpus.py write it,
#:                v2 corpus concept with that label column (block choice), shipped control knob)
DESCRIPTORS = {
    "onset_rate": ("desc.onset_rate", "onset_rate", "percussive"),
    "centroid": ("desc.centroid", "bright", "bright"),
    "lra": ("dyn.lra", "dynamic", "dense_arrangement"),
}
ALPHAS = np.logspace(-1, 6, 15)


def stepmean_blocks(means, need: list, rows: np.ndarray, chunk: int = 512) -> dict:
    """``{local block: [len(rows), H] float32}`` step-mean features, one pass over the memmap."""
    out = {b: np.empty((len(rows), means.shape[-1]), dtype=np.float32) for b in need}
    for s in range(0, len(rows), chunk):
        r = rows[s:s + chunk]
        x = np.asarray(means[r[0]:r[-1] + 1], dtype=np.float32)[r - r[0]]   # [c, steps, blocks, H]
        for b in need:
            out[b][s:s + len(r)] = x[:, :, b, :].mean(1)
    return out


def group_folds(groups: np.ndarray, k: int = 5, seed: int = 0) -> np.ndarray:
    """Fold id per row; whole groups (prompt base_id) per fold, shuffled."""
    u = np.unique(groups)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(u))
    gf = dict(zip(u[perm], np.arange(len(u)) % k))
    return np.array([gf[g] for g in groups])


def cv_ridge(x: np.ndarray, y: np.ndarray, folds: np.ndarray, alphas=ALPHAS) -> np.ndarray:
    """Pooled out-of-fold R^2 per alpha (ridge with intercept, by eigendecomposition)."""
    pred = np.zeros((len(alphas), len(y)))
    for f in np.unique(folds):
        tr, te = folds != f, folds == f
        xm, ym = x[tr].mean(0), y[tr].mean()
        xt = (x[tr] - xm).astype(np.float64)
        w, v = np.linalg.eigh(xt.T @ xt)
        xty = v.T @ (xt.T @ (y[tr] - ym))
        xe = (x[te] - xm).astype(np.float64) @ v
        for i, a in enumerate(alphas):
            pred[i, te] = xe @ (xty / (w + a)) + ym
    ss = ((y - y.mean()) ** 2).sum()
    return 1.0 - ((pred - y[None]) ** 2).sum(1) / ss


def fit(x: np.ndarray, y: np.ndarray, folds: np.ndarray) -> dict:
    from sklearn.linear_model import Ridge

    r2 = cv_ridge(x, y, folds)
    a = float(ALPHAS[int(np.argmax(r2))])
    m = Ridge(alpha=a).fit(x.astype(np.float64), y)
    coef = m.coef_.astype(np.float64)
    return {"alpha": a, "cv_r2": float(r2.max()), "cv_curve": {f"{al:.3g}": round(float(v), 5) for al, v in zip(ALPHAS, r2)},
            "coef": coef, "in_sample_r2": float(m.score(x.astype(np.float64), y))}


def residualise(y: np.ndarray, conf: np.ndarray) -> tuple:
    c = np.column_stack([np.ones(len(y)), (conf - conf.mean(0)) / conf.std(0)])
    beta, *_ = np.linalg.lstsq(c, y, rcond=None)
    res = y - c @ beta
    return res, float(1 - res.var() / y.var())


def zs(v):
    return (v - v.mean()) / v.std()


def unit(v):
    return v / max(float(np.linalg.norm(v)), 1e-12)


def cos(a, b) -> float:
    a, b = np.asarray(a, np.float64).ravel(), np.asarray(b, np.float64).ravel()
    return float(a @ b / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-12))


def quartile_gap(xb: np.ndarray, raw: np.ndarray, u: np.ndarray) -> tuple:
    lo, hi = np.quantile(raw, [0.25, 0.75])
    p = xb @ u
    return float(p[raw >= hi].mean() - p[raw <= lo].mean()), float(p.std(ddof=1))


def write(pack_cls, save_pack, load_pack, path, *, name, block, blocks, vec, norm, label, blurb, prov):
    import torch

    pack = pack_cls(family="sa3", checkpoint="medium", block=int(block), hidden_size=int(vec.shape[-1]),
                    blocks=[int(b) for b in blocks], name=name, vector=torch.from_numpy(vec.astype(np.float32)),
                    label=label, blurb=blurb, hook="post_block_residual", method="caa_diff_means",
                    norm=float(norm), magnitude=float(norm) * KNOB_UNIT,
                    policy={"kind": "range", "start": 0.0, "end": 1.0}, provenance=prov)
    p = save_pack(pack, path)
    back = load_pack(p)
    c = cos(back.vector.numpy(), vec)
    if back.block != int(block) or c < 0.99999 or back.provenance.get("method") != "ridge":
        raise SystemExit(f"{p}: re-read mismatch (block {back.block}, cos {c:.6f})")
    return p


#: --per-sign (v3 smoke batch S1): subsets of the fit rows by raw-descriptor quantile, and the render sides they serve.
#: kind -> (row rule, render sides); terciles / median over the descriptor's own fit rows.
PER_SIGN = {
    "ridge": ("all rows (symmetric, first-smoke comparison row)", [1, -1]),
    "ridge_up": ("middle + high tercile (high vs middle)", [1]),
    "ridge_down": ("low + middle tercile (low vs middle)", [-1]),
    "ridge_lowtail": ("bottom half only", [-1]),
}


def per_sign_rows(kind: str, raw: np.ndarray) -> np.ndarray:
    t1, t2 = np.quantile(raw, [1 / 3, 2 / 3])
    if kind == "ridge":
        return np.ones(len(raw), bool)
    if kind == "ridge_up":
        return raw > t1
    if kind == "ridge_down":
        return raw <= t2
    if kind == "ridge_lowtail":
        return raw <= np.median(raw)
    raise KeyError(kind)


def per_sign(args) -> int:
    """S1: symmetric / per-sign / low-tail ridge vectors at the variant a best block, format-1 packs in ``--out``.

    Each subset's target is the descriptor z-scored within the subset; the fit is the same CV ridge as the symmetric
    vector. Provenance carries ``kind`` and ``sides`` (the knob signs the vector is meant for: up = +1, down / low-tail
    = -1; sign +1 still raises the descriptor), which smoke_drive.py reads to render only those sides."""
    from acestep.steering.packs import SteeringPack, load_pack, save_pack

    t0 = time.time()
    cap, out = Path(args.cap), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    means, meta, ids, cats = load_corpus(cap)
    bid = block_ids(meta, means.shape[2])
    labels = load_labels(Path(args.labels) if args.labels else cap / "labels" / "merged.parquet", ids)
    with open(cap / "dirs" / "index.csv", newline="", encoding="utf-8") as f:
        index = {r["name"]: r for r in csv.DictReader(f)}
    pr = json.loads((cap / "prompts.json").read_text())[: means.shape[0]]
    base = np.array([int(r.get("base_id", r["id"])) for r in pr])
    today = _dt.date.today().isoformat()
    report = {"date": today, "cap": str(cap), "mode": "per-sign (v3 smoke batch S1)",
              "kinds": {k: v[0] for k, v in PER_SIGN.items()}, "descriptors": {}, "packs": []}
    for d in args.descriptors:
        col, concept, control = DESCRIPTORS[d]
        row = index[concept]
        pop = row.get("population", "music") or "music"
        cols = [col] + [c for c in CONFOUNDS if c != col]   # same row set as the first smoke
        ok = population_mask(pop, cats) & np.isfinite(labels[cols].to_numpy(dtype=np.float64)).all(1)
        rows = np.flatnonzero(ok)
        best = int(row["best_block"])
        xb = stepmean_blocks(means, [bid.index(best)], rows)[bid.index(best)]
        raw = labels.iloc[rows][col].to_numpy(dtype=np.float64)
        groups = base[rows]
        rep = {"label_col": col, "block": best, "n": int(len(rows)), "control": control,
               "terciles": [float(x) for x in np.quantile(raw, [1 / 3, 2 / 3])], "median": float(np.median(raw))}
        units = {}
        for kind, (rule, sides) in PER_SIGN.items():
            m = per_sign_rows(kind, raw)
            xs, ys = xb[m], zs(raw[m])
            fr = fit(xs, ys, group_folds(groups[m], args.folds))
            u = unit(fr["coef"])
            units[kind] = u
            gap, pstd = quartile_gap(xb, raw, u)
            name = f"{kind}_{d}"
            prov = {"tool": "scripts/steering_bench/ridge_directions.py --per-sign", "method": "ridge", "kind": kind,
                    "sides": sides, "rows_rule": rule, "date": today, "label_col": col, "confounds": [],
                    "sign": 1, "sign_rule": "positive knob raises the descriptor", "population": pop,
                    "n": int(m.sum()), "blocks": [best], "ridge_alpha": fr["alpha"], "cv_r2": round(fr["cv_r2"], 5),
                    "in_sample_r2": round(fr["in_sample_r2"], 5), "capture": str(cap), "knob_unit": KNOB_UNIT,
                    "norm_rule": "projected quartile gap of the raw descriptor over all fit rows",
                    "proj_std_at_block": round(pstd, 5), "control": control, "calibrated_gain": None,
                    "caveat": "estimated at 10 s ARC sam.generate, audio tokens only; unscreened (v3 smoke S1)"}
            path = write(SteeringPack, save_pack, load_pack, out / f"{name}.safetensors", name=name, block=best,
                         blocks=[], vec=u, norm=gap, label=f"{d} ({kind})", blurb=f"ridge direction ({kind}) on {col}",
                         prov=prov)
            rep[kind] = {"n": int(m.sum()), "alpha": fr["alpha"], "cv_r2": round(fr["cv_r2"], 4),
                         "in_sample_r2": round(fr["in_sample_r2"], 4), "norm": round(gap, 4), "sides": sides,
                         "pack": str(path)}
            report["packs"].append(str(path))
            print(f"[{d}/{kind}] b{best} n {m.sum()} alpha {fr['alpha']:.3g} CV R2 {fr['cv_r2']:.4f} -> {path}", flush=True)
        rep["cos"] = {f"{a}|{b}": round(cos(units[a], units[b]), 4) for i, a in enumerate(units) for b in list(units)[i + 1:]}
        report["descriptors"][d] = rep
    (out / "fit_report.json").write_text(json.dumps(report, indent=1))
    md = ["# v3 smoke S1: per-sign ridge fit report", "", f"{today}; corpus {cap}; kinds: " +
          "; ".join(f"{k} = {v[0]}" for k, v in PER_SIGN.items()), "",
          "| descriptor | block | kind | n | CV R^2 | sides | cos vs symmetric |", "|---|---|---|---|---|---|---|"]
    for d, r in report["descriptors"].items():
        for kind in PER_SIGN:
            c = 1.0 if kind == "ridge" else r["cos"].get(f"ridge|{kind}")
            md.append(f"| {d} | {r['block']} | {kind} | {r[kind]['n']} | {r[kind]['cv_r2']} | {r[kind]['sides']} | {c} |")
    md += ["", "cos(up, down) / cos(up, lowtail) / cos(down, lowtail): " + "; ".join(
        f"{d} {r['cos']['ridge_up|ridge_down']} / {r['cos']['ridge_up|ridge_lowtail']} / {r['cos']['ridge_down|ridge_lowtail']}"
        for d, r in report["descriptors"].items()), ""]
    (out / "fit_report.md").write_text("\n".join(md), encoding="utf-8")
    print(f"wrote {out / 'fit_report.md'} in {time.time() - t0:.0f} s", flush=True)
    return 0


def _load_fit_rows(cap: Path, labels_path, d: str):
    """(means, block ids, row indices, raw descriptor, base ids, population, label col) for descriptor ``d``,
    the same row rule as the first smoke (population of the v2 concept, finite descriptor + confounds)."""
    col, concept, control = DESCRIPTORS[d]
    means, meta, ids, cats = load_corpus(cap)
    bid = block_ids(meta, means.shape[2])
    labels = load_labels(Path(labels_path) if labels_path else cap / "labels" / "merged.parquet", ids)
    pr = json.loads((cap / "prompts.json").read_text())[: means.shape[0]]
    base = np.array([int(r.get("base_id", r["id"])) for r in pr])
    cols = [col] + [c for c in CONFOUNDS if c != col]
    return means, bid, labels, base, cats, cols


def extras(args) -> int:
    """S1 additions (lit review Q1, Q6). Packs are up-side only (provenance sides [1], kind names the variant).

    --reg-path: per descriptor at the first-smoke block on the v2 corpus, ridge at the CV lambda, 10x, 100x, and the
      lambda -> inf limit (unit X_c^T y, the covariance / mean-difference-style direction).
    --block-sweep B...: centroid and onset_rate (``--sweep-descriptors``) at the CV lambda of each block, fit on
      ``--sweep-cap`` (default: the v1 corpus, the only local capture with all 24 blocks; the v2 corpus holds
      blocks 9-19, 22, 23 only), so every block of the sweep shares one row set."""
    from acestep.steering.packs import SteeringPack, load_pack, save_pack

    t0 = time.time()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    today = _dt.date.today().isoformat()
    report = {"date": today, "mode": "extras (v3 smoke S1, lit review Q1/Q6)", "reg_path": {}, "block_sweep": {}, "packs": []}

    def emit(name, d, kind, block, u, xb, raw, prov_extra, cap, pop, n):
        col, _concept, control = DESCRIPTORS[d]
        gap, pstd = quartile_gap(xb, raw, u)
        prov = {"tool": "scripts/steering_bench/ridge_directions.py --extras", "method": "ridge", "kind": kind,
                "sides": [1], "date": today, "label_col": col, "confounds": [], "sign": 1,
                "sign_rule": "positive knob raises the descriptor", "population": pop, "n": int(n), "blocks": [block],
                "capture": str(cap), "knob_unit": KNOB_UNIT, "proj_std_at_block": round(pstd, 5), "control": control,
                "calibrated_gain": None, "caveat": "v3 smoke S1 extras; unscreened", **prov_extra}
        p = write(SteeringPack, save_pack, load_pack, out / f"{name}.safetensors", name=name, block=block, blocks=[],
                  vec=u, norm=gap, label=f"{d} ({kind})", blurb=f"ridge direction ({kind}) on {col}", prov=prov)
        report["packs"].append(str(p))
        return p

    def rows_for(cap, d):
        means, bid, labels, base, cats, cols = _load_fit_rows(cap, args.labels if cap == Path(args.cap) else
                                                               args.sweep_labels, d)
        with open(Path(args.cap) / "dirs" / "index.csv", newline="", encoding="utf-8") as f:
            row = {r["name"]: r for r in csv.DictReader(f)}[DESCRIPTORS[d][1]]
        pop = row.get("population", "music") or "music"
        ok = population_mask(pop, cats) & np.isfinite(labels[cols].to_numpy(dtype=np.float64)).all(1)
        rows = np.flatnonzero(ok)
        raw = labels.iloc[rows][DESCRIPTORS[d][0]].to_numpy(dtype=np.float64)
        return means, bid, rows, raw, base[rows], pop, int(row["best_block"])

    if args.reg_path:
        cap = Path(args.cap)
        for d in args.descriptors:
            means, bid, rows, raw, groups, pop, best = rows_for(cap, d)
            xb = stepmean_blocks(means, [bid.index(best)], rows)[bid.index(best)]
            y = zs(raw)
            folds = group_folds(groups, args.folds)
            fr = fit(xb, y, folds)
            from sklearn.linear_model import Ridge

            rep = {"block": best, "n": int(len(rows)), "cv_alpha": fr["alpha"], "cv_r2": round(fr["cv_r2"], 4)}
            units = {}
            for mult, kind in ((1, "ridge"), (10, "ridge_l10"), (100, "ridge_l100"), (None, "ridge_cov")):
                if mult is None:
                    xc = (xb - xb.mean(0)).astype(np.float64)
                    u = unit(xc.T @ (y - y.mean()))
                    lam = "inf"
                else:
                    lam = fr["alpha"] * mult
                    u = unit(Ridge(alpha=lam).fit(xb.astype(np.float64), y).coef_)
                units[kind] = u
                r2 = float(cv_ridge(xb, y, folds, alphas=[lam])[0]) if mult is not None else None
                p = emit(f"{kind}_{d}", d, kind, best, u, xb, raw,
                         {"ridge_alpha": lam, "cv_alpha": fr["alpha"], "lambda_mult": mult if mult else "inf",
                          "cv_r2_at_lambda": r2}, cap, pop, len(rows))
                rep[kind] = {"lambda": lam, "cv_r2": None if r2 is None else round(r2, 4), "pack": str(p)}
                print(f"[reg {d}/{kind}] b{best} lambda {lam} CV R2 {r2} -> {p}", flush=True)
            rep["cos_vs_cv"] = {k: round(cos(units["ridge"], u), 4) for k, u in units.items()}
            report["reg_path"][d] = rep
    if args.block_sweep:
        cap = Path(args.sweep_cap)
        for d in args.sweep_descriptors:
            means, bid, rows, raw, groups, pop, best = rows_for(cap, d)
            miss = [b for b in args.block_sweep if b not in bid]
            if miss:
                raise SystemExit(f"{cap}: blocks {miss} not captured (has {bid})")
            loc = [bid.index(b) for b in args.block_sweep]
            feats = stepmean_blocks(means, loc, rows)
            y = zs(raw)
            folds = group_folds(groups, args.folds)
            rep = {"capture": str(cap), "n": int(len(rows))}
            for b, lb in zip(args.block_sweep, loc):
                fr = fit(feats[lb], y, folds)
                kind = f"ridge_sb{b}"
                p = emit(f"{kind}_{d}", d, kind, b, unit(fr["coef"]), feats[lb], raw,
                         {"ridge_alpha": fr["alpha"], "cv_r2": round(fr["cv_r2"], 5), "sweep_capture": str(cap)},
                         cap, pop, len(rows))
                rep[str(b)] = {"alpha": fr["alpha"], "cv_r2": round(fr["cv_r2"], 4), "pack": str(p)}
                print(f"[sweep {d}] b{b} alpha {fr['alpha']:.3g} CV R2 {fr['cv_r2']:.4f} -> {p}", flush=True)
            report["block_sweep"][d] = rep
    (out / "fit_report.json").write_text(json.dumps(report, indent=1))
    md = ["# v3 smoke S1 extras: regularisation path (Q1) and block sweep (Q6)", ""]
    for d, r in report["reg_path"].items():
        md.append(f"- reg {d} b{r['block']} n {r['n']}: " + "; ".join(
            f"{k} lambda {r[k]['lambda']} CV R2 {r[k]['cv_r2']} cos-vs-CV {r['cos_vs_cv'][k]}"
            for k in ("ridge", "ridge_l10", "ridge_l100", "ridge_cov")))
    for d, r in report["block_sweep"].items():
        md.append(f"- sweep {d} ({r['capture']}, n {r['n']}): " + "; ".join(
            f"b{b} CV R2 {v['cv_r2']}" for b, v in r.items() if isinstance(v, dict)))
    (out / "fit_report.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"wrote {out / 'fit_report.md'} in {time.time() - t0:.0f} s", flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cap", required=True, help="v2 corpus root (means.npy, meta.json, prompts.json, dirs/index.csv)")
    ap.add_argument("--labels", help="label table (parquet or csv, column id); default $CAP/labels/merged.parquet")
    ap.add_argument("--controls", required=True, help="shipped pack dir (packs_final/sa3/medium)")
    ap.add_argument("--manifest", help="packs_final manifest.csv (lra proxy check); default <controls>/../../manifest.csv")
    ap.add_argument("--out", required=True)
    ap.add_argument("--descriptors", nargs="*", default=list(DESCRIPTORS))
    ap.add_argument("--variant-b", action="store_true", help="also fit the summed top-3 block variant")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--per-sign", action="store_true",
                    help="S1: symmetric / up (mid+high tercile) / down (low+mid tercile) / low-tail (bottom half) "
                         "ridge packs at the best block only (no resid, no variant b)")
    ap.add_argument("--reg-path", action="store_true", help="S1 extras (Q1): CV, 10x, 100x lambda and X^T y, up side")
    ap.add_argument("--block-sweep", type=int, nargs="*", default=None, help="S1 extras (Q6): blocks to fit at CV lambda")
    ap.add_argument("--sweep-descriptors", nargs="*", default=["centroid", "onset_rate"])
    ap.add_argument("--sweep-cap", default="E:/Projects/DEMON/steering-bench/many_knobs/corpus",
                    help="capture with every swept block (default v1 corpus, all 24 blocks)")
    ap.add_argument("--sweep-labels", help="labels for --sweep-cap (default --labels)")
    args = ap.parse_args()
    if args.per_sign:
        return per_sign(args)
    if args.reg_path or args.block_sweep:
        args.sweep_labels = args.sweep_labels or args.labels
        return extras(args)

    from acestep.steering.packs import SteeringPack, load_pack, save_pack

    t0 = time.time()
    cap, out = Path(args.cap), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    means, meta, ids, cats = load_corpus(cap)
    bid = block_ids(meta, means.shape[2])
    labels = load_labels(Path(args.labels) if args.labels else cap / "labels" / "merged.parquet", ids)
    with open(cap / "dirs" / "index.csv", newline="", encoding="utf-8") as f:
        index = {r["name"]: r for r in csv.DictReader(f)}
    pr = json.loads((cap / "prompts.json").read_text())[: means.shape[0]]
    base = np.array([int(r.get("base_id", r["id"])) for r in pr])
    ctrl_dir = Path(args.controls)

    plan = {}
    for d in args.descriptors:
        col, concept, control = DESCRIPTORS[d]
        row = index[concept]
        if row["label_col"] != col:
            raise SystemExit(f"{concept}: index label_col {row['label_col']} != {col}")
        pop = row.get("population", "music") or "music"
        cols = [col] + [c for c in CONFOUNDS if c != col]
        ok = population_mask(pop, cats) & np.isfinite(labels[cols].to_numpy(dtype=np.float64)).all(1)
        blocks = [int(b) for b in row["blocks"].split()]
        plan[d] = dict(col=col, concept=concept, control=control, pop=pop, cols=cols, rows=np.flatnonzero(ok),
                       best=int(row["best_block"]), top3=blocks[:3], v2_dir=row["path"])
    rows = np.unique(np.concatenate([p["rows"] for p in plan.values()]))
    print(f"[load] means {means.shape}, blocks {bid}, {len(rows)} usable rows", flush=True)
    feats = stepmean_blocks(means, list(range(len(bid))), rows)
    pos_of = {r: i for i, r in enumerate(rows)}
    print(f"[feat] step-mean features for {len(bid)} blocks in {time.time() - t0:.0f} s", flush=True)

    ctrl_packs = {}
    for d, p in plan.items():
        cp = load_pack(ctrl_dir / f"{p['control']}.safetensors")
        ctrl_packs[d] = cp

    today = _dt.date.today().isoformat()
    report = {"date": today, "cap": str(cap), "n_corpus": int(means.shape[0]), "block_ids": bid,
              "feature": "mean over the 8 sampler steps of the post_block_residual audio-token mean at one block",
              "cv": f"{args.folds}-fold GroupKFold on prompts.json base_id, pooled out-of-fold R^2; alpha grid "
                    f"logspace(-1, 6, 15)", "confounds": list(CONFOUNDS), "descriptors": {}, "packs": []}
    for d, p in plan.items():
        idx = np.array([pos_of[r] for r in p["rows"]])
        lab = labels.iloc[p["rows"]]
        raw = lab[p["col"]].to_numpy(dtype=np.float64)
        conf = lab[[c for c in p["cols"] if c != p["col"]]].to_numpy(dtype=np.float64)
        y = zs(raw)
        yres, conf_r2 = residualise(y, conf)
        yres = zs(yres)
        folds = group_folds(base[p["rows"]], args.folds)
        lb = bid.index(p["best"])
        xb = feats[lb][idx]
        rep = {"label_col": p["col"], "v2_concept": p["concept"], "population": p["pop"], "n": int(len(idx)),
               "n_prompts": int(len(np.unique(base[p["rows"]]))), "block": p["best"],
               "block_choice": f"v2 variant a best block of corpus concept {p['concept']} (same label column)",
               "confound_r2": round(conf_r2, 4), "control": p["control"]}
        per_block = {}
        for k, b in enumerate(bid):
            per_block[str(b)] = round(float(cv_ridge(feats[k][idx], y, folds).max()), 4)
        rep["cv_r2_by_block"] = per_block
        fits = {}
        for kind, target in (("ridge", y), ("ridge_resid", yres)):
            fr = fit(xb, target, folds)
            u = unit(fr["coef"])
            gap, pstd = quartile_gap(xb, raw, u)
            fits[kind] = dict(fr, unit=u, gap=gap, pstd=pstd)
            print(f"[{d}/{kind}] b{p['best']} alpha {fr['alpha']:.3g} CV R2 {fr['cv_r2']:.4f} "
                  f"(in-sample {fr['in_sample_r2']:.4f}) gap {gap:.3f}", flush=True)
        cp = ctrl_packs[d]
        cvec = cp.vector.numpy().reshape(-1, cp.vector.shape[-1])[0]
        ctrl = {"block": int(cp.block), "cos_ridge": round(cos(fits["ridge"]["unit"], cvec), 4),
                "cos_ridge_resid": round(cos(fits["ridge_resid"]["unit"], cvec), 4),
                "note": "cross-block when control block != ridge block"}
        if int(cp.block) in bid and int(cp.block) != p["best"]:
            xc = feats[bid.index(int(cp.block))][idx]
            for kind, target in (("ridge", y), ("ridge_resid", yres)):
                fc = fit(xc, target, folds)
                ctrl[f"cos_{kind}_at_control_block"] = round(cos(unit(fc["coef"]), cvec), 4)
                ctrl[f"cv_r2_{kind}_at_control_block"] = round(fc["cv_r2"], 4)
        elif int(cp.block) not in bid:
            ctrl["at_control_block"] = f"control block {cp.block} not captured in the v2 corpus"
        rep["control_cos"] = ctrl
        z = np.load(p["v2_dir"]) if Path(p["v2_dir"]).exists() else None
        if z is None:
            alt = cap / "dirs" / Path(p["v2_dir"]).name
            z = np.load(alt) if alt.exists() else None
        if z is not None:
            v2 = z["vec"][0] * int(z["sign"])          # sign-adjusted so +1 = raises the label
            rep["cos_v2_classmean_same_label"] = {k: round(cos(fits[k]["unit"], v2), 4) for k in fits}
        rep["cos_ridge_vs_resid"] = round(cos(fits["ridge"]["unit"], fits["ridge_resid"]["unit"]), 4)
        for kind, f in fits.items():
            name = f"{kind}_{d}"
            prov = {"tool": "scripts/steering_bench/ridge_directions.py", "method": "ridge", "date": today,
                    "label_col": p["col"], "target": "z-scored descriptor" if kind == "ridge" else
                    "z-scored OLS residual of the z-scored descriptor on the confounds (z-scored)",
                    "confounds": [] if kind == "ridge" else [c for c in p["cols"] if c != p["col"]],
                    "confound_r2": None if kind == "ridge" else round(conf_r2, 4),
                    "sign": 1, "sign_rule": "positive knob raises the descriptor",
                    "population": p["pop"], "n": int(len(idx)), "blocks": [p["best"]],
                    "block_choice": rep["block_choice"], "ridge_alpha": f["alpha"],
                    "cv_r2": round(f["cv_r2"], 5), "cv": report["cv"], "in_sample_r2": round(f["in_sample_r2"], 5),
                    "capture": str(cap), "feature": report["feature"], "knob_unit": KNOB_UNIT,
                    "norm_rule": "projected gap top-quartile minus bottom-quartile mean of the raw descriptor",
                    "proj_std_at_block": round(f["pstd"], 5), "control": p["control"],
                    "cos_control": ctrl["cos_" + kind], "calibrated_gain": None,
                    "caveat": "estimated at 10 s ARC sam.generate, audio tokens only; unscreened (v3 smoke)"}
            path = write(SteeringPack, save_pack, load_pack, out / f"{name}.safetensors", name=name,
                         block=p["best"], blocks=[], vec=f["unit"], norm=f["gap"],
                         label=f"{d} ({kind})", blurb=f"ridge direction raising {p['col']}", prov=prov)
            rep[kind] = {"alpha": f["alpha"], "cv_r2": round(f["cv_r2"], 4), "in_sample_r2": round(f["in_sample_r2"], 4),
                         "cv_curve": f["cv_curve"], "norm": round(f["gap"], 4), "proj_std": round(f["pstd"], 4),
                         "pack": str(path)}
            report["packs"].append(str(path))
            print(f"  wrote {path} (re-read OK)", flush=True)
        if args.variant_b:
            l3 = [bid.index(b) for b in p["top3"]]
            xj = np.concatenate([feats[k][idx] for k in l3], 1)
            rep["variant_b"] = {"blocks": p["top3"]}
            for kind, target in (("ridge", y), ("ridge_resid", yres)):
                fb = fit(xj, target, folds)
                h = xb.shape[1]
                rowsv = fb["coef"].reshape(len(l3), h)
                n0 = max(float(np.linalg.norm(rowsv[0])), 1e-12)
                vec = (rowsv / n0)[:, None, :]                   # [3 blocks, 1 step, H], row 0 unit
                gap, pstd = quartile_gap(xj, raw, fb["coef"] / n0)   # all rows applied together
                name = f"{kind}_{d}"
                prov = {"tool": "scripts/steering_bench/ridge_directions.py", "method": "ridge", "variant": "b",
                        "date": today, "label_col": p["col"], "sign": 1, "population": p["pop"], "n": int(len(idx)),
                        "blocks": p["top3"], "block_choice": "joint ridge over the v2 top-3 blocks of "
                        f"{p['concept']}; rows scaled relative to the best block's row", "ridge_alpha": fb["alpha"],
                        "cv_r2": round(fb["cv_r2"], 5), "cv": report["cv"], "capture": str(cap),
                        "confounds": [] if kind == "ridge" else [c for c in p["cols"] if c != p["col"]],
                        "knob_unit": KNOB_UNIT, "calibrated_gain": None,
                        "norm_rule": "projected quartile gap of the joint (all-block) projection, rows scaled so row 0 is unit"}
                pth = write(SteeringPack, save_pack, load_pack, out / "variant_b" / f"{name}.safetensors", name=name,
                            block=p["top3"][0], blocks=p["top3"], vec=vec, norm=gap, label=f"{d} ({kind}, top-3)",
                            blurb=f"ridge direction raising {p['col']} (3 blocks)", prov=prov)
                rep["variant_b"][kind] = {"alpha": fb["alpha"], "cv_r2": round(fb["cv_r2"], 4),
                                          "row_norms": [round(float(np.linalg.norm(r)), 4) for r in vec[:, 0]],
                                          "pack": str(pth)}
                print(f"[{d}/{kind}/b] blocks {p['top3']} CV R2 {fb['cv_r2']:.4f} -> {pth}", flush=True)
        report["descriptors"][d] = rep

    # lra proxy check: shipped packs whose label column correlates best with dyn.lra (music population)
    if "lra" in plan:
        man = Path(args.manifest) if args.manifest else ctrl_dir.parents[1] / "manifest.csv"
        if man.exists():
            with open(man, newline="", encoding="utf-8") as f:
                shipped = [r["name"] for r in csv.DictReader(f) if r["status"] == "shipped"]
            p = plan["lra"]
            lab = labels.iloc[p["rows"]]
            lra = lab["dyn.lra"].to_numpy(dtype=np.float64)
            cands = []
            for nm in shipped:
                r = index.get(nm)
                if not r or r["label_col"] not in labels.columns:
                    continue
                v = lab[r["label_col"]].to_numpy(dtype=np.float64) * int(float(r["sign"]))
                m = np.isfinite(v)
                if m.sum() > 100 and np.std(v[m]) > 0:
                    cands.append((nm, r["label_col"], round(float(np.corrcoef(v[m], lra[m])[0, 1]), 4)))
            cands.sort(key=lambda t: -abs(t[2]))
            ctl = index.get(p["control"])
            report["lra_proxy_check"] = {
                "control": p["control"], "control_label_corr": next((c[2] for c in cands if c[0] == p["control"]), None),
                "top_shipped_by_abs_label_corr": cands[:5],
                "note": "corr of the shipped pack's sign-adjusted label column with dyn.lra over the lra fit rows; "
                        "legacy packs carry no label column and are skipped" + (
                            "" if ctl else f" ({p['control']} not in v2 index)")}

    (out / "fit_report.json").write_text(json.dumps(report, indent=1))
    md = ["# v3 smoke: ridge directions fit report", "",
          f"{report['date']}; corpus {cap} ({report['n_corpus']} rows); {report['cv']}.", "",
          "| descriptor | block | n | CV R^2 ridge | CV R^2 resid | confound R^2 | cos(ridge, resid) | control | "
          "cos(ridge, control) | cos(resid, control) | same-block cos ridge/resid | cos v2 class-mean ridge/resid |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for d, r in report["descriptors"].items():
        c = r["control_cos"]
        sb = (f"{c.get('cos_ridge_at_control_block')} / {c.get('cos_ridge_resid_at_control_block')} (b{c['block']})"
              if "cos_ridge_at_control_block" in c else c.get("at_control_block", "same block"))
        v2 = r.get("cos_v2_classmean_same_label", {})
        md.append(f"| {d} | {r['block']} | {r['n']} | {r['ridge']['cv_r2']} | {r['ridge_resid']['cv_r2']} | "
                  f"{r['confound_r2']} | {r['cos_ridge_vs_resid']} | {r['control']} (b{c['block']}) | {c['cos_ridge']} | "
                  f"{c['cos_ridge_resid']} | {sb} | {v2.get('ridge')} / {v2.get('ridge_resid')} |")
    if any("variant_b" in r for r in report["descriptors"].values()):
        md += ["", "Variant b (joint ridge over the v2 top-3 blocks, packs in variant_b/):", ""]
        for d, r in report["descriptors"].items():
            vb = r.get("variant_b")
            if vb:
                md.append(f"- {d}: blocks {vb['blocks']}, CV R^2 ridge {vb['ridge']['cv_r2']} (rows "
                          f"{vb['ridge']['row_norms']}), resid {vb['ridge_resid']['cv_r2']}")
    md += ["", "CV R^2 by block (ridge, raw descriptor):", ""]
    for d, r in report["descriptors"].items():
        md.append(f"- {d}: " + ", ".join(f"b{b} {v}" for b, v in r["cv_r2_by_block"].items()))
    if "lra_proxy_check" in report:
        lp = report["lra_proxy_check"]
        md += ["", f"lra proxy check: {lp['control']} label corr with dyn.lra = {lp['control_label_corr']}; "
                   "top shipped by |corr|: " + "; ".join(f"{a} ({b}) {c}" for a, b, c in lp["top_shipped_by_abs_label_corr"])]
    md += ["", "Packs: " + ", ".join(Path(x).name for x in report["packs"]), ""]
    (out / "fit_report.md").write_text("\n".join(md), encoding="utf-8")
    print(f"wrote {out / 'fit_report.json'} and fit_report.md in {time.time() - t0:.0f} s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
