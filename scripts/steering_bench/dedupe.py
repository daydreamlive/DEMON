"""Dedupe the many-knobs directions (phase 4, CPU).

Reads ``dirs/index.csv`` + ``dirs/<name>.npz`` (build_directions.py --cap)
and the capture, and writes under ``dirs/``:

* ``cosine_matrix.npy`` [C, C] float32 + ``cosine_names.json``: cosine of the
  step-mean unit directions. Each concept's direction is recomputed from the
  capture at any block needed (same classes as build_directions: the npz
  ``pos_idx`` / ``neg_idx``);
  pair (i, j) is compared at block(i) and at block(j) and the value with the
  larger magnitude is kept (signed). Clustering uses ``|cos|``: an antipodal
  pair (bright vs dark) is one knob.
* ``clusters.csv``: greedy keeper clustering. Concepts sorted by effect at
  their best block (descending); each not-yet-assigned concept becomes a
  keeper and takes every unassigned concept with ``|cos| > --thresh`` to it.
  Columns: cluster, name, keeper, is_keeper, cos_to_keeper, effect,
  best_block, label_col.
* ``cross_projection.npy`` [C, C] + ``.csv``: row i = direction of concept i
  (step-mean unit at its best block), column j = concept j's label: the
  correlation over the corpus between the clips' projection on direction i
  and ``sign_j * label_j``. That is the label-j slope per unit of projection
  std in label-j stds, i.e. which labels a knob would drag along. Diagonal
  = own label.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_directions import _class_diff, block_ids, load_corpus, load_labels, parse_sign  # noqa: E402


def read_index(path: Path) -> list:
    with open(path, newline="", encoding="utf-8") as f:
        return [r for r in csv.DictReader(f) if r.get("status", "ok") == "ok"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cap", required=True, help="corpus root ($CAP)")
    ap.add_argument("--labels", help="default $CAP/labels/merged.parquet")
    ap.add_argument("--dirs", help="default $CAP/dirs")
    ap.add_argument("--thresh", type=float, default=0.8, help="|cos| above which two concepts merge")
    args = ap.parse_args()

    cap = Path(args.cap)
    dirs = Path(args.dirs) if args.dirs else cap / "dirs"
    rows = read_index(dirs / "index.csv")
    if not rows:
        raise SystemExit("index.csv has no usable concepts")
    means, _meta, ids, _cats = load_corpus(cap)
    n, steps, nb, h = means.shape
    loc_of = {b: j for j, b in enumerate(block_ids(_meta, nb))}
    labels = load_labels(Path(args.labels) if args.labels else cap / "labels" / "merged.parquet", ids)
    names = [r["name"] for r in rows]
    c = len(rows)
    best = np.array([int(r["best_block"]) for r in rows])
    effect = np.array([float(r["effect_best"]) for r in rows])
    signs = np.array([parse_sign(r["sign"]) for r in rows])
    Y = np.stack([signs[i] * labels[r["label_col"]].to_numpy(dtype=np.float64) for i, r in enumerate(rows)], 1)
    # the classes build_directions used (population + class rule), stored in each npz
    P, Q = np.zeros((c, n), np.float32), np.zeros((c, n), np.float32)
    for i, r in enumerate(rows):
        z = np.load(r["path"])
        P[i, z["pos_idx"]] = 1.0
        Q[i, z["neg_idx"]] = 1.0

    cos_at = {}                      # block -> [C, C] cosine of step-mean units at that block
    proj = np.zeros((n, c))          # clip projections on each concept's own best-block direction
    for b in sorted(set(best.tolist())):
        xbm = np.asarray(means[:, :, loc_of[b], :], dtype=np.float32).mean(1)  # [N, H] step mean
        d = _class_diff(xbm, P, Q)
        u = d / np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-12)
        cos_at[b] = u @ u.T
        own = np.flatnonzero(best == b)
        proj[:, own] = xbm @ u[own].T
        print(f"[block {b:02d}] {len(own)} concepts have it as best block", flush=True)
        # consistency with the stored npz (step-mean of the stored per-step units)
        for i in own[:3]:
            z = np.load(rows[i]["path"])
            k = list(z["blocks"]).index(b)
            sm = (z["unit"][:, k].astype(np.float32) * z["norm"][:, k][:, None]).mean(0)
            cs = float(sm @ u[i] / max(np.linalg.norm(sm), 1e-12))
            if cs < 0.99:
                print(f"  WARN {names[i]}: recomputed vs stored cos {cs:.4f}", flush=True)

    cos = np.zeros((c, c), dtype=np.float32)
    for i in range(c):
        a = cos_at[best[i]][i]
        for j in range(c):
            x, y = a[j], cos_at[best[j]][i, j]
            cos[i, j] = x if abs(x) >= abs(y) else y
    np.save(dirs / "cosine_matrix.npy", cos)
    (dirs / "cosine_names.json").write_text(json.dumps(names))

    order = np.argsort(-effect, kind="stable")
    cluster = -np.ones(c, dtype=int)
    keeper_of = np.zeros(c, dtype=int)
    k = 0
    for i in order:
        if cluster[i] >= 0:
            continue
        members = [j for j in order if cluster[j] < 0 and (j == i or abs(cos[i, j]) > args.thresh)]
        for j in members:
            cluster[j], keeper_of[j] = k, i
        k += 1
    with open(dirs / "clusters.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["cluster", "name", "keeper", "is_keeper", "cos_to_keeper", "effect", "best_block", "label_col"])
        for i in sorted(range(c), key=lambda j: (cluster[j], -effect[j])):
            kp = keeper_of[i]
            w.writerow([cluster[i], names[i], names[kp], int(kp == i), f"{cos[i, kp]:+.4f}",
                        f"{effect[i]:.4f}", best[i], rows[i]["label_col"]])

    # cross-projection: corr(projection on direction i, sign_j * label_j), NaN labels skipped
    pz = (proj - proj.mean(0)) / np.maximum(proj.std(0), 1e-12)
    ok = np.isfinite(Y)
    mu = np.nanmean(Y, 0)
    sd = np.nanstd(Y, 0)
    yz = np.where(ok, (Y - mu) / np.maximum(sd, 1e-12), 0.0)
    cnt = ok.sum(0).astype(np.float64)
    cross = (pz.T @ yz) / np.maximum(cnt[None, :], 1)
    np.save(dirs / "cross_projection.npy", cross.astype(np.float32))
    with open(dirs / "cross_projection.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["direction \\ label"] + names)
        for i in range(c):
            w.writerow([names[i]] + [f"{x:+.3f}" for x in cross[i]])

    n_clusters = int(cluster.max()) + 1
    merged = [(names[i], names[keeper_of[i]], float(cos[i, keeper_of[i]])) for i in range(c) if keeper_of[i] != i]
    print(f"{c} concepts -> {n_clusters} clusters (|cos| > {args.thresh}); merged: "
          + ", ".join(f"{a}->{b} ({x:+.2f})" for a, b, x in merged), flush=True)
    print(f"wrote {dirs / 'cosine_matrix.npy'}, clusters.csv, cross_projection.csv", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
