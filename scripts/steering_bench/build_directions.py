"""Per-block, per-step steering directions for the five production SA3 knobs
from a ``capture_resid.py`` capture (post_block_residual, audio-token means).

Labels (one of):

* MusicCaps (default): keyword classes over ``caption`` + ``aspect_list``
  (regexes in :mod:`concepts`), rows aligned through ``meta.json
  caption_index``; two-sided concepts keep captions matching exactly one
  pole, percussive is presence vs none; classes balanced to
  ``min(pos, neg, --cap)`` with rng 0 (as E5). method ``class_mean_musiccaps``.
* ``--self-label``: the five descriptors (:mod:`proxies`) measured on the
  capture's saved audio; positive class = top quartile of ``sign *
  descriptor``, negative = bottom quartile (density: positive = sparse =
  LOWER onset rate). method ``mass_mean_quartile``.

Per concept, step s and block b: ``diff = mean(pos) - mean(neg)``,
``unit = diff / |diff|``, ``raw_norm = |diff|``, ``proj_std`` = pooled
within-class std of the projection on ``unit`` (for std-normalised alpha),
``proj_std_all`` = std over every captured row.

Writes under ``--out`` (use the bench ``--out`` so ``sa3_tada_run.py
--vec-dir`` finds them):

* ``vec_<tag>/<concept>.pt``          all 24 blocks (``--sites all``, or ``loc --loc b``)
* ``vec_<tag>_b<bb>/<concept>.pt``    block bb only (24 dirs)
* ``vec_<tag>_prod/<concept>.pt``     the production pack's block (or ``--block``)
* ``packs_<tag>/sa3/medium/<concept>.safetensors``  production pack format 1
* ``packs_<tag>_b<bb>/<concept>.safetensors``  format 1 at block bb (24 dirs; the per-block
  sweep uses these with ``--method pack``, the only bench path that steers post_block_residual)
* ``packs_<tag>_perstep/...``          format 2, ``vector [steps, 1536]`` (``--per-step-pack``;
  production does NOT read format 2 yet)
* ``<tag>_summary.json``              classes, norms, effect sizes, alpha scales, cos to production

Vector files are ``{"vectors": {step: {block: unit fp32 [H]}}, ...}`` as
``sa3_tada_run.py`` ``_load_vectors`` reads them (torch.save).
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from concepts import CONCEPTS, DESCRIPTORS, SPEC  # noqa: E402

DEFAULT_CAPTIONS = "E:/Projects/tada-replication/data/musiccaps-public.csv"
DEFAULT_PROD_PACKS = Path.home() / ".daydream-scope/models/demon/steering_packs/sa3/medium"
PACK_METADATA_KEY = "steering_pack"
KNOB_UNIT = 0.1
CHUNK = 128


# ---------------------------------------------------------------------------
# labels
# ---------------------------------------------------------------------------

def musiccaps_classes(csv_path, caption_index, n, cap, rng):
    import pandas as pd

    df = pd.read_csv(csv_path)
    asp = df["aspect_list"].astype(str) if "aspect_list" in df.columns else ""
    text = (df["caption"].astype(str) + " " + asp).str.lower().tolist()
    low = [text[i] for i in caption_index[:n]]
    out = {}
    for c in CONCEPTS:
        sp = SPEC[c]
        p = np.array([bool(re.search(sp["pos"], t)) for t in low])
        if sp["kind"] == "presence":
            neg_mask = np.array([not re.search(sp["exclude"], t) for t in low]) & ~p
            pos_mask = p
            rule = {"pos": sp["pos"], "neg": f"none of {sp['exclude']}"}
        else:
            q = np.array([bool(re.search(sp["neg"], t)) for t in low])
            pos_mask, neg_mask = p & ~q, q & ~p
            rule = {"pos": sp["pos"], "neg": sp["neg"], "exclusive": True}
        pos, neg = np.flatnonzero(pos_mask), np.flatnonzero(neg_mask)
        m = min(len(pos), len(neg), cap)
        if m < 2:
            raise SystemExit(f"{c}: classes too small ({len(pos)} / {len(neg)})")
        out[c] = {"pos": np.sort(rng.choice(pos, m, replace=False)),
                  "neg": np.sort(rng.choice(neg, m, replace=False)),
                  "n_pos_all": int(len(pos)), "n_neg_all": int(len(neg)), "rule": rule}
    return out


def measure_descriptors(cap_dir: Path, n: int, cache: Path) -> np.ndarray:
    """``[n, 5]`` descriptor values (order DESCRIPTORS) for the capture's audio."""
    if cache.exists():
        d = np.load(cache)
        if d.shape == (n, len(DESCRIPTORS)):
            print(f"[desc] reuse {cache}", flush=True)
            return d
    from desc_pool import measure_clips

    vals = np.full((n, len(DESCRIPTORS)), np.nan)
    files = sorted((cap_dir / "audio").glob("batch_*.npz"))
    if not files:
        raise SystemExit(f"no audio under {cap_dir / 'audio'} (capture with --self-label)")
    rows, clips, sr = [], [], None
    for f in files:
        z = np.load(f)
        sr = int(z["sr"])
        for row, a in zip(z["rows"], z["audio"]):
            if row < n:
                rows.append(int(row))
                clips.append(a)
    print(f"[desc] {len(clips)} clips, five descriptors (parallel)", flush=True)
    vals[rows] = measure_clips(clips, sr)
    if np.isnan(vals).any():
        raise SystemExit(f"{int(np.isnan(vals).any(1).sum())} rows have no audio")
    np.save(cache, vals)
    return vals


def quartile_classes(desc: np.ndarray, q: float):
    out = {}
    for c in CONCEPTS:
        sp = SPEC[c]
        d = sp["sign"] * desc[:, DESCRIPTORS.index(sp["descriptor"])]
        lo, hi = np.quantile(d, q), np.quantile(d, 1.0 - q)
        pos, neg = np.flatnonzero(d >= hi), np.flatnonzero(d <= lo)
        raw = desc[:, DESCRIPTORS.index(sp["descriptor"])]
        out[c] = {"pos": pos, "neg": neg, "n_pos_all": int(len(pos)), "n_neg_all": int(len(neg)),
                  "rule": {"descriptor": sp["descriptor"], "sign": sp["sign"], "quantile": q,
                           "pos_mean": float(raw[pos].mean()), "neg_mean": float(raw[neg].mean()),
                           "all_mean": float(raw.mean())}}
    return out


# ---------------------------------------------------------------------------
# estimation
# ---------------------------------------------------------------------------

def estimate(means, classes):
    """Class means (pass 1) then projections (pass 2), chunked over rows."""
    n, steps, nb, h = means.shape
    sums = {c: {k: np.zeros((steps, nb, h)) for k in ("pos", "neg")} for c in classes}
    masks = {c: {k: np.isin(np.arange(n), v[k]) for k in ("pos", "neg")} for c, v in classes.items()}
    for i in range(0, n, CHUNK):
        x = np.asarray(means[i:i + CHUNK], dtype=np.float64)
        for c in classes:
            for k in ("pos", "neg"):
                m = masks[c][k][i:i + CHUNK]
                if m.any():
                    sums[c][k] += x[m].sum(0)
    res = {}
    for c, v in classes.items():
        diff = sums[c]["pos"] / len(v["pos"]) - sums[c]["neg"] / len(v["neg"])
        raw = np.linalg.norm(diff, axis=-1)                       # [steps, nb]
        res[c] = {"diff": diff, "raw": raw, "unit": diff / np.maximum(raw[..., None], 1e-12)}
    proj = {c: np.zeros((n, steps, nb), dtype=np.float32) for c in classes}
    units = {c: torch.from_numpy(res[c]["unit"]).float() for c in classes}
    for i in range(0, n, CHUNK):
        x = torch.from_numpy(np.asarray(means[i:i + CHUNK], dtype=np.float32))
        for c in classes:
            proj[c][i:i + CHUNK] = torch.einsum("nsbh,sbh->nsb", x, units[c]).numpy()
    for c, v in classes.items():
        pp, pn = proj[c][v["pos"]], proj[c][v["neg"]]
        dof = max(len(pp) + len(pn) - 2, 1)
        pooled = np.sqrt((pp.var(0, ddof=1) * (len(pp) - 1) + pn.var(0, ddof=1) * (len(pn) - 1)) / dof)
        res[c]["proj_std"] = pooled
        res[c]["proj_std_all"] = proj[c].std(0, ddof=1)
        res[c]["effect"] = res[c]["raw"] / np.maximum(pooled, 1e-12)
    return res


# ---------------------------------------------------------------------------
# outputs
# ---------------------------------------------------------------------------

def nested(arr, blocks):
    """``{step: {block: value}}`` from ``arr[step, block, ...]``."""
    f = (lambda a: torch.from_numpy(np.ascontiguousarray(a)).float()) if arr.ndim == 3 else float
    return {s: {b: f(arr[s, b]) for b in blocks} for s in range(arr.shape[0])}


def write_vec(path: Path, r, blocks, concept, info) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"vectors": nested(r["unit"], blocks), "raw_norm": nested(r["raw"], blocks),
                "proj_std": nested(r["proj_std"], blocks), "proj_std_all": nested(r["proj_std_all"], blocks),
                "concept": concept, "pairs": None, "blocks": list(blocks),
                "hook": "post_block_residual", **info}, path)


def pack_header(concept, block, norm, provenance, *, fmt=1, extra=None) -> dict:
    sp = SPEC[concept]
    meta = {"family": "sa3", "checkpoint": "medium", "block": int(block), "hidden_size": 1536,
            "name": concept, "label": sp["label"], "blurb": sp["blurb"], "hook": "post_block_residual",
            "method": "caa_diff_means", "norm": float(norm), "magnitude": float(norm) * KNOB_UNIT,
            "policy": {"kind": "range", "start": 0.0, "end": 1.0}, "provenance": provenance,
            "format": fmt}
    meta.update(extra or {})
    return meta


def write_pack(path: Path, vector: torch.Tensor, meta: dict) -> Path:
    """Production ``save_pack`` layout: one F32 tensor ``vector``, metadata
    ``steering_pack`` = JSON (sort_keys) of the SteeringPack fields + format."""
    from safetensors.torch import save_file

    path.parent.mkdir(parents=True, exist_ok=True)
    save_file({"vector": vector.detach().float().contiguous()}, str(path),
              metadata={PACK_METADATA_KEY: json.dumps(meta, sort_keys=True)})
    return path


def prod_vector(prod_dir: Path, concept: str):
    p = prod_dir / f"{concept}.safetensors"
    if not p.exists():
        return None, None
    from safetensors import safe_open

    with safe_open(str(p), framework="pt") as f:
        meta = json.loads(f.metadata()[PACK_METADATA_KEY])
        return f.get_tensor("vector").float(), int(meta["block"])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--capture", required=True, help="capture_resid.py --out dir")
    ap.add_argument("--out", required=True, help="bench root (sa3_tada_run.py --out)")
    ap.add_argument("--self-label", action="store_true")
    ap.add_argument("--captions", default=DEFAULT_CAPTIONS)
    ap.add_argument("--tag", default=None, help="default resid_mc / resid_self")
    ap.add_argument("--concepts", nargs="+", default=list(CONCEPTS), choices=CONCEPTS)
    ap.add_argument("--cap", type=int, default=1000, help="MusicCaps: max rows per class")
    ap.add_argument("--quantile", type=float, default=0.25, help="self-label: top/bottom fraction")
    ap.add_argument("--block", nargs="*", default=[], metavar="CONCEPT=B",
                    help="pack/prod block override per concept (default: production pack block)")
    ap.add_argument("--per-step-pack", action="store_true")
    ap.add_argument("--prod-packs", type=Path, default=DEFAULT_PROD_PACKS,
                    help="production packs (read-only) for the cosine report")
    args = ap.parse_args()

    cap_dir, out = Path(args.capture), Path(args.out)
    tag = args.tag or ("resid_self" if args.self_label else "resid_mc")
    info_cap = json.loads((cap_dir / "meta.json").read_text())
    means = np.load(cap_dir / "means.npy", mmap_mode="r")
    n, steps, nb, h = means.shape
    print(f"[load] {cap_dir} means {means.shape}", flush=True)
    if args.self_label:
        desc = measure_descriptors(cap_dir, n, out / f"descriptors_{tag}.npy")
        classes = quartile_classes(desc, args.quantile)
        method = "mass_mean_quartile"
    else:
        classes = musiccaps_classes(args.captions, info_cap["caption_index"], n, args.cap,
                                    np.random.default_rng(0))
        method = "class_mean_musiccaps"
    classes = {c: classes[c] for c in args.concepts}
    over = {k: int(v) for k, v in (s.split("=") for s in args.block)}
    res = estimate(means, classes)

    today = _dt.date.today().isoformat()
    cap_summary = {k: info_cap.get(k) for k in ("site", "mode", "n", "steps", "tokens", "duration_s",
                                                "sampler", "checkpoint", "seed", "batch", "date")}
    blocks = list(range(nb))
    summary = {"tag": tag, "method": method, "capture": str(cap_dir), "date": today, "concepts": {}}
    for c in args.concepts:
        r, v = res[c], classes[c]
        pb = over.get(c, SPEC[c]["prod_block"])
        info = {"method": method, "site": "post_block_residual", "steps": steps, "duration": 10.0,
                "classes": [len(v["pos"]), len(v["neg"])], "rule": v["rule"], "capture": str(cap_dir),
                "tokens": "audio only", "date": today}
        write_vec(out / f"vec_{tag}" / f"{c}.pt", r, blocks, c, info)
        for b in blocks:
            write_vec(out / f"vec_{tag}_b{b:02d}" / f"{c}.pt", r, [b], c, info)
        write_vec(out / f"vec_{tag}_prod" / f"{c}.pt", r, [pb], c, info)

        sm = r["diff"].mean(0)                                    # step-mean raw diff [nb, h]
        sm_norm = np.linalg.norm(sm, axis=-1)
        unit = torch.from_numpy(sm[pb] / sm_norm[pb]).float()
        prov = {
            "tool": "scripts/steering_bench/build_directions.py", "method": method,
            "method_source": ("MusicCaps keyword-class difference of means over captured activations"
                              if method == "class_mean_musiccaps" else
                              "descriptor top-vs-bottom quartile difference of means over self-rendered audio"),
            "date": today, "capture": str(cap_dir), "capture_meta": cap_summary,
            "n_pos": int(len(v["pos"])), "n_neg": int(len(v["neg"])),
            "n_pos_all": v["n_pos_all"], "n_neg_all": v["n_neg_all"], "label_rule": v["rule"],
            "block_choice": "override" if c in over else "production pack block (unchanged)",
            "steps": steps, "step_reduction": "mean of per-step raw differences over all steps",
            "knob_unit": KNOB_UNIT,
            "raw_norm_per_block": [round(float(x), 4) for x in sm_norm],
            "proj_std_at_block": float(r["proj_std"][:, pb].mean()),
            "caveat": ("estimated at 10 s ARC sam.generate, audio tokens only; production applies at "
                       "54 s pingpong StreamPipeline to every token incl. 64 memory tokens"),
        }
        pdir = out / f"packs_{tag}" / "sa3" / "medium"
        write_pack(pdir / f"{c}.safetensors", unit, pack_header(c, pb, sm_norm[pb], prov))
        # one format-1 pack per block (flat dirs, for sa3_tada_run.py --method pack --pack <dir>,
        # which steers at the pack's hook = post_block_residual)
        for b in blocks:
            ub = torch.from_numpy(sm[b] / sm_norm[b]).float()
            write_pack(out / f"packs_{tag}_b{b:02d}" / f"{c}.safetensors", ub,
                       pack_header(c, b, sm_norm[b], {**prov, "block_choice": f"per-block sweep candidate b{b}",
                                                      "proj_std_at_block": float(r["proj_std"][:, b].mean())}))
        if args.per_step_pack:
            rows = torch.from_numpy(r["unit"][:, pb]).float()      # [steps, h]
            write_pack(out / f"packs_{tag}_perstep" / "sa3" / "medium" / f"{c}.safetensors", rows,
                       pack_header(c, pb, sm_norm[pb], prov, fmt=2, extra={
                           "norm_per_step": [float(x) for x in r["raw"][:, pb]],
                           "magnitude_per_step": [float(x) * KNOB_UNIT for x in r["raw"][:, pb]],
                           "note": "format 2: vector [steps, hidden], one unit row per sampler step; "
                                   "the production loader reads format 1 only (not wired yet)"}))
        pv, pblk = prod_vector(args.prod_packs, c)
        cos_prod = None
        if pv is not None:
            cos_prod = {"prod_block": pblk,
                        "cos_stepmean_at_prod_block": float(torch.dot(
                            torch.from_numpy(sm[pblk] / sm_norm[pblk]).float(), pv / pv.norm()))}
        summary["concepts"][c] = {
            "classes": [len(v["pos"]), len(v["neg"])], "n_pos_all": v["n_pos_all"],
            "n_neg_all": v["n_neg_all"], "rule": v["rule"], "pack_block": pb,
            "pack_norm": float(sm_norm[pb]), "stepmean_norm_per_block": [round(float(x), 4) for x in sm_norm],
            "effect_size_per_block": [round(float(x), 4) for x in r["effect"].mean(0)],
            "alpha_scale_per_block": [round(float(x), 4) for x in r["proj_std"].mean(0)],
            "alpha_scale_all_per_block": [round(float(x), 4) for x in r["proj_std_all"].mean(0)],
            "step_cos_to_stepmean_at_pack_block": [
                round(float(np.dot(r["unit"][s, pb], sm[pb] / sm_norm[pb])), 4) for s in range(steps)],
            "vs_production": cos_prod,
        }
        best = int(np.argmax(r["effect"].mean(0)))
        print(f"[{c}] classes {len(v['pos'])}/{len(v['neg'])} pack b{pb} norm {sm_norm[pb]:.3f} "
              f"effect b{pb} {r['effect'][:, pb].mean():.2f} (max b{best} {r['effect'][:, best].mean():.2f})"
              + (f" cos prod {cos_prod['cos_stepmean_at_prod_block']:+.3f}" if cos_prod else ""), flush=True)
    (out / f"{tag}_summary.json").write_text(json.dumps(summary, indent=1))
    print(f"wrote {out / f'{tag}_summary.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
