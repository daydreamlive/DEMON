"""Per-block, per-step steering directions for the five production SA3 knobs
from a ``capture_resid.py`` capture (post_block_residual, audio-token means).

Labels (one of):

* MusicCaps (default): keyword classes over ``caption`` + ``aspect_list``
  (regexes in :mod:`concepts`), rows aligned through ``meta.json
  caption_index``; two-sided concepts keep captions matching exactly one
  pole, percussive is presence vs none; classes balanced to
  ``min(pos, neg, --class-cap)`` with rng 0 (as E5). method ``class_mean_musiccaps``.
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

Generic mode (many knobs, ``--cap``; notes/steering_pr/many_knobs_master_plan.md):
any label column of ``labels/merged.parquet`` (one row per clip id), from a
``--catalogue`` csv (columns ``name, label_col`` required; ``sign`` default +1;
``category, second_scorer_col`` and any extra columns are passed through to
``dirs/index.csv``) or one ``--label-col C --sign +1|-1``. Per concept, step and
block: ``diff = mean(top quartile of sign*label) - mean(bottom quartile)``,
``unit = diff/|diff|``; ``std`` = projection std of every clip along ``unit``;
``effect`` = class gap of the projection / pooled within-class std (2-fold
cross-fitted, see block_directions). Blocks are
processed one at a time off the fp16 memmap. Writes ``dirs/<name>.npz``
(``unit`` [steps, 4, H] fp16 at the 4 best blocks by step-mean effect,
``blocks`` [4], ``best_block``, ``norm`` [steps, 4], ``std`` / ``effect``
[steps, 24], ``n_pos``, ``n_neg``, ``label_col``, ``sign``, ``quantile``) and
``dirs/index.csv``.
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


# ---------------------------------------------------------------------------
# generic mode: any label column, many concepts (--cap)
# ---------------------------------------------------------------------------

KEEP_BLOCKS = 4
INDEX_COLS = ("name", "category", "label_col", "sign", "second_scorer_col", "status", "n_pos", "n_neg",
              "best_block", "blocks", "effect_best", "std_best", "norm_best", "label_std", "second_cols",
              "second_corrs", "second_std", "quantile", "path")


def load_corpus(cap: Path):
    """``(means memmap [N, steps, blocks, H], meta, ids, categories)``; ids
    and categories in row order (prompts.json), else ``arange(N)`` / ""."""
    means = np.load(cap / "means.npy", mmap_mode="r")
    meta = json.loads((cap / "meta.json").read_text()) if (cap / "meta.json").exists() else {}
    pj = cap / "prompts.json"
    if pj.exists():
        pr = json.loads(pj.read_text())[: means.shape[0]]
        ids = np.array([int(r["id"]) for r in pr])
        cats = np.array([str(r.get("category", "")) for r in pr])
    else:
        ids, cats = np.arange(means.shape[0]), np.array([""] * means.shape[0])
    if len(ids) != means.shape[0]:
        raise SystemExit(f"prompts.json has {len(ids)} rows, means {means.shape[0]}")
    return means, meta, ids, cats


def block_ids(meta: dict, nb: int) -> list:
    """Model block index of each stored block (capture_resid.py --blocks
    writes ``block_ids``; older captures stored all blocks in order)."""
    b = meta.get("block_ids") if meta else None
    return [int(x) for x in b] if b else list(range(nb))


def load_labels(path: Path, ids: np.ndarray):
    """Label table (parquet or csv, column ``id``) reindexed to the capture's
    row order; clips without labels are NaN."""
    import pandas as pd

    df = pd.read_parquet(path) if str(path).endswith(".parquet") else pd.read_csv(path)
    if "id" in df.columns:
        df = df.set_index("id")
    df.index = df.index.astype(int)
    return df.reindex(ids)


def slug(s: str) -> str:
    """Knob-safe name (``^[a-z][a-z0-9_]{0,31}$``, the pack name rule)."""
    out = re.sub(r"[^a-z0-9_]+", "_", str(s).lower()).strip("_") or "concept"
    if not out[0].isalpha():
        out = "k_" + out
    return out[:32]


#: planner catalogue header (notes/steering_pr/concept_catalogue.csv) -> generic names
CATALOGUE_ALIASES = {"primary_label": "label_col", "second_scorer": "second_scorer_col"}
SECOND_COLS = ("second_scorer_col", "third_scorer")


def read_catalogue(path: Path) -> list:
    """Catalogue rows as dicts, header-driven: ``name`` and ``label_col``
    (or the planner's ``primary_label``) required; ``second_scorer`` is
    read as ``second_scorer_col``; every other column is kept and passed
    through to index.csv (population, class_rule, screen_set, anchors, ...)."""
    import csv

    with open(path, newline="", encoding="utf-8") as f:
        rows = [{k.strip(): (v or "").strip() for k, v in r.items() if k} for r in csv.DictReader(f)]
    for r in rows:
        for a, b in CATALOGUE_ALIASES.items():
            if a in r and not r.get(b):
                r[b] = r.pop(a)
    for need in ("name", "label_col"):
        if rows and need not in rows[0]:
            raise SystemExit(f"{path}: catalogue lacks column {need!r} (has {list(rows[0])})")
    return [r for r in rows if r.get("name") and r.get("label_col") and not r["name"].startswith("#")]


def population_mask(pop: str, cats: np.ndarray) -> np.ndarray:
    """``all`` (or empty) = every clip; ``sfx`` = prompt category sfx;
    ``music`` = every other category (music, solo, hybrid, musiccaps,
    abstract, tada_test: the catalogue's 3700); else that category."""
    pop = (pop or "all").strip().lower()
    if pop in ("all", "", "*"):
        return np.ones(len(cats), dtype=bool)
    if pop == "music":
        return ~np.isin(cats, ["sfx", "sfx_holdout"])
    if pop == "sfx":
        return np.isin(cats, ["sfx", "sfx_holdout"])
    m = cats == pop
    if not m.any():
        raise SystemExit(f"population {pop!r} matches no prompt category ({sorted(set(cats))})")
    return m


TOP_N_SCALE = 1.0


def class_masks(y: np.ndarray, rule: str, q: float):
    """Positive / negative rows of ``y`` (sign-adjusted; NaN = outside the
    population or unlabelled) from the catalogue ``class_rule``:
    ``top N by score vs bottom P% of population`` (sparse labels) or
    ``quartile (top P% vs bottom P% ...)``; empty rule = ``q`` both sides."""
    rule = (rule or "").lower()
    ok = np.isfinite(y)
    k = int(ok.sum())
    if k < 8:
        return None, None
    m_top_n = re.search(r"top\s+(\d+)\s+by", rule)
    m_top_p = re.search(r"top\s+(\d+(?:\.\d+)?)\s*%", rule)
    m_bot_p = re.search(r"bottom\s+(\d+(?:\.\d+)?)\s*%", rule)
    q_bot = float(m_bot_p.group(1)) / 100 if m_bot_p else q
    lo = np.quantile(y[ok], q_bot)
    neg = ok & (y <= lo)
    if m_top_n:
        n = min(int(round(int(m_top_n.group(1)) * TOP_N_SCALE)), k // 2)
        order = np.argsort(np.where(ok, -y, np.inf), kind="stable")[:n]
        pos = np.zeros_like(ok)
        pos[order] = True
        hi = y[order].min() if n else np.inf
    else:
        q_top = float(m_top_p.group(1)) / 100 if m_top_p else q
        hi = np.quantile(y[ok], 1.0 - q_top)
        pos = ok & (y >= hi)
    neg &= ~pos
    if hi <= lo or pos.sum() < 4 or neg.sum() < 4:   # >= 2 per class per fold
        return None, None
    return pos, neg


def parse_sign(v) -> int:
    v = str(v).strip() if v is not None else ""
    if v in ("", "+", "+1", "1", "1.0", "+1.0", "pos"):
        return 1
    if v in ("-", "-1", "-1.0", "neg"):
        return -1
    raise SystemExit(f"bad sign {v!r}")


def _class_diff(flat: np.ndarray, P: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """``mean(pos) - mean(neg)`` per concept: [C, D] from ``flat`` [N, D]."""
    return (P @ flat) / P.sum(1)[:, None] - (Q @ flat) / Q.sum(1)[:, None]


def _proj_effect(proj: np.ndarray, P: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """Class gap of ``proj`` [N, C] / pooled within-class std, per concept."""
    npos, nneg = P.sum(1), Q.sum(1)
    sp, sq = np.einsum("cn,nc->c", P, proj), np.einsum("cn,nc->c", Q, proj)
    sp2, sq2 = np.einsum("cn,nc->c", P, proj * proj), np.einsum("cn,nc->c", Q, proj * proj)
    mp, mn = sp / np.maximum(npos, 1), sq / np.maximum(nneg, 1)
    vp = (sp2 - npos * mp * mp) / np.maximum(npos - 1, 1)
    vn = (sq2 - nneg * mn * mn) / np.maximum(nneg - 1, 1)
    pooled = np.sqrt(np.maximum((vp * (npos - 1) + vn * (nneg - 1)) / np.maximum(npos + nneg - 2, 1), 1e-24))
    return (mp - mn) / pooled


def block_directions(xb: np.ndarray, pos: np.ndarray, neg: np.ndarray):
    """Every concept at one block.

    ``xb`` [N, steps, H] float32; ``pos`` / ``neg`` [C, N] bool. Returns
    ``unit`` [C, steps, H] and ``norm`` [C, steps] (raw class-mean gap) from
    all rows, ``std`` [C, steps] (projection std over all clips along
    ``unit``) and ``effect`` [C, steps]: projection class gap / pooled
    within-class std, CROSS-FITTED (direction from even rows scored on odd
    rows and vice versa, mean of the two). The in-sample gap is biased up by
    ``sqrt(H * 2 / n_class)`` pooled stds of pure noise (1536 dims), which
    would make every block look alike; held-out rows remove that bias."""
    n, steps, h = xb.shape
    P, Q = pos.astype(np.float32), neg.astype(np.float32)
    flat = xb.reshape(n, steps * h)
    diff = _class_diff(flat, P, Q).reshape(-1, steps, h)
    norm = np.linalg.norm(diff, axis=-1)
    unit = diff / np.maximum(norm[..., None], 1e-12)
    std = np.zeros_like(norm)
    for s in range(steps):
        std[:, s] = (xb[:, s, :] @ unit[:, s, :].T).std(0, ddof=1)
    effect = np.zeros_like(norm)
    fold = (np.arange(n) % 2).astype(bool)
    for f in (False, True):
        fit, held = fold == f, fold != f
        Pf, Qf = P * fit, Q * fit
        Ph, Qh = P[:, held], Q[:, held]
        df = _class_diff(flat[fit], Pf[:, fit], Qf[:, fit]).reshape(-1, steps, h)
        uf = df / np.maximum(np.linalg.norm(df, axis=-1, keepdims=True), 1e-12)
        for s in range(steps):
            effect[:, s] += 0.5 * _proj_effect(xb[held, s, :] @ uf[:, s, :].T, Ph, Qh)
    return unit, norm, std, effect


def resolve_col(col: str, columns, aliases: dict) -> str:
    """``col`` if the label table has it, else with its scorer prefix mapped
    through ``aliases`` (``--col-alias proxies=desc``), else ``col``."""
    if col in columns or "." not in col:
        return col
    pre, rest = col.split(".", 1)
    for alt in aliases.get(pre, []):
        if f"{alt}.{rest}" in columns:
            return f"{alt}.{rest}"
    return col


def concept_specs(rows: list, labels, quantile: float, cats: np.ndarray, aliases: dict | None = None) -> list:
    """Catalogue rows + class masks (``_pos``/``_neg``), label stds and the
    usable independent scorer columns (``second_cols``: second then third
    scorer, those present in the label table; ``second_corrs``: their corpus
    correlation with ``sign * label`` = the direction they should move)."""
    specs = []
    for r in rows:
        r = dict(r)
        r["name"] = slug(r["name"])
        r["sign"] = parse_sign(r.get("sign"))
        aliases = aliases or {}
        col = resolve_col(r["label_col"], labels.columns, aliases)
        if col != r["label_col"]:
            r["label_col_catalogue"], r["label_col"] = r["label_col"], col
        if col not in labels.columns:
            r["status"] = "missing_label"
            specs.append(r)
            continue
        raw = labels[col].to_numpy(dtype=np.float64)
        y = r["sign"] * raw
        y[~population_mask(r.get("population", ""), cats)] = np.nan
        pos, neg = class_masks(y, r.get("class_rule", ""), quantile)
        r["status"] = "ok" if pos is not None else "too_few"
        r["_pos"], r["_neg"], r["_y"] = pos, neg, y
        r["label_std"] = round(float(np.nanstd(raw)), 6)
        cols, corrs = [], []
        for key in SECOND_COLS:
            for sec in (r.get(key) or "").replace("|", ";").split(";"):
                sec = resolve_col(sec.strip(), labels.columns, aliases)
                if not sec or sec not in labels.columns or sec in cols or sec == col:
                    continue
                y2 = labels[sec].to_numpy(dtype=np.float64)
                both = np.isfinite(y) & np.isfinite(y2)
                if both.sum() > 2 and np.nanstd(y2[both]) > 0:
                    cols.append(sec)
                    corrs.append(round(float(np.corrcoef(y[both], y2[both])[0, 1]), 4))
        r["second_cols"] = ";".join(cols)
        r["second_corrs"] = ";".join(str(c) for c in corrs)
        r["second_std"] = (round(float(np.nanstd(labels[cols[0]].to_numpy(dtype=np.float64))), 6)
                           if cols else "")
        specs.append(r)
    return specs


def generic_main(args) -> int:
    import csv
    import time

    cap = Path(args.cap)
    means, meta, ids, cats = load_corpus(cap)
    n, steps, nb, h = means.shape
    labels = load_labels(Path(args.labels) if args.labels else cap / "labels" / "merged.parquet", ids)
    out = Path(args.dirs_out) if args.dirs_out else cap / "dirs"
    groups = None
    if args.center_prompt:
        # variant d: labels and residual means centred on their prompt's mean over its seeds
        pr = json.loads((cap / "prompts.json").read_text())[: means.shape[0]]
        groups = np.array([int(r.get("base_id", r["id"])) for r in pr])
        num = labels.select_dtypes("number")
        labels = num - num.groupby(groups).transform("mean")
        _u, g_inv = np.unique(groups, return_inverse=True)
        g_cnt = np.bincount(g_inv).astype(np.float32)
        print(f"[center] {len(_u)} prompt groups (median size {int(np.median(g_cnt))})", flush=True)
    out.mkdir(parents=True, exist_ok=True)
    if args.catalogue:
        rows = read_catalogue(Path(args.catalogue))
    else:
        rows = [{"name": args.name or args.label_col, "label_col": args.label_col, "sign": str(args.sign),
                 "category": "", "second_scorer_col": args.second_col or ""}]
    aliases = {}
    for a in args.col_alias:
        k, v = a.split("=", 1)
        aliases.setdefault(k.strip(), []).extend(x.strip() for x in v.split(",") if x.strip())
    specs = concept_specs(rows, labels, args.quantile, cats, aliases)
    live = [r for r in specs if r["status"] == "ok"]
    print(f"[load] {cap} means {means.shape}; {len(specs)} concepts, {len(live)} with usable labels", flush=True)
    if not live:
        raise SystemExit("no concept has a usable label column")
    pos = np.stack([r["_pos"] for r in live])
    neg = np.stack([r["_neg"] for r in live])
    c = len(live)
    bid = block_ids(meta, nb)
    nb_model = int(meta.get("model_blocks", max(bid) + 1)) if meta else nb
    print(f"[load] stored blocks {bid} (model has {nb_model})", flush=True)
    # all blocks held as fp16 until the best 4 are known (500 concepts ~ 0.3 GB)
    units = np.zeros((c, nb, steps, h), dtype=np.float16)
    norms = np.zeros((c, nb, steps), dtype=np.float32)
    stds = np.zeros((c, steps, nb), dtype=np.float32)
    effects = np.zeros((c, steps, nb), dtype=np.float32)
    t0 = time.time()
    for b in range(nb):
        xb = np.asarray(means[:, :, b, :], dtype=np.float32)
        if groups is not None:
            flat = xb.reshape(n, -1)
            gsum = np.zeros((len(g_cnt), flat.shape[1]), dtype=np.float32)
            np.add.at(gsum, g_inv, flat)
            xc = (flat - (gsum / g_cnt[:, None])[g_inv]).reshape(xb.shape)
            u, nr, sd, ef = block_directions(xc, pos, neg)
            for st in range(steps):        # alpha unit = projection std of the RAW activations
                sd[:, st] = (xb[:, st, :] @ u[:, st, :].T).std(0, ddof=1)
            del xc, flat, gsum
        else:
            u, nr, sd, ef = block_directions(xb, pos, neg)
        units[:, b], norms[:, b] = u.astype(np.float16), nr
        stds[:, :, b], effects[:, :, b] = sd, ef
        print(f"[block {bid[b]:02d}] {time.time() - t0:.0f} s", flush=True)

    keep = min(args.keep_blocks, nb)
    stds_full = np.full((c, steps, nb_model), np.nan, dtype=np.float32)
    effects_full = np.full((c, steps, nb_model), np.nan, dtype=np.float32)
    stds_full[:, :, bid], effects_full[:, :, bid] = stds, effects
    variants = [v for v in (args.variants or "a").split(",") if v]
    best_local = np.array([int(np.argmax(effects[i].mean(0))) for i in range(c)])
    var_c = {}
    if "c" in variants:
        var_c = asym_directions(means, live, pos, neg, best_local, bid, cats)
    extra = []
    for r in specs:
        for k in r:
            if not k.startswith("_") and k not in INDEX_COLS and k not in extra:
                extra.append(k)
    out_rows = []
    var_rows = {}
    for r in specs:
        row = {k: v for k, v in r.items() if not k.startswith("_")}
        if r["status"] == "ok":
            i = next(j for j, x in enumerate(live) if x is r)
            score = effects[i].mean(0)                                 # step-mean effect per block
            loc = [int(x) for x in np.argsort(-score)[:keep]]
            blocks = [bid[x] for x in loc]
            best = blocks[0]
            path = out / f"{r['name']}.npz"
            # variant a: step-mean unit at the best block (what screen_knobs / make_packs use)
            sm = (units[i, loc[0]].astype(np.float64) * norms[i, loc[0]][:, None]).mean(0)
            vec_a = (sm / max(np.linalg.norm(sm), 1e-12))[None].astype(np.float32)
            std_best = float(stds[i, :, loc[0]].mean())
            np.savez(path, unit=units[i, loc].transpose(1, 0, 2), blocks=np.array(blocks),
                     best_block=np.int64(best), norm=norms[i, loc].T.astype(np.float32),
                     vec=vec_a, vec_blocks=np.array([best]), vec_std=np.float64(std_best),
                     variant=np.array(args.variant_tag),
                     std=stds_full[i], effect=effects_full[i], n_pos=np.int64(r["_pos"].sum()),
                     n_neg=np.int64(r["_neg"].sum()), label_col=np.array(r["label_col"]),
                     pos_idx=np.flatnonzero(r["_pos"]).astype(np.int32),
                     neg_idx=np.flatnonzero(r["_neg"]).astype(np.int32),
                     sign=np.int64(r["sign"]), quantile=np.float64(args.quantile),
                     name=np.array(r["name"]), capture=np.array(str(cap)),
                     method=np.array("mass_mean_quartile"),
                     hook=np.array(meta.get("hook", "post_block_residual")))
            row.update(n_pos=int(r["_pos"].sum()), n_neg=int(r["_neg"].sum()), best_block=best,
                       blocks=" ".join(map(str, blocks)), effect_best=round(float(score[loc[0]]), 5),
                       std_best=round(std_best, 5),
                       norm_best=round(float(norms[i, loc[0]].mean()), 5), quantile=args.quantile,
                       path=str(path), variant=args.variant_tag, base=r["name"])
            if "b" in variants:
                # variant b: the top-3 blocks together, each scaled by its raw step-mean norm relative to the best block
                l3 = loc[:3]
                vb = []
                for x in l3:
                    smx = (units[i, x].astype(np.float64) * norms[i, x][:, None]).mean(0)
                    vb.append(smx / max(float(np.linalg.norm(sm)), 1e-12))   # unit_b x |raw_b| / |raw_best|
                pb = out.parent / (out.name + "_b") / f"{r['name']}.npz"
                pb.parent.mkdir(parents=True, exist_ok=True)
                np.savez(pb, unit=units[i, loc].transpose(1, 0, 2), blocks=np.array(blocks),
                         best_block=np.int64(best), norm=norms[i, loc].T.astype(np.float32),
                         vec=np.stack(vb).astype(np.float32), vec_blocks=np.array([bid[x] for x in l3]),
                         vec_std=np.float64(std_best), variant=np.array("b"),
                         std=stds_full[i], effect=effects_full[i], n_pos=np.int64(r["_pos"].sum()),
                         n_neg=np.int64(r["_neg"].sum()), label_col=np.array(r["label_col"]),
                         pos_idx=np.flatnonzero(r["_pos"]).astype(np.int32),
                         neg_idx=np.flatnonzero(r["_neg"]).astype(np.int32), sign=np.int64(r["sign"]),
                         name=np.array(r["name"]), hook=np.array(meta.get("hook", "post_block_residual")))
                var_rows.setdefault("b", []).append({**row, "path": str(pb), "variant": "b", "base": r["name"],
                                                     "vec_blocks": " ".join(str(bid[x]) for x in l3),
                                                     "k_scale": 0.5})
            if i in var_c:
                for side, vc in var_c[i].items():
                    nm = f"{r['name'][:29]}_{side}"
                    pc = out.parent / (out.name + "_c") / f"{nm}.npz"
                    pc.parent.mkdir(parents=True, exist_ok=True)
                    sgn = r["sign"] if side == "up" else -r["sign"]
                    np.savez(pc, unit=vc["unit"][:, None].astype(np.float16), blocks=np.array([best]),
                             best_block=np.int64(best), norm=vc["norm"][:, None].astype(np.float32),
                             vec=vc["vec"][None].astype(np.float32), vec_blocks=np.array([best]),
                             vec_std=np.float64(vc["std"]), variant=np.array("c"),
                             std=stds_full[i], effect=effects_full[i], n_pos=np.int64(len(vc["pos_idx"])),
                             n_neg=np.int64(len(vc["neg_idx"])), label_col=np.array(r["label_col"]),
                             pos_idx=vc["pos_idx"].astype(np.int32), neg_idx=vc["neg_idx"].astype(np.int32),
                             sign=np.int64(sgn), name=np.array(nm),
                             hook=np.array(meta.get("hook", "post_block_residual")))
                    var_rows.setdefault("c", []).append({
                        **row, "name": nm, "sign": sgn, "path": str(pc), "variant": "c",
                        "base": r["name"], "side": side, "std_best": round(vc["std"], 5),
                        "norm_best": round(float(vc["norm"].mean()), 5), "vec_blocks": str(best),
                        "effect_best": round(float(score[loc[0]]), 5)})
            print(f"[{r['name']}] {r['label_col']} sign {r['sign']:+d} n {row['n_pos']}/{row['n_neg']} "
                  f"best b{best} effect {score[loc[0]]:.2f} std {row['std_best']:.3f}", flush=True)
        else:
            print(f"[{r['name']}] SKIP {r['status']} ({r['label_col']})", flush=True)
        out_rows.append(row)
    idx = out / "index.csv"
    cols = list(INDEX_COLS) + extra + [k for k in ("variant", "base") if k not in extra]
    with open(idx, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, restval="")
        w.writeheader()
        for row in out_rows:
            w.writerow({k: row.get(k, "") for k in cols})
    miss = {}
    for r in specs:
        if r["status"] == "missing_label":
            pre = r["label_col"].split(".", 1)[0]
            miss[pre] = miss.get(pre, 0) + 1
    if miss:
        print(f"WARN {sum(miss.values())} concepts have no label column in the table, by scorer prefix: {miss} "
              "(see --col-alias)", flush=True)
    print(f"wrote {idx} ({len(live)} directions)", flush=True)
    for v, vrows in var_rows.items():
        vcols = cols + [k for k in ("variant", "base", "side", "vec_blocks", "k_scale") if k not in cols]
        vidx = out.parent / (out.name + f"_{v}") / "index.csv"
        with open(vidx, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=vcols, restval="")
            w.writeheader()
            for row in vrows:
                w.writerow({k: row.get(k, "") for k in vcols})
        print(f"wrote {vidx} ({len(vrows)} variant-{v} directions)", flush=True)
    return 0


def asym_directions(means, live, pos, neg, best_local, bid, cats, lo_q=0.375, hi_q=0.625) -> dict:
    """Variant c (asymmetric per sign), at each concept's best block:
    ``up`` = mean(positive class) - mean(median band), ``dn`` = mean(negative
    class) - mean(median band); median band = clips of the concept's
    population whose sign-adjusted label lies between the ``lo_q`` and
    ``hi_q`` quantiles (outside both classes). Returns ``{i: {"up"|"dn": {unit
    [steps,H] per-step units, norm [steps], vec [H] step-mean unit, std
    (projection std of every clip, step mean), pos_idx, neg_idx (= median band)}}}``."""
    n = means.shape[0]
    mids = []
    for i, r in enumerate(live):
        y = r["_y"]
        ok = np.isfinite(y) & ~pos[i] & ~neg[i]
        if ok.sum() < 8:
            mids.append(None)
            continue
        a, b = np.quantile(y[np.isfinite(y)], [lo_q, hi_q])
        m = ok & (y >= a) & (y <= b)
        mids.append(m if m.sum() >= 4 else None)
    out = {}
    for lb in sorted(set(best_local.tolist())):
        idx = [i for i in range(len(live)) if best_local[i] == lb and mids[i] is not None]
        if not idx:
            continue
        xb = np.asarray(means[:, :, lb, :], dtype=np.float32)
        steps, h = xb.shape[1], xb.shape[2]
        flat = xb.reshape(n, steps * h)
        M = np.stack([mids[i] for i in idx]).astype(np.float32)
        mid_mean = (M @ flat) / M.sum(1)[:, None]
        for side, cls in (("up", pos), ("dn", neg)):
            C = cls[idx].astype(np.float32)
            d = ((C @ flat) / C.sum(1)[:, None] - mid_mean).reshape(len(idx), steps, h)
            nrm = np.linalg.norm(d, axis=-1)
            unit = d / np.maximum(nrm[..., None], 1e-12)
            sm = d.mean(1)
            vec = sm / np.maximum(np.linalg.norm(sm, axis=-1, keepdims=True), 1e-12)
            std = np.zeros(len(idx))
            for st in range(steps):
                std += (xb[:, st, :] @ vec.T).std(0, ddof=1) / steps
            for j, i in enumerate(idx):
                out.setdefault(i, {})[side] = {
                    "unit": unit[j], "norm": nrm[j], "vec": vec[j], "std": float(std[j]),
                    "pos_idx": np.flatnonzero(cls[i]), "neg_idx": np.flatnonzero(mids[i])}
        print(f"[variant c] block {bid[lb]:02d}: {len(idx)} concepts", flush=True)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--capture", help="legacy modes: capture_resid.py --out dir")
    ap.add_argument("--out", help="legacy modes: bench root (sa3_tada_run.py --out)")
    g = ap.add_argument_group("generic mode (many knobs)")
    g.add_argument("--cap", help="corpus root ($CAP: means.npy, meta.json, prompts.json, labels/)")
    g.add_argument("--labels", help="label table (default $CAP/labels/merged.parquet)")
    g.add_argument("--catalogue", help="concept catalogue csv (name, label_col, sign, ...)")
    g.add_argument("--label-col", help="one concept: label column")
    g.add_argument("--sign", type=int, choices=(1, -1), default=1, help="one concept: +1 | -1")
    g.add_argument("--name", help="one concept: name (default from --label-col)")
    g.add_argument("--second-col", help="one concept: second scorer column")
    g.add_argument("--dirs-out", help="default $CAP/dirs")
    g.add_argument("--keep-blocks", type=int, default=KEEP_BLOCKS, help="blocks stored per concept")
    g.add_argument("--variants", default="a",
                   help="comma list: a = best block (dirs/), b = top-3 blocks (dirs_b/), "
                        "c = asymmetric per sign vs the median band (dirs_c/, <name>_up / <name>_dn)")
    g.add_argument("--center-prompt", action="store_true",
                   help="variant d: centre labels and means on the prompt mean over its seeds (prompts.json base_id)")
    g.add_argument("--variant-tag", default="a", help="variant name written for the --dirs-out directions")
    g.add_argument("--top-n-scale", type=float, default=1.0,
                   help="multiply 'top N by score' class sizes (4 for a corpus 4x the catalogue's 5000)")
    g.add_argument("--col-alias", nargs="*", default=[], metavar="PREFIX=ALT[,ALT]",
                   help="catalogue scorer prefix -> label table prefix when the exact column is missing, "
                        "e.g. proxies=desc spectral=desc level=dyn rhythm=dyn tonal=dyn")
    ap.add_argument("--self-label", action="store_true")
    ap.add_argument("--captions", default=DEFAULT_CAPTIONS)
    ap.add_argument("--tag", default=None, help="default resid_mc / resid_self")
    ap.add_argument("--concepts", nargs="+", default=list(CONCEPTS), choices=CONCEPTS)
    ap.add_argument("--class-cap", type=int, default=1000, help="MusicCaps: max rows per class (was --cap)")
    ap.add_argument("--quantile", type=float, default=0.25, help="self-label: top/bottom fraction")
    ap.add_argument("--block", nargs="*", default=[], metavar="CONCEPT=B",
                    help="pack/prod block override per concept (default: production pack block)")
    ap.add_argument("--per-step-pack", action="store_true")
    ap.add_argument("--prod-packs", type=Path, default=DEFAULT_PROD_PACKS,
                    help="production packs (read-only) for the cosine report")
    args = ap.parse_args()
    if args.cap:
        if not (args.catalogue or args.label_col):
            ap.error("--cap needs --catalogue or --label-col")
        global TOP_N_SCALE
        TOP_N_SCALE = float(args.top_n_scale)
        return generic_main(args)
    if not (args.capture and args.out):
        ap.error("legacy modes need --capture and --out (or use --cap for the generic mode)")

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
        classes = musiccaps_classes(args.captions, info_cap["caption_index"], n, args.class_cap,
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
