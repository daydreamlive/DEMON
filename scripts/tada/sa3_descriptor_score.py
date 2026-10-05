"""DSP descriptor alignment for the SA3 steering bench (CPU only).

Scores eval directories in the bench layout (``<dir>/alpha_<a>/audios.npz``,
int16 mono ``[N, 1, T]`` plus ``sr``, written by ``sa3_tada_run.py``) with
the production steering proxies (``proxies.py``, copied from DEMON-steer
``scripts/steering/proxies.py``), per clip:

* ``centroid_st``  spectral centroid in semitones, ``12 log2(Hz / 440) + 69``
* ``lowhigh_db``   energy below 250 Hz over energy above 2.5 kHz, dB
* ``perc_ratio``   HPSS percussive energy share
* ``onset_rate``   onsets per second
* ``flatness``     mean spectral flatness
* ``tm_brightness``, ``tm_warmth``, ``tm_roughness``, ``tm_hardness``:
  AudioCommons ``timbral_models`` (optional, ``--timbral``, only when
  ``import timbral_models`` works)

Subcommands:

* ``score``: per descriptor, ``<dir>/protocol_results/desc_<name>.csv`` in
  the column layout the reference AUC reads for an alignment metric
  (``alpha, mean, std, scores``; ``auc.py: load_and_merge_alignment``
  merges ``<metric>.csv`` with ``lpaps.csv`` on ``alpha``), plus
  ``desc_clips.csv`` (one row per clip). ``--auc`` chains ``auc``.
* ``auc``: for every directory under ``--root`` with ``lpaps.csv``, the
  reference alignment AUC (``auc.py: compute_alignment_auc_direction``)
  of each descriptor against the concept's PCI cutoff
  (``min(PCI-all, PCI-loc)`` max LPAPS; ``lpaps_endpoints.csv`` accepted),
  with the descriptor normalised by its PCI-all full-swap gap for that
  concept, ``G = mean(k = +max) - mean(k = -max)`` (signed, so a concept
  whose positive pole lowers the descriptor, e.g. density/onset_rate,
  still scores positive). The normalised column is written as
  ``desc_<name>_norm.csv`` and the AUC is reported with its steer/PCI-all
  ratio (dimensionless). MuQ (``muqt.csv``) is a secondary column when
  present. Also writes the cross-effect table (``cross_effect.csv``): for
  each knob directory x descriptor, the slope of the descriptor vs alpha
  near 0 and the descriptor delta (vs alpha 0) at the PCI cutoff.

    python scripts/tada/sa3_descriptor_score.py score --root <out>/<eval_sub> --workers 16
    python scripts/tada/sa3_descriptor_score.py auc --root <out>/<eval_sub>
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

BASE = ("centroid_st", "lowhigh_db", "perc_ratio", "onset_rate", "flatness")
TIMBRAL = ("tm_brightness", "tm_warmth", "tm_roughness", "tm_hardness")
#: The concept each descriptor measures (its "home" knob), for the
#: cross-effect normalisation.
HOME = {
    "centroid_st": "bright", "lowhigh_db": "warm", "perc_ratio": "percussive",
    "onset_rate": "density", "flatness": "rough",
    "tm_brightness": "bright", "tm_warmth": "warm", "tm_roughness": "rough",
    "tm_hardness": "percussive",
}
REF = Path(os.environ.get("TADA_REF", r"E:\Projects\tada-replication\steer-audio"))


def timbral_available() -> bool:
    try:
        import timbral_models  # noqa: F401
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# per-clip descriptors (worker side)
# ---------------------------------------------------------------------------

_NPZ: dict = {}


def _clip(job):
    """``job = (audios.npz path, row, timbral)`` -> {descriptor: value}.
    Each worker keeps the last npz it read (jobs arrive grouped by
    strength), so audio is read once per worker and strength instead of
    being pickled through the pool."""
    import proxies

    path, row, timbral = job
    if _NPZ.get("path") != path:
        z = np.load(path)
        _NPZ.clear()
        _NPZ.update(path=path, audio=z["audio"], sr=int(z["sr"]))
    a, sr = _NPZ["audio"][row], _NPZ["sr"]
    y = a.astype(np.float32) / 32768.0
    if y.ndim > 1:
        y = y.mean(axis=0)
    out = {}
    c = proxies.measure("centroid", y, sr)
    out["centroid_st"] = 12.0 * math.log2(c / 440.0) + 69.0 if c > 0 else float("nan")
    for name in ("lowhigh_db", "perc_ratio", "onset_rate", "flatness"):
        out[name] = proxies.measure(name, y, sr)
    if timbral:
        import timbral_models as tm

        for name, fn in (("tm_brightness", "timbral_brightness"), ("tm_warmth", "timbral_warmth"),
                         ("tm_roughness", "timbral_roughness"), ("tm_hardness", "timbral_hardness")):
            try:
                out[name] = float(getattr(tm, fn)(y, fs=sr))
            except Exception:
                out[name] = float("nan")
    return out


# ---------------------------------------------------------------------------
# layout helpers
# ---------------------------------------------------------------------------

def _alpha_dirs(d: Path):
    got = []
    for p in d.glob("alpha_*"):
        if (p / "audios.npz").exists():
            got.append((float(p.name[len("alpha_"):]), p))
    return sorted(got)


def _eval_dirs(root: Path):
    return [d for d in sorted(root.iterdir()) if d.is_dir() and _alpha_dirs(d)]


def _split(name: str):
    """``method_site_concept`` (the bench naming; concept may hold ``_``)."""
    method, site, concept = name.split("_", 2)
    return method, site, concept


def _read_curve(f: Path) -> dict:
    """``{alpha: mean}`` from a reference-layout csv."""
    if not f.exists():
        return {}
    return {round(float(r["alpha"]), 6): float(r["mean"]) for r in csv.DictReader(f.open())}


# ---------------------------------------------------------------------------
# score
# ---------------------------------------------------------------------------

def score_dir(d: Path, pool, names, timbral: bool, force: bool) -> bool:
    pr = d / "protocol_results"
    if not force and all((pr / f"desc_{n}.csv").exists() for n in names):
        return False
    t0 = time.time()
    keys, jobs = [], []
    for alpha, ad in _alpha_dirs(d):
        f = ad / "audios.npz"
        with np.load(f) as z:
            n = int(z["audio"].shape[0])
        keys += [(alpha, i) for i in range(n)]
        jobs += [(str(f), i, timbral) for i in range(n)]
    # every strength in one map so all workers stay busy
    rows = [(a, i, r) for (a, i), r in zip(keys, pool.map(_clip, jobs, chunksize=8))]
    pr.mkdir(parents=True, exist_ok=True)
    with (pr / "desc_clips.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["alpha", "clip", *names])
        for alpha, i, r in rows:
            w.writerow([alpha, i, *[r.get(n, float("nan")) for n in names]])
    alphas = sorted({a for a, _, _ in rows})
    for n in names:
        with (pr / f"desc_{n}.csv").open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["alpha", "mean", "std", "scores"])
            for a in alphas:
                v = np.array([r.get(n, np.nan) for al, _, r in rows if al == a], dtype=float)
                w.writerow([a, float(np.nanmean(v)) if np.isfinite(v).any() else float("nan"),
                            float(np.nanstd(v)) if np.isfinite(v).any() else float("nan"),
                            json.dumps([None if not np.isfinite(x) else float(x) for x in v])])
    print(f"score {d.name}: {len(rows)} clips, {len(alphas)} strengths, {time.time() - t0:.1f}s", flush=True)
    return True


def cmd_score(args) -> None:
    timbral = bool(args.timbral)
    if timbral and not timbral_available():
        print("timbral_models not importable: AudioCommons descriptors skipped", flush=True)
        timbral = False
    names = list(BASE) + (list(TIMBRAL) if timbral else [])
    dirs = [Path(p) for p in args.dirs] if args.dirs else _eval_dirs(Path(args.root))
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for d in dirs:
            score_dir(d, pool, names, timbral, args.force)
    if args.auc:
        if not args.root:
            raise SystemExit("--auc needs --root")
        cmd_auc(args)


# ---------------------------------------------------------------------------
# auc + cross-effect
# ---------------------------------------------------------------------------

def _reference_auc():
    """The reference ``src/steering/eval/auc.py`` loaded by path (numpy +
    pandas only; the package ``__init__`` pulls in the ACE pipeline)."""
    f = REF / "src" / "steering" / "eval" / "auc.py"
    if not f.exists():
        raise SystemExit(f"reference auc.py not found at {f} (set TADA_REF)")
    spec = importlib.util.spec_from_file_location("ref_auc", f)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _lpaps_curve(d: Path) -> dict:
    pr = d / "protocol_results"
    return _read_curve(pr / "lpaps.csv") or _read_curve(pr / "lpaps_endpoints.csv")


def pci_cutoff(root: Path, concept: str, direction: str) -> float:
    """``min(PCI-all, PCI-loc)`` of the max mean LPAPS in one direction
    (``lpaps.csv``, else ``lpaps_endpoints.csv``), as
    ``sa3_tada_score.py: _pci_cutoff``."""
    sign = 1 if direction == "pos" else -1
    vals = []
    for s in ("all", "loc"):
        cur = _lpaps_curve(root / f"pci_{s}_{concept}")
        side = [v for a, v in cur.items() if sign * a > 0]
        if side:
            vals.append(max(side))
    return min(vals) if vals else float("nan")


def pci_gap(root: Path, concept: str, name: str) -> float:
    """PCI-all full-swap gap of one descriptor: mean at the strongest
    positive switch minus mean at the strongest negative switch."""
    cur = _read_curve(root / f"pci_all_{concept}" / "protocol_results" / f"desc_{name}.csv")
    if not cur or min(cur) >= 0 or max(cur) <= 0:
        return float("nan")
    return cur[max(cur)] - cur[min(cur)]


def _write_norm(d: Path, name: str, gap: float) -> str:
    pr = d / "protocol_results"
    metric = f"desc_{name}_norm"
    with (pr / f"desc_{name}.csv").open() as f, (pr / f"{metric}.csv").open("w", newline="") as g:
        w = csv.writer(g)
        w.writerow(["alpha", "mean", "std"])
        for r in csv.DictReader(f):
            w.writerow([r["alpha"], float(r["mean"]) / gap, float(r["std"]) / abs(gap)])
    return metric


def _at_cutoff(desc: dict, lp: dict, direction: str, cutoff: float):
    """Descriptor delta vs alpha 0 where the sweep's LPAPS first reaches
    ``cutoff`` (linear in LPAPS, in order of increasing |alpha|); the
    strongest alpha when it never does. Returns (delta, alpha, reached)."""
    sign = 1 if direction == "pos" else -1
    if 0.0 not in desc:
        return float("nan"), float("nan"), False
    base = desc[0.0]
    side = sorted((abs(a), a) for a in desc if sign * a > 0)
    if not side:
        return float("nan"), float("nan"), False
    prev_lp, prev_v, prev_a = lp.get(0.0, 0.0), base, 0.0
    if lp and math.isfinite(cutoff):
        for _, a in side:
            if a not in lp:
                continue
            if lp[a] >= cutoff:
                span = lp[a] - prev_lp
                fr = (cutoff - prev_lp) / span if span > 1e-12 else 1.0
                v = prev_v + fr * (desc[a] - prev_v)
                return v - base, prev_a + fr * (a - prev_a), True
            prev_lp, prev_v, prev_a = lp[a], desc[a], a
    a = side[-1][1]
    return desc[a] - base, a, False


def _slope0(desc: dict) -> float:
    pos = [a for a in desc if a > 0]
    neg = [a for a in desc if a < 0]
    if pos and neg:
        a1, a0 = min(pos), max(neg)
        return (desc[a1] - desc[a0]) / (a1 - a0)
    if pos and 0.0 in desc:
        a1 = min(pos)
        return (desc[a1] - desc[0.0]) / a1
    return float("nan")


def _names_present(d: Path):
    pr = d / "protocol_results"
    return [n for n in (*BASE, *TIMBRAL) if (pr / f"desc_{n}.csv").exists()]


def cmd_auc(args) -> None:
    root = Path(args.root)
    dirs = _eval_dirs(root)
    concepts = sorted({_split(d.name)[2] for d in dirs})
    if args.concepts:
        concepts = [c for c in concepts if c in args.concepts]
    cut = {c: {dn: pci_cutoff(root, c, dn) for dn in ("pos", "neg")} for c in concepts}
    names_all = sorted({n for d in dirs for n in _names_present(d)}, key=(*BASE, *TIMBRAL).index)
    gaps = {c: {n: pci_gap(root, c, n) for n in names_all} for c in set(concepts) | set(HOME.values())}

    try:
        ref = _reference_auc()
    except SystemExit as e:
        ref = None
        print(f"AUC skipped: {e}", flush=True)

    auc_rows, cross_rows = [], []
    auc = {}
    for d in dirs:
        method, site, concept = _split(d.name)
        if concept not in concepts:
            continue
        pr = d / "protocol_results"
        has_lp = (pr / "lpaps.csv").exists()
        lp = _lpaps_curve(d)
        meta = json.loads((d / "sweep.json").read_text()) if (d / "sweep.json").exists() else {}
        mag = (meta.get("pack") or {}).get("magnitude")
        for n in _names_present(d):
            desc = _read_curve(pr / f"desc_{n}.csv")
            g = gaps.get(concept, {}).get(n, float("nan"))
            # AUC (needs the full lpaps curve and a usable PCI gap)
            if ref is not None and has_lp and math.isfinite(g) and abs(g) > 1e-12:
                metric = _write_norm(d, n, g)
                metrics = [metric] + (["muqt"] if n == BASE[0] and (pr / "muqt.csv").exists() else [])
                for dn in ("pos", "neg"):
                    c = cut[concept][dn]
                    if not math.isfinite(c):
                        continue
                    res = ref.compute_alignment_auc_direction(pr, dn, c, metrics)
                    e = auc.setdefault(d.name, {}).setdefault(n, {})
                    e[dn] = res.get(metric, float("nan"))
                    e[f"cutoff_{dn}"] = c
                    if "muqt" in res:
                        auc.setdefault(d.name, {}).setdefault("muqt", {})[dn] = res["muqt"]
            # cross effect
            hg = gaps.get(HOME.get(n, ""), {}).get(n, float("nan"))
            row = {"dir": d.name, "method": method, "site": site, "knob": concept, "descriptor": n,
                   "pci_gap_own": g, "pci_gap_home": hg, "slope0_per_alpha": _slope0(desc),
                   "slope0_per_knob_unit": _slope0(desc) * mag if mag else float("nan")}
            for dn in ("pos", "neg"):
                delta, a, reached = _at_cutoff(desc, lp, dn, cut[concept][dn])
                row[f"delta_at_cut_{dn}"] = delta
                row[f"delta_at_cut_{dn}_home_norm"] = (delta / hg if math.isfinite(hg) and abs(hg) > 1e-12
                                                       else float("nan"))
                row[f"alpha_at_cut_{dn}"] = a
                row[f"knob_at_cut_{dn}"] = a / mag if mag else float("nan")
                row[f"reached_{dn}"] = reached
            cross_rows.append(row)

    # steer / PCI-all ratios
    for name, per in auc.items():
        method, site, concept = _split(name)
        pci = auc.get(f"pci_all_{concept}", {})
        for n, e in per.items():
            avg = np.nanmean([e.get("pos", np.nan), e.get("neg", np.nan)])
            p = pci.get(n, {})
            pavg = np.nanmean([p.get("pos", np.nan), p.get("neg", np.nan)]) if p else float("nan")
            e["avg"] = float(avg)
            e["pci_all_avg"] = float(pavg)
            e["ratio"] = float(avg / pavg) if pavg and math.isfinite(pavg) and abs(pavg) > 1e-12 else float("nan")
            auc_rows.append({"dir": name, "knob": concept, "metric": n, **e})

    (root / "auc_desc.json").write_text(json.dumps(auc, indent=1))
    _write_rows(root / "auc_desc.csv", auc_rows)
    _write_rows(root / "cross_effect.csv", cross_rows)
    for r in auc_rows:
        print(f"{r['dir']:28s} {r['metric']:14s} pos {r.get('pos', float('nan')):+.4f} "
              f"neg {r.get('neg', float('nan')):+.4f} avg {r['avg']:+.4f} "
              f"pci {r['pci_all_avg']:+.4f} ratio {r['ratio']:+.2f}")
    _print_cross(cross_rows)


def _write_rows(f: Path, rows) -> None:
    keys = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    with f.open("w", newline="") as g:
        w = csv.DictWriter(g, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {f} ({len(rows)} rows)", flush=True)


def _print_cross(rows) -> None:
    """Knob dirs x descriptors: delta at the positive cutoff in units of
    the descriptor's home-concept PCI gap (raw delta when no gap)."""
    knobs = [r for r in rows if r["method"] != "pci"]
    if not knobs:
        return
    dirs = sorted({r["dir"] for r in knobs})
    names = [n for n in (*BASE, *TIMBRAL) if any(r["descriptor"] == n for r in knobs)]
    print("cross effect: delta at +cutoff / home PCI gap  [slope0 per alpha]")
    print(" " * 28 + "".join(f"{n:>22s}" for n in names))
    for d in dirs:
        cells = []
        for n in names:
            r = next((x for x in knobs if x["dir"] == d and x["descriptor"] == n), None)
            if r is None:
                cells.append(f"{'':>22s}")
                continue
            v = r["delta_at_cut_pos_home_norm"]
            if not math.isfinite(v):
                v = r["delta_at_cut_pos"]
            cells.append(f"{v:+9.3f} [{r['slope0_per_alpha']:+9.4f}]")
        print(f"{d:28s}" + "".join(cells))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=("score", "auc"))
    ap.add_argument("dirs", nargs="*", help="score: eval dirs (default: every dir under --root)")
    ap.add_argument("--root", default=None, help="eval root (<out>/<eval_sub>)")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--timbral", action="store_true", help="also AudioCommons timbral_models (slow)")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--auc", action="store_true", help="score: run auc on --root afterwards")
    ap.add_argument("--concepts", nargs="*", default=None)
    args = ap.parse_args()
    if args.cmd == "score" and not (args.dirs or args.root):
        raise SystemExit("score needs dirs or --root")
    if args.cmd == "auc" and not args.root:
        raise SystemExit("auc needs --root")
    {"score": cmd_score, "auc": cmd_auc}[args.cmd](args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
