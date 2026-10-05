"""Production steering packs from screened directions (phase 6 input).

For each concept: ``dirs/<name>.npz`` (build_directions.py --cap) and
``screen/<name>.json`` (screen_knobs.py) -> ``packs/sa3/medium/<name>.safetensors``,
production format 1 via the branch's ``acestep.steering.packs.save_pack``:
the step-mean unit vector at the best block (``mean_s(unit[s] * norm[s])``,
normalised), ``norm`` = its raw L2 norm, ``magnitude`` = norm x 0.1 (knob 10 =
one raw quartile mean gap), hook ``post_block_residual``, policy range 0..1.
Provenance: method mass_mean_quartile, label_col, sign, n_pos, n_neg, capture
path, screening numbers, and the calibrated gain per sign = alpha at the
screening LPAPS cutoff / magnitude (the knob value that reaches the cutoff;
the negative side is mirrored from the positive one when screening ran
``--signs pos``). Each pack is re-read with ``load_pack`` and checked.

Concepts: ``--concepts`` names, else every row of ``screen/accepted.csv``
with accepted = 1 (screen_knobs.py --finalize), else every screened json
with pass = true. ``--pci-descriptors`` also writes ``pci_descriptors.json``
(name -> [positive, negative] text from catalogue columns ``pos_text`` /
``neg_text``, else ``pos_anchor`` / ``neg_anchor``, in dirs/index.csv) for
``sa3_tada_run.py --pci-descriptors``.
"""

from __future__ import annotations

import argparse
import csv
import datetime as _dt
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
KNOB_UNIT = 0.1


def pick(args, screen: Path) -> list:
    if args.concepts:
        return list(args.concepts)
    acc = screen / "accepted.csv"
    if acc.exists():
        with open(acc, newline="", encoding="utf-8") as f:
            return [r["name"] for r in csv.DictReader(f) if r["accepted"] == "1"]
    out = []
    for p in sorted(screen.glob("*.json")):
        if json.loads(p.read_text()).get("pass"):
            out.append(p.stem)
    return out


def stepmean(npz) -> tuple:
    b = int(npz["best_block"])
    k = list(npz["blocks"]).index(b)
    sm = (npz["unit"][:, k].astype(np.float64) * npz["norm"][:, k][:, None]).mean(0)
    n = float(np.linalg.norm(sm))
    return sm / max(n, 1e-12), n, b


def _lpaps_curve(d: Path):
    for fn in ("lpaps.csv", "lpaps_endpoints.csv"):
        f = d / "protocol_results" / fn
        if f.exists():
            with open(f, newline="") as fh:
                return {float(r["alpha"]): float(r["mean"]) for r in csv.DictReader(fh)}
    return None


def phase6_gain(res: Path, name: str, mag: float) -> dict:
    """Per-sign knob value at the PCI cutoff, as ``sa3_tada_run.py calibrate``
    picks it: the smallest probed |alpha| whose mean LPAPS reaches that sign's
    PCI-all max LPAPS on the 50 test prompts, else the largest probe
    (``reached`` false). Read from the phase 6 per-knob results dir."""
    pci = _lpaps_curve(res / "eval" / f"pci_all_{name}")
    probe = _lpaps_curve(res / "calib" / f"pack_pack_{name}")
    if not pci or not probe:
        raise SystemExit(f"{name}: no phase 6 PCI / probe LPAPS under {res}")
    gain = {"unit": "knob value at which mean LPAPS vs the unsteered render reaches the PCI-all cutoff",
            "source": "phase 6 full protocol (probe 20 holdout prompts, PCI-all on 50 TADA test prompts)"}
    for direction, sign in (("pos", 1), ("neg", -1)):
        cut = max(v for a, v in pci.items() if sign * a > 0)
        probes = sorted((abs(a), v) for a, v in probe.items() if sign * a > 0)
        hit = [a for a, v in probes if v >= cut]
        a = hit[0] if hit else probes[-1][0]
        gain[direction] = a / mag
        gain[f"{direction}_alpha"] = sign * a
        gain[f"{direction}_cutoff"] = cut
        gain[f"{direction}_reached"] = bool(hit)
    return gain


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cap", required=True, help="corpus root ($CAP)")
    ap.add_argument("--dirs", help="default $CAP/dirs")
    ap.add_argument("--screen", help="default $CAP/screen")
    ap.add_argument("--out", help="default $CAP/packs/sa3/medium")
    ap.add_argument("--concepts", nargs="*", default=[])
    ap.add_argument("--allow-unscreened", action="store_true", help="pack without a screen json (no gain)")
    ap.add_argument("--pci-descriptors", action="store_true")
    ap.add_argument("--phase6-results", help="many_knobs_eval/results: calibrated_gain from the phase 6 PCI cutoff "
                                             "(sa3_tada_run calibrate rule) instead of the screening estimate")
    args = ap.parse_args()

    import torch
    from acestep.steering.packs import SteeringPack, load_pack, save_pack

    cap = Path(args.cap)
    dirs = Path(args.dirs) if args.dirs else cap / "dirs"
    screen = Path(args.screen) if args.screen else cap / "screen"
    out = Path(args.out) if args.out else cap / "packs" / "sa3" / "medium"
    with open(dirs / "index.csv", newline="", encoding="utf-8") as f:
        index = {r["name"]: r for r in csv.DictReader(f)}
    names = pick(args, screen)
    if not names:
        raise SystemExit("no concepts to pack (no accepted.csv / passing screen json, no --concepts)")
    today = _dt.date.today().isoformat()
    pci = {}
    for name in names:
        row = index.get(name)
        if row is None or row.get("status", "ok") != "ok":
            raise SystemExit(f"{name}: not a usable row of {dirs / 'index.csv'}")
        z = np.load(row["path"])
        unit, norm, block = stepmean(z)
        mag = norm * KNOB_UNIT
        sj = screen / f"{name}.json"
        scr = json.loads(sj.read_text()) if sj.exists() else None
        if scr is None and not args.allow_unscreened:
            raise SystemExit(f"{name}: no {sj} (pass --allow-unscreened to pack anyway)")
        screening, gain = None, None
        if scr is not None:
            if int(scr["block"]) != block:
                raise SystemExit(f"{name}: screened at b{scr['block']}, npz best block b{block}")
            ap_pos = scr["alpha_at_cut"]["pos"]
            ap_neg = scr["alpha_at_cut"].get("neg")
            gain = {"pos": ap_pos / mag, "neg": (ap_neg if ap_neg is not None else ap_pos) / mag,
                    "pos_reached": scr["cut_reached"]["pos"],
                    "neg_reached": scr["cut_reached"].get("neg"),
                    "neg_source": "screened" if ap_neg is not None else "mirrored from pos (screened --signs pos)",
                    "unit": "knob value at which mean LPAPS vs the unsteered render reaches lpaps_cut"}
            screening = {
                "tool": "scripts/steering_bench/screen_knobs.py", "date": scr.get("date"),
                "lpaps_cut": scr["lpaps_cut"], "n_prompts": scr["n_prompts"], "prompts": scr["prompts"],
                "k": scr["k"], "alphas": scr["alphas"],
                "lpaps": [round(e["lpaps_mean"], 4) for e in scr["per_alpha"]],
                "own": scr["own"], "second": scr["second"], "slope_per_lpaps": scr["slope_per_lpaps"],
                "max_cos_accepted": scr["max_cos_accepted"], "pass": scr["pass"], "flags": scr["flags"],
                "dry_run": scr.get("dry_run", False)}
        if args.phase6_results:
            gain = phase6_gain(Path(args.phase6_results) / name, name, mag)
        label = row.get("label") or name.replace("_", " ").title()
        blurb = row.get("blurb") or f"positive raises {row['label_col']}" + (
            " (sign -1: lowers it)" if int(float(row["sign"])) < 0 else "")
        prov = {
            "tool": "scripts/steering_bench/make_packs.py", "method": "mass_mean_quartile",
            "method_source": "top vs bottom quartile difference of means of a label column over a "
                             "self-rendered, self-labelled capture (build_directions.py --cap)",
            "date": today, "label_col": row["label_col"], "sign": int(float(row["sign"])),
            "category": row.get("category", ""), "second_cols": row.get("second_cols", ""),
            "population": row.get("population", ""), "class_rule": row.get("class_rule", ""),
            "quantile": float(z["quantile"]), "n_pos": int(z["n_pos"]), "n_neg": int(z["n_neg"]),
            "capture": str(z["capture"]), "directions": str(row["path"]), "block_choice":
            "best block by cross-fitted step-mean effect (build_directions.py)",
            "effect_best": float(row["effect_best"]), "proj_std_at_block": float(z["std"][:, block].mean()),
            "step_reduction": "mean of per-step raw differences over all steps", "knob_unit": KNOB_UNIT,
            "screening": screening, "calibrated_gain": gain,
            "caveat": "estimated at 10 s ARC sam.generate, audio tokens only; production applies at 54 s "
                      "pingpong StreamPipeline to every token incl. 64 memory tokens",
        }
        pack = SteeringPack(family="sa3", checkpoint="medium", block=block, hidden_size=int(unit.shape[0]),
                            name=name, vector=torch.from_numpy(unit.astype(np.float32)), label=label,
                            blurb=blurb, hook=str(z["hook"]) if "hook" in z else "post_block_residual",
                            method="caa_diff_means", norm=norm, magnitude=mag,
                            policy={"kind": "range", "start": 0.0, "end": 1.0}, provenance=prov)
        path = save_pack(pack, out / f"{name}.safetensors")
        back = load_pack(path)
        cos = float(torch.dot(back.vector, pack.vector) / back.vector.norm() / pack.vector.norm())
        ok = (back.block == block and abs(back.magnitude - mag) < 1e-6 and cos > 0.99999
              and back.hook == pack.hook and back.provenance["label_col"] == row["label_col"])
        if not ok:
            raise SystemExit(f"{path}: re-read mismatch (block {back.block}, cos {cos:.6f})")
        g = f" gain +{gain['pos']:.2f} / -{gain['neg']:.2f}" if gain else ""
        print(f"[{name}] b{block} norm {norm:.3f} magnitude {mag:.3f}{g} -> {path} (re-read OK)", flush=True)
        pt = row.get("pos_text") or row.get("pos_anchor")
        nt = row.get("neg_text") or row.get("neg_anchor")
        if pt and nt:
            pci[name] = [pt, nt]
    if args.pci_descriptors:
        (out / "pci_descriptors.json").write_text(json.dumps(pci, indent=1))
        print(f"wrote {out / 'pci_descriptors.json'} ({len(pci)} of {len(names)} have pos/neg text)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
