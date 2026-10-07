"""Ship pass: every shipped SA3 steering knob measured under one protocol, at the gain that ships.

Runbook: notes/steering_pr/ship_uniform_rigor_runbook.md (DEMON). Works on the many-knobs v2 eval tree
($E2, default /dev/shm/steerbench/out/many_knobs_v2_eval); everything new goes under $E2/ship.

Vectors (``vectors.json``):
  new      the 63 new_v2 knobs of the 2026-10-06 shipped set (bench packs, protocol seeds 0,1,2 done)
  v2gate   the other v2 vectors that compete with a v1 vector for the same concept (regression gate)
  v1       the 46 v1 vectors (unchanged bytes, ``v1_<name>``), measured with the v2 counterpart's descriptors
  legacy   the 5 legacy vectors (bench packs ``legacy_*``)

Subcommands:
  init                 vectors.json, packs/ (v1 copies + bench symlinks + pci_descriptors.json), cols.json
  rules                rules.json + rules.md (decision rules, frozen before any render)
  gains                gains.json: interpolated shipped gain per vector-sign (log-linear in alpha between the
                       two probes that bracket the PCI-all cutoff; median over seeds 0,1,2)
  skiplist             skip_c.json: new knobs whose 07:45Z P4 / 08:25Z cross-effect gains are within 5% of the
                       new shipped gain on both signs (their C parts are reused)
  fixedplan VID OFF    ranges json for the fixed-gain pass; prints "<pack> <lo> <hi>" per pack
  sfx VID OFF          SFX prompt pass: app_sfx (12 prompts) at seed 2115+OFF, both signs, CLAP + MuQ
  capp VID             applicability (36 prompts) + 30 s hold (6 prompts) at the shipped gain
  cfx VID|base         cross-effect on screen_music at the shipped gain, scored on the common label universe
  combos               d/combos.json: 200 pairs + 100 triples over the final set, random signs, seed 0
  stack K              D chunk K (10 combos x 4 rules: none, inv_sqrt, inv_n, norm_budget)
  loop                 CPU loop: appends P1/P2/P3/P4 items as their inputs land, retries, hourly accounting
  final                bar, gate, packs (one metadata shape), results_ship.md
"""
from __future__ import annotations

import csv
import datetime as dt
import glob
import json
import math
import os
import random
import re
import shutil
import statistics
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

E2 = Path(os.environ.get("E2", "/dev/shm/steerbench/out/many_knobs_v2_eval"))
SH = E2 / "ship"
PKB = E2 / "packs_bench"
PKS = SH / "packs"
V1 = Path(os.environ.get("V1", "/dev/shm/steerbench/out/many_knobs_eval"))
B = os.environ.get("B", "/root/DEMON-steerbench")
EV, DM = os.environ.get("EV", "/root/evalenv/bin/python"), os.environ.get("DM", "/root/demonenv/bin/python")
EVAL_SEED = 2115
SEEDS = (0, 1, 2)
FRAC_MIN = 0.6
SKIP_TOL = 0.05
LEGACY = {"legacy_bright": "bright", "legacy_density": "dense_arrangement", "legacy_percussive": "percussive",
          "legacy_rough": "rough", "legacy_warm": "warm"}
APP_SETS = {"app_music": "music", "app_sfx": "sfx", "app_abstract": "abstract"}
RULES = ("none", "inv_sqrt", "inv_n", "norm_budget")
SIDES = ("pos", "neg")


def jl(p):
    try:
        return json.loads(Path(p).read_text())
    except (OSError, ValueError):
        return None


def jw(p, obj):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=float))
    os.replace(tmp, p)


def now():
    return dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")


def catalogue():
    cat = {r["name"]: r for r in csv.DictReader(open(E2 / "concept_catalogue_v2_sfx.csv", encoding="utf-8"))}
    for n, r in cat.items():  # _r2 rows inherit the v1 twin's anchors (as final.fill_anchors)
        if not r.get("pos_anchor"):
            tw = cat.get(n[:-3], {}) if n.endswith("_r2") else {}
            r["pos_anchor"] = tw.get("pos_anchor") or (n[:-3] if n.endswith("_r2") else n).replace("_", " ")
            r["neg_anchor"] = r.get("neg_anchor") or tw.get("neg_anchor", "")
    return cat


def vectors():
    return jl(SH / "vectors.json")


def vec(vid):
    return vectors()[vid]


def magnitude(pack):
    from safetensors import safe_open
    with safe_open(str(PKS / f"{pack}.safetensors"), framework="pt") as f:
        return float(json.loads(f.metadata()["steering_pack"])["magnitude"])


# ---------------------------------------------------------------- init / rules

def cmd_init():
    SH.mkdir(parents=True, exist_ok=True)
    PKS.mkdir(parents=True, exist_ok=True)
    cat = catalogue()
    man = list(csv.DictReader(open(E2 / "packs_final" / "manifest.csv", encoding="utf-8")))
    kt = list(csv.DictReader(open(E2 / "knob_table.csv", encoding="utf-8")))
    gate = list(csv.DictReader(open(E2 / "gate.csv", encoding="utf-8")))
    packs_of = defaultdict(dict)
    for r in kt:
        packs_of[r["knob"]][r["sign"]] = (r["pack"], r["variant"])
    pci = json.loads((PKB / "pci_descriptors.json").read_text())
    out = {}

    def bench(vid, group, concept, final):
        p = packs_of[vid]
        pos, neg = p["+"][0], p["-"][0]
        out[vid] = {"vid": vid, "group": group, "concept": concept, "final_name": final, "variant": p["+"][1],
                    "pos_pack": pos, "neg_pack": neg, "c": pos != neg}
    for r in man:
        if r["status"] == "shipped" and not r["kept_from"]:
            bench(r["knob"], "new", r["knob"], r["name"])
    for r in gate:
        if r.get("legacy"):
            continue
        k, k1 = r["knob"], r["v1"]
        if k in packs_of and k not in out:
            bench(k, "v2gate", k, k[:-3] if k.endswith("_r2") else k)
        vid = f"v1_{k1}"
        src = V1 / "packs_screening_copy" / f"{k1}.safetensors"
        shutil.copy2(src, PKS / f"{vid}.safetensors")
        twin = k if k in packs_of else None
        if twin:
            desc = pci[packs_of[twin]["+"][0]]
            pci_src = packs_of[twin]["+"][0]
        else:
            c = cat[k1]
            desc, pci_src = [c["pos_anchor"], c["neg_anchor"]], None
        pci[vid] = desc
        out[vid] = {"vid": vid, "group": "v1", "concept": k if twin else k1, "final_name": k1, "variant": "a",
                    "pos_pack": vid, "neg_pack": vid, "c": False, "v1_name": k1, "twin": twin, "pci_src": pci_src}
    for leg, cp in LEGACY.items():
        out[leg] = {"vid": leg, "group": "legacy", "concept": cp, "final_name": cp, "variant": "legacy",
                    "pos_pack": leg, "neg_pack": leg, "c": False}
    for v in out.values():
        for p in {v["pos_pack"], v["neg_pack"]}:
            dst = PKS / f"{p}.safetensors"
            if not dst.exists():
                os.symlink(PKB / f"{p}.safetensors", dst)
        c = cat.get(v["concept"]) or cat.get(v["concept"][:-3] if v["concept"].endswith("_r2") else "") or {}
        v["category"] = c.get("category", "")
        v["primary_label"] = c.get("primary_label") or c.get("label_col") or ""
        v["pos_anchor"], v["neg_anchor"] = pci[v["pos_pack"]]
        v["label"] = c.get("label", "")
        v["blurb"] = c.get("blurb", "")
    (PKS / "pci_descriptors.json").write_text(json.dumps(pci, indent=1))
    jw(SH / "vectors.json", out)
    cols = sorted({v["primary_label"] for v in out.values()} - {""})
    jw(SH / "cols.json", cols)
    print(Counter(v["group"] for v in out.values()), len(cols), "label cols")


RULES_TEXT = """# Ship pass decision rules (frozen {ts}, before any render)

- Shipped gain per vector-sign: for each seed 0,1,2, the knob value at which the probe LPAPS curve (20 holdout prompts)
  reaches that seed's PCI-all cutoff (max PCI LPAPS on that sign, 50 test prompts), interpolated log-linearly in alpha
  between the two probes that bracket it (linear from alpha 0 when the first probe already reaches it; the largest probe,
  reached = false, when none does). Shipped gain = median over seeds 0,1,2. reached[] per seed.
- Measurement at the shipped gain (fixed-gain pass): 50 TADA test prompts, seed 2115+off for off in 0,1,2, alpha 0 and
  the shipped alpha of each sign; CLAP and MuQ intended-sign fraction (vs alpha 0) and LPAPS per seed.
- Quality bar (absolute, every origin, no exemptions): on at least one sign, the median over seeds 0,1,2 of the CLAP
  fraction >= 0.6 AND of the MuQ fraction >= 0.6, AND on that same sign the cutoff reached on >= 2 of 3 seeds.
- Regression gate (only where a v1 and a v2 vector exist for the same concept, both measured here): v2 wins iff on both
  signs both median fractions >= v1's - 0.05, on both signs median(LPAPS/cutoff) <= 1.1 x v1's, and the number of signs
  reached (>= 2 of 3 seeds) >= v1's. The bar is applied first: a vector failing the bar is never a candidate; if only one
  vector of a concept passes, it ships (gate "decided by the bar"); if none passes, the concept ships nothing (this is
  the sfx_laden case; "never regress" yields to the bar). No counterpart = "not compared", never "won".
- Hold drift (signed, 0-10 s vs 20-30 s) and cross-effect are reported, not gating.
- applies_to: the measured categories (music / sfx / abstract) whose CLAP and MuQ fractions, averaged over signs, are
  >= 0.6 (sfx judged on CLAP alone, as last night); an empty measurement stays empty (no catalogue default).
- SFX prompt pass: CLAP and MuQ fractions on the 12 SFX applicability prompts x 3 seeds, recorded, not gating.
- c packs: header neg gain is stored in the dn vector's own magnitude units; the runtime applies it to vectors_neg.
- Skip list for C (new knobs only): a knob's applicability/hold is reused when its 07:45Z P4 gain is within 5% of the
  new shipped gain on both signs; its cross-effect is reused when its 08:25Z cross-effect gain is within 5% on both signs.
"""


def cmd_rules():
    ts = now()
    rules = {"frozen": ts, "frac_min": FRAC_MIN, "bar_reach_min_seeds": 2, "seeds": list(SEEDS),
             "gate": {"frac_margin": 0.05, "lpaps_ratio_factor": 1.1, "reach": ">= v1"},
             "gain": "median over seeds of the log-linear interpolated alpha at the PCI-all cutoff / magnitude",
             "skip_tol": SKIP_TOL, "c_neg_unit": "dn vector's own magnitude"}
    if (SH / "rules.json").exists():
        print("rules already frozen at", jl(SH / "rules.json")["frozen"])
        return
    jw(SH / "rules.json", rules)
    (SH / "rules.md").write_text(RULES_TEXT.format(ts=ts))
    print("frozen", ts)


# ---------------------------------------------------------------- gains

def lp_curve(f):
    if not f.exists():
        return None
    with open(f, newline="") as fh:
        return {float(r["alpha"]): float(r["mean"]) for r in csv.DictReader(fh)}


def res_dir(pack, off):
    for root in (SH / f"results_s{off}", E2 / f"results_s{off}"):
        d = root / pack
        if (d / "done").exists():
            return d
    return None


def interp_side(res, pack, sign):
    """(alpha_interp, alpha_first_hit, cutoff, reached) for one sign of one pack at one seed."""
    pr = res / "eval" / f"pci_all_{pack}" / "protocol_results"
    pci = lp_curve(pr / "lpaps.csv") or lp_curve(pr / "lpaps_endpoints.csv")
    probe = lp_curve(res / "calib" / f"pack_pack_{pack}" / "protocol_results" / "lpaps.csv")
    if not pci or not probe:
        return None
    cut = max(v for a, v in pci.items() if sign * a > 0)
    v0 = probe.get(0.0, 0.0)
    pts = sorted((abs(a), v) for a, v in probe.items() if sign * a > 0)
    for i, (a, v) in enumerate(pts):
        if v >= cut:
            if i == 0:
                ai = a * (cut - v0) / (v - v0) if v > v0 else a
            else:
                a0, w0 = pts[i - 1]
                t = (cut - w0) / (v - w0) if v > w0 else 1.0
                ai = math.exp(math.log(a0) + t * (math.log(a) - math.log(a0)))
            return {"alpha": ai, "alpha_hit": a, "cutoff": cut, "reached": True}
    return {"alpha": pts[-1][0], "alpha_hit": pts[-1][0], "cutoff": cut, "reached": False}


def vector_gain(v, seeds=SEEDS):
    """Per side: per-seed interpolated gains and their median, or None when a seed is missing."""
    out = {}
    for side in SIDES:
        pack = v["pos_pack"] if side == "pos" else v["neg_pack"]
        sign = 1 if (side == "pos" or v["c"]) else -1
        mag = magnitude(pack)
        per = {}
        for off in seeds:
            d = res_dir(pack, off)
            if d is None:
                return None
            r = interp_side(d, pack, sign)
            if r is None:
                return None
            per[off] = {**r, "gain": r["alpha"] / mag, "gain_first_hit": r["alpha_hit"] / mag}
        g = [per[o]["gain"] for o in seeds]
        out[side] = {"pack": pack, "magnitude": mag, "median": statistics.median(g), "min": min(g), "max": max(g),
                     "seeds": list(seeds), "reached": [per[o]["reached"] for o in seeds],
                     "cutoff": [per[o]["cutoff"] for o in seeds], "per_seed": per,
                     "first_hit_median": statistics.median(per[o]["gain_first_hit"] for o in seeds),
                     "steer_sign": 1 if (side == "pos" or v["c"]) else -1}
    return out


def cmd_gains(quiet=False):
    vs = vectors()
    gp = SH / "gains.json"
    gains = jl(gp) or {}
    new = []
    for vid, v in vs.items():
        if vid in gains:
            continue
        g = vector_gain(v)
        if g:
            gains[vid] = g
            new.append(vid)
    if new:
        jw(gp, gains)
    if not quiet:
        print(f"gains: {len(gains)}/{len(vs)} vectors ({len(new)} new)")
    return new


def steer_of(vid, side, gains=None):
    """[pack path, signed gain] for set_tools render."""
    g = (gains or jl(SH / "gains.json"))[vid][side]
    return [str(PKS / f"{g['pack']}.safetensors"), g["steer_sign"] * g["median"]]


# ---------------------------------------------------------------- skip list

def old_gains():
    """Gains that last night's P4 knob items (p4/knob_<base>_r.json) and cross-effect items (cfx*_r.json) used,
    per pack path and signed gain -> knob base and side; the latest file wins."""
    p4, cfx = {}, defaultdict(dict)
    for f in sorted(glob.glob(str(E2 / "slot_g*" / "p4" / "knob_*_r.json")), key=os.path.getmtime):
        base = Path(f).name[5:-7]
        jobs = jl(f) or []
        d = {}
        for j in jobs:
            if j["set"] == "app_music":
                (path, g), = j["steer"]
                d["pos" if "pos" not in d else "neg"] = (Path(path).stem, g)
        p4[base] = d
    for f in sorted(glob.glob(str(E2 / "slot_g*" / "p4" / "cfx*_r.json")), key=os.path.getmtime):
        if "base" in f:
            continue
        for j in jl(f) or []:
            (path, g), = j["steer"]
            cfx[Path(path).stem][1 if g > 0 else -1] = g
    return p4, cfx


def cmd_skiplist():
    vs, gains = vectors(), jl(SH / "gains.json")
    p4, cfx = old_gains()
    out = {}
    for vid, v in vs.items():
        if v["group"] != "new":
            continue
        g = gains[vid]
        o = p4.get(vid, {})
        rel = {}
        ok_app = bool(o)
        for side in SIDES:
            if side not in o:
                ok_app = False
                continue
            pk, og = o[side]
            if pk != g[side]["pack"]:
                ok_app = False
                continue
            rel[f"app_{side}"] = abs(abs(og) - g[side]["median"]) / g[side]["median"]
            ok_app &= rel[f"app_{side}"] <= SKIP_TOL
        ok_cfx = True
        for side in SIDES:
            pk = g[side]["pack"]
            og = cfx.get(pk, {}).get(g[side]["steer_sign"])
            if og is None:
                ok_cfx = False
                continue
            rel[f"cfx_{side}"] = abs(abs(og) - g[side]["median"]) / g[side]["median"]
            ok_cfx &= rel[f"cfx_{side}"] <= SKIP_TOL
        out[vid] = {"skip_app_hold": ok_app, "skip_crossfx": ok_cfx, "rel_diff": rel}
    jw(SH / "skip_c.json", out)
    print("skip app/hold:", sum(x["skip_app_hold"] for x in out.values()), "skip crossfx:",
          sum(x["skip_crossfx"] for x in out.values()), "of", len(out))


def reuse_skipped():
    """Copy last night's P4 app/hold and cross-effect results of skipped new knobs into ship/c (marked reused)."""
    sk = jl(SH / "skip_c.json") or {}
    for vid, s in sk.items():
        if s["skip_app_hold"]:
            src, dst = E2 / "p4" / "app" / vid, SH / "c" / "app" / vid
            if src.is_dir() and not dst.exists():
                shutil.copytree(src, dst)
            for n in (f"{vid}.json", f"{vid}_base.json"):
                if (E2 / "p4" / "hold" / n).exists():
                    (SH / "c" / "hold").mkdir(parents=True, exist_ok=True)
                    shutil.copy2(E2 / "p4" / "hold" / n, SH / "c" / "hold" / n)
            (dst / "REUSED").write_text("p4 07:45Z, gain within 5%\n")
        if s["skip_crossfx"]:
            for sg in ("+1", "-1"):
                f = E2 / "p4" / "crossfx" / f"{vid}_{sg}.json"
                if f.exists():
                    (SH / "c" / "crossfx").mkdir(parents=True, exist_ok=True)
                    shutil.copy2(f, SH / "c" / "crossfx" / f"{vid}_{sg}.json")


# ---------------------------------------------------------------- fixed-gain pass

def cmd_fixedplan(vid, off):
    v, g = vec(vid), jl(SH / "gains.json")[vid]
    out = Path(os.environ["OUT"])
    ranges = {"pack": {}}
    lines = []
    if v["c"]:
        for side in SIDES:
            p = g[side]["pack"]
            a = round(g[side]["median"] * g[side]["magnitude"], 4)
            ranges["pack"][p] = {"pack": [0.0, a]}
            lines.append(f"{p} 0.0 {a}")
    else:
        p = v["pos_pack"]
        lo = -round(g["neg"]["median"] * g["neg"]["magnitude"], 4)
        hi = round(g["pos"]["median"] * g["pos"]["magnitude"], 4)
        ranges["pack"][p] = {"pack": [lo, hi]}
        lines.append(f"{p} {lo} {hi}")
    jw(out / "ship_ranges.json", ranges)
    print("\n".join(lines))


def per_prompt(f):
    import ast
    import pandas as pd
    d = pd.read_csv(f)
    return {float(a): np.array(ast.literal_eval(s), float) for a, s in zip(d["alpha"], d["scores"])}


def fixed_metrics(vid, off, gains=None):
    """Per side at the shipped gain, seed off: CLAP / MuQ intended-sign fractions and effects, LPAPS, LPAPS/cutoff."""
    v = vec(vid) if gains is None else vectors()[vid]
    gains = gains or jl(SH / "gains.json")
    d = SH / f"fixed_s{off}" / vid
    if not (d / "done").exists():
        return None
    out = {}
    for side in SIDES:
        g = gains[vid][side]
        pack, sign = g["pack"], g["steer_sign"]
        pr = d / pack / "protocol_results"
        clap, muq, lp = per_prompt(pr / "clap.csv"), per_prompt(pr / "muqt.csv"), lp_curve(pr / "lpaps.csv")
        al = [a for a in clap if a * sign > 0]
        a = max(al) if sign > 0 else min(al)
        dc, dm = (clap[a] - clap[0.0]) * sign, (muq[a] - muq[0.0]) * sign
        cut = g["cutoff"][off] if off < len(g["cutoff"]) else statistics.median(g["cutoff"])
        out[side] = {"alpha": a, "frac_clap": float(np.mean(dc > 0)), "frac_muq": float(np.mean(dm > 0)),
                     "effect_clap": float(dc.mean()), "effect_muq": float(dm.mean()), "lpaps": lp[a],
                     "lpaps_over_cutoff": lp[a] / cut}
    return out


# ---------------------------------------------------------------- set_tools jobs (SFX, C, D)

def run_jobs(rjobs, sjobs, tag):
    d = Path(os.environ.get("OUT", str(SH / "slot_x"))) / "ship"
    d.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, PYTHONPATH=B)
    if rjobs:
        (d / f"{tag}_r.json").write_text(json.dumps(rjobs))
        subprocess.run([DM, "scripts/steering_bench/set_tools.py", "render", "--jobs", str(d / f"{tag}_r.json")],
                       cwd=B, env=env, check=True)
    if sjobs:
        (d / f"{tag}_s.json").write_text(json.dumps(sjobs))
        subprocess.run([EV, "scripts/steering_bench/set_tools.py", "score", "--jobs", str(d / f"{tag}_s.json")],
                       cwd=B, env=env, check=True)


def slot():
    return Path(os.environ.get("OUT", str(SH / "slot_x"))) / "ship"


def baseline(st, seed, duration=None):
    tag = f"{st}_30s" if duration else st
    return E2 / "baselines" / f"{tag}_s{seed}.npz"


def cmd_sfx(vid, off):
    v, seed = vec(vid), EVAL_SEED + int(off)
    sets = str(E2 / "eval_sets.json")
    anchors = [a for a in (v["pos_anchor"], v["neg_anchor"]) if a]
    base = baseline("app_sfx", seed)
    rj, sj = [], []
    if not base.exists():  # first SFX item of this seed renders the unsteered reference
        tmp = slot() / f"base_app_sfx_s{seed}.npz"
        run_jobs([{"sets": sets, "set": "app_sfx", "seed": seed, "out": str(tmp)}], [], f"sfxbase{seed}")
        if not base.exists():
            shutil.copy2(tmp, str(base) + ".tmp")
            os.replace(str(base) + ".tmp", base)
    for s, side in ((1, "pos"), (-1, "neg")):
        out = slot() / f"sfx_{s}.npz"
        rj.append({"sets": sets, "set": "app_sfx", "seed": seed, "out": str(out), "steer": [steer_of(vid, side)]})
        sj.append({"npz": str(out), "ref": str(base), "anchors": anchors,
                   "json": str(SH / "sfx" / vid / f"s{off}_{s:+d}.json"), "meta": {"vid": vid, "sign": s, "seed": seed}})
    sj.append({"npz": str(base), "anchors": anchors, "lpaps": False, "json": str(SH / "sfx" / vid / f"s{off}_base.json"),
               "meta": {"vid": vid, "sign": 0, "seed": seed}})
    run_jobs(rj, sj, f"sfx_{vid}_{off}")


def cmd_capp(vid):
    v = vec(vid)
    sets = str(E2 / "eval_sets.json")
    anchors = [a for a in (v["pos_anchor"], v["neg_anchor"]) if a]
    prim = [v["primary_label"]] if v["primary_label"] else []
    rj, sj = [], []
    for s, side in ((1, "pos"), (-1, "neg")):
        for st in APP_SETS:
            out = slot() / f"app_{st}_{s}.npz"
            rj.append({"sets": sets, "set": st, "seed": EVAL_SEED, "out": str(out), "steer": [steer_of(vid, side)]})
            sj.append({"npz": str(out), "ref": str(baseline(st, EVAL_SEED)), "anchors": anchors, "cols": prim,
                       "json": str(SH / "c" / "app" / vid / f"{st}_{s:+d}.json"),
                       "meta": {"vid": vid, "sign": s, "set": st}})
    for st in APP_SETS:
        sj.append({"npz": str(baseline(st, EVAL_SEED)), "anchors": anchors, "cols": prim, "lpaps": False,
                   "json": str(SH / "c" / "app" / vid / f"{st}_base.json"), "meta": {"vid": vid, "sign": 0, "set": st}})
    h0 = baseline("hold6", EVAL_SEED, 30)
    sj.append({"npz": str(h0), "anchors": anchors, "cols": prim, "windows": ["0:10", "20:30"], "lpaps": False,
               "json": str(SH / "c" / "hold" / f"{vid}_base.json"), "meta": {"vid": vid, "sign": 0}})
    out = slot() / "hold.npz"
    rj.append({"sets": sets, "set": "hold6", "seed": EVAL_SEED, "duration": 30.0, "out": str(out),
               "steer": [steer_of(vid, "pos")]})
    sj.append({"npz": str(out), "ref": str(h0), "anchors": anchors, "cols": prim, "windows": ["0:10", "20:30"],
               "lpaps": False, "json": str(SH / "c" / "hold" / f"{vid}.json"), "meta": {"vid": vid, "sign": 1}})
    run_jobs(rj, sj, f"capp_{vid}")


def cmd_cfx(vid):
    cols = jl(SH / "cols.json")
    ref = str(baseline("screen_music", EVAL_SEED))
    d = SH / "c" / "crossfx"
    if vid == "base":
        run_jobs([], [{"npz": ref, "cols": cols, "lpaps": False, "json": str(d / "base.json"),
                       "meta": {"vid": "__base__", "sign": 0}}], "cfx_base")
        return
    sets = str(E2 / "eval_sets.json")
    rj, sj = [], []
    for s, side in ((1, "pos"), (-1, "neg")):
        out = slot() / f"cfx_{s}.npz"
        rj.append({"sets": sets, "set": "screen_music", "seed": EVAL_SEED, "out": str(out),
                   "steer": [steer_of(vid, side)]})
        sj.append({"npz": str(out), "ref": ref, "cols": cols, "json": str(d / f"{vid}_{s:+d}.json"),
                   "meta": {"vid": vid, "sign": s}})
    run_jobs(rj, sj, f"cfx_{vid}")


def cmd_combos():
    fs = jl(SH / "final_set.json")
    names = sorted(fs["shipped_vids"])
    rng = random.Random(0)
    combos = [[[x, rng.choice((1, -1))] for x in rng.sample(names, 2)] for _ in range(200)]
    combos += [[[x, rng.choice((1, -1))] for x in rng.sample(names, 3)] for _ in range(100)]
    jw(SH / "d" / "combos.json", combos)
    print(len(combos), "combos over", len(names))


def norm_budget_scale(members):
    """Shared norm budget: the combined perturbation (all steps and blocks) is held to the Frobenius norm of the
    largest single member: scale = max_i ||a_i V_i|| / ||sum_i a_i V_i||, capped at 1."""
    sys.path[:0] = [B, B + "/scripts/tada"]
    import sa3_tada_run as R
    import torch
    tot, single = {}, []
    for path, gain in members:
        pv = R._pack_vectors(path, 8)
        alpha = float(gain) * float(R._PACK["magnitude"])
        n2 = 0.0
        for s, dd in pv.items():
            for b, v in dd.items():
                x = v.float() * alpha
                tot[(s, b)] = tot.get((s, b), 0) + x
                n2 += float((x * x).sum())
        single.append(math.sqrt(n2))
    nt = math.sqrt(sum(float((x * x).sum()) for x in tot.values()))
    return min(1.0, max(single) / nt) if nt > 0 else 1.0


def cmd_stack(k):
    k = int(k)
    gains = jl(SH / "gains.json")
    combos = jl(SH / "d" / "combos.json")[k * 10:(k + 1) * 10]
    sets = str(E2 / "eval_sets.json")
    ref = str(baseline("screen_music", EVAL_SEED))
    rj, sj = [], []
    for i, cb in enumerate(combos):
        members = [steer_of(n, "pos" if s > 0 else "neg", gains) for n, s in cb]
        cut = min(statistics.median(gains[n]["pos" if s > 0 else "neg"]["cutoff"]) for n, s in cb)
        for rule in RULES:
            sc = {"none": 1.0, "inv_sqrt": len(cb) ** -0.5, "inv_n": 1.0 / len(cb)}.get(rule)
            if rule == "norm_budget":
                sc = norm_budget_scale(members)
            out = slot() / f"stack_{i}_{rule}.npz"
            rj.append({"sets": sets, "set": "screen_music", "seed": EVAL_SEED, "out": str(out), "scale": sc,
                       "steer": members})
            sj.append({"npz": str(out), "ref": ref, "json": str(SH / "d" / "stack" / f"c{k * 10 + i:03d}_{rule}.json"),
                       "meta": {"combo": cb, "rule": rule, "scale": sc, "cutoff_min": cut}})
    run_jobs(rj, sj, f"stack{k}")


# ---------------------------------------------------------------- queue

def qline(prio, iid, cmd):
    return f"{prio} {iid} {cmd}"


ENV = f"source {E2}/env2.sh; export E2={E2}; cd {B}"
TOOLS = f"$DM scripts/steering_bench/ship_tools.py"
ITEM = f"bash {B}/scripts/steering_bench/ship_item.sh"


def item_proto(pack, off, pci_src=None):
    return f"{ITEM} proto {pack} {off} {pci_src or '-'}"


def item_fixed(vid, off):
    return f"{ITEM} fixed {vid} {off}"


def item_py(*a):
    return f"{ENV}; {TOOLS} " + " ".join(str(x) for x in a)


def queued_ids():
    q = SH / "queue.txt"
    return {l.split()[1] for l in open(q)} if q.exists() else set()


def append(lines):
    q = queued_ids()
    lines = [l for l in lines if l.split()[1] not in q]
    if lines:
        with open(SH / "queue.txt", "a") as f:
            for l in lines:
                f.write(l + "\n")
    return lines


def post_gain_items(vid, prio_b):
    """B (3 seeds), SFX (sound_effect vectors), C parts, and the P4 seed-3 filler for a vector with a gain."""
    v = vec(vid)
    sk = (jl(SH / "skip_c.json") or {}).get(vid, {})
    out = [qline(prio_b, f"B_{vid}_s{o}", item_fixed(vid, o)) for o in SEEDS]
    if v["category"] == "sound_effect":
        out += [qline(prio_b, f"S_{vid}_s{o}", item_py("sfx", vid, o)) for o in SEEDS]
    if not sk.get("skip_app_hold"):
        out.append(qline(2, f"C_{vid}", item_py("capp", vid)))
    if not sk.get("skip_crossfx"):
        out.append(qline(2, f"X_{vid}", item_py("cfx", vid)))
    out.append(qline(4, f"B_{vid}_s3", item_fixed(vid, 3)))
    return out


def cmd_enqueue0():
    vs = vectors()
    a_items = []
    order = ([v for v in vs.values() if v["group"] == "v1" and v["v1_name"] in KEPT_V1] +
             [v for v in vs.values() if v["group"] == "v1" and v["v1_name"] in DISPLACED] +
             [v for v in vs.values() if v["group"] == "v1" and v["v1_name"] == "sfx_laden"] +
             [v for v in vs.values() if v["group"] == "v1" and v["v1_name"] not in KEPT_V1 | DISPLACED | {"sfx_laden"}])
    for v in order:
        for o in SEEDS:
            a_items.append(qline(0, f"A_{v['vid']}_s{o}", item_proto(v["vid"], o, v.get("pci_src"))))
    # seeds missing on bench vectors (legacy warm s2, v2gate vectors without seed 2)
    for v in vs.values():
        if v["group"] in ("legacy", "v2gate", "new"):
            for p in {v["pos_pack"], v["neg_pack"]}:
                for o in SEEDS:
                    if res_dir(p, o) is None:
                        a_items.append(qline(0, f"A_{p}_s{o}", item_proto(p, o)))
    b_items = []
    for vid, v in vs.items():
        if v["group"] in ("new", "v2gate", "legacy") and vid in (jl(SH / "gains.json") or {}):
            b_items += [qline(0, f"B_{vid}_s{o}", item_fixed(vid, o)) for o in SEEDS]
    s_items = []
    for vid, v in vs.items():
        if v["group"] in ("new", "v2gate", "legacy") and v["category"] == "sound_effect" and vid in (jl(SH / "gains.json") or {}):
            s_items += [qline(0, f"S_{vid}_s{o}", item_py("sfx", vid, o)) for o in SEEDS]
    lines = [qline(0, f"base_app_sfx_s{EVAL_SEED + o}",
                   f"{ENV}; $DM scripts/steering_bench/set_tools.py render --sets {E2}/eval_sets.json --set app_sfx "
                   f"--seed {EVAL_SEED + o} --out {E2}/baselines/app_sfx_s{EVAL_SEED + o}.npz") for o in (1, 2)]
    # round-robin A and B so the P1 pipeline (which needs A) fills early
    ia, ib = iter(a_items), iter(b_items + s_items)
    while True:
        a, b = next(ia, None), next(ib, None)
        if a is None and b is None:
            break
        lines += [x for x in (a, b) if x]
    lines.append(qline(2, "X_base", item_py("cfx", "base")))
    for vid, v in vs.items():
        if v["group"] in ("new", "v2gate", "legacy") and vid in (jl(SH / "gains.json") or {}):
            lines += [l for l in post_gain_items(vid, 0) if l.split()[1].startswith(("C_", "X_"))
                      or l.split()[1].endswith("_s3")]
    added = append(lines)
    print("enqueued", Counter(l.split()[1].split("_")[0] for l in added))


KEPT_V1 = set()
DISPLACED = {"aggressive", "black_metal", "jet", "polyrhythmic", "rain_on_surface", "ukulele"}


def _load_kept():
    man = E2 / "packs_final" / "manifest.csv"
    if man.exists():
        KEPT_V1.update(r["name"] for r in csv.DictReader(open(man, encoding="utf-8"))
                       if r["status"] == "shipped" and r["kept_from"] == "v1")


_load_kept()


def acct(since=None):
    T = lambda s: dt.datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ")
    t_now = dt.datetime.utcnow()
    t0 = T(since) if since else None
    kinds, idle = defaultdict(Counter), {}
    for f in sorted(glob.glob(str(SH / "drivers" / "g*_*.log"))):
        drv = Path(f).stem
        g = drv.split("_")[0]
        t_idle, tot = None, 0.0
        for line in open(f):
            p = line.split()
            if len(p) < 2:
                continue
            t = T(p[0])
            if p[1] == "end" and (t0 is None or t >= t0):
                kinds[g][p[2].split("_")[0]] += 1
            if p[1] == "idle" and p[2] == "start":
                t_idle = t
            elif p[1] == "idle" and p[2] == "end" and t_idle is not None:
                a = max(t_idle, t0) if t0 else t_idle
                tot += max(0.0, (t - a).total_seconds())
                t_idle = None
        if t_idle is not None:
            a = max(t_idle, t0) if t0 else t_idle
            tot += max(0.0, (t_now - a).total_seconds())
        idle[drv] = round(tot / 60, 1)
    return {g: dict(c) for g, c in sorted(kinds.items())}, idle


def status(msg):
    with open(SH / "status_box.md", "a") as f:
        f.write(f"- {now()}(box) {msg}\n")


def cmd_loop():
    last_acct = time.time()
    t_acct = now()
    final_done = False
    while True:
        if (SH / "FREEZE").exists():
            return 0
        lines = []
        for vid in cmd_gains(quiet=True):
            lines += post_gain_items(vid, 1)
            status(f"gain ready {vid}: + {jl(SH / 'gains.json')[vid]['pos']['median']:.3g} "
                   f"- {jl(SH / 'gains.json')[vid]['neg']['median']:.3g}; B/C items appended")
        # one retry of a failed item
        q = SH / "queue.txt"
        ids = queued_ids()
        for l in open(q):
            sp = l.split()
            if len(sp) < 3 or sp[1].endswith("_retry"):
                continue
            rc = SH / "claims" / sp[1] / "rc"
            if rc.exists() and rc.read_text().strip() not in ("0", "") and f"{sp[1]}_retry" not in ids:
                lines.append(l.rstrip("\n").replace(f" {sp[1]} ", f" {sp[1]}_retry ", 1))
                status(f"item {sp[1]} failed (rc {rc.read_text().strip()}), one retry appended")
        added = append(lines)
        # P3: final set known once every B seed 0-2 of every vector has a result (or failed twice)
        vs = vectors()
        gains = jl(SH / "gains.json") or {}
        blocked = {vid for vid, v in vs.items() if vid not in gains and any(
            _dead(f"A_{p}_s{o}") for p in {v["pos_pack"], v["neg_pack"]} for o in SEEDS)}
        if not (SH / "d" / "combos.json").exists() and len(gains) + len(blocked) == len(vs):
            pending = [f"B_{vid}_s{o}" for vid in gains for o in SEEDS if not (SH / f"fixed_s{o}" / vid / "done").exists()
                       and not _dead(f"B_{vid}_s{o}")]
            if not pending:
                subprocess.run([DM, __file__, "decide"], check=False)
                if (SH / "final_set.json").exists():
                    subprocess.run([DM, __file__, "combos"], check=False)
                    n = len(jl(SH / "d" / "combos.json") or [])
                    append([qline(3, f"D_{k:02d}", item_py("stack", k)) for k in range((n + 9) // 10)])
                    status(f"final set known ({len(jl(SH / 'final_set.json')['shipped_vids'])} vectors pass the bar "
                           f"and gate); {(n + 9) // 10} stacking chunks appended (P3)")
        # completion: every queued item has an rc (P4 filler included) -> CPU finish once
        if (SH / "d" / "combos.json").exists() and not final_done:
            open_items = [l.split()[1] for l in open(q) if not (SH / "claims" / l.split()[1] / "rc").exists()]
            if not open_items:
                status("queue drained: running final (bar, gate, packs, results) and the B2 mirror")
                r = subprocess.run([DM, __file__, "final"], capture_output=True, text=True)
                status("final rc %d: %s" % (r.returncode, (r.stdout.strip().splitlines() or [""])[-1][:300]))
                (SH / "FINAL_DONE").write_text(now())
                final_done = True
                r = subprocess.run(["bash", f"{B}/scripts/steering_bench/b2_ship.sh", "run"], capture_output=True, text=True)
                status(f"B2 mirror rc {r.returncode}: {r.stdout.strip()[-300:]}")
                (SH / "DONE").write_text(now())
                return 0
        if time.time() - last_acct >= 3600:
            kinds, idle = acct(t_acct)
            nq = sum(1 for l in open(q) if not (SH / "claims" / l.split()[1]).exists())
            status(f"hourly GPU accounting since {t_acct}: items done {kinds}; idle min {idle}; unclaimed {nq}"
                   + ("; FLAG idle > 5 min with queue non-empty" if nq and any(x > 5 for x in idle.values()) else ""))
            last_acct, t_acct = time.time(), now()
        time.sleep(60)


def _dead(iid):
    """True when an item and its retry both failed."""
    r = [SH / "claims" / x / "rc" for x in (iid, f"{iid}_retry")]
    return all(p.exists() and p.read_text().strip() != "0" for p in r)


# ---------------------------------------------------------------- decide / final

def measured(vid, gains):
    """Medians over seeds 0,1,2 of the fixed-gain metrics per side, the bar, and reach."""
    per = {o: fixed_metrics(vid, o, gains) for o in SEEDS}
    have = [o for o in SEEDS if per[o]]
    out = {"seeds_measured": have}
    if not have:
        return out
    for side in SIDES:
        m = {k: statistics.median(per[o][side][k] for o in have)
             for k in ("frac_clap", "frac_muq", "effect_clap", "effect_muq", "lpaps", "lpaps_over_cutoff")}
        m["frac_clap_seeds"] = [per[o][side]["frac_clap"] for o in have]
        m["frac_muq_seeds"] = [per[o][side]["frac_muq"] for o in have]
        m["lpaps_seeds"] = [per[o][side]["lpaps"] for o in have]
        m["reached_n"] = int(sum(gains[vid][side]["reached"]))
        m["good"] = m["frac_clap"] >= FRAC_MIN and m["frac_muq"] >= FRAC_MIN and m["reached_n"] >= 2
        out[side] = m
    s3 = fixed_metrics(vid, 3, gains)
    if s3:
        out["seed3"] = s3
    out["bar"] = len(have) == 3 and any(out[s]["good"] for s in SIDES)
    out["good_signs"] = [s for s in SIDES if out.get(s, {}).get("good")]
    why = []
    if len(have) < 3:
        why.append(f"only seeds {have} measured")
    elif not out["bar"]:
        for s in SIDES:
            m = out[s]
            why.append(f"{s}: CLAP {m['frac_clap']:.2f} MuQ {m['frac_muq']:.2f} reached {m['reached_n']}/3")
    out["bar_why"] = "; ".join(why)
    return out


def gate_cmp(m2, m1):
    why = []
    for s in SIDES:
        for f in ("frac_clap", "frac_muq"):
            if m2[s][f] < m1[s][f] - 0.05:
                why.append(f"{s} {f[5:]} {m2[s][f]:.2f} < v1 {m1[s][f]:.2f} - 0.05")
        if m2[s]["lpaps_over_cutoff"] > 1.1 * m1[s]["lpaps_over_cutoff"]:
            why.append(f"{s} LPAPS/cutoff {m2[s]['lpaps_over_cutoff']:.2f} > 1.1 x v1 {m1[s]['lpaps_over_cutoff']:.2f}")
    r2 = sum(m2[s]["reached_n"] >= 2 for s in SIDES)
    r1 = sum(m1[s]["reached_n"] >= 2 for s in SIDES)
    if r2 < r1:
        why.append(f"reached signs {r2} < v1 {r1}")
    return not why, "; ".join(why)


def cmd_decide():
    vs, gains = vectors(), jl(SH / "gains.json")
    meas = {vid: measured(vid, gains) for vid in vs if vid in gains}
    jw(SH / "measured.json", meas)
    concepts = defaultdict(dict)
    for vid, v in vs.items():
        if v["group"] == "v1":
            concepts[v["v1_name"]]["v1"] = vid
        elif v["group"] == "legacy":
            concepts[f"{v['concept']} (legacy)"]["legacy"] = vid
        else:
            concepts[v["final_name"]]["v2"] = vid
    gate_rows, shipped = [], []
    for key, d in sorted(concepts.items()):
        cand = {r: vid for r, vid in d.items() if meas.get(vid, {}).get("bar")}
        row = {"concept": key, **{f"{r}_vid": vid for r, vid in d.items()},
               **{f"{r}_bar": bool(meas.get(vid, {}).get("bar")) for r, vid in d.items()}}
        if "v1" in d and "v2" in d:
            m1, m2 = meas.get(d["v1"], {}), meas.get(d["v2"], {})
            if m1.get("bar") is not None and "pos" in m1 and "pos" in m2:
                won, why = gate_cmp(m2, m1)
                row["gate_v2_wins"], row["gate_why"] = won, why
            if len(cand) == 2:
                row["outcome"] = "compared: " + ("v2 wins" if row.get("gate_v2_wins") else "v1 kept")
                row["ships"] = cand["v2"] if row.get("gate_v2_wins") else cand["v1"]
            elif len(cand) == 1:
                r = next(iter(cand))
                row["outcome"] = f"decided by the bar: only {r} passes"
                row["ships"] = cand[r]
            else:
                row["outcome"] = "neither vector passes the bar: ships nothing"
                row["ships"] = None
        else:
            r, vid = next(iter(d.items()))
            row["outcome"] = "not compared (no counterpart)" + (" (blocked: protocol failed)" if vid not in gains else "")
            row["ships"] = vid if meas.get(vid, {}).get("bar") else None
            if not row["ships"]:
                row["outcome"] += "; fails the bar: ships nothing"
        gate_rows.append(row)
        if row["ships"]:
            shipped.append(row["ships"])
    names = Counter(vs[x]["final_name"] for x in shipped)
    dup = [n for n, c in names.items() if c > 1]
    if dup:
        print("FLAG: two shipped vectors share a final name:", dup)
    jw(SH / "gate.json", gate_rows)
    jw(SH / "final_set.json", {"decided": now(), "shipped_vids": sorted(shipped)})
    print(f"decide: {len(shipped)} vectors ship; bar pass {sum(bool(m.get('bar')) for m in meas.values())}"
          f"/{len(meas)}")


def _margin(block, fam, anchors):
    d = (block or {}).get(fam) or {}
    if not anchors or anchors[0] not in d:
        return None
    m = np.array(d[anchors[0]], float)
    if len(anchors) > 1 and anchors[1] in d:
        m = m - np.array(d[anchors[1]], float)
    return m


def product(vid):
    """applicability (+ applies_to), signed hold drift, SFX pass, cross-effect own/off-target for one vector."""
    v = vec(vid)
    anchors = [a for a in (v["pos_anchor"], v["neg_anchor"]) if a]
    out = {"app": {}, "applies_to": []}
    kd = SH / "c" / "app" / vid
    out["app_source"] = "reused 07:45Z" if (kd / "REUSED").exists() else "ship pass"
    for st, label in APP_SETS.items():
        b = jl(kd / f"{st}_base.json")
        if not b:
            continue
        fr = {}
        for fam in ("clap", "muq"):
            f = []
            for s in (1, -1):
                j = jl(kd / f"{st}_{s:+d}.json")
                m1, m0 = (_margin(j["all"], fam, anchors) if j else None), _margin(b["all"], fam, anchors)
                if m1 is not None and m0 is not None:
                    f.append(float(np.mean((m1 - m0) * s > 0)))
            fr[fam] = float(np.mean(f)) if f else None
        out["app"][label] = fr
        need = ("clap",) if label == "sfx" else ("clap", "muq")
        if all((fr.get(x) or 0) >= FRAC_MIN for x in need):
            out["applies_to"].append(label)
    h, h0 = jl(SH / "c" / "hold" / f"{vid}.json"), jl(SH / "c" / "hold" / f"{vid}_base.json")
    if h and h0:
        eff = {}
        for w in ("0-10", "20-30"):
            e = [_margin(h[w], fam, anchors) - _margin(h0[w], fam, anchors) for fam in ("clap", "muq")
                 if _margin(h[w], fam, anchors) is not None and _margin(h0[w], fam, anchors) is not None]
            eff[w] = float(np.mean([x.mean() for x in e])) if e else None
        if eff.get("0-10") not in (None, 0.0) and eff.get("20-30") is not None:
            out["hold"] = {"effect_0_10": eff["0-10"], "effect_20_30": eff["20-30"],
                           "drift_signed": (eff["20-30"] - eff["0-10"]) / abs(eff["0-10"]),
                           "sign_crossed": eff["0-10"] * eff["20-30"] < 0}
    sfx = {}
    for o in SEEDS:
        b = jl(SH / "sfx" / vid / f"s{o}_base.json")
        for s in (1, -1):
            j = jl(SH / "sfx" / vid / f"s{o}_{s:+d}.json")
            if not (b and j):
                continue
            for fam in ("clap", "muq"):
                m1, m0 = _margin(j["all"], fam, anchors), _margin(b["all"], fam, anchors)
                if m1 is not None and m0 is not None:
                    sfx.setdefault(f"{'pos' if s > 0 else 'neg'}_{fam}", []).append(float(np.mean((m1 - m0) * s > 0)))
    if sfx:
        out["sfx_prompts"] = {k: statistics.median(x) for k, x in sfx.items()}
        out["sfx_prompts"]["seeds"] = len(next(iter(sfx.values())))
    return out


def crossfx(vids):
    d = SH / "c" / "crossfx"
    base = jl(d / "base.json")
    if not base:
        return {}
    import pandas as pd
    b = {c: np.array(x, float) for c, x in base["all"]["cols"].items()}
    try:
        lab = pd.read_parquet(Path("/dev/shm/steerbench/out/corpus_v2") / "labels" / "merged_all.parquet",
                              columns=[c for c in b if c])
        sd = {c: float(lab[c].std()) if c in lab else float("nan") for c in b}
    except Exception:  # noqa: BLE001
        sd = {c: float(np.std(x)) for c, x in b.items()}
    out = {}
    for vid in vids:
        own = vec(vid)["primary_label"]
        for s in (1, -1):
            j = jl(d / f"{vid}_{s:+d}.json")
            if not j:
                continue
            z = {c: float(np.nanmean(np.array(x, float) - b[c])) / sd[c] if sd.get(c) else float("nan")
                 for c, x in j["all"]["cols"].items() if c in b}
            off = {c: x for c, x in z.items() if c != own and x == x}
            worst = max(off, key=lambda c: abs(off[c])) if off else ""
            out[f"{vid}{'+' if s > 0 else '-'}"] = {
                "own_label": own, "own_z": z.get(own), "max_off_label": worst, "max_off_z": off.get(worst),
                "n_off_over_own": sum(abs(x) > abs(z.get(own) or 0) for x in off.values()),
                "n_cols": len(z), "z": z}
    return out


def stack_table():
    st = [j for j in (jl(p) for p in sorted((SH / "d" / "stack").glob("*.json"))) if j]
    out = {}
    for rule in RULES:
        js = [j for j in st if j["meta"]["rule"] == rule]
        under = [float(np.mean(j["all"]["lpaps"])) <= float(j["meta"]["cutoff_min"]) for j in js]
        out[rule] = {"n": len(js), "frac_under_cutoff": float(np.mean(under)) if under else None,
                     "median_scale": statistics.median(j["meta"]["scale"] for j in js) if js else None}
    ok = [r for r in ("none", "inv_sqrt", "norm_budget", "inv_n") if (out[r]["frac_under_cutoff"] or 0) >= 0.95]
    out["chosen"] = ok[0] if ok else (max(RULES, key=lambda r: out[r]["frac_under_cutoff"] or -1) if st else None)
    out["note"] = ("least attenuating rule keeping >= 95% of combos under the strictest member's cutoff" if ok
                   else "no rule reached 95%; the best observed is named")
    return out


def cmd_final():
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file
    sys.path.insert(0, str(E2))
    from steer8c.packs import PACK_METADATA_KEY, SteeringPack, load_pack

    cmd_decide()
    vs, gains = vectors(), jl(SH / "gains.json")
    meas, gate = jl(SH / "measured.json"), jl(SH / "gate.json")
    shipped = jl(SH / "final_set.json")["shipped_vids"]
    prod = {vid: product(vid) for vid in vs if vid in gains}
    cfx = crossfx(list(vs))
    stk = stack_table()
    jw(SH / "product.json", {"product": prod, "crossfx": {k: {x: y for x, y in v.items() if x != "z"}
                                                         for k, v in cfx.items()}, "stack": stk})
    gate_of = {}
    for r in gate:
        for role in ("v1", "v2", "legacy"):
            if r.get(f"{role}_vid"):
                gate_of[r[f"{role}_vid"]] = r
    out = SH / "packs_final" / "sa3" / "medium"
    if out.exists():
        shutil.move(str(out), str(SH / f"packs_final_prev_{int(time.time())}"))
    out.mkdir(parents=True)

    def read(path):
        with safe_open(str(path), framework="pt") as f:
            meta = json.loads(f.metadata()["steering_pack"])
            t = {k: f.get_tensor(k).to(torch.float32) if f.get_tensor(k).dtype != torch.int64 else f.get_tensor(k)
                 for k in f.keys()}
        return meta, t

    def gstat(vid, side):
        g = gains[vid][side]
        return {"median": g["median"], "min": g["min"], "max": g["max"], "reached": g["reached"],
                "seeds": g["seeds"], "cutoff": g["cutoff"]}

    manifest = []
    for vid in shipped:
        v = vs[vid]
        name = v["final_name"]
        src = V1 / "packs_screening_copy" / f"{v['v1_name']}.safetensors" if v["group"] == "v1" else \
            PKB / f"{v['pos_pack']}.safetensors"
        bm, bt = read(src)
        m, p = meas[vid], prod[vid]
        g = gate_of.get(vid, {})
        prov = dict(bm.get("provenance") or {})
        prov.pop("blocks", None)
        prov.pop("calibrated_gain", None)
        prov["calibrated_gain"] = {
            "pos": gstat(vid, "pos"), "neg": gstat(vid, "neg"),
            "unit": "knob value at which mean LPAPS vs the unsteered render reaches the PCI-all cutoff, interpolated "
                    "log-linearly in alpha between bracketing probes, median over seeds 0,1,2"
                    + ("; neg in the dn vector's own magnitude units (applied to vectors_neg)" if v["c"] else "")}
        prov["measured_at_shipped_gain"] = {s: {k: m[s][k] for k in ("frac_clap", "frac_muq", "frac_clap_seeds",
                                                                    "frac_muq_seeds", "lpaps", "lpaps_over_cutoff",
                                                                    "effect_clap", "effect_muq", "reached_n")}
                                            for s in SIDES}
        prov["quality_bar"] = {"pass": True, "good_signs": m["good_signs"], "rule": "ship_uniform_rigor 2026-10-06"}
        prov["gate"] = g.get("outcome", "not compared (no counterpart)")
        prov["knob_source"] = vid
        prov["origin"] = v["group"] if v["group"] != "v2gate" else "new_v2"
        if v["group"] == "v1":
            prov["kept_from"] = "v1"
        elif v["group"] == "legacy":
            prov["kept_from"] = "legacy"
            prov["legacy_pack"] = vid
        if p.get("hold"):
            prov["hold_30s"] = p["hold"]
        if p.get("app"):
            prov["applicability"] = p["app"]
        if p.get("sfx_prompts"):
            prov["sfx_prompts"] = p["sfx_prompts"]
        cx = {s: cfx.get(f"{vid}{'+' if s == 'pos' else '-'}") for s in SIDES}
        prov["cross_effect"] = {s: ({k: cx[s][k] for k in ("own_label", "own_z", "max_off_label", "max_off_z",
                                                           "n_off_over_own", "n_cols")} if cx[s] else None)
                                for s in SIDES}
        pos, neg = v["pos_anchor"], v["neg_anchor"]
        desc = f"+ {pos} / - {neg}" if neg else f"+ {pos}"
        kw = {"family": "sa3", "checkpoint": "medium", "block": int(bm["block"]), "hidden_size": int(bm["hidden_size"]),
              "name": name, "label": v.get("label") or name.replace("_", " ").title(), "blurb": desc,
              "hook": bm.get("hook", "post_block_residual"), "method": bm.get("method", "caa_diff_means"),
              "norm": float(bm["norm"]), "magnitude": float(bm["magnitude"]),
              "policy": bm.get("policy") or {"kind": "range", "start": 0.0, "end": 1.0}, "provenance": prov,
              "category": v["category"], "applies_to": p.get("applies_to", []), "seeds": list(SEEDS),
              "variant": v["variant"], "description": desc, "pos_anchor": pos, "neg_anchor": neg}
        if v["c"]:
            md, td = read(PKB / f"{v['neg_pack']}.safetensors")
            if int(md["block"]) != int(bm["block"]):
                manifest.append({"name": name, "vid": vid, "status": "not shipped", "why": "c up/dn blocks differ"})
                continue
            u, w = bt["vector"].reshape(-1), td["vector"].reshape(-1)
            prov.update(dn_pack=v["neg_pack"], dn_magnitude_own=float(md["magnitude"]), dn_norm_own=float(md["norm"]))
            kw.update(vector=u / u.norm(), vectors=(u / u.norm())[None], blocks=(int(bm["block"]),),
                      vectors_neg=(w / w.norm())[None], norms=(float(bm["norm"]),))
            tensors = {"vectors": kw["vectors"], "blocks": torch.tensor([int(bm["block"])], dtype=torch.int64),
                       "vectors_neg": kw["vectors_neg"]}
            expect = [(int(bm["block"]), float(bm["magnitude"]), True)]
        else:
            vv = bt["vector"]
            if v["variant"] == "b" and vv.ndim == 3 and vv.shape[0] > 1:
                rows = vv[:, 0, :]
                r = rows.norm(dim=1)
                blocks = [int(x) for x in bm.get("blocks") or (bm.get("provenance") or {}).get("blocks")]
                kw.update(vector=rows[0] / r[0], vectors=rows / r[:, None], blocks=tuple(blocks),
                          norms=tuple(float(bm["norm"]) * float(x) for x in r))
                tensors = {"vectors": kw["vectors"], "blocks": torch.tensor(blocks, dtype=torch.int64)}
                expect = [(b_, float(bm["magnitude"]) * float(x), None) for b_, x in zip(blocks, r)]
            else:
                kw["vector"] = vv.reshape(-1)
                tensors = {"vector": kw["vector"]}
                expect = [(kw["block"], kw["magnitude"], None)]
        pk = SteeringPack(**kw)
        pk.validate()
        meta = pk.metadata()
        meta.update({"label_note": v.get("blurb", ""), "applies_to": p.get("applies_to", [])})
        path = out / f"{name}.safetensors"
        save_file({k: x.detach().cpu().to(torch.float32).contiguous() if x.dtype != torch.int64 else x
                   for k, x in tensors.items()}, str(path), metadata={PACK_METADATA_KEY: json.dumps(meta, sort_keys=True)})
        back = load_pack(path)
        for term, (blk, pm, has_neg) in zip(back.terms(), expect):
            assert term.block == blk and abs(term.pos_magnitude - pm) < 1e-4 * max(1, pm), (name, term, blk, pm)
            assert abs(float(term.pos.norm()) - 1) < 1e-3, (name, "non-unit pos row")
            if has_neg:
                assert term.neg is not None and abs(float(term.neg.norm()) - 1) < 1e-3, (name, "neg row")
        if v["group"] == "v1":  # vector bytes unchanged
            assert torch.equal(read(path)[1]["vector"].reshape(-1), bt["vector"].reshape(-1)), name
        manifest.append({"name": name, "vid": vid, "origin": prov["origin"], "variant": v["variant"],
                         "status": "shipped", "gate": prov["gate"]})
    for vid, v in vs.items():
        if vid not in shipped and vid in meas:
            manifest.append({"name": v["final_name"], "vid": vid, "origin": v["group"], "variant": v["variant"],
                             "status": "not shipped", "why": meas[vid].get("bar_why") or gate_of.get(vid, {}).get("outcome", "")})
    import pandas as pd
    pd.DataFrame(manifest).to_csv(SH / "packs_final" / "manifest.csv", index=False)
    # one metadata shape: every shipped header carries the dict-per-sign gain
    for f in out.glob("*.safetensors"):
        with safe_open(str(f), framework="pt") as fh:
            g = json.loads(fh.metadata()["steering_pack"])["provenance"]["calibrated_gain"]
        assert all(isinstance(g[s], dict) and set(g[s]) >= {"median", "min", "max", "reached", "seeds", "cutoff"}
                   for s in SIDES), f
    results(manifest, meas, prod, cfx, stk, gate)
    print(f"final: {sum(m['status'] == 'shipped' for m in manifest)} shipped, "
          f"{sum(m['status'] != 'shipped' for m in manifest)} not shipped")


def results(manifest, meas, prod, cfx, stk, gate):
    vs, gains = vectors(), jl(SH / "gains.json")
    sk = jl(SH / "skip_c.json") or {}
    L = ["# Ship pass results: every shipped SA3 steering knob under one protocol (2026-10-06)", "",
         "Rules: ship/rules.md (frozen before any render). Gain = interpolated at the PCI-all cutoff, median over seeds "
         "0,1,2; fractions, LPAPS and reach measured AT that gain on 50 test prompts x 3 seeds (fixed-gain pass).", ""]
    sh = [m for m in manifest if m["status"] == "shipped"]
    L.append(f"Shipped {len(sh)}; vectors measured {len(meas)}; bar pass "
             f"{sum(bool(m.get('bar')) for m in meas.values())}. Origins shipped: "
             f"{dict(Counter(m['origin'] for m in sh))}.")
    L.append(f"Gate outcomes: {dict(Counter(r['outcome'].split(':')[0] for r in gate))}.")
    L += ["", "## Knob table (all measured vectors)", "",
          "| pack | vector | origin | variant | blocks | gain + median (min-max) | gain - median (min-max) | reached +/- "
          "| CLAP + / - | MuQ + / - | LPAPS/cut + / - | bar | gate | hold drift (signed) | cross-effect own z +/- "
          "(n off > own) | applies_to | ships |",
          "|" + "---|" * 18]
    shipped = {m["vid"] for m in sh}
    gate_of = {}
    for r in gate:
        for role in ("v1", "v2", "legacy"):
            if r.get(f"{role}_vid"):
                gate_of[r[f"{role}_vid"]] = r
    for vid in sorted(meas, key=lambda x: (vs[x]["final_name"], x)):
        v, m, p = vs[vid], meas[vid], prod.get(vid, {})
        g = gains[vid]

        def gs(s):
            return f"{g[s]['median']:.2f} ({g[s]['min']:.2f}-{g[s]['max']:.2f})"

        def f2(s, k):
            return f"{m[s][k]:.2f}" if s in m else ""
        rch = "/".join("".join("y" if x else "n" for x in g[s]["reached"]) for s in SIDES)
        h = p.get("hold")
        hd = f"{h['drift_signed']:+.2f}{' x' if h['sign_crossed'] else ''}" if h else "not measured"
        cx = []
        for s in "+-":
            c = cfx.get(f"{vid}{s}")
            cx.append(f"{c['own_z']:.2f} ({c['n_off_over_own']})" if c and c["own_z"] is not None else "-")
        blocks = v.get("blocks") or ""
        L.append(f"| {v['final_name']} | {vid} | {v['group']} | {v['variant']} | {blocks} | {gs('pos')} | {gs('neg')} | "
                 f"{rch} | {f2('pos', 'frac_clap')} / {f2('neg', 'frac_clap')} | {f2('pos', 'frac_muq')} / "
                 f"{f2('neg', 'frac_muq')} | {f2('pos', 'lpaps_over_cutoff')} / {f2('neg', 'lpaps_over_cutoff')} | "
                 f"{'pass' if m.get('bar') else 'FAIL'} | {gate_of.get(vid, {}).get('outcome', '')} | {hd} | "
                 f"{' / '.join(cx)} | {','.join(p.get('applies_to', [])) or '(none measured >= 0.6)'} | "
                 f"{'yes' if vid in shipped else 'no'} |")
    L += ["", "## Regression gate", "", "| concept | v1 | v2 | legacy | outcome | why |", "|---|---|---|---|---|---|"]
    for r in gate:
        L.append(f"| {r['concept']} | {r.get('v1_vid', '')} {'(bar)' if r.get('v1_bar') else ''} | "
                 f"{r.get('v2_vid', '')} {'(bar)' if r.get('v2_bar') else ''} | {r.get('legacy_vid', '')} | "
                 f"{r['outcome']} | {r.get('gate_why', '')} |")
    L += ["", "## Stacking (300 combos over the final set, LPAPS vs the strictest member's cutoff)", "",
          "| rule | combos | fraction under cutoff | median scale |", "|---|---|---|---|"]
    for rule in RULES:
        x = stk.get(rule) or {}
        L.append(f"| {rule} | {x.get('n', 0)} | {x.get('frac_under_cutoff')} | {x.get('median_scale')} |")
    L += ["", f"Chosen: {stk.get('chosen')} ({stk.get('note', '')}). norm_budget: the combined perturbation's Frobenius "
          "norm over all steps and blocks is held to the largest single member's.", ""]
    sfx = {vid: p["sfx_prompts"] for vid, p in prod.items() if p.get("sfx_prompts")}
    if sfx:
        L += ["## SFX prompt pass (12 SFX prompts x 3 seeds at the shipped gain; median intended-sign fraction; "
              "recorded, not gating)", "", "| vector | CLAP + | CLAP - | MuQ + | MuQ - |", "|---|---|---|---|---|"]
        for vid, s in sorted(sfx.items()):
            L.append(f"| {vid} | {s.get('pos_clap', float('nan')):.2f} | {s.get('neg_clap', float('nan')):.2f} | "
                     f"{s.get('pos_muq', float('nan')):.2f} | {s.get('neg_muq', float('nan')):.2f} |")
        L.append("")
    L += ["## Notes and deviations", "",
          f"- C parts reused from last night (gain within 5%): applicability/hold for "
          f"{sum(x['skip_app_hold'] for x in sk.values())} new knobs, cross-effect for "
          f"{sum(x['skip_crossfx'] for x in sk.values())}; reused cross-effect rows lack label columns added to the "
          "universe for the old vectors (n_cols shows it).",
          "- v2gate: the v2 vectors that lost last night's gate (and sfx_laden v2) were measured too, so the gate "
          "compares like with like; their missing seed-2 protocol runs were added to P0.",
          "- v1 vectors were measured with their v2 counterpart's PCI descriptors and anchors (PCI results reused per "
          "seed where last night rendered them); the 11 v1 concepts without a v2 counterpart use the catalogue anchors.",
          "- Hold drift and cross-effect are reported, not gating. applies_to is empty when no category passes.",
          "- c packs: neg gain in the dn vector's own magnitude units.", ""]
    (SH / "results_ship.md").write_text("\n".join(L), encoding="utf-8")
    print("results_ship.md", len(L), "lines")


def main():
    c, a = sys.argv[1], sys.argv[2:]
    fn = {"init": cmd_init, "rules": cmd_rules, "gains": cmd_gains, "skiplist": cmd_skiplist,
          "reuse": reuse_skipped, "fixedplan": cmd_fixedplan, "sfx": cmd_sfx, "capp": cmd_capp, "cfx": cmd_cfx,
          "combos": cmd_combos, "stack": cmd_stack, "enqueue0": cmd_enqueue0, "loop": cmd_loop,
          "decide": cmd_decide, "final": cmd_final}[c]
    r = fn(*a)
    return r if isinstance(r, int) else 0


if __name__ == "__main__":
    sys.exit(main())
