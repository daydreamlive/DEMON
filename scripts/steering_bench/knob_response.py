"""S4: perceptual knob mapping from the ship-pass probe curves (CPU only).

For every measured vector-sign and seed 0,1,2 the ship pass rendered a probe LPAPS curve (20 holdout prompts, alpha in
+-{2,4,...,128} x the pack's alpha units) and interpolated the shipped gain where that curve crosses the PCI-all cutoff
(ship_tools.interp_side: log-linear in alpha between bracketing probes, linear from alpha 0 below the first probe). This
script reuses exactly that interpolant, after a monotone (pool-adjacent-violators) clean-up, so u = 1 lands on the
shipped gain by construction, and:

  * raw curvature: the UI maps slider position u in [0, 1] linearly to knob = u x gain. f(u) = L(u A) / L(A) (A = alpha at
    the cutoff) is how much of the LPAPS budget the raw slider has spent at u; linear would be f(u) = u.
  * perceptual map: r(u) = A^-1 L^-1(u L(A)), the knob multiple of the shipped gain that puts LPAPS at u x cutoff.
    Median over seeds per grid point. Past u = 1 the map stays linear in knob (the UI headroom is unchanged).

Writes <out>/knob_response.json (proposed sidecar), <out>/curvature.json (per vector-sign stats) and <out>/s4.md.

Usage:
    python scripts/steering_bench/knob_response.py \
        --mk E:/Projects/DEMON/steering-bench/many_knobs_v2 --out E:/Projects/DEMON/steering-bench/v3_smoke/s4
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
import statistics
from pathlib import Path

SEEDS = (0, 1, 2)
HEADROOM = 1.25  # acestep.streaming.knobs.STEERING_PACK_HEADROOM
FINE = [i / 400 for i in range(401)]
GRIDS = {6: [i / 5 for i in range(6)], 11: [i / 10 for i in range(11)], 21: [i / 20 for i in range(21)],
         41: [i / 40 for i in range(41)]}
TOL = 0.02  # max |LPAPS error| / cutoff a sidecar grid may introduce vs the fine map


def lp_curve(f: Path):
    if not f.exists():
        return None
    with open(f, newline="") as fh:
        return {float(r["alpha"]): float(r["mean"]) for r in csv.DictReader(fh)}


def res_dir(mk: Path, pack: str, off: int):
    for root in (mk / "ship" / f"results_s{off}", mk / "eval" / f"results_s{off}"):
        d = root / pack
        # the local mirror carries no "done" markers; the probe csv is the evidence
        if (d / "calib" / f"pack_pack_{pack}" / "protocol_results" / "lpaps.csv").exists():
            return d
    return None


def pav(ys):
    """Non-decreasing least-squares fit (pool adjacent violators); returns fitted values and violation count."""
    blocks = [[y, 1] for y in ys]
    i, viol = 0, sum(1 for a, b in zip(ys, ys[1:]) if b < a)
    while i < len(blocks) - 1:
        if blocks[i][0] > blocks[i + 1][0]:
            v = (blocks[i][0] * blocks[i][1] + blocks[i + 1][0] * blocks[i + 1][1]) / (blocks[i][1] + blocks[i + 1][1])
            blocks[i] = [v, blocks[i][1] + blocks[i + 1][1]]
            del blocks[i + 1]
            i = max(i - 1, 0)
        else:
            i += 1
    out = []
    for v, n in blocks:
        out += [v] * n
    return out, viol


class Curve:
    """Monotone LPAPS(alpha) on one sign: linear on [0, a1], log-linear between probes (the ship interpolant)."""

    def __init__(self, probe: dict, sign: int):
        pts = sorted((abs(a), v) for a, v in probe.items() if sign * a > 0)
        ys, self.violations = pav([v for _, v in pts])
        self.a = [a for a, _ in pts]
        self.y = ys
        self.v0 = probe.get(0.0, 0.0)

    def L(self, alpha):
        a, y = self.a, self.y
        if alpha <= a[0]:
            return self.v0 + (y[0] - self.v0) * alpha / a[0]
        for i in range(1, len(a)):
            if alpha <= a[i]:
                t = (math.log(alpha) - math.log(a[i - 1])) / (math.log(a[i]) - math.log(a[i - 1]))
                return y[i - 1] + t * (y[i] - y[i - 1])
        return y[-1]  # no extrapolation past the largest probe

    def inv(self, target):
        """Smallest alpha with L(alpha) >= target (None when never reached)."""
        a, y = self.a, self.y
        if target <= self.v0:
            return 0.0
        if target <= y[0]:
            return a[0] * (target - self.v0) / (y[0] - self.v0) if y[0] > self.v0 else a[0]
        for i in range(1, len(a)):
            if y[i] >= target:
                if y[i] <= y[i - 1]:
                    return a[i - 1]
                t = (target - y[i - 1]) / (y[i] - y[i - 1])
                return math.exp(math.log(a[i - 1]) + t * (math.log(a[i]) - math.log(a[i - 1])))
        return None


def interp(xs, ys, x):
    for i in range(1, len(xs)):
        if x <= xs[i]:
            t = (x - xs[i - 1]) / (xs[i] - xs[i - 1])
            return ys[i - 1] + t * (ys[i] - ys[i - 1])
    return ys[-1]


def side_seed(mk, pack, sign, off, cut):
    d = res_dir(mk, pack, off)
    if d is None:
        return None
    probe = lp_curve(d / "calib" / f"pack_pack_{pack}" / "protocol_results" / "lpaps.csv")
    if not probe:
        return None
    c = Curve(probe, sign)
    A = c.inv(cut)
    reached = A is not None
    if not reached:
        A = c.a[-1]
    top = c.L(A)
    f = [c.L(u * A) / top for u in FINE]                      # raw slider: share of budget spent at u
    r = [(lambda x: A if x is None else x)(c.inv(u * top)) / A for u in FINE]             # perceptual map: knob multiple at LPAPS u x top
    return {"A": A, "cut": cut, "top": top, "reached": reached, "violations": c.violations, "f": f, "r": r,
            "u_first_probe": c.a[0] / A, "f_first_probe": c.L(c.a[0]) / top, "curve": c}


def stats_of(f):
    dev = [fi - u for fi, u in zip(f, FINE)]
    u50 = next(u for u, fi in zip(FINE, f) if fi >= 0.5)
    return {"max_dev": max(dev), "mean_abs_dev": sum(abs(x) for x in dev) / len(dev), "u50": u50,
            "f_at_0.1": f[40], "f_at_0.25": f[100], "f_at_0.5": f[200]}


def q(xs, p):
    xs = sorted(xs)
    k = (len(xs) - 1) * p
    lo, hi = math.floor(k), math.ceil(k)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mk", type=Path, default=Path("E:/Projects/DEMON/steering-bench/many_knobs_v2"))
    ap.add_argument("--out", type=Path, default=Path("E:/Projects/DEMON/steering-bench/v3_smoke/s4"))
    a = ap.parse_args()
    sh = a.mk / "ship"
    gains = json.loads((sh / "gains.json").read_text())
    vecs = json.loads((sh / "vectors.json").read_text())
    meas = json.loads((sh / "measured.json").read_text())
    shipped = set(json.loads((sh / "final_set.json").read_text())["shipped_vids"])
    sets = json.loads((a.mk / "eval" / "eval_sets.json").read_text())
    a.out.mkdir(parents=True, exist_ok=True)

    rows, per = [], {}
    for vid, g in sorted(gains.items()):
        for side in ("pos", "neg"):
            gs = g[side]
            seeds = []
            for off in SEEDS:
                ps = gs["per_seed"][str(off)]
                s = side_seed(a.mk, gs["pack"], gs["steer_sign"], off, ps["cutoff"])
                if s is None:
                    continue
                s["alpha_ship"] = ps["alpha"]
                seeds.append(s)
            if not seeds:
                continue
            st = [stats_of(s["f"]) for s in seeds]
            med = {k: statistics.median(x[k] for x in st) for k in st[0]}
            # perceptual map relative to the shipped (median-over-seeds) gain: knob = gain_median * R(u)
            mag, gmed = gs["magnitude"], gs["median"]
            R = [statistics.median(s["r"][i] * s["A"] / mag / gmed for s in seeds) for i in range(len(FINE))]
            spread05 = max(s["r"][200] for s in seeds) - min(s["r"][200] for s in seeds)
            good = bool(meas.get(vid, {}).get(side, {}).get("good"))
            row = {"vid": vid, "name": vecs[vid].get("final_name") or vid, "side": side, "pack": gs["pack"],
                   "shipped": vid in shipped, "good": good, "category": vecs[vid].get("category"),
                   "n_seeds": len(seeds), "reached": [s["reached"] for s in seeds],
                   "pav_violations": sum(s["violations"] for s in seeds),
                   "alpha_fit_over_ship": statistics.median(s["A"] / s["alpha_ship"] for s in seeds),
                   "u_first_probe": statistics.median(s["u_first_probe"] for s in seeds),
                   "f_first_probe": statistics.median(s["f_first_probe"] for s in seeds),
                   "r_seed_spread_at_0.5": spread05, "R_at_1": R[400], **med}
            rows.append(row)
            per[(vid, side)] = {"R": R, "seeds": seeds, "gain": gmed, "mag": mag}

    core = [r for r in rows if r["shipped"] and r["good"]]
    if not core:
        raise SystemExit("no shipped good sides found")

    # grid choice: smallest grid whose piecewise-linear R reproduces the fine map within TOL of the LPAPS budget
    def grid_err(grid, key):
        errs = []
        for r in core:
            p = per[(r["vid"], r["side"])]
            Rg = [interp(FINE, p["R"], u) for u in grid]
            s = p["seeds"][0]
            c, top, A = s["curve"], s["top"], s["A"]
            scale = p["gain"] * p["mag"] / A
            errs.append(max(abs(c.L(interp(grid, Rg, u) * scale * A) / top - c.L(interp(FINE, p["R"], u) * scale * A) / top)
                            for u in FINE))
        return max(errs) if key == "max" else statistics.median(errs)
    grid_report = {n: {"max": grid_err(g, "max"), "median": grid_err(g, "median")} for n, g in GRIDS.items()}
    n_pick = next((n for n in sorted(GRIDS) if grid_report[n]["max"] <= TOL), max(GRIDS))
    grid = GRIDS[n_pick]

    # one global curve vs per-knob: LPAPS error of applying the median R to every knob
    Rglob = [statistics.median(per[(r["vid"], r["side"])]["R"][i] for r in core) for i in range(len(FINE))]
    gerr = []
    for r in core:
        p = per[(r["vid"], r["side"])]
        s = p["seeds"][0]
        scale = p["gain"] * p["mag"] / s["A"]
        gerr.append(max(abs(s["curve"].L(Rglob[i] * scale * s["A"]) / s["top"] - u) for i, u in enumerate(FINE)))

    # sidecar: keyed by bundle knob name; shipped vectors only, both signs (the runtime clamps failing sides anyway)
    knobs = {}
    for r in rows:
        if not r["shipped"]:
            continue
        p = per[(r["vid"], r["side"])]
        knobs.setdefault(r["name"], {"vector": r["vid"]})[r["side"]] = {
            "r": [round(interp(FINE, p["R"], u), 5) for u in grid],
            "reached": r["reached"], "bar_pass": r["good"], "u50_raw": round(r["u50"], 4)}
    side_car = {
        "version": 1, "kind": "steering_knob_response", "model": "sa3/medium",
        "created": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "basis": "LPAPS on the ship-pass probe (20 holdout prompts), seeds 0,1,2, median per grid point",
        "semantics": ("UI position u in [0,1] per sign; knob = sign * calibrated_gain[sign].median * interp(u_grid, r, u). "
                      "u = 1 is the shipped gain (r = 1). For 1 < u <= headroom the knob stays linear: "
                      "knob = sign * calibrated_gain[sign].median * u. A knob absent from this file keeps the linear map."),
        "headroom": HEADROOM, "u_grid": grid, "interp": "piecewise_linear",
        "source": {"gains": str(sh / "gains.json"), "probe": "results_s{0,1,2}/<pack>/calib/pack_pack_<pack>/protocol_results/lpaps.csv",
                   "interpolant": "ship_tools.interp_side (log-linear between probes, linear below the first probe) after PAV"},
        "knobs": dict(sorted(knobs.items())),
    }
    (a.out / "knob_response.json").write_text(json.dumps(side_car, indent=1))

    agg = lambda k: {"median": statistics.median(r[k] for r in core), "p10": q([r[k] for r in core], 0.1),
                     "p90": q([r[k] for r in core], 0.9), "min": min(r[k] for r in core), "max": max(r[k] for r in core)}
    summary = {k: agg(k) for k in ("max_dev", "mean_abs_dev", "u50", "f_at_0.1", "f_at_0.25", "f_at_0.5",
                                   "u_first_probe", "f_first_probe", "r_seed_spread_at_0.5", "alpha_fit_over_ship")}
    worst = sorted(core, key=lambda r: -r["max_dev"])[:8]
    least = sorted(core, key=lambda r: r["max_dev"])[:5]
    curv = {"n_core": len(core), "n_all": len(rows), "summary": summary, "grid_report": grid_report, "grid_points": n_pick,
            "global_curve_max_err": {"median": statistics.median(gerr), "max": max(gerr)},
            "global_R": {str(u): round(interp(FINE, Rglob, u), 5) for u in grid},
            "rows": [{k: v for k, v in r.items()} for r in rows]}
    (a.out / "curvature.json").write_text(json.dumps(curv, indent=1))

    # listening list: 5 most curved shipped good sides, one per category where possible, with a prompt from that category
    pick, cats = [], set()
    for r in sorted(core, key=lambda r: -r["max_dev"]):
        if r["category"] in cats and len(cats) < 3 or any(x["name"] == r["name"] for x in pick):
            continue
        pick.append(r)
        cats.add(r["category"])
        if len(pick) == 5:
            break
    listen = []
    for i, r in enumerate(pick):
        p = per[(r["vid"], r["side"])]
        sgn = 1 if r["side"] == "pos" else -1
        cat = (r["category"] or "music").split(",")[0]
        pool = sets.get(f"app_{cat}", sets["app_music"])
        mid = interp(FINE, p["R"], 0.5)  # perceptual midpoint: LPAPS at half the budget
        listen.append({"knob": r["name"], "side": r["side"], "category": cat, "prompt": pool[i % len(pool)], "seed": 0,
                       "a_knob": round(sgn * p["gain"] * mid, 3), "a_raw_u": round(mid, 3),
                       "b_knob": round(sgn * p["gain"], 3),
                       "question": "with dry (knob 0) as reference: does A sound about halfway from dry to B?"})
    curv["listening"] = listen
    (a.out / "curvature.json").write_text(json.dumps(curv, indent=1))

    nseed = {}
    for r in core:
        nseed[r["n_seeds"]] = nseed.get(r["n_seeds"], 0) + 1
    S = summary
    L = ["# S4: perceptual knob mapping (CPU, ship-pass probe curves)", "",
         f"Scope: {len(core)} shipped bar-passing vector-signs (of {len(rows)} measured), seeds 0,1,2, probe LPAPS on 20 holdout "
         "prompts. Interpolant = ship_tools.interp_side after PAV, so u = 1 is the shipped gain "
         f"(median fit/ship alpha {S['alpha_fit_over_ship']['median']:.3f}). Seeds per side (core): {nseed}.", "",
         "## Raw slider curvature (knob linear in UI position, f(u) = share of the LPAPS budget spent at u)", "",
         "| stat | median | p10 | p90 | min | max |", "|---|---|---|---|---|---|"]
    for k in ("max_dev", "mean_abs_dev", "u50", "f_at_0.1", "f_at_0.25", "f_at_0.5", "u_first_probe", "f_first_probe",
              "r_seed_spread_at_0.5"):
        L.append(f"| {k} | {S[k]['median']:.3f} | {S[k]['p10']:.3f} | {S[k]['p90']:.3f} | {S[k]['min']:.3f} | {S[k]['max']:.3f} |")
    L += ["", "max_dev = max over u of f(u) - u (0 = linear). u50 = slider position that spends half the budget (0.5 = linear). "
          "u_first_probe = slider position of the smallest probe (alpha 2): below it the curve is a straight line from "
          "(0, 0), not data.", "",
          "Most curved (shipped, bar-passing): " + ", ".join(f"{r['name']} {r['side']} ({r['max_dev']:.2f}, u50 {r['u50']:.3f})" for r in worst[:5]),
          "", "Least curved: " + ", ".join(f"{r['name']} {r['side']} ({r['max_dev']:.2f}, u50 {r['u50']:.3f})" for r in least), "",
          f"Grid: {n_pick} points reproduce the fine map within {TOL:.0%} of the budget (max err per grid: " +
          ", ".join(f"{n}: {v['max']:.3f}" for n, v in grid_report.items()) + ").",
          f"One global curve for every knob instead: LPAPS error median {statistics.median(gerr):.3f}, max {max(gerr):.3f} of the budget.",
          "", "Global median map r(u) (knob multiple of the shipped gain): " +
          ", ".join(f"{u:g}:{interp(FINE, Rglob, u):.3f}" for u in grid), "",
          "## Listening list (dry = knob 0; A = perceptual midpoint (LPAPS at half the budget); B = shipped gain; seed 0)", "",
          "Question per item: does A sound about halfway from dry to B? The raw slider would put A at the listed raw u.", "",
          "| knob | side | A knob (raw u) | B knob | prompt |", "|---|---|---|---|---|"]
    for x in listen:
        L.append(f"| {x['knob']} | {x['side']} | {x['a_knob']} ({x['a_raw_u']}) | {x['b_knob']} | {x['prompt']} |")
    (a.out / "s4.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
