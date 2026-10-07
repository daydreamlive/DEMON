"""v3 smoke S3: stacking rules for the shipped SA3 knob set, LPAPS budget and own-label effect retained.

Runbook: notes/steering_pr/v3_smoke_batch_runbook.md (DEMON), section S3. Reuses the ship pass stacking design
(ship_tools.py cmd_combos / cmd_stack / stack_table): the same 300 combos (200 pairs + 100 triples over the 97
shipped vectors, random signs), the 12 screen_music prompts, seed 2115 + offset, 10 s, 8 steps, every member at its
shipped gain (gains.json median), LPAPS of the combo vs the alpha-0 render, pass = mean LPAPS over the 12 prompts <=
the strictest member's cutoff (median over seeds of that member-sign's cutoff).

Vectors come from the installed bundle (DEMON-steer ``load_bundle``, ``export`` subcommand, run with DEMON-steer on
sys.path). Shift per unit knob value = magnitude x unit direction, per block, as the runtime's ``build_configs``,
EXCEPT the negative side of a variant-c pack: the shipped neg gain is in the dn vector's own magnitude units
(``provenance.dn_magnitude_own``), so the shift here is g_neg x dn_magnitude_own x neg (what the ship pass calibrated
and stacked). Policies are all range 0..1 (every step weight 1).

Rules (scale applied to every member's shift):
  none       1 (shipped gains only)
  inv_sqrt   1/sqrt(k)
  inv_n      1/k
  shared_l2  min(1, min_i ||S_i|| / ||sum_i S_i||): the norm of the summed shift equals the strictest member's
             single-knob shift norm at its shipped gain (Frobenius over all steps and blocks)
  shared_l1  min(1, min_i ||S_i|| / sum_i ||S_i||): the sum of the member shift norms equals that budget
             (triangle-inequality bound; = inv_n when the member norms are equal)
The ship pass also ran ``norm_budget`` (= shared_l2 with the LARGEST member's norm as budget); its numbers are read
from the ship tree for the table, not re-rendered.

Own-label effect retained, per member of a combo under a rule: mean over prompts of (own label, combo - base) /
mean over prompts of (own label, member alone at its shipped gain - base); own label = vectors.json primary_label.
The singles are rendered here too (same GPU, same scorers).

Subcommands:
  export   bundle -> <out>/shifts/<name>.npz + index.json   (DEMON-steer python path)
  plan     <out>/plan.json: combos, members, cutoffs, scales per rule, phases
  run      render + score (demon env renders in-process; evalenv worker scores); resumable, phase A then B (then C)
  serve    internal: the evalenv scoring worker
  analyse  <out>/results.json + results.md
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
SHIP = Path("E:/Projects/DEMON/steering-bench/many_knobs_v2/ship")
EVAL = Path("E:/Projects/DEMON/steering-bench/many_knobs_v2/eval")
OUT = Path("E:/Projects/DEMON/steering-bench/v3_smoke/s3")
BUNDLE = Path.home() / ".daydream-scope/models/demon/steering_packs/sa3/medium/bundle.safetensors"
STEER_REPO = Path("C:/_dev/projects/DEMON-steer")
SR, STEPS, DURATION, EVAL_SEED = 44100, 8, 10.0, 2115
RULES = ("none", "inv_sqrt", "inv_n", "shared_l2", "shared_l1")
SUBSET = list(range(0, 40)) + list(range(200, 220))   # phase A: 40 pairs + 20 triples, every rule
SUBSET_ORDER = [x for k in range(20) for x in (2 * k, 2 * k + 1, 200 + k)]   # 2 pairs, 1 triple, repeat


def _log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


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


# ------------------------------------------------------------------------------------------------ export

def cmd_export(a):
    sys.path.insert(0, str(Path(a.steer_repo)))
    from acestep.steering import packs as P

    d = Path(a.out) / "shifts"
    d.mkdir(parents=True, exist_ok=True)
    idx = {}
    for p in P.load_bundle(Path(a.bundle)):
        if p.hook != "post_block_residual" or any(w != 1.0 for w in P.policy_weights(p.policy, STEPS)):
            raise SystemExit(f"{p.name}: hook {p.hook} / policy {p.policy} not handled")
        g = p.provenance["calibrated_gain"]
        dn_mag = p.provenance.get("dn_magnitude_own")
        pos_b, pos_v, neg_v = [], [], []
        for t in p.terms():
            pos_b.append(int(t.block))
            pos_v.append((t.pos.float() * t.pos_magnitude).numpy())
            if t.neg is not None:
                if dn_mag is None:
                    raise SystemExit(f"{p.name}: neg vector without dn_magnitude_own")
                neg_v.append((t.neg.float() / t.neg.float().norm() * float(dn_mag)).numpy())   # applied at +g_neg
            else:
                neg_v.append(-(t.pos.float() * t.pos_magnitude).numpy())                        # -g_neg x pos
        np.savez(d / f"{p.name}.npz", blocks=np.array(pos_b, np.int64), pos=np.stack(pos_v), neg=np.stack(neg_v))
        idx[p.name] = {"variant": p.variant, "blocks": pos_b, "c": bool(neg_v and dn_mag is not None),
                       "runtime_neg_magnitude": [t.neg_magnitude for t in p.terms()] if dn_mag else None,
                       "dn_magnitude_own": dn_mag,
                       "gain": {s: float(g[s]["median"]) for s in ("pos", "neg")},
                       "cutoff": {s: float(statistics.median(g[s]["cutoff"])) for s in ("pos", "neg")},
                       "bar_pass": p.bar_pass}
    jw(d / "index.json", idx)
    print(f"exported {len(idx)} knobs -> {d}")
    return 0


# ------------------------------------------------------------------------------------------------ plan

def shift_of(out, name, sign):
    z = np.load(Path(out) / "shifts" / f"{name}.npz")
    return list(z["blocks"]), z["pos" if sign > 0 else "neg"].astype(np.float64)


def member_norm(blocks, vecs, gain):
    return gain * float(np.sqrt((vecs ** 2).sum()))


def scales(out, members):
    """members: [(name, sign, gain)] -> {rule: scale}, plus norms."""
    tot, norms = {}, []
    for name, s, g in members:
        bl, v = shift_of(out, name, s)
        norms.append(member_norm(bl, v, g))
        for b, x in zip(bl, v):
            tot[b] = tot.get(b, 0) + g * x
    nt = math.sqrt(sum(float((x ** 2).sum()) for x in tot.values()))
    k = len(members)
    return {"none": 1.0, "inv_sqrt": k ** -0.5, "inv_n": 1.0 / k,
            "shared_l2": min(1.0, min(norms) / nt) if nt > 0 else 1.0,
            "shared_l1": min(1.0, min(norms) / sum(norms))}, norms, nt


def cmd_plan(a):
    out = Path(a.out)
    idx = jl(out / "shifts" / "index.json")
    vs, gains = jl(SHIP / "vectors.json"), jl(SHIP / "gains.json")
    combos = jl(SHIP / "d" / "combos.json")
    plan, bad = [], []
    for ci, cb in enumerate(combos):
        mem = []
        for vid, s in cb:
            v = vs[vid]
            name, side = v["final_name"], "pos" if s > 0 else "neg"
            k = idx[name]
            g_ship = gains[vid][side]["median"]
            if abs(k["gain"][side] - g_ship) > 1e-6 * max(1.0, abs(g_ship)):
                bad.append((vid, side, k["gain"][side], g_ship))
            mem.append({"vid": vid, "name": name, "sign": s, "gain": k["gain"][side], "cutoff": k["cutoff"][side],
                        "own": v["primary_label"]})
        sc, norms, nt = scales(out, [(m["name"], m["sign"], m["gain"]) for m in mem])
        for m, n in zip(mem, norms):
            m["shift_norm"] = n
        plan.append({"i": ci, "members": mem, "cutoff_min": min(m["cutoff"] for m in mem), "scale": sc,
                     "sum_norm": nt, "phase": "A" if ci in SUBSET else "B"})
    if bad:
        raise SystemExit(f"bundle gains differ from ship gains.json: {bad[:5]}")
    # cross-check vs the ship pass stack meta (scale of inv_n and the cutoff)
    ship = {}
    for f in (SHIP / "d" / "stack").glob("c*_none.json"):
        j = jl(f)
        ship[int(f.name[1:4])] = j["meta"]["cutoff_min"]
    worst = max(abs(ship[p["i"]] - p["cutoff_min"]) for p in plan if p["i"] in ship)
    jw(out / "plan.json", plan)
    singles = sorted({(m["name"], m["sign"]) for p in plan if p["phase"] == "A" for m in p["members"]})
    print(f"plan: {len(plan)} combos, phase A {sum(p['phase'] == 'A' for p in plan)}; {len(singles)} singles; "
          f"cutoff vs ship max |diff| {worst:.2e}")
    for r in RULES:
        print(f"  {r}: median scale {statistics.median(p['scale'][r] for p in plan):.3f}")
    return 0


# ------------------------------------------------------------------------------------------------ scoring worker

def cmd_serve(a):
    out = os.fdopen(os.dup(1), "w", buffering=1)
    os.dup2(2, 1)
    sys.stdout = sys.stderr
    if a.tada_root:
        os.environ["TADA_ROOT"] = a.tada_root
    sys.path.insert(0, str(HERE))
    from screen_knobs import RealBackend

    be = RealBackend(None, {"label_workers": a.label_workers})
    out.write(json.dumps({"ready": True}) + "\n")
    for line in sys.stdin:
        if not line.strip():
            continue
        req = json.loads(line)
        if req.get("quit"):
            break
        try:
            wav = np.load(req["npz"])["wav"]
            res = {"id": req["id"], "cols": be.cols(req.get("cols") or [], wav, SR) if req.get("cols") else {}}
            if req.get("ref"):
                res["lpaps"] = be.lpaps(np.load(req["ref"])["wav"], wav, SR)
        except Exception as e:  # noqa: BLE001
            import traceback

            traceback.print_exc()
            res = {"id": req.get("id"), "error": f"{type(e).__name__}: {e}"}
        out.write(json.dumps(res, default=float) + "\n")
        if "torch" in sys.modules:
            sys.modules["torch"].cuda.empty_cache()
    return 0


class Worker:
    def __init__(self, python, tada_root, label_workers):
        cmd = [python, str(Path(__file__).resolve()), "serve", "--label-workers", str(label_workers)]
        if tada_root:
            cmd += ["--tada-root", tada_root]
        ck = Path(OUT) / "ckpt"
        env = dict(os.environ, PYTHONUTF8="1", PYTHONPATH=os.pathsep.join([str(REPO), str(HERE), str(Path(OUT) / "pylib")]),
                   CLAP_GENERAL_CKPT=str(ck / "630k-audioset-best.pt"), AUDIOSET_LABELS=str(ck / "class_labels_indices.csv"),
                   CLAP_MUSIC_CKPT="E:/Projects/tada-replication/steer-audio/res/clap/pretrained/"
                                   "music_audioset_epoch_15_esc_90.14.pt")
        for k in ("CLAP_GENERAL_CKPT", "AUDIOSET_LABELS", "CLAP_MUSIC_CKPT"):
            env[k] = os.environ.get(k, env[k])
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1, env=env)
        if not self.proc.stdout.readline():
            raise SystemExit("scorer worker failed to start")
        self.done, self.cv, self.n = {}, threading.Condition(), 0
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self):
        for line in self.proc.stdout:
            if line.strip():
                r = json.loads(line)
                with self.cv:
                    self.done[r["id"]] = r
                    self.cv.notify_all()
        with self.cv:
            self.done["__eof__"] = True
            self.cv.notify_all()

    def submit(self, **req):
        self.n += 1
        req["id"] = f"r{self.n}"
        self.proc.stdin.write(json.dumps(req) + "\n")
        self.proc.stdin.flush()
        return req["id"]

    def get(self, rid):
        with self.cv:
            while rid not in self.done:
                if "__eof__" in self.done:
                    raise SystemExit("scorer worker exited")
                self.cv.wait()
            r = self.done.pop(rid)
        if "error" in r:
            raise SystemExit(f"scorer error on {rid}: {r['error']}")
        return r

    def close(self):
        try:
            self.proc.stdin.write(json.dumps({"quit": True}) + "\n")
            self.proc.stdin.close()
            self.proc.wait(timeout=120)
        except Exception:  # noqa: BLE001
            self.proc.kill()


# ------------------------------------------------------------------------------------------------ run

def cmd_run(a):
    sys.path.insert(0, str(REPO))
    sys.path.insert(0, str(REPO / "scripts" / "tada"))
    import torch

    import sa3_tada_run as R
    from acestep.engine import sa3_tada

    torch.backends.cuda.matmul.allow_tf32 = True
    out = Path(a.out)
    plan = jl(out / "plan.json")
    prompts = json.loads((EVAL / "eval_sets.json").read_text())["screen_music"]
    seed = EVAL_SEED + a.seed_offset
    res = out / "res" / f"seed{a.seed_offset}"
    slots = out / "slots" / f"seed{a.seed_offset}"
    slots.mkdir(parents=True, exist_ok=True)
    phases = set(a.phases)
    shift_cache = {}

    def shift(name, s):
        if (name, s) not in shift_cache:
            bl, v = shift_of(out, name, s)
            shift_cache[(name, s)] = [(int(b), torch.from_numpy(x.astype(np.float32))) for b, x in zip(bl, v)]
        return shift_cache[(name, s)]

    _log(f"loading SA3 medium (seed {seed}, chunks of {a.chunk} prompts, cap {a.mem_gib} GiB)")
    sam = R._load_sam()
    torch.cuda.empty_cache()
    _log(f"model loaded: {torch.cuda.memory_allocated() / 2**30:.2f} GiB allocated, load peak "
         f"{torch.cuda.max_memory_reserved() / 2**30:.2f} GiB; capping at {a.mem_gib} GiB")
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.set_per_process_memory_fraction(a.mem_gib * 2**30 / torch.cuda.get_device_properties(0).total_memory)

    def render(members, scale):
        return np.concatenate([render1(members, scale, prompts[k:k + a.chunk])
                               for k in range(0, len(prompts), a.chunk)])

    def render1(members, scale, prompts):
        if members:
            per_block = {}
            for name, s, g in members:
                for b, x in shift(name, s):
                    per_block[b] = per_block.get(b, 0) + x * float(g * scale)
            vectors = {st: dict(per_block) for st in range(STEPS)}
            with sa3_tada.steer_offline(sam, vectors, 1.0, hook="post_block_residual"):
                audio = sa3_tada.generate(sam, prompts, seed=seed, duration=DURATION, steps=STEPS)
        else:
            audio = sa3_tada.generate(sam, prompts, seed=seed, duration=DURATION, steps=STEPS)
        torch.cuda.empty_cache()
        return (audio.mean(dim=1) * 32768.0).round().clamp(-32768, 32767).to(torch.int16).numpy()

    worker = Worker(a.eval_python, a.tada_root, a.label_workers)
    _log("scorer worker ready")
    base = slots / "base.npz"
    all_own = sorted({m["own"] for p in plan for m in p["members"]})
    if not (res / "base.json").exists():
        np.savez(base, wav=render([], 1.0))
        r = worker.get(worker.submit(npz=str(base), cols=all_own))
        jw(res / "base.json", {"cols": r["cols"], "seed": seed})
        _log(f"base done, render peak reserved {torch.cuda.max_memory_reserved() / 2**30:.1f} GB")

    # job list, in phase order
    jobs = []
    if "A" in phases:   # combo by combo (anytime-valid): the members' singles first, then the five rules
        seen = set()
        order = sorted((p for p in plan if p["phase"] == "A"), key=lambda p: SUBSET_ORDER.index(p["i"]))
        for p in order:
            for m in p["members"]:
                key = (m["name"], m["sign"])
                if key not in seen:
                    seen.add(key)
                    jobs.append((res / "single" / f"{m['name']}_{m['sign']:+d}.json",
                                 [(m["name"], m["sign"], m["gain"])], 1.0, [m["own"]],
                                 {"name": m["name"], "sign": m["sign"], "gain": m["gain"]}))
            for rule in RULES:
                jobs.append((res / "combo" / f"c{p['i']:03d}_{rule}.json",
                             [(m["name"], m["sign"], m["gain"]) for m in p["members"]], p["scale"][rule],
                             [m["own"] for m in p["members"]], {"i": p["i"], "rule": rule}))
    if "B" in phases:
        for p in plan:
            if p["phase"] == "B":
                for rule in a.b_rules:
                    jobs.append((res / "combo" / f"c{p['i']:03d}_{rule}.json",
                                 [(m["name"], m["sign"], m["gain"]) for m in p["members"]], p["scale"][rule],
                                 [m["own"] for m in p["members"]], {"i": p["i"], "rule": rule}))
    jobs = [j for j in jobs if not j[0].exists()]
    _log(f"{len(jobs)} renders to do")
    inflight, t0 = [], time.time()
    nslot = 3

    def drain(item):
        path, rid, meta, scale = item
        r = worker.get(rid)
        jw(path, {"meta": meta, "scale": scale, "lpaps": r["lpaps"], "cols": r["cols"]})

    for n, (path, members, scale, cols, meta) in enumerate(jobs):
        if len(inflight) >= nslot:
            drain(inflight.pop(0))
        sl = slots / f"slot{n % (nslot + 1)}.npz"
        np.savez(sl, wav=render(members, scale))
        rid = worker.submit(npz=str(sl), ref=str(base), cols=sorted(set(cols)))
        inflight.append((path, rid, meta, scale))
        if (n + 1) % 20 == 0:
            el = time.time() - t0
            _log(f"{n + 1}/{len(jobs)} renders, {el / (n + 1):.1f} s each, ETA {el / (n + 1) * (len(jobs) - n - 1) / 60:.0f} min, "
                 f"render peak reserved {torch.cuda.max_memory_reserved() / 2**30:.1f} GB")
    for it in inflight:
        drain(it)
    worker.close()
    _log("run done")
    return 0


# ------------------------------------------------------------------------------------------------ analyse

def cmd_analyse(a):
    out = Path(a.out)
    plan = {p["i"]: p for p in jl(out / "plan.json")}
    res_all = {}
    for sd in sorted((out / "res").glob("seed*")):
        base = jl(sd / "base.json")
        if not base:
            continue
        bc = {c: np.array(v, float) for c, v in base["cols"].items()}
        singles = {}
        for f in (sd / "single").glob("*.json"):
            j = jl(f)
            m = j["meta"]
            c = next(iter(j["cols"]))
            singles[(m["name"], m["sign"])] = {"delta": float(np.nanmean(np.array(j["cols"][c], float) - bc[c])),
                                               "lpaps": float(np.mean(j["lpaps"]))}
        combos = {}
        for f in (sd / "combo").glob("*.json"):
            j = jl(f)
            combos[(j["meta"]["i"], j["meta"]["rule"])] = j
        per_rule = {}
        for rule in RULES:
            rows = [(i, j) for (i, r), j in combos.items() if r == rule]
            under = {i: float(np.mean(j["lpaps"])) <= plan[i]["cutoff_min"] for i, j in rows}
            ratio = {i: float(np.mean(j["lpaps"])) / plan[i]["cutoff_min"] for i, j in rows}
            ret, ret_sign, ret_mem, pool_c, pool_s = [], [], {}, [], []
            for i, j in rows:
                if plan[i]["phase"] != "A":
                    continue
                for m in plan[i]["members"]:
                    s = singles.get((m["name"], m["sign"]))
                    if s is None:
                        continue
                    dc = float(np.nanmean(np.array(j["cols"][m["own"]], float) - bc[m["own"]]))
                    r = dc / s["delta"] if s["delta"] else float("nan")
                    ret.append(r)
                    pool_c.append(dc * np.sign(s["delta"]))
                    pool_s.append(abs(s["delta"]))
                    ret_sign.append(np.sign(dc) == np.sign(s["delta"]))
                    ret_mem.setdefault(f"{m['name']}{'+' if m['sign'] > 0 else '-'}", []).append(r)
            subA = [i for i, _ in rows if plan[i]["phase"] == "A"]
            # Measured-Sliders style: every requested knob's own label moves in its intended sign (= the knob sign;
            # vs base), per combo; and on the subset, every member keeps the sign its single render had
            all_int = [all(np.sign(np.nanmean(np.array(j["cols"][m["own"]], float) - bc[m["own"]])) == m["sign"]
                           for m in plan[i]["members"]) for i, j in rows]
            all_single = []
            for i, j in rows:
                ss = [singles.get((m["name"], m["sign"])) for m in plan[i]["members"]]
                if plan[i]["phase"] == "A" and all(ss):
                    all_single.append(all(np.sign(np.nanmean(np.array(j["cols"][m["own"]], float) - bc[m["own"]]))
                                          == np.sign(x["delta"]) for m, x in zip(plan[i]["members"], ss)))
            per_rule[rule] = {
                "n": len(rows), "frac_under": float(np.mean(list(under.values()))) if rows else None,
                "n_A": len(subA), "frac_under_A": float(np.mean([under[i] for i in subA])) if subA else None,
                "frac_under_pairs": _frac([under[i] for i, _ in rows if len(plan[i]["members"]) == 2]),
                "frac_under_triples": _frac([under[i] for i, _ in rows if len(plan[i]["members"]) == 3]),
                "median_lpaps_over_cut": float(np.median(list(ratio.values()))) if rows else None,
                "median_scale": float(np.median([plan[i]["scale"][rule] for i, _ in rows])) if rows else None,
                "retained_median": float(np.nanmedian(ret)) if ret else None,
                "retained_mean_clipped": float(np.nanmean(np.clip(ret, -1, 2))) if ret else None,
                "retained_n": len(ret),
                "retained_pooled": float(np.sum(pool_c) / np.sum(pool_s)) if pool_s else None,
                "retained_median_per_knob": float(np.nanmedian([np.nanmedian(v) for v in ret_mem.values()]))
                if ret_mem else None,
                "own_sign_kept": float(np.mean(ret_sign)) if ret_sign else None,
                "frac_all_intended": _frac(all_int), "frac_all_intended_A": _frac(
                    [x for (i, _), x in zip(rows, all_int) if plan[i]["phase"] == "A"]),
                "frac_all_as_single_A": _frac(all_single),
                "under": {str(i): bool(u) for i, u in under.items()}}
        # shipped-pass reproducibility on the subset (ship had none/inv_sqrt/inv_n/norm_budget at seed 2115)
        repro = {}
        if sd.name == "seed0":
            for rule in ("none", "inv_sqrt", "inv_n"):
                pairs = []
                for i in SUBSET:
                    sj = jl(SHIP / "d" / "stack" / f"c{i:03d}_{rule}.json")
                    if sj and (i, rule) in combos:
                        pairs.append((float(np.mean(sj["all"]["lpaps"])), float(np.mean(combos[(i, rule)]["lpaps"]))))
                if pairs:
                    x, y = np.array(pairs).T
                    repro[rule] = {"n": len(pairs), "median_rel_diff": float(np.median(np.abs(y - x) / x)),
                                   "corr": float(np.corrcoef(x, y)[0, 1]) if len(pairs) > 2 else None}
        single_ok = {f"{k[0]}{k[1]:+d}": v for k, v in singles.items()}
        res_all[sd.name] = {"rules": per_rule, "repro_vs_ship": repro, "n_singles": len(singles),
                            "singles_lpaps_over_cut_median": None, "singles": single_ok}
    ship = {}
    st = jl(SHIP / "product.json")
    if st and "stack" in st:
        ship = st["stack"]
    jw(out / "results.json", {"seeds": res_all, "ship_pass_stack": ship})
    write_md(out / "results.md", res_all, ship)
    print((out / "results.md").read_text())
    return 0


def _frac(x):
    return float(np.mean(x)) if x else None


def _f(x, nd=3):
    return "-" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{nd}f}"


def write_md(path, res_all, ship):
    L = ["# v3 smoke S3: stacking rules (shipped bundle, 97 knobs, shipped gains)", "",
         "Pass = mean LPAPS over the 12 screen_music prompts <= the strictest member's cutoff. Retained = own-label "
         "delta vs base in the combo / the same member alone at its shipped gain (phase A subset: 40 pairs + 20 "
         "triples). Rule definitions: stacking_rules.py docstring.", ""]
    for sd, r in res_all.items():
        L += [f"## {sd}", "", "| rule | combos | frac under cutoff | pairs | triples | subset A frac | median LPAPS/cut "
              "| median scale | retained median (n) | retained pooled | retained median per knob | retained mean (clip -1..2) | own sign kept "
              "| all knobs intended sign (all / subset) | all knobs as single (subset) |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for rule, x in r["rules"].items():
            L.append(f"| {rule} | {x['n']} | {_f(x['frac_under'])} | {_f(x['frac_under_pairs'])} | "
                     f"{_f(x['frac_under_triples'])} | {_f(x['frac_under_A'])} | {_f(x['median_lpaps_over_cut'])} | "
                     f"{_f(x['median_scale'])} | {_f(x['retained_median'])} ({x['retained_n']}) | {_f(x['retained_pooled'])} | "
                     f"{_f(x['retained_median_per_knob'])} | {_f(x['retained_mean_clipped'])} | {_f(x['own_sign_kept'])} | "
                     f"{_f(x['frac_all_intended'])} / {_f(x['frac_all_intended_A'])} | {_f(x['frac_all_as_single_A'])} |")
        if r["repro_vs_ship"]:
            L += ["", "Reproduction of the ship pass on the subset (same combos, seed 2115; ship box GPU vs local 5090): "
                  + "; ".join(f"{k} n={v['n']} median |rel diff| {_f(v['median_rel_diff'])} corr {_f(v['corr'])}"
                              for k, v in r["repro_vs_ship"].items())]
        L.append("")
    if ship:
        L += ["## Ship pass (2026-10-06, box, 300 combos, seed 2115) for reference", "",
              "| rule | frac under cutoff | median scale |", "|---|---|---|"]
        for k, v in ship.items():
            if isinstance(v, dict):
                L.append(f"| {k} | {_f(v.get('frac_under_cutoff'))} | {_f(v.get('median_scale'))} |")
        L.append("")
    Path(path).write_text("\n".join(L), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("export")
    e.add_argument("--bundle", default=str(BUNDLE))
    e.add_argument("--steer-repo", default=str(STEER_REPO))
    for name in ("export", "plan", "run", "analyse"):
        p = e if name == "export" else sub.add_parser(name)
        p.add_argument("--out", default=str(OUT))
    r = sub.choices["run"]
    r.add_argument("--phases", nargs="+", default=["A"])
    r.add_argument("--b-rules", nargs="+", default=["shared_l2", "shared_l1"])
    r.add_argument("--seed-offset", type=int, default=0)
    r.add_argument("--chunk", type=int, default=4, help="prompts per render batch (each chunk seeded alike)")
    r.add_argument("--mem-gib", type=float, default=6.0)
    r.add_argument("--eval-python", default="E:/Projects/tada-replication/evalenv/Scripts/python.exe")
    r.add_argument("--tada-root", default="E:/Projects/tada-replication")
    r.add_argument("--label-workers", type=int, default=8)
    s = sub.add_parser("serve")
    s.add_argument("--tada-root", default=None)
    s.add_argument("--label-workers", type=int, default=8)
    a = ap.parse_args()
    return {"export": cmd_export, "plan": cmd_plan, "run": cmd_run, "serve": cmd_serve, "analyse": cmd_analyse}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
