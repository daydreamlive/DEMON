"""v3 smoke batch S2: minimal-pair steering directions from the synthetic-corpus recipe grammar.

Runbook: notes/steering_pr/v3_smoke_batch_runbook.md (DEMON), section S2.

Grammar: instrument-controlnet ``scripts/build_synthetic_overnight_manifest.py`` (``make_row``, RHYTHMS, banks) and
the reviewed tables it hash-pins (``build_synthetic_bank_manifest.py``), imported READ-ONLY (no bytecode written).
A pair = one grammar recipe rendered twice, same seed, exactly one grammar field changed:

* tempo        ``scenario.tempo``: the slowest vs the fastest tempo of the recipe's own rhythm cell (fixed-meter cells),
               i.e. only the ``tempo: N <unit> per minute`` sentence differs. Scored by dyn.tempo and desc.onset_rate.
* drums        ``requested.instruments`` of a drum-free sparse / full ensemble, without vs with ``drum kit`` appended
               (the prompt's instrument list is the only change). Scored by desc.perc_ratio. Control: percussive.
* epiano       pitched-solo instrument name: ``acoustic grand piano`` vs ``tine electric piano`` (rows of both grammar
               entries; the other entry's register / technique text is kept, only the name swaps). Scored by CLAP
               cos("electric piano") - cos("piano") (anchors.json instruments). Control: electric_piano.

Side A (``side = 0``) is the first member (slow / no drums / acoustic piano), side B the second; direction = mean over
training pairs of (B - A) per block; pos knob = toward B.

Subcommands:
  prompts   write <out>/corpus/prompts.json (rows {id, prompt, seed, category, tags, pair, concept, side}; ids 2k, 2k+1)
  verify    per-sample seeding bit-identity check (GPU)
  fit       directions + held-out pair-separation AUC per block + packs (CPU)
  drive     render / score the minimal-pair packs vs controls (GPU)
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
for _p in (str(HERE), str(REPO)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

S2 = Path("E:/Projects/DEMON/steering-bench/v3_smoke/s2")
ICN = Path("C:/_dev/projects/instrument-controlnet")
N_PAIRS = 150
SEED0 = 7000
DRUM_FAMILIES = {"drum_kit", "frame_drums", "hand_percussion", "orchestral_percussion", "electronic_drums",
                 "body_percussion", "wood_percussion"}
CONCEPTS = {
    # concept: (side A, side B, control knob or None, primary score, secondary scores)
    "tempo": ("slowest tempo of the cell", "fastest tempo of the cell", None, "dyn.tempo", ["desc.onset_rate"]),
    "drums": ("no drum kit", "drum kit appended", "percussive", "desc.perc_ratio", ["desc.onset_rate"]),
    "epiano": ("acoustic grand piano", "tine electric piano", "electric_piano", "clap:electric piano|piano", []),
}


def _h(key: str) -> int:
    return int(hashlib.sha256(key.encode()).hexdigest()[:16], 16)


def load_grammar():
    sys.dont_write_bytecode = True
    path = ICN / "scripts" / "build_synthetic_overnight_manifest.py"
    spec = importlib.util.spec_from_file_location("s2_overnight_grammar", path)
    g = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(g)
    t = g.load_tables()            # hash-pinned reviewed tables
    return g, t, g.banks(t)


def _tempo_sentence(t, row, tempo):
    q = row["requested"]
    return t.tempo_sentence({"meter": q["meter"], "unit": q["tempo_beat_unit"], "tempo": tempo})


def _swap_once(text: str, old: str, new: str) -> str:
    if text.count(old) != 1:
        raise RuntimeError(f"expected exactly one {old!r} in {text!r}")
    return text.replace(old, new)


def pairs_tempo(g, t, banks, n):
    cand = []
    for material, entries in banks.items():
        for e in entries:
            for visit in range(96):
                cand.append((_h(f"s2:tempo:{material}:{e['slug']}:{visit}"), material, e, visit))
    cand.sort(key=lambda c: c[0])
    out, per = [], {}
    for _k, material, e, visit in cand:
        if len(out) >= n:
            break
        if per.get((material, e["slug"]), 0) >= 1:
            continue                                  # at most one recipe per instrument group (140 groups)...
        row = g.make_row(t, material, e, visit, {})
        q = row["requested"]
        if q["meter"] == "free" or q["tempo_bpm"] is None:
            continue
        rh = next(r for r in g.RHYTHMS if row["design_scenario"].startswith(r[0] + "-"))
        lo, hi = min(rh[5]), max(rh[5])
        old = _tempo_sentence(t, row, q["tempo_bpm"])
        a = _swap_once(row["prompt"], old, _tempo_sentence(t, row, lo))
        b = _swap_once(row["prompt"], old, _tempo_sentence(t, row, hi))
        per[(material, e["slug"])] = 1
        out.append(dict(recipe=row["recipe_id"], material=material, group=e["slug"], a=a, b=b,
                        field="scenario.tempo", va=lo, vb=hi, rhythm=rh[0]))
    if len(out) < n:                                  # ...then a second recipe for the first groups in hash order
        for _k, material, e, visit in cand:
            if len(out) >= n:
                break
            if per.get((material, e["slug"]), 0) >= 2:
                continue
            row = g.make_row(t, material, e, visit, {})
            q = row["requested"]
            if q["meter"] == "free" or q["tempo_bpm"] is None or any(o["recipe"] == row["recipe_id"] for o in out):
                continue
            rh = next(r for r in g.RHYTHMS if row["design_scenario"].startswith(r[0] + "-"))
            lo, hi = min(rh[5]), max(rh[5])
            old = _tempo_sentence(t, row, q["tempo_bpm"])
            a = _swap_once(row["prompt"], old, _tempo_sentence(t, row, lo))
            b = _swap_once(row["prompt"], old, _tempo_sentence(t, row, hi))
            per[(material, e["slug"])] = per.get((material, e["slug"]), 0) + 1
            out.append(dict(recipe=row["recipe_id"], material=material, group=e["slug"], a=a, b=b,
                            field="scenario.tempo", va=lo, vb=hi, rhythm=rh[0]))
    return out


def pairs_drums(g, t, banks, n):
    ents = [(m, e) for m in ("sparse_arrangement", "full_instrumental") for e in banks[m]
            if not DRUM_FAMILIES.intersection(e["families"])]
    cand = sorted(((_h(f"s2:drums:{m}:{e['slug']}:{v}"), m, e, v) for m, e in ents for v in range(96)),
                  key=lambda c: c[0])
    cap = -(-n // len(ents))
    out, per = [], {}
    for _k, m, e, v in cand:
        if len(out) >= n:
            break
        if per.get(e["slug"], 0) >= cap:
            continue
        row = g.make_row(t, m, e, v, {})
        name = e["name"]
        a = row["prompt"]
        b = _swap_once(a, f": {name}. Roles:", f": {name}, drum kit. Roles:")
        per[e["slug"]] = per.get(e["slug"], 0) + 1
        out.append(dict(recipe=row["recipe_id"], material=m, group=e["slug"], a=a, b=b,
                        field="requested.instruments", va=name, vb=name + ", drum kit"))
    return out


def pairs_epiano(g, t, banks, n):
    solo = {e["slug"]: e for e in banks["pitched_solo"]}
    pa, pe = solo["acoustic-piano"], solo["electric-piano"]
    A, B = pa["name"], pe["name"]
    cand = sorted(((_h(f"s2:epiano:{e['slug']}:{v}"), e, v) for e in (pa, pe) for v in range(96)), key=lambda c: c[0])
    out, per = [], {}
    for _k, e, v in cand:
        if len(out) >= n:
            break
        if per.get(e["slug"], 0) >= -(-n // 2):
            continue
        row = g.make_row(t, "pitched_solo", e, v, {})
        p = row["prompt"]
        if e is pa:
            a, b = p, _swap_once(p, f"Unaccompanied {A}.", f"Unaccompanied {B}.")
        else:
            a, b = _swap_once(p, f"Unaccompanied {B}.", f"Unaccompanied {A}."), p
        per[e["slug"]] = per.get(e["slug"], 0) + 1
        out.append(dict(recipe=row["recipe_id"], material="pitched_solo", group=e["slug"], a=a, b=b,
                        field="pitched_solo instrument name", va=A, vb=B))
    return out


def cmd_prompts(args) -> int:
    g, t, banks = load_grammar()
    out = Path(args.out) / "corpus"
    out.mkdir(parents=True, exist_ok=True)
    rows, pairs = [], []
    k = 0
    for concept, fn in (("tempo", pairs_tempo), ("drums", pairs_drums), ("epiano", pairs_epiano)):
        ps = fn(g, t, banks, args.n_pairs)
        if len(ps) != args.n_pairs:
            raise SystemExit(f"{concept}: only {len(ps)} pairs")
        for j, p in enumerate(ps):
            seed = SEED0 + k
            for side, text in ((0, p["a"]), (1, p["b"])):
                rows.append({"id": 2 * k + side, "prompt": text, "seed": seed, "category": f"s2_{concept}",
                             "tags": [concept, "B" if side else "A"], "pair": k, "concept": concept, "side": side,
                             "pair_in_concept": j, "recipe": p["recipe"], "material": p["material"],
                             "group": p["group"], "base_id": k})
            pairs.append({"pair": k, "concept": concept, **{x: p[x] for x in p if x not in ("a", "b")}})
            k += 1
    (out / "prompts.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")
    (out / "pairs.json").write_text(json.dumps(pairs, indent=1), encoding="utf-8")
    gp = ICN / "scripts" / "build_synthetic_overnight_manifest.py"
    src = {"grammar": str(gp), "grammar_sha256": hashlib.sha256(gp.read_bytes()).hexdigest(),
           "tables_sha256": g.TABLE_SHA, "n_pairs_per_concept": args.n_pairs, "seed_rule": f"pair k -> seed {SEED0} + k",
           "concepts": {c: list(v[:2]) for c, v in CONCEPTS.items()}}
    (out / "grammar_source.json").write_text(json.dumps(src, indent=1), encoding="utf-8")
    for c in CONCEPTS:
        ex = next(r for r in rows if r["concept"] == c)
        exb = rows[rows.index(ex) + 1]
        print(f"[{c}] e.g.\n  A: {ex['prompt']}\n  B: {exb['prompt']}")
    print(f"wrote {len(rows)} rows ({k} pairs) -> {out / 'prompts.json'}")
    return 0


def mem_cap(gib: float) -> None:
    """Runbook memory rule: hard cap this process's CUDA allocator (an over-cap allocation raises OOM)."""
    import torch

    if gib and gib > 0:
        tot = torch.cuda.get_device_properties(0).total_memory
        torch.cuda.set_per_process_memory_fraction(min(1.0, gib * 2 ** 30 / tot), 0)


def cmd_verify(args) -> int:
    """Bit-identity of per-sample seeding: one batch of ``--batch`` corpus rows (as capture_resid renders them) vs
    single-clip renders of two of its rows with the same seeds, each rendered twice; plus the noise itself."""
    import torch

    sys.path.insert(0, str(REPO / "scripts" / "sa3"))
    import capture_resid as CR
    from acestep.engine import sa3_tada
    from sa3_reference_generate import checkpoint_dir, load_local_model

    rows = json.loads((Path(args.out) / "corpus" / "prompts.json").read_text(encoding="utf-8"))[: args.batch]
    torch.backends.cuda.matmul.allow_tf32 = True
    sam = load_local_model(checkpoint_dir("medium"), device="cuda", model_half=True)
    sam.model.eval()
    torch.cuda.empty_cache()           # the fp32 load transient (~9 GB) is freed here; the cap applies after it
    mem_cap(args.mem_gib)
    res0 = {"weights_alloc_gib": round(torch.cuda.memory_allocated() / 2 ** 30, 2)}
    torch.cuda.reset_peak_memory_stats()

    def render(rs, seed_first):
        with torch.no_grad(), CR.PerRowNoise([r["seed"] for r in rs], "cuda") as pr:
            a = sa3_tada.generate(sam, [r["prompt"] for r in rs], seed=seed_first, duration=10.0, steps=8)
        return a

    # noise only: the first draw for the batch vs alone
    with CR.PerRowNoise([r["seed"] for r in rows], "cuda"):
        nb = torch.randn([len(rows), 64, 256], device="cuda")
    with CR.PerRowNoise([rows[1]["seed"]], "cuda"):
        n1 = torch.randn([1, 64, 256], device="cuda")
    noise_equal = bool(torch.equal(nb[1:2], n1)) and bool(torch.equal(nb[0], nb[1]))   # rows 0,1 = one pair
    t0 = time.time()
    batch = render(rows, rows[0]["seed"])
    tb = time.time() - t0
    peak_b = torch.cuda.max_memory_allocated() / 2 ** 30
    res = {**res0, "batch": len(rows), "noise_pair_equal_and_batch_vs_single_equal": noise_equal, "batch_s": round(tb, 2),
           "peak_alloc_gib_batch": round(peak_b, 2), "checks": []}
    for i in sorted({0, 1, len(rows) - 1}):
        s1 = render([rows[i]], 12345)       # the call-level seed is ignored under PerRowNoise
        s2 = render([rows[i]], 999)
        res["checks"].append({
            "row": i, "id": rows[i]["id"], "single_vs_single_bit_identical": bool(torch.equal(s1, s2)),
            "single_vs_batched_bit_identical": bool(torch.equal(s1[0], batch[i])),
            "single_vs_batched_max_abs": float((s1[0] - batch[i]).abs().max()),
            "single_vs_batched_corr": float(np.corrcoef(s1[0].flatten().numpy(), batch[i].flatten().numpy())[0, 1])})
    # the same batch twice
    batch2 = render(rows, 1)
    res["batch_vs_batch_bit_identical"] = bool(torch.equal(batch, batch2))
    # sub-batch of 2 (one pair) vs the full batch
    sub = render(rows[:2], 1)
    res["pair_batch2_vs_batched_bit_identical"] = bool(torch.equal(sub, batch[:2]))
    res["pair_batch2_vs_batched_max_abs"] = float((sub - batch[:2]).abs().max())
    res["peak_alloc_gib"] = round(torch.cuda.max_memory_allocated() / 2 ** 30, 2)
    res["peak_reserved_gib"] = round(torch.cuda.max_memory_reserved() / 2 ** 30, 2)
    p = Path(args.out) / "verify_seed.json"
    p.write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))
    return 0


# ----------------------------------------------------------------------------------------------------- fit (CPU)

def _auc(pos: np.ndarray, neg: np.ndarray) -> float:
    """P(pos > neg) over all pos x neg (ties half)."""
    d = pos[:, None] - neg[None, :]
    return float(((d > 0).sum() + 0.5 * (d == 0).sum()) / d.size)


def _unit(v):
    return v / max(float(np.linalg.norm(v)), 1e-12)


def _cos(a, b) -> float:
    a, b = np.asarray(a, np.float64).ravel(), np.asarray(b, np.float64).ravel()
    return float(a @ b / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-12))


def cmd_fit(args) -> int:
    import torch
    from acestep.steering.packs import SteeringPack, load_pack, save_pack

    t0 = time.time()
    cap = Path(args.out) / "corpus"
    meta = json.loads((cap / "meta.json").read_text())
    if not meta.get("complete"):
        raise SystemExit(f"{cap}: capture not complete")
    rows = json.loads((cap / "prompts.json").read_text(encoding="utf-8"))
    means = np.load(cap / "means.npy", mmap_mode="r")
    bid = list(meta["block_ids"])
    x = np.asarray(means, dtype=np.float32).mean(1)                      # [N, blocks, H] step-mean
    print(f"[fit] means {means.shape} -> step-mean {x.shape} ({time.time() - t0:.0f} s)", flush=True)
    ctrl_dir = Path(args.controls)
    packs = Path(args.out) / "packs"
    packs.mkdir(parents=True, exist_ok=True)
    rep = {"date": time.strftime("%Y-%m-%d"), "corpus": str(cap), "block_ids": bid,
           "feature": "mean over the 8 sampler steps of the post_block_residual audio-token mean",
           "split": "held-out = pair_in_concept % 3 == 0 (50 pairs), train = the other 100",
           "auc": "held-out unpaired AUC = P(proj(B) > proj(A)) over all held-out B x A clips, projection on the "
                  "train mean (B - A) direction at that block; paired = fraction of held-out pairs with proj(B-A) > 0",
           "consistency": "mean pairwise cosine between the individual pair-difference vectors (all 150 pairs, "
                          "off-diagonal), per block (lit review Q5)", "concepts": {}}
    for concept, (sa, sb, control, _p, _s) in CONCEPTS.items():
        rs = [r for r in rows if r["concept"] == concept]
        pairs = sorted({r["pair"] for r in rs})
        ia = np.array([next(r["id"] for r in rs if r["pair"] == p and r["side"] == 0) for p in pairs])
        ib = np.array([next(r["id"] for r in rs if r["pair"] == p and r["side"] == 1) for p in pairs])
        pic = np.array([next(r["pair_in_concept"] for r in rs if r["pair"] == p) for p in pairs])
        ho, tr = pic % 3 == 0, pic % 3 != 0
        per = {}
        for k, b in enumerate(bid):
            a, bb = x[ia, k], x[ib, k]
            d = bb - a
            u = _unit(d[tr].mean(0))
            pa, pb = a[ho] @ u, bb[ho] @ u
            pd = pb - pa
            dn = d / np.linalg.norm(d, axis=1, keepdims=True)
            g = dn @ dn.T
            n = len(dn)
            cons = float((g.sum() - np.trace(g)) / (n * (n - 1)))
            # unit-norm consistency of the train vs held-out halves (split-half direction agreement)
            per[b] = {"auc": round(_auc(pb, pa), 4), "paired": round(float((pd > 0).mean()), 4),
                      "dprime": round(float(pd.mean() / max(pd.std(ddof=1), 1e-12)), 3),
                      "consistency": round(cons, 4),
                      "split_half_cos": round(_cos(d[tr].mean(0), d[ho].mean(0)), 4),
                      "gap_over_std": round(float(pd.mean() / max(np.concatenate([pa, pb]).std(ddof=1), 1e-12)), 3)}
        best = max(bid, key=lambda b: (per[b]["auc"], per[b]["dprime"]))
        kb = bid.index(best)
        d_all = x[ib, kb] - x[ia, kb]
        u = _unit(d_all.mean(0))                                          # final vector: all 150 pairs
        gap = float((d_all @ u).mean())
        crep = {"side_a": sa, "side_b": sb, "n_pairs": len(pairs), "n_train": int(tr.sum()), "n_heldout": int(ho.sum()),
                "best_block": best, "block_rule": "max held-out AUC, ties by held-out d'", "per_block": per,
                "mean_pair_gap_along_u": round(gap, 4), "control": control}
        if control:
            cp = load_pack(ctrl_dir / f"{control}.safetensors")
            cvec = cp.vector.numpy().reshape(-1, cp.vector.shape[-1])[0]
            cb = int(cp.block)
            crep["control_block"] = cb
            crep["cos_control_at_best_block"] = round(_cos(u, cvec), 4)
            if cb in bid:
                uc = _unit((x[ib, bid.index(cb)] - x[ia, bid.index(cb)]).mean(0))
                crep["cos_control_at_control_block"] = round(_cos(uc, cvec), 4)
                crep["control_block_auc"] = per[cb]["auc"]
                crep["control_block_consistency"] = per[cb]["consistency"]
        name = f"pair_{concept}"
        prov = {"tool": "scripts/steering_bench/pairs_s2.py fit", "method": "minimal_pair", "date": rep["date"],
                "concept": concept, "side_a": sa, "side_b": sb, "sign": 1, "sign_rule": "positive knob = toward side B",
                "n_pairs": len(pairs), "blocks": [best], "block_choice": crep["block_rule"],
                "heldout_auc": per[best]["auc"], "consistency": per[best]["consistency"], "capture": str(cap),
                "feature": rep["feature"], "knob_unit": 0.1, "norm_rule": "mean pair gap (B - A) along the unit vector",
                "control": control, "calibrated_gain": None,
                "caveat": "estimated at 10 s ARC sam.generate from grammar minimal pairs; unscreened (v3 smoke S2)"}
        pk = SteeringPack(family="sa3", checkpoint="medium", block=int(best), hidden_size=int(u.shape[0]), blocks=[],
                          name=name, vector=torch.from_numpy(u.astype(np.float32)), label=f"{concept} (minimal pair)",
                          blurb=f"{sa} -> {sb}", hook="post_block_residual", method="caa_diff_means", norm=gap,
                          magnitude=gap * 0.1, policy={"kind": "range", "start": 0.0, "end": 1.0}, provenance=prov)
        p = save_pack(pk, packs / f"{name}.safetensors")
        back = load_pack(p)
        if back.block != best or _cos(back.vector.numpy(), u) < 0.99999:
            raise SystemExit(f"{p}: re-read mismatch")
        crep["pack"] = str(p)
        rep["concepts"][concept] = crep
        print(f"[fit] {concept}: best block {best} held-out AUC {per[best]['auc']} paired {per[best]['paired']} "
              f"d' {per[best]['dprime']} consistency {per[best]['consistency']} -> {p}", flush=True)
    (Path(args.out) / "fit_report.json").write_text(json.dumps(rep, indent=1))
    md = ["# S2 minimal-pair fit report", "", rep["split"] + "; " + rep["auc"] + ".", "Consistency: " + rep["consistency"] + ".", ""]
    for c, r in rep["concepts"].items():
        md += [f"## {c} ({r['side_a']} -> {r['side_b']}), best block {r['best_block']}", "",
               "| block | held-out AUC | paired | d' | consistency | split-half cos |", "|---|---|---|---|---|---|"]
        md += [f"| {b} | {v['auc']} | {v['paired']} | {v['dprime']} | {v['consistency']} | {v['split_half_cos']} |"
               for b, v in r["per_block"].items()]
        md += ["", "control: " + json.dumps({k: r[k] for k in r if k.startswith("cos_") or k.startswith("control")}), ""]
    (Path(args.out) / "fit_report.md").write_text("\n".join(md), encoding="utf-8")
    print(f"[fit] done in {time.time() - t0:.0f} s", flush=True)
    return 0


# ----------------------------------------------------------------------------------------------------- drive (GPU)

COLS = ["dyn.tempo", "desc.onset_rate", "desc.perc_ratio"]
CLAP_TEXTS = ["electric piano", "piano"]
FRACTIONS = (0.2, 0.4, 0.6, 0.8, 1.0)
PROBE_GRID = (2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0)
#: concept -> scores reported (primary first). "clap_ep" = cos(electric piano) - cos(piano).
SCORES = {"tempo": ["dyn.tempo", "desc.onset_rate"], "drums": ["desc.perc_ratio", "desc.onset_rate"],
          "epiano": ["clap_ep"]}
TEMPO_CUTOFF_FROM = "percussive"     # no shipped tempo knob: the PCI budget of the nearest rhythmic knob


def _scores(r: dict) -> dict:
    out = {c: r["cols"][c] for c in COLS}
    out["clap_ep"] = (np.asarray(r["clap"]["electric piano"]) - np.asarray(r["clap"]["piano"])).tolist()
    return out


class ChunkRenderer:
    """SA3 medium eager, set_tools / smoke_drive render path (pack vectors x alpha under steer_offline), but in chunks
    of ``batch`` clips under PerRowNoise so the noise of clip i is seeded by seeds[i] whatever the chunking."""

    def __init__(self, batch: int, mem_gib: float):
        sys.path.insert(0, str(REPO / "scripts" / "tada"))
        sys.path.insert(0, str(REPO / "scripts" / "sa3"))
        import torch

        import capture_resid as CR
        import sa3_tada_run as R
        from acestep.engine import sa3_tada

        torch.backends.cuda.matmul.allow_tf32 = True
        self.R, self.E, self.torch, self.CR, self.batch = R, sa3_tada, torch, CR, batch
        self.cache = {}
        self.sam = R._load_sam()
        torch.cuda.empty_cache()
        mem_cap(mem_gib)

    def render(self, prompts, seeds, pack, alpha) -> np.ndarray:
        out = []
        vectors = hook = None
        if pack is not None and alpha != 0.0:
            if pack not in self.cache:
                self.cache[pack] = (self.R._pack_vectors(pack, 8), self.R._PACK["hook"])
            pv, hook = self.cache[pack]
            vectors = {s: {b: v * float(alpha) for b, v in d.items()} for s, d in pv.items()}
        for i in range(0, len(prompts), self.batch):
            p, s = prompts[i:i + self.batch], seeds[i:i + self.batch]
            with self.torch.no_grad(), self.CR.PerRowNoise(s, "cuda"):
                if vectors is None:
                    a = self.E.generate(self.sam, p, seed=s[0], duration=10.0, steps=8)
                else:
                    with self.E.steer_offline(self.sam, vectors, 1.0, hook=hook):
                        a = self.E.generate(self.sam, p, seed=s[0], duration=10.0, steps=8)
            out.append((a.mean(dim=1) * 32768.0).round().clamp(-32768, 32767).to(self.torch.int16).numpy())
        return np.concatenate(out)


def _save(d: Path, wav: np.ndarray, prompts) -> Path:
    d.mkdir(parents=True, exist_ok=True)
    p = d / "audios.npz"
    np.savez(p, audio=wav[:, None, :], sr=np.int64(44100), names=np.array(list(prompts)))
    return p


def cmd_drive(args) -> int:
    import smoke_drive as SDR
    import ship_tools as ST
    from acestep.tada import concepts as C

    t0 = time.time()
    out = Path(args.out).resolve()
    R = Path(args.renders_dir).resolve() if args.renders_dir else out / "renders"
    fit = json.loads((out / "fit_report.json").read_text())
    vecs = {f"pair_{c}": {"path": r["pack"], "concept": c, "control": r["control"]} for c, r in fit["concepts"].items()}
    cargs = argparse.Namespace(bundle=SDR.BUNDLE, steer_repo=SDR.STEER_REPO,
                               controls=[r["control"] for r in fit["concepts"].values() if r["control"]] + [TEMPO_CUTOFF_FROM])
    cargs.controls = list(dict.fromkeys(cargs.controls))
    ctrl = SDR.load_controls(cargs, out)
    worker = SDR.Worker(args.eval_python, False, args.tada_root, 8)
    hold = C.benchmark_prompts(holdout=True)[0][:12]
    test = C.benchmark_prompts(holdout=False)[0][:12]
    base = hold + test
    prompts = base * args.n_seeds
    seeds = [2115 + 1000 * s + i for s in range(args.n_seeds) for i in range(len(base))]
    n0 = min(len(base), args.probe_n)                    # probe on the first probe_n seed-0 clips
    gen = ChunkRenderer(args.batch, args.mem_gib)
    nclips = 0

    def render_to(d, pack, alpha, ps, ss):
        nonlocal nclips
        f = d / f"alpha_{float(alpha)}" / "audios.npz"
        if f.exists():                                   # resume: same per-clip seeds and batch -> same audio
            return f
        nclips += len(ps)
        return _save(d / f"alpha_{float(alpha)}", gen.render(ps, ss, pack, alpha), ps)

    # corpus scoring (pair label effect) goes to the worker first; collected later
    ref_npz = render_to(R / "ref", None, 0.0, prompts, seeds)
    ref0_npz = _save(R / "ref_seed0", np.load(ref_npz)["audio"][:n0, 0], prompts[:n0])
    print(f"[drive] ref rendered ({len(prompts)} clips, {time.time() - t0:.0f} s)", flush=True)
    ref = _scores(worker.get(worker.submit(npz=str(ref_npz), ref=str(ref_npz), cols=COLS, anchors=CLAP_TEXTS, cand="")))
    print(f"[drive] ref rendered + scored ({len(prompts)} clips, {time.time() - t0:.0f} s)", flush=True)
    recs, pending = {}, []
    for vid in list(vecs) + list(ctrl):
        isc = vid in ctrl
        if isc and vid == TEMPO_CUTOFF_FROM and vid not in [v["control"] for v in vecs.values()]:
            continue
        pack = ctrl[vid]["path"] if isc else vecs[vid]["path"]
        cutsrc = vid if isc else (vecs[vid]["control"] or TEMPO_CUTOFF_FROM)
        vdir = R / vid
        probe = {0.0: [0.0] * n0}
        for s in (1, -1):
            side = SDR.SIDES[s]
            cut = ctrl[cutsrc]["side"][side]["cutoff"]
            if isc:
                a_c, src = ctrl[vid]["side"][side]["alpha_cut"], "calibrated gain x magnitude"
            else:
                for a in [g for g in PROBE_GRID if g >= args.probe_start]:
                    npz = render_to(vdir / "calib" / f"pack_pack_{vid}", pack, s * a, prompts[:n0], seeds[:n0])
                    lp = worker.get(worker.submit(npz=str(npz), ref=str(ref0_npz), cand=vid))["lpaps"]
                    probe[s * a] = lp
                    if float(np.mean(lp)) >= cut:
                        break
                SDR.write_lpaps_csv(vdir / "calib" / f"pack_pack_{vid}" / "protocol_results" / "lpaps.csv", probe)
                SDR.write_lpaps_csv(vdir / "eval" / f"pci_all_{vid}" / "protocol_results" / "lpaps.csv",
                                    {1.0: [ctrl[cutsrc]["side"]["pos"]["cutoff"]], -1.0: [ctrl[cutsrc]["side"]["neg"]["cutoff"]]})
                r = ST.interp_side(vdir, vid, s)
                a_c = r["alpha"]
                src = f"probe {'reached' if r['reached'] else 'NOT reached'} (seed 0, {n0} prompts), interp_side"
            alphas = [s * a_c * f for f in args.fractions]
            rec = {"vid": vid, "sign": s, "a_c": a_c, "a_c_source": src, "cutoff": cut, "cutoff_from": cutsrc,
                   "alphas": alphas, "lpaps": {}, "scores": {}}
            for a in alphas:
                npz = render_to(vdir / "eval" / f"pack_pack_{vid}", pack, a, prompts, seeds)
                pending.append((rec, a, worker.submit(npz=str(npz), ref=str(ref_npz), cols=COLS, anchors=CLAP_TEXTS, cand=vid)))
            recs[(vid, s)] = rec
            print(f"[drive] {vid} {side}: a_c {a_c:.3f} ({src}) cutoff {cut:.3f}; {nclips} clips {time.time() - t0:.0f} s",
                  flush=True)
    cap = out / "corpus"
    corpus_req = []  # after the renders: the probe must not queue behind it
    for f in sorted(cap.glob("audio_*.npz")):
        z = np.load(f)
        tmp = Path(args.tmp_dir) / "s2_corpus_score" / f.name
        tmp.parent.mkdir(parents=True, exist_ok=True)
        np.savez(tmp, audio=z["wav"][:, None, :], sr=np.int64(44100))
        corpus_req.append((z["ids"].tolist(), worker.submit(npz=str(tmp), ref=str(tmp), cols=COLS, anchors=CLAP_TEXTS, cand="")))
    for rec, a, rid in pending:
        r = worker.get(rid)
        rec["lpaps"][SDR._akey(a)] = r["lpaps"]
        rec["scores"][SDR._akey(a)] = _scores(r)
    corpus = {}
    for ids, rid in corpus_req:
        sc = _scores(worker.get(rid))
        for j, i in enumerate(ids):
            corpus[int(i)] = {k: v[j] for k, v in sc.items()}
    worker.close()
    res = analyse_drive(vecs, ctrl, recs, ref, corpus, out)
    res["meta"] = {"finished": time.strftime("%Y-%m-%d %H:%M:%S"), "seconds": round(time.time() - t0, 1),
                   "clips": nclips, "prompts": base, "n_seeds": args.n_seeds, "seed_rule": "clip seed 2115 + 1000 s + i "
                   "under PerRowNoise", "batch": args.batch, "mem_cap_gib": args.mem_gib,
                   "probe": f"seed 0, first {n0} prompts, grid {[g for g in PROBE_GRID if g >= args.probe_start]}",
                   "tempo_cutoff_from": TEMPO_CUTOFF_FROM, "renders": str(R), "fractions": list(args.fractions)}
    res["sweeps"] = {f"{k[0]}|{SDR.SIDES[k[1]]}": v for k, v in recs.items()}
    res["ref"] = ref
    (out / "results.json").write_text(json.dumps(res, indent=1, default=float))
    write_drive_md(out / "results.md", res, fit)
    print(f"[drive] done: {nclips} clips in {time.time() - t0:.0f} s -> {out / 'results.md'}", flush=True)
    return 0


def _metrics(rec, score, direction, refv):
    import smoke_drive as SDR

    alphas = rec["alphas"]
    vals = np.array([rec["scores"][SDR._akey(a)][score] for a in alphas], dtype=float)      # [5, n]
    lp = [float(np.mean(rec["lpaps"][SDR._akey(a)])) for a in alphas]
    rho = [direction * SDR.spearman(vals[:, i], np.abs(alphas)) for i in range(vals.shape[1])]
    rho_f = [r for r in rho if np.isfinite(r)]
    dl = [float(direction * np.nanmean(vals[k] - refv)) for k in range(len(alphas))]
    return {"rho_median": float(np.median(rho_f)) if rho_f else float("nan"), "lpaps": lp, "lpaps_max": max(lp),
            "lpaps_top": lp[-1], "delta": dl, "delta_top": dl[-1]}


def analyse_drive(vecs, ctrl, recs, ref, corpus, out) -> dict:
    import smoke_drive as SDR

    rows = []
    for vid, v in vecs.items():
        c, control = v["concept"], v["control"]
        for score in SCORES[c]:
            refv = np.asarray(ref[score], dtype=float)
            for direction in (1, -1):
                m = _metrics(recs[(vid, direction)], score, direction, refv)
                rec = recs[(vid, direction)]
                row = {"concept": c, "score": score, "direction": SIDES_TXT[direction], "vector": vid,
                       "alpha_top": rec["alphas"][-1], "a_c_source": rec["a_c_source"], "cutoff": rec["cutoff"],
                       "cutoff_from": rec["cutoff_from"], "rho_median": m["rho_median"],
                       "lpaps_top_over_cut": m["lpaps_top"] / rec["cutoff"], "delta_top": m["delta_top"], "metrics": m}
                if control:
                    cands = [(_metrics(recs[(control, s)], score, direction, refv), s) for s in (1, -1)]
                    cm, cs = max(cands, key=lambda x: x[0]["delta_top"])
                    lstar = min(rec["cutoff"], m["lpaps_max"], cm["lpaps_max"])
                    dr, dc = SDR.delta_at(m["lpaps"], m["delta"], lstar), SDR.delta_at(cm["lpaps"], cm["delta"], lstar)
                    row.update(control=control, control_sign=SDR.SIDES[cs], control_rho=cm["rho_median"],
                               control_lpaps_top_over_cut=cm["lpaps_top"] / ctrl[control]["side"][SDR.SIDES[cs]]["cutoff"],
                               control_delta_top=cm["delta_top"], matched_lpaps=lstar, delta_matched=dr,
                               control_delta_matched=dc, beats_control=bool(dr >= dc), control_metrics=cm)
                row["pass_rho"] = bool(np.isfinite(m["rho_median"]) and m["rho_median"] >= 0.9)
                row["pass_lpaps"] = bool(m["lpaps_top"] <= 1.1 * rec["cutoff"])
                rows.append(row)
    # pair label effect in the corpus: B - A per pair on each score
    rowsj = json.loads((out / "corpus" / "prompts.json").read_text(encoding="utf-8"))
    eff = {}
    for c in CONCEPTS:
        rs = [r for r in rowsj if r["concept"] == c]
        a = {r["pair"]: r["id"] for r in rs if r["side"] == 0}
        b = {r["pair"]: r["id"] for r in rs if r["side"] == 1}
        eff[c] = {}
        for score in ["dyn.tempo", "desc.onset_rate", "desc.perc_ratio", "clap_ep"]:
            d = np.array([corpus[b[p]][score] - corpus[a[p]][score] for p in a if a[p] in corpus and b[p] in corpus], float)
            d = d[np.isfinite(d)]
            eff[c][score] = {"n": int(len(d)), "mean": float(d.mean()) if len(d) else None,
                             "frac_pos": float((d > 0).mean()) if len(d) else None,
                             "median": float(np.median(d)) if len(d) else None}
    return {"rows": rows, "pair_label_effect": eff}


SIDES_TXT = {1: "pos (toward B)", -1: "neg (toward A)"}


def write_drive_md(path: Path, res: dict, fit: dict) -> None:
    def f(x, nd=3):
        return "nan" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{nd}f}"

    L = ["# S2 minimal-pair render results", "", f"{res['meta']['finished']}; {res['meta']['clips']} clips; 24 prompts "
         f"(12 holdout + 12 test) x {res['meta']['n_seeds']} seeds; alphas = sign x a_c x {res['meta']['fractions']}; a_c for "
         "pair vectors from the LPAPS probe to the PCI cutoff (tempo uses the percussive cutoff: no tempo knob), "
         "controls at calibrated gain x magnitude.", "",
         "| concept | score | direction | vector | Spearman med | LPAPS top/cut | delta top | matched LPAPS | delta @ matched | "
         "control (sign) | control Spearman | control delta @ matched | beats control |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in res["rows"]:
        L.append(f"| {r['concept']} | {r['score']} | {r['direction']} | {r['vector']} | {f(r['rho_median'])} | "
                 f"{f(r['lpaps_top_over_cut'])} | {f(r['delta_top'], 4)} | {f(r.get('matched_lpaps'))} | "
                 f"{f(r.get('delta_matched'), 4)} | {r.get('control', '-')} ({r.get('control_sign', '-')}) | "
                 f"{f(r.get('control_rho'))} | {f(r.get('control_delta_matched'), 4)} | {r.get('beats_control', '-')} |")
    L += ["", "Deltas are direction-signed (positive = moved the intended way), mean over clips vs alpha 0.", "",
          "## Pair label effect in the corpus (B - A per pair)", "", "| concept | score | n | mean | median | frac > 0 |",
          "|---|---|---|---|---|---|"]
    for c, e in res["pair_label_effect"].items():
        for s, v in e.items():
            L.append(f"| {c} | {s} | {v['n']} | {f(v['mean'], 4)} | {f(v['median'], 4)} | {f(v['frac_pos'])} |")
    L += ["", "## Fit (best block per concept)", "", "| concept | block | held-out AUC | paired | consistency |", "|---|---|---|---|---|"]
    for c, r in fit["concepts"].items():
        b = r["per_block"][str(r["best_block"])]
        L.append(f"| {c} | {r['best_block']} | {b['auc']} | {b['paired']} | {b['consistency']} |")
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


# ----------------------------------------------------------------------------------------------- independent rescoring

MUQ_TEXTS = ["electric piano", "acoustic piano", "fast tempo", "slow tempo", "drum kit", "solo instruments without drums"]
#: concept -> (fit-target-like scores from the drive, independent scores: rescore_independent.py q2 columns + MuQ)
INDEP = {
    "tempo": (["dyn.tempo", "desc.onset_rate"], ["onset_superflux", "tempo_bt", "clap.fast_tempo", "muq.fast_minus_slow"]),
    "drums": (["desc.perc_ratio"], ["onset_superflux", "clap.percussive", "muq.drums_minus_nodrums"]),
    "epiano": (["clap_ep"], ["muq.epiano_minus_piano"]),
}
INDEP_FIT_COLS = {"dyn.tempo", "desc.onset_rate", "desc.perc_ratio", "clap_ep"}


def cmd_indep_muq(args) -> int:
    """evalenv: MuQ-MuLan text similarity per clip (a different model from the drive's CLAP), template as
    screen_knobs ("This is a music of {t}"), for every npz rescore_independent.index finds."""
    import csv as _csv
    import os

    import rescore_independent as RI

    os.environ["TADA_ROOT"] = args.tada_root
    sys.path.insert(0, str(REPO / "scripts" / "tada"))
    import sa3_tada_score as S  # noqa: F401
    import torch

    import editing.eval as ev

    out = Path(args.out).resolve() / "independent"
    out.mkdir(parents=True, exist_ok=True)
    rows = RI.index(Path(args.renders_dir).resolve() if args.renders_dir else Path(args.out).resolve() / "renders")
    # editing.eval.get_mulan, but the model and the text embeddings loaded once (it reloads per call)
    mulan = ev.MuQMuLan.from_pretrained("OpenMuQ/MuQ-MuLan-large").to("cuda").eval()
    with torch.no_grad():
        te = mulan(texts=[f"This is a music of {t}" for t in MUQ_TEXTS])
    with open(out / "muq.csv", "w", newline="", encoding="utf-8") as fh, torch.no_grad():
        w = _csv.writer(fh)
        w.writerow(["vec", "sign", "alpha", "clip"] + MUQ_TEXTS)
        for n, r in enumerate(rows):
            wav, sr, _ = RI.load(r["npz"])
            rs = ev.Resample(sr, 24000)
            for i, c in enumerate(wav):
                x = rs(torch.from_numpy(c.astype(np.float32)[None, None] / 32768.0)).squeeze(1).to(mulan.device)
                sim = mulan.calc_similarity(mulan(wavs=x), te).cpu().numpy().reshape(-1)
                w.writerow([r["vec"], r["sign"], r["alpha"], i] + [float(v) for v in sim])
            fh.flush()
            print(f"muq {n + 1}/{len(rows)}", flush=True)
    return 0


def cmd_indep_sum(args) -> int:
    """Fit-target-like vs independent scores side by side, per concept / sign: delta at the top alpha
    (direction-signed), median per-clip Spearman over the 6 alphas incl. 0 (rescore_independent convention),
    and delta at matched LPAPS vs the control knob's best sign."""
    import smoke_drive as SDR
    import rescore_independent as RI

    out = Path(args.out).resolve()
    ind = out / "independent"
    clips = RI._read(ind / "q2_clips.csv")
    anchors = json.loads((HERE / "anchors.json").read_text(encoding="utf-8"))["concepts"]
    if (ind / "q2_clap.csv").exists():
        for k, v in RI._read(ind / "q2_clap.csv").items():
            for c in RI.CLAP_PAIRS:
                pos, neg = anchors[c][:2]
                clips.setdefault(k, {})[f"clap.{c}"] = v[pos] - v[neg]
    if (ind / "muq.csv").exists():
        for k, v in RI._read(ind / "muq.csv").items():
            clips.setdefault(k, {}).update({
                "muq.epiano_minus_piano": v["electric piano"] - v["acoustic piano"],
                "muq.fast_minus_slow": v["fast tempo"] - v["slow tempo"],
                "muq.drums_minus_nodrums": v["drum kit"] - v["solo instruments without drums"]})
    res = json.loads((out / "results.json").read_text())
    sw = res["sweeps"]
    ncl = 1 + max(k[3] for k in clips)
    ref_drive = res["ref"]

    def series(vid, sign, m):
        rec = sw[f"{vid}|{SDR.SIDES[sign]}"]
        alphas = rec["alphas"]
        if m in INDEP_FIT_COLS:
            ref = np.asarray(ref_drive[m], float)
            mat = np.array([rec["scores"][SDR._akey(a)][m] for a in alphas], float)
        else:
            ref = np.array([clips[("ref", 0, 0.0, i)][m] for i in range(ncl)])
            mat = np.array([[clips[(vid, sign, float(a), i)][m] for i in range(ncl)] for a in alphas])
        lp = [float(np.mean(rec["lpaps"][SDR._akey(a)])) for a in alphas]
        return alphas, ref, mat, lp, rec["cutoff"]

    def summ(vid, sign, m, direction):
        alphas, ref, mat, lp, cut = series(vid, sign, m)
        d = direction * (mat - ref[None])
        xs = np.array([0.0] + [abs(x) for x in alphas])
        rho = [direction * RI._spearman(xs, np.concatenate([[ref[i]], mat[:, i]])) for i in range(mat.shape[1])]
        return {"delta_top": float(np.nanmean(d[-1])), "rho_med": float(np.nanmedian(rho)),
                "frac_top": float(np.mean(d[-1] > 0)), "lp": lp, "dcurve": np.nanmean(d, axis=1).tolist(), "cut": cut}

    fit = json.loads((out / "fit_report.json").read_text())
    table = []
    for c, (fcols, icols) in INDEP.items():
        vid, control = f"pair_{c}", fit["concepts"][c]["control"]
        for direction in (1, -1):
            for m in fcols + icols:
                s = summ(vid, direction, m, direction)
                row = {"concept": c, "direction": SIDES_TXT[direction], "score": m,
                       "kind": "fit-target" if m in fcols else "independent", "delta_top": s["delta_top"],
                       "rho_med": s["rho_med"], "frac_top": s["frac_top"], "lpaps_top_over_cut": s["lp"][-1] / s["cut"]}
                if control:
                    cands = [(summ(control, cs, m, direction), cs) for cs in (1, -1)]
                    cm, cs = max(cands, key=lambda x: x[0]["delta_top"])
                    lstar = min(s["cut"], max(s["lp"]), max(cm["lp"]))
                    row.update(control=control, control_sign=SDR.SIDES[cs], control_delta_top=cm["delta_top"],
                               control_rho_med=cm["rho_med"], matched_lpaps=lstar,
                               delta_matched=SDR.delta_at(s["lp"], s["dcurve"], lstar),
                               control_delta_matched=SDR.delta_at(cm["lp"], cm["dcurve"], lstar))
                    row["beats_control"] = bool(row["delta_matched"] >= row["control_delta_matched"])
                table.append(row)
    (out / "independent.json").write_text(json.dumps(table, indent=1, default=float))

    def f(x, nd=3):
        return "" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:+.{nd}g}"

    L = ["# S2 fit-target vs independent scores", "",
         "Direction-signed (positive = moved toward the intended side). delta top = mean over the clips at the top "
         "alpha vs alpha 0; rho = median per-clip Spearman over the 6 alphas incl. 0; frac = share of clips moved the "
         "right way at the top alpha; delta @ matched = at matched LPAPS (interpolated) vs the control knob's best sign.",
         "Fit-target-like = the scorer each concept was chosen by (dyn.tempo / onset_rate, perc_ratio, CLAP e-piano); "
         "independent = rescore_independent.py q2 (superflux onsets, librosa beat_track tempo, CLAP anchors.json "
         "concepts) and MuQ-MuLan text similarity (a different model).", "",
         "| concept | direction | kind | score | delta top | rho | frac | LPAPS/cut | delta @ matched | control (sign) delta @ matched | beats control |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in table:
        L.append(f"| {r['concept']} | {r['direction']} | {r['kind']} | {r['score']} | {f(r['delta_top'])} | "
                 f"{f(r['rho_med'], 2)} | {r['frac_top']:.2f} | {r['lpaps_top_over_cut']:.2f} | {f(r.get('delta_matched'))} | "
                 f"{r.get('control', '-')} ({r.get('control_sign', '-')}) {f(r.get('control_delta_matched'))} | "
                 f"{r.get('beats_control', '-')} |")
    (out / "independent.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"wrote {out / 'independent.md'}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eval-python", default="E:/Projects/tada-replication/evalenv/Scripts/python.exe")
    ap.add_argument("--tada-root", default="E:/Projects/tada-replication")
    ap.add_argument("--n-seeds", type=int, default=2)
    ap.add_argument("--renders-dir", default=None, help="drive / indep-muq: render tree (default <out>/renders)")
    ap.add_argument("--tmp-dir", default="C:/Users/ryanf/AppData/Local/Temp/claude/C---dev-projects-DEMON-notes-steering-pr/"
                    "a4a78375-a437-4f1c-bcdd-7d7cbe1a990e/scratchpad", help="drive: temp copies for corpus scoring (off E:)")
    ap.add_argument("--fractions", type=float, nargs="*", default=list(FRACTIONS), help="drive: sweep alphas / a_c")
    ap.add_argument("--probe-n", type=int, default=12, help="drive: seed-0 clips used by the LPAPS probe")
    ap.add_argument("--probe-start", type=float, default=8.0, help="drive: smallest |alpha| on the probe grid")
    ap.add_argument("cmd", choices=["prompts", "verify", "fit", "drive", "indep-muq", "indep-sum"])
    ap.add_argument("--controls", default="E:/Projects/DEMON/steering-bench/many_knobs_v2/packs_final/sa3/medium")
    ap.add_argument("--out", default=str(S2))
    ap.add_argument("--n-pairs", type=int, default=N_PAIRS)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--mem-gib", type=float, default=6.0, help="CUDA allocator cap for this process (0 = none)")
    args, _rest = ap.parse_known_args()
    if args.cmd == "prompts":
        return cmd_prompts(args)
    if args.cmd == "verify":
        return cmd_verify(args)
    if args.cmd == "fit":
        return cmd_fit(args)
    if args.cmd == "drive":
        return cmd_drive(args)
    if args.cmd == "indep-muq":
        return cmd_indep_muq(args)
    if args.cmd == "indep-sum":
        return cmd_indep_sum(args)
    raise SystemExit(f"{args.cmd}: not implemented yet")


if __name__ == "__main__":
    raise SystemExit(main())
