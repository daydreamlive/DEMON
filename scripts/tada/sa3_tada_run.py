"""TADA on Stable Audio 3 medium: the generation side of the replication.

TADA (Staniszewski et al., arXiv 2602.11910; reference code
github.com/luk-st/steer-audio, MIT). This driver renders every audio set
the paper's protocol needs, on SA3 medium (eager, offline; TADA's Stable
Audio Open settings where they transfer, SA3's own sampler where they do
not: 8 steps, ``cfg_scale=1``, so "conditional pass only" is every row):

* ``patch``: the activation-patching sweep for layer localisation
  (Sec. 3-4). Per concept, prompt pair and seed: one clean run records
  every block's cross-attention conditioning, then the corrupted prompt
  is rendered ``L + 2`` times (none patched, all patched, each block
  alone). Audio: ``<out>/patch/<concept>/<set>/audios.npz``.
* ``caa``: contrastive activation addition vectors (Eq. 6) for the nine
  benchmark concepts: 50 prompt pairs, seed 10, cross-attention outputs
  averaged over rows and time, per step and block, unit norm.
  ``<out>/caa/<concept>.pt``.
* ``pci``: the prompt-swap baseline that sets the LPAPS cutoff (PCI-all
  swaps the prompt at every block for the last ``k`` steps; PCI-loc only
  at the localised blocks). ``<out>/eval/pci_{all,loc}_<concept>/``.
* ``sweep``: CAA strength sweeps on the benchmark prompts for one site
  (``all``, ``loc``, ``ablated``). ``<out>/eval/caa_<site>_<concept>/``.
* ``calibrate``: largest strength per site whose mean LPAPS on the
  held-out prompts reaches the PCI cutoff (the paper calibrates ranges to
  the prompt-swap distortion), via the reference LPAPS (``--score-python``).

Eval directories follow the reference layout (``alpha_<a>/audios.npz``),
so the reference scoring (``eval_steering_protocol.py``, ``auc.py``)
runs on them unchanged; see ``scripts/tada/sa3_tada_score.py``.

Run with the GPU lock held, no server:
    .venv/Scripts/python.exe scripts/tada/sa3_tada_run.py patch --pairs 34 --seeds 4
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
for _p in (str(REPO), str(REPO / "scripts" / "sa3")):
    while _p in sys.path:
        sys.path.remove(_p)
sys.path.insert(0, str(REPO / "scripts" / "sa3"))
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from acestep.engine import sa3_tada  # noqa: E402
from acestep.tada import ActivationRecorder, ConditioningPatcher, forward_counter  # noqa: E402
from acestep.tada import concepts as C  # noqa: E402
from acestep.tada.caa import caa_vectors  # noqa: E402
from acestep.tada.patching import ALL, NONE, run_sets  # noqa: E402
from acestep.tada.target import HOOK_CROSS_ATTN_COND  # noqa: E402

DEFAULT_OUT = Path("E:/Projects/tada-replication/sa3")
DEFAULT_PROMPTS_CSV = Path("E:/Projects/tada-replication/data/patching_prompts.csv")
STEPS = 8
DURATION = 10.0
SA3_SR = 44100

#: Paper Fig. 8 Stable Audio Open localisation concepts. ``guitar`` and
#: ``techno`` have no patch data or prompt pairs in the reference release
#: (features.py has 21 features, none of them), so they cannot be rebuilt.
SAO_LOC_CONCEPTS = ("female", "male", "fast", "slow", "happy", "sad",
                    "violin", "flute", "maracas", "reggae")
#: Reference ``configs/patch_config/stableaudio.yaml`` seed.
PATCH_SEED = 222
#: Reference CAA compute seed and eval seed.
CAA_SEED = 10
EVAL_SEED = 2115


def _load_sam():
    from sa3_reference_generate import checkpoint_dir, load_local_model

    sam = load_local_model(checkpoint_dir("medium"), device="cuda", model_half=True)
    sam.model.eval()
    return sam


def _save_npz(directory: Path, audio: torch.Tensor, names, *, mono: bool) -> None:
    """Reference ``audio_io.save_alpha_audios`` layout (int16 PCM)."""
    directory.mkdir(parents=True, exist_ok=True)
    a = audio.mean(dim=1, keepdim=True) if mono else audio
    arr = (a.float() * 32768.0).round().clamp(-32768, 32767).to(torch.int16).numpy()
    np.savez(directory / "audios.npz", audio=arr, sr=np.int64(SA3_SR),
             names=np.array(list(names)))


def _batches(n: int, b: int):
    for i in range(0, n, b):
        yield range(i, min(n, i + b))


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# patch
# ---------------------------------------------------------------------------

def _patch_pairs(concept: str, csv_path: Path, limit: int):
    import pandas as pd

    d = C.load("localization")["patch_data"][concept]
    df = pd.read_csv(csv_path)
    df = df[df["original_feature"] == d["feature"]]
    clean = [f"{d['prefix_clean']}{p}{d['suffix_clean']}" for p in df["clean_prompt"]]
    corr = [f"{d['prefix_corrupted']}{p}{d['suffix_corrupted']}" for p in df["corrupted_prompt"]]
    return clean[:limit], corr[:limit], d


def cmd_patch(args, sam) -> None:
    target = sa3_tada.sa3_target(sam)
    nb = target.num_blocks
    sets = run_sets(nb)
    for concept in args.concepts:
        root = args.out / "patch" / concept
        if (root / "done.json").exists() and not args.force:
            _log(f"patch {concept}: done, skipping")
            continue
        clean, corr, d = _patch_pairs(concept, args.prompts_csv, args.pairs)
        n = len(clean)
        audio = {key: [] for key, _ in sets}
        names = []
        t0 = time.time()
        for k in range(args.seeds):
            seed = PATCH_SEED + k
            for idx in _batches(n, args.batch):
                cp = [clean[i] for i in idx]
                xp = [corr[i] for i in idx]
                rec = ConditioningPatcher(target, list(range(nb)), hook=HOOK_CROSS_ATTN_COND)
                with rec.record():
                    sa3_tada.generate(sam, cp, seed=seed, duration=DURATION, steps=STEPS)
                for key, blocks in sets:
                    if not blocks:
                        out = sa3_tada.generate(sam, xp, seed=seed, duration=DURATION, steps=STEPS)
                    else:
                        p = ConditioningPatcher(target, list(blocks), hook=HOOK_CROSS_ATTN_COND)
                        p.cache = {b: list(rec.cache[b]) for b in blocks}
                        with p.patch():
                            out = sa3_tada.generate(sam, xp, seed=seed, duration=DURATION, steps=STEPS)
                        if p.leftover:
                            raise RuntimeError(f"patch misaligned: {p.leftover}")
                    audio[key].append(out)
                names += [f"p{i:03d}_s{k}.wav" for i in idx]
            _log(f"patch {concept}: seed {k + 1}/{args.seeds} ({n} pairs) {time.time() - t0:.0f}s")
        for key, _ in sets:
            sub = key if key in (NONE, ALL) else f"tf{key}"
            _save_npz(root / sub, torch.cat(audio[key]), names, mono=True)
        (root / "done.json").write_text(json.dumps({
            "concept": concept, "pairs": n, "seeds": args.seeds, "seed0": PATCH_SEED,
            "steps": STEPS, "duration": DURATION, "cfg_scale": 1.0,
            "eval_clap_prompts": d["eval_clap_prompts"],
            "eval_muqt_prompts": d["eval_muqt_prompts"],
            "example_pair": [clean[0], corr[0]],
            "seconds": round(time.time() - t0, 1),
        }, indent=2))
        _log(f"patch {concept}: saved {len(names)} x {len(sets)} sets")


# ---------------------------------------------------------------------------
# caa
# ---------------------------------------------------------------------------

def cmd_caa(args, sam) -> None:
    target = sa3_tada.sa3_target(sam)
    (args.out / "caa").mkdir(parents=True, exist_ok=True)
    for concept in args.concepts:
        path = args.out / "caa" / f"{concept}.pt"
        if path.exists() and not args.force:
            _log(f"caa {concept}: done, skipping")
            continue
        pos, neg, _lyrics = C.prompt_pairs(concept)
        n = min(len(pos), args.pairs)
        pos_runs, neg_runs = [], []
        for idx in _batches(n, args.batch):
            for prompts, runs in (([pos[i] for i in idx], pos_runs), ([neg[i] for i in idx], neg_runs)):
                with ActivationRecorder(target, context=_PerRow()) as rec:
                    sa3_tada.generate(sam, prompts, seed=CAA_SEED, duration=DURATION, steps=STEPS)
                st = rec.steps()
                # One run per prompt: the mean over prompts is the reference
                # mean, and the per-prompt activations stay available for
                # other localized methods (AUSteer) without re-rendering.
                for r in range(len(prompts)):
                    runs.append({s_: {b: [calls[r]] for b, calls in per.items()} for s_, per in st.items()})
        acts = {
            pole: torch.stack([torch.stack([torch.stack([run[s_][b][0] for b in sorted(run[s_])])
                                            for s_ in sorted(run)]) for run in runs]).half()
            for pole, runs in (("pos", pos_runs), ("neg", neg_runs))
        }
        torch.save({"acts": acts, "layout": "[prompt, step, block, hidden] time-mean cross-attn output, fp16",
                    "positive": pos[:n], "negative": neg[:n], "seed": CAA_SEED},
                   args.out / "caa" / f"{concept}.acts.pt")
        vec = caa_vectors(pos_runs, neg_runs, normalize=True)
        raw = caa_vectors(pos_runs, neg_runs, normalize=False)
        torch.save({
            "vectors": vec,
            "raw_norm": {s: {b: float(v.norm()) for b, v in per.items()} for s, per in raw.items()},
            "concept": concept, "pairs": n, "seed": CAA_SEED, "steps": STEPS,
            "duration": DURATION, "example": [pos[0], neg[0]],
        }, path)
        _log(f"caa {concept}: {n} pairs, steps {sorted(vec)} blocks {len(vec[0])}")


def cmd_austeer(args, sam) -> None:
    """AUSteer vectors (paper Eq. 7, core :mod:`acestep.tada.austeer`)
    from the same 50 prompt pairs and seed as CAA, every (pair, frame) a
    sample. Momentum signs are counted batch by batch (pair rows aligned:
    same seed and length), which equals pooling every sample at once;
    the core ``select_top_s`` then keeps the global top-s per step over
    each site's blocks."""
    from acestep.tada.austeer import select_top_s
    from acestep.tada.target import frames

    target = sa3_tada.sa3_target(sam)
    nb = target.num_blocks
    for concept in args.concepts:
        path = args.out / "caa" / f"{concept}.austeer.pt"
        if path.exists() and not args.force:
            _log(f"austeer {concept}: done, skipping")
            continue
        pos, neg, _ = C.prompt_pairs(concept)
        n = min(len(pos), args.pairs)
        cnt_p, cnt_n, total = {}, {}, {}
        for idx in _batches(n, args.batch):
            stores = []
            for prompts in ([pos[i] for i in idx], [neg[i] for i in idx]):
                with ActivationRecorder(target, context=forward_counter(1), reduce=frames) as rec:
                    sa3_tada.generate(sam, prompts, seed=CAA_SEED, duration=DURATION, steps=STEPS)
                stores.append(rec.steps())
            sp, sn = stores
            for st in sp:
                for b in sp[st]:
                    m = sp[st][b][0] - sn[st][b][0]
                    key = (st, b)
                    cnt_p[key] = cnt_p.get(key, 0) + (m > 0).sum(dim=0)
                    cnt_n[key] = cnt_n.get(key, 0) + (m < 0).sum(dim=0)
                    total[key] = total.get(key, 0) + m.shape[0]
            del stores, sp, sn
        betas = {}
        for (st, b), cp in cnt_p.items():
            rp = cp.float() / total[(st, b)]
            rn = cnt_n[(st, b)].float() / total[(st, b)]
            sc = torch.maximum(rp, rn)
            betas.setdefault(st, {})[b] = torch.where(rp >= rn, sc, -sc)
        vecs = {
            "loc": select_top_s(betas, args.top_s_loc, list(args.loc)),
            "all": select_top_s(betas, args.top_s_all, list(range(nb))),
        }
        torch.save({"betas": betas, "vectors": vecs, "top_s": {"loc": args.top_s_loc, "all": args.top_s_all},
                    "loc": list(args.loc), "pairs": n, "seed": CAA_SEED, "samples": total[(0, 0)]}, path)
        kept = {site: sorted({b for per in v.values() for b, x in per.items() if x.abs().sum() > 0})
                for site, v in vecs.items()}
        _log(f"austeer {concept}: {n} pairs, {total[(0, 0)]} samples/step-block, blocks kept {kept}")


class _PerRow:
    """Recorder context: one forward per sampler step (``cfg_scale=1``),
    every batch row recorded as its own call (row order kept)."""

    def __init__(self):
        self.forwards = -1

    def tick(self) -> None:
        self.forwards += 1

    def __call__(self, batch: int):
        k = max(0, self.forwards)
        return [(slice(i, i + 1), k) for i in range(int(batch))]


# ---------------------------------------------------------------------------
# eval sets (pci, sweep)
# ---------------------------------------------------------------------------

def _site_blocks(site: str, loc, nb: int):
    if site == "all":
        return list(range(nb))
    if site == "loc":
        return list(loc)
    if site == "ablated":
        return [b for b in range(nb) if b not in set(loc)]
    raise ValueError(site)


def _eval_prompts(args):
    prompts, _ = C.benchmark_prompts(holdout=args.holdout)
    return prompts[: args.n_prompts]


def _fmt_alpha(a: float) -> str:
    return f"alpha_{float(a)}"


def cmd_pci(args, sam) -> None:
    """Reference PCI (``pci/core.py``): neutral prompt for the first
    ``N - k`` steps, the concept's positive prompt for the last ``k``
    (negative prompt for ``-k``). PCI-loc swaps only at the localised
    blocks' cross-attention (the same conditioning substitution as
    patching, applied from the switch step on)."""
    from acestep.tada.target import ConditioningPatcher as CP

    target = sa3_tada.sa3_target(sam)
    nb = target.num_blocks
    tests = _eval_prompts(args)
    # Both directions, as the reference PCI cells (cutoff per direction).
    ks = list(range(-STEPS, STEPS + 1))
    for concept in args.concepts:
        triples = [_pci_triple(p, concept) for p in tests]
        for site in ("all", "loc"):
            root = args.out / args.eval_sub / f"pci_{site}_{concept}{args.suffix}"
            blocks = _site_blocks(site, args.loc, nb)
            for k in ks:
                d = root / _fmt_alpha(k)
                if (d / "audios.npz").exists() and not args.force:
                    continue
                outs = []
                for idx in _batches(len(tests), args.batch):
                    neutral = [triples[i][0] for i in idx]
                    swap = [triples[i][1 if k > 0 else 2] for i in idx]
                    if k == 0:
                        outs.append(sa3_tada.generate(sam, neutral, seed=EVAL_SEED, duration=DURATION, steps=STEPS))
                        continue
                    rec = CP(target, blocks, hook=HOOK_CROSS_ATTN_COND)
                    with rec.record():
                        sa3_tada.generate(sam, swap, seed=EVAL_SEED, duration=DURATION, steps=STEPS)
                    switch = STEPS - abs(k)
                    # Calls before the switch step keep the neutral context:
                    # patch only from call ``switch`` on.
                    p = _SwitchPatcher(target, blocks, rec.cache, switch)
                    with p.patch():
                        outs.append(sa3_tada.generate(sam, neutral, seed=EVAL_SEED, duration=DURATION, steps=STEPS))
                _save_npz(d, torch.cat(outs), [f"p{i:03d}.wav" for i in range(len(tests))], mono=True)
            _log(f"pci {site} {concept}: {len(ks)} strengths x {len(tests)} prompts")


def _pci_triple(p: str, concept: str):
    """Reference ``build_prompt_triple``: (neutral, positive, negative)."""
    table = {
        "piano": (f"{p}, with instrument", f"{p}, with piano", f"{p}"),
        "mood": (f"a song, {p}", f"happy song, {p}", f"sad song, {p}"),
        "tempo": (f"a song, {p}", f"fast song, {p}", f"slow song, {p}"),
        "vocal_gender": (f"{p}, with clean vocal", f"{p}, with female vocal", f"{p}, with male vocal"),
        "vocal_style": (f"{p}, with clean vocal", f"{p}, with rap vocal", f"{p}, with sing vocal"),
        "guitar_electronic": (f"{p}, with a guitar", f"{p}, with acoustic guitar", f"{p}, with electric guitar"),
        "violin": (f"{p}, with instrument", f"{p}, with violin", f"{p}"),
        "rock_genre": (f"a song, {p}", f"jazz song, {p}", f"rock song, {p}"),
        "electronic_music": (f"a song, {p}", f"classical song, {p}", f"electronic song, {p}"),
    }
    return table[concept]


class _SwitchPatcher:
    """Conditioning substitution from forward ``switch`` on (PCI)."""

    def __init__(self, target, blocks, cache, switch: int):
        self.target, self.blocks, self.cache, self.switch = target, list(blocks), cache, int(switch)

    def patch(self):
        from contextlib import contextmanager

        @contextmanager
        def ctx():
            mods = self.target.modules(HOOK_CROSS_ATTN_COND)
            keys = tuple(self.target.patch_keys(HOOK_CROSS_ATTN_COND))
            calls = {b: 0 for b in self.blocks}
            handles = []

            def make(b):
                def pre(_m, a, kw):
                    i = calls[b]
                    calls[b] += 1
                    if i < self.switch:
                        return None
                    saved = self.cache[b][i]
                    for k in keys:
                        if saved.get(k) is not None:
                            kw[k] = saved[k]
                    return a, kw
                return pre

            for b in self.blocks:
                handles.append(mods[b].register_forward_pre_hook(make(b), with_kwargs=True))
            try:
                yield
            finally:
                for h in handles:
                    h.remove()
        return ctx()


def _load_vectors(args, concept, site="loc"):
    """``{step: {block: [H]}}`` for ``args.method``: the CAA unit vectors
    (every block; the site picks blocks), or the AUSteer sparse vectors
    selected for that site (top-s is global over the site's blocks)."""
    if getattr(args, "method", "caa") == "austeer":
        d = torch.load(args.out / "caa" / f"{concept}.austeer.pt", weights_only=False)
        return d["vectors"][site]
    d = torch.load(args.out / "caa" / f"{concept}.pt", weights_only=False)
    return d["vectors"]


def _sweep_dir(args, site, concept):
    return args.out / args.eval_sub / f"{args.method}_{site}_{concept}{args.suffix}"


#: Set from ``--renorm`` (reference SAO CAA configs: off).
_RENORM = False


def _render_alpha(sam, prompts, vectors, blocks, alpha, batch):
    sel = {s: {b: v for b, v in per.items() if b in set(blocks)} for s, per in vectors.items()}
    outs = []
    for idx in _batches(len(prompts), batch):
        with sa3_tada.steer_offline(sam, sel, alpha, renorm=_RENORM):
            outs.append(sa3_tada.generate(sam, [prompts[i] for i in idx], seed=EVAL_SEED,
                                          duration=DURATION, steps=STEPS))
    return torch.cat(outs)


def cmd_sweep(args, sam) -> None:
    nb = len(sa3_tada.cross_attn_modules(sam))
    tests = _eval_prompts(args)
    ranges = json.loads(Path(args.ranges).read_text()) if args.ranges else {}
    for concept in args.concepts:
        for site in args.sites:
            vectors = _load_vectors(args, concept, site)
            blocks = _site_blocks(site, args.loc, nb)
            from acestep.tada.caa import alphas_from_range

            rng = ranges.get(args.method, {}).get(concept, {}).get(site)
            if rng is None:
                if not args.alpha_max:
                    raise SystemExit(f"no strength range for {args.method}/{concept}/{site}")
                rng = [0.0 if args.pos_only else -args.alpha_max, args.alpha_max]
            alphas = alphas_from_range(float(rng[0]), float(rng[1]), args.points)
            amax = rng
            root = _sweep_dir(args, site, concept)
            for a in alphas:
                d = root / _fmt_alpha(a)
                if (d / "audios.npz").exists() and not args.force:
                    continue
                audio = _render_alpha(sam, tests, vectors, blocks, a, args.batch)
                _save_npz(d, audio, [f"p{i:03d}.wav" for i in range(len(tests))], mono=True)
            (root / "sweep.json").write_text(json.dumps({
                "concept": concept, "site": site, "blocks": blocks, "alphas": alphas,
                "prompts": len(tests), "seed": EVAL_SEED, "steps": STEPS,
                "duration": DURATION, "holdout": bool(args.holdout),
            }, indent=2))
            _log(f"sweep {site} {concept}: {len(alphas)} strengths x {len(tests)} prompts (max {amax})")


def cmd_probe(args, sam) -> None:
    """Calibration probe: render the held-out prompts at the given
    strengths for every site, into ``calib_<site>_<concept>``."""
    nb = len(sa3_tada.cross_attn_modules(sam))
    tests = _eval_prompts(args)
    for concept in args.concepts:
        for site in args.sites:
            vectors = _load_vectors(args, concept, site)
            blocks = _site_blocks(site, args.loc, nb)
            root = args.out / "calib" / f"{args.method}_{site}_{concept}"
            for a in [0.0] + [float(x) for x in args.alphas]:
                d = root / _fmt_alpha(a)
                if (d / "audios.npz").exists() and not args.force:
                    continue
                _save_npz(d, _render_alpha(sam, tests, vectors, blocks, a, args.batch),
                          [f"p{i:03d}.wav" for i in range(len(tests))], mono=True)
            _log(f"probe {site} {concept}: {len(args.alphas)} strengths")


def cmd_calibrate(args, _sam=None) -> None:
    """Strength ranges from the held-out probes (reference ranges are
    calibrated so the sweep reaches the prompt-swap distortion): per
    method, concept, site and direction, the smallest probed ``|alpha|``
    whose mean LPAPS reaches that direction's PCI cutoff
    (``min(PCI-all, PCI-loc)`` max LPAPS on the test prompts); the largest
    probe when none does (recorded as ``reached: false``). Writes
    ``<out>/ranges.json`` as ``{method: {concept: {site: [min, max]}}}``."""
    import csv

    def lp(d):
        f = d / "protocol_results" / "lpaps.csv"
        if not f.exists():
            f = d / "protocol_results" / "lpaps_endpoints.csv"
        if not f.exists():
            return None
        return {float(r["alpha"]): float(r["mean"]) for r in csv.DictReader(f.open())}

    ranges, notes = {}, {}
    for method in ("caa", "austeer"):
        for concept in args.concepts:
            cut = {}
            for direction, sign in (("pos", 1), ("neg", -1)):
                vals = [max(v for a, v in (lp(args.out / args.eval_sub / f"pci_{site}_{concept}") or {}).items()
                            if sign * a > 0) for site in ("all", "loc")
                        if lp(args.out / args.eval_sub / f"pci_{site}_{concept}")]
                cut[direction] = min(vals) if vals else None
            for site in ("all", "loc", "ablated"):
                pr = lp(args.out / "calib" / f"{method}_{site}_{concept}")
                if not pr or None in cut.values():
                    continue
                rng = []
                for direction, sign in (("neg", -1), ("pos", 1)):
                    probes = sorted((abs(a), v) for a, v in pr.items() if sign * a > 0)
                    hit = [a for a, v in probes if v >= cut[direction]]
                    a = hit[0] if hit else probes[-1][0]
                    rng.append(sign * a)
                    notes.setdefault(method, {}).setdefault(concept, {})[f"{site}_{direction}"] = {
                        "cutoff": cut[direction], "alpha": sign * a, "reached": bool(hit)}
                ranges.setdefault(method, {}).setdefault(concept, {})[site] = rng
    (args.out / f"ranges_{args.eval_sub}.json").write_text(json.dumps(ranges, indent=1))
    (args.out / f"ranges_{args.eval_sub}_notes.json").write_text(json.dumps(notes, indent=1))
    for m, per in ranges.items():
        for c, sites in per.items():
            print(m, c, sites)


def cmd_subset(args, _sam=None) -> None:
    """Copy the first ``--n-prompts`` rows of every PCI strength under
    ``eval`` into ``--eval-sub``: the PCI cutoff for a sweep run on the
    first N benchmark prompts is measured on the same N prompts."""
    src = args.out / "eval"
    for d in sorted(src.glob("pci_*")):
        for a in sorted(d.glob("alpha_*")):
            f = a / "audios.npz"
            dst = args.out / args.eval_sub / d.name / a.name / "audios.npz"
            if dst.exists() and not args.force:
                continue
            z = np.load(f)
            dst.parent.mkdir(parents=True, exist_ok=True)
            np.savez(dst, audio=z["audio"][: args.n_prompts], sr=z["sr"],
                     names=z["names"][: args.n_prompts])
    print(f"subset: {args.n_prompts} prompts into {args.out / args.eval_sub}")


def _clap_index(concept: str, prompts) -> int:
    """The ``eval_clap_prompts`` entry that names the clean concept (the
    reference plots ``clap_1_mean`` or ``clap_2_mean`` accordingly; e.g.
    ``sad`` lists ``["happy song", "sad song"]``)."""
    for i, p in enumerate(prompts):
        if concept.lower() in p.lower():
            return i
    return 0


def cmd_localize(args, _sam=None) -> dict:
    """Eq. 2 impacts from ``sa3_tada_score.py patch`` scores, averaged
    over concepts; functional blocks at ``tau`` (core patching API)."""
    from acestep.tada.patching import TAU, aggregate_impacts, layer_impacts, select_layers

    per = {}
    raw = {}
    for concept in args.concepts:
        f = args.out / "patch" / concept / "scores.json"
        if not f.exists():
            continue
        js = json.loads(f.read_text())
        sc = js["scores"]
        i = _clap_index(concept, js["prompts"])
        scores = {NONE: sc[NONE][i], ALL: sc[ALL][i]}
        nb = 0
        while f"tf{nb}" in sc:
            scores[nb] = sc[f"tf{nb}"][i]
            nb += 1
        per[concept] = layer_impacts(scores, nb)
        raw[concept] = {"none": scores[NONE], "all": scores[ALL]}
    agg = aggregate_impacts(per)
    sel = select_layers(agg, TAU)
    res = {"tau": TAU, "blocks": sel, "impacts": agg, "per_concept": per, "refs": raw,
           "ranked": sorted(range(len(agg)), key=lambda l: -agg[l])[:8]}
    (args.out / "localization.json").write_text(json.dumps(res, indent=1))
    print("I(l): " + " ".join(f"tf{l}={agg[l]:.3f}" for l in res["ranked"]))
    print(f"functional blocks (tau {TAU}): {sel}")
    for c, v in per.items():
        top = sorted(range(len(v)), key=lambda l: -(v[l] if v[l] == v[l] else -1))[:3]
        print(f"  {c:8s} none {raw[c]['none']:.3f} all {raw[c]['all']:.3f} top " +
              " ".join(f"tf{l}={v[l]:.2f}" for l in top))
    return res


def cmd_pack(args, _sam=None) -> None:
    """CAA vectors at the localised blocks -> TADA packs
    (``<packs>/sa3/<checkpoint>/<concept>.safetensors``). ``magnitude``
    is the calibrated strength for the site (knob 1 = that alpha)."""
    from acestep.tada.caa import stack_vectors
    from acestep.tada.packs import caa_pack, write_caa_pack

    ranges = json.loads(Path(args.ranges).read_text()) if args.ranges else {}
    for concept in args.concepts:
        vec = _load_vectors(args, concept)
        blocks = list(args.loc)
        mag = float(ranges.get(concept, {}).get("loc", args.alpha_max or 1.0))
        pack = caa_pack(
            stack_vectors(vec, blocks), family="sa3", checkpoint=args.checkpoint,
            concept=concept, blocks=blocks, magnitude=mag, renorm=False, cond_only=True,
            blurb=f"TADA CAA ({concept}) at cross-attention outputs of blocks {blocks}",
            provenance={"pairs": 50, "seed": CAA_SEED, "steps": STEPS,
                        "renorm": "off (reference Stable Audio Open eval configs)"},
        )
        print(write_caa_pack(pack, args.packs))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=("patch", "caa", "austeer", "pci", "sweep", "probe", "localize", "pack", "calibrate", "subset"))
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--concepts", nargs="+", default=None)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--force", action="store_true")
    # patch
    ap.add_argument("--prompts-csv", type=Path, default=DEFAULT_PROMPTS_CSV)
    ap.add_argument("--pairs", type=int, default=34)
    ap.add_argument("--seeds", type=int, default=4)
    # eval
    ap.add_argument("--loc", type=int, nargs="*", default=[])
    ap.add_argument("--sites", nargs="+", default=["all", "loc", "ablated"])
    ap.add_argument("--n-prompts", type=int, default=100)
    ap.add_argument("--holdout", action="store_true")
    ap.add_argument("--pos-only", action="store_true", help="positive strengths only (SAO configs)")
    ap.add_argument("--points", type=int, default=15, help="strengths per side")
    ap.add_argument("--alpha-max", type=float, default=None)
    ap.add_argument("--ranges", default=None, help="json {concept: {site: max}}")
    ap.add_argument("--alphas", nargs="*", default=[])
    ap.add_argument("--suffix", default="")
    ap.add_argument("--eval-sub", default="eval", help="eval directory under --out")
    ap.add_argument("--packs", type=Path, default=Path("E:/Projects/tada-replication/packs"))
    ap.add_argument("--checkpoint", default="medium")
    ap.add_argument("--method", choices=("caa", "austeer"), default="caa")
    ap.add_argument("--renorm", action="store_true")
    ap.add_argument("--top-s-loc", type=int, default=1024)
    ap.add_argument("--top-s-all", type=int, default=2048)
    args = ap.parse_args()

    if args.concepts is None:
        args.concepts = list(SAO_LOC_CONCEPTS if args.cmd in ("patch", "localize")
                             else C.STEERING_CONCEPTS)
    if args.cmd in ("localize", "pack", "calibrate", "subset"):
        {"localize": cmd_localize, "pack": cmd_pack, "calibrate": cmd_calibrate,
         "subset": cmd_subset}[args.cmd](args)
        return 0
    if args.cmd in ("caa", "austeer") and args.pairs == 34:
        args.pairs = 50
    torch.backends.cuda.matmul.allow_tf32 = True
    _log(f"{args.cmd}: loading SA3 medium")
    sam = _load_sam()
    global _RENORM
    _RENORM = bool(args.renorm)
    {"patch": cmd_patch, "caa": cmd_caa, "austeer": cmd_austeer, "pci": cmd_pci, "sweep": cmd_sweep,
     "probe": cmd_probe}[args.cmd](args, sam)
    _log("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
