"""Q10 (v3 smoke batch): a Concept-Sliders LoRA on SA3 medium (served ARC checkpoint, cfg 1), trained from S2's
grammar minimal pairs, evaluated like the activation-steering knobs and stacked with them.

Runbooks: notes/steering_pr/v3_smoke_runbook.md, v3_smoke_batch_runbook.md (DEMON); lit_review_v3_2026-10-07.md
(Concept Sliders, Gandikota et al. 2023, arXiv 2311.12092; Measured Sliders 2609.05234 for the composition metric).

Objective (Concept Sliders adapted to rectified flow, no CFG). For a minimal pair (prompt A = concept absent, prompt B =
concept present, one shared seed) the base model is run with S2's exact noise (capture_resid.PerRowNoise, pair seed) along
the A and the B trajectory; at every sampler step k and side s we keep x_t, the base velocity under its own prompt v_t and
under the other prompt, i.e. v_A(x_t) and v_B(x_t) on the same x_t. The LoRA (lora_strength = slider scale) is trained so that
    v_LoRA(+1)(x_t, c_t) -> v_t + eta (v_B - v_A)        v_LoRA(-1)(x_t, c_t) -> v_t - eta (v_B - v_A)
with c_t the prompt of side s. The scale is the vendored ``lora_strength`` buffer, the same live knob DEMON's
SA3LoRAManager.set_strength writes, so a negative strength is the other side of the slider. Adapter: vendored
stable_audio_3 LoRAParametrization (adapter_type "lora", rank 8) on the DiT trunk self-attention (to_qkv, to_out) and
feed-forward linears of all 24 layers; cross-attention, conditioner and embedders untouched (Concept Sliders' "noxattn"
choice, so the slider does not key on the prompt text). Saved with the vendored ``save_lora_safetensors`` (trainer format,
keys of ``get_lora_state_dict(sam.model.model)``), i.e. loadable by the existing SA3 LoRA path.

Subcommands:
  train      collect the base trajectories (cache_<concept>.pt) then train under a wall-clock budget (GPU)
  drive      knob grid (slider vs S2 pair vector vs the S2 control, 12 protocol prompts, seed 0 per-row, 5 strengths per
             sign up to the PCI cutoff) + stacking table (GPU; scorer worker on the evalenv)
  indep-sum  independent-scorer table from rescore_independent q2-cpu / q2-clap and pairs_s2 indep-muq outputs (CPU)

Dry runs: ``train --device cpu --duration 1 --steps 2 --n-pairs 2 --max-steps 2`` runs the REAL model on CPU (fp32);
``drive --dry-run`` stubs renderer and scorer (smoke_drive / screen_knobs fakes) and a missing S2 pack.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import json
import os
import sys
import time
from functools import partial
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
for _p in (str(REPO / "scripts" / "tada"), str(REPO / "scripts" / "sa3"), str(HERE), str(REPO)):
    while _p in sys.path:
        sys.path.remove(_p)
    sys.path.insert(0, _p)

Q10 = Path("E:/Projects/DEMON/steering-bench/v3_smoke/q10")
S2 = Path("E:/Projects/DEMON/steering-bench/v3_smoke/s2")
INCLUDE = ["self_attn.to_qkv", "self_attn.to_out", "ff.ff"]
EXCLUDE = ["cross_attn", "to_local_embed"]
LORA_PROBE = (0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0)
FRACTIONS = (0.2, 0.4, 0.6, 0.8, 1.0)
EVAL_SEED = 2115
#: concept -> anchors.json concept used as its CLAP check (pos text = concept side B)
CONCEPT_CLAP = {"tempo": "fast_tempo", "drums": "percussive", "epiano": "electric_piano"}
STACK_KNOBS = ["warm", "plate_reverb"]


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _akey(a: float) -> str:
    return f"alpha_{float(a)}"


# ----------------------------------------------------------------------------------------------- data

def pick_concept(s2: Path, override: str | None) -> tuple[str, dict]:
    if not (s2 / "fit_report.json").exists() and override:      # CPU dry run before S2's fit exists
        return override, {}
    fit = json.loads((s2 / "fit_report.json").read_text())
    best = {c: r["per_block"][str(r["best_block"])] for c, r in fit["concepts"].items()}
    c = override or max(best, key=lambda k: (best[k]["auc"], best[k]["dprime"]))
    return c, {k: {"block": fit["concepts"][k]["best_block"], **v} for k, v in best.items()}


def load_pairs(s2: Path, concept: str) -> list[dict]:
    rows = json.loads((s2 / "corpus" / "prompts.json").read_text(encoding="utf-8"))
    by = {}
    for r in rows:
        if r["concept"] == concept:
            by.setdefault(r["pair"], {})[int(r["side"])] = r
    out = []
    for p in sorted(by):
        a, b = by[p][0], by[p][1]
        if int(a["seed"]) != int(b["seed"]):
            raise SystemExit(f"pair {p}: sides have different seeds")
        out.append({"pair": p, "pic": int(a["pair_in_concept"]), "a": a["prompt"], "b": b["prompt"], "seed": int(a["seed"]),
                    "heldout": int(a["pair_in_concept"]) % 3 == 0})        # S2's split (fit_report "split")
    return out


# ----------------------------------------------------------------------------------------------- model + LoRA

def load_sam(device: str):
    import torch

    from sa3_reference_generate import checkpoint_dir, load_local_model

    torch.backends.cuda.matmul.allow_tf32 = True
    sam = load_local_model(checkpoint_dir("medium"), device=device, model_half=device != "cpu")
    if device == "cpu":                       # CPU dry run only: flash-attn has no CPU kernel, use the fallback path
        import stable_audio_3.models.transformer as TR

        TR.flash_attn_func = TR.flash_attn_varlen_func = None
    sam.model.eval()
    return sam


def _vendored():
    from acestep.engine.sa3_helpers import ensure_sa3_paths

    ensure_sa3_paths()
    from stable_audio_3.models import lora as L

    return L


def attach_lora(sam, rank: int, alpha: float) -> list:
    """Vendored add_lora on the DiT tree (the trainer's own call), base frozen. Returns the trainable LoRA params."""
    import torch

    L = _vendored()
    dit = sam.model.model
    dit.requires_grad_(False)
    sam.model.conditioner.requires_grad_(False)
    cfg = {torch.nn.Linear: {"weight": partial(L.LoRAParametrization.from_linear, rank=rank, lora_alpha=alpha,
                                               adapter_type="lora")}}
    L.add_lora(dit, cfg, include=INCLUDE, exclude=EXCLUDE)
    params = list(L.get_lora_params(dit))
    for p in params:
        p.requires_grad_(True)
    L.set_lora_strength(dit, 0.0)
    return params


def set_strength(sam, s: float) -> None:
    _vendored().set_lora_strength(sam.model.model, float(s))


def lora_config(rank, alpha, extra=None) -> dict:
    return {"rank": rank, "alpha": alpha, "adapter_type": "lora", "include": INCLUDE, "exclude": EXCLUDE, **(extra or {})}


def save_lora(sam, path: Path, cfg: dict) -> Path:
    L = _vendored()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.safetensors")
    L.save_lora_safetensors(L.get_lora_state_dict(sam.model.model), cfg, tmp)
    os.replace(tmp, path)
    return path


def load_lora(sam, path: Path) -> dict:
    """Attach + load a slider file the way the vendored trainer resumes (add_lora with the file's own config, then
    load_state_dict strict=False); every LoRA key of the model must come from the file."""
    import safetensors
    from safetensors.torch import load_file

    with safetensors.safe_open(str(path), framework="pt") as f:
        cfg = json.loads((f.metadata() or {})["lora_config"])
    if cfg["include"] != INCLUDE or cfg["exclude"] != EXCLUDE:
        raise SystemExit(f"{path}: include/exclude {cfg['include']}/{cfg['exclude']} differ from this script's")
    attach_lora(sam, int(cfg["rank"]), float(cfg["alpha"]))
    sd = load_file(str(path))
    res = sam.model.model.load_state_dict(sd, strict=False)
    want = set(_vendored().get_lora_state_dict(sam.model.model))
    if want != set(sd) or res.unexpected_keys:
        raise SystemExit(f"{path}: key mismatch ({len(want ^ set(sd))} differ, unexpected {res.unexpected_keys[:3]})")
    sam.model.model.requires_grad_(False)
    set_strength(sam, 0.0)
    return cfg


def mem_cap(gib: float) -> None:
    import torch

    if gib and gib > 0 and torch.cuda.is_available():
        tot = torch.cuda.get_device_properties(0).total_memory
        torch.cuda.set_per_process_memory_fraction(min(1.0, gib * 2 ** 30 / tot), 0)


# ----------------------------------------------------------------------------------------------- collect

def collect(sam, pairs: list, args) -> dict:
    """Base trajectories for every pair, both sides, all steps: x_t, v under its own prompt, v under the other prompt
    (same x_t, conditioning rows swapped inside the pair), plus the per-row conditioning kwargs."""
    import torch

    import capture_resid as CR

    dit = sam.model.model
    st = {"busy": False, "cap": []}

    def hook(mod, a, kw, out):
        if st["busy"]:
            return out
        x, t = a[0], a[1]
        n = int(x.shape[0])
        perm = torch.tensor([i ^ 1 for i in range(n)], device=x.device)
        kw2 = {k: (v.index_select(0, perm) if torch.is_tensor(v) and v.dim() >= 1 and v.shape[0] == n else v)
               for k, v in kw.items()}
        st["busy"] = True
        try:
            other = mod(x, t, **kw2)
        finally:
            st["busy"] = False
        st["cap"].append({"x": x.detach().to("cpu", torch.float16), "t": t.detach().float().cpu(),
                          "own": out.detach().to("cpu", torch.float16), "other": other.detach().to("cpu", torch.float16),
                          "kw": kw if not st["cap"] else None})
        return out

    h = dit.register_forward_hook(hook, with_kwargs=True)
    P, K = len(pairs), args.steps
    X = OWN = OTH = None
    T = torch.zeros(K)
    cond, static = {}, {}
    t0 = time.time()
    try:
        for i in range(0, P, args.collect_pairs):
            chunk = pairs[i:i + args.collect_pairs]
            prompts = [p[s] for p in chunk for s in ("a", "b")]
            seeds = [p["seed"] for p in chunk for _ in (0, 1)]
            st["cap"] = []
            with torch.no_grad(), CR.PerRowNoise(seeds, args.device):
                sam.generate(prompt=prompts, duration=float(args.duration), steps=K, seed=seeds[0], cfg_scale=1.0,
                             batch_size=len(prompts), return_latents=True)
            if len(st["cap"]) != K:
                raise SystemExit(f"saw {len(st['cap'])} DiT forwards, expected {K}")
            if X is None:
                shp = tuple(st["cap"][0]["x"].shape[1:])
                X, OWN, OTH = (torch.zeros((P, 2, K) + shp, dtype=torch.float16) for _ in range(3))
            kw = st["cap"][0]["kw"]
            n = len(prompts)
            for k, v in kw.items():
                if torch.is_tensor(v) and v.dim() >= 1 and v.shape[0] == n:
                    v = v.detach().cpu()
                    if k not in cond:
                        cond[k] = torch.zeros((P, 2) + tuple(v.shape[1:]), dtype=v.dtype)
                    if tuple(cond[k].shape[2:]) != tuple(v.shape[1:]):
                        raise SystemExit(f"conditioning {k}: shape {tuple(v.shape[1:])} vs {tuple(cond[k].shape[2:])}")
                    cond[k][i:i + len(chunk)] = v.reshape((len(chunk), 2) + tuple(v.shape[1:]))
                elif i == 0:
                    static[k] = v.detach().cpu() if torch.is_tensor(v) else v
            for k, c in enumerate(st["cap"]):
                T[k] = c["t"][0]
                for name, dst in (("x", X), ("own", OWN), ("other", OTH)):
                    dst[i:i + len(chunk), :, k] = c[name].reshape((len(chunk), 2) + tuple(c[name].shape[1:]))
            _log(f"collect {min(i + len(chunk), P)}/{P} pairs ({time.time() - t0:.0f} s)")
    finally:
        h.remove()
    return {"X": X, "OWN": OWN, "OTH": OTH, "T": T, "cond": cond, "static": static,
            "pairs": pairs, "steps": K, "duration": args.duration}


# ----------------------------------------------------------------------------------------------- train

def _batch(data, idx, dev, dtype, eta):
    """idx: list of (pair i, side s, step k). Returns model inputs and (v_t, d = eta (v_B - v_A))."""
    import torch

    pi = torch.tensor([i for i, _s, _k in idx])
    si = torch.tensor([s for _i, s, _k in idx])
    ki = torch.tensor([k for _i, _s, k in idx])
    x = data["X"][pi, si, ki].to(dev, dtype)
    own = data["OWN"][pi, si, ki].to(dev, torch.float32)
    oth = data["OTH"][pi, si, ki].to(dev, torch.float32)
    sgn = (2 * si - 1).to(dev, torch.float32).view(-1, *([1] * (own.dim() - 1)))  # side B: v_B - v_A = own - oth
    d = eta * sgn * (own - oth)
    t = data["T"][ki].to(dev, dtype)
    kw = {k: v[pi, si].to(dev) for k, v in data["cond"].items()}
    kw = {k: (v.to(dtype) if v.is_floating_point() else v) for k, v in kw.items()}
    for k, v in data["static"].items():
        kw[k] = v.to(dev) if hasattr(v, "to") else v
    return x, t, kw, own, d


def train(args) -> int:
    import torch

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    concept, aucs = pick_concept(Path(args.s2), args.concept)
    pairs = load_pairs(Path(args.s2), concept)
    if args.n_pairs:
        pairs = pairs[:max(2, args.n_pairs)]
        if not any(p["heldout"] for p in pairs):
            pairs[0]["heldout"] = True
        if all(p["heldout"] for p in pairs):
            pairs[-1]["heldout"] = False
    dev = args.device
    dtype = torch.float32 if dev == "cpu" else torch.float16
    _log(f"concept {concept} (S2 best-block held-out AUCs {json.dumps(aucs)}); "
         f"{len(pairs)} pairs, {sum(p['heldout'] for p in pairs)} held out")
    sam = load_sam(dev)
    mem_cap(args.mem_gib if dev != "cpu" else 0)
    cache = out / f"cache_{concept}{'_' + args.tag if args.tag else ''}.pt"
    if cache.exists() and not args.recollect:
        data = torch.load(cache, weights_only=False)
        if [p["pair"] for p in data["pairs"]] != [p["pair"] for p in pairs] or data["steps"] != args.steps:
            raise SystemExit(f"{cache}: made for other pairs/steps; pass --recollect")
        _log(f"cache {cache} loaded")
    else:
        t0 = time.time()
        data = collect(sam, pairs, args)
        torch.save(data, cache)
        _log(f"collected {len(pairs)} pairs x 2 sides x {args.steps} steps in {time.time() - t0:.0f} s -> {cache}")
    if dev != "cpu":                                    # conditioning is cached: keep only the DiT on the GPU
        sam.model.conditioner.to("cpu")
        sam.model.pretransform.to("cpu")
        torch.cuda.empty_cache()
    params = attach_lora(sam, args.rank, args.alpha)
    dit = sam.model.model
    nparam = sum(p.numel() for p in params)
    _log(f"LoRA rank {args.rank} alpha {args.alpha}: {len(params) // 2} layers, {nparam / 1e6:.2f} M params")
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    scaler = torch.amp.GradScaler("cuda", enabled=dev != "cpu")
    K = data["steps"]
    tr = [(i, s, k) for i, p in enumerate(pairs) if not p["heldout"] for s in (0, 1) for k in range(K)]
    ho = [(i, s, k) for i, p in enumerate(pairs) if p["heldout"] for s in (0, 1) for k in range(K)]
    rng = np.random.default_rng(0)
    val = [ho[j] for j in rng.permutation(len(ho))[: args.val_records]]

    @torch.no_grad()
    def evaluate():
        """Held-out: rel MSE per sign (1.0 = untrained slider), floor (strength 0 vs cached v_t: batch-composition
        noise), and cos((v(+1) - v(-1)) / 2, d) = how well the slider's own direction matches the pair difference."""
        acc = {"pos": 0.0, "neg": 0.0, "floor": 0.0, "dd": 0.0, "cos": []}
        for j in range(0, len(val), args.batch):
            x, t, kw, own, d = _batch(data, val[j:j + args.batch], dev, dtype, args.eta)
            pr = {}
            for name, s in (("pos", 1.0), ("neg", -1.0), ("floor", 0.0)):
                set_strength(sam, s)
                pr[name] = dit(x, t, **kw).float()
            acc["pos"] += float(((pr["pos"] - own - d) ** 2).sum())
            acc["neg"] += float(((pr["neg"] - own + d) ** 2).sum())
            acc["floor"] += float(((pr["floor"] - own) ** 2).sum())
            acc["dd"] += float((d ** 2).sum())
            half = ((pr["pos"] - pr["neg"]) / 2).flatten(1)
            acc["cos"] += torch.nn.functional.cosine_similarity(half, d.flatten(1), dim=1).tolist()
        set_strength(sam, 0.0)
        dd = max(acc["dd"], 1e-12)
        return {"val_rel_pos": acc["pos"] / dd, "val_rel_neg": acc["neg"] / dd, "val_floor": acc["floor"] / dd,
                "val_cos": float(np.mean(acc["cos"]))}

    log_path = out / f"train_log_{concept}{'_' + args.tag if args.tag else ''}.csv"
    fields = ["step", "seconds", "train_rel_pos", "train_rel_neg", "skipped", "val_rel_pos", "val_rel_neg",
              "val_floor", "val_cos"]
    lf = open(log_path, "w", newline="")
    w = csv.DictWriter(lf, fieldnames=fields)
    w.writeheader()
    t0 = time.time()
    v = evaluate()
    w.writerow({"step": 0, "seconds": 0, **v})
    lf.flush()
    _log(f"step 0: {v}")
    name = f"slider_{concept}{'_' + args.tag if args.tag else ''}"
    meta = {"concept": concept, "eta": args.eta, "lr": args.lr, "batch": args.batch, "n_train_pairs": len(pairs) - len(
        {i for i, _s, _k in ho}), "n_heldout_pairs": len({i for i, _s, _k in ho}), "steps": K, "duration": args.duration,
        "objective": "concept sliders (rf): v(+s) -> v_t + s eta (v_B - v_A), both signs", "s2": str(args.s2)}
    best, run = {"val": float("inf"), "step": 0}, {"pos": [], "neg": [], "skip": 0}
    step = 0
    while step < args.max_steps and (time.time() - t0) < args.minutes * 60:
        step += 1
        idx = [tr[j] for j in rng.integers(0, len(tr), args.batch)]
        x, t, kw, own, d = _batch(data, idx, dev, dtype, args.eta)
        dd = float((d ** 2).mean().clamp_min(1e-12))
        rel = {}
        for name_s, s in (("pos", 1.0), ("neg", -1.0)):
            set_strength(sam, s)
            pred = dit(x, t, **kw).float()
            loss = ((pred - own - s * d) ** 2).mean()
            rel[name_s] = float(loss.detach()) / dd
            scaler.scale(loss).backward()
        set_strength(sam, 0.0)
        if not all(np.isfinite(list(rel.values()))):
            run["skip"] += 1
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)
        run["pos"].append(rel["pos"])
        run["neg"].append(rel["neg"])
        if step % args.eval_every == 0 or step >= args.max_steps or (time.time() - t0) >= args.minutes * 60:
            v = evaluate()
            row = {"step": step, "seconds": round(time.time() - t0, 1), "train_rel_pos": float(np.mean(run["pos"])),
                   "train_rel_neg": float(np.mean(run["neg"])), "skipped": run["skip"], **v}
            w.writerow(row)
            lf.flush()
            run = {"pos": [], "neg": [], "skip": run["skip"]}
            vv = (v["val_rel_pos"] + v["val_rel_neg"]) / 2
            save_lora(sam, out / "lora" / f"{name}_last.safetensors", lora_config(args.rank, args.alpha, {**meta, "step": step}))
            if vv < best["val"]:
                best = {"val": vv, "step": step}
                save_lora(sam, out / "lora" / f"{name}.safetensors", lora_config(args.rank, args.alpha, {**meta, "step": step}))
            _log(f"step {step} ({row['seconds']:.0f} s): train {row['train_rel_pos']:.3f}/{row['train_rel_neg']:.3f} "
                 f"val {v['val_rel_pos']:.3f}/{v['val_rel_neg']:.3f} floor {v['val_floor']:.3f} cos {v['val_cos']:.3f}")
    lf.close()
    summary = {**meta, "rank": args.rank, "alpha": args.alpha, "train_steps": step, "seconds": round(time.time() - t0, 1),
               "best_step": best["step"], "best_val_rel_mean": best["val"], "lora": str(out / "lora" / f"{name}.safetensors"),
               "log": str(log_path), "s2_auc": aucs, "include": INCLUDE, "exclude": EXCLUDE,
               "peak_alloc_gib": (round(torch.cuda.max_memory_allocated() / 2 ** 30, 2) if dev != "cpu" else None)}
    (out / f"train_{name}.json").write_text(json.dumps(summary, indent=1))
    # round trip: reload into a fresh attach must reproduce the trained deltas
    _log(f"train done: {step} steps, best step {best['step']} (val rel {best['val']:.3f}) -> {summary['lora']}")
    return 0


# ----------------------------------------------------------------------------------------------- drive

class Gen:
    """SA3 medium eager with the slider attached; renders under PerRowNoise (clip i seeded by seeds[i], any chunking),
    slider strength + any number of steering packs (each pack its own steer_offline context, nested)."""

    def __init__(self, lora: Path, args):
        import torch

        import capture_resid as CR
        import sa3_tada_run as R
        from acestep.engine import sa3_tada

        self.torch, self.CR, self.R, self.E, self.args = torch, CR, R, sa3_tada, args
        self.sam = load_sam(args.device)
        self.cfg = load_lora(self.sam, lora)
        mem_cap(args.mem_gib if args.device != "cpu" else 0)
        self.cache = {}

    def _pv(self, pack):
        if pack not in self.cache:
            self.cache[pack] = (self.R._pack_vectors(pack, self.args.steps), self.R._PACK["hook"])
        return self.cache[pack]

    def render(self, prompts, seeds, strength: float, vecs=()) -> np.ndarray:
        torch = self.torch
        out = []
        set_strength(self.sam, strength)
        try:
            for i in range(0, len(prompts), self.args.batch):
                p, s = prompts[i:i + self.args.batch], seeds[i:i + self.args.batch]
                with contextlib.ExitStack() as es:
                    for pack, alpha in vecs:
                        if alpha == 0.0:
                            continue
                        pv, hook = self._pv(pack)
                        es.enter_context(self.E.steer_offline(
                            self.sam, {st: {b: v * float(alpha) for b, v in d.items()} for st, d in pv.items()}, 1.0, hook=hook))
                    es.enter_context(torch.no_grad())
                    es.enter_context(self.CR.PerRowNoise(s, self.args.device))
                    a = self.E.generate(self.sam, p, seed=s[0], duration=self.args.duration, steps=self.args.steps)
                out.append((a.mean(dim=1) * 32768.0).round().clamp(-32768, 32767).to(torch.int16).numpy())
        finally:
            set_strength(self.sam, 0.0)
        if self.args.device != "cpu":
            torch.cuda.empty_cache()
        return np.concatenate(out)


class FakeGen:
    def __init__(self, lora, args):
        from screen_knobs import FakeGenerator

        self.g = FakeGenerator()

    def render(self, prompts, seeds, strength, vecs=()) -> np.ndarray:
        names = (["slider"] if strength else []) + [Path(p).stem for p, a in vecs if a]
        self.g.cand = "+".join(names)
        amp = abs(strength) * 8.0 + sum(abs(a) for _p, a in vecs) / 4.0
        return self.g.render(prompts, None, None, amp * (1 if (strength or 0) >= 0 else -1))


def _save(d: Path, wav: np.ndarray, prompts) -> Path:
    d.mkdir(parents=True, exist_ok=True)
    p = d / "audios.npz"
    np.savez(p, audio=wav[:, None, :], sr=np.int64(44100), names=np.array(list(prompts)))
    return p


def _scores(r: dict, clap_pairs: dict) -> dict:
    import pairs_s2 as P2

    out = {c: r["cols"][c] for c in P2.COLS}
    out["clap_ep"] = (np.asarray(r["clap"]["electric piano"]) - np.asarray(r["clap"]["piano"])).tolist()
    for name, (pos, neg) in clap_pairs.items():
        out[f"clap:{name}"] = (np.asarray(r["clap"][pos]) - np.asarray(r["clap"][neg])).tolist()
    return out


def _metrics(rec, score, direction, refv):
    import smoke_drive as SDR

    alphas = rec["alphas"]
    vals = np.array([rec["scores"][_akey(a)][score] for a in alphas], dtype=float)
    lp = [float(np.mean(rec["lpaps"][_akey(a)])) for a in alphas]
    rho = [direction * SDR.spearman(vals[:, i], np.abs(alphas)) for i in range(vals.shape[1])]
    rho_f = [r for r in rho if np.isfinite(r)]
    dl = [float(direction * np.nanmean(vals[k] - refv)) for k in range(len(alphas))]
    return {"rho_median": float(np.median(rho_f)) if rho_f else float("nan"), "lpaps": lp, "lpaps_max": max(lp),
            "lpaps_top": lp[-1], "delta": dl, "delta_top": dl[-1]}


def stub_pair_pack(dest: Path, concept: str) -> Path:
    import torch
    from acestep.steering.packs import SteeringPack, save_pack

    v = np.random.default_rng(1).standard_normal(1536).astype(np.float32)
    v /= np.linalg.norm(v)
    return save_pack(SteeringPack(family="sa3", checkpoint="medium", block=12, hidden_size=1536, name=f"pair_{concept}",
                                  vector=torch.from_numpy(v), label="stub", hook="post_block_residual", norm=1.0,
                                  magnitude=0.1, provenance={"method": "minimal_pair", "stub": True}),
                     dest / f"pair_{concept}.safetensors")


def drive(args) -> int:
    import pairs_s2 as P2
    import ship_tools as ST
    import smoke_drive as SDR

    t0 = time.time()
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    R = out / "renders"
    notes = []
    s2 = Path(args.s2)
    if (s2 / "fit_report.json").exists():
        concept, aucs = pick_concept(s2, args.concept)
        fit = json.loads((s2 / "fit_report.json").read_text())["concepts"][concept]
        pair_pack, control = Path(fit["pack"]), fit["control"]
    elif args.dry_run:
        concept, aucs, control = args.concept or "drums", {}, None
        pair_pack = stub_pair_pack(out / "stub", concept)
        control = P2.CONCEPTS[concept][2]
        notes.append("dry run: no S2 fit_report.json; stub pair pack")
    else:
        raise SystemExit(f"{s2 / 'fit_report.json'} missing: S2 fit has not run")
    lora = Path(args.lora) if args.lora else out / "lora" / f"slider_{concept}.safetensors"
    if not lora.exists() and not args.dry_run:
        raise SystemExit(f"{lora} missing: train first")
    cutsrc = control or P2.TEMPO_CUTOFF_FROM
    anchors = json.loads((HERE / "anchors.json").read_text(encoding="utf-8"))["concepts"]
    stack = list(args.stack_knobs)
    cargs = argparse.Namespace(bundle=SDR.BUNDLE, steer_repo=SDR.STEER_REPO,
                               controls=list(dict.fromkeys([c for c in (control, cutsrc) if c] + stack)))
    ctrl = SDR.load_controls(cargs, out)
    cclap = CONCEPT_CLAP[concept]
    clap_pairs = {cclap: tuple(anchors[cclap][:2])}
    for k in stack + ([control] if control else []):
        clap_pairs[k] = tuple(anchors[k][:2])
    texts = list(dict.fromkeys(P2.CLAP_TEXTS + [t for pr in clap_pairs.values() for t in pr]))
    worker = SDR.Worker(args.eval_python if not args.dry_run else (args.eval_python or sys.executable), args.dry_run,
                        args.tada_root, args.label_workers)
    gen = (FakeGen if args.dry_run else Gen)(lora, args)
    if args.dry_run:
        prompts = [f"dry prompt {i}" for i in range(args.n_prompts)]
    else:
        from acestep.tada import concepts as C

        prompts = C.benchmark_prompts(holdout=True)[0][: args.n_prompts]
    seeds = [EVAL_SEED + i for i in range(len(prompts))]
    nclips = 0

    def submit(npz, ref, cand):
        return worker.submit(npz=str(npz), ref=str(ref), cols=P2.COLS, anchors=texts, cand=cand)

    def render_to(d: Path, strength, vecs, alpha_key):
        nonlocal nclips
        p = d / _akey(alpha_key) / "audios.npz"
        if args.resume and p.exists():
            return p
        nclips += len(prompts)
        return _save(d / _akey(alpha_key), gen.render(prompts, seeds, strength, vecs), prompts)

    ref_npz = render_to(R / "ref", 0.0, (), 0.0)
    ref = _scores(worker.get(submit(ref_npz, ref_npz, "")), clap_pairs)
    _log(f"ref rendered + scored ({len(prompts)} clips)")
    knobs = {"slider": {"kind": "lora", "path": str(lora)}, f"pair_{concept}": {"kind": "vector", "path": str(pair_pack)}}
    if control:
        knobs[control] = {"kind": "control", "path": ctrl[control]["path"]}
    recs, pending = {}, []
    for vid, kn in knobs.items():
        vdir = R / vid

        def rend(a, d, _kn=kn):
            if _kn["kind"] == "lora":
                return render_to(d, a, (), a)
            return render_to(d, 0.0, ((_kn["path"], a),), a)

        probe = {0.0: [0.0] * len(prompts)}
        for s in (1, -1):
            side = SDR.SIDES[s]
            cut = ctrl[cutsrc]["side"][side]["cutoff"]
            if kn["kind"] == "control":
                a_c, src = ctrl[vid]["side"][side]["alpha_cut"], "calibrated gain x magnitude"
            else:
                for a in (LORA_PROBE if kn["kind"] == "lora" else P2.PROBE_GRID):
                    npz = rend(s * a, vdir / "calib" / f"pack_pack_{vid}")
                    lp = worker.get(worker.submit(npz=str(npz), ref=str(ref_npz), cand=vid))["lpaps"]
                    probe[s * a] = lp
                    if float(np.mean(lp)) >= cut:
                        break
                SDR.write_lpaps_csv(vdir / "calib" / f"pack_pack_{vid}" / "protocol_results" / "lpaps.csv", probe)
                SDR.write_lpaps_csv(vdir / "eval" / f"pci_all_{vid}" / "protocol_results" / "lpaps.csv",
                                    {1.0: [ctrl[cutsrc]["side"]["pos"]["cutoff"]], -1.0: [ctrl[cutsrc]["side"]["neg"]["cutoff"]]})
                r = ST.interp_side(vdir, vid, s)
                a_c = r["alpha"]
                src = f"probe {'reached' if r['reached'] else 'NOT reached (largest probe)'}, interp_side"
            alphas = [s * a_c * f for f in FRACTIONS]
            rec = {"vid": vid, "kind": kn["kind"], "sign": s, "a_c": a_c, "a_c_source": src, "cutoff": cut,
                   "cutoff_from": cutsrc, "alphas": alphas, "lpaps": {}, "scores": {}}
            for a in alphas:
                npz = rend(a, vdir / "eval" / f"pack_pack_{vid}")
                pending.append((rec, a, submit(npz, ref_npz, vid)))
            recs[(vid, s)] = rec
            _log(f"{vid} {side}: a_c {a_c:.4g} ({src}) cutoff {cut:.3f}; {nclips} clips")
    for rec, a, rid in pending:
        r = worker.get(rid)
        rec["lpaps"][_akey(a)] = r["lpaps"]
        rec["scores"][_akey(a)] = _scores(r, clap_pairs)

    # ------------------------------------------------------------------ stacking
    S, V = "slider", f"pair_{concept}"
    members = {S: {"kind": "lora", "a": recs[(S, 1)]["a_c"], "cut": recs[(S, 1)]["cutoff"], "label": P2.SCORES[concept][0]},
               V: {"kind": "vector", "path": str(pair_pack), "a": recs[(V, 1)]["a_c"], "cut": recs[(V, 1)]["cutoff"],
                   "label": P2.SCORES[concept][0]}}
    for k in stack:
        members[k] = {"kind": "vector", "path": ctrl[k]["path"], "a": ctrl[k]["side"]["pos"]["alpha_cut"],
                      "cut": ctrl[k]["side"]["pos"]["cutoff"], "label": f"clap:{k}"}
    combos = [[S], [V]] + [[k] for k in stack] + [[S, V], [S] + stack, [V] + stack, [S, V] + stack]
    srecs, spend = [], []
    for rule in ("full", "inv_n"):
        for cm in combos:
            if rule == "inv_n" and len(cm) == 1:
                continue
            f = 1.0 if rule == "full" else 1.0 / len(cm)
            strength = members[S]["a"] * f if S in cm else 0.0
            vecs = tuple((members[m]["path"], members[m]["a"] * f) for m in cm if m != S)
            cid = f"{rule}__{'+'.join(cm)}"
            npz = render_to(out / "stack" / cid, strength, vecs, 1.0)
            srec = {"id": cid, "rule": rule, "members": cm, "factor": f, "cutoff": min(members[m]["cut"] for m in cm)}
            srecs.append(srec)
            spend.append((srec, submit(npz, ref_npz, cid)))
    for srec, rid in spend:
        r = worker.get(rid)
        srec["lpaps"] = r["lpaps"]
        srec["scores"] = _scores(r, clap_pairs)
    worker.close()

    # ------------------------------------------------------------------ analysis
    rows = []
    for vid, kn in knobs.items():
        for score in P2.SCORES[concept] + [f"clap:{cclap}"]:
            refv = np.asarray(ref[score], dtype=float)
            for direction in (1, -1):
                rec = recs[(vid, direction)]
                m = _metrics(rec, score, direction, refv)
                row = {"knob": vid, "kind": kn["kind"], "score": score, "direction": P2.SIDES_TXT[direction],
                       "a_c": rec["a_c"], "cutoff": rec["cutoff"], "rho_median": m["rho_median"],
                       "lpaps_top_over_cut": m["lpaps_top"] / rec["cutoff"], "delta_top": m["delta_top"], "metrics": m}
                for other in ([V] if vid == S else []) + ([control] if control and vid != control else []):
                    cands = [(_metrics(recs[(other, s)], score, direction, refv), s) for s in (1, -1)] if other == control \
                        else [(_metrics(recs[(other, direction)], score, direction, refv), direction)]
                    om, os_ = max(cands, key=lambda x: x[0]["delta_top"])
                    lstar = min(rec["cutoff"], m["lpaps_max"], om["lpaps_max"])
                    tag = "vector" if other == V else "control"
                    row.update({f"{tag}_sign": SDR.SIDES[os_], f"{tag}_rho": om["rho_median"],
                                f"matched_lpaps_vs_{tag}": lstar, f"delta_at_matched_vs_{tag}": SDR.delta_at(m["lpaps"], m["delta"], lstar),
                                f"{tag}_delta_at_matched": SDR.delta_at(om["lpaps"], om["delta"], lstar)})
                rows.append(row)
    alone = {}
    for srec in srecs:
        if len(srec["members"]) == 1:
            m = srec["members"][0]
            alone[m] = np.asarray(srec["scores"][members[m]["label"]], float) - np.asarray(ref[members[m]["label"]], float)
    stab = []
    for srec in srecs:
        eff, ok = {}, None
        for m in srec["members"]:
            lab = members[m]["label"]
            d = np.asarray(srec["scores"][lab], float) - np.asarray(ref[lab], float)
            eff[m] = {"label": lab, "delta": float(np.nanmean(d)), "alone": float(np.nanmean(alone[m])),
                      "retained": float(np.nanmean(d) / np.nanmean(alone[m])) if np.nanmean(alone[m]) else float("nan")}
            ok = (d > 0) if ok is None else (ok & (d > 0))
        lp = float(np.mean(srec["lpaps"]))
        stab.append({"id": srec["id"], "rule": srec["rule"], "members": srec["members"], "lpaps": lp,
                     "cutoff": srec["cutoff"], "lpaps_over_cut": lp / srec["cutoff"], "effects": eff,
                     "all_preserved_frac": float(np.mean(ok)) if ok is not None else None})
    res = {"meta": {"finished": time.strftime("%Y-%m-%d %H:%M:%S"), "seconds": round(time.time() - t0, 1), "clips": nclips,
                    "dry_run": args.dry_run, "concept": concept, "s2_auc": aucs, "lora": str(lora), "pair_pack": str(pair_pack),
                    "control": control, "cutoff_from": cutsrc, "prompts": prompts, "seeds": seeds,
                    "seed_rule": "clip i seed 2115 + i under PerRowNoise (S2 drive seed 0)", "stack_knobs": stack,
                    "members": {k: {kk: vv for kk, vv in v.items() if kk != "path"} for k, v in members.items()},
                    "notes": notes},
           "rows": rows, "stack": stab, "sweeps": {f"{k[0]}|{SDR.SIDES[k[1]]}": v for k, v in recs.items()}, "ref": ref}
    (out / "results.json").write_text(json.dumps(res, indent=1, default=float))
    write_md(out / "results.md", res)
    _log(f"drive done: {nclips} clips in {time.time() - t0:.0f} s -> {out / 'results.md'}")
    return 0


def _f(x, nd=3):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "nan"
    return f"{x:.{nd}g}"


def write_md(path: Path, res: dict) -> None:
    m = res["meta"]
    L = [f"# Q10 Concept-Sliders LoRA on SA3 medium: {m['concept']}{' (DRY RUN, numbers meaningless)' if m['dry_run'] else ''}",
         "", f"{m['finished']}; {m['clips']} clips; {len(m['prompts'])} TADA holdout prompts x seed 0 ({m['seed_rule']}); "
         f"5 strengths = sign x a_c x {list(FRACTIONS)}; a_c from the LPAPS probe to the PCI cutoff of {m['cutoff_from']} "
         "(slider: lora_strength, pair vector: alpha), control at calibrated gain.", "",
         "## Knob grid", "", "| knob | score | direction | a_c | Spearman med | LPAPS top/cut | delta top | delta @ matched vs vector "
         "(vector) | delta @ matched vs control (control, sign) |", "|---|---|---|---|---|---|---|---|---|"]
    for r in res["rows"]:
        vv = (f"{_f(r.get('delta_at_matched_vs_vector'))} ({_f(r.get('vector_delta_at_matched'))})"
              if "delta_at_matched_vs_vector" in r else "")
        cc = (f"{_f(r.get('delta_at_matched_vs_control'))} ({_f(r.get('control_delta_at_matched'))}, {r.get('control_sign')})"
              if "delta_at_matched_vs_control" in r else "")
        L.append(f"| {r['knob']} | {r['score']} | {r['direction']} | {_f(r['a_c'])} | {_f(r['rho_median'])} | "
                 f"{_f(r['lpaps_top_over_cut'])} | {_f(r['delta_top'])} | {vv} | {cc} |")
    L += ["", "Deltas direction-signed (positive = intended way), mean over clips vs strength 0; matched LPAPS = min(cutoff, "
          "both knobs' largest LPAPS), interpolated in LPAPS.", "", "## Stacking (all members on their pos side)", "",
          "Members: " + "; ".join(f"{k} a={_f(v['a'])} cut={_f(v['cut'])} label={v['label']}" for k, v in m["members"].items()),
          "", "| rule | combo | LPAPS | cutoff (strictest) | LPAPS/cut | own-label delta (retained vs alone) | all preserved |",
          "|---|---|---|---|---|---|---|"]
    for s in res["stack"]:
        eff = "; ".join(f"{k} {_f(v['delta'])} ({_f(v['retained'], 2)})" for k, v in s["effects"].items())
        L.append(f"| {s['rule']} | {'+'.join(s['members'])} | {_f(s['lpaps'])} | {_f(s['cutoff'])} | {_f(s['lpaps_over_cut'])} | "
                 f"{eff} | {_f(s['all_preserved_frac'], 2)} |")
    L += ["", "all preserved = share of clips where every member's own label moved its requested way (Measured Sliders)."]
    if m["notes"]:
        L += ["", "## Notes", ""] + [f"- {n}" for n in m["notes"]]
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


# ----------------------------------------------------------------------------------------------- independent rescoring

def indep_sum(args) -> int:
    """Like pairs_s2 indep-sum, for the slider, the pair vector and the control on this run's renders."""
    import pairs_s2 as P2
    import rescore_independent as RI
    import smoke_drive as SDR

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
    concept = res["meta"]["concept"]
    fcols, icols = P2.INDEP[concept]
    sw = res["sweeps"]
    ncl = len(res["meta"]["prompts"])

    def summ(vid, sign, m, direction):
        rec = sw[f"{vid}|{SDR.SIDES[sign]}"]
        alphas = rec["alphas"]
        if m in P2.INDEP_FIT_COLS:
            ref = np.asarray(res["ref"][m], float)
            mat = np.array([rec["scores"][_akey(a)][m] for a in alphas], float)
        else:
            ref = np.array([clips[("ref", 0, 0.0, i)][m] for i in range(ncl)])
            mat = np.array([[clips[(vid, sign, float(a), i)][m] for i in range(ncl)] for a in alphas])
        lp = [float(np.mean(rec["lpaps"][_akey(a)])) for a in alphas]
        d = direction * (mat - ref[None])
        xs = np.array([0.0] + [abs(x) for x in alphas])
        rho = [direction * RI._spearman(xs, np.concatenate([[ref[i]], mat[:, i]])) for i in range(mat.shape[1])]
        return {"delta_top": float(np.nanmean(d[-1])), "rho_med": float(np.nanmedian(rho)), "frac_top": float(np.mean(d[-1] > 0)),
                "lp": lp, "dcurve": np.nanmean(d, axis=1).tolist(), "cut": rec["cutoff"]}

    knobs = ["slider", f"pair_{concept}"] + ([res["meta"]["control"]] if res["meta"]["control"] else [])
    table = []
    for direction in (1, -1):
        for m in fcols + icols:
            base = summ("slider", direction, m, direction)
            for vid in knobs:
                if vid == res["meta"]["control"]:
                    s, cs = max([(summ(vid, c, m, direction), c) for c in (1, -1)], key=lambda x: x[0]["delta_top"])
                else:
                    s, cs = summ(vid, direction, m, direction), direction
                lstar = min(base["cut"], max(base["lp"]), max(s["lp"]))
                table.append({"direction": P2.SIDES_TXT[direction], "score": m, "kind": "fit-target" if m in fcols else "independent",
                              "knob": vid, "knob_sign": SDR.SIDES[cs], "delta_top": s["delta_top"], "rho_med": s["rho_med"],
                              "frac_top": s["frac_top"], "lpaps_top_over_cut": s["lp"][-1] / s["cut"], "matched_lpaps": lstar,
                              "delta_at_matched_with_slider": SDR.delta_at(s["lp"], s["dcurve"], lstar)})
    (out / "independent.json").write_text(json.dumps(table, indent=1, default=float))
    L = ["# Q10 slider vs pair vector vs control: fit-target and independent scorers", "",
         "Direction-signed. delta top = mean at the top strength vs 0; rho = median per-clip Spearman over the 6 strengths "
         "incl. 0; frac = clips moved the right way at the top; delta @ matched = at the LPAPS level matched with the "
         "slider's (min of cutoff and both max LPAPS).", "",
         "| direction | kind | score | knob (sign) | delta top | rho | frac | LPAPS/cut | delta @ matched w/ slider |",
         "|---|---|---|---|---|---|---|---|---|"]
    for r in table:
        L.append(f"| {r['direction']} | {r['kind']} | {r['score']} | {r['knob']} ({r['knob_sign']}) | {_f(r['delta_top'])} | "
                 f"{_f(r['rho_med'], 2)} | {r['frac_top']:.2f} | {r['lpaps_top_over_cut']:.2f} | {_f(r['delta_at_matched_with_slider'])} |")
    (out / "independent.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"wrote {out / 'independent.md'}")
    return 0


# ----------------------------------------------------------------------------------------------- main

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["train", "drive", "indep-sum"])
    ap.add_argument("--out", default=str(Q10))
    ap.add_argument("--s2", default=str(S2))
    ap.add_argument("--concept", default=None, help="default: S2 concept with the best held-out pair-separation AUC")
    ap.add_argument("--tag", default="", help="suffix for cache / log / LoRA names (dry runs)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--duration", type=float, default=10.0)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--mem-gib", type=float, default=6.0)
    # train
    ap.add_argument("--n-pairs", type=int, default=0, help="limit pairs (dry run)")
    ap.add_argument("--collect-pairs", type=int, default=4, help="pairs per collect generate call")
    ap.add_argument("--recollect", action="store_true")
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--alpha", type=float, default=8.0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--eta", type=float, default=1.0)
    ap.add_argument("--batch", type=int, default=4, help="train records per step / drive clips per generate call")
    ap.add_argument("--minutes", type=float, default=60.0)
    ap.add_argument("--max-steps", type=int, default=100000)
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--val-records", type=int, default=96)
    # drive
    ap.add_argument("--lora", default=None)
    ap.add_argument("--n-prompts", type=int, default=12)
    ap.add_argument("--stack-knobs", nargs=2, default=STACK_KNOBS)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--eval-python", default="E:/Projects/tada-replication/evalenv/Scripts/python.exe")
    ap.add_argument("--tada-root", default="E:/Projects/tada-replication")
    ap.add_argument("--label-workers", type=int, default=8)
    args = ap.parse_args()
    if args.cmd == "train":
        return train(args)
    if args.cmd == "drive":
        return drive(args)
    return indep_sum(args)


if __name__ == "__main__":
    raise SystemExit(main())
