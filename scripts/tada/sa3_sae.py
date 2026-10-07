"""TADA SAE pipeline on Stable Audio 3 medium (offline, eager).

Runs TADA's sparse-autoencoder recipe (Staniszewski et al., arXiv
2602.11910, App. I.1.4 and I.2; reference luk-st/steer-audio, MIT) on SA3
medium with the family-generic pieces in ``acestep.tada.sae`` and the SA3
target in ``acestep.engine.sa3_tada``:

    time    one timed caching batch and one timed training slice, with the
            extrapolated cost of the full run (write it down before running)
    cache   cross-attention outputs of the localised blocks over MusicCaps
            captions, every token, every ``--every``-th step, conditional
            pass, during generation
    train   the (m, k) sweep per block, held-out FVU per noise bucket with
            the absolute gate, the configuration choice
    score   per-concept TF-IDF feature means over TADA's contrastive prompts

Sampling setting (SA3's equivalent of TADA's 30-step CFG 5): SA3 medium is
post-trained for ``cfg_scale = 1`` with its own few-step sampler, so every
forward is one conditional pass and the step count is SA3's (``--steps``,
default the value the SA3 lane's CAA collection uses). TADA's ACE cache
keeps 5 of 30 steps (every 6th) and its Stable Audio Open cache 10 of 100;
SA3's equivalent at 8 steps is every step (``--every 1``, the default).
Checkpoint: the served SA3 medium (ARC post-trained, ``stable-audio-3-
medium``, the same 8-step CFG-free render the CAA vectors used), not the
base model. Every cached sample and every per-step quantity (score tables,
vectors) carries the step's noise level ``sigma`` next to its step index.

Token means and training samples use AUDIO tokens only: SA3 prepends 64
learned memory tokens inside every block's sequence (the reference models
have none), and padding is excluded when present
(:mod:`acestep.engine.sa3_tada_tokens`). On SA3, MuQ is the primary
alignment metric for every decision (CLAP barely separates even the real
positive and negative prompts on this model; status_tada_skeptic.md A).

Outputs go under ``--root`` (default E:/Projects/tada-replication).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(_REPO_ROOT), str(_REPO_ROOT / "scripts" / "sa3")):
    while _p in sys.path:
        sys.path.remove(_p)
sys.path.insert(0, str(_REPO_ROOT / "scripts" / "sa3"))
sys.path.insert(0, str(_REPO_ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

FAMILY = "sa3"
CHECKPOINT = "medium"
DEFAULT_ROOT = Path("E:/Projects/tada-replication")
#: Heavy outputs (activation cache, SAEs, score tables, eval audio).
DEFAULT_WORK = Path("D:/tada-replication")
MUSICCAPS = "data/musiccaps-public.csv"


# ---------------------------------------------------------------------------
# model
# ---------------------------------------------------------------------------

def load_sam():
    from sa3_reference_generate import checkpoint_dir, load_local_model

    sam = load_local_model(checkpoint_dir(CHECKPOINT), device="cuda", model_half=True)
    sam.model.eval()
    return sam


class SigmaProbe:
    """Noise level of the current DiT forward (its ``t`` argument)."""

    def __init__(self, sam):
        from acestep.engine.sa3_internals import trunk_module

        self.sigma = float("nan")
        self._h = trunk_module(sam).register_forward_pre_hook(self._pre, with_kwargs=True)

    def _pre(self, _m, args, kwargs):
        t = kwargs.get("t", args[1] if len(args) > 1 else None)
        if t is not None:
            self.sigma = float(torch.as_tensor(t).flatten()[0])
        return None

    def __call__(self) -> float:
        return self.sigma

    def remove(self) -> None:
        self._h.remove()


def hook_target(sam, hook: str):
    """An :class:`~acestep.tada.target.ActivationTarget` exposing ``hook``
    on SA3 (the site the SA3 lane's ``SITE:`` line names). Uses the SA3
    lane's resolver when it provides one; otherwise ``cross_attn_output``
    (each block's ``cross_attn`` module) or ``post_block_residual`` (each
    trunk block's output)."""
    from acestep.engine import sa3_tada
    from acestep.tada import ModuleTarget

    for name in ("hook_target", "site_target"):
        fn = getattr(sa3_tada, name, None)
        if fn is not None:
            return fn(sam, hook)
    if hook == "cross_attn_output":
        return sa3_tada.sa3_target(sam)
    if hook == "post_block_residual":
        return ModuleTarget({hook: list(sa3_tada.sa3_blocks(sam))})
    raise SystemExit(f"no SA3 module mapping for hook {hook!r}")


def make_generate(sam, *, duration: float, steps: int):
    from acestep.engine.sa3_tada import generate

    def _gen(prompts, seed):
        return generate(sam, list(prompts), seed=int(seed), duration=duration, steps=steps)
    return _gen


def captions(root: Path) -> list:
    import pandas as pd

    return pd.read_csv(root / MUSICCAPS)["caption"].astype(str).tolist()


def prompt_order(n: int, seed: int = 42) -> list:
    """The reference shuffles the caption set with seed 42 before caching."""
    return np.random.default_rng(seed).permutation(n).tolist()


# ---------------------------------------------------------------------------
# commands
# ---------------------------------------------------------------------------

def cmd_cache(args, *, limit=None, store_dir=None, sam=None) -> dict:
    from acestep.tada.sae import ActivationStore, TokenRecorder, cache_generations
    from acestep.tada.target import forward_counter

    root = Path(args.root)
    caps = captions(root)
    order = prompt_order(len(caps))
    prompts = [caps[i] for i in order]
    if limit is not None:
        prompts = prompts[:limit]
    store = ActivationStore(store_dir or (Path(args.work) / "sae_cache" / FAMILY))
    cfg = {
        "family": FAMILY, "checkpoint": CHECKPOINT, "blocks": args.blocks,
        "hook": args.hook, "tokens": "audio only (64 memory tokens and padding excluded)",
        "steps": args.steps, "every": args.every,
        "duration": args.duration, "cfg_scale": 1.0, "batch": args.batch,
        "prompts": "MusicCaps captions, shuffled seed 42", "n_prompts": len(prompts),
        "caption_index": [int(i) for i in order[:len(prompts)]],
        "seed": "1000 + batch index",
    }
    store.write_config(cfg)
    start = store.shard_count(args.blocks[0]) * args.batch
    sam = sam if sam is not None else load_sam()
    from acestep.engine.sa3_tada_tokens import sa3_audio_tokens

    target = hook_target(sam, args.hook)
    probe = SigmaProbe(sam)
    audio = sa3_audio_tokens(sam)
    gen = make_generate(sam, duration=args.duration, steps=args.steps)
    seeds = [1000 + j for j in range((len(prompts) + args.batch - 1) // args.batch)]
    t0 = time.perf_counter()
    times = []

    def on_batch(i, n):
        times.append(time.perf_counter())
        if len(times) % 20 == 0:
            rate = (times[-1] - t0) / len(times)
            left = (len(prompts) - i - n) / args.batch * rate
            print(f"[cache] {i + n}/{len(prompts)} {rate:.2f}s/batch eta {left / 60:.1f} min",
                  flush=True)

    with torch.no_grad():
        done = cache_generations(
            gen, prompts,
            lambda: TokenRecorder(target, args.blocks, every=args.every, hook=args.hook,
                                  context=forward_counter(1), sigma_fn=probe, tokens=audio),
            store, seeds=seeds, batch_size=args.batch, start=start, on_batch=on_batch,
        )
    probe.remove()
    audio.remove()
    secs = time.perf_counter() - t0
    return {"cached_prompts": done, "seconds": secs, "store": str(store.root)}


def _sweep(args):
    from acestep.tada.sae import SaeConfig, sweep_configs

    if args.ms or args.ks:
        ms = args.ms or [2, 4, 8, 16]
        ks = args.ks or [16, 32, 64]
        return [SaeConfig(expansion_factor=m, k=k, batch_topk=True) for m in ms for k in ks]
    return sweep_configs()


def split_prompts(meta_prompt: np.ndarray, frac: float = 0.05, seed: int = 0):
    ids = np.unique(meta_prompt)
    rng = np.random.default_rng(seed)
    held = set(rng.choice(ids, size=max(1, int(round(frac * len(ids)))), replace=False).tolist())
    return [int(i) for i in ids if int(i) not in held], sorted(int(i) for i in held)


def _normalize_steps(tr, tr_steps, va, va_steps) -> dict:
    """Divide every sample by its denoise step's RMS (over all train
    tokens and channels of that step), in place; returns ``{step: rms}``.
    Control for the per-sigma variance spread (not part of TADA)."""
    rms = {}
    for s in sorted(set(int(x) for x in tr_steps)):
        idx = np.nonzero(tr_steps == s)[0]
        acc, n = 0.0, 0
        for i in range(0, len(idx), 512):
            x = tr[torch.from_numpy(idx[i:i + 512])].float()
            acc += float(x.pow(2).sum())
            n += x.numel()
        rms[s] = (acc / n) ** 0.5
    for acts, steps in ((tr, tr_steps), (va, va_steps)):
        for s, r in rms.items():
            idx = torch.from_numpy(np.nonzero(steps == s)[0])
            for i in range(0, len(idx), 512):
                j = idx[i:i + 512]
                acts[j] = (acts[j].float() / r).to(acts.dtype)
    return {str(k): v for k, v in rms.items()}


def cmd_train(args, *, store_dir=None, out_dir=None, max_steps=None) -> dict:
    from acestep.tada.sae import (
        ActivationStore, Sae, SaeTrainer, TrainConfig, absolute_bucket_gate, choose_config,
        config_name, fvu_by_bucket,
    )

    root = Path(args.root)
    store = ActivationStore(store_dir or (Path(args.work) / "sae_cache" / FAMILY))
    out_root = Path(out_dir or args.sae_dir or (Path(args.work) / "sae" / FAMILY))
    prev = out_root / "sweep.json"
    report = (json.loads(prev.read_text(encoding="utf-8"))
              if out_dir is not False and prev.exists() else {})
    for block in args.blocks:
        acts_all, meta = store.load(block)
        train_ids, held_ids = split_prompts(meta["prompt"])
        held_mask = np.isin(meta["prompt"], held_ids)
        tr = acts_all[torch.from_numpy(~held_mask)]
        va = acts_all[torch.from_numpy(held_mask)]
        va_steps = meta["step"][held_mask]
        del acts_all
        step_rms = None
        if args.step_rms_norm:
            step_rms = _normalize_steps(tr, meta["step"][~held_mask], va, va_steps)
        if args.gpu_resident:
            tr = tr.to(args.device)
        d_in = tr.shape[-1]
        cfgs = _sweep(args)
        torch.manual_seed(0)
        saes = {config_name(c): Sae(d_in, c, device=args.device) for c in cfgs}
        tcfg = TrainConfig(lr=args.lr, num_epochs=args.epochs, device=args.device)
        print(f"[train] block {block}: {tr.shape[0]} train / {va.shape[0]} held-out samples, "
              f"seq {tr.shape[1]}, d {d_in}, {len(saes)} SAEs", flush=True)
        t0 = time.perf_counter()
        stats = SaeTrainer(saes, tcfg).fit(tr, log=lambda m: print(f"[train] {m}", flush=True),
                                           max_steps=max_steps)
        rows = []
        for name, sae in saes.items():
            bf = fvu_by_bucket(sae, va, va_steps, device=args.device)
            st = stats[name]
            row = {
                "name": name, "m": sae.cfg.expansion_factor, "k": sae.cfg.k,
                "train_fvu": st.fvu, "val_fvu": bf["all"],
                "val_fvu_by_step": {str(k): v for k, v in bf.items() if k != "all"},
                "dead_pct": st.dead_pct, "fire_pct": st.fire_pct,
                "high_freq_pct": st.high_freq_pct, "l0": st.l0, "steps": st.steps,
                "gate": absolute_bucket_gate(bf),
            }
            rows.append(row)
            if out_dir is not False:
                sae.save_to_disk(out_root / f"block_{block}" / name)
        names = {r["name"] for r in rows}
        old = report.get(str(block), {}).get("rows", [])
        rows = [r for r in old if r["name"] not in names] + rows
        best = choose_config(rows)
        report[str(block)] = {
            "rows": rows, "chosen": best["name"] if best else None, "step_rms": step_rms,
            "seconds": time.perf_counter() - t0, "train_samples": int(tr.shape[0]),
            "held_out_prompts": len(held_ids), "epochs": args.epochs, "lr": args.lr,
        }
        if out_dir is not False:
            out_root.mkdir(parents=True, exist_ok=True)
            (out_root / "sweep.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        del tr, va
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return report


def cmd_time(args) -> dict:
    """One caching batch and one training slice, timed, plus the
    extrapolated full-run cost."""
    import tempfile

    from acestep.tada.sae import ActivationStore

    root = Path(args.root)
    tmp = Path(tempfile.mkdtemp(prefix="sae_time_", dir=str(args.work)))
    out = {"batch": args.batch, "steps": args.steps, "every": args.every,
           "duration": args.duration, "blocks": args.blocks}
    sam = load_sam()
    # Batch 1 warms up; batch 2 is timed on its own.
    cmd_cache(args, limit=args.batch, store_dir=tmp / "cache", sam=sam)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    cmd_cache(args, limit=2 * args.batch, store_dir=tmp / "cache", sam=sam)
    torch.cuda.synchronize()
    warm = time.perf_counter() - t0
    del sam
    torch.cuda.empty_cache()
    store = ActivationStore(tmp / "cache")
    shards = sorted((tmp / "cache" / f"block_{args.blocks[0]}").glob("shard_*.npy"))
    acts, meta = store.load(args.blocks[0])
    n_caps = len(captions(root))
    per_prompt_samples = acts.shape[0] / (2 * args.batch)
    sample_bytes = acts[0].numel() * 2
    out.update({
        "seq": int(acts.shape[1]), "d": int(acts.shape[2]),
        "samples_per_prompt": per_prompt_samples,
        "steps_recorded": sorted(set(int(x) for x in meta["step"])),
        "sigmas_recorded": sorted(set(round(float(x), 4) for x in meta["sigma"])),
        "cache_seconds_one_warm_batch": warm,
        "tmp_dir": str(tmp),
    })
    # Training slice: 50 steps of the whole sweep on the cached samples, repeated.
    reps = max(1, int(np.ceil(50 * 4096 / acts.shape[1] / acts.shape[0])))
    big = acts.repeat(reps, 1, 1)
    t0 = time.perf_counter()
    from acestep.tada.sae import Sae, SaeTrainer, TrainConfig, config_name
    cfgs = _sweep(args)
    saes = {config_name(c2): Sae(acts.shape[-1], c2, device="cuda") for c2 in cfgs}
    SaeTrainer(saes, TrainConfig(num_epochs=1)).fit(big, max_steps=60)
    torch.cuda.synchronize()
    train60 = time.perf_counter() - t0
    tok_per_step = (4096 // acts.shape[1]) * acts.shape[1]
    total_samples = per_prompt_samples * n_caps
    steps_per_epoch = total_samples * acts.shape[1] / tok_per_step
    out.update({
        "train_seconds_60_steps_whole_sweep": train60,
        "sweep_size": len(saes),
        "steps_per_epoch_full": steps_per_epoch,
        "est_cache_hours_full": warm / 3600 * n_caps / args.batch,
        "est_train_hours_per_block_per_epoch_whole_sweep": train60 / 60 * steps_per_epoch / 3600,
        "cache_gb_per_block": total_samples * sample_bytes / 1e9,
        "n_captions": n_caps, "shards_written": len(shards),
    })
    return out


def _chosen(entry: dict) -> str:
    """The block's SAE: the sweep's choice (paper criterion plus the
    absolute per-sigma-bucket gate) when one passes; otherwise the paper's
    criterion alone (lowest held-out FVU with dead <= 1 percent and fire
    <= 25 percent), the gate failure being recorded in the sweep and the
    status file; lowest held-out FVU if nothing meets even that."""
    if entry.get("chosen"):
        return entry["chosen"]
    from acestep.tada.sae import choose_config

    rows = [{**r, "gate": True} for r in entry["rows"]]
    best = choose_config(rows, bar=float("inf"))
    return (best or min(rows, key=lambda r: r["val_fvu"]))["name"]


def cmd_score(args) -> dict:
    """Feature means over TADA's contrastive prompts per concept (Eq. 12
    inputs), every step, for the chosen SAE of each block."""
    from acestep.tada import concepts
    from acestep.tada.sae import FeatureMeanRecorder, Sae, score_tables
    from acestep.tada.target import forward_counter

    root = Path(args.work)
    sweep = json.loads((_sae_root(args) / "sweep.json").read_text(encoding="utf-8"))
    saes = {}
    for b in args.blocks:
        name = args.sae or _chosen(sweep[str(b)])
        saes[b] = Sae.load_from_disk(_sae_root(args) / f"block_{b}" / name, device="cuda")
    sam = load_sam()
    from acestep.engine.sa3_tada_tokens import sa3_audio_tokens

    target = hook_target(sam, args.hook)
    audio = sa3_audio_tokens(sam)
    probe = SigmaProbe(sam)
    gen = make_generate(sam, duration=args.score_duration, steps=args.steps)
    out_dir = root / "sae_scores" / (FAMILY + args.tag)
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {}
    for concept in args.concepts:
        pos, neg, _lyrics = concepts.prompt_pairs(concept)
        means = {}
        for label, prompts in (("pos", pos), ("neg", neg)):
            scale = {b: sweep[str(b)]["step_rms"] for b in args.blocks
                     if sweep[str(b)].get("step_rms")}
            rec = FeatureMeanRecorder(target, saes, hook=args.hook, context=forward_counter(1),
                                      tokens=audio, sigma_fn=probe, step_scale=scale)
            with rec, torch.no_grad():
                for i in range(0, len(prompts), args.batch):
                    rec.reset_steps()
                    # Reference: seed 10, one shared latent batch.
                    gen(prompts[i:i + args.batch], 10)
            means[label] = rec.result()
            sigmas = rec.step_sigmas()
        payload = {}
        for b in args.blocks:
            t = score_tables(means["pos"][b], means["neg"][b])
            payload[str(b)] = {k: v.cpu() for k, v in t.items()}
            payload[str(b)]["mean_neg"] = means["neg"][b].cpu()
        torch.save({"concept": concept, "tables": payload,
                    "sae": {str(b): args.sae or _chosen(sweep[str(b)]) for b in args.blocks},
                    "steps": args.steps, "sigmas": sigmas,
                    "duration": args.score_duration, "seed": 10},
                   out_dir / f"{concept}.pt")
        report[concept] = {str(b): payload[str(b)]["tfidf"].shape[0] for b in args.blocks}
        print(f"[score] {concept} done", flush=True)
    return report


def cmd_vectors(args) -> dict:
    """v_SAE (Eq. 13) for every concept and ``k_c`` in ``--k-grid``, in
    the per-step form (top-``k_c`` per step, the reference) and the pooled
    single-vector form (the paper's equation), written where the SA3
    lane's driver reads vectors: ``<root>/sae_sel/<form>_k<k>/caa/
    <concept>.sae.pt`` as ``{"vectors": {step: {block: [H]}}}``. The same
    ``k_c`` is used for every localised block (TADA sweeps a pair of
    blocks; three blocks would make the grid 6**3)."""
    from acestep.tada.sae import Sae, concept_vectors

    root = Path(args.work)
    sweep = json.loads((_sae_root(args) / "sweep.json").read_text(encoding="utf-8"))
    saes = {b: Sae.load_from_disk(_sae_root(args) / f"block_{b}" /
                                  (args.sae or _chosen(sweep[str(b)])))
            for b in args.blocks}
    out = {}
    for concept in args.concepts:
        d = torch.load(root / "sae_scores" / (FAMILY + args.tag) / f"{concept}.pt", weights_only=False)
        tab = d["tables"]
        mp = {b: tab[str(b)]["mean_pos"] for b in args.blocks}
        mn = {b: tab[str(b)]["mean_neg"] for b in args.blocks}
        n_steps = int(next(iter(mp.values())).shape[0])
        sigmas = list(d.get("sigmas") or [float("nan")] * n_steps)
        for form in args.forms:
            for k in args.k_grid:
                v = concept_vectors(saes, mp, mn, {b: k for b in args.blocks},
                                    per_step=(form == "perstep"))
                vec = {s: {b: (v[b][s] if v[b].ndim == 2 else v[b]).float()
                           for b in args.blocks} for s in range(n_steps)}
                dst = root / "sae_sel" / f"{form}_k{k}{args.tag}" / "caa" / f"{concept}.sae.pt"
                dst.parent.mkdir(parents=True, exist_ok=True)
                torch.save({"vectors": vec, "k": int(k), "form": form, "blocks": args.blocks,
                            "sigmas": {s: sigmas[s] for s in range(n_steps)},
                            "sae": {str(b): args.sae or _chosen(sweep[str(b)])
                                    for b in args.blocks}}, dst)
                out[f"{concept}/{form}/k{k}"] = round(float(
                    torch.stack([vec[0][b].norm() for b in args.blocks]).mean()), 3)
    return out


# ---------------------------------------------------------------------------
# evaluation through the SA3 lane's driver and protocol layout
# ---------------------------------------------------------------------------

def _run_module(args):
    """The SA3 lane's driver (``sa3_tada_run.py``) with the MECHANISM
    line's renorm and guidance, so SAE vectors render exactly like CAA."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import sa3_tada_run as R

    R._RENORM = bool(args.renorm)
    if hasattr(R, "_GUIDANCE"):
        R._GUIDANCE = float(args.guidance)
    R.STEPS = int(args.steps)
    return R


def _sel_vectors(args, concept, form, k):
    root = Path(args.work)
    d = torch.load(root / "sae_sel" / f"{form}_k{k}{args.tag}" / "caa" / f"{concept}.sae.pt",
                   weights_only=False)
    return d["vectors"]


def _vec_scale(vectors, blocks) -> float:
    return float(torch.stack([vectors[0][b].float().norm() for b in blocks]).mean())


_TAG = {"tag": ""}


def _site(form: str, k: int) -> str:
    """Directory site token (no underscore: the scorer splits on '_');
    a tagged variant (``--tag``) appends its letters."""
    return f"{form[0]}{int(k)}{_TAG['tag'].replace('_', '')}"


def _sae_root(args) -> Path:
    return Path(args.sae_dir) if args.sae_dir else Path(args.work) / "sae" / FAMILY


def cmd_probe(args) -> dict:
    """Calibration probes on the held-out prompts: strengths chosen so
    that ``alpha * |v_SAE|`` spans ``--probe-mults`` (CAA's unit-vector
    probes reach its cutoffs between 4 and 32), into
    ``<sa3>/calib_sae/sae_<form><k>_<concept>/alpha_*``."""
    from acestep.tada import concepts as C

    R = _run_module(args)
    _copy_pci(args, Path(args.sa3_out) / "calib_sae")
    sam = R._load_sam()
    prompts, _ = C.benchmark_prompts(holdout=True)
    out = {}
    for concept in args.concepts:
        for form in args.forms:
            for k in args.k_grid:
                vec = _sel_vectors(args, concept, form, k)
                scale = _vec_scale(vec, args.blocks)
                root = Path(args.sa3_out) / "calib_sae" / f"sae_{_site(form, k)}_{concept}"
                alphas = [0.0] + [round(sg * m / scale, 4)
                                  for m in args.probe_mults for sg in (1, -1)]
                for a in alphas:
                    d = root / R._fmt_alpha(a)
                    if (d / "audios.npz").exists():
                        continue
                    audio = R._render_alpha(sam, prompts, vec, args.blocks, a, args.batch)
                    R._save_npz(d, audio, [f"p{i:03d}.wav" for i in range(len(prompts))],
                                mono=True)
                meta = json.dumps({"concept": concept, "form": form, "k": k,
                                   "scale": scale, "alphas": alphas}, indent=1)
                (root / "probe.json").write_text(meta)
                # The SA3 scorer treats a directory as complete once sweep.json exists.
                (root / "sweep.json").write_text(meta)
                out[f"{concept}/{form}/k{k}"] = scale
                print(f"[probe] {root.name} scale {scale:.2f}", flush=True)
    return out


def cmd_calibrate(args) -> dict:
    """Per concept / form / k, the smallest probed ``|alpha|`` per
    direction whose mean LPAPS reaches the concept's PCI cutoff (the SA3
    lane's rule and its eval50 cutoffs); the largest probe otherwise."""
    import csv

    sa3 = Path(args.sa3_out)

    def lp(d):
        for name in ("lpaps.csv", "lpaps_endpoints.csv"):
            f = d / "protocol_results" / name
            if f.exists():
                with f.open() as fh:
                    return {float(r["alpha"]): float(r["mean"]) for r in csv.DictReader(fh)}
        return None

    ranges, notes = {}, {}
    for concept in args.concepts:
        cut = {}
        for direction, sign in (("pos", 1), ("neg", -1)):
            vals = []
            for site in ("all", "loc"):
                t = lp(Path(args.pci_src) / f"pci_{site}_{concept}")
                if t:
                    vals.append(max(v for a, v in t.items() if sign * a > 0))
            cut[direction] = min(vals) if vals else None
        for form in args.forms:
            for k in args.k_grid:
                pr = lp(sa3 / "calib_sae" / f"sae_{_site(form, k)}_{concept}")
                if not pr or None in cut.values():
                    continue
                rng = []
                for direction, sign in (("neg", -1), ("pos", 1)):
                    probes = sorted((abs(a), v) for a, v in pr.items() if sign * a > 0)
                    hit = [a for a, v in probes if v >= cut[direction]]
                    a = hit[0] if hit else probes[-1][0]
                    rng.append(sign * a)
                    notes[f"{concept}/{_site(form, k)}/{direction}"] = {
                        "cutoff": cut[direction], "alpha": sign * a, "reached": bool(hit)}
                ranges.setdefault(concept, {})[_site(form, k)] = rng
    (sa3 / "ranges_sae.json").write_text(json.dumps(ranges, indent=1))
    (sa3 / "ranges_sae_notes.json").write_text(json.dumps(notes, indent=1))
    return ranges


def cmd_sweep(args) -> dict:
    """The scaled protocol's strength sweep for SAE vectors, in the SA3
    lane's layout ``<sa3>/<sub>/sae_<site>_<concept>/alpha_*`` (site =
    form initial + k_c, or ``--site-name``), with the eval50 PCI
    ``protocol_results`` copied in so the scorer finds the cutoffs."""
    from acestep.tada import concepts as C
    from acestep.tada.caa import alphas_from_range

    R = _run_module(args)
    sa3 = Path(args.sa3_out)
    sub = sa3 / args.sub
    sub.mkdir(parents=True, exist_ok=True)
    _copy_pci(args, sub)
    ranges = json.loads((sa3 / "ranges_sae.json").read_text())
    prompts, _ = C.benchmark_prompts(holdout=args.holdout)
    prompts = prompts[: args.n_prompts]
    sam = R._load_sam()
    done = {}
    sel = _selection(args)
    for concept in args.concepts:
        for form in args.forms:
            for k in ([sel[concept]["k"]] if sel else args.k_grid):
                site = _site(form, k)
                rng = ranges[concept][site]
                alphas = alphas_from_range(float(rng[0]), float(rng[1]), args.points)
                vec = _sel_vectors(args, concept, form, k)
                root = sub / f"sae_{args.site_name or site}_{concept}"
                for a in alphas:
                    d = root / R._fmt_alpha(a)
                    if (d / "audios.npz").exists():
                        continue
                    audio = R._render_alpha(sam, prompts, vec, args.blocks, a, args.batch)
                    R._save_npz(d, audio, [f"p{i:03d}.wav" for i in range(len(prompts))],
                                mono=True)
                (root / "sweep.json").write_text(json.dumps({
                    "concept": concept, "form": form, "k": k, "blocks": args.blocks,
                    "alphas": alphas, "prompts": len(prompts), "seed": R.EVAL_SEED,
                    "steps": args.steps, "renorm": bool(args.renorm),
                    "guidance": float(args.guidance), "holdout": bool(args.holdout),
                    "range": rng}, indent=1))
                done[f"{concept}/{site}"] = len(alphas)
                print(f"[sweep] {root.name}: {len(alphas)} strengths x {len(prompts)}", flush=True)
    return done


def _copy_pci(args, sub: Path) -> None:
    """The PCI rows' ``protocol_results`` (``--pci-src``, the SA3 lane's
    eval50) into ``sub`` so the scorer finds SA3's own PCI cutoffs and
    reports the PCI AUC beside the SAE rows."""
    import shutil

    sub.mkdir(parents=True, exist_ok=True)
    for pci in Path(args.pci_src).glob("pci_*"):
        dst = sub / pci.name / "protocol_results"
        if not dst.exists() and (pci / "protocol_results").exists():
            shutil.copytree(pci / "protocol_results", dst)


def _selection(args):
    if not args.selection:
        return None
    return json.loads(Path(args.selection).read_text())


def _auc_avg(row: dict, metric: str) -> float:
    return 0.5 * (float(row["pos"][metric]) + float(row["neg"][metric]))


def selection_metric(concept: str) -> str:
    """The paper's benchmark metric per concept (Sec. 4 / App. I): CLAP for
    vocal gender, MuQ-T for everything else (MuQ primary on SA3)."""
    return "clap" if concept == "vocal_gender" else "muqt"


def cmd_select(args) -> dict:
    """``k_c`` per concept on the 20 held-out prompts (paper App. I.2): the
    ``k_c`` whose calibration-grid alignment AUC (mean of the two
    directions, SA3's own PCI cutoffs) is highest, by the concept's
    benchmark metric. Reads ``<sa3-out>/calib_sae/auc.json`` (scorer
    ``auc --sub calib_sae``); writes ``--selection``."""
    auc = json.loads((Path(args.sa3_out) / "calib_sae" / "auc.json").read_text())
    out = {}
    for concept in args.concepts:
        metric = selection_metric(concept)
        rows = {}
        for form in args.forms:
            for k in args.k_grid:
                r = auc.get(concept, {}).get(f"sae_{_site(form, k)}")
                if r and "pos" in r and "neg" in r:
                    rows[(form, k)] = _auc_avg(r, metric)
        if not rows:
            continue
        (form, k), best = max(rows.items(), key=lambda kv: kv[1])
        out[concept] = {"k": int(k), "form": form, "metric": metric, "holdout_auc": best,
                        "all": {f"{f}{kk}": round(v, 4) for (f, kk), v in rows.items()}}
    Path(args.selection).write_text(json.dumps(out, indent=1))
    return out


def cmd_report(args) -> dict:
    """SAE/PCI ratios on the eval prompts: per concept the alignment AUC
    (mean of the two directions) of ``sae_<site>`` and of PCI-all from the
    same ``auc.json`` (scorer ``auc --sub <sub>``), MuQ and CLAP, plus the
    corrected CAA-loc row when ``--caa-auc`` exists. Paper reference
    (Table 1, ACE-Step): SAE-loc / PCI = 0.118 / 0.084 = 1.40 (MuQ)."""
    sub = Path(args.sa3_out) / args.sub
    auc = json.loads((sub / "auc.json").read_text())
    caa = {}
    if args.caa_auc and Path(args.caa_auc).exists():
        caa = json.loads(Path(args.caa_auc).read_text())
    label = f"sae_{args.site_name or 'loc'}"
    rows = {}
    acc = {"sae_muqt": [], "pci_muqt": [], "sae_clap": [], "pci_clap": [], "caa_muqt": []}
    for concept in args.concepts:
        r = auc.get(concept, {})
        if label not in r or "pci_all" not in r:
            continue
        row = {}
        for m in ("muqt", "clap"):
            row[f"sae_{m}"] = _auc_avg(r[label], m)
            row[f"pci_{m}"] = _auc_avg(r["pci_all"], m)
            acc[f"sae_{m}"].append(row[f"sae_{m}"])
            acc[f"pci_{m}"].append(row[f"pci_{m}"])
        row["ratio_muqt"] = row["sae_muqt"] / row["pci_muqt"] if row["pci_muqt"] else float("nan")
        c = caa.get(concept, {}).get("caa_loc")
        if c and "pos" in c and "neg" in c:
            row["caa_loc_muqt"] = _auc_avg(c, "muqt")
            acc["caa_muqt"].append(row["caa_loc_muqt"])
        rows[concept] = row
    mean = {k: float(np.mean(v)) for k, v in acc.items() if v}
    if mean.get("pci_muqt"):
        mean["ratio_muqt_of_means"] = mean["sae_muqt"] / mean["pci_muqt"]
    rep = {"rows": rows, "mean": mean, "n_concepts": len(rows),
           "paper_ref": "ACE-Step Table 1: SAE-loc/PCI 0.118/0.084 = 1.40 (MuQ)"}
    (sub / "sae_report.json").write_text(json.dumps(rep, indent=1))
    return rep


def cmd_pack(args) -> dict:
    """Per-step ``v_SAE`` packs (hook ``cross_attn_output``, renorm off,
    cond pass) for the selected ``k_c``, with the eval strength range as
    the knob magnitude, into ``--packs``."""
    from acestep.tada.sae import sae_pack, write_sae_pack

    sel = _selection(args)
    ranges = json.loads((Path(args.sa3_out) / "ranges_sae.json").read_text())
    out = {}
    for concept in args.concepts:
        s = sel[concept]
        d = torch.load(Path(args.work) / "sae_sel" / f"{s['form']}_k{s['k']}" / "caa" /
                       f"{concept}.sae.pt", weights_only=False)
        vec = d["vectors"]
        steps = sorted(vec)
        v = torch.stack([torch.stack([vec[st][b].float() for st in steps]) for b in args.blocks])
        rng = [float(x) for x in ranges[concept][_site(s["form"], s["k"])]]
        mag = max(abs(x) for x in rng)
        pack = sae_pack(v, family=FAMILY, checkpoint=CHECKPOINT, concept=concept,
                        blocks=args.blocks, k_per_block={b: int(s["k"]) for b in args.blocks},
                        magnitude=mag, renorm=False,
                        policy={"kind": "range", "start": rng[0] / mag, "end": rng[1] / mag},
                        sigmas=[float(d["sigmas"][st]) for st in steps],
                        provenance={"sae": d["sae"], "selection": s, "range": rng,
                                    "negative": "negated positive vector (alpha < 0)"})
        out[concept] = str(write_sae_pack(pack, args.packs))
    return out


def cmd_listen(args) -> dict:
    """Ear package: per concept, four files of one benchmark prompt from
    the same seed (2115) and batch layout: zero (strength 0 of the SAE
    sweep), the real positive prompt (PCI reference at full switch length,
    ``sa3_tada_run.py swap`` into the same eval directory), SAE at the
    strongest positive strength within the PCI cutoff, SAE at the largest
    positive grid strength. Prompt = largest MuQ gain zero -> admitted."""
    import ast

    import pandas as pd
    import soundfile as sf

    sub = Path(args.sa3_out) / args.sub
    auc = json.loads((sub / "auc.json").read_text())
    bench = json.loads((_REPO_ROOT / "acestep" / "tada" / "data" / "benchmark_prompts.json")
                       .read_text(encoding="utf-8"))["test_prompts"]
    sel = _selection(args) or {}
    dst = Path(args.dst)
    dst.mkdir(parents=True, exist_ok=True)
    label = f"sae_{args.site_name or 'loc'}"
    lines = ["# TADA SAE on Stable Audio 3 medium: ear package", "",
             "SAE = sum of the top-k_c TF-IDF SAE decoder rows per step (unit weights), added at the",
             f"cross-attention output of blocks {args.blocks}, cond pass, renorm off. Positive direction.", ""]
    for c in args.concepts:
        d = sub / f"{label}_{c}"
        pr = d / "protocol_results"
        lp = pd.read_csv(pr / "lpaps.csv")
        mq = pd.read_csv(pr / "muqt.csv")
        cut = auc[c][label]["pos"]["cutoff"]
        adm = lp[(lp.alpha > 0) & (lp["mean"] <= cut)]
        a_adm = float(adm.alpha.max()) if len(adm) else float(lp[lp.alpha > 0].alpha.min())
        a_max = float(lp.alpha.max())
        sc = {float(r.alpha): np.array(ast.literal_eval(r.scores)) for r in mq.itertuples()}
        k = int(np.argmax(sc[a_adm] - sc[0.0]))

        def lpm(a):
            return float(lp[lp.alpha == a]["mean"].iloc[0])

        note = "" if len(adm) else " (no positive strength within the cutoff: file 3 is the smallest)"
        lines.append(f"## {c}")
        lines.append(f"Prompt {k}: \"{bench[k]}\". k_c {sel.get(c, {}).get('k', '?')}, MuQ query "
                     f"\"{mq.prompt_used.iloc[0]}\", PCI cutoff (mean LPAPS) {cut:.2f}{note}.")
        files = [("1_zero", d / "alpha_0.0", f"strength 0 (the prompt as written), MuQ {sc[0.0][k]:.3f}"),
                 ("2_pci_real_prompt", sub / f"swap_full_{c}" / "pos",
                  "REFERENCE: the real positive prompt of the PCI triple rendered from step 0 "
                  "(its own wording, same seed and batch layout)"),
                 (f"3_sae_admitted_alpha{a_adm:+g}", d / f"alpha_{a_adm}",
                  f"SAE at the strongest strength within the PCI cutoff (mean LPAPS {lpm(a_adm):.2f}), "
                  f"MuQ {sc[a_adm][k]:.3f}"),
                 (f"4_sae_max_alpha{a_max:+g}", d / f"alpha_{a_max}",
                  f"SAE at the largest grid strength (mean LPAPS {lpm(a_max):.2f}), MuQ {sc[a_max][k]:.3f}")]
        for tag, src, what in files:
            z = np.load(src / "audios.npz")
            x = z["audio"][k].astype(np.float32) / 32768.0
            name = f"{c}_{tag}.wav"
            sf.write(dst / name, x.T, int(z["sr"]))
            lines.append(f"- `{name}`: {what}")
        lines.append("- Listen for: files 3 and 4 moving toward the positive pole relative to file 1, "
                     "against what the real prompt (file 2, the reference) does; the piece should stay "
                     "recognisable in file 3.")
        lines.append("")
    lines += ["All files: SA3 medium (served ARC checkpoint), 8 steps, cfg 1, seed 2115, 10 s mono,",
              "batch 25, so zero, reference and steered files share their initial noise."]
    (dst / "README.md").write_text("\n".join(lines), encoding="utf-8")
    return {"dst": str(dst), "files": len(list(dst.glob("*.wav")))}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("command", choices=("time", "cache", "train", "score", "vectors", "probe",
                                        "calibrate", "select", "sweep", "report", "pack", "listen"))
    ap.add_argument("--root", default=str(DEFAULT_ROOT), help="data root (MusicCaps captions)")
    ap.add_argument("--work", default=str(DEFAULT_WORK),
                    help="cache, SAEs, score tables and selection vectors")
    ap.add_argument("--pci-src", default=str(DEFAULT_ROOT / "sa3" / "eval50"),
                    help="the SA3 lane's PCI rows (cutoffs and the PCI AUC)")
    ap.add_argument("--selection", default=None, help="k_c selection json (select writes it)")
    ap.add_argument("--sae-dir", default=None, help="train: output directory (default <work>/sae/sa3)")
    ap.add_argument("--caa-auc", default=str(DEFAULT_ROOT / "sa3" / "eval_e3" / "auc.json"))
    ap.add_argument("--packs", default=str(DEFAULT_ROOT / "packs" / "sa3_sae"))
    ap.add_argument("--dst", default=str(DEFAULT_ROOT / "listen_sae"))
    ap.add_argument("--blocks", type=int, nargs="+", required=True)
    ap.add_argument("--hook", default="cross_attn_output")
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--every", type=int, default=1)
    ap.add_argument("--duration", type=float, default=10.0)
    ap.add_argument("--score-duration", type=float, default=10.0)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--ms", type=int, nargs="*")
    ap.add_argument("--ks", type=int, nargs="*")
    ap.add_argument("--gpu-resident", action="store_true")
    ap.add_argument("--step-rms-norm", action="store_true",
                    help="train: divide each step's tokens by that step's RMS (control)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--concepts", nargs="*", default=[])
    ap.add_argument("--sae", default=None, help="SAE config name to score with (default: chosen)")
    ap.add_argument("--k-grid", type=int, nargs="*", default=[5, 10, 20, 50, 100, 500])
    ap.add_argument("--forms", nargs="*", default=["pooled", "perstep"])
    ap.add_argument("--sa3-out", default=str(DEFAULT_ROOT / "sa3"))
    ap.add_argument("--renorm", action="store_true")
    ap.add_argument("--guidance", type=float, default=1.0)
    ap.add_argument("--probe-mults", type=float, nargs="*", default=[2, 4, 8, 16, 32, 64, 128])
    ap.add_argument("--sub", default="sae_sel")
    ap.add_argument("--site-name", default=None)
    ap.add_argument("--holdout", action="store_true")
    ap.add_argument("--n-prompts", type=int, default=50)
    ap.add_argument("--points", type=int, default=10)
    ap.add_argument("--json", default=None)
    ap.add_argument("--tag", default="", help="variant suffix for score, vector and site names")
    args = ap.parse_args()
    _TAG["tag"] = args.tag
    fn = {"time": cmd_time, "cache": cmd_cache, "train": cmd_train, "score": cmd_score,
          "vectors": cmd_vectors, "probe": cmd_probe, "calibrate": cmd_calibrate,
          "select": cmd_select, "sweep": cmd_sweep, "report": cmd_report, "pack": cmd_pack,
          "listen": cmd_listen}
    res = fn[args.command](args)
    text = json.dumps(res, indent=2, default=str)
    print(text if len(text) < 4000 else text[:4000])
    if args.json:
        Path(args.json).write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
