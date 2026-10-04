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
``--every 2`` keeps 4 of 8 here.

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
    store = ActivationStore(store_dir or (root / "sae_cache" / FAMILY))
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


def cmd_train(args, *, store_dir=None, out_dir=None, max_steps=None) -> dict:
    from acestep.tada.sae import (
        ActivationStore, Sae, SaeTrainer, TrainConfig, absolute_bucket_gate, choose_config,
        config_name, fvu_by_bucket,
    )

    root = Path(args.root)
    store = ActivationStore(store_dir or (root / "sae_cache" / FAMILY))
    out_root = Path(out_dir or (root / "sae" / FAMILY))
    report = {}
    for block in args.blocks:
        acts_all, meta = store.load(block)
        train_ids, held_ids = split_prompts(meta["prompt"])
        held_mask = np.isin(meta["prompt"], held_ids)
        tr = acts_all[torch.from_numpy(~held_mask)]
        va = acts_all[torch.from_numpy(held_mask)]
        va_steps = meta["step"][held_mask]
        del acts_all
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
        best = choose_config(rows)
        report[str(block)] = {
            "rows": rows, "chosen": best["name"] if best else None,
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
    tmp = Path(tempfile.mkdtemp(prefix="sae_time_", dir=str(root)))
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


def cmd_score(args) -> dict:
    """Feature means over TADA's contrastive prompts per concept (Eq. 12
    inputs), every step, for the chosen SAE of each block."""
    from acestep.tada import concepts
    from acestep.tada.sae import FeatureMeanRecorder, Sae, score_tables
    from acestep.tada.target import forward_counter

    root = Path(args.root)
    sweep = json.loads((root / "sae" / FAMILY / "sweep.json").read_text(encoding="utf-8"))
    saes = {}
    for b in args.blocks:
        name = args.sae or sweep[str(b)]["chosen"]
        saes[b] = Sae.load_from_disk(root / "sae" / FAMILY / f"block_{b}" / name, device="cuda")
    sam = load_sam()
    from acestep.engine.sa3_tada_tokens import sa3_audio_tokens

    target = hook_target(sam, args.hook)
    audio = sa3_audio_tokens(sam)
    gen = make_generate(sam, duration=args.score_duration, steps=args.steps)
    out_dir = root / "sae_scores" / FAMILY
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {}
    for concept in args.concepts:
        pos, neg, _lyrics = concepts.prompt_pairs(concept)
        means = {}
        for label, prompts in (("pos", pos), ("neg", neg)):
            rec = FeatureMeanRecorder(target, saes, hook=args.hook, context=forward_counter(1),
                                      tokens=audio)
            with rec, torch.no_grad():
                for i in range(0, len(prompts), args.batch):
                    rec.reset_steps()
                    # Reference: seed 10, one shared latent batch.
                    gen(prompts[i:i + args.batch], 10)
            means[label] = rec.result()
        payload = {}
        for b in args.blocks:
            t = score_tables(means["pos"][b], means["neg"][b])
            payload[str(b)] = {k: v.cpu() for k, v in t.items()}
            payload[str(b)]["mean_neg"] = means["neg"][b].cpu()
        torch.save({"concept": concept, "tables": payload,
                    "sae": {str(b): args.sae or sweep[str(b)]["chosen"] for b in args.blocks},
                    "steps": args.steps, "duration": args.score_duration, "seed": 10},
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

    root = Path(args.root)
    sweep = json.loads((root / "sae" / FAMILY / "sweep.json").read_text(encoding="utf-8"))
    saes = {b: Sae.load_from_disk(root / "sae" / FAMILY / f"block_{b}" /
                                  (args.sae or sweep[str(b)]["chosen"]))
            for b in args.blocks}
    out = {}
    for concept in args.concepts:
        d = torch.load(root / "sae_scores" / FAMILY / f"{concept}.pt", weights_only=False)
        tab = d["tables"]
        mp = {b: tab[str(b)]["mean_pos"] for b in args.blocks}
        mn = {b: tab[str(b)]["mean_neg"] for b in args.blocks}
        n_steps = int(next(iter(mp.values())).shape[0])
        for form in args.forms:
            for k in args.k_grid:
                v = concept_vectors(saes, mp, mn, {b: k for b in args.blocks},
                                    per_step=(form == "perstep"))
                vec = {s: {b: (v[b][s] if v[b].ndim == 2 else v[b]).float()
                           for b in args.blocks} for s in range(n_steps)}
                dst = root / "sae_sel" / f"{form}_k{k}" / "caa" / f"{concept}.sae.pt"
                dst.parent.mkdir(parents=True, exist_ok=True)
                torch.save({"vectors": vec, "k": int(k), "form": form, "blocks": args.blocks,
                            "sae": {str(b): args.sae or sweep[str(b)]["chosen"]
                                    for b in args.blocks}}, dst)
                out[f"{concept}/{form}/k{k}"] = round(float(
                    torch.stack([vec[0][b].norm() for b in args.blocks]).mean()), 3)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("command", choices=("time", "cache", "train", "score", "vectors"))
    ap.add_argument("--root", default=str(DEFAULT_ROOT))
    ap.add_argument("--blocks", type=int, nargs="+", required=True)
    ap.add_argument("--hook", default="cross_attn_output")
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--every", type=int, default=2)
    ap.add_argument("--duration", type=float, default=10.0)
    ap.add_argument("--score-duration", type=float, default=10.0)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--ms", type=int, nargs="*")
    ap.add_argument("--ks", type=int, nargs="*")
    ap.add_argument("--gpu-resident", action="store_true")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--concepts", nargs="*", default=[])
    ap.add_argument("--sae", default=None, help="SAE config name to score with (default: chosen)")
    ap.add_argument("--k-grid", type=int, nargs="*", default=[5, 10, 20, 50, 100, 500])
    ap.add_argument("--forms", nargs="*", default=["pooled", "perstep"])
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    fn = {"time": cmd_time, "cache": cmd_cache, "train": cmd_train, "score": cmd_score,
          "vectors": cmd_vectors}
    res = fn[args.command](args)
    text = json.dumps(res, indent=2, default=str)
    print(text if len(text) < 4000 else text[:4000])
    if args.json:
        Path(args.json).write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
