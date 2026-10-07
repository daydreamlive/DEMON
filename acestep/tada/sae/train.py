"""SAE training, evaluation and configuration choice as TADA does them.

Port of the reference ``SaeTrainer`` (``src/steering/methods/sae/lib/sae/
trainer.py``) for one process and several SAEs fed the same batches (so a
whole ``(m, k)`` sweep reads the cache once). Kept from the reference:

* one sample = all tokens of one cached activation; batch = ``4096 //
  sample_size`` samples (``effective_batch_size`` 4096 activation vectors),
  samples shuffled once with the run seed (the reference shuffles the
  dataset with seed 42 and then iterates it in order every epoch);
* ``b_dec`` initialised to the geometric median of the first batch;
* decoder rows renormalised before every batch, gradient parallel to them
  removed before every step, grad norm clipped to 1;
* Adam with ``eps = 6.25e-10`` in fp32; linear schedule with ``lr_warmup_
  steps`` = 1000 warmup steps then linear decay to zero;
* loss ``fvu + auxk_alpha * auxk + multi_topk_fvu / 8``, ``auxk_alpha`` =
  1/32, dead mask = latents not fired for more than
  ``dead_feature_threshold`` = 10**7 tokens (each step adds
  ``effective_batch_size`` to every counter, as the reference does).

TADA's ACE-Step settings (``train_ace.py``, paper App. I.2): BatchTopK,
lr 3e-5, 10 or 15 epochs, sweep ``m in {2, 4, 8, 16}``, ``k in {16, 32,
64}``; chosen ``(2, 32)`` for layer 7 and ``(4, 64)`` for layer 8 with FVU
0.216 / 0.243, dead 0.37% / 0.25%, fire 19.4% / 20.1% (Table 18). Their
criterion: "minimizes reconstruction error while keeping the fraction of
dead and high-frequency features low". :func:`choose_config` makes that
explicit with stated thresholds.

Added by this port (the prior DEMON SAE lane's lesson, kept as a hard
gate): :func:`fvu_by_bucket` on HELD-OUT prompts, one bucket per cached
denoise step (one noise level), and :func:`absolute_bucket_gate`, which
fails a dictionary when any bucket's FVU exceeds an absolute bar. A
relative gate alone missed a dictionary whose low-noise buckets diverged
while its training FVU looked excellent.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Mapping, Optional, Sequence

import numpy as np
import torch
from torch import Tensor

from .model import Sae, SaeConfig, geometric_median


@dataclass
class TrainConfig:
    """Reference ``TrainConfig`` fields used by a single-process run, with
    TADA's ACE-Step training values as defaults."""

    effective_batch_size: int = 4096
    lr: float = 3e-5
    lr_warmup_steps: int = 1000
    auxk_alpha: float = 1.0 / 32.0
    dead_feature_threshold: int = 10_000_000
    num_epochs: int = 10
    feature_sampling_window: int = 100
    seed: int = 42
    device: str = "cuda"


def linear_schedule(step: int, warmup: int, total: int) -> float:
    """``transformers.get_linear_schedule_with_warmup`` multiplier."""
    if step < warmup:
        return float(step) / float(max(1, warmup))
    return max(0.0, float(total - step) / float(max(1, total - warmup)))


def sample_batches(n_samples: int, sample_size: int, effective_batch_size: int,
                   seed: int) -> List[np.ndarray]:
    """Sample-index batches for one epoch: one fixed shuffle (seed), then
    consecutive slices of ``effective_batch_size // sample_size``; the
    trailing partial batch is dropped as the reference's step count does."""
    bs = max(1, effective_batch_size // sample_size)
    order = np.random.default_rng(seed).permutation(n_samples)
    return [order[i:i + bs] for i in range(0, n_samples - bs + 1, bs)]


@dataclass
class RunStats:
    """End-of-training metrics of one SAE (reference wandb names)."""

    fvu: float = float("nan")
    l0: float = float("nan")
    auxk: float = float("nan")
    dead_pct: float = float("nan")
    fire_pct: float = float("nan")
    high_freq_pct: float = float("nan")
    steps: int = 0
    seconds: float = 0.0
    history: list = field(default_factory=list)


def _to_floats(values: list) -> None:
    """Convert device scalars in ``values`` to floats in place, in one
    transfer."""
    pos = [i for i, v in enumerate(values) if isinstance(v, Tensor)]
    if pos:
        vals = torch.stack([values[i].float() for i in pos]).cpu().tolist()
        for i, v in zip(pos, vals):
            values[i] = v


class SaeTrainer:
    """Trains ``saes`` (name -> :class:`Sae`) on the same batches."""

    def __init__(self, saes: Mapping[str, Sae], cfg: TrainConfig):
        self.cfg = cfg
        self.saes = dict(saes)
        dev = torch.device(cfg.device)
        self.opts = {}
        for name, sae in self.saes.items():
            kw = {"eps": 6.25e-10}
            if dev.type == "cuda":
                kw["fused"] = True
            self.opts[name] = torch.optim.Adam(sae.parameters(), lr=cfg.lr, **kw)
        self.since_fired = {
            n: torch.zeros(s.num_latents, dtype=torch.long, device=dev) for n, s in self.saes.items()
        }
        self.global_step = 0

    def dead_mask(self, name: str) -> Optional[Tensor]:
        if self.cfg.auxk_alpha <= 0:
            return None
        return self.since_fired[name] > self.cfg.dead_feature_threshold

    def fit(self, acts: Tensor, *, log: Optional[Callable[[str], None]] = None,
            max_steps: Optional[int] = None) -> Dict[str, RunStats]:
        """Train on ``acts`` (``[N, sample_size, d_in]``, any dtype, any
        device; batches are moved to ``cfg.device`` as fp32). Gradients are
        enabled here whatever the caller's grad mode."""
        with torch.enable_grad():
            return self._fit(acts, log=log, max_steps=max_steps)

    def _fit(self, acts: Tensor, *, log: Optional[Callable[[str], None]],
             max_steps: Optional[int]) -> Dict[str, RunStats]:
        torch.set_float32_matmul_precision("high")
        dev = torch.device(self.cfg.device)
        n, sample_size, _ = acts.shape
        batches = sample_batches(n, sample_size, self.cfg.effective_batch_size, self.cfg.seed)
        if not batches:
            raise ValueError(
                f"{n} samples of {sample_size} tokens do not fill one batch of "
                f"{self.cfg.effective_batch_size} activation vectors"
            )
        total = len(batches) * self.cfg.num_epochs
        if max_steps is not None:
            total = min(total, int(max_steps))
        stats = {name: RunStats() for name in self.saes}
        window: Dict[str, List[Tensor]] = {name: [] for name in self.saes}
        fired_frac: Dict[str, List[float]] = {name: [] for name in self.saes}
        run_fvu: Dict[str, List[float]] = {name: [] for name in self.saes}
        run_l0: Dict[str, List[float]] = {name: [] for name in self.saes}
        run_aux: Dict[str, List[float]] = {name: [] for name in self.saes}
        t0 = time.perf_counter()
        step = 0
        done = False
        for _epoch in range(self.cfg.num_epochs):
            for idx in batches:
                if step >= total:
                    done = True
                    break
                x = acts[torch.from_numpy(idx)].to(dev, non_blocking=True).float()
                lr_mult = linear_schedule(step, self.cfg.lr_warmup_steps, total)
                for name, sae in self.saes.items():
                    if step == 0:
                        sae.b_dec.data = geometric_median(x.reshape(-1, x.shape[-1])).to(sae.dtype)
                    if sae.cfg.normalize_decoder:
                        sae.set_decoder_norm_to_unit_norm()
                    out = sae(x, dead_mask=self.dead_mask(name))
                    loss = out.fvu + self.cfg.auxk_alpha * out.auxk_loss + out.multi_topk_fvu / 8
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(sae.parameters(), 1.0)
                    if sae.cfg.normalize_decoder:
                        sae.remove_gradient_parallel_to_decoder_directions()
                    opt = self.opts[name]
                    for g in opt.param_groups:
                        g["lr"] = self.cfg.lr * lr_mult
                    opt.step()
                    opt.zero_grad(set_to_none=True)
                    with torch.no_grad():
                        did = torch.zeros(sae.num_latents, dtype=torch.bool, device=dev)
                        did[out.latent_indices.flatten()] = True
                        self.since_fired[name] += self.cfg.effective_batch_size
                        self.since_fired[name][did] = 0
                        cnt = torch.bincount(out.latent_indices.flatten(),
                                             minlength=sae.num_latents)
                    w = window[name]
                    w.append(cnt)
                    if len(w) > self.cfg.feature_sampling_window:
                        w.pop(0)
                    # Device scalars; converted in bulk (no per-step sync).
                    fired_frac[name].append(did.float().mean())
                    run_fvu[name].append(out.fvu.detach())
                    run_l0[name].append(out.l0_loss.detach())
                    run_aux[name].append(out.auxk_loss.detach())
                step += 1
                self.global_step += 1
                if step % 500 == 0 or step == total:
                    for nm in self.saes:
                        for lst in (run_fvu[nm], run_l0[nm], run_aux[nm], fired_frac[nm]):
                            _to_floats(lst)
                if log is not None and (step % 500 == 0 or step == total):
                    msg = " ".join(
                        f"{nm}: fvu {np.mean(run_fvu[nm][-500:]):.4f}" for nm in self.saes
                    )
                    log(f"step {step}/{total} {msg} ({time.perf_counter() - t0:.0f}s)")
            if done:
                break
        secs = time.perf_counter() - t0
        for nm in self.saes:
            for lst in (run_fvu[nm], run_l0[nm], run_aux[nm], fired_frac[nm]):
                _to_floats(lst)
        last = max(1, min(len(batches), 500))
        for name, sae in self.saes.items():
            dens = torch.stack(window[name]).sum(0).double() / (
                len(window[name]) * self.cfg.effective_batch_size
            )
            stats[name] = RunStats(
                fvu=float(np.mean(run_fvu[name][-last:])),
                l0=float(np.mean(run_l0[name][-last:])),
                auxk=float(np.mean(run_aux[name][-last:])),
                dead_pct=float((self.since_fired[name] > self.cfg.dead_feature_threshold)
                               .float().mean()),
                fire_pct=float(np.mean(fired_frac[name][-last:])),
                high_freq_pct=float((dens > 1e-2).float().mean()),
                steps=step,
                seconds=secs,
                history=[float(np.mean(run_fvu[name][i:i + 100]))
                         for i in range(0, len(run_fvu[name]), 100)],
            )
        return stats


@torch.no_grad()
def fvu_by_bucket(sae: Sae, acts: Tensor, buckets: Sequence, *, device: str = "cuda",
                  batch_samples: Optional[int] = None) -> Dict[object, float]:
    """Absolute FVU per bucket (``sum ||x - x_hat||^2 / sum ||x -
    mean_bucket||^2`` over every token of the bucket's samples), with the
    SAE in its training TopK mode; ``"all"`` pools every sample."""
    buckets = np.asarray(buckets)
    n, seq, d = acts.shape
    bs = batch_samples or max(1, 4096 // seq)
    out: Dict[object, float] = {}
    keys = list(dict.fromkeys(buckets.tolist())) + ["all"]
    for key in keys:
        idx = np.arange(n) if key == "all" else np.nonzero(buckets == key)[0]
        s1 = torch.zeros(d, dtype=torch.float64)
        s2 = torch.zeros((), dtype=torch.float64)
        err = torch.zeros((), dtype=torch.float64)
        count = 0
        for i in range(0, len(idx), bs):
            x = acts[torch.from_numpy(idx[i:i + bs])].to(device).float()
            flat = x.reshape(-1, d)
            o = sae(x)
            err += (o.sae_out.float() - flat).pow(2).sum().double().cpu()
            s1 += flat.sum(0).double().cpu()
            s2 += flat.pow(2).sum().double().cpu()
            count += flat.shape[0]
        var = s2 - (s1.pow(2).sum() / max(1, count))
        out[key] = float(err / var) if var > 0 else float("nan")
    return out


def absolute_bucket_gate(bucket_fvu: Mapping[object, float], bar: float = 0.5) -> bool:
    """True when every bucket's held-out FVU is finite and at most
    ``bar`` (an absolute bar, not a comparison to another set)."""
    return all(math.isfinite(v) and v <= bar for v in bucket_fvu.values())


def choose_config(results: Sequence[Mapping], *, max_dead: float = 0.01,
                  max_fire: float = 0.25, bar: float = 0.5) -> Optional[Mapping]:
    """TADA's choice over the ``(m, k)`` sweep: lowest held-out FVU among
    dictionaries with few dead features (``dead_pct <= max_dead``; the
    paper's chosen SAEs have 0.25-0.37%), few always-on ones
    (``fire_pct <= max_fire``; theirs ~20%) and a passing absolute bucket
    gate. ``results`` items carry ``val_fvu``, ``dead_pct``, ``fire_pct``
    and ``gate``. Returns None when nothing qualifies."""
    ok = [r for r in results
          if r.get("gate") and r["dead_pct"] <= max_dead and r["fire_pct"] <= max_fire
          and math.isfinite(r["val_fvu"]) and r["val_fvu"] <= bar]
    if not ok:
        return None
    return min(ok, key=lambda r: r["val_fvu"])


SWEEP_M = (2, 4, 8, 16)
SWEEP_K = (16, 32, 64)


def sweep_configs(ms: Sequence[int] = SWEEP_M, ks: Sequence[int] = SWEEP_K) -> List[SaeConfig]:
    """The paper's sweep grid as BatchTopK configs."""
    return [SaeConfig(expansion_factor=int(m), k=int(k), batch_topk=True) for m in ms for k in ks]


def config_name(cfg: SaeConfig) -> str:
    return f"m{cfg.expansion_factor}_k{cfg.k}"


__all__ = [
    "RunStats", "SWEEP_K", "SWEEP_M", "SaeTrainer", "TrainConfig", "absolute_bucket_gate",
    "choose_config", "config_name", "fvu_by_bucket", "linear_schedule", "sample_batches",
    "sweep_configs",
]
