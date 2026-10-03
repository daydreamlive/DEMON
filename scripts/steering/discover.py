#!/usr/bin/env python3
"""Discover activation-steering vectors for any model family and write packs.

Prompt-pair difference-of-means vectors on the generic steering slot:
paired positive/negative generations, the mean difference of each
block's post-block residual, and one block chosen by a residual patching
sweep. This is not a replication of TADA (arXiv 2602.11910,
github.com/luk-st/steer-audio); it borrows only the general idea of
choosing a layer by patching.

For one concept:

1. Generate ``--pairs`` paired runs (same seed for the positive and the
   negative prompt; the prompts are concept descriptors appended to a
   rotating set of neutral base prompts) through the family's REAL
   streaming pipeline (``StreamPipeline`` + its ``ModelAdapter``, the
   ring's own schedule and sampler), capturing the post-block residual of
   every block at every step (mean over tokens).
2. Difference of means per block (averaged over steps and pairs).
3. Residual patching sweep: for ``--patch-pairs`` pairs and every block
   ``b``, re-run the negative prompt with block ``b``'s contribution
   (output minus input) replaced by the positive run's, at every step,
   decode, and measure the concept proxy. The block whose patch moves the
   proxy furthest toward the positive run (normalised by the pooled
   pos/neg gap, median over pairs) carries the pack.
4. Write ``<out>/<family>/<checkpoint>/<concept>.safetensors``
   (:mod:`acestep.steering.packs`), with provenance.

The core is family-agnostic: it only uses the pipeline's steering layout
and eager block list (``StreamPipeline.steering_layout`` /
``_steering_blocks``). A family adds a small driver (generate one latent
for a prompt+seed through its pipeline, decode it to audio); ``sa3`` and
``acestep`` drivers live below.

Proxies are log-mel/STFT statistics (``scripts/steering/proxies.py``);
CLAP is not a repo dependency, so semantic concepts need a spectral proxy
that tracks them.

Usage (idle GPU, eager forward):

    python scripts/steering/discover.py --family sa3 --checkpoint medium \\
        --concept bright --concept warm --concept rough --concept density

    python scripts/steering/discover.py --family sa3 --checkpoint medium \\
        --concept mything --pos "glassy bells" --neg "muffled thuds" \\
        --proxy centroid --label "Glassy"
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from proxies import measure  # noqa: E402

torch.set_grad_enabled(False)

#: Built-in concepts: the four ACE demo axes (same knob directions as
#: acestep/steering/policy.py AUTO_AXES) plus one extra.
CONCEPTS = {
    "bright": dict(
        label="Bright", proxy="centroid",
        pos="bright, crisp, sparkling highs, airy, shimmering treble",
        neg="dark, muffled, dull, lowpassed, murky, no treble",
        blurb="positive brightens (spectral centroid up)",
    ),
    "warm": dict(
        label="Warm", proxy="lowhigh_db",
        pos="warm, deep round bass, mellow full low end",
        neg="thin, tinny, no bass, harsh brittle highs",
        blurb="positive tilts the spectrum toward bass (warmer)",
    ),
    "rough": dict(
        label="Rough", proxy="flatness",
        pos="gritty, distorted, noisy, saturated, rough lo-fi texture",
        neg="clean, smooth, pure, polished, pristine",
        blurb="positive adds grit and noise (spectral flatness up)",
    ),
    "density": dict(
        label="Density", proxy="onset_rate",
        pos="sparse, minimal, few instruments, lots of space and silence",
        neg="dense, busy, layered, many instruments, wall of sound",
        blurb="positive thins the texture toward sparse/minimal "
              "(same direction as the ACE steer_density axis)",
    ),
    "percussive": dict(
        label="Percussive", proxy="perc_ratio",
        pos="heavy drums, punchy percussion, driving beat",
        neg="no drums, beatless, ambient sustained pads",
        blurb="positive pushes toward drums and percussion",
    ),
}

#: Neutral base prompts the concept descriptors are appended to; pairs
#: rotate through them so the vector is about the concept, not a genre.
BASES = (
    "lofi hip hop beat",
    "cinematic synthwave",
    "jazz piano trio",
    "deep house groove",
    "acoustic folk song",
    "orchestral film score",
    "indie rock band",
    "ambient electronic music",
)


# ---------------------------------------------------------------------------
# Block recorder (family-agnostic: hooks the pipeline's eager blocks)
# ---------------------------------------------------------------------------


class BlockRecorder:
    """Forward hooks on a block list, keyed by the pipeline's row step.

    Modes: ``mean`` accumulates the token-mean of every block output per
    (block, step); ``delta`` caches each block's full contribution
    (output minus input) per (block, step, call); ``patch`` replaces block
    ``patch_block``'s contribution with a cached one.
    """

    def __init__(self, blocks):
        self.blocks = blocks
        self.pipe = None
        self.mode = None
        self.sums = None
        self.count = None
        self.cache = {}
        self.patch_block = -1
        self.patch_src = None
        self._calls = {}
        self.handles = [
            b.register_forward_hook(self._make(i)) for i, b in enumerate(blocks)
        ]

    def close(self):
        for h in self.handles:
            h.remove()

    def start(self, mode, pipe, *, patch_block=-1, patch_src=None):
        self.mode, self.pipe = mode, pipe
        self.patch_block, self.patch_src = patch_block, patch_src
        self._calls = {}
        if mode == "mean":
            self.sums = {}
            self.count = {}
        if mode == "delta":
            self.cache = {}

    def stop(self):
        self.mode, self.pipe = None, None

    def _step(self):
        rows = self.pipe._current_step_per_row if self.pipe is not None else []
        return rows[0] if rows else None

    def _make(self, i):
        def hook(_m, inputs, output):
            if self.mode is None:
                return None
            step = self._step()
            if step is None:
                return None
            hs = output[0] if isinstance(output, tuple) else output
            call = self._calls.get((i, step), 0)
            self._calls[(i, step)] = call + 1
            if self.mode == "mean":
                v = hs.float().mean(dim=(0, 1))
                key = (i, step)
                self.sums[key] = self.sums.get(key, 0) + v
                self.count[key] = self.count.get(key, 0) + 1
                return None
            x_in = inputs[0]
            if self.mode == "delta":
                self.cache[(i, step, call)] = (hs - x_in).detach().clone()
                return None
            if self.mode == "patch" and i == self.patch_block:
                d = self.patch_src.get((i, step, call))
                if d is None:
                    return None
                new = x_in + d.to(hs.dtype)
                return (new,) + tuple(output[1:]) if isinstance(output, tuple) else new
            return None
        return hook

    def means(self, n_blocks):
        """``[n_blocks, H]`` mean over steps of the per-step token means."""
        out = []
        for b in range(n_blocks):
            keys = [k for k in self.sums if k[0] == b]
            out.append(torch.stack([self.sums[k] / self.count[k] for k in keys]).mean(0))
        return torch.stack(out)


# ---------------------------------------------------------------------------
# Family drivers
# ---------------------------------------------------------------------------


class SA3Driver:
    """Stable Audio 3 through the production StreamPipeline + SA3Adapter
    (eager DiT, pingpong sampler with seeded renoise, full denoise from
    noise: the ring's text-to-music path at sa3_denoise 1.0)."""

    family = "sa3"

    def __init__(self, checkpoint: str, *, duration: float, steps: int):
        from acestep.engine.sa3_adapter import SA3Adapter
        from acestep.engine.sa3_context import SA3Context

        self.checkpoint = checkpoint
        self.duration = float(duration)
        self.steps = int(steps)
        self.ctx = SA3Context(checkpoint)
        self.sample_rate = self.ctx.sample_rate
        self.adapter = SA3Adapter(
            self.ctx.dit, schedule_builder=None,
            device=self.ctx.device, dtype=self.ctx.dtype,
        )
        self.codec = self.ctx.make_codec(backend="tensorrt")
        self._conds = {}
        self.sampler = "pingpong"

    def _cond(self, prompt):
        c = self._conds.get(prompt)
        if c is None:
            c = self.ctx.prepare_cond(prompt=prompt, duration=self.duration, steps=self.steps)
            self._conds[prompt] = c
        return c

    def build_pipeline(self, prompt=None):
        from acestep.engine.diffusion import DiffusionConfig
        from acestep.engine.stream import StreamPipeline

        cond = self._cond(prompt or BASES[0])
        self.adapter.schedule_builder = self.ctx.make_schedule_builder(cond, self.steps)
        cfg = DiffusionConfig(
            infer_steps=self.steps, infer_method="sde", noise_on_cpu=True,
            dcw_enabled=False,
        )
        return StreamPipeline(None, cfg, pipeline_depth=1, adapter=self.adapter)

    def generate(self, prompt, seed, *, on_pipeline=None, steering=None):
        from acestep.engine.stream import SlotRequest

        cond = self._cond(prompt)
        pipe = self.build_pipeline(prompt)
        if steering:
            pipe.set_steering(steering)
        if on_pipeline is not None:
            on_pipeline(pipe)
        pipe.submit(SlotRequest(
            seed=int(seed), denoise=1.0, aux_cond=cond.cond_bundle,
            latent_frames=cond.latent_frames, sde_noise_seeded=True,
        ))
        latent = None
        for _ in range(self.steps + 2):
            latent = pipe.tick()
            if latent is not None:
                break
        pipe.close()
        if latent is None:
            raise RuntimeError("pipeline produced no latent")
        return latent

    def decode(self, latent_btc):
        audio = self.codec.decode_full(latent_btc.movedim(1, 2), decode_seed=1528)
        y = audio.float().mean(dim=0).cpu().numpy()
        return y[: int(self.duration * self.sample_rate)]


class ACEDriver:
    """ACE-Step v1.5 through ``Session.stream`` (eager decoder, the
    streaming handle's own pipeline): text-to-music over a silent source
    at full denoise, CFG with the learned null embedding."""

    family = "acestep"

    def __init__(self, checkpoint: str, *, duration: float, steps: int):
        from acestep.engine.session import Session
        from acestep.nodes import Audio

        self.checkpoint = checkpoint
        self.duration = float(duration)
        self.steps = int(steps)
        self.session = Session(checkpoint=checkpoint, decoder_backend="pytorch",
                               vae_backend="pytorch")
        self.sample_rate = 48000
        silence = torch.zeros(1, 2, int(self.duration * 48000))
        self.source = self.session.prepare_source(Audio(waveform=silence, sample_rate=48000))
        self._conds = {}
        self.sampler = "ode"

    def _cond(self, prompt):
        c = self._conds.get(prompt)
        if c is None:
            c = self.session.encode_text(tags=prompt, lyrics="[instrumental]",
                                         duration=self.duration)
            self._conds[prompt] = c
        return c

    def _handle(self, prompt):
        return self.session.stream(
            source=self.source, conditioning=self._cond(prompt),
            steps=self.steps, pipeline_depth=1,
        )

    def build_pipeline(self, prompt=None):
        h = self._handle(prompt or BASES[0])
        h.tick(seed=0, denoise=1.0)  # materialize the pipeline
        return h.pipeline

    def generate(self, prompt, seed, *, on_pipeline=None, steering=None):
        h = self._handle(prompt)
        latent = None
        first = True
        for _ in range(self.steps + 3):
            if first:
                latent = h.tick(seed=int(seed), denoise=1.0)
                first = False
                pipe = h.pipeline
                if steering:
                    pipe.set_steering(steering)
                if on_pipeline is not None:
                    on_pipeline(pipe)
            else:
                latent = h.tick(seed=int(seed), denoise=1.0)
            if latent is not None:
                break
        if latent is None:
            raise RuntimeError("pipeline produced no latent")
        return latent.tensor if hasattr(latent, "tensor") else latent

    def decode(self, latent_btc):
        from acestep.nodes.types import Latent

        audio = self.session.decode(Latent(tensor=latent_btc))
        wav = audio.waveform
        if wav.dim() == 3:
            wav = wav[0]
        return wav.float().mean(dim=0).cpu().numpy()


DRIVERS = {"sa3": SA3Driver, "acestep": ACEDriver}


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


def _pairs(n_pairs, spec, seed0):
    out = []
    for k in range(n_pairs):
        base = BASES[k % len(BASES)]
        seed = seed0 + k
        out.append((f"{base}, {spec['pos']}", f"{base}, {spec['neg']}", seed))
    return out


def discover_concept(driver, name, spec, *, pairs, patch_pairs, seed0, log):
    layout_pipe = driver.build_pipeline()
    layout = layout_pipe.steering_layout()
    blocks = layout_pipe._steering_blocks()
    layout_pipe.close()
    if layout is None or blocks is None:
        raise RuntimeError(f"family {driver.family} declares no eager steering layout")
    nb, hidden = layout.num_blocks, layout.hidden_size
    rec = BlockRecorder(blocks)
    proxy = spec["proxy"]
    t0 = time.time()
    try:
        pair_list = _pairs(pairs, spec, seed0)
        pos_means, neg_means, pos_p, neg_p = [], [], [], []
        for k, (pp, nn, seed) in enumerate(pair_list):
            for prompt, means, prox in ((pp, pos_means, pos_p), (nn, neg_means, neg_p)):
                lat = driver.generate(prompt, seed, on_pipeline=lambda p: rec.start("mean", p))
                rec.stop()
                means.append(rec.means(nb).cpu())
                prox.append(measure(proxy, driver.decode(lat), driver.sample_rate))
            if k % 8 == 7:
                log(f"  [{name}] pairs {k + 1}/{pairs} ({time.time() - t0:.0f}s) "
                    f"proxy pos {np.mean(pos_p):.4g} neg {np.mean(neg_p):.4g}")
        pos_m = torch.stack(pos_means).mean(0)   # [nb, H]
        neg_m = torch.stack(neg_means).mean(0)
        diffs = pos_m - neg_m
        gap = float(np.mean(pos_p) - np.mean(neg_p))
        paired_gap_sign = float(np.mean(np.sign(np.array(pos_p) - np.array(neg_p))))
        log(f"  [{name}] proxy {proxy}: pos {np.mean(pos_p):.4g} neg {np.mean(neg_p):.4g} "
            f"gap {gap:.4g} (paired sign agreement {paired_gap_sign:+.2f})")

        # Activation patching over blocks.
        effects = np.zeros((patch_pairs, nb))
        for k, (pp, nn, seed) in enumerate(pair_list[:patch_pairs]):
            driver.generate(pp, seed, on_pipeline=lambda p: rec.start("delta", p))
            rec.stop()
            src = rec.cache
            rec.cache = {}
            # Normalise by the pooled pos/neg gap (a single pair's gap can
            # be near zero and blow the ratio up).
            p_neg = neg_p[k]
            denom = gap if abs(gap) > 1e-9 else 1.0
            for b in range(nb):
                lat = driver.generate(
                    nn, seed,
                    on_pipeline=lambda p, b=b: rec.start("patch", p, patch_block=b, patch_src=src),
                )
                rec.stop()
                val = measure(proxy, driver.decode(lat), driver.sample_rate)
                effects[k, b] = (val - p_neg) / denom
            del src
            torch.cuda.empty_cache()
            log(f"  [{name}] patched pair {k + 1}/{patch_pairs} ({time.time() - t0:.0f}s) "
                f"best block {int(np.argmax(np.median(effects[: k + 1], axis=0)))}")
        # Median over pairs: robust to one pair whose patch overshoots.
        mean_eff = np.median(effects, axis=0)
        block = int(np.argmax(mean_eff))
    finally:
        rec.close()
    vec = diffs[block]
    norm = float(vec.norm())
    return {
        "block": block, "vector": vec / max(norm, 1e-12), "norm": norm,
        "hidden": hidden, "num_blocks": nb, "effects": mean_eff.tolist(),
        "effects_mean": effects.mean(0).tolist(),
        "effects_std": effects.std(0).tolist(), "gap": gap,
        "pos_proxy_mean": float(np.mean(pos_p)), "neg_proxy_mean": float(np.mean(neg_p)),
        "paired_sign_agreement": paired_gap_sign,
        "diff_norms": [float(d.norm()) for d in diffs],
        "elapsed_s": time.time() - t0,
    }


def calibrate(driver, res, spec, *, seeds, alphas, log):
    """Proxy vs CAA strength (multiples of the raw mean difference) on
    neutral base prompts, through the pipeline's steering slot."""
    proxy = spec["proxy"]
    out = {}
    for a in alphas:
        vals = []
        for k, seed in enumerate(seeds):
            prompt = BASES[k % len(BASES)]
            cfg = [] if a == 0 else [{
                "layer": res["block"], "step": -1,
                "weights": tuple(1.0 for _ in range(driver.steps)),
                "vector": res["vector"], "magnitude": res["norm"], "alpha": float(a),
            }]
            lat = driver.generate(prompt, seed, steering=cfg)
            vals.append(measure(proxy, driver.decode(lat), driver.sample_rate))
        out[float(a)] = vals
        log(f"    alpha {a:+.2f} x meandiff: {proxy} mean {np.mean(vals):.4g}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--family", required=True, choices=sorted(DRIVERS))
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--concept", action="append", required=True,
                    help="concept name (built-in: %s); repeatable" % ", ".join(CONCEPTS))
    ap.add_argument("--pos", default=None, help="positive descriptor (custom concept)")
    ap.add_argument("--neg", default=None, help="negative descriptor (custom concept)")
    ap.add_argument("--proxy", default=None, help="proxy for a custom concept")
    ap.add_argument("--label", default=None)
    ap.add_argument("--blurb", default="")
    ap.add_argument("--pairs", type=int, default=32)
    ap.add_argument("--patch-pairs", type=int, default=8)
    ap.add_argument("--duration", type=float, default=54.0,
                    help="render seconds (default 54: the streaming session window)")
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--knob-unit", type=float, default=0.1,
                    help="mean-difference multiples per knob unit (pack magnitude = "
                         "norm * knob_unit; default 0.1: knob 10 = one mean difference)")
    ap.add_argument("--calibrate", default="-2,-1,0,1,2",
                    help="CAA strengths (x mean difference) to report, '' to skip")
    ap.add_argument("--calib-seeds", type=int, default=4)
    ap.add_argument("--out", default=None, help="pack root (default: the configured "
                    "steering packs dir)")
    ap.add_argument("--report", default=None, help="write a JSON report here")
    args = ap.parse_args()

    from acestep.paths import steering_packs_dir
    from acestep.steering.packs import SteeringPack, save_pack

    out_root = Path(args.out) if args.out else steering_packs_dir()

    def log(msg):
        print(msg, flush=True)

    concepts = []
    for name in args.concept:
        spec = dict(CONCEPTS.get(name, {}))
        if args.pos:
            spec["pos"] = args.pos
        if args.neg:
            spec["neg"] = args.neg
        if args.proxy:
            spec["proxy"] = args.proxy
        if args.label:
            spec["label"] = args.label
        if args.blurb:
            spec["blurb"] = args.blurb
        missing = [k for k in ("pos", "neg", "proxy") if k not in spec]
        if missing:
            ap.error(f"concept {name!r}: missing {missing} (not built-in; pass --pos/--neg/--proxy)")
        concepts.append((name, spec))

    log(f"[load] {args.family} {args.checkpoint} duration={args.duration}s steps={args.steps}")
    driver = DRIVERS[args.family](args.checkpoint, duration=args.duration, steps=args.steps)
    report = {}
    for name, spec in concepts:
        log(f"[concept] {name}: +'{spec['pos']}' / -'{spec['neg']}' proxy={spec['proxy']}")
        res = discover_concept(
            driver, name, spec, pairs=args.pairs, patch_pairs=args.patch_pairs,
            seed0=args.seed, log=log,
        )
        log(f"  [{name}] block {res['block']} (effect {res['effects'][res['block']]:.3f}) "
            f"norm {res['norm']:.3g}")
        calib = {}
        if args.calibrate:
            alphas = [float(a) for a in args.calibrate.split(",") if a.strip()]
            calib = calibrate(
                driver, res, spec,
                seeds=[args.seed + 10_000 + i for i in range(args.calib_seeds)],
                alphas=alphas, log=log,
            )
        pack = SteeringPack(
            family=driver.family, checkpoint=args.checkpoint, block=res["block"],
            hidden_size=res["hidden"], name=name, vector=res["vector"],
            label=spec.get("label", name), blurb=spec.get("blurb", ""),
            norm=res["norm"], magnitude=res["norm"] * args.knob_unit,
            policy={"kind": "range", "start": 0.0, "end": 1.0},
            provenance={
                "tool": "scripts/steering/discover.py",
                "method_source": "prompt-pair difference of means + residual patching block choice",
                "date": _dt.date.today().isoformat(),
                "pos": spec["pos"], "neg": spec["neg"], "bases": list(BASES),
                "pairs": args.pairs, "patch_pairs": args.patch_pairs,
                "seed0": args.seed, "steps": args.steps, "duration_s": args.duration,
                "sampler": driver.sampler, "denoise": 1.0,
                "proxy": spec["proxy"], "proxy_pos_mean": res["pos_proxy_mean"],
                "proxy_neg_mean": res["neg_proxy_mean"],
                "patching_effect_per_block": [round(e, 4) for e in res["effects"]],
                "knob_unit": args.knob_unit,
            },
        )
        path = save_pack(pack, out_root / driver.family / args.checkpoint / name)
        log(f"  [{name}] wrote {path}")
        report[name] = {**{k: v for k, v in res.items() if k != "vector"},
                        "calibration": calib, "path": str(path), "spec": spec}
    if args.report:
        Path(args.report).write_text(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
