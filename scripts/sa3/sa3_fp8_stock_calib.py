"""Activation-scale calibration for the stock SA3 medium DiT (fp8 producer proof).

Loads the stock medium DiT through DEMON's ``SA3Context`` (the production
load: fp16 weights, song-length label, loop-wrap outro pad), conditions 14
own-domain Music prompts plus one render prompt at the demo window, and
runs every one through an 8-step pingpong rollout at cfg 1.0 inside
``acestep.engine.trt.sa3_fp8.capture_activation_scales``. The result is the
per-Linear max absmax dict that ``graft_fp8`` consumes, written as plain
JSON (``--out``), with provenance beside it (``<out>.meta.json``).

The prompts follow upstream's small-DiT graft (vendor
``make_dit_fp8_smalldit.py``): ``_extract_examples(SYSTEM_PROMPTS["Music"])[:14]``
plus its House render prompt. Max calibration only, margin applied later by
the graft (percentile clipping collapsed parity in June 2026).

Window: the demo runs a 10 s loop with the 3 s loop wrap, i.e. L=140
latent frames (``--duration 10`` under the default song-length label).
``--expect-latents`` asserts it so a changed env can't silently calibrate
a different window. The capture module is ``sam.model.model`` (the
``DiTWrapper`` the engines replace); its qualified names are the JSON keys,
so the ONNX that gets grafted must be exported from the same module root.

Run (GPU; the conductor only):
    python scripts/sa3/sa3_fp8_stock_calib.py \
        --out D:/hot/DEMON/musetimbre/fp8_2026-10-12/calib/stock_act_scales.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable, Sequence

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

STEPS = 8
N_PROMPTS = 14
DOMAIN = "Music"
# Upstream make_dit_fp8_smalldit.py's Music render prompt.
RENDER_PROMPT = (
    "Genre: House, Subgenre: Deep House, BPM: 122 BPM, Tempo: Medium, "
    "VocalType: Instrumental, TrackType: Music, Grade: Neutral"
)
PROMPT_SEED_BASE = 1000
RENDER_SEED = 6000
DEMO_DURATION_S = 10.0
DEMO_LATENTS = 140
MODULE_ROOT = "sam.model.model"
DEFAULT_OUT = Path(r"D:\hot\DEMON\musetimbre\fp8_2026-10-12\calib\stock_act_scales.json")


def own_domain_prompts(n: int = N_PROMPTS, domain: str = DOMAIN) -> list[str]:
    """Upstream's few-shot examples for ``domain`` (vendor reprompt module)."""
    from stable_audio_3.interface.reprompt import SYSTEM_PROMPTS, _extract_examples

    prompts = list(_extract_examples(SYSTEM_PROMPTS[domain]))[:n]
    if len(prompts) < n:
        raise RuntimeError(f"only {len(prompts)} {domain} examples upstream, need {n}")
    return prompts


def pingpong_rollout(dit: Callable, bundle: dict, sched: Sequence[float], *,
                     latent_shape: tuple, seed: int, device, dtype):
    """One rectified-flow pingpong rollout (denoise to x0, renoise to the
    next sigma), cfg 1.0: the sampler the demo runs. Returns the final latent."""
    import torch

    gen = torch.Generator(device=device).manual_seed(int(seed))
    x = torch.randn(latent_shape, device=device, generator=gen)
    with torch.no_grad():
        for i in range(len(sched) - 1):
            t_cur, t_nxt = float(sched[i]), float(sched[i + 1])
            t_b = torch.tensor([t_cur], device=device, dtype=dtype)
            v = dit(x.to(dtype), t_b, **bundle).float()
            denoised = x - t_cur * v
            noise = torch.randn(x.shape, device=device, generator=gen)
            x = (1 - t_nxt) * denoised + t_nxt * noise
    return x


def make_calibration_run(dit: Callable, jobs: Sequence[dict], *, device, dtype,
                         log: Callable[[str], None] = print) -> Callable[[], None]:
    """The ``run`` callable handed to ``capture_activation_scales``.

    ``jobs`` entries: ``{"name", "bundle", "sched", "latent_shape", "seed"}``."""

    def run() -> None:
        for i, job in enumerate(jobs):
            t0 = time.perf_counter()
            pingpong_rollout(dit, job["bundle"], job["sched"],
                             latent_shape=job["latent_shape"], seed=job["seed"],
                             device=device, dtype=dtype)
            log(f"[calib] {i + 1}/{len(jobs)} {job['name']} seed={job['seed']} "
                f"{time.perf_counter() - t0:.1f}s")

    return run


def write_scales(scales: dict, out: Path, meta: dict) -> tuple[Path, Path]:
    """Write the plain ``{module_name: absmax}`` dict ``graft_fp8`` reads and
    a provenance sidecar ``<out stem>.meta.json`` carrying its sha256."""
    if not scales:
        raise ValueError("no activation scales captured")
    bad = {k: v for k, v in scales.items()
           if not isinstance(v, (int, float)) or v != v or v in (float("inf"), float("-inf"))}
    if bad:
        raise ValueError(f"non-finite activation scales: {sorted(bad)[:5]}")
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps({k: float(scales[k]) for k in sorted(scales)}, indent=1)
    out.write_text(payload, encoding="utf-8")
    sha = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    meta = dict(meta)
    meta.update({"scales_file": out.name, "scales_sha256": sha,
                 "n_modules": len(scales),
                 "absmax_max": max(float(v) for v in scales.values()),
                 "absmax_min": min(float(v) for v in scales.values())})
    meta_path = out.with_name(out.stem + ".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return out, meta_path


def _git_head() -> str:
    try:
        return subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"],
                                       text=True).strip()
    except Exception:
        return "unknown"


def _load_capture():
    try:
        from acestep.engine.trt.sa3_fp8 import capture_activation_scales
    except ImportError as exc:
        raise SystemExit(
            "acestep.engine.trt.sa3_fp8.capture_activation_scales is not available "
            f"yet (lane D1, d1_capture.ready): {exc}"
        )
    return capture_activation_scales


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="medium")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--duration", type=float, default=DEMO_DURATION_S,
                    help="loop seconds per calibration render (demo: 10)")
    ap.add_argument("--render-duration", type=float, default=None,
                    help="seconds for the render-prompt pass (default: --duration, "
                         "the full demo window)")
    ap.add_argument("--expect-latents", type=int, default=DEMO_LATENTS,
                    help="assert the --duration window is this many latent frames "
                         "(0 disables)")
    ap.add_argument("--n-prompts", type=int, default=N_PROMPTS)
    ap.add_argument("--min-features", type=int, default=128)
    args = ap.parse_args(argv)

    capture = _load_capture()
    import torch

    from acestep.engine.sa3_context import SA3Context

    print(f"[load] SA3 {args.model} via SA3Context", flush=True)
    context = SA3Context(args.model)
    dit = context.dit
    dit.eval()
    prompts = own_domain_prompts(args.n_prompts)
    render_s = args.duration if args.render_duration is None else args.render_duration
    specs = [(f"music_{i:02d}", p, args.duration, PROMPT_SEED_BASE + i)
             for i, p in enumerate(prompts)]
    specs.append(("render", RENDER_PROMPT, render_s, RENDER_SEED))

    jobs = []
    for name, prompt, dur, seed in specs:
        cond = context.prepare_cond(prompt=prompt, duration=dur, steps=STEPS)
        L = int(cond.latent_frames)
        if args.expect_latents and dur == args.duration and L != args.expect_latents:
            raise SystemExit(f"window for {dur}s is L={L}, expected {args.expect_latents} "
                             "(check DEMON_SA3_SONG_SECONDS / DEMON_SA3_OUTRO_PAD_S)")
        sched = context.make_schedule_builder(cond, STEPS)(1.0).tolist()
        jobs.append({"name": name, "prompt": prompt, "bundle": cond.cond_bundle,
                     "sched": sched, "latent_shape": (1, context.latent_channels, L),
                     "seed": seed, "latent_frames": L,
                     "seconds_label": context.cond_seconds_total(dur)})
    print(f"[calib] {len(jobs)} rollouts x {STEPS} steps, L={[j['latent_frames'] for j in jobs]}",
          flush=True)

    run = make_calibration_run(dit, jobs, device=context.device, dtype=context.dtype)
    t0 = time.perf_counter()
    scales = capture(dit, run, min_features=args.min_features)
    elapsed = time.perf_counter() - t0
    meta = {
        "model": args.model, "checkpoint_dir": str(context.checkpoint_dir),
        "module_root": MODULE_ROOT, "min_features": args.min_features,
        "calibration": "max absmax (no percentile); margin applied by graft_fp8",
        "sampler": "pingpong", "steps": STEPS, "cfg_scale": 1.0,
        "domain": DOMAIN, "renders": [
            {k: j[k] for k in ("name", "prompt", "seed", "latent_frames", "seconds_label")}
            for j in jobs],
        "outro_pad_s": context.outro_pad_s, "song_seconds": context.song_seconds,
        "dtype": str(context.dtype), "torch": torch.__version__,
        "demon_commit": _git_head(), "elapsed_s": round(elapsed, 1),
    }
    out, meta_path = write_scales(scales, args.out, meta)
    print(f"[calib] {len(scales)} Linear scales -> {out} (meta {meta_path.name})", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
