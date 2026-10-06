"""SA3 streaming throughput: depth sweep in process, no server.

Builds the production :class:`~acestep.streaming.sa3_backend.SA3Backend`
over the process-cached :class:`~acestep.engine.sa3_context.SA3Context`
(the same assembly ``sa3_session`` does) and drives ``produce`` the way
the runner does, one ``generate`` per tick. Per depth it reports:

* ``tick_ms``     wall time of one produce (DiT step(s) + bookkeeping),
* ``gens_per_s``  fresh results per second of wall time,
* ``conv_ms``     knob-flip convergence: flip ``seed`` and time until the
                  first fresh result whose originating request carries
                  the new seed (``SA3Backend._emerged_request``), i.e.
                  every DiT step of that result ran after the flip.

The JSON shape matches the medium reference runs
(``{model, duration_s, steps, dit_backend, measure_ticks, flips, rows}``)
so rows compare directly.

Run (repo root):
    python scripts/sa3/sa3_throughput_probe.py --model small-sfx \
        --duration 4 --backend tensorrt --depths 1 2 4 8 --out run.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import torch  # noqa: E402

PROMPT = "Steady heavy rain on a tin roof"


def _stats(xs):
    xs = list(xs)
    if not xs:
        return {"n": 0}
    return {
        "mean": round(statistics.fmean(xs), 2),
        "min": round(min(xs), 2),
        "p50": round(statistics.median(xs), 2),
        "max": round(max(xs), 2),
        "n": len(xs),
    }


def run_depth(context, cond, source, args, depth: int) -> dict:
    from acestep.streaming.generator_backend import TickContext
    from acestep.streaming.knobs import KnobState
    from acestep.streaming.sa3_backend import SA3Backend, sa3_knob_specs

    backend = SA3Backend.from_context(
        context,
        prompt=args.prompt,
        duration_s=args.duration,
        knob_state=KnobState(sa3_knob_specs()),
        cond=cond,
        source_latent_bct=source,
        steps=args.steps,
        depth=depth,
        dit_backend=args.backend,
        codec_backend=args.backend,
    )
    ctx = TickContext(playhead_s=0.0, buffer_duration_s=args.duration)
    knobs = {"sa3_denoise": 1.0, "seed": 1000}

    def tick():
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fresh = backend.produce(knobs, ctx, "generate")
        torch.cuda.synchronize()
        return fresh, (time.perf_counter() - t0) * 1000.0

    for _ in range(args.warmup_ticks):
        tick()

    tick_ms, gens = [], 0
    wall0 = time.perf_counter()
    for _ in range(args.measure_ticks):
        fresh, ms = tick()
        tick_ms.append(ms)
        gens += int(fresh)
    wall = time.perf_counter() - wall0

    conv_ticks, conv_ms = [], []
    for k in range(args.flips):
        new_seed = 2000 + k
        knobs["seed"] = new_seed
        n, t0 = 0, time.perf_counter()
        while True:
            fresh, _ = tick()
            n += 1
            req = getattr(backend, "_emerged_request", None)
            if fresh and getattr(req, "seed", None) == new_seed:
                break
            if n > 50 * (depth + args.steps):
                raise RuntimeError("seed flip never emerged")
        conv_ticks.append(n)
        conv_ms.append((time.perf_counter() - t0) * 1000.0)
        for _ in range(args.steps * 2):  # settle before the next flip
            tick()

    dit_name = type(backend.adapter.dit).__name__
    backend.close()
    return {
        "depth": depth,
        "dit": dit_name,
        "gens_per_s": round(gens / wall, 3),
        "ticks_per_s": round(args.measure_ticks / wall, 2),
        "tick_ms": _stats(tick_ms),
        "gens_per_tick": round(gens / args.measure_ticks, 3),
        "conv_ticks": _stats(conv_ticks),
        "conv_ms": _stats(conv_ms),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", default="small-sfx")
    ap.add_argument("--duration", type=float, default=54.0)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--backend", choices=("eager", "tensorrt"), default="tensorrt")
    ap.add_argument("--depths", type=int, nargs="+", default=[1, 2, 4, 8])
    ap.add_argument("--warmup-ticks", type=int, default=40)
    ap.add_argument("--measure-ticks", type=int, default=150)
    ap.add_argument("--flips", type=int, default=4)
    ap.add_argument("--prompt", default=PROMPT)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from acestep.streaming.sa3_session import get_sa3_context
    from acestep.streaming.source import SAMPLE_RATE

    context = get_sa3_context(args.model)
    cond = context.prepare_cond(prompt=args.prompt, duration=args.duration, steps=args.steps)
    silence = torch.zeros(2, int(args.duration * SAMPLE_RATE))
    source = context.encode_source((SAMPLE_RATE, silence), cond.audio_sample_size)
    print(f"[probe] model={args.model} duration={args.duration}s "
          f"latent_frames={cond.latent_frames} backend={args.backend}", flush=True)

    rows = []
    for depth in args.depths:
        row = run_depth(context, cond, source, args, depth)
        rows.append(row)
        print(json.dumps(row), flush=True)

    result = {
        "model": args.model,
        "duration_s": args.duration,
        "latent_frames": int(cond.latent_frames),
        "steps": args.steps,
        "dit_backend": args.backend,
        "measure_ticks": args.measure_ticks,
        "flips": args.flips,
        "gpu": torch.cuda.get_device_name(0),
        "rows": rows,
    }
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(result, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
