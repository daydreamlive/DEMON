#!/usr/bin/env python3
"""Sanity-check SA3 steering packs on the live streaming backend (TensorRT).

Builds the production ``SA3Backend`` (``from_context``, TRT DiT + TRT
SAME-L) exactly as a session does, attaches the configured packs through
the family-agnostic factory helper, and drives ``produce()`` ticks:

* knob 0 with packs attached vs the same engine with NO packs attached:
  the emitted latents must be bit-identical (the zero path is a no-op);
* each pack at knob -K/+K (default 1 and 10): the concept proxy of the
  decoded audio must move to the pack's positive side for +K and the
  other way for -K (per prompt, against knob 0);
* steady-state tick time: steering engine at knob 0 and with a knob
  active, and the default engine selection without packs (fp8 when built).

Run (idle GPU, no server):
    python scripts/steering/sanity_sa3_stream.py [--knobs 1,10] [--json out.json]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from proxies import measure  # noqa: E402

torch.set_grad_enabled(False)

PROMPTS = (
    "cinematic synthwave, analog arpeggios",
    "lofi hip hop beat",
    "acoustic folk song",
)
DURATION = 54.0
STEPS = 8
DEPTH = 4


def _backend(context, prompt, *, packs_dir):
    """A fresh production backend; ``packs_dir`` decides what it attaches
    (and therefore which DiT engine it selects)."""
    from acestep.paths import STEERING_PACKS_ENV
    from acestep.streaming.families import _attach_steering_packs
    from acestep.streaming.knobs import KnobState
    from acestep.streaming.sa3_backend import SA3Backend, sa3_knob_specs

    os.environ[STEERING_PACKS_ENV] = str(packs_dir)
    b = SA3Backend.from_context(
        context, prompt=prompt, duration_s=DURATION,
        knob_state=KnobState(sa3_knob_specs()),
        dit_backend="tensorrt", codec_backend="tensorrt",
        steps=STEPS, depth=DEPTH, vae_window_s=1.0,
    )
    _attach_steering_packs(
        b, family="sa3", checkpoint=context.model_id,
        layout=b.pipeline.steering_layout(),
    )
    return b


def _drop(b):
    """Release a backend's TRT execution context before the next one."""
    import gc

    b.close()
    b.adapter = None
    b.pipeline = None
    gc.collect()
    torch.cuda.empty_cache()


def _run(b, knobs_over, *, n_fresh=2, max_ticks=40):
    from acestep.streaming.generator_backend import TickContext

    base = {**{s.name: s.default for s in b.knob_specs()}, "steps_override": STEPS}
    knobs = {**base, **knobs_over}
    ctx = TickContext(playhead_s=0.0, buffer_duration_s=DURATION)
    fresh, ticks_ms = [], []
    for _ in range(max_ticks):
        if b.produce(knobs, ctx, "generate"):
            fresh.append(b._last_result_latent.clone())
            ticks_ms.append(b.last_tick_ms)
            if len(fresh) >= n_fresh:
                break
        else:
            ticks_ms.append(b.last_tick_ms)
    return fresh, ticks_ms


def _proxy(context, latent_btc, proxy):
    codec = context.make_codec(backend="tensorrt")
    audio = codec.decode_full(latent_btc.movedim(1, 2), decode_seed=1528)
    y = audio.float().mean(dim=0).cpu().numpy()[: int(DURATION * context.sample_rate)]
    return measure(proxy, y, context.sample_rate)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--knobs", default="1,10", help="knob magnitudes to test")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    mags = [float(k) for k in args.knobs.split(",") if k.strip()]

    from acestep.engine.sa3_context import SA3Context
    from acestep.paths import steering_packs_dir
    from acestep.steering.packs import discover_packs

    packs_dir = steering_packs_dir()
    packs = discover_packs(packs_dir, family="sa3", checkpoint="medium")
    if not packs:
        raise SystemExit(f"no sa3/medium packs under {packs_dir}")
    empty_dir = Path(tempfile.mkdtemp(prefix="nopacks_"))
    context = SA3Context("medium")
    result = {"packs_dir": str(packs_dir), "packs": {}, "timing": {}}

    # 1. Zero is a no-op: packs attached at 0 vs no packs, same engine.
    b0 = _backend(context, PROMPTS[0], packs_dir=packs_dir)
    engine = b0.adapter.dit.engine_path.parent.name
    print(f"[engine] with packs: {engine}; knobs {[p.knob_name for p in b0.steering_packs.packs]}")
    lat_a, _ = _run(b0, {})
    b_np = _backend(context, PROMPTS[0], packs_dir=packs_dir)
    from acestep.steering.packs import PackSteering

    b_np.attach_steering_packs(PackSteering())
    lat_b, _ = _run(b_np, {})
    same = all(torch.equal(x, y) for x, y in zip(lat_a, lat_b)) and len(lat_a) == len(lat_b) > 0
    print(f"[zero] knob 0 vs no packs (same engine {engine}): bit-identical={same}")
    result["zero_bit_identical"] = bool(same)
    result["engine"] = engine

    # 2. Direction per pack.
    for p in packs:
        proxy = p.provenance.get("proxy")
        sign = np.sign(p.provenance.get("proxy_pos_mean", 0) - p.provenance.get("proxy_neg_mean", 0))
        rows = []
        for prompt in PROMPTS:
            vals = {}
            for k in [0.0] + [m for m in mags] + [-m for m in mags]:
                b = _backend(context, prompt, packs_dir=packs_dir)
                lat, _ = _run(b, {p.knob_name: k}, n_fresh=1)
                vals[k] = _proxy(context, lat[0], proxy)
                _drop(b)
            rows.append({"prompt": prompt, "proxy": vals})
            print(f"  [{p.name}] {prompt!r}: " + ", ".join(
                f"{k:+g}:{v:.4g}" for k, v in sorted(vals.items())))
        ok = {}
        for m in mags:
            up = [np.sign(r["proxy"][m] - r["proxy"][0.0]) == sign for r in rows]
            dn = [np.sign(r["proxy"][-m] - r["proxy"][0.0]) == -sign for r in rows]
            ok[m] = {"plus_right_way": int(sum(up)), "minus_right_way": int(sum(dn)),
                     "of": len(rows)}
        print(f"  [{p.name}] direction ({proxy}, positive side {'up' if sign > 0 else 'down'}): {ok}")
        result["packs"][p.name] = {"block": p.block, "proxy": proxy, "rows": rows, "direction": ok}

    # 3. Tick time (steady state).
    def steady(b, over):
        _, ticks = _run(b, over, n_fresh=6, max_ticks=60)
        tail = ticks[-12:]
        return float(np.mean(tail)), float(np.median(tail))

    _drop(b0)
    _drop(b_np)
    b = _backend(context, PROMPTS[0], packs_dir=packs_dir)
    result["timing"]["steer_engine_knob0_ms"] = steady(b, {})
    _drop(b)
    b = _backend(context, PROMPTS[0], packs_dir=packs_dir)
    result["timing"]["steer_engine_active_ms"] = steady(b, {packs[0].knob_name: 5.0})
    _drop(b)
    b = _backend(context, PROMPTS[0], packs_dir=empty_dir)
    result["timing"]["default_engine"] = b.adapter.dit.engine_path.parent.name
    result["timing"]["default_engine_ms"] = steady(b, {})
    print(f"[timing] tick ms (mean, median): {json.dumps(result['timing'])}")
    if args.json:
        Path(args.json).write_text(json.dumps(result, indent=2, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
