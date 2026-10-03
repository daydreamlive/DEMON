"""M1: YuE2Adapter + the real StreamPipeline vs upstream, eager and TRT.

Composes one song (upstream ``examples/song.json``), then measures:

* per-forward parity: the TRT velocity vs upstream ``CachedNAR.velocity``;
* E1: ring output (production adapter, real StreamPipeline, depth 1 and
  4) vs an explicit midpoint solve on the SAME backend (eager: upstream
  ``CachedNAR.solve``; TRT: a graph_solve-style loop over the bound
  engine), plus TRT vs upstream eager;
* E9: windowed VAE decode (TRT window engine and eager window) vs the
  full FP32 decode, at the song start, middle and end.

Writes one JSON. Run with the YuE2 env vars set (see docs/FAMILIES.md):

    python scripts/spikes/yue2_ring_parity.py --out notes/family_merge/yue2_out/ring/m1.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch

from acestep.engine.diffusion import DiffusionConfig
from acestep.engine.stream import SlotRequest, StreamPipeline
from acestep.engine.yue2_adapter import YuE2Adapter
from acestep.engine.yue2_velocity import raw_time, upstream_noise


def errors(actual, expected) -> dict:
    a, b = actual.detach().float().cpu(), expected.detach().float().cpu()
    d = a - b
    return {"max_abs": float(d.abs().max()), "relative_l2": float(d.norm() / b.norm().clamp_min(1e-12)),
            "exact": bool(torch.equal(a, b))}


def explicit_solve(velocity, bundle, noise_btc, steps=32):
    """graph_solve's arithmetic over any velocity backend, B identical rows."""
    x = noise_btc.clone()
    dt = 1 / steps
    for step in range(steps):
        t = 1 - step * dt
        first = velocity(bundle, x, [raw_time(t)] * x.shape[0])
        mid = x - first * (dt / 2)
        second = velocity(bundle, mid, [raw_time(t - dt / 2)] * x.shape[0])
        x = x - second * dt
    return x


def ring_output(velocity, bundle, seed, depth, *, emit_index=0, device="cuda"):
    """The ``emit_index``-th latent the production ring emits."""
    adapter = YuE2Adapter(velocity, steps=32, device=device, dtype=torch.bfloat16)
    config = DiffusionConfig(infer_steps=32, dcw_enabled=False, noise_on_cpu=True)
    pipe = StreamPipeline(None, config, pipeline_depth=depth, adapter=adapter, queue_cap=1)
    emitted = []
    ticks = []
    for _ in range(32 * (emit_index + 2) + 8):
        pipe.submit(SlotRequest(seed=seed, latent_frames=bundle.frames, aux_cond=bundle))
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = pipe.tick()
        torch.cuda.synchronize()
        ticks.append((time.perf_counter() - t0) * 1000)
        if out is not None:
            emitted.append(out.clone())
            if len(emitted) > emit_index:
                return emitted[emit_index], ticks
    raise RuntimeError("ring emitted too few latents")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--song", type=Path, help="request JSON (default: upstream examples/song.json)")
    args = parser.parse_args()

    from acestep.engine.yue2_context import YuE2Context
    from acestep.engine.yue2_runtime import import_paths, trt_dir, weights_root

    ctx = YuE2Context(weights_root(), trt_dir=trt_dir())
    song_path = args.song or Path(import_paths()[0]).parent / "examples" / "song.json"
    request = json.loads(song_path.read_text())
    report = {"song": str(song_path), "load_s": ctx.load_s, "gpu": torch.cuda.get_device_name()}

    comp = ctx.compose(style=request["style"], lyrics=request["lyrics"], seed=int(request["seed"]),
                       max_frames=2500)
    bundle = ctx.bundle(comp)
    report.update(frames=bundle.frames, cond_tokens=bundle.cond_tokens, truncated=bundle.truncated,
                  compose_ms=comp.timings_ms)
    seed = bundle.seed
    noise = upstream_noise(seed, bundle.frames).to("cuda", torch.bfloat16)[None]
    eager, trt = ctx.eager_velocity, (ctx.velocity if ctx.has_trt_nar else None)

    with torch.inference_mode():
        # Per-forward parity vs upstream CachedNAR.velocity.
        fwd = {}
        for s in (1.0, 0.5, 1 / 32):
            ref = bundle.nar.velocity(noise[0], raw_time(s))
            fwd[f"eager_s{s:.4f}"] = errors(eager(bundle, noise, [raw_time(s)])[0], ref)
            if trt is not None:
                fwd[f"trt_s{s:.4f}"] = errors(trt(bundle, noise, [raw_time(s)])[0].clone(), ref)
        report["forward_parity"] = fwd

        # E1 eager: ring vs upstream solve.
        upstream = bundle.nar.solve()  # CPU fp32 of the bf16 state
        e1 = {}
        for depth in (1, 4):
            out, ticks = ring_output(eager, bundle, seed, depth)
            e1[f"eager_ring_d{depth}_vs_upstream_solve"] = errors(out[0], upstream)
            e1[f"eager_ring_d{depth}_tick_p50_ms"] = sorted(ticks)[len(ticks) // 2]
        if trt is not None:
            for depth in (1, 4):
                ref = explicit_solve(trt, bundle, noise.repeat(depth, 1, 1))
                emit_index = 0 if depth == 1 else 4  # every step at full batch 4
                out, ticks = ring_output(trt, bundle, seed, depth, emit_index=emit_index)
                e1[f"trt_ring_d{depth}_vs_trt_explicit_solve"] = errors(out[0], ref[0])
                e1[f"trt_explicit_d{depth}_vs_upstream_solve"] = errors(ref[0], upstream)
                e1[f"trt_ring_d{depth}_tick_p50_ms"] = sorted(ticks)[len(ticks) // 2]
        report["e1"] = e1

        # E9: window decode vs full decode at the edges and the middle.
        anchor = ctx.solve(bundle)
        full = ctx.decode_full(anchor).float()
        T = bundle.frames
        e9 = {}
        spans = {"start": (0, 20), "middle": (T // 2, 20), "end": (T - 20, 20)}
        for kind, decode_ctx in (("trt", ctx.window_vae), ("eager", ctx.vae.decode)):
            if decode_ctx is None:
                continue
            from acestep.engine.yue2_trt import decode_span

            for name, (start, n) in spans.items():
                win = decode_span(decode_ctx, anchor, start, n)
                lo, hi = start * 1920, min((start + n) * 1920, full.shape[-1])
                e9[f"{kind}_{name}"] = errors(win[:, : hi - lo], full[:, lo:hi])
        report["e9"] = e9
        report["peak_vram_gib"] = torch.cuda.max_memory_allocated() / 2**30

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
