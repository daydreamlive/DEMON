"""SA3 ring-buffer throughput + knob-flip convergence probe (in-process).

Builds the production :class:`~acestep.streaming.sa3_backend.SA3Backend`
over a process-loaded :class:`~acestep.engine.sa3_context.SA3Context`
(the same ``from_context`` assembly the server's create path runs) and
drives its generate step directly, without the server, the runner
thread or any decode, so the number is the DiT ring buffer alone:

* ``tick_ms``: wall time of one ``_generate`` call (submit + one
  ``StreamPipeline.tick``), CUDA-synchronized, over ``--measure-ticks``
  ticks after a warm-up of ``2 * steps`` ticks.
* ``gens_per_s``: finished generations (ticks that returned a latent)
  per second over the measured window.
* ``conv_ticks`` / ``conv_ms``: after flipping ``sa3_denoise``, ticks
  (and wall ms) until the first finished generation whose request was
  submitted with the new value. That is the ring-buffer floor of a
  knob change: queued requests drain first (the queue holds ``depth``
  requests), then a fresh slot needs ``steps`` steps.

With ``--decode eager|tensorrt`` every finished generation is also
decoded the way the server renders it (``SA3Backend._rendered_audio``:
SAME full decode + 48 kHz resample + host copy) inside the timed tick,
with the codec built for that backend; ``dec_ms`` reports that part.
``--decode none`` (default) keeps the DiT-only numbers.

Source: a 48 kHz server fixture as the audio-to-audio anchor (the flips
move ``sa3_denoise`` between two values below 1.0 so the anchor is in
play). Run one process per backend; each process sweeps the depths:

    .venv/Scripts/python.exe scripts/sa3/sa3_throughput_probe.py \
        --model small-music --backend tensorrt --depths 1 2 4 8 \
        --out E:/Projects/sa3-variants/small/throughput/small-trt.json
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

import numpy as np  # noqa: E402
import torch  # noqa: E402

PROMPT = "lofi hip hop, mellow, instrumental"
FIXTURE = "low_fi_Gm_loop_60s_gnm.wav"
FLIP_VALUES = (0.9, 0.6)


def _stats(values) -> dict:
    if not values:
        return {}
    arr = sorted(float(v) for v in values)
    return {
        "mean": round(statistics.fmean(arr), 2),
        "min": round(arr[0], 2),
        "p50": round(float(np.percentile(arr, 50)), 2),
        "max": round(arr[-1], 2),
        "n": len(arr),
    }


def _load_fixture() -> tuple[int, torch.Tensor]:
    import soundfile as sf

    from acestep.fixtures import audio_fixture

    data, sr = sf.read(str(audio_fixture(FIXTURE)), dtype="float32", always_2d=True)
    return int(sr), torch.from_numpy(data.T.copy())


def _make_backend(context, *, depth, steps, duration_s, source, backend,
                  codec_backend=None):
    from acestep.streaming.knobs import KnobState
    from acestep.streaming.sa3_backend import SA3Backend, sa3_knob_specs

    return SA3Backend.from_context(
        context,
        prompt=PROMPT,
        duration_s=duration_s,
        knob_state=KnobState(sa3_knob_specs()),
        source_audio=source,
        dit_backend=backend,
        codec_backend=codec_backend or backend,
        steps=steps,
        depth=depth,
    )


def _prep(denoise: float, steps: int) -> dict:
    return {
        "denoise": float(denoise), "seed": 1528, "steps": int(steps),
        "shift": 1.0, "x0_target": 0.0, "feedback": 0.0, "feedback_depth": 1,
    }


_DEC_MS: list = []


def _tick(be, prep, decode: bool = False) -> tuple[float, bool, float | None]:
    t0 = time.perf_counter()
    lat = be._generate(prep)
    torch.cuda.synchronize()
    if decode and lat is not None:
        t1 = time.perf_counter()
        be._rendered_audio(lat)
        _DEC_MS.append((time.perf_counter() - t1) * 1000.0)
    dt = (time.perf_counter() - t0) * 1000.0
    req = getattr(be.pipeline, "last_finished_request", None) if lat is not None else None
    return dt, lat is not None, (float(req.denoise) if req is not None else None)


def probe_depth(context, *, depth, steps, duration_s, source, backend,
                measure_ticks, flips, decode: str = "none") -> dict:
    dec = decode != "none"
    be = _make_backend(context, depth=depth, steps=steps, duration_s=duration_s,
                       source=source, backend=backend,
                       codec_backend=decode if dec else None)
    codec_kind = type(be.codec).__name__
    if getattr(be.codec, "uses_trt", False):
        codec_kind += ":" + be.codec._trt.engine_path.parent.name
    dit_kind = type(be.adapter.dit).__name__
    engine = getattr(getattr(be.adapter.dit, "engine_path", None), "parent", None)
    if engine is not None:
        dit_kind = f"{dit_kind}:{engine.name}"
    try:
        cur = FLIP_VALUES[0]
        for _ in range(2 * steps + depth):  # warm-up: fill the ring
            _tick(be, _prep(cur, steps), dec)
        tick_ms, gens = [], 0
        _DEC_MS.clear()
        t_start = time.perf_counter()
        for _ in range(measure_ticks):
            dt, done, _ = _tick(be, _prep(cur, steps), dec)
            tick_ms.append(dt)
            gens += int(done)
        wall = time.perf_counter() - t_start
        dec_ms = list(_DEC_MS)

        conv_ticks, conv_ms = [], []
        for i in range(flips):
            new = FLIP_VALUES[(i + 1) % 2]
            n, ms = 0, 0.0
            while True:
                dt, done, den = _tick(be, _prep(new, steps), dec)
                n += 1
                ms += dt
                if done and den is not None and abs(den - new) < 1e-6:
                    break
                if n > 20 * (steps + depth):
                    raise RuntimeError("flip never converged")
            conv_ticks.append(n)
            conv_ms.append(ms)
            cur = new
            # settle so the next flip starts from a steady ring
            for _ in range(steps + depth):
                _tick(be, _prep(cur, steps), dec)
        return {
            "depth": depth,
            "dit": dit_kind,
            "latent_frames": int(be._cond.latent_frames),
            "gens_per_s": round(gens / wall, 3),
            "ticks_per_s": round(measure_ticks / wall, 2),
            "tick_ms": _stats(tick_ms),
            "gens_per_tick": round(gens / measure_ticks, 3),
            "decode": decode,
            "codec": codec_kind if dec else None,
            "dec_ms": _stats(dec_ms),
            "conv_ticks": _stats(conv_ticks),
            "conv_ms": _stats(conv_ms),
        }
    finally:
        be.close()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", default="small-music")
    ap.add_argument("--backend", choices=("tensorrt", "eager"), default="tensorrt")
    ap.add_argument("--depths", type=int, nargs="+", default=[1, 2, 4, 8])
    ap.add_argument("--duration", type=float, default=54.0)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--measure-ticks", type=int, default=150)
    ap.add_argument("--flips", type=int, default=4)
    ap.add_argument("--decode", choices=("none", "eager", "tensorrt"), default="none",
                    help="also decode every finished generation inside the "
                         "timed tick with this codec backend (default: none, "
                         "DiT ring buffer only)")
    ap.add_argument("--no-fp8", action="store_true",
                    help="hide fp8 DiT engines from discovery (medium: "
                         "measure the fp16mixed engine production would "
                         "otherwise skip in favor of fp8)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    if args.no_fp8:
        import re

        import acestep.engine.sa3_trt as sa3_trt

        sa3_trt._DIT_FP8_DIR_RE = re.compile(r"(?!)")  # never matches

    from acestep.streaming.sa3_session import get_sa3_context

    context = get_sa3_context(args.model)
    sr, wav = _load_fixture()
    source = (sr, wav[:, : int(args.duration * sr)])
    out = {
        "model": args.model, "duration_s": args.duration, "steps": args.steps,
        "dit_backend": args.backend, "measure_ticks": args.measure_ticks,
        "flips": args.flips, "gpu": torch.cuda.get_device_name(0),
        "no_fp8": bool(args.no_fp8), "decode": args.decode,
        "flip_values": list(FLIP_VALUES), "rows": [],
    }
    for d in args.depths:
        row = probe_depth(
            context, depth=d, steps=args.steps, duration_s=args.duration,
            source=source, backend=args.backend,
            measure_ticks=args.measure_ticks, flips=args.flips,
            decode=args.decode,
        )
        print(json.dumps(row), flush=True)
        out["rows"].append(row)
    out["torch_max_allocated_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
