"""End-to-end streaming proof for the MiniMax-Music3 backend.

Drives the real :class:`~acestep.streaming.minimax_backend.MiniMaxBackend`
through the real :class:`~acestep.engine.stream.StreamPipeline` with real
weights, reproducing what ``PipelineRunner`` does each tick — advance a
playhead, produce one step over the whole ring, render a window just
ahead of the listener, crossfade it into a looping buffer — and writes
the result out as audio.

This is the artifact that distinguishes "the seam type-checks" from "the
model streams". It reports the two numbers that decide whether the
family is viable in real time: milliseconds per tick, and the ratio of
wall-clock spent to audio committed.

Run with a saved conditioning capture (no autoregressive stage needed):

    .venv/Scripts/python.exe scripts/minimax/minimax_stream_smoke.py \
        --capture <path.safetensors> --seconds 20 --out out/minimax_stream.wav

Or unconditioned, which needs no capture at all and still exercises
every moving part -- zeros is the model's own unconditional CFG branch:

    .venv/Scripts/python.exe scripts/minimax/minimax_stream_smoke.py \
        --uncond --seconds 20 --out out/minimax_stream.wav
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# A sibling ACE-Step checkout shadows `acestep` otherwise.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from acestep.engine.minimax_context import get_minimax_context  # noqa: E402
from acestep.streaming.knobs import KnobState  # noqa: E402
from acestep.streaming.generator_backend import TickContext  # noqa: E402
from acestep.streaming.minimax_backend import (  # noqa: E402
    DELIVERY_SAMPLE_RATE,
    MiniMaxBackend,
    minimax_knob_specs,
    minimax_latent_frames,
)
from acestep.streaming.minimax_session import (  # noqa: E402
    MINIMAX_DURATION_S,
    MINIMAX_VAE_WINDOW_S,
)


def _crossfade_into(buf: np.ndarray, chunk: np.ndarray, start: int) -> None:
    """Patch ``chunk`` into the looping buffer with 25 ms edges.

    The same treatment pipeline_runner applies, and for the same reason:
    the region being overwritten is already audible, so a hard splice
    clicks.
    """
    n = chunk.shape[0]
    total = buf.shape[0]
    if n <= 0:
        return
    xf = min(1200, n // 4)
    patch = chunk.copy()
    if xf > 0:
        ramp = np.linspace(0.0, 1.0, xf, dtype=np.float32)[:, None]
        head = np.arange(start, start + xf) % total
        tail = np.arange(start + n - xf, start + n) % total
        patch[:xf] = buf[head] * (1.0 - ramp) + patch[:xf] * ramp
        patch[n - xf:] = buf[tail] * ramp[::-1] + patch[n - xf:] * (1.0 - ramp[::-1])
    idx = np.arange(start, start + n) % total
    buf[idx] = patch


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--capture", default=None, help="saved conditioning capture")
    ap.add_argument("--uncond", action="store_true",
                    help="stream the model's unconditional branch (zeros)")
    ap.add_argument("--seconds", type=float, default=20.0,
                    help="wall-clock seconds of streaming to simulate")
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--denoise", type=float, default=0.6,
                    help="cover strength once the anchor exists")
    ap.add_argument("--dtype", default="bfloat16", choices=("bfloat16", "float32"))
    ap.add_argument("--sweep", default=None,
                    choices=("minimax_denoise", "minimax_cond_strength",
                             "minimax_shift", "feedback"),
                    help="ramp this knob across the run to prove live steering")
    ap.add_argument("--sweep-from", type=float, default=0.15)
    ap.add_argument("--sweep-to", type=float, default=0.95)
    ap.add_argument("--accel", default="eager",
                    choices=("eager", "tensorrt"),
                    help="renderer backend; tensorrt needs a built engine")
    ap.add_argument("--out", default="out/minimax_stream.wav")
    args = ap.parse_args()

    if not args.capture and not args.uncond:
        ap.error("pass --capture <path> or --uncond")

    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32
    duration_s = MINIMAX_DURATION_S
    frames = minimax_latent_frames(duration_s)

    print(f"[load] context dtype={args.dtype}")
    t0 = time.perf_counter()
    ctx = get_minimax_context(dtype=dtype, ar_policy="absent")
    print(f"[load] done in {time.perf_counter() - t0:.1f}s")

    if args.uncond:
        cond = {
            "encoder_hidden_states": torch.zeros(
                1, frames, 2048, device=ctx.device, dtype=ctx.dtype,
            )
        }
        print(f"[cond] unconditional zeros, {frames} frames")
    else:
        cond = ctx.load_capture(args.capture)
        print(f"[cond] capture {args.capture}, "
              f"{cond['encoder_hidden_states'].shape[1]} frames")

    backend = MiniMaxBackend.from_context(
        ctx,
        cond=cond,
        knob_state=KnobState(minimax_knob_specs()),
        duration_s=duration_s,
        steps=args.steps,
        depth=args.depth,
        vae_window_s=MINIMAX_VAE_WINDOW_S,
        dit_backend=args.accel,
    )
    backend.knob_state.update({"minimax_denoise": args.denoise})

    total = int(round(duration_s * DELIVERY_SAMPLE_RATE))
    buf = np.zeros((total, 2), dtype=np.float32)

    # The runner's shape: a playhead advancing in real time, and a
    # decode target placed a lead ahead of it.
    lead_s = 0.35
    playhead = 0.0
    tick_ms: list = []
    render_ms: list = []
    fresh = 0
    written = 0
    sweep_trace: list = []

    print(f"[stream] simulating {args.seconds}s at depth={args.depth} "
          f"steps={args.steps} denoise={args.denoise} accel={args.accel}")
    wall0 = time.perf_counter()
    while playhead < args.seconds:
        if args.sweep:
            frac = min(1.0, playhead / max(args.seconds, 1e-9))
            val = args.sweep_from + frac * (args.sweep_to - args.sweep_from)
            backend.knob_state.update({args.sweep: val})
            sweep_trace.append((playhead, val))
        knobs = backend.read_knobs()
        ctx_t = TickContext(playhead_s=playhead % duration_s,
                            buffer_duration_s=duration_s)

        t = time.perf_counter()
        is_fresh = backend.produce(knobs, ctx_t, "generate")
        tick = (time.perf_counter() - t) * 1000.0
        tick_ms.append(tick)
        fresh += int(is_fresh)

        target = (playhead + lead_s) % duration_s
        t = time.perf_counter()
        chunk = backend.render_window(target)
        render_ms.append((time.perf_counter() - t) * 1000.0)

        if chunk is not None:
            _crossfade_into(buf, chunk.pcm, chunk.start_sample)
            written += chunk.pcm.shape[0]

        # A tick commits its own wall-clock worth of playhead. This is
        # the honest accounting: if a tick takes longer than the audio
        # it covers, the stream cannot keep up with a listener.
        playhead += tick / 1000.0 + render_ms[-1] / 1000.0
    wall = time.perf_counter() - wall0

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    try:
        import soundfile as sf
        sf.write(args.out, buf, DELIVERY_SAMPLE_RATE)
    except ImportError:
        import torchaudio
        torchaudio.save(
            args.out, torch.from_numpy(buf.T), DELIVERY_SAMPLE_RATE,
        )

    peak = float(np.abs(buf).max())
    rms = float(np.sqrt((buf.astype(np.float64) ** 2).mean()))
    med_tick = float(np.median(tick_ms))
    med_render = float(np.median(render_ms))

    print()
    print(f"  ticks              {len(tick_ms)}  ({fresh} fresh generations)")
    print(f"  tick   median      {med_tick:.1f} ms   "
          f"(p90 {np.percentile(tick_ms, 90):.1f})")
    print(f"  render median      {med_render:.1f} ms")
    print(f"  audio committed    {written / DELIVERY_SAMPLE_RATE:.1f}s "
          f"over {wall:.1f}s wall")
    print(f"  peak / rms         {peak:.4f} / {rms:.5f} "
          f"({20 * np.log10(max(rms, 1e-12)):.1f} dBFS)")
    if sweep_trace:
        lo, hi = sweep_trace[0], sweep_trace[-1]
        print(f"  swept {args.sweep:22s} {lo[1]:.2f} -> {hi[1]:.2f} "
              f"over {len(sweep_trace)} ticks")
    print(f"  wrote              {args.out}")

    if peak < 1e-4:
        print("\n  FAIL: output is silence")
        return 1
    # A full generation lands every steps/depth ticks; that cadence is
    # what has to outrun the playhead, not a single tick.
    gen_s = med_tick * (args.steps / args.depth) / 1000.0
    print(f"\n  full generation every ~{gen_s:.2f}s of compute for "
          f"{duration_s:.1f}s of audio "
          f"= {duration_s / max(gen_s, 1e-9):.1f}x realtime headroom")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
