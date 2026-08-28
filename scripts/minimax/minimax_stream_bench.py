"""End-to-end streaming proof and cost report for MiniMax-Music3.

Drives the real :class:`~acestep.streaming.minimax_backend.MiniMaxBackend`
with real weights through the loop ``PipelineRunner`` runs each tick --
advance a playhead, sync the source, read knobs, produce, render the
next frontier chunk, crossfade it into a looping buffer -- and writes
the result out as audio.

This backend is append-only: the AR stage writes 25 Hz frames, the DiT
renders them in overlapping chunks, and the frontier advances. So the
numbers that matter are not "tick milliseconds" (there is no ring to
tick) but:

``ar_realtime``
    The autoregressive stage's rate against the 40 ms of audio each
    frame represents. This is the pipeline's bottleneck.
``render_realtime``
    Committed audio per second of chunk-render wall clock.
``pipeline_realtime``
    Both together: audio committed per second of wall clock. **Below
    1.0 the frontier cannot keep ahead of a playhead**, the rolling
    window laps, and the listener hears earlier material repeat. The
    run reports how much of the output was fresh.
``knob_to_frontier_s``
    Measured, not derived. At ``--sweep-at`` the harness moves a knob
    and times how long until audio written under the new setting
    reaches the delivery frontier. For an AR knob that is the wait for
    the current AR frontier to be committed; for a renderer knob it is
    the wait for one more chunk. Add the playback lead for knob-to-EAR.

Two ways to run. With the language model, which is the real thing::

    .venv/Scripts/python.exe scripts/minimax/minimax_stream_bench.py \\
        --prompt "bpm is 140 ... darkwave" --seconds 60 --accel tensorrt \\
        --sweep minimax_temperature --out out/native_stream.wav

Or replaying a saved capture, which exercises the renderer, the chunk
geometry and the whole frontier path without 21 GB of LM resident::

    .venv/Scripts/python.exe scripts/minimax/minimax_stream_bench.py \\
        --capture <path.safetensors> --seconds 40
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# A sibling ACE-Step checkout shadows `acestep` otherwise.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from acestep.engine.minimax_ar import load_replay_stream  # noqa: E402
from acestep.engine.minimax_context import get_minimax_context  # noqa: E402
from acestep.engine.minimax_render import (  # noqa: E402
    AR_FRAME_RATE_HZ,
    CARRY_LATENT_FRAMES,
    CHUNK_AR_FRAMES,
    DEFAULT_GUIDANCE,
    DEFAULT_SHIFT,
    DEFAULT_STEPS,
    HOP_AR_FRAMES,
    MiniMaxChunkRenderer,
)
from acestep.streaming.generator_backend import TickContext  # noqa: E402
from acestep.streaming.knobs import KnobState  # noqa: E402
from acestep.streaming.minimax_backend import (  # noqa: E402
    DELIVERY_SAMPLE_RATE,
    MiniMaxBackend,
    minimax_knob_specs,
)

DEFAULT_PROMPT = (
    "bpm is 140. key is F, and scale is minor. Darkwave / Coldwave. "
    "Cold analog synthesizers, gated reverb on a mechanical drum machine, "
    "a wide stereo pad bed and a narrow, close-mic'd baritone vocal. "
    "The low end is tight and the mid-range is deliberately hollow."
)

# Knobs the sweep can move, and which stage each one reaches. The
# distinction is the whole latency story: an AR knob changes what the
# language model writes NEXT and waits for that to be rendered; a
# renderer knob changes how already-written frames are rendered and
# waits only for the next chunk.
# RenderControls field each renderer knob writes, so the probe can
# confirm a committed chunk actually carried the new value.
_RENDER_FIELD = {
    "minimax_guidance": "guidance",
    "minimax_shift": "shift",
    "minimax_cond_strength": "cond_strength",
}

SWEEPABLE = {
    "minimax_temperature": "ar",
    "minimax_top_k": "ar",
    "minimax_ar_guidance": "ar",
    "minimax_guidance": "render",
    "minimax_shift": "render",
    "minimax_cond_strength": "render",
}


def _crossfade_into(buf: np.ndarray, chunk: np.ndarray, start: int) -> None:
    """Patch ``chunk`` into the looping buffer with 25 ms edges -- the
    same treatment pipeline_runner applies, and for the same reason: the
    region being overwritten is already audible, so a hard splice
    clicks."""
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
    buf[np.arange(start, start + n) % total] = patch


def _build(args):
    ctx = get_minimax_context(
        dtype=torch.bfloat16 if args.dtype == "bfloat16" else torch.float32,
        ar_policy="absent" if args.capture else "resident",
    )
    knobs = KnobState(minimax_knob_specs())

    if args.capture:
        renderer = MiniMaxChunkRenderer(
            ctx.make_dit(
                latent_frames=ctx.chunk_latent_frames, backend=args.accel,
            ),
            ctx.condition_encoder,
            device=ctx.device, dtype=ctx.dtype,
            chunk_ar_frames=CHUNK_AR_FRAMES,
            carry_latent_frames=CARRY_LATENT_FRAMES,
            latent_channels=ctx.latent_channels,
        )
        # rate 0 = serve frames as fast as asked, so the run measures the
        # RENDERER's ceiling rather than the replay's pacing.
        stream = load_replay_stream(args.capture, rate_x_realtime=args.replay_rate)
        print(f"[ar] replay capture frames={stream.max_frames} "
              f"({stream.max_frames / AR_FRAME_RATE_HZ:.1f}s of audio)")
        return MiniMaxBackend(
            ar_stream=stream, renderer=renderer,
            codec=ctx.make_codec(backend=args.accel),
            knob_state=knobs, window_s=args.window, steps=args.steps,
        )

    print("[ar] live autoregressive stage (resident)")
    return MiniMaxBackend.from_context(
        ctx,
        prompt=args.prompt,
        lyrics=args.lyrics,
        knob_state=knobs,
        window_s=args.window,
        steps=args.steps,
        max_ar_frames=args.max_ar_frames,
        dit_backend=args.accel,
        codec_backend=args.accel,
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--capture", default=None,
                    help="replay a saved capture instead of running the LM")
    ap.add_argument("--replay-rate", type=float, default=0.0,
                    help="pace a replayed capture at this multiple of "
                         "realtime; 0 serves frames on demand")
    ap.add_argument("--prompt", default=DEFAULT_PROMPT)
    ap.add_argument("--lyrics", default="[instrumental]")
    ap.add_argument("--seconds", type=float, default=45.0,
                    help="wall-clock seconds to run the stream")
    ap.add_argument("--window", type=float, default=60.0,
                    help="rolling-window length in seconds")
    ap.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    ap.add_argument("--shift", type=float, default=DEFAULT_SHIFT)
    ap.add_argument("--guidance", type=float, default=DEFAULT_GUIDANCE)
    ap.add_argument("--hop", type=int, default=HOP_AR_FRAMES,
                    help="AR frames committed per render (the latency lever)")
    ap.add_argument("--lead", type=float, default=8.0,
                    help="generation lead target; high enough not to bind "
                         "on hardware slower than realtime")
    ap.add_argument("--max-ar-frames", type=int, default=9000)
    ap.add_argument("--sweep", default=None, choices=sorted(SWEEPABLE))
    ap.add_argument("--sweep-at", type=float, default=None,
                    help="wall seconds at which to move --sweep "
                         "(default: a third of the way in)")
    ap.add_argument("--sweep-to", type=float, default=None,
                    help="value to move it to (default: the spec maximum)")
    ap.add_argument("--reprompt", default=None,
                    help="swap the caption at --sweep-at (live AR only)")
    ap.add_argument("--dtype", default="bfloat16",
                    choices=("bfloat16", "float32"))
    ap.add_argument("--accel", default="tensorrt", choices=("eager", "tensorrt"))
    ap.add_argument("--tick-hz", type=float, default=30.0,
                    help="runner tick rate to simulate")
    ap.add_argument("--out", default="out/native_stream.wav")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    print(f"[load] context dtype={args.dtype} accel={args.accel}")
    t0 = time.perf_counter()
    backend = _build(args)
    load_s = time.perf_counter() - t0
    print(f"[load] ready in {load_s:.1f}s")

    knobs = backend.knob_state
    knobs.update({
        "minimax_shift": args.shift,
        "minimax_guidance": args.guidance,
        "minimax_hop": args.hop,
        "minimax_lead": args.lead,
        "minimax_steps": args.steps,
    })

    buf = np.zeros((backend.window_samples, 2), dtype=np.float32)
    written = np.zeros(backend.window_samples, dtype=bool)

    sweep_at = args.sweep_at
    if sweep_at is None:
        sweep_at = args.seconds / 3.0
    sweep_done = False
    sweep_report = None
    pending_probe = None

    tick_dt = 1.0 / args.tick_hz
    start = time.perf_counter()
    first_audio_s = None
    ticks = 0
    emitted_samples = 0
    # Wall clock at the last chunk commit. A replayed capture runs out of
    # AR frames and then generates nothing, so dividing committed audio
    # by the whole run would report the idle tail as slowness.
    last_commit_s = 0.0
    seen_chunks = 0

    while True:
        now = time.perf_counter()
        elapsed = now - start
        if elapsed >= args.seconds:
            break

        # A playhead running in real time from t=0, wrapped at the window.
        playhead = elapsed % backend.window_s
        ctx = TickContext(
            playhead_s=playhead, buffer_duration_s=backend.window_s,
        )
        backend.sync_source(ctx)
        raw = backend.read_knobs()
        backend.rebuild_imminent(raw)
        fresh = backend.produce(raw, ctx, "generate")

        while True:
            chunk = backend.render_window(playhead)
            if chunk is None:
                break
            _crossfade_into(buf, chunk.pcm, chunk.start_sample)
            idx = np.arange(
                chunk.start_sample, chunk.start_sample + chunk.pcm.shape[0],
            ) % backend.window_samples
            written[idx] = True
            emitted_samples += int(chunk.pcm.shape[0])
            if first_audio_s is None:
                first_audio_s = elapsed
                print(f"[first audio] {elapsed:.1f}s after start")

        if fresh:
            backend.on_fresh_generation(raw)
        if backend.chunks != seen_chunks:
            seen_chunks = backend.chunks
            last_commit_s = elapsed
        ticks += 1

        # ---- live-steering probe -------------------------------------
        if not sweep_done and elapsed >= sweep_at and (args.sweep or args.reprompt):
            sweep_done = True
            frontier_before = backend.frontier_s()
            ar_frontier_s = backend.ar.frames_emitted / AR_FRAME_RATE_HZ
            if args.sweep:
                spec = {s.name: s for s in minimax_knob_specs()}[args.sweep]
                target = args.sweep_to
                if target is None:
                    target = spec.max_val
                knobs.update({args.sweep: target})
                print(f"[probe] {args.sweep} -> {target} at {elapsed:.1f}s "
                      f"(stage={SWEEPABLE[args.sweep]})")
            if args.reprompt:
                backend.handle_set_prompt(args.reprompt)
                print(f"[probe] set_prompt at {elapsed:.1f}s")
            pending_probe = {
                "at_s": elapsed,
                "stage": SWEEPABLE.get(args.sweep, "ar" if args.reprompt else None),
                "frontier_before_s": frontier_before,
                # For an AR knob, every frame from here on is written
                # under the new setting, so the wait is for the frontier
                # to reach the AR stage's current position.
                "target_frontier_s": ar_frontier_s,
                # For a renderer knob it is enough for the frontier to
                # advance -- but only under a chunk that was actually
                # rendered with the new value. A chunk already in flight
                # when the knob moved carries the old one, so waiting on
                # the frontier alone reports the in-flight chunk's
                # commit and undercounts the real latency.
                "target_render_value": (
                    target if SWEEPABLE.get(args.sweep) == "render" else None
                ),
            }

        if pending_probe is not None:
            want = pending_probe["target_render_value"]
            if want is not None:
                landed = (
                    backend.frontier_s() > pending_probe["frontier_before_s"]
                    and getattr(backend.last_commit_controls, _RENDER_FIELD[args.sweep])
                    == want
                )
            else:
                landed = backend.frontier_s() >= pending_probe["target_frontier_s"]
            if landed:
                sweep_report = dict(pending_probe)
                sweep_report["knob_to_frontier_s"] = round(
                    elapsed - pending_probe["at_s"], 2,
                )
                print(
                    f"[probe] reached the frontier after "
                    f"{sweep_report['knob_to_frontier_s']:.2f}s "
                    f"(+ playback lead for knob-to-EAR)"
                )
                pending_probe = None

        slack = tick_dt - (time.perf_counter() - now)
        if slack > 0:
            time.sleep(slack)

    wall = time.perf_counter() - start
    committed_s = backend.frontier_s()
    ar_frames = backend.ar_frames
    # Measure the rate over the period generation was actually running.
    gen_wall = last_commit_s if backend.ar_finished else wall
    gen_wall = max(gen_wall, 1e-6)

    fresh_frac = float(written.mean())
    report = {
        "wall_s": round(wall, 2),
        "load_s": round(load_s, 1),
        "accel": args.accel,
        "steps": args.steps,
        "hop_ar_frames": args.hop,
        "hop_audio_s": round(args.hop / AR_FRAME_RATE_HZ, 2),
        "ticks": ticks,
        "first_audio_s": None if first_audio_s is None else round(first_audio_s, 2),
        "ar_frames": ar_frames,
        "ar_ms_per_frame": round(backend.mean_ar_ms_per_frame, 2),
        "ar_realtime": round(
            (1000.0 / AR_FRAME_RATE_HZ)
            / max(backend.mean_ar_ms_per_frame, 1e-9), 3,
        ),
        "chunks": backend.chunks,
        "chunk_render_ms": round(backend.mean_chunk_render_ms, 1),
        "chunk_commit_s": round(args.hop / AR_FRAME_RATE_HZ, 2),
        "render_realtime": round(
            (args.hop / AR_FRAME_RATE_HZ)
            / max(backend.mean_chunk_render_ms / 1000.0, 1e-9), 2,
        ),
        "committed_audio_s": round(committed_s, 2),
        "generating_wall_s": round(gen_wall, 2),
        # The headline: audio committed per second of wall clock spent
        # generating. Measured over `generating_wall_s`, not the whole
        # run, so a replayed capture that runs dry does not report its
        # idle tail as slowness.
        "pipeline_realtime": round(committed_s / gen_wall, 3),
        "window_s": backend.window_s,
        "window_fresh_fraction": round(fresh_frac, 3),
        "emitted_samples": emitted_samples,
        "ar_finished": backend.ar_finished,
        "probe": sweep_report,
    }

    print()
    print(f"ticks              : {ticks} at {args.tick_hz:g} Hz")
    print(f"AR                 : {ar_frames} frames, "
          f"{report['ar_ms_per_frame']:.1f} ms/frame -> "
          f"{report['ar_realtime']:.2f}x realtime")
    print(f"render             : {backend.chunks} chunks, "
          f"{report['chunk_render_ms']:.0f} ms per "
          f"{report['chunk_commit_s']:.1f}s commit -> "
          f"{report['render_realtime']:.1f}x realtime")
    print(f"PIPELINE           : {committed_s:.1f}s of audio in {gen_wall:.1f}s "
          f"of generation -> {report['pipeline_realtime']:.2f}x realtime"
          + ("  [AR exhausted]" if backend.ar_finished else ""))
    if report["pipeline_realtime"] < 1.0:
        print("                     BELOW REALTIME: the playhead laps the "
              "frontier and the window repeats")
    print(f"window written     : {fresh_frac * 100:.0f}% of the "
          f"{backend.window_s:g}s window")
    if sweep_report:
        print(f"knob -> frontier   : {sweep_report['knob_to_frontier_s']:.2f}s "
              f"({sweep_report['stage']} stage)")
    elif pending_probe is not None:
        print("knob -> frontier   : did not reach the frontier before the run "
              "ended (raise --seconds)")

    backend.close()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    import soundfile as sf

    sf.write(str(out), buf, DELIVERY_SAMPLE_RATE)
    print(f"wrote {out}")

    if args.json:
        path = Path(args.json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
