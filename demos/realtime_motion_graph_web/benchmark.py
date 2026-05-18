"""Headless performance benchmark for the realtime motion graph demo.

This mirrors the server-side inference path used by
``demos.realtime_motion_graph_web`` without HTTP, WebSocket, browser, or
audio-device overhead:

1. Resolve the same accel/checkpoint/TRT engine combination as backend.py.
2. Load the same fixture/config defaults as the web demo.
3. Build Session -> prepare_source -> encode_text -> stream.
4. Time stream.tick() and optional VAE decode per completed generation.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch

torch.set_grad_enabled(False)
torch._dynamo.config.disable = True

from acestep.audio.key_detection import detect_key
from acestep.constants import TASK_INSTRUCTIONS
from acestep.engine.session import Session
from acestep.fixtures import audio_fixture
from acestep.nodes.types import Audio
from acestep.paths import (
    EngineNotBuiltError,
    available_dreamvae_decode_engine,
    available_trt_engines,
    checkpoints_dir,
    dreamvae_decode_engine_name,
    max_profile_duration_s,
    smallest_fitting_profile_duration_s,
)

from .protocol import SAMPLE_RATE


VALID_ACCEL = ("tensorrt", "compile", "eager")
DEFAULT_CONFIG = Path(__file__).parent / "static" / "config.json"
DEFAULT_FIXTURE = "inside_confusion_loop_60s_gsm.wav"


def cuda_sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


@contextmanager
def timed(label: str, timings: dict[str, float]):
    cuda_sync()
    t0 = time.perf_counter()
    try:
        yield
    finally:
        cuda_sync()
        ms = (time.perf_counter() - t0) * 1000
        timings[label] = ms
        print(f"  [{label}] {ms:.1f}ms")


def percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * pct
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] * (hi - pos) + ordered[hi] * (pos - lo)


def describe(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {
            "count": 0,
            "mean_ms": None,
            "median_ms": None,
            "p90_ms": None,
            "p95_ms": None,
            "min_ms": None,
            "max_ms": None,
        }
    return {
        "count": len(values),
        "mean_ms": statistics.fmean(values),
        "median_ms": percentile(values, 0.50),
        "p90_ms": percentile(values, 0.90),
        "p95_ms": percentile(values, 0.95),
        "min_ms": min(values),
        "max_ms": max(values),
    }


def fmt_stat(value: float | int | None) -> str:
    return "n/a" if value is None else f"{float(value):.1f}"


def load_demo_config(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def duration_cap_for(
    *,
    decoder_backend: str,
    vae_backend: str,
    checkpoint: str,
) -> tuple[float, str]:
    use_trt = decoder_backend == "tensorrt" or vae_backend == "tensorrt"
    trt_profile_checkpoint = (
        checkpoint if decoder_backend == "tensorrt" else "acestep-v15-turbo"
    )
    if use_trt:
        return (
            max_profile_duration_s(checkpoint=trt_profile_checkpoint),
            trt_profile_checkpoint,
        )
    return max_profile_duration_s(), trt_profile_checkpoint


def load_audio(path: Path, *, duration_s: float) -> Audio:
    data, sr = sf.read(str(path), dtype="float32")
    if data.ndim == 1:
        waveform = torch.from_numpy(data.reshape(1, -1))
    else:
        waveform = torch.from_numpy(data.T)

    if sr != SAMPLE_RATE:
        import torchaudio

        waveform = torchaudio.transforms.Resample(sr, SAMPLE_RATE)(waveform)

    waveform = waveform[:2, : int(duration_s * SAMPLE_RATE)]
    pool = 1920 * 5
    rem = waveform.shape[-1] % pool
    if rem:
        waveform = waveform[:, : waveform.shape[-1] - rem]
    if waveform.shape[-1] <= 0:
        raise SystemExit(
            "Audio is empty after trimming to the 5-frame ACE-Step boundary."
        )
    return Audio(waveform=waveform, sample_rate=SAMPLE_RATE)


def resolve_trt_engines(
    *,
    decoder_backend: str,
    vae_backend: str,
    checkpoint: str,
    duration_s: float,
    fast_vae: bool,
) -> tuple[dict[str, str] | None, float | None, bool]:
    use_trt = decoder_backend == "tensorrt" or vae_backend == "tensorrt"
    if not use_trt:
        if fast_vae:
            print("[Setup] WARNING: --fast-vae requires --vae-accel tensorrt; ignoring")
        return None, None, False

    trt_profile_checkpoint = (
        checkpoint if decoder_backend == "tensorrt" else "acestep-v15-turbo"
    )
    needs: list[str] = []
    if decoder_backend == "tensorrt":
        needs.append("decoder")
    if vae_backend == "tensorrt":
        needs.extend(["vae_encode", "vae_decode"])

    try:
        trt_engines, picked_dur = available_trt_engines(
            duration_s=duration_s,
            needs=tuple(needs),
            checkpoint=trt_profile_checkpoint,
        )
    except EngineNotBuiltError as exc:
        print(f"[Setup] {exc}")
        if exc.build_command:
            print(f"[Setup] Build command: {exc.build_command}")
        raise SystemExit(2) from exc

    ideal_dur = smallest_fitting_profile_duration_s(
        duration_s,
        checkpoint=trt_profile_checkpoint,
    )
    if picked_dur > ideal_dur:
        print(
            f"[Setup] WARNING: using {picked_dur:.0f}s TRT profile for "
            f"{duration_s:.1f}s audio (ideal {ideal_dur:.0f}s profile not built)"
        )

    if decoder_backend != "tensorrt":
        trt_engines.pop("decoder", None)
    if vae_backend != "tensorrt":
        trt_engines.pop("vae_encode", None)
        trt_engines.pop("vae_decode", None)

    if fast_vae and vae_backend == "tensorrt":
        dreamvae = available_dreamvae_decode_engine(picked_dur)
        if dreamvae is not None:
            trt_engines["vae_decode"] = str(dreamvae)
        else:
            wanted = dreamvae_decode_engine_name(int(picked_dur))
            fallback = Path(trt_engines["vae_decode"]).stem
            print(
                f"[Setup] WARNING: {wanted} engine missing, using {fallback}"
            )
            fast_vae = False
    elif fast_vae:
        print("[Setup] WARNING: --fast-vae requires --vae-accel tensorrt; ignoring")
        fast_vae = False

    return trt_engines, picked_dur, fast_vae


def denoise_at(args: argparse.Namespace, index: int, total: int) -> float:
    if args.denoise_pattern == "constant":
        return float(args.denoise)
    if total <= 1:
        return float(args.denoise)
    phase = 2.0 * math.pi * (index / total)
    value = float(args.denoise) + float(args.denoise_amplitude) * math.sin(phase)
    return max(0.0, min(1.0, value))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark the realtime motion graph ACE-Step inference path."
    )
    parser.add_argument("--accel", choices=VALID_ACCEL, default="tensorrt")
    parser.add_argument("--decoder-accel", choices=VALID_ACCEL)
    parser.add_argument("--vae-accel", choices=VALID_ACCEL)
    parser.add_argument("--checkpoint", default="acestep-v15-turbo")

    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Demo config.json for default prompt/engine/control values.",
    )
    parser.add_argument(
        "--no-config",
        action="store_true",
        help="Ignore static/config.json and use script fallbacks.",
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--audio", type=Path, help="Local WAV/FLAC/etc source audio.")
    source.add_argument(
        "--fixture",
        default=DEFAULT_FIXTURE,
        help="Fixture name from daydreamlive/demon-fixtures.",
    )
    parser.add_argument(
        "--duration",
        type=float,
        help="Seconds of source audio to use. Default: source length capped like the web server.",
    )

    parser.add_argument("--prompt", help="Cover prompt. Defaults to config prompts.a.")
    parser.add_argument("--bpm", type=int, help="Override detected BPM.")
    parser.add_argument("--key", help="Override detected key.")
    parser.add_argument(
        "--no-detect-metadata",
        action="store_true",
        help="Skip librosa/key detection and use --bpm/--key or config defaults.",
    )

    parser.add_argument("--steps", type=int, help="Diffusion steps.")
    parser.add_argument("--depth", type=int, help="Streaming pipeline depth.")
    parser.add_argument("--vae-window", type=float, help="Windowed VAE decode size.")
    parser.add_argument(
        "--fast-vae",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Use DreamVAE TRT decode when available.",
    )

    parser.add_argument("--warmup", type=int, default=4)
    parser.add_argument("--iters", type=int, default=24)
    parser.add_argument("--seed", type=int, default=557)
    parser.add_argument(
        "--vary-seed",
        action="store_true",
        help="Increment seed each submitted tick.",
    )
    parser.add_argument("--denoise", type=float, help="Center/default denoise value.")
    parser.add_argument(
        "--denoise-pattern",
        choices=("sine", "constant"),
        default="sine",
        help="Sine avoids decode-skip artifacts in repeated identical ticks.",
    )
    parser.add_argument("--denoise-amplitude", type=float, default=0.20)
    parser.add_argument(
        "--shift",
        dest="shift",
        type=float,
        help="Diffusion flow shift, passed verbatim to the solver. Useful range ~[1, 6].",
    )
    parser.add_argument("--noise-share", type=float, default=0.0)

    parser.add_argument(
        "--skip-threshold",
        type=float,
        default=-1.0,
        help=(
            "Decode-skip MSE threshold. Default -1 disables skip so VAE "
            "decode metrics are measured every generation. Use 1e-3 to "
            "mirror PipelineRunner."
        ),
    )
    parser.add_argument("--no-decode", action="store_true")
    parser.add_argument(
        "--progress-every",
        type=int,
        default=5,
        help="Print one progress row per N measured generations; 0 disables.",
    )
    parser.add_argument("--json", type=Path, help="Write full metrics as JSON.")

    parser.add_argument(
        "--dcw-enabled",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable/disable wavelet-domain correction.",
    )
    parser.add_argument("--dcw-mode", default=None)
    parser.add_argument("--dcw-scaler", type=float, default=None)
    parser.add_argument("--dcw-high-scaler", type=float, default=None)
    parser.add_argument("--dcw-wavelet", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_demo_config(None if args.no_config else args.config)

    decoder_backend = args.decoder_accel or args.accel
    vae_backend = args.vae_accel or args.accel
    cap_s, trt_profile_checkpoint = duration_cap_for(
        decoder_backend=decoder_backend,
        vae_backend=vae_backend,
        checkpoint=args.checkpoint,
    )
    requested_duration = args.duration if args.duration is not None else cap_s
    duration_s = min(requested_duration, cap_s)

    engine_cfg = config.get("engine", {})
    controls_cfg = config.get("controls", {})
    prompts_cfg = config.get("prompts", {})

    steps = args.steps if args.steps is not None else int(engine_cfg.get("steps", 8))
    depth = args.depth if args.depth is not None else int(engine_cfg.get("depth", 4))
    vae_window = (
        args.vae_window
        if args.vae_window is not None
        else float(engine_cfg.get("vae_window", 3.0))
    )
    fast_vae = (
        bool(args.fast_vae)
        if args.fast_vae is not None
        else bool(engine_cfg.get("fast_vae", False))
    )
    prompt = args.prompt or prompts_cfg.get("a") or "instrumental music"
    fallback_bpm = args.bpm if args.bpm is not None else 120
    key = args.key or engine_cfg.get("key") or "C major"
    denoise = (
        args.denoise
        if args.denoise is not None
        else float(controls_cfg.get("denoise", 0.7))
    )
    args.denoise = denoise
    shift = (
        args.shift
        if args.shift is not None
        else float(controls_cfg.get("shift", 3.0))
    )
    dcw_enabled = (
        bool(args.dcw_enabled)
        if args.dcw_enabled is not None
        else bool(controls_cfg.get("dcw_enabled", True))
    )
    dcw_mode = args.dcw_mode or str(controls_cfg.get("dcw_mode", "double"))
    dcw_scaler = (
        args.dcw_scaler
        if args.dcw_scaler is not None
        else float(controls_cfg.get("dcw_scaler", 0.05))
    )
    dcw_high_scaler = (
        args.dcw_high_scaler
        if args.dcw_high_scaler is not None
        else float(controls_cfg.get("dcw_high_scaler", 0.02))
    )
    dcw_wavelet = args.dcw_wavelet or str(controls_cfg.get("dcw_wavelet", "haar"))

    if requested_duration > cap_s:
        print(
            f"[Setup] Requested duration {requested_duration:.1f}s exceeds "
            f"{trt_profile_checkpoint} profile cap {cap_s:.0f}s; clipping."
        )

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    timings: dict[str, float] = {}

    print("=" * 72)
    print("Realtime Motion Graph Headless Benchmark")
    print("=" * 72)
    print(
        f"[Setup] checkpoint={args.checkpoint} "
        f"decoder={decoder_backend} vae={vae_backend}"
    )
    print(
        f"[Setup] steps={steps} depth={depth} vae_window={vae_window:.1f}s "
        f"fast_vae={fast_vae}"
    )
    print(
        f"[Setup] denoise={denoise:.3f} pattern={args.denoise_pattern} "
        f"shift={shift:.2f} noise_share={args.noise_share:.2f}"
    )

    with timed("load_audio", timings):
        if args.audio is not None:
            audio_path = args.audio
        else:
            audio_path = audio_fixture(args.fixture)
        audio = load_audio(audio_path, duration_s=duration_s)

    waveform = audio.waveform
    actual_duration_s = waveform.shape[-1] / SAMPLE_RATE
    print(
        f"[Setup] audio={audio_path} "
        f"duration={actual_duration_s:.1f}s channels={waveform.shape[0]}"
    )

    trt_engines, picked_dur, fast_vae = resolve_trt_engines(
        decoder_backend=decoder_backend,
        vae_backend=vae_backend,
        checkpoint=args.checkpoint,
        duration_s=actual_duration_s,
        fast_vae=fast_vae,
    )
    if trt_engines:
        for key_name, engine_path in sorted(trt_engines.items()):
            print(f"[Setup] {key_name}={Path(engine_path).stem}")
        if picked_dur is not None:
            print(f"[Setup] picked_trt_profile={picked_dur:.0f}s")

    with timed("model_load", timings):
        session = Session(
            project_root=str(checkpoints_dir()),
            config_path=args.checkpoint,
            decoder_backend=decoder_backend,
            vae_backend=vae_backend,
            trt_engines=trt_engines,
            vae_window=vae_window,
        )

    if args.no_detect_metadata:
        bpm = fallback_bpm
        print(f"[Setup] metadata detection skipped; bpm={bpm} key={key}")
    else:
        with timed("detect_metadata", timings):
            import librosa

            mono_np = waveform.mean(dim=0).numpy()
            detected_bpm, _ = librosa.beat.beat_track(y=mono_np, sr=SAMPLE_RATE)
            bpm = int(round(float(np.asarray(detected_bpm).flat[0])))
            key = args.key or detect_key(mono_np, SAMPLE_RATE)
        if args.bpm is not None:
            bpm = args.bpm
        print(f"[Setup] metadata bpm={bpm} key={key}")

    with timed("prepare_source", timings):
        source = session.prepare_source(audio)
    print(
        f"[Setup] latent_frames={source.latent.tensor.shape[1]} "
        f"({source.latent.tensor.shape[1] / 25.0:.1f}s)"
    )

    with timed("text_encode", timings):
        conditioning = session.encode_text(
            tags=prompt,
            instruction=TASK_INSTRUCTIONS["cover"],
            refer_latent=source.latent,
            bpm=bpm,
            duration=actual_duration_s,
            key=key,
        )

    with timed("stream_setup", timings):
        stream = session.stream(
            source=source,
            conditioning=conditioning,
            steps=steps,
            shift=3.0,
            pipeline_depth=depth,
            dcw_enabled=dcw_enabled,
            dcw_mode=dcw_mode,
            dcw_scaler=dcw_scaler,
            dcw_high_scaler=dcw_high_scaler,
            dcw_wavelet=dcw_wavelet,
        )
    print("[Run] Stream handle ready; first tick builds the pipeline")

    target_completed = args.warmup + args.iters
    max_ticks = target_completed + depth + steps + 20
    measured_tick_ms: list[float] = []
    measured_decode_ms: list[float] = []
    measured_total_ms: list[float] = []
    warmup_tick_ms: list[float] = []
    skipped_measured = 0
    decoded_measured = 0
    completed = 0
    measured = 0
    last_latent = None
    last_wav_seen = False

    print(
        f"[Run] warmup={args.warmup} measured={args.iters} "
        f"decode={'off' if args.no_decode else 'on'} "
        f"skip_threshold={args.skip_threshold:g}"
    )
    run_t0 = time.perf_counter()

    for tick_idx in range(max_ticks):
        denoise_value = denoise_at(args, tick_idx, max(1, target_completed))
        seed = args.seed + tick_idx if args.vary_seed else args.seed

        cuda_sync()
        tick_t0 = time.perf_counter()
        result_latent = stream.tick(
            denoise=denoise_value,
            seed=seed,
            shift=shift,
            noise_sharing=args.noise_share,
        )
        cuda_sync()
        tick_ms = (time.perf_counter() - tick_t0) * 1000

        if result_latent is None:
            continue

        completed += 1
        is_measured = completed > args.warmup
        if is_measured:
            measured += 1
            measured_tick_ms.append(tick_ms)
        else:
            warmup_tick_ms.append(tick_ms)

        skipped = False
        if args.skip_threshold >= 0:
            result = result_latent.tensor
            if last_latent is not None and last_wav_seen:
                mse = (result - last_latent).pow(2).mean().item()
                skipped = mse < args.skip_threshold
            last_latent = result.clone()

        dec_ms = 0.0
        if args.no_decode:
            skipped = True
        elif skipped:
            if is_measured:
                skipped_measured += 1
        else:
            cuda_sync()
            dec_t0 = time.perf_counter()
            audio_out = session.decode(result_latent, t_start=0.0, cyclic=True)
            cuda_sync()
            dec_ms = (time.perf_counter() - dec_t0) * 1000
            last_wav_seen = True
            _ = audio_out.start_sample
            if is_measured:
                decoded_measured += 1
                measured_decode_ms.append(dec_ms)

        if is_measured:
            measured_total_ms.append(tick_ms + dec_ms)
            if args.progress_every and (
                measured == 1
                or measured == args.iters
                or measured % args.progress_every == 0
            ):
                dec_label = "skip" if skipped else f"{dec_ms:.1f}ms"
                print(
                    f"  #{measured:3d}/{args.iters} "
                    f"tick={tick_ms:.1f}ms decode={dec_label} "
                    f"denoise={denoise_value:.3f}"
                )

        if measured >= args.iters:
            break

    if measured < args.iters:
        raise SystemExit(
            f"Only completed {measured} measured generations after {max_ticks} ticks."
        )

    run_wall_ms = (time.perf_counter() - run_t0) * 1000
    tick_stats = describe(measured_tick_ms)
    decode_stats = describe(measured_decode_ms)
    total_stats = describe(measured_total_ms)

    mem_stats: dict[str, float | None] = {
        "cuda_peak_allocated_gb": None,
        "cuda_peak_reserved_gb": None,
    }
    if torch.cuda.is_available():
        mem_stats = {
            "cuda_peak_allocated_gb": torch.cuda.max_memory_allocated() / (1024**3),
            "cuda_peak_reserved_gb": torch.cuda.max_memory_reserved() / (1024**3),
        }

    print("\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)
    print(f"Completed: {completed} ({args.warmup} warmup, {measured} measured)")
    print(f"Wall time: {run_wall_ms:.1f}ms")
    print(f"Decode: {decoded_measured} decoded, {skipped_measured} skipped")
    print()
    print(f"{'metric':<14s} {'mean':>9s} {'p50':>9s} {'p90':>9s} {'p95':>9s} {'min':>9s} {'max':>9s}")
    print("-" * 72)
    for label, stats in (
        ("tick", tick_stats),
        ("decode", decode_stats),
        ("tick+decode", total_stats),
    ):
        print(
            f"{label:<14s} "
            f"{fmt_stat(stats['mean_ms']):>9s} "
            f"{fmt_stat(stats['median_ms']):>9s} "
            f"{fmt_stat(stats['p90_ms']):>9s} "
            f"{fmt_stat(stats['p95_ms']):>9s} "
            f"{fmt_stat(stats['min_ms']):>9s} "
            f"{fmt_stat(stats['max_ms']):>9s}"
        )
    if mem_stats["cuda_peak_allocated_gb"] is not None:
        print(
            f"\nCUDA peak: allocated={mem_stats['cuda_peak_allocated_gb']:.2f} GiB "
            f"reserved={mem_stats['cuda_peak_reserved_gb']:.2f} GiB"
        )

    payload = {
        "config": {
            "checkpoint": args.checkpoint,
            "decoder_backend": decoder_backend,
            "vae_backend": vae_backend,
            "trt_profile_checkpoint": trt_profile_checkpoint,
            "picked_trt_profile_s": picked_dur,
            "trt_engines": trt_engines,
            "fast_vae": fast_vae,
            "audio": str(audio_path),
            "duration_s": actual_duration_s,
            "steps": steps,
            "depth": depth,
            "vae_window_s": vae_window,
            "prompt": prompt,
            "bpm": bpm,
            "key": key,
            "denoise": denoise,
            "denoise_pattern": args.denoise_pattern,
            "shift": shift,
            "noise_share": args.noise_share,
            "seed": args.seed,
            "vary_seed": args.vary_seed,
            "skip_threshold": args.skip_threshold,
            "decode": not args.no_decode,
            "dcw_enabled": dcw_enabled,
            "dcw_mode": dcw_mode,
            "dcw_scaler": dcw_scaler,
            "dcw_high_scaler": dcw_high_scaler,
            "dcw_wavelet": dcw_wavelet,
        },
        "setup_timings_ms": timings,
        "warmup_tick_ms": warmup_tick_ms,
        "tick_ms": measured_tick_ms,
        "decode_ms": measured_decode_ms,
        "tick_decode_ms": measured_total_ms,
        "stats": {
            "tick": tick_stats,
            "decode": decode_stats,
            "tick_decode": total_stats,
            "decoded_measured": decoded_measured,
            "skipped_measured": skipped_measured,
            "run_wall_ms": run_wall_ms,
            **mem_stats,
        },
    }

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nWrote JSON metrics: {args.json}")


if __name__ == "__main__":
    main()
