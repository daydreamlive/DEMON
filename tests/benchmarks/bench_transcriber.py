"""Benchmark ACE-Step Transcriber on a fixture clip.

Loads the local ``acestep-transcriber`` checkpoint, runs it on a single
fixture, and prints load/inference timing, peak VRAM, and the raw
structured output for eyeball review.

Usage::

    python tests/benchmarks/bench_transcriber.py
    python tests/benchmarks/bench_transcriber.py --fixture prog_rock_loop_60s_enm.wav
    python tests/benchmarks/bench_transcriber.py --audio path/to/song.wav

Run with ``PYTHONUTF8=1`` if your console is cp1252 — the chat template
contains UTF-8 markers that crash on Windows otherwise.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from acestep.fixtures import audio_fixture
from acestep.transcriber import Transcriber


DEFAULT_FIXTURE = "thrash_metal_loop_120s_enm.wav"


def _resolve_audio(args: argparse.Namespace) -> Path:
    if args.audio:
        return Path(args.audio)
    return audio_fixture(args.fixture)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", default=DEFAULT_FIXTURE,
                        help=f"named fixture (default: {DEFAULT_FIXTURE})")
    parser.add_argument("--audio", default=None,
                        help="explicit audio path (overrides --fixture)")
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--dtype", default="bfloat16",
                        choices=["bfloat16", "float16"])
    parser.add_argument("--attn", default="sdpa",
                        choices=["sdpa", "eager", "flash_attention_2"])
    parser.add_argument("--json-out", default=None,
                        help="optional path to write a JSON summary")
    args = parser.parse_args()

    audio_path = _resolve_audio(args)
    print(f"audio:        {audio_path}")
    print(f"size on disk: {audio_path.stat().st_size / 1e6:.2f} MB")

    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}[args.dtype]

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    free_before, total = torch.cuda.mem_get_info()
    print(f"GPU free before load: {free_before / 1024**3:.2f} / "
          f"{total / 1024**3:.2f} GiB")

    print(f"\nloading transcriber (dtype={args.dtype}, attn={args.attn})...")
    t0 = time.perf_counter()
    transcriber = Transcriber(dtype=dtype, attn_implementation=args.attn)
    load_s = time.perf_counter() - t0
    weight_vram = torch.cuda.max_memory_allocated() / 1024**3
    free_after_load, _ = torch.cuda.mem_get_info()
    print(f"  load time:    {load_s:.2f} s")
    print(f"  weights VRAM: {weight_vram:.2f} GiB (peak alloc)")
    print(f"  free now:     {free_after_load / 1024**3:.2f} GiB")
    print(f"  audio_sr:     {transcriber.audio_sr}")

    print(f"\ntranscribing (max_new_tokens={args.max_new_tokens})...")
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    output = transcriber.transcribe(audio_path,
                                    max_new_tokens=args.max_new_tokens)
    infer_s = time.perf_counter() - t0
    peak_vram = torch.cuda.max_memory_allocated() / 1024**3

    audio_len_s = float(_audio_seconds(audio_path))
    rtf = infer_s / audio_len_s if audio_len_s > 0 else float("nan")

    print(f"  inference:    {infer_s:.2f} s")
    print(f"  audio len:    {audio_len_s:.2f} s")
    print(f"  RTF:          {rtf:.3f} x  (lower = faster)")
    print(f"  peak VRAM:    {peak_vram:.2f} GiB")
    print(f"  out chars:    {len(output)}")

    print("\n" + "=" * 72)
    print("MODEL OUTPUT")
    print("=" * 72)
    print(output)
    print("=" * 72)

    if args.json_out:
        Path(args.json_out).write_text(json.dumps({
            "audio": str(audio_path),
            "audio_seconds": audio_len_s,
            "load_seconds": load_s,
            "inference_seconds": infer_s,
            "rtf": rtf,
            "weights_vram_gib": weight_vram,
            "peak_vram_gib": peak_vram,
            "dtype": args.dtype,
            "attn": args.attn,
            "max_new_tokens": args.max_new_tokens,
            "output_chars": len(output),
            "output": output,
        }, indent=2), encoding="utf-8")
        print(f"\nJSON summary -> {args.json_out}")

    return 0


def _audio_seconds(path: Path) -> float:
    import soundfile as sf
    info = sf.info(str(path))
    return info.frames / info.samplerate


if __name__ == "__main__":
    raise SystemExit(main())
