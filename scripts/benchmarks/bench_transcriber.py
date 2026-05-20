"""Benchmark ACE-Step Transcriber on a fixture clip.

Loads the local ``acestep-transcriber`` checkpoint, runs ``--passes``
back-to-back inferences on a single fixture, and prints VRAM + timing
for each step. Used to characterize:

  - cold load + first inference latency
  - warm-cache inference latency (subsequent passes)
  - CPU↔CUDA round-trip cost (with ``--shuttle``), which models the
    detect-while-live path where the demo parks the transcriber on CPU
    between presses

Usage::

    python scripts/benchmarks/bench_transcriber.py
    python scripts/benchmarks/bench_transcriber.py --fixture prog_rock_loop_60s_enm.wav
    python scripts/benchmarks/bench_transcriber.py --audio path/to/song.wav --passes 3 --shuttle

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


DEFAULT_FIXTURE = "thrash_metal_loop_60s_enm.wav"


def _resolve_audio(args: argparse.Namespace) -> Path:
    if args.audio:
        return Path(args.audio)
    return audio_fixture(args.fixture)


def _vram_snapshot(label: str) -> dict:
    """One-line VRAM read-out + a structured dict for the JSON summary.

    ``mem_get_info()`` reports physical CUDA free/total which is the
    same lens the OS sees, so we don't get fooled by torch's caching
    allocator holding pre-freed slabs.
    """
    if not torch.cuda.is_available():
        return {"label": label, "free_gib": None, "total_gib": None}
    free, total = torch.cuda.mem_get_info()
    free_g = free / 1024**3
    total_g = total / 1024**3
    allocated_g = torch.cuda.memory_allocated() / 1024**3
    reserved_g = torch.cuda.memory_reserved() / 1024**3
    print(
        f"  VRAM {label:<24} free={free_g:5.2f} / {total_g:5.2f} GiB  "
        f"(torch alloc={allocated_g:.2f}, reserved={reserved_g:.2f})"
    )
    return {
        "label": label,
        "free_gib": free_g,
        "total_gib": total_g,
        "torch_alloc_gib": allocated_g,
        "torch_reserved_gib": reserved_g,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", default=DEFAULT_FIXTURE,
                        help=f"named fixture (default: {DEFAULT_FIXTURE})")
    parser.add_argument("--audio", default=None,
                        help="explicit audio path (overrides --fixture)")
    parser.add_argument("--max-new-tokens", type=int, default=4096,
                        help="generation ceiling (default 4096)")
    parser.add_argument("--passes", type=int, default=3,
                        help="how many inference passes to run after load "
                             "(default 3 — first is the 'cold' first-token "
                             "pass, subsequent ones are warm)")
    parser.add_argument("--shuttle", action="store_true",
                        help="between each pass, move the model to CPU + "
                             "empty_cache + move back. Models the live demo's "
                             "detect-while-streaming path where ACE-Step + "
                             "Qwen swap places on the GPU per press.")
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
    audio_len_s = float(_audio_seconds(audio_path))
    print(f"audio length: {audio_len_s:.2f} s")

    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}[args.dtype]

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    snapshot_pre_load = _vram_snapshot("pre-load")

    print(f"\nloading transcriber (dtype={args.dtype}, attn={args.attn})...")
    t0 = time.perf_counter()
    transcriber = Transcriber(dtype=dtype, attn_implementation=args.attn)
    load_s = time.perf_counter() - t0
    weight_vram = torch.cuda.max_memory_allocated() / 1024**3
    snapshot_post_load = _vram_snapshot("post-load")
    print(f"  load time:    {load_s:.2f} s")
    print(f"  weights VRAM: {weight_vram:.2f} GiB (peak alloc since reset)")
    print(f"  audio_sr:     {transcriber.audio_sr}")

    passes = []
    for i in range(args.passes):
        print(f"\n--- pass {i + 1}/{args.passes} ---")

        # Optional CPU shuttle round-trip. This is what the live demo
        # path does between presses; modeling it here lets us cost the
        # PCIe transfer without confounding it with the actual inference.
        cpu_to_gpu_s = None
        if args.shuttle and i > 0:
            t0 = time.perf_counter()
            transcriber.to_device("cuda")
            cpu_to_gpu_s = time.perf_counter() - t0
            print(f"  CPU→CUDA:    {cpu_to_gpu_s:.2f} s")
            _vram_snapshot(f"pre-pass-{i + 1}")

        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        output = transcriber.transcribe(
            audio_path, max_new_tokens=args.max_new_tokens,
        )
        infer_s = time.perf_counter() - t0
        peak_vram = torch.cuda.max_memory_allocated() / 1024**3
        rtf = infer_s / audio_len_s if audio_len_s > 0 else float("nan")

        print(f"  inference:   {infer_s:.2f} s")
        print(f"  RTF:         {rtf:.3f}  (lower = faster than realtime)")
        print(f"  peak VRAM:   {peak_vram:.2f} GiB")
        print(f"  out chars:   {len(output)}")

        gpu_to_cpu_s = None
        if args.shuttle:
            t0 = time.perf_counter()
            transcriber.to_device("cpu")
            torch.cuda.empty_cache()
            gpu_to_cpu_s = time.perf_counter() - t0
            print(f"  CUDA→CPU:    {gpu_to_cpu_s:.2f} s (+ empty_cache)")
            _vram_snapshot(f"post-pass-{i + 1}")

        passes.append({
            "pass": i + 1,
            "inference_s": infer_s,
            "rtf": rtf,
            "peak_vram_gib": peak_vram,
            "output_chars": len(output),
            "cpu_to_gpu_s": cpu_to_gpu_s,
            "gpu_to_cpu_s": gpu_to_cpu_s,
            # Keep the first pass's full output for review; later passes
            # just count chars (deterministic decode, do_sample=False).
            "output": output if i == 0 else None,
        })

    print("\n" + "=" * 72)
    print(f"PASS 1 OUTPUT (do_sample=False so all passes share this text)")
    print("=" * 72)
    print(passes[0]["output"])
    print("=" * 72)

    if args.json_out:
        Path(args.json_out).write_text(json.dumps({
            "audio": str(audio_path),
            "audio_seconds": audio_len_s,
            "load_seconds": load_s,
            "weights_vram_gib": weight_vram,
            "dtype": args.dtype,
            "attn": args.attn,
            "max_new_tokens": args.max_new_tokens,
            "shuttle": args.shuttle,
            "snapshots": {
                "pre_load": snapshot_pre_load,
                "post_load": snapshot_post_load,
            },
            "passes": passes,
        }, indent=2), encoding="utf-8")
        print(f"\nJSON summary -> {args.json_out}")

    return 0


def _audio_seconds(path: Path) -> float:
    import soundfile as sf
    info = sf.info(str(path))
    return info.frames / info.samplerate


if __name__ == "__main__":
    raise SystemExit(main())
