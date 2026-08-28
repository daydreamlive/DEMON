"""What the MiniMax-Music3 autoregressive stage actually costs.

This is the measurement the whole integration turns on, and it was never
taken directly before: the renderer's throughput was benchmarked in
detail while the stage that FEEDS it was assumed.

MiniMax-Music3 is autoregressive. Its 8.58B Global LM emits one 25 Hz
acoustic frame at a time over a KV cache -- one LM forward at batch 2
(the classifier-free twin) plus seven depth-decoder forwards per frame.
Everything downstream is bounded by how fast that loop runs, so the
question "can this model stream in real time" is first and mostly this
number.

What the script reports:

``ms_per_frame`` / ``realtime``
    Steady-state emission cost, and it against the 40 ms of audio a
    frame represents. Below 1.0 means the stage cannot keep ahead of a
    playhead on its own.
``blocks``
    The same cost bucketed by position in the piece. Attention over a
    growing KV cache could in principle make late frames dearer than
    early ones; this says whether it does.
``prefill_s``
    The prompt's one-time cost, which is also what a live re-prompt
    costs per :meth:`MiniMaxARStream.reprompt`.
``reprompt_s``
    Measured directly with ``--reprompt``: swap the caption mid-stream
    and rebuild the cache against the audio already written. This is the
    number that decides whether ``set_prompt`` is a live control on this
    family or a session restart.

Run::

    .venv/Scripts/python.exe scripts/minimax/minimax_ar_bench.py \
        --frames 400 --reprompt --json out/ar_bench.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

# A sibling ACE-Step checkout shadows `acestep` otherwise.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch  # noqa: E402

from acestep.engine.minimax_ar import (  # noqa: E402
    AR_FRAME_RATE_HZ,
    ARControls,
    MiniMaxAR,
)
from acestep.engine.minimax_helpers import resolve_model_dir  # noqa: E402

DEFAULT_PROMPT = (
    "bpm is 92. key is E, and scale is minor. Electric Blues / Blues Rock. "
    "The production favors a live, organic feel with a moderately wide "
    "soundstage. The frequency response is warm and mid-range heavy, "
    "emphasizing the grit of the guitar tubes and the body of the vocals, "
    "while the low end remains tight but not overpowering."
)
DEFAULT_PROMPT_B = (
    "bpm is 140. key is F, and scale is minor. Darkwave / Coldwave. "
    "Cold analog synthesizers, gated reverb on a mechanical drum machine, "
    "a wide stereo pad bed and a narrow, close-mic'd baritone vocal."
)
DEFAULT_LYRICS = "[instrumental]"

# LM 17.2 GB + depth decoder 1.3 GB + the lm_head logits and KV cache.
STACK_VRAM_GB = 21.0


def _wait_for_vram(index: int, need_gb: float, timeout_s: float) -> float:
    deadline = time.monotonic() + timeout_s
    while True:
        free = torch.cuda.mem_get_info(index)[0] / 1024**3
        if free >= need_gb:
            return free
        if time.monotonic() >= deadline:
            raise SystemExit(
                f"cuda:{index} has {free:.1f} GB free, need {need_gb:.1f} GB. "
                "Another job is holding the card."
            )
        print(f"  waiting for VRAM: {free:.1f} GB free", flush=True)
        time.sleep(15.0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=400,
                    help="AR frames to emit (25 per second of audio)")
    ap.add_argument("--block", type=int, default=50,
                    help="bucket size for the cost-vs-position report")
    ap.add_argument("--prompt", default=DEFAULT_PROMPT)
    ap.add_argument("--prompt-b", default=DEFAULT_PROMPT_B,
                    help="caption to swap to for the --reprompt measurement")
    ap.add_argument("--lyrics", default=DEFAULT_LYRICS)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--reprompt", action="store_true",
                    help="measure a live caption swap at the halfway point")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--ar-guidance", type=float, default=1.5)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--json", default=None, help="write the report here")
    args = ap.parse_args()

    device = torch.device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit("CUDA requested but unavailable")
        free = _wait_for_vram(device.index or 0, STACK_VRAM_GB, 600.0)
        print(f"free VRAM      : {free:.1f} GB (need ~{STACK_VRAM_GB:.0f} GB)")
        torch.cuda.reset_peak_memory_stats(device.index or 0)

    root = resolve_model_dir()
    print(f"checkpoint     : {root}")

    started = time.perf_counter()
    ar = MiniMaxAR.from_pretrained(
        root, dtype=torch.bfloat16, device="cpu", seed=args.seed,
    )
    load_s = time.perf_counter() - started
    started = time.perf_counter()
    ar.to(device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    page_s = time.perf_counter() - started
    print(f"load / page    : {load_s:.1f}s / {page_s:.1f}s")

    controls = ARControls(
        temperature=args.temperature,
        top_k=args.top_k,
        guidance=args.ar_guidance,
    )
    stream = ar.stream(
        prompt=args.prompt, lyrics=args.lyrics, seed=args.seed,
        max_frames=args.frames, controls=controls,
    )
    print(f"prompt tokens  : {stream.prompt_tokens}")
    print(f"prefill        : {stream.last_prefill_s * 1000:.0f} ms")

    per_frame = []
    reprompt_s = None
    halfway = args.frames // 2
    while not stream.finished:
        t0 = time.perf_counter()
        emitted = stream.advance(1)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        if emitted is None:
            continue
        per_frame.append(time.perf_counter() - t0)

        if args.reprompt and reprompt_s is None and len(per_frame) >= halfway:
            # Swap the caption against everything written so far. The
            # audio history is kept; only the text prefix changes.
            t0 = time.perf_counter()
            stream.reprompt(args.prompt_b)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            reprompt_s = time.perf_counter() - t0
            print(
                f"reprompt       : {reprompt_s * 1000:.0f} ms over "
                f"{len(per_frame)} frames of history"
            )

        if len(per_frame) % 100 == 0:
            print(f"  frame {len(per_frame)}/{args.frames}", flush=True)

    total_s = sum(per_frame)
    frames = len(per_frame)
    mean_ms = statistics.mean(per_frame) * 1000
    report = {
        "frames": frames,
        "audio_s": frames / AR_FRAME_RATE_HZ,
        "wall_s": round(total_s, 3),
        "ms_per_frame": round(mean_ms, 2),
        "ms_per_frame_median": round(statistics.median(per_frame) * 1000, 2),
        # 40 ms of audio per frame. Below 1.0 the stage cannot keep
        # ahead of a playhead.
        "realtime": round((1000.0 / AR_FRAME_RATE_HZ) / mean_ms, 4),
        "prompt_tokens": stream.prompt_tokens,
        "prefill_ms": round(stream.last_prefill_s * 1000, 1),
        "reprompt_ms": None if reprompt_s is None else round(reprompt_s * 1000, 1),
        "reprompt_over_frames": halfway if reprompt_s is not None else None,
        "stopped_early": stream.stopped_early,
        "controls": vars(controls),
        "blocks_ms": {
            f"{lo}-{min(lo + args.block, frames)}": round(
                statistics.mean(per_frame[lo:lo + args.block]) * 1000, 2,
            )
            for lo in range(0, frames, args.block)
            if per_frame[lo:lo + args.block]
        },
        "peak_vram_gb": (
            round(torch.cuda.max_memory_allocated(device.index or 0) / 1024**3, 2)
            if device.type == "cuda" else None
        ),
    }
    stream.close()

    print()
    print(f"frames         : {frames}  ({report['audio_s']:.1f}s of audio)")
    print(f"ms per frame   : {report['ms_per_frame']:.1f} "
          f"(median {report['ms_per_frame_median']:.1f})")
    print(f"REALTIME       : {report['realtime']:.3f}x  "
          f"{'(cannot keep ahead of a playhead)' if report['realtime'] < 1 else ''}")
    print(f"peak VRAM      : {report['peak_vram_gb']} GB")
    print("cost by position (ms/frame):")
    for label, ms in report["blocks_ms"].items():
        print(f"  {label:>12}  {ms:6.1f}")

    if args.json:
        path = Path(args.json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
