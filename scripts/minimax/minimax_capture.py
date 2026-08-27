"""Capture a MiniMax-Music3 composition: run the AR stage once, keep the tensor.

DEMON never streams MiniMax's 8.58B autoregressive LM. It runs it once per
composition and covers the result forever, which makes the capture — not the
prompt — the reusable artifact. This CLI produces one.

Output is a ``.safetensors`` file with two keys:

``frame_hiddens``
    ``[1, frames, 8*4096]`` bf16 — the RAW fused per-frame hidden states
    straight off the AR stage. This is the durable form: it survives any
    change to the ConditionEncoder and can be re-projected later.

``encoder_hidden_states``
    ``[1, latent_frames, 2048]`` — ``frame_hiddens`` pushed through the
    ConditionEncoder, i.e. exactly what the renderer wants. Written only when
    :mod:`acestep.engine.minimax_dit` is importable; the raw key is always
    written, so a capture taken before that module exists is not wasted.

Usage::

    python scripts/minimax/minimax_capture.py \
        --prompt "bpm is 92. key is E, and scale is minor. Electric Blues." \
        --lyrics "[verse]\\nI'm learning how to fill up" \
        --seconds 4 --out out/capture.safetensors

``--prompt`` and ``--lyrics`` accept a literal string or a path to a text
file; captions that actually work on this checkpoint run to several hundred
words, which is not a command line.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
# A sibling ACE-Step checkout on sys.path shadows `acestep`; force ours first.
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402
from safetensors.torch import save_file  # noqa: E402

from acestep.engine.minimax_ar import AR_FRAME_RATE_HZ, MiniMaxAR  # noqa: E402
from acestep.engine.minimax_helpers import resolve_model_dir  # noqa: E402

# LM 17.2 GB + depth decoder 1.3 GB + the lm_head logits and KV cache. The
# card is shared, so refuse to start rather than OOM someone else's run.
STACK_VRAM_GB = 21.0


def _text_arg(value: str) -> str:
    """A literal string, or the contents of a file if the value is a path."""
    try:
        candidate = Path(value)
        if candidate.is_file():
            return candidate.read_text(encoding="utf-8")
    except OSError:
        pass
    return value


def _free_vram_gb(device_index: int) -> float:
    free, _total = torch.cuda.mem_get_info(device_index)
    return free / 1024**3


def _wait_for_vram(device_index: int, need_gb: float, timeout_s: float) -> float:
    """Block until the card has room. Other agents share this GPU."""
    deadline = time.monotonic() + timeout_s
    while True:
        free = _free_vram_gb(device_index)
        if free >= need_gb:
            return free
        if time.monotonic() >= deadline:
            raise RuntimeError(
                f"cuda:{device_index} has {free:.1f} GB free, need {need_gb:.1f} GB "
                f"and waited {timeout_s:.0f}s. Another job is holding the card; "
                "retry later or pass --device cpu."
            )
        print(
            f"  waiting for VRAM: {free:.1f} GB free, need {need_gb:.1f} GB",
            flush=True,
        )
        time.sleep(15.0)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--prompt", required=True, help="Music description, or a path to one.")
    parser.add_argument("--lyrics", required=True, help="Lyrics, or a path to them.")
    parser.add_argument("--seconds", type=float, default=4.0, help="Audio length to capture.")
    parser.add_argument("--out", type=Path, required=True, help="Destination .safetensors.")
    parser.add_argument("--seed", type=int, default=0, help="Sampling seed (upstream default is 0).")
    parser.add_argument("--model-dir", default=None, help="Checkpoint root; auto-resolved otherwise.")
    parser.add_argument("--device", default="cuda", help="cuda, cuda:N, or cpu.")
    parser.add_argument(
        "--sample-on-cpu",
        action="store_true",
        help="Draw every sample from a CPU generator so the capture reproduces "
        "across devices. Costs eight device syncs per frame.",
    )
    parser.add_argument(
        "--no-condition-encoder",
        action="store_true",
        help="Write only the raw frame_hiddens, even if minimax_dit is importable.",
    )
    parser.add_argument(
        "--vram-timeout",
        type=float,
        default=600.0,
        help="Seconds to wait for a busy shared GPU before giving up.",
    )
    return parser


def main() -> int:
    args = _parser().parse_args()

    prompt = _text_arg(args.prompt)
    lyrics = _text_arg(args.lyrics)
    if args.seconds <= 0:
        raise SystemExit("--seconds must be positive")
    frames = int(round(args.seconds * AR_FRAME_RATE_HZ))
    if frames < 1:
        raise SystemExit(
            f"--seconds {args.seconds} is shorter than one frame (1/{AR_FRAME_RATE_HZ:g} s)"
        )

    root = Path(args.model_dir) if args.model_dir else resolve_model_dir()
    device = torch.device(args.device)
    print(f"checkpoint     : {root}")
    print(f"device         : {device}")
    print(f"frames         : {frames}  ({args.seconds:g}s @ {AR_FRAME_RATE_HZ:g} Hz)")
    print(f"seed           : {args.seed}")

    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit("CUDA requested but unavailable")
        index = device.index or 0
        free = _wait_for_vram(index, STACK_VRAM_GB, args.vram_timeout)
        print(f"free VRAM      : {free:.1f} GB (need ~{STACK_VRAM_GB:.0f} GB)")
        torch.cuda.reset_peak_memory_stats(index)

    # Load on the CPU, then page across in one move — the same path
    # MiniMaxContext takes under ar_policy="offload".
    load_started = time.perf_counter()
    ar = MiniMaxAR.from_pretrained(
        root,
        dtype=torch.bfloat16,
        device="cpu",
        seed=args.seed,
        sample_on_cpu=args.sample_on_cpu,
    )
    load_s = time.perf_counter() - load_started
    print(f"loaded (cpu)   : {load_s:.1f}s")

    page_started = time.perf_counter()
    ar.to(device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    page_s = time.perf_counter() - page_started
    print(f"paged to device: {page_s:.1f}s")

    last_report = [0.0]

    def _progress(done: int, total: int) -> None:
        now = time.monotonic()
        if now - last_report[0] >= 2.0 or done == total:
            last_report[0] = now
            print(f"  frame {done}/{total}", flush=True)

    capture_started = time.perf_counter()
    frame_hiddens = ar.generate_frame_hiddens(
        prompt=prompt, lyrics=lyrics, frames=frames, seed=args.seed, progress=_progress
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    capture_s = time.perf_counter() - capture_started
    stats = dict(ar.last_stats)

    peak_gb = 0.0
    if device.type == "cuda":
        peak_gb = torch.cuda.max_memory_allocated(device.index or 0) / 1024**3

    tensors = {"frame_hiddens": frame_hiddens.detach().to("cpu").contiguous()}

    cond_shape = None
    if not args.no_condition_encoder:
        try:
            from acestep.engine.minimax_dit import MiniMaxConditionEncoder
        except Exception as exc:  # module is another agent's; may not exist yet
            print(f"condition enc. : skipped ({type(exc).__name__}: {exc})")
        else:
            # The encoder is ~25M params; run it wherever the capture already
            # is rather than paging the AR stack out first.
            encoder = MiniMaxConditionEncoder.from_pretrained(
                root, dtype=torch.float32, device=device
            )
            with torch.no_grad():
                cond = encoder(frame_hiddens.to(device=device, dtype=torch.float32))
            cond_shape = tuple(cond.shape)
            tensors["encoder_hidden_states"] = cond.detach().to("cpu").contiguous()
            del encoder
            print(f"condition enc. : {cond_shape} {cond.dtype}")

    # Free the card before touching disk; the write can take a moment and the
    # GPU is shared.
    ar.to("cpu")
    if device.type == "cuda":
        torch.cuda.empty_cache()

    out = args.out.expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "producer": "scripts/minimax/minimax_capture.py",
        "frame_rate_hz": str(AR_FRAME_RATE_HZ),
        "seed": str(args.seed),
        "frames": str(stats["frames"]),
        "requested_frames": str(frames),
        "stopped_early": str(bool(stats["stopped_early"])),
        "prompt_tokens": str(stats["prompt_tokens"]),
        "checkpoint": str(root),
    }
    save_file(tensors, str(out), metadata=metadata)

    print("-" * 62)
    print(f"frame_hiddens  : {tuple(frame_hiddens.shape)} {frame_hiddens.dtype}")
    if cond_shape is not None:
        print(f"encoder_hidden : {cond_shape} {tensors['encoder_hidden_states'].dtype}")
    print(f"audio captured : {stats['audio_seconds']:.2f}s"
          + (" (LM stopped early)" if stats["stopped_early"] else ""))
    print(f"wall clock     : {capture_s:.2f}s capture, {load_s + page_s + capture_s:.1f}s total")
    print(f"throughput     : {stats['lm_tokens_per_s']:.2f} LM tok/s"
          f" | {stats['codes_per_s']:.1f} codes/s"
          f" | {stats['realtime_factor']:.3f}x realtime")
    print(f"peak VRAM      : {peak_gb:.2f} GB")
    print(f"written        : {out}  ({out.stat().st_size / 1e6:.1f} MB)")
    print(json.dumps(stats, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
