"""Convert an SA3 checkpoint's fp32 weights to fp16 on disk.

The medium DiT ships as ~9 GB of **float32** safetensors, but DEMON runs
it in fp16 (``load_diffusion_cond(..., model_half=True)`` does
``model.to(torch.float16)`` after loading). Storing fp32 therefore costs
2x the disk/VRAM-transfer bytes for precision we immediately discard.

This script rewrites the checkpoint with fp32 tensors cast to fp16,
which:
  * halves the DiT file (~8.6 GiB -> ~4.3 GiB) — smaller Docker image
    and less to read at load, and
  * is numerically identical at runtime: today's path already does
    fp16(fp32_weights); pre-converting yields fp16(fp32_weights) stored,
    and load_state_dict upcasts fp16->fp32 (exact) before the same
    ``.to(fp16)`` — so the resident weights are bit-for-bit the same.

Only **float32** tensors are converted. bf16 tensors (the bundled
T5Gemma encoder) are left alone: bf16 is already 2 bytes, so converting
saves no space and would needlessly change its stored precision (runtime
still casts it to fp16 at load, exactly as before). int/bool tensors and
the safetensors ``__metadata__`` header are preserved verbatim.

Usage:
    # write an fp16 copy next to the source (default: <dir>-fp16/)
    python scripts/sa3/sa3_convert_fp16.py --model-id medium

    # overwrite the canonical checkpoint in place (image-bake use case)
    python scripts/sa3/sa3_convert_fp16.py --model-id medium --in-place
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import struct
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = next(p for p in (_HERE, *_HERE.parents) if (p / "pyproject.toml").exists())
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from acestep.engine.sa3_helpers import sa3_checkpoint_dir  # noqa: E402


def _header_dtypes(path: Path) -> set[str]:
    """The set of tensor dtype strings in a safetensors file, read from
    the JSON header only (no tensor data loaded)."""
    with open(path, "rb") as fh:
        (n,) = struct.unpack("<Q", fh.read(8))
        header = json.loads(fh.read(n))
    return {v["dtype"] for k, v in header.items() if k != "__metadata__"}


def convert_or_copy_safetensors(src: Path, dst: Path) -> tuple[int, int]:
    """Write ``dst`` from ``src`` with float32 tensors cast to fp16.

    Returns ``(converted, total)`` tensor counts. If the file has no
    float32 tensors, it is copied byte-for-byte (no re-serialization) so
    bf16/int payloads and header ordering are preserved exactly.
    """
    if "F32" not in _header_dtypes(src):
        shutil.copy2(src, dst)
        return 0, 0

    import torch  # noqa: PLC0415
    from safetensors import safe_open  # noqa: PLC0415
    from safetensors.torch import load_file, save_file  # noqa: PLC0415

    with safe_open(str(src), framework="pt") as f:
        metadata = f.metadata()  # preserved (may be None)
    tensors = load_file(str(src))
    out: dict = {}
    converted = 0
    for key, value in tensors.items():
        if value.dtype == torch.float32:
            value = value.to(torch.float16)
            converted += 1
        out[key] = value.contiguous()
    save_file(out, str(dst), metadata=metadata)
    return converted, len(tensors)


def convert_checkpoint(src_dir: Path, dst_dir: Path) -> None:
    """Materialize an fp16 copy of an SA3 checkpoint dir at ``dst_dir``.

    Every ``*.safetensors`` has its float32 tensors cast to fp16; all
    other files (model_config.json, tokenizer, t5gemma bf16 weights, …)
    are copied verbatim. The HF ``.cache`` download dir is skipped.
    """
    total_before = total_after = 0
    for src_path in sorted(src_dir.rglob("*")):
        rel = src_path.relative_to(src_dir)
        if ".cache" in rel.parts:
            continue
        dst_path = dst_dir / rel
        if src_path.is_dir():
            dst_path.mkdir(parents=True, exist_ok=True)
            continue
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        if src_path.suffix == ".safetensors":
            before = src_path.stat().st_size
            converted, total = convert_or_copy_safetensors(src_path, dst_path)
            after = dst_path.stat().st_size
            total_before += before
            total_after += after
            print(
                f"  {rel}: {converted}/{total} fp32->fp16  "
                f"{before / 1e9:.2f} GB -> {after / 1e9:.2f} GB"
            )
        else:
            shutil.copy2(src_path, dst_path)
    if total_before:
        print(
            f"[convert] safetensors total: {total_before / 1e9:.2f} GB -> "
            f"{total_after / 1e9:.2f} GB "
            f"({100 * (1 - total_after / total_before):.0f}% smaller)"
        )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--model-id", default="medium", help='e.g. "medium", "small-music"')
    ap.add_argument("--src", default=None, help="checkpoint dir (overrides --model-id)")
    ap.add_argument("--out", default=None, help="output dir (default: <src>-fp16)")
    ap.add_argument(
        "--in-place", action="store_true",
        help="overwrite the source checkpoint's safetensors instead of copying",
    )
    ap.add_argument("--force", action="store_true", help="overwrite an existing --out")
    args = ap.parse_args()

    src = Path(args.src) if args.src else sa3_checkpoint_dir(args.model_id)
    if not (src / "model.safetensors").is_file():
        print(f"[convert] no checkpoint at {src}", file=sys.stderr)
        return 1

    if args.in_place:
        # Convert each safetensors to a temp sibling, then atomically
        # replace — never leave a half-written weight file at the path
        # the loader/preflight reads.
        print(f"[convert] in-place: {src}")
        for st in sorted(src.rglob("*.safetensors")):
            if ".cache" in st.relative_to(src).parts:
                continue
            tmp = st.with_suffix(".safetensors.fp16.tmp")
            converted, total = convert_or_copy_safetensors(st, tmp)
            if converted:
                os.replace(tmp, st)
                print(f"  {st.relative_to(src)}: {converted}/{total} fp32->fp16")
            else:
                tmp.unlink(missing_ok=True)
        return 0

    dst = Path(args.out) if args.out else src.parent / f"{src.name}-fp16"
    if dst.exists() and not args.force:
        print(f"[convert] {dst} exists (use --force to overwrite)", file=sys.stderr)
        return 1
    print(f"[convert] {src}  ->  {dst}")
    convert_checkpoint(src, dst)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
