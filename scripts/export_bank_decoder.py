"""Export the bank-aware DiT decoder to ONNX.

Loads the v15-turbo decoder, wraps it with ``BankAwareDecoderForExport``,
and traces to ONNX with bank K/V as engine I/O. The output ONNX has
8 inputs / 3 outputs (vs the stock 4-in / 1-out export).

Run:

    .venv/Scripts/python.exe scripts/export_bank_decoder.py \\
        --out C:/_dev/projects/DEMON_alt/trt_engines/_onnx_bank/bank_decoder.onnx \\
        --seq-len 250 --num-steps 8

The ONNX file is the input to a TRT engine build; that's a separate
step (``scripts/build_bank_engine.py``, written after we confirm the
trace succeeds).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Suppress flash_attn import (not needed for export).
import importlib, importlib.util
_orig = importlib.util.find_spec
def _patch(name, *a, **k):
    if "flash_attn" in str(name):
        return None
    return _orig(name, *a, **k)
importlib.util.find_spec = _patch

import torch  # noqa: E402
from loguru import logger  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True, help="Output ONNX path")
    p.add_argument("--seq-len", type=int, default=1500, help="Latent T (must be even; 1500 = 60s)")
    p.add_argument("--enc-len", type=int, default=64, help="Encoder seq length for trace")
    p.add_argument("--num-steps", type=int, default=8, help="Bank step axis size")
    p.add_argument("--batch-size", type=int, default=1, help="Trace batch size")
    p.add_argument("--precision", default="bf16_mixed", choices=["bf16", "bf16_mixed", "fp32"])
    p.add_argument("--device", default="cuda")
    p.add_argument(
        "--checkpoint", default=None,
        help="Path to checkpoints/ dir (default: acestep.paths.checkpoints_dir)",
    )
    args = p.parse_args()

    if args.seq_len % 2 != 0:
        raise SystemExit(f"--seq-len must be even (got {args.seq_len})")

    # Load just the decoder. Going through Session pulls in VAE / text
    # encoder / etc. which we don't need for export.
    from transformers import AutoModel

    if args.checkpoint is None:
        from acestep.paths import checkpoints_dir
        ckpt = Path(checkpoints_dir())
    else:
        ckpt = Path(args.checkpoint)
    dit_path = ckpt / "acestep-v15-turbo"
    if not dit_path.exists():
        raise SystemExit(f"DiT checkpoint not found at {dit_path}")

    logger.info("Loading model from %s ...", dit_path)
    model = AutoModel.from_pretrained(
        str(dit_path),
        trust_remote_code=True,
        attn_implementation="sdpa",
        dtype="bfloat16" if args.precision == "bf16" else "float32",
    )
    model.eval()
    decoder = model.decoder
    decoder = decoder.to(args.device)

    from acestep.engine.trt.bank_export import (
        export_bank_aware_decoder_to_onnx,
        DEFAULT_BANKED_LAYERS,
    )

    logger.info(
        "Exporting bank-aware decoder: T=%d L=%d num_steps=%d "
        "banked_layers=%s precision=%s",
        args.seq_len, args.enc_len, args.num_steps,
        list(DEFAULT_BANKED_LAYERS), args.precision,
    )
    export_bank_aware_decoder_to_onnx(
        decoder=decoder,
        onnx_path=args.out,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        enc_len=args.enc_len,
        num_steps=args.num_steps,
        banked_layers=DEFAULT_BANKED_LAYERS,
        precision=args.precision,
        device=args.device,
    )
    logger.info("Done.")


if __name__ == "__main__":
    main()
