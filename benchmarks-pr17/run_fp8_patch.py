"""Standalone driver: re-run the FP8 patch on the bf16 XL decoder ONNX.

Useful for iterating on the patch logic without re-invoking the full
build CLI (which would also rebuild the 8 GB engine).

Pass ``--w8a8`` to run the activation-quantized variant using the
per-Linear absmax JSON from ``scripts/collect_activation_absmax.py``.
"""
import argparse
from pathlib import Path

from acestep.engine.trt.fp8_onnx import patch_bf16_onnx_to_fp8

SRC = Path(
    r"C:\Users\ryanf\.daydream-scope\models\demon\trt_engines"
    r"\_onnx_acestep-v15-xl-turbo\decoder_refit\decoder_refit_dynbatch.onnx"
)
AMAX_JSON = Path(
    r"C:\Users\ryanf\.daydream-scope\models\demon\calibration"
    r"\decoder_xl_fp8\activation_absmax.json"
)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--w8a8", action="store_true",
                    help="Use activation absmax JSON to insert activation "
                         "Q->DQ alongside the weight DQ (W8A8 mode).")
    ap.add_argument("--percentile",
                    choices=("absmax", "p99", "p99_9", "p99_99"),
                    default="absmax",
                    help="Activation amax field to drive the FP8 scale.")
    ap.add_argument("--outlier-skip-ratio", type=float, default=0.0,
                    help="If >0, Linears with absmax/p99.9 > ratio skip "
                         "activation Q-DQ (fall back to W8A16). Use ~10 "
                         "to route the worst transformer 'massive "
                         "activation' layers around FP8.")
    ap.add_argument("--smoothquant-alpha", type=float, default=0.0,
                    help="SmoothQuant migration strength (0.0 = off, "
                         "0.5 = standard).")
    ap.add_argument("--quantize-attention", action="store_true",
                    help="Also insert Q-DQ on the 128 dynamic-input "
                         "attention MatMuls so TRT picks FP8 GEMM tactics "
                         "for them too (currently bf16).")
    ap.add_argument("--attention-softmax-max", type=float, default=1.05,
                    help="Per-tensor activation amax for softmax-output "
                         "operands (default 1.05; range is [0,1]).")
    ap.add_argument("--attention-generic-max", type=float, default=10.0,
                    help="Per-tensor activation amax for non-softmax "
                         "attention operands (scaled Q, K, V views). "
                         "Default 10.0.")
    args = ap.parse_args()

    amax = AMAX_JSON if args.w8a8 else None
    out = patch_bf16_onnx_to_fp8(
        SRC,
        activation_absmax_json_path=amax,
        activation_percentile=args.percentile,
        activation_outlier_skip_ratio=args.outlier_skip_ratio,
        smoothquant_alpha=args.smoothquant_alpha,
        quantize_attention=args.quantize_attention,
        attention_softmax_max=args.attention_softmax_max,
        attention_generic_max=args.attention_generic_max,
        force=True,
    )
    print(f"Patched ONNX: {out}")
