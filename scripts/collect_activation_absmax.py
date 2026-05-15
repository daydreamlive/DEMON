"""Per-Linear activation absmax capture for W8A8 FP8 patching.

Loads the PyTorch XL DiT, hooks every ``nn.Linear`` in
``model.decoder`` with a forward pre-hook that tracks the running
``max(|input|)``, replays the calibration .npz through the model, and
writes a JSON of ``{linear_module_path -> {absmax, weight_shape,
weight_l2_bf16}}`` for the FP8 patch to consume.

Why weight_l2: the dynamo ONNX export anonymizes most Linear weight
initializers as ``val_NNN``. The FP8 patch needs a way to map an ONNX
weight initializer back to its source PyTorch Linear so it can look
up the right activation amax. Shape alone is ambiguous (XL DiT has
hundreds of (2560,1024) MatMuls); shape + L2 norm of the bf16-cast
weight bytes is unique in practice.

Usage::

    uv run python scripts/collect_activation_absmax.py
    uv run python scripts/collect_activation_absmax.py --batch 4
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

os.environ.setdefault("PYTHONUTF8", "1")
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import numpy as np
import torch
import torch.nn as nn
torch.set_grad_enabled(False)

from acestep.engine.model_context import ModelContext
from acestep.paths import models_dir, checkpoints_dir


def _hash_weight(w: torch.Tensor) -> dict:
    """Stable shape + L2-norm signature for a Linear weight in bf16.

    We compute the L2 norm of the weight tensor AFTER casting to bf16,
    because the ONNX export stores weights as bf16 and the patcher will
    re-load them as bf16. That way the signature here matches what the
    FP8 patch computes on the ONNX side, even though the live PyTorch
    weights are nominally fp32.
    """
    w_bf16 = w.detach().to(torch.bfloat16).contiguous()
    w_fp32 = w_bf16.to(torch.float32)
    return {
        "shape": list(w.shape),
        "l2_bf16": float(w_fp32.pow(2).sum().sqrt().item()),
        # First 4 elements as a tiebreaker; nonzero L2 collisions are
        # vanishingly unlikely but defense in depth costs nothing.
        "head4_bf16": [float(x) for x in w_fp32.flatten()[:4].tolist()],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--calibration",
        type=str,
        default=None,
        help="Path to calibration .npz (default: "
        "<MODELS_DIR>/calibration/decoder_xl_fp8/calibration.npz)",
    )
    ap.add_argument(
        "--checkpoint",
        type=str,
        default="acestep-v15-xl-turbo",
        help="Model checkpoint directory name (default: acestep-v15-xl-turbo)",
    )
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument(
        "--batch",
        type=int,
        default=4,
        help="Batch size to re-shape calibration samples into (default: 4)",
    )
    ap.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON path (default: <output-dir>/activation_absmax.json, "
        "or <calibration parent>/activation_absmax.json when --output-dir "
        "is unset).",
    )
    ap.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to write activation_absmax.json into. Use this to "
        "land per-profile JSONs in subdirectories (e.g., "
        "<MODELS_DIR>/calibration/decoder_xl_fp8/120s/). Ignored when "
        "--output is set.",
    )
    args = ap.parse_args()

    subdir = "decoder_xl_fp8" if "xl" in args.checkpoint else "decoder_2b_fp8"
    cal_path = Path(args.calibration) if args.calibration else (
        models_dir() / "calibration" / subdir / "calibration.npz"
    )
    if not cal_path.exists():
        raise FileNotFoundError(f"Calibration .npz not found: {cal_path}")
    if args.output:
        out_path = Path(args.output)
    elif args.output_dir:
        out_path = Path(args.output_dir) / "activation_absmax.json"
    else:
        out_path = cal_path.parent / "activation_absmax.json"

    print(f"[setup] calibration: {cal_path}")
    print(f"[setup] output:      {out_path}")
    print(f"[setup] checkpoint:  {args.checkpoint}")

    cal = np.load(str(cal_path))
    keys = ("hidden_states", "timestep", "encoder_hidden_states", "context_latents")
    arrs = {k: cal[k] for k in keys}
    n_samples = arrs["hidden_states"].shape[0]
    if n_samples % args.batch != 0:
        # Drop the tail so reshape works.
        n_use = (n_samples // args.batch) * args.batch
        for k in keys:
            arrs[k] = arrs[k][:n_use]
        n_samples = n_use
    n_batches = n_samples // args.batch
    print(f"[setup] calibration samples: {n_samples} -> {n_batches} batches of {args.batch}")

    handler = ModelContext(
        project_root=str(checkpoints_dir()),
        config_path=args.checkpoint,
        device=args.device,
        use_flash_attention=False,
        compile_decoder=False,
        compile_vae=False,
        skip_vae=True,
    )
    print("[setup] model loaded")

    with handler._load_model_context("model"):
        model = handler.model
        device = handler.device
        dtype = handler.dtype
        print(f"[setup] device={device}  dtype={dtype}")

        # Discover all Linear modules in model.decoder.
        decoder = model.decoder
        linear_modules: dict[str, nn.Linear] = {}
        for name, mod in decoder.named_modules():
            if isinstance(mod, nn.Linear):
                linear_modules[name] = mod
        print(f"[setup] found {len(linear_modules)} nn.Linear modules in model.decoder")

        # Register forward pre-hooks that record absmax and tail-percentile
        # statistics. Activation outliers in DiT (one rogue token at one
        # position) blow up the absmax for some layers — e.g. layer 16
        # mlp.down_proj has absmax ~10000 while 99.9% of its activations
        # fit under ~200. Percentile-clipped scales let FP8 use its
        # precision budget on the bulk distribution, saturating only the
        # rare outliers. We record p99/p99.9/p99.99 so the patch can
        # choose.
        absmax_state: dict[str, float] = {n: 0.0 for n in linear_modules}
        p99_state: dict[str, float] = {n: 0.0 for n in linear_modules}
        p99_9_state: dict[str, float] = {n: 0.0 for n in linear_modules}
        p99_99_state: dict[str, float] = {n: 0.0 for n in linear_modules}
        # SmoothQuant needs per-input-channel activation absmax: max over
        # batch+seq dims for each in_features channel. We keep this in
        # fp32 on CPU as a torch tensor — total memory across all
        # linears is small (~10 MB) and using CPU avoids GPU pressure
        # since these accumulators live for the whole capture run.
        per_chan_state: dict[str, torch.Tensor] = {}
        # Output absmax for each Linear (used by the attention quantization
        # path to derive scales for the post-Mul / post-View tensors
        # feeding the attention bmm/baddbmm matmuls).
        output_absmax_state: dict[str, float] = {n: 0.0 for n in linear_modules}
        hooks = []

        def _make_output_hook(linear_name: str):
            def _hook(module, inputs, output):
                if not isinstance(output, torch.Tensor):
                    return
                m = float(output.detach().abs().float().max().item())
                if m > output_absmax_state[linear_name]:
                    output_absmax_state[linear_name] = m
            return _hook

        def _make_hook(linear_name: str):
            def _hook(module, inputs):
                x = inputs[0]
                if not isinstance(x, torch.Tensor):
                    return
                abs_x = x.detach().abs().float()
                abs_flat = abs_x.flatten()
                n = abs_flat.numel()
                # absmax: literal max.
                cur_max = float(abs_flat.max().item())
                if cur_max > absmax_state[linear_name]:
                    absmax_state[linear_name] = cur_max
                # Per-batch tail quantiles via a single topk on the
                # largest k we need (the 99 percentile = the top 1% =
                # the largest k_99 = ceil(n * 0.01) elements; the
                # k-th element of that descending list is the q-th
                # quantile). max-across-batches gives a conservative
                # upper bound the engine should target.
                k99 = max(1, int(n * 0.01))
                k999 = max(1, int(n * 0.001))
                k9999 = max(1, int(n * 0.0001))
                max_k = max(k99, k999, k9999)
                if max_k >= n:
                    # Tiny tensor: percentile == absmax.
                    p99 = p999 = p9999 = cur_max
                else:
                    topk_vals = abs_flat.topk(max_k, largest=True, sorted=True).values
                    # topk_vals is sorted descending: [0]=largest, [k-1]=k-th largest.
                    p99 = float(topk_vals[k99 - 1].item())
                    p999 = float(topk_vals[k999 - 1].item())
                    p9999 = float(topk_vals[k9999 - 1].item())
                if p99 > p99_state[linear_name]:
                    p99_state[linear_name] = p99
                if p999 > p99_9_state[linear_name]:
                    p99_9_state[linear_name] = p999
                if p9999 > p99_99_state[linear_name]:
                    p99_99_state[linear_name] = p9999
                # Per-input-channel absmax for SmoothQuant. Reduce over
                # every axis except the last (which is in_features).
                if abs_x.ndim < 2:
                    # 1-D activation (rare); treat as single channel.
                    per_chan = abs_x.flatten().max().reshape(1)
                else:
                    reduce_axes = tuple(range(abs_x.ndim - 1))
                    per_chan = abs_x.amax(dim=reduce_axes)  # shape [in]
                per_chan_cpu = per_chan.detach().cpu()
                prev = per_chan_state.get(linear_name)
                if prev is None or prev.shape != per_chan_cpu.shape:
                    per_chan_state[linear_name] = per_chan_cpu.clone()
                else:
                    torch.maximum(prev, per_chan_cpu, out=prev)
            return _hook

        for name, mod in linear_modules.items():
            hooks.append(mod.register_forward_pre_hook(_make_hook(name)))
            hooks.append(mod.register_forward_hook(_make_output_hook(name)))

        # Replay calibration data through the decoder.
        for k, arr in arrs.items():
            arrs[k] = arr.reshape(n_batches, args.batch, *arr.shape[1:])
        timestep_arr = arrs["timestep"]
        # The exported wrapper uses `timestep` as both timestep and timestep_r,
        # so we replicate the same call shape here.

        print(f"[capture] running {n_batches} batches through model.decoder...")
        for bi in range(n_batches):
            hs = torch.from_numpy(arrs["hidden_states"][bi]).to(device).to(dtype)
            ts = torch.from_numpy(timestep_arr[bi]).to(device).to(dtype)
            enc = torch.from_numpy(arrs["encoder_hidden_states"][bi]).to(device).to(dtype)
            ctx = torch.from_numpy(arrs["context_latents"][bi]).to(device).to(dtype)
            decoder(
                hidden_states=hs,
                timestep=ts,
                timestep_r=ts,
                attention_mask=None,
                encoder_hidden_states=enc,
                encoder_attention_mask=None,
                context_latents=ctx,
                use_cache=False,
                past_key_values=None,
                output_attentions=False,
            )
            if (bi + 1) % 4 == 0:
                print(f"  batch {bi + 1}/{n_batches} done")

        for h in hooks:
            h.remove()

        # Build the output JSON.
        records: dict[str, dict] = {}
        for name, mod in linear_modules.items():
            w_sig = _hash_weight(mod.weight)
            per_chan = per_chan_state.get(name)
            per_chan_list = per_chan.tolist() if per_chan is not None else None
            records[name] = {
                "absmax": absmax_state[name],
                "p99": p99_state[name],
                "p99_9": p99_9_state[name],
                "p99_99": p99_99_state[name],
                "per_channel_absmax": per_chan_list,  # [in_features] for SmoothQuant
                "in_features": int(per_chan.numel()) if per_chan is not None else None,
                "weight_shape": w_sig["shape"],
                "weight_l2_bf16": w_sig["l2_bf16"],
                "weight_head4_bf16": w_sig["head4_bf16"],
                "output_absmax": output_absmax_state[name],
            }

        nonzero = sum(1 for r in records.values() if r["absmax"] > 0)
        print(f"[capture] linear modules with nonzero absmax: {nonzero}/{len(records)}")
        amaxes_sorted = sorted(
            [(r["absmax"], r["p99"], r["p99_9"], r["p99_99"], n)
             for n, r in records.items()],
            reverse=True,
        )
        print("[capture] top 5 by absmax (absmax, p99, p99.9, p99.99):")
        for amax, p99, p999, p9999, n in amaxes_sorted[:5]:
            print(f"  amax={amax:>9.2f}  p99={p99:>9.2f}  p99.9={p999:>9.2f}  "
                  f"p99.99={p9999:>9.2f}  {n}")
        print("[capture] bottom 5 by absmax (nonzero):")
        nz = [t for t in amaxes_sorted if t[0] > 0]
        for amax, p99, p999, p9999, n in nz[-5:]:
            print(f"  amax={amax:>9.2e}  p99={p99:>9.2e}  p99.9={p999:>9.2e}  "
                  f"p99.99={p9999:>9.2e}  {n}")
        # Outlier diagnostic: how much does each quantile shrink vs absmax?
        ratios = []
        for name, r in records.items():
            if r["absmax"] > 0:
                ratios.append((r["absmax"] / max(r["p99_9"], 1e-12), name, r))
        ratios.sort(key=lambda t: t[0], reverse=True)
        print("[capture] top 10 outlier ratios (absmax / p99.9):")
        for ratio, name, r in ratios[:10]:
            print(f"  ratio={ratio:7.1f}x  amax={r['absmax']:>9.2f}  "
                  f"p99.9={r['p99_9']:>9.2f}  {name}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "checkpoint": args.checkpoint,
                "calibration_npz": str(cal_path),
                "batch": args.batch,
                "n_batches": n_batches,
                "n_samples": n_samples,
                "linears": records,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[save] wrote {out_path}")


if __name__ == "__main__":
    main()
