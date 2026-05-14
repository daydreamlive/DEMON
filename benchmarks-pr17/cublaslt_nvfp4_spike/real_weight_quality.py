"""Per-layer NVFP4 quality test on real ACE-Step XL turbo DiT weights.

For each Linear (sampled across layers and projection types):
  - Load real bf16 weight
  - Quantize to NVFP4 (per-block-16 FP8 + per-tensor FP32)
  - Run a synthetic activation (gaussian, scaled to match calibrated absmax)
    through both bf16 and NVFP4 paths
  - Report cos sim and rel L2

Output identifies layers that may need FP8 fallback under NVFP4.
"""

import json
import os
import sys

import torch
from safetensors import safe_open

sys.path.insert(0, os.path.dirname(__file__))
from nvfp4_e2e import cublaslt_nvfp4_matmul, quantize_to_nvfp4  # noqa: E402

CKPT_DIR = r"C:/Users/ryanf/.daydream-scope/models/demon/checkpoints/acestep-v15-xl-turbo"
ABSMAX_JSON = r"C:/Users/ryanf/.daydream-scope/models/demon/calibration/decoder_xl_fp8/activation_absmax.json"


def load_weight_map():
    """Returns dict: param_name -> file path."""
    index_path = os.path.join(CKPT_DIR, "model.safetensors.index.json")
    with open(index_path) as f:
        idx = json.load(f)
    weight_map = idx["weight_map"]
    return {k: os.path.join(CKPT_DIR, v) for k, v in weight_map.items()}


def load_tensor(name, weight_map) -> torch.Tensor:
    path = weight_map[name]
    with safe_open(path, framework="pt") as f:
        return f.get_tensor(name)


def per_layer_test(weight_name, weight, input_absmax, M=6000):
    """weight shape: (N, K). Runs Y = X @ W^T comparison.
    input_absmax: per-tensor absmax used to scale a Gaussian input to match the
    real activation distribution at this layer.
    """
    device = torch.device("cuda")
    weight = weight.to(device=device, dtype=torch.float32)
    N, K = weight.shape

    if K % 64 != 0 or N % 32 != 0:
        return {"name": weight_name, "skip_reason": f"shape {N}x{K} not aligned"}

    # Build synthetic input scaled to typical absmax (gaussian inputs with absmax matching cal)
    # IID gaussian has expected absmax ~ sqrt(2*log(M*K)) = ~5.5 for M*K~1e7
    torch.manual_seed(0)
    x = torch.randn((M, K), dtype=torch.float32, device=device)
    if input_absmax is not None and input_absmax > 0:
        x = x * (input_absmax / 5.5)

    # bf16 reference
    ref = (x.to(torch.bfloat16) @ weight.to(torch.bfloat16).T).float()

    # NVFP4 path
    x_pack, x_blk, x_g = quantize_to_nvfp4(x, block=16)
    w_pack, w_blk, w_g = quantize_to_nvfp4(weight, block=16)
    from torchao.prototype.mx_formats.utils import to_blocked
    x_blk_swz = to_blocked(x_blk).contiguous()
    w_blk_swz = to_blocked(w_blk).contiguous()
    alpha = float(x_g) * float(w_g)
    D = cublaslt_nvfp4_matmul(x_pack, w_pack, x_blk_swz, w_blk_swz, alpha, M, N, K).float()

    cos = torch.cosine_similarity(ref.flatten(), D.flatten(), dim=0).item()
    rel = ((D - ref).norm() / ref.norm()).item()

    return {
        "name": weight_name,
        "N": N, "K": K,
        "input_absmax": input_absmax,
        "weight_absmax": weight.abs().max().item(),
        "x_global_scale": float(x_g),
        "w_global_scale": float(w_g),
        "cos": cos,
        "rel_l2": rel,
    }


def main():
    # Load absmax data (per Linear)
    with open(ABSMAX_JSON) as f:
        absmax_data = json.load(f)["linears"]
    print(f"Loaded absmax for {len(absmax_data)} Linears")

    # Load weight index
    weight_map = load_weight_map()
    print(f"Loaded weight map with {len(weight_map)} params")

    # Pick a representative subset: layer 0, 15 (known outlier zone per handoff), 26 (last) -
    # and the 5 main projection types per layer
    sample_layers = [0, 5, 10, 15, 20, 26]
    proj_types = [
        "self_attn.q_proj",
        "self_attn.k_proj",
        "self_attn.v_proj",
        "self_attn.out_proj",
        "mlp.gate_proj",
        "mlp.up_proj",
        "mlp.down_proj",
    ]

    results = []
    for layer in sample_layers:
        for pt in proj_types:
            # Try canonical and onnx-style names
            absmax_key = f"layers.{layer}.{pt}"
            weight_key = f"decoder.layers.{layer}.{pt}.weight"
            if weight_key not in weight_map:
                continue
            absmax_entry = absmax_data.get(absmax_key, {})
            input_absmax = absmax_entry.get("absmax", None)

            w = load_tensor(weight_key, weight_map)
            r = per_layer_test(weight_key, w, input_absmax)
            results.append(r)
            if "skip_reason" in r:
                print(f"  SKIP {r['name']}: {r['skip_reason']}")
            else:
                print(
                    f"  {r['name']:50s}  N={r['N']:5d} K={r['K']:5d}  "
                    f"x_amax={r['input_absmax']:.2f}  cos={r['cos']:.5f}  rel_l2={r['rel_l2']:.4f}"
                )

    # Summary
    valid = [r for r in results if "cos" in r]
    if valid:
        print(f"\nSummary over {len(valid)} layers:")
        cos_vals = [r["cos"] for r in valid]
        rel_vals = [r["rel_l2"] for r in valid]
        print(f"  cos sim:   min={min(cos_vals):.4f}  mean={sum(cos_vals)/len(cos_vals):.4f}  max={max(cos_vals):.4f}")
        print(f"  rel L2:    min={min(rel_vals):.4f}  mean={sum(rel_vals)/len(rel_vals):.4f}  max={max(rel_vals):.4f}")
        worst = sorted(valid, key=lambda r: r["cos"])[:5]
        print(f"\n  Worst 5 layers by cos sim:")
        for r in worst:
            print(f"    {r['name']:50s}  cos={r['cos']:.5f}  rel_l2={r['rel_l2']:.4f}  w_amax={r['weight_absmax']:.3f}")

    # Save for follow-up analysis
    out_path = os.path.join(os.path.dirname(__file__), "real_weight_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
