"""Diagnose where the cos~0.89 error in nvfp4_e2e.py comes from.

For a tiny case: K=16 (single block), M=N=1. Compute D three ways:
  1. FP32 reference
  2. FP32 reconstruction from the quantized FP4 + FP8 block scale + alpha
  3. cublasLt NVFP4 GEMM
If 2 ~ 1 but 3 != 2, the API call is wrong. If 2 != 1, the quant is wrong.
"""

import torch
import os, sys

sys.path.insert(0, os.path.dirname(__file__))
from nvfp4_e2e import (
    quantize_to_nvfp4, cublaslt_nvfp4_matmul,
    fp8_e4m3_decode, FP4_LEVELS, FP4_CODES,
)


def unpack_fp4(packed: torch.Tensor):
    """Inverse of pack_fp4_pairs: byte -> two FP32 values."""
    fp4_levels = FP4_LEVELS.to(packed.device)
    fp4_codes = FP4_CODES.to(packed.device)
    # code -> level index (inverse of FP4_CODES)
    code_to_idx = torch.zeros(16, dtype=torch.long, device=packed.device)
    for i, c in enumerate(fp4_codes.tolist()):
        code_to_idx[c] = i
    lo = packed & 0xF
    hi = (packed >> 4) & 0xF
    lo_val = fp4_levels[code_to_idx[lo.long()]]
    hi_val = fp4_levels[code_to_idx[hi.long()]]
    # interleave: (..., K/2) -> (..., K)
    out = torch.empty(
        (*packed.shape[:-1], packed.shape[-1] * 2),
        dtype=torch.float32, device=packed.device,
    )
    out[..., 0::2] = lo_val
    out[..., 1::2] = hi_val
    return out


def reconstruct_fp32(packed, blk_scale_e4m3, G, K, block=16):
    """Decode (packed FP4, FP8 block scale, FP32 global) back to FP32."""
    fp4_vals = unpack_fp4(packed)  # (M, K)
    # blk_scale shape (M, K/block); broadcast along block
    M = fp4_vals.shape[0]
    fp8_dec = fp8_e4m3_decode(blk_scale_e4m3.flatten()).reshape(M, K // block)
    expanded = fp8_dec.repeat_interleave(block, dim=-1)  # (M, K)
    return fp4_vals * expanded * float(G)


def run():
    device = torch.device("cuda")
    torch.manual_seed(0)
    M, N, K = 128, 128, 128
    x = torch.randn((M, K), dtype=torch.float32, device=device)
    w = torch.randn((N, K), dtype=torch.float32, device=device) / (K ** 0.5)

    ref = (x @ w.T)
    print("ref[0]:", ref[0])

    x_pack, x_blk, x_g = quantize_to_nvfp4(x, block=16)
    w_pack, w_blk, w_g = quantize_to_nvfp4(w, block=16)
    print(f"x global={x_g.item():.4g} w global={w_g.item():.4g}")
    print(f"x_pack shape={x_pack.shape} x_blk shape={x_blk.shape}")

    # Reference-decode and recompute
    x_dec = reconstruct_fp32(x_pack, x_blk, x_g, K)
    w_dec = reconstruct_fp32(w_pack, w_blk, w_g, K)
    print("x_dec[0,:8]:", x_dec[0, :8])
    print("x   [0,:8]:", x[0, :8])
    rec_quality_x = ((x_dec - x).norm() / x.norm()).item()
    rec_quality_w = ((w_dec - w).norm() / w.norm()).item()
    print(f"x recon rel err: {rec_quality_x:.4f}")
    print(f"w recon rel err: {rec_quality_w:.4f}")

    ref_dec = x_dec @ w_dec.T
    print("ref_dec[0]:", ref_dec[0])

    # Apply cuBLAS SF swizzle (128 x 4 tile -> 32 x 16 layout, flattened)
    from torchao.prototype.mx_formats.utils import to_blocked
    x_blk_swz = to_blocked(x_blk).contiguous()
    w_blk_swz = to_blocked(w_blk).contiguous()

    alpha = float(x_g) * float(w_g)
    D = cublaslt_nvfp4_matmul(x_pack, w_pack, x_blk_swz, w_blk_swz, alpha, M, N, K).float()
    print("D    [0,:8]:", D[0, :8])
    print("ref  [0,:8]:", ref[0, :8])

    print()
    print(f"cos(ref, ref_dec):  {torch.cosine_similarity(ref.flatten(), ref_dec.flatten(), dim=0).item():.4f}")
    print(f"cos(ref, D):        {torch.cosine_similarity(ref.flatten(), D.flatten(), dim=0).item():.4f}")
    print(f"cos(ref_dec, D):    {torch.cosine_similarity(ref_dec.flatten(), D.flatten(), dim=0).item():.4f}")

    # --- Uniform-scale test: if all FP8 scales are exactly 1.0 (0x38), then the
    # scale tensor layout doesn't matter and any correct GEMM impl should give
    # the same result regardless of swizzle. This isolates the data path from
    # the scale-layout path.
    print()
    print("=== Uniform-scale (all 1.0) test ===")
    # Build x, w directly in FP4-representable range so quant is exact.
    # Use values from FP4_LEVELS for both. Pack as we normally do.
    x_uni_idx = torch.randint(0, 16, (M, K), dtype=torch.uint8, device=device)
    w_uni_idx = torch.randint(0, 16, (N, K), dtype=torch.uint8, device=device)
    # Get FP32 values for the reference
    from nvfp4_e2e import FP4_LEVELS
    levels = FP4_LEVELS.to(device)
    x_uni_fp32 = levels[x_uni_idx.long()]
    w_uni_fp32 = levels[w_uni_idx.long()]
    ref_uni = x_uni_fp32 @ w_uni_fp32.T
    # Pack
    from nvfp4_e2e import pack_fp4_pairs
    x_uni_pack = pack_fp4_pairs(x_uni_idx)
    w_uni_pack = pack_fp4_pairs(w_uni_idx)
    # All FP8 E4M3 = 1.0 (bit 0x38)
    x_uni_scale = torch.full((M, K // 16), 0x38, dtype=torch.uint8, device=device)
    w_uni_scale = torch.full((N, K // 16), 0x38, dtype=torch.uint8, device=device)
    D_uni = cublaslt_nvfp4_matmul(
        x_uni_pack, w_uni_pack, x_uni_scale, w_uni_scale, 1.0, M, N, K
    ).float()
    print("ref_uni[0,:4]:", ref_uni[0, :4])
    print("D_uni  [0,:4]:", D_uni[0, :4])
    cos_uni = torch.cosine_similarity(ref_uni.flatten(), D_uni.flatten(), dim=0).item()
    rel_uni = ((D_uni - ref_uni).norm() / ref_uni.norm()).item()
    print(f"cos(ref_uni, D_uni):  {cos_uni:.6f}")
    print(f"rel L2:                {rel_uni:.6f}")


if __name__ == "__main__":
    run()
