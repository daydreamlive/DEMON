"""Empirically reverse-engineer the CUTLASS NVFP4 scale-factor swizzle.

Plant a "2.0" scale in exactly one position of scale_A while leaving all other
scales at "1.0" and all data at unit FP4 value. Then observe which output row
gets doubled — that tells us where the swizzled scale (0,0) actually maps to.

Repeat across a grid of probe positions to fully map the swizzle.
"""

import torch
import os, sys

sys.path.insert(0, os.path.dirname(__file__))
from nvfp4_e2e import (
    cublaslt_nvfp4_matmul, pack_fp4_pairs, FP4_LEVELS, FP4_CODES,
)


def run_probe(M, N, K, probe_row, probe_blk):
    """Set scale_A[probe_row, probe_blk] = 2.0, all else 1.0; A = all 1's,
    B = all 1's, scale_B = all 1's. Output entry that ends up == 2*K means
    that row's scale was doubled. Anywhere it ended up != K (1*K) is the
    image of (probe_row, probe_blk)."""
    device = torch.device("cuda")
    # FP4 value 1.0 = code 0x2 (positive, ord=2). Pack two of those:
    #   byte = (0x2 << 4) | 0x2 = 0x22
    A_bytes = torch.full((M, K // 2), 0x22, dtype=torch.uint8, device=device)
    B_bytes = torch.full((N, K // 2), 0x22, dtype=torch.uint8, device=device)
    # Scales: all 1.0 (FP8 E4M3 = 0x38), except A[probe_row, probe_blk] = 2.0 (0x40)
    scale_A = torch.full((M, K // 16), 0x38, dtype=torch.uint8, device=device)
    scale_B = torch.full((N, K // 16), 0x38, dtype=torch.uint8, device=device)
    scale_A[probe_row, probe_blk] = 0x40  # 2.0 in E4M3
    D = cublaslt_nvfp4_matmul(
        A_bytes, B_bytes, scale_A, scale_B, alpha=1.0, M=M, N=N, K=K
    ).float()
    # Expected baseline: every output = K (sum of K 1*1 products)
    # If swizzle is identity row-major, row=probe_row should jump by 16 (one block of 16 ones doubled)
    # i.e., row probe_row should == K + 16 (with the +16 located at probe_blk's 16 sub-products)
    return D


def find_doubled_row(D, K):
    """Return the row indices where row-sum is increased above K*N."""
    row_sums = D.sum(dim=-1)
    expected = float(K) * D.shape[-1]  # K * N
    diff = row_sums - expected
    # find rows where diff > 5% of expected
    doubled = torch.nonzero(diff.abs() > expected * 0.005).flatten().tolist()
    return doubled, diff


def main():
    M, N, K = 128, 64, 128  # 8 blocks per row in scale_A
    print(f"Probing M={M} N={N} K={K} ({K//16} scale blocks per row)")
    print()

    # Probe each (row, blk) cell of scale_A
    print("scale_A[r, b] = 2.0 -> which output rows show >1.0 sum perturbation?")
    print()
    # Sample a few key cells
    cells = []
    for r in [0, 1, 2, 8, 16, 32, 64, 127]:
        for b in [0, 1, 2, 3, 4, 7]:
            if b >= K // 16:
                continue
            cells.append((r, b))

    for r, b in cells:
        D = run_probe(M, N, K, r, b)
        doubled, diff = find_doubled_row(D, K)
        # Top-5 most-perturbed rows
        top = sorted(range(M), key=lambda i: -abs(diff[i].item()))[:5]
        msg_top = ", ".join(f"r{i}(+{diff[i].item():+.1f})" for i in top)
        print(f"  probe(r={r:3d}, b={b}): top perturbed rows: {msg_top}")


if __name__ == "__main__":
    main()
