"""End-to-end NVFP4 Linear correctness + speed check.

Builds a real bf16 activation x weight matmul, quantizes both to NVFP4 (FP4
E2M1 packed + per-block-16 FP8 E4M3 scales + per-tensor FP32 global scale),
runs the cuBLASLt NVFP4 GEMM via the ctypes wrapper, and compares against
the bf16 reference. Reports relative L2 error and TFLOPS.

This is the "is the quantization scheme actually usable" check on top of the
pure-throughput nvfp4_gemm.py spike. The TRT plugin will replicate this same
quant scheme at runtime, with weights pre-quantized at engine build time.
"""

import ctypes
import os
import sys

import torch

# Re-use bindings from the throughput spike
sys.path.insert(0, os.path.dirname(__file__))
from nvfp4_gemm import (  # noqa: E402
    CUBLAS_COMPUTE_32F,
    CUBLAS_OP_N,
    CUBLAS_OP_T,
    CUBLAS_STATUS_SUCCESS,
    CUBLASLT_MATMUL_DESC_A_SCALE_MODE,
    CUBLASLT_MATMUL_DESC_A_SCALE_POINTER,
    CUBLASLT_MATMUL_DESC_B_SCALE_MODE,
    CUBLASLT_MATMUL_DESC_B_SCALE_POINTER,
    CUBLASLT_MATMUL_DESC_TRANSA,
    CUBLASLT_MATMUL_DESC_TRANSB,
    CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3,
    CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
    CUBLASLT_MATRIX_LAYOUT_ORDER,
    CUBLASLT_ORDER_ROW,
    CUDA_R_16BF,
    CUDA_R_32F,
    CUDA_R_4F_E2M1,
    HeuristicResult,
    check,
    cublasLtCreate,
    cublasLtDestroy,
    cublasLtMatmul,
    cublasLtMatmulAlgoGetHeuristic,
    cublasLtMatmulDescCreate,
    cublasLtMatmulDescDestroy,
    cublasLtMatmulDescSetAttribute,
    cublasLtMatmulPreferenceCreate,
    cublasLtMatmulPreferenceDestroy,
    cublasLtMatmulPreferenceSetAttribute,
    cublasLtMatrixLayoutCreate,
    cublasLtMatrixLayoutDestroy,
    cublasLtMatrixLayoutSetAttribute,
)


# --- FP4 / FP8 quantization helpers (CPU-side reference) --------------------

# NVFP4 E2M1 representable positive values: 0, 0.5, 1, 1.5, 2, 3, 4, 6
FP4_LEVELS = torch.tensor(
    [-6.0, -4.0, -3.0, -2.0, -1.5, -1.0, -0.5, 0.0,
     0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
    dtype=torch.float32,
)
# Encoded bit patterns (4 bits): sign|exp(2)|mantissa(1). Map level index -> code.
# Per OCP MX-FP4 / NV FP4: negative levels 0..7 = 0x8..0xF (sign=1), positive 0..7 = 0x0..0x7
FP4_CODES = torch.tensor(
    [0xF, 0xE, 0xD, 0xC, 0xB, 0xA, 0x9, 0x8,
     0x0, 0x1, 0x2, 0x3, 0x4, 0x5, 0x6, 0x7],
    dtype=torch.uint8,
)
FP4_MAX = 6.0
FP8_E4M3_MAX = 448.0  # standard E4M3 max representable


def fp8_e4m3_encode(x: torch.Tensor) -> torch.Tensor:
    """Encode FP32 tensor to FP8 E4M3 bit pattern (uint8). Clamps to range."""
    fp8 = x.clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX).to(torch.float8_e4m3fn)
    return fp8.view(torch.uint8)


def fp8_e4m3_decode(x_uint8: torch.Tensor) -> torch.Tensor:
    return x_uint8.view(torch.float8_e4m3fn).to(torch.float32)


def fp4_quantize(values: torch.Tensor) -> torch.Tensor:
    """Round each value to the nearest FP4 E2M1 level. Returns indices 0..15."""
    # values: any-shape FP32 tensor, already pre-scaled into FP4 range
    flat = values.flatten()
    levels = FP4_LEVELS.to(values.device)
    # nearest level per element
    diff = (flat.unsqueeze(-1) - levels.unsqueeze(0)).abs()
    idx = diff.argmin(dim=-1)
    return idx.reshape(values.shape).to(torch.uint8)


def pack_fp4_pairs(codes: torch.Tensor) -> torch.Tensor:
    """Pack low-nibble FP4 codes (shape (..., 2K)) into bytes (..., K).

    Convention: lower nibble = even index, upper nibble = odd index.
    """
    assert codes.shape[-1] % 2 == 0
    fp4_codes = FP4_CODES.to(codes.device)
    nibbles = fp4_codes[codes.long()]
    lo = nibbles[..., 0::2]
    hi = nibbles[..., 1::2]
    return (lo | (hi << 4)).to(torch.uint8)


def quantize_to_nvfp4(x: torch.Tensor, block: int = 16):
    """Block-quantize an FP32 tensor along its last dim to NVFP4.

    Returns (packed_bytes, block_scale_e4m3, global_scale_fp32).
    - packed_bytes: (..., last_dim/2) uint8, 2 FP4 vals per byte
    - block_scale_e4m3: (..., last_dim/block) uint8, FP8 E4M3 bit pattern
    - global_scale_fp32: scalar tensor
    """
    assert x.shape[-1] % block == 0
    n_blocks_last = x.shape[-1] // block
    shaped = x.reshape(*x.shape[:-1], n_blocks_last, block)

    # Per-block absmax (FP32)
    blk_amax = shaped.abs().amax(dim=-1)  # (..., n_blocks_last)

    # Per-block scale s_b such that x_b / s_b is in [-FP4_MAX, FP4_MAX].
    # We want s_b stored as FP8 E4M3 (max value FP8_E4M3_MAX).
    # Define s_b = blk_amax / FP4_MAX (then we still need to compress s_b itself).
    # Use a global scale G so that (s_b / G) fits in FP8 E4M3 range.
    # Then effective per-element scale is G * fp8(s_b/G).
    raw_block_scale = blk_amax / FP4_MAX
    raw_block_scale = raw_block_scale.clamp_min(1e-30)
    G = raw_block_scale.max() / FP8_E4M3_MAX
    G = G.clamp_min(1e-30)
    blk_scale_norm = (raw_block_scale / G).clamp_min(1e-30)
    blk_scale_e4m3_bytes = fp8_e4m3_encode(blk_scale_norm.to(torch.float32))

    # Decode the FP8-roundtripped scale to actually quantize x with it
    blk_scale_actual = fp8_e4m3_decode(blk_scale_e4m3_bytes) * G  # (..., n_blocks)
    # Broadcast back to (..., n_blocks, block)
    x_normed = shaped / blk_scale_actual.unsqueeze(-1).clamp_min(1e-30)
    fp4_idx = fp4_quantize(x_normed)
    fp4_idx_flat = fp4_idx.reshape(*x.shape[:-1], x.shape[-1])
    packed = pack_fp4_pairs(fp4_idx_flat)

    return packed, blk_scale_e4m3_bytes, G.to(torch.float32)


# --- cublasLt NVFP4 GEMM call -------------------------------------------------


def cublaslt_nvfp4_matmul(
    A_bytes: torch.Tensor,
    B_bytes: torch.Tensor,
    scale_A: torch.Tensor,
    scale_B: torch.Tensor,
    alpha: float,
    M: int,
    N: int,
    K: int,
):
    """Run D = alpha * A @ B^T via cuBLASLt NVFP4 GEMM.

    A_bytes shape: (M, K/2) uint8, FP4 packed
    B_bytes shape: (N, K/2) uint8, FP4 packed (stored transposed, K-contiguous)
    scale_A shape: (M, K/16) uint8, FP8 E4M3 per-block-16
    scale_B shape: (N, K/16) uint8, FP8 E4M3 per-block-16
    Returns: D (M, N) bf16
    """
    device = A_bytes.device
    D = torch.empty((M, N), dtype=torch.bfloat16, device=device)
    workspace_size = 32 * 1024 * 1024
    workspace = torch.empty(workspace_size, dtype=torch.uint8, device=device)

    alpha_c = ctypes.c_float(float(alpha))
    beta_c = ctypes.c_float(0.0)

    handle = ctypes.c_void_p()
    check(cublasLtCreate(ctypes.byref(handle)), "cublasLtCreate")

    def make_layout(dtype, rows, cols, ld):
        h = ctypes.c_void_p()
        check(
            cublasLtMatrixLayoutCreate(ctypes.byref(h), dtype, rows, cols, ld),
            "MatrixLayoutCreate",
        )
        order = ctypes.c_int32(CUBLASLT_ORDER_ROW)
        check(
            cublasLtMatrixLayoutSetAttribute(
                h, CUBLASLT_MATRIX_LAYOUT_ORDER, ctypes.byref(order), ctypes.sizeof(order)
            ),
            "Layout set ORDER",
        )
        return h

    Adesc = make_layout(CUDA_R_4F_E2M1, M, K, K)
    Bdesc = make_layout(CUDA_R_4F_E2M1, N, K, K)
    Cdesc = make_layout(CUDA_R_16BF, M, N, N)
    Ddesc = make_layout(CUDA_R_16BF, M, N, N)

    op_desc = ctypes.c_void_p()
    check(
        cublasLtMatmulDescCreate(ctypes.byref(op_desc), CUBLAS_COMPUTE_32F, CUDA_R_32F),
        "MatmulDescCreate",
    )

    def set_attr(attr, c_val):
        check(
            cublasLtMatmulDescSetAttribute(
                op_desc, attr, ctypes.byref(c_val), ctypes.sizeof(c_val)
            ),
            f"DescSetAttribute {attr}",
        )

    set_attr(CUBLASLT_MATMUL_DESC_TRANSA, ctypes.c_int32(CUBLAS_OP_N))
    set_attr(CUBLASLT_MATMUL_DESC_TRANSB, ctypes.c_int32(CUBLAS_OP_T))
    sm = ctypes.c_int32(CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3)
    set_attr(CUBLASLT_MATMUL_DESC_A_SCALE_MODE, sm)
    set_attr(CUBLASLT_MATMUL_DESC_B_SCALE_MODE, sm)
    set_attr(CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, ctypes.c_void_p(scale_A.data_ptr()))
    set_attr(CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, ctypes.c_void_p(scale_B.data_ptr()))

    pref = ctypes.c_void_p()
    check(cublasLtMatmulPreferenceCreate(ctypes.byref(pref)), "PrefCreate")
    ws = ctypes.c_size_t(workspace_size)
    check(
        cublasLtMatmulPreferenceSetAttribute(
            pref,
            CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
            ctypes.byref(ws),
            ctypes.sizeof(ws),
        ),
        "Pref set workspace",
    )

    results = (HeuristicResult * 8)()
    returned = ctypes.c_int(0)
    status = cublasLtMatmulAlgoGetHeuristic(
        handle, op_desc, Adesc, Bdesc, Cdesc, Ddesc, pref, 8, results, ctypes.byref(returned)
    )
    if status != 0 or returned.value == 0:
        raise RuntimeError(f"heuristic failed status={status} returned={returned.value}")
    chosen = next((i for i in range(returned.value) if results[i].state == 0), None)
    if chosen is None:
        raise RuntimeError("no algo reported SUCCESS")
    algo = results[chosen].algo
    algo_ptr = ctypes.cast(ctypes.pointer(algo), ctypes.c_void_p)
    stream = torch.cuda.current_stream().cuda_stream

    st = cublasLtMatmul(
        handle, op_desc, ctypes.byref(alpha_c),
        ctypes.c_void_p(A_bytes.data_ptr()), Adesc,
        ctypes.c_void_p(B_bytes.data_ptr()), Bdesc,
        ctypes.byref(beta_c),
        ctypes.c_void_p(D.data_ptr()), Cdesc,
        ctypes.c_void_p(D.data_ptr()), Ddesc,
        algo_ptr,
        ctypes.c_void_p(workspace.data_ptr()), workspace_size,
        ctypes.c_void_p(stream),
    )
    if st != CUBLAS_STATUS_SUCCESS:
        raise RuntimeError(f"matmul failed status={st}")
    torch.cuda.synchronize()

    cublasLtMatmulPreferenceDestroy(pref)
    cublasLtMatmulDescDestroy(op_desc)
    for h in (Adesc, Bdesc, Cdesc, Ddesc):
        cublasLtMatrixLayoutDestroy(h)
    cublasLtDestroy(handle)
    return D


def relative_l2(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.float()
    b = b.float()
    return ((a - b).norm() / b.norm().clamp_min(1e-30)).item()


def cosine_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.float().flatten()
    b = b.float().flatten()
    return (a @ b / (a.norm() * b.norm()).clamp_min(1e-30)).item()


def run_case(M, N, K, seed=0):
    print(f"\n=== M={M} N={N} K={K} ===")
    torch.manual_seed(seed)
    device = torch.device("cuda")
    # Realistic DiT-like scaling: activations ~ N(0, 1), weights ~ N(0, 1/sqrt(K))
    x = torch.randn((M, K), dtype=torch.float32, device=device)
    w = torch.randn((N, K), dtype=torch.float32, device=device) / (K ** 0.5)

    # Reference bf16 D = x @ w^T (logically equivalent to Linear)
    ref = (x.to(torch.bfloat16) @ w.to(torch.bfloat16).T).float()

    # Quantize both to NVFP4 (block-16 along K dim)
    x_pack, x_blk_scale, x_global = quantize_to_nvfp4(x, block=16)
    w_pack, w_blk_scale, w_global = quantize_to_nvfp4(w, block=16)
    print(
        f"  global scale A={x_global.item():.6f} B={w_global.item():.6f}"
    )

    # cuBLAS expects per-block scales in the swizzled "128 x 4 tile -> 32 x 16"
    # SF layout (NVIDIA cuBLAS docs: "block scaling factors layout").
    from torchao.prototype.mx_formats.utils import to_blocked
    x_blk_swz = to_blocked(x_blk_scale).contiguous()
    w_blk_swz = to_blocked(w_blk_scale).contiguous()

    # cuBLASLt only consumes per-block FP8 scales; fold per-tensor global
    # scales into alpha.
    alpha = float(x_global) * float(w_global)
    D = cublaslt_nvfp4_matmul(
        x_pack, w_pack, x_blk_swz, w_blk_swz,
        alpha=alpha, M=M, N=N, K=K,
    )
    cos = cosine_sim(D, ref)
    rel = relative_l2(D, ref)
    print(f"  cos sim vs bf16: {cos:.6f}")
    print(f"  rel L2:          {rel:.6f}")

    # Timing (using arbitrary FP4 bits and unit block scales is fine: same TC ops)
    iters = 50
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    # Pre-call once
    cublaslt_nvfp4_matmul(x_pack, w_pack, x_blk_swz, w_blk_swz, alpha, M, N, K)
    torch.cuda.synchronize()
    start.record()
    for _ in range(iters):
        cublaslt_nvfp4_matmul(x_pack, w_pack, x_blk_swz, w_blk_swz, alpha, M, N, K)
    end.record()
    torch.cuda.synchronize()
    per_ms = start.elapsed_time(end) / iters
    tflops = (2.0 * M * N * K) / (per_ms * 1e-3) / 1e12
    print(
        f"  {per_ms:.3f} ms/iter (incl. python overhead per call)  ~{tflops:.0f} TFLOPS"
    )
    # Note: cublasLtMatmul-only timing is in nvfp4_gemm.py; this number is e2e
    # with descriptor recreation each iter and is intentionally pessimistic.
    return cos, rel


if __name__ == "__main__":
    print("[env] torch", torch.__version__)
    print("[env] device", torch.cuda.get_device_name(0))

    # Production-relevant Linear shapes from XL turbo DiT
    # M = batch * seq (e.g. 4 * 1500 = 6000 ~= 6144 padded)
    # K, N from typical hidden dims: 2560, 3072, 4096
    shapes = [
        (6144, 3072, 3072),   # benchmark reference shape
        (6000, 2560, 2560),   # typical self-attn Linear
        (6000, 4096, 2560),   # MLP up proj
        (6000, 2560, 4096),   # MLP down proj
    ]
    for shp in shapes:
        run_case(*shp)
