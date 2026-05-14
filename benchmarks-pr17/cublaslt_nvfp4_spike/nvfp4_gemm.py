"""cuBLASLt NVFP4 GEMM verification spike.

Calls cublasLtMatmul directly via ctypes with NVFP4 (FP4 E2M1 packed, FP8 E4M3
per-block 16 scale) data types to measure achievable throughput on RTX 5090
(Blackwell SM_120).

Reference shape 6144 x 3072 x 3072 matches benchmarks-pr17/fp8_stub.py which
hit 414 TFLOPS in FP8. Target: >= 500 TFLOPS gates the TRT plugin path.

Bench is timed with torch CUDA events; memory is allocated as torch tensors and
.data_ptr() is handed to cuBLASLt. We use CUDA 12.8 cublasLt bundled with
torch (cublasLt64_12.dll, lib version 120804).
"""

import ctypes
import os
import sys
import time

import torch

# --- Locate and load cublasLt -------------------------------------------------

TORCH_LIB = os.path.join(os.path.dirname(torch.__file__), "lib")
os.add_dll_directory(TORCH_LIB)
# Touch CUDA so the runtime is initialized before we call into cublasLt
torch.cuda.init()
_ = torch.empty(1, device="cuda")
torch.cuda.synchronize()

LT = ctypes.CDLL(os.path.join(TORCH_LIB, "cublasLt64_12.dll"))

# --- Enum values from CUDA 12.8 headers --------------------------------------

# cudaDataType_t (library_types.h)
CUDA_R_32F = 0
CUDA_R_16BF = 14
CUDA_R_8F_E4M3 = 28
CUDA_R_8F_UE4M3 = 28  # same value (unsigned alias)
CUDA_R_4F_E2M1 = 33

# cublasComputeType_t (cublas_api.h)
CUBLAS_COMPUTE_32F = 68

# cublasOperation_t (cublas_api.h)
CUBLAS_OP_N = 0
CUBLAS_OP_T = 1

# cublasLtOrder_t (cublasLt.h)
CUBLASLT_ORDER_ROW = 1

# cublasLtMatmulMatrixScale_t
CUBLASLT_MATMUL_MATRIX_SCALE_SCALAR_32F = 0
CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3 = 1

# cublasLtMatrixLayoutAttribute_t
CUBLASLT_MATRIX_LAYOUT_ORDER = 1

# cublasLtMatmulDescAttributes_t
CUBLASLT_MATMUL_DESC_TRANSA = 3
CUBLASLT_MATMUL_DESC_TRANSB = 4
CUBLASLT_MATMUL_DESC_A_SCALE_POINTER = 17
CUBLASLT_MATMUL_DESC_B_SCALE_POINTER = 18
CUBLASLT_MATMUL_DESC_A_SCALE_MODE = 31
CUBLASLT_MATMUL_DESC_B_SCALE_MODE = 32

# cublasLtMatmulPreferenceAttributes_t
CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES = 1

# cublasStatus_t  (cublas_api.h enum). 0 == success.
CUBLAS_STATUS_SUCCESS = 0
CUBLAS_STATUS_NOT_SUPPORTED = 15

# --- Function prototypes ------------------------------------------------------

def _proto(name, restype, argtypes):
    fn = getattr(LT, name)
    fn.restype = restype
    fn.argtypes = argtypes
    return fn


cublasLtCreate = _proto(
    "cublasLtCreate", ctypes.c_int, [ctypes.POINTER(ctypes.c_void_p)]
)
cublasLtDestroy = _proto("cublasLtDestroy", ctypes.c_int, [ctypes.c_void_p])

cublasLtMatrixLayoutCreate = _proto(
    "cublasLtMatrixLayoutCreate",
    ctypes.c_int,
    [
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.c_int,
        ctypes.c_uint64,
        ctypes.c_uint64,
        ctypes.c_int64,
    ],
)
cublasLtMatrixLayoutDestroy = _proto(
    "cublasLtMatrixLayoutDestroy", ctypes.c_int, [ctypes.c_void_p]
)
cublasLtMatrixLayoutSetAttribute = _proto(
    "cublasLtMatrixLayoutSetAttribute",
    ctypes.c_int,
    [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_size_t],
)

cublasLtMatmulDescCreate = _proto(
    "cublasLtMatmulDescCreate",
    ctypes.c_int,
    [ctypes.POINTER(ctypes.c_void_p), ctypes.c_int, ctypes.c_int],
)
cublasLtMatmulDescDestroy = _proto(
    "cublasLtMatmulDescDestroy", ctypes.c_int, [ctypes.c_void_p]
)
cublasLtMatmulDescSetAttribute = _proto(
    "cublasLtMatmulDescSetAttribute",
    ctypes.c_int,
    [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_size_t],
)

cublasLtMatmulPreferenceCreate = _proto(
    "cublasLtMatmulPreferenceCreate",
    ctypes.c_int,
    [ctypes.POINTER(ctypes.c_void_p)],
)
cublasLtMatmulPreferenceDestroy = _proto(
    "cublasLtMatmulPreferenceDestroy", ctypes.c_int, [ctypes.c_void_p]
)
cublasLtMatmulPreferenceSetAttribute = _proto(
    "cublasLtMatmulPreferenceSetAttribute",
    ctypes.c_int,
    [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_size_t],
)


class HeuristicResult(ctypes.Structure):
    _fields_ = [
        ("algo", ctypes.c_uint64 * 8),  # cublasLtMatmulAlgo_t
        ("workspaceSize", ctypes.c_size_t),
        ("state", ctypes.c_int),  # cublasStatus_t
        ("wavesCount", ctypes.c_float),
        ("reserved", ctypes.c_int * 4),
    ]


cublasLtMatmulAlgoGetHeuristic = _proto(
    "cublasLtMatmulAlgoGetHeuristic",
    ctypes.c_int,
    [
        ctypes.c_void_p,  # handle
        ctypes.c_void_p,  # operationDesc
        ctypes.c_void_p,  # Adesc
        ctypes.c_void_p,  # Bdesc
        ctypes.c_void_p,  # Cdesc
        ctypes.c_void_p,  # Ddesc
        ctypes.c_void_p,  # preference
        ctypes.c_int,  # requestedAlgoCount
        ctypes.POINTER(HeuristicResult),
        ctypes.POINTER(ctypes.c_int),
    ],
)

cublasLtMatmul = _proto(
    "cublasLtMatmul",
    ctypes.c_int,
    [
        ctypes.c_void_p,  # handle
        ctypes.c_void_p,  # computeDesc
        ctypes.c_void_p,  # alpha
        ctypes.c_void_p,  # A
        ctypes.c_void_p,  # Adesc
        ctypes.c_void_p,  # B
        ctypes.c_void_p,  # Bdesc
        ctypes.c_void_p,  # beta
        ctypes.c_void_p,  # C
        ctypes.c_void_p,  # Cdesc
        ctypes.c_void_p,  # D
        ctypes.c_void_p,  # Ddesc
        ctypes.c_void_p,  # algo (pointer to cublasLtMatmulAlgo_t, 64 bytes)
        ctypes.c_void_p,  # workspace
        ctypes.c_size_t,  # workspaceSizeInBytes
        ctypes.c_void_p,  # stream
    ],
)


def check(status, where):
    if status != CUBLAS_STATUS_SUCCESS:
        raise RuntimeError(f"{where} failed with cublasStatus={status}")


# --- Bench --------------------------------------------------------------------


def bench_nvfp4(M, N, K, iters=50, warmup=10, verbose=True, layout="TN"):
    """Run NVFP4 GEMM (D = A * B) and return TFLOPS.

    layout="TN": A is (M, K) row-major, B is (N, K) row-major + TransB=T.
    This puts K as the contiguous dim for both operands, which low-precision
    tensor core paths typically require. scale_A is (M, K/16) row-major;
    scale_B is (N, K/16) row-major.
    """
    assert K % 16 == 0, f"K={K} must be divisible by 16"
    device = torch.device("cuda")

    # Allocate packed FP4 storage as uint8 (2 FP4 vals per byte along K)
    A_bytes = torch.empty((M, K // 2), dtype=torch.uint8, device=device)
    if layout == "TN":
        # B stored as (N, K) row-major (so K is contiguous), TransB=T at use
        B_bytes = torch.empty((N, K // 2), dtype=torch.uint8, device=device)
    else:
        B_bytes = torch.empty((K, N // 2), dtype=torch.uint8, device=device)
    A_bytes.fill_(0x44)
    B_bytes.fill_(0x33)

    # FP8 E4M3 per-block scales (one scale per 16 K-elements)
    scale_A = torch.empty((M, K // 16), dtype=torch.uint8, device=device)
    if layout == "TN":
        scale_B = torch.empty((N, K // 16), dtype=torch.uint8, device=device)
    else:
        scale_B = torch.empty((K // 16, N), dtype=torch.uint8, device=device)
    # 0x38 in E4M3 = 1.0 (sign=0, exp=0111=bias 7, mantissa=000)
    scale_A.fill_(0x38)
    scale_B.fill_(0x38)

    D = torch.empty((M, N), dtype=torch.bfloat16, device=device)

    workspace_size = 32 * 1024 * 1024  # 32 MiB
    workspace = torch.empty(workspace_size, dtype=torch.uint8, device=device)

    alpha = ctypes.c_float(1.0)
    beta = ctypes.c_float(0.0)

    # --- Create handle ---
    handle = ctypes.c_void_p()
    check(cublasLtCreate(ctypes.byref(handle)), "cublasLtCreate")

    # --- Matrix layouts (row-major) ---
    def make_layout(dtype, rows, cols, ld):
        h = ctypes.c_void_p()
        check(
            cublasLtMatrixLayoutCreate(
                ctypes.byref(h), dtype, rows, cols, ld
            ),
            "MatrixLayoutCreate",
        )
        order = ctypes.c_int32(CUBLASLT_ORDER_ROW)
        check(
            cublasLtMatrixLayoutSetAttribute(
                h,
                CUBLASLT_MATRIX_LAYOUT_ORDER,
                ctypes.byref(order),
                ctypes.sizeof(order),
            ),
            "Layout set ORDER",
        )
        return h

    Adesc = make_layout(CUDA_R_4F_E2M1, M, K, K)
    if layout == "TN":
        # B logical shape (K, N), but stored as (N, K) row-major + TransB=T
        Bdesc = make_layout(CUDA_R_4F_E2M1, N, K, K)
    else:
        Bdesc = make_layout(CUDA_R_4F_E2M1, K, N, N)
    Cdesc = make_layout(CUDA_R_16BF, M, N, N)
    Ddesc = make_layout(CUDA_R_16BF, M, N, N)

    # --- Matmul desc ---
    op_desc = ctypes.c_void_p()
    check(
        cublasLtMatmulDescCreate(
            ctypes.byref(op_desc), CUBLAS_COMPUTE_32F, CUDA_R_32F
        ),
        "MatmulDescCreate",
    )

    def set_desc_attr(attr, c_val):
        check(
            cublasLtMatmulDescSetAttribute(
                op_desc, attr, ctypes.byref(c_val), ctypes.sizeof(c_val)
            ),
            f"DescSetAttribute {attr}",
        )

    transN = ctypes.c_int32(CUBLAS_OP_N)
    transT = ctypes.c_int32(CUBLAS_OP_T)
    set_desc_attr(CUBLASLT_MATMUL_DESC_TRANSA, transN)
    set_desc_attr(
        CUBLASLT_MATMUL_DESC_TRANSB, transT if layout == "TN" else transN
    )

    # Block-scale mode VEC16_UE4M3 for both A and B
    scale_mode = ctypes.c_int32(CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3)
    set_desc_attr(CUBLASLT_MATMUL_DESC_A_SCALE_MODE, scale_mode)
    set_desc_attr(CUBLASLT_MATMUL_DESC_B_SCALE_MODE, scale_mode)

    # Scale pointers (FP8 E4M3 block scale tensors)
    sa_ptr = ctypes.c_void_p(scale_A.data_ptr())
    sb_ptr = ctypes.c_void_p(scale_B.data_ptr())
    set_desc_attr(CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, sa_ptr)
    set_desc_attr(CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, sb_ptr)

    # --- Preference ---
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

    # --- Heuristic ---
    n_results = 8
    results = (HeuristicResult * n_results)()
    returned = ctypes.c_int(0)
    status = cublasLtMatmulAlgoGetHeuristic(
        handle,
        op_desc,
        Adesc,
        Bdesc,
        Cdesc,
        Ddesc,
        pref,
        n_results,
        results,
        ctypes.byref(returned),
    )
    if verbose:
        print(f"[heuristic] status={status} returned={returned.value}")
    if status != CUBLAS_STATUS_SUCCESS or returned.value == 0:
        print(
            f"[heuristic] FAIL: status={status} returned={returned.value} — "
            f"NVFP4 likely unsupported for this M/N/K/layout combination"
        )
        return None

    for i in range(returned.value):
        if verbose:
            r = results[i]
            print(
                f"  algo[{i}]: state={r.state} workspace={r.workspaceSize} "
                f"waves={r.wavesCount:.2f}"
            )

    valid_idx = [i for i in range(returned.value) if results[i].state == CUBLAS_STATUS_SUCCESS]
    if not valid_idx:
        print("[heuristic] No algo reported SUCCESS state")
        return None

    stream = torch.cuda.current_stream().cuda_stream

    def launch_with(algo_ptr):
        return cublasLtMatmul(
            handle,
            op_desc,
            ctypes.byref(alpha),
            ctypes.c_void_p(A_bytes.data_ptr()),
            Adesc,
            ctypes.c_void_p(B_bytes.data_ptr()),
            Bdesc,
            ctypes.byref(beta),
            ctypes.c_void_p(D.data_ptr()),
            Cdesc,
            ctypes.c_void_p(D.data_ptr()),
            Ddesc,
            algo_ptr,
            ctypes.c_void_p(workspace.data_ptr()),
            workspace_size,
            ctypes.c_void_p(stream),
        )

    best_tflops = 0.0
    best_idx = -1
    flops_per_iter = 2.0 * M * N * K
    for idx in valid_idx:
        algo = results[idx].algo
        algo_ptr = ctypes.cast(ctypes.pointer(algo), ctypes.c_void_p)
        st = launch_with(algo_ptr)
        if st != CUBLAS_STATUS_SUCCESS:
            if verbose:
                print(f"  algo[{idx}]: launch failed status={st}")
            continue
        torch.cuda.synchronize()
        for _ in range(warmup):
            launch_with(algo_ptr)
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            launch_with(algo_ptr)
        end.record()
        torch.cuda.synchronize()
        elapsed_ms = start.elapsed_time(end)
        per_iter_ms = elapsed_ms / iters
        tflops_i = flops_per_iter / (per_iter_ms * 1e-3) / 1e12
        if verbose:
            print(f"  algo[{idx}]: {per_iter_ms:.3f} ms/iter  {tflops_i:.1f} TFLOPS")
        if tflops_i > best_tflops:
            best_tflops = tflops_i
            best_idx = idx
    per_iter_ms = flops_per_iter / (best_tflops * 1e12) * 1e3 if best_tflops > 0 else float("nan")
    tflops = best_tflops

    # Cleanup
    cublasLtMatmulPreferenceDestroy(pref)
    cublasLtMatmulDescDestroy(op_desc)
    for h in (Adesc, Bdesc, Cdesc, Ddesc):
        cublasLtMatrixLayoutDestroy(h)
    cublasLtDestroy(handle)

    if verbose:
        print(
            f"[nvfp4] M={M} N={N} K={K}  best algo[{best_idx}]: "
            f"{per_iter_ms:.3f} ms/iter  {tflops:.1f} TFLOPS"
        )
    return tflops


if __name__ == "__main__":
    print("[env] torch", torch.__version__)
    print("[env] device", torch.cuda.get_device_name(0))
    print("[env] capability", torch.cuda.get_device_capability(0))

    # Reference shape matches benchmarks-pr17/fp8_stub.py (FP8 was 414 TFLOPS)
    tf = bench_nvfp4(6144, 3072, 3072, iters=100, warmup=20)
    if tf is None:
        sys.exit(2)

    # Try a few more shapes to characterize scaling
    print()
    for shape in [(2048, 2048, 2048), (4096, 4096, 4096), (8192, 8192, 8192)]:
        bench_nvfp4(*shape, iters=50, warmup=10)
