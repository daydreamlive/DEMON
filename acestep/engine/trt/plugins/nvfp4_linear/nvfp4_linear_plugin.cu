// NVFP4 Linear plugin for TensorRT 10.16+ (RTX 5090 / Blackwell SM_120).
// Implements y = x @ W^T with both operands in NVFP4 (FP4 E2M1 + per-block-16
// FP8 E4M3 scales + per-tensor FP32 global scale). Uses cuBLASLt for the
// matmul kernel.
//
// Build-time plugin attributes:
//   K, N (int32)        - matmul dims (K = inner, N = out)
//   weight_fp4 (bytes)  - packed FP4 weights, shape (N, K/2) flattened
//   weight_scale (bytes)- CUTLASS-swizzled FP8 E4M3 per-block-16 scales for W
//   weight_global_scale (float) - per-tensor FP32 scale for W
//
// Inputs (1):  bf16, (M, K), M dynamic
// Outputs (1): bf16, (M, N)
//
// See benchmarks-pr17/cublaslt_nvfp4_spike/nvfp4_e2e.py for the Python
// reference and benchmarks-pr17/cublaslt_nvfp4_spike/RESULT.md for the
// throughput / correctness verification this implementation is based on.

#include <cuda_bf16.h>
#include <cuda_fp4.h>
#include <cuda_fp8.h>
#include <cublasLt.h>
#include <cuda_runtime.h>

#include <NvInferRuntime.h>

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

using namespace nvinfer1;

namespace nvfp4_plugin {

constexpr float kFP4Max = 6.0f;
constexpr float kFP8E4M3Max = 448.0f;
constexpr float kGlobalScaleDivisor = kFP4Max * kFP8E4M3Max;  // 2688

inline void cudaCheck(cudaError_t e, char const* where) {
    if (e != cudaSuccess) {
        fprintf(stderr, "[NVFP4Linear] CUDA error at %s: %s\n", where, cudaGetErrorString(e));
    }
}
inline void cublasLtCheck(cublasStatus_t s, char const* where) {
    if (s != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "[NVFP4Linear] cuBLASLt error at %s: status=%d\n", where, (int)s);
    }
}

// ---------------------------------------------------------------------------
// CUDA kernels
//
// Activation quantization uses a STATIC cal-baked global scale (passed as a
// kernel arg by host). This eliminates the per-tensor max reduction pass
// (~30 us/call) that an earlier version of this plugin did at runtime.
// The static scale is derived from cal2 `activation_absmax.json` per-Linear.
// ---------------------------------------------------------------------------

// Phase 2: per-block FP4 quant + swizzled FP8 scale write.
// Layout: thread y in 0..127 = intra-row-block, thread x in 0..3 = intra-col-block.
// Block (gridDim.y, gridDim.x) = (n_row_blocks, n_col_blocks).
// Each thread quantizes 16 K-elements for one (m, k_block16).
__device__ __forceinline__ uint8_t quantize_to_fp4_code(float xn) {
    // Round xn to nearest FP4 E2M1 level: positive levels [0,.5,1,1.5,2,3,4,6].
    // FP4 encoding: bit3 = sign, bits 0..2 = ordinal index (0..7).
    uint8_t sign = (xn < 0.0f) ? 0x8 : 0x0;
    float a = fabsf(xn);
    // Boundaries (from ModelOpt _cast_fp4): [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0]
    uint8_t ord;
    if (a < 0.25f) ord = 0;
    else if (a < 0.75f) ord = 1;
    else if (a < 1.25f) ord = 2;
    else if (a < 1.75f) ord = 3;
    else if (a < 2.5f)  ord = 4;
    else if (a < 3.5f)  ord = 5;
    else if (a < 5.0f)  ord = 6;
    else                ord = 7;
    // Round-to-nearest-even adjustment at odd boundaries (0.75, 1.75, 2.5)
    if (a == 0.75f || a == 1.75f || a == 2.5f) ord += 1;
    return sign | ord;
}

// Vectorized variant: takes G as a scalar kernel arg (no device pointer load),
// uses uint4 reinterpretation to read 16 bf16 elements in a single 32-byte load,
// then does the quant + pack + swizzle.
__global__ void quantize_act_kernel(
    const __nv_bfloat16* __restrict__ x,        // (M, K)
    int M, int K,
    float G,                                     // baked global scale (host-passed)
    uint8_t* __restrict__ packed_fp4,            // (M, K/2)
    uint8_t* __restrict__ swz_scales,            // (32 * rb * cb, 16) flat
    int n_row_blocks, int n_col_blocks)
{
    int m = blockIdx.y * 128 + threadIdx.y;
    int k_blk = blockIdx.x * 4 + threadIdx.x;
    if (m >= M || k_blk >= (K / 16)) return;

    int k_base = k_blk * 16;
    int row_off_x = m * K;
    int row_off_p = m * (K / 2);

    // Single 32-byte (uint4 = 16 bytes? actually uint4 is 16 bytes -> 8 bf16)
    // bf16 is 2 bytes; we need 16 elements = 32 bytes = 2x uint4.
    // Use two uint4 loads. uint4 is a 128-bit aligned load.
    const uint4* xv = reinterpret_cast<const uint4*>(x + row_off_x + k_base);
    uint4 vec0 = __ldg(xv + 0);  // 8 bf16 elements (kk 0..7)
    uint4 vec1 = __ldg(xv + 1);  // 8 bf16 elements (kk 8..15)
    __nv_bfloat16 bv[16];
    *reinterpret_cast<uint4*>(bv + 0) = vec0;
    *reinterpret_cast<uint4*>(bv + 8) = vec1;

    // Per-block absmax
    float blk_max = 0.0f;
    float fv[16];
    #pragma unroll
    for (int kk = 0; kk < 16; ++kk) {
        fv[kk] = __bfloat162float(bv[kk]);
        float a = fabsf(fv[kk]);
        blk_max = fmaxf(blk_max, a);
    }

    // stored_scale_norm = (blk_max / FP4_MAX) / G
    float scale_norm = (blk_max * (1.0f / kFP4Max)) / G;
    if (scale_norm > kFP8E4M3Max) scale_norm = kFP8E4M3Max;
    if (scale_norm < 0.0f) scale_norm = 0.0f;
    __nv_fp8_e4m3 fp8_scale(scale_norm);
    uint8_t e4m3_bits = fp8_scale.__x;

    // Reconstruct actual per-element scale.
    __nv_fp8_e4m3 fp8_decode;
    fp8_decode.__x = e4m3_bits;
    float actual_scale = static_cast<float>(fp8_decode) * G;
    if (actual_scale < 1e-30f) actual_scale = 1.0f;
    float inv_scale = __frcp_rn(actual_scale);

    // Quantize and pack 16 elements into 8 bytes
    #pragma unroll
    for (int kk = 0; kk < 8; ++kk) {
        uint8_t lo = quantize_to_fp4_code(fv[2 * kk + 0] * inv_scale);
        uint8_t hi = quantize_to_fp4_code(fv[2 * kk + 1] * inv_scale);
        packed_fp4[row_off_p + k_base / 2 + kk] = (lo & 0xF) | ((hi & 0xF) << 4);
    }

    // CUTLASS SF swizzle. Source scale is at (m, k_blk) in row-major (M, K/16).
    // Per torchao.to_blocked:
    //   view (n_row_blocks, 128, n_col_blocks, 4).permute(0,2,1,3)
    //   -> (n_row_blocks, n_col_blocks, 128, 4) at (rb, cb, rb_intra, cb_intra)
    //   reshape (-1, 4, 32, 4).transpose(1,2).reshape(-1, 32, 16)
    //   rb_intra split as (a=rb_intra/32, b=rb_intra%32); cb_intra=c
    //   final position: tile=rb*n_cb+cb, row=b, col=a*4+c
    int rb = m / 128, rb_intra = m % 128;
    int cb = k_blk / 4, cb_intra = k_blk % 4;
    int a = rb_intra / 32;
    int b = rb_intra % 32;
    int tile_idx = rb * n_col_blocks + cb;
    int swz_offset = (tile_idx * 32 + b) * 16 + (a * 4 + cb_intra);
    swz_scales[swz_offset] = e4m3_bits;
}


// ---------------------------------------------------------------------------
// cuBLASLt NVFP4 GEMM wrapper (persistent handle + descriptors)
// ---------------------------------------------------------------------------

struct CublasLtGemmState {
    cublasLtHandle_t handle = nullptr;
    cublasLtMatmulDesc_t op_desc = nullptr;
    cublasLtMatrixLayout_t adesc = nullptr, bdesc = nullptr, cdesc = nullptr, ddesc = nullptr;
    cublasLtMatmulPreference_t pref = nullptr;
    cublasLtMatmulHeuristicResult_t algo;
    bool algo_ready = false;
    int cached_M = 0, cached_N = 0, cached_K = 0;

    void destroy() {
        if (pref) cublasLtMatmulPreferenceDestroy(pref);
        if (op_desc) cublasLtMatmulDescDestroy(op_desc);
        if (adesc) cublasLtMatrixLayoutDestroy(adesc);
        if (bdesc) cublasLtMatrixLayoutDestroy(bdesc);
        if (cdesc) cublasLtMatrixLayoutDestroy(cdesc);
        if (ddesc) cublasLtMatrixLayoutDestroy(ddesc);
        if (handle) cublasLtDestroy(handle);
        pref = nullptr; op_desc = nullptr; adesc = bdesc = cdesc = ddesc = nullptr; handle = nullptr;
        algo_ready = false;
    }

    bool prepare(int M, int N, int K, void const* swz_scale_A_dev, void const* swz_scale_B_dev, size_t workspace_bytes) {
        if (handle == nullptr) {
            cublasLtCheck(cublasLtCreate(&handle), "cublasLtCreate");
        }
        // Recreate layouts if shape changed
        if (M != cached_M || N != cached_N || K != cached_K) {
            if (adesc) cublasLtMatrixLayoutDestroy(adesc);
            if (bdesc) cublasLtMatrixLayoutDestroy(bdesc);
            if (cdesc) cublasLtMatrixLayoutDestroy(cdesc);
            if (ddesc) cublasLtMatrixLayoutDestroy(ddesc);
            cublasLtCheck(cublasLtMatrixLayoutCreate(&adesc, CUDA_R_4F_E2M1, M, K, K), "MLC A");
            cublasLtCheck(cublasLtMatrixLayoutCreate(&bdesc, CUDA_R_4F_E2M1, N, K, K), "MLC B");
            cublasLtCheck(cublasLtMatrixLayoutCreate(&cdesc, CUDA_R_16BF, M, N, N), "MLC C");
            cublasLtCheck(cublasLtMatrixLayoutCreate(&ddesc, CUDA_R_16BF, M, N, N), "MLC D");
            int32_t order_row = CUBLASLT_ORDER_ROW;
            cublasLtCheck(cublasLtMatrixLayoutSetAttribute(adesc, CUBLASLT_MATRIX_LAYOUT_ORDER, &order_row, sizeof(order_row)), "A ORDER");
            cublasLtCheck(cublasLtMatrixLayoutSetAttribute(bdesc, CUBLASLT_MATRIX_LAYOUT_ORDER, &order_row, sizeof(order_row)), "B ORDER");
            cublasLtCheck(cublasLtMatrixLayoutSetAttribute(cdesc, CUBLASLT_MATRIX_LAYOUT_ORDER, &order_row, sizeof(order_row)), "C ORDER");
            cublasLtCheck(cublasLtMatrixLayoutSetAttribute(ddesc, CUBLASLT_MATRIX_LAYOUT_ORDER, &order_row, sizeof(order_row)), "D ORDER");
            algo_ready = false;
            cached_M = M; cached_N = N; cached_K = K;
        }
        if (op_desc == nullptr) {
            cublasLtCheck(cublasLtMatmulDescCreate(&op_desc, CUBLAS_COMPUTE_32F, CUDA_R_32F), "MDC");
            int32_t transN = CUBLAS_OP_N;
            int32_t transT = CUBLAS_OP_T;
            int32_t scale_mode = CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3;
            cublasLtCheck(cublasLtMatmulDescSetAttribute(op_desc, CUBLASLT_MATMUL_DESC_TRANSA, &transN, sizeof(transN)), "TA");
            cublasLtCheck(cublasLtMatmulDescSetAttribute(op_desc, CUBLASLT_MATMUL_DESC_TRANSB, &transT, sizeof(transT)), "TB");
            cublasLtCheck(cublasLtMatmulDescSetAttribute(op_desc, CUBLASLT_MATMUL_DESC_A_SCALE_MODE, &scale_mode, sizeof(scale_mode)), "AsmA");
            cublasLtCheck(cublasLtMatmulDescSetAttribute(op_desc, CUBLASLT_MATMUL_DESC_B_SCALE_MODE, &scale_mode, sizeof(scale_mode)), "AsmB");
            // alpha/beta are HOST scalars (mAlpha, mBeta) - no per-call sync needed
            // since the static cal-baked global scale removes the runtime reduction.
        }
        // Always set the scale pointers (they live in workspace and can change)
        cublasLtCheck(cublasLtMatmulDescSetAttribute(op_desc, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &swz_scale_A_dev, sizeof(void*)), "ApA");
        cublasLtCheck(cublasLtMatmulDescSetAttribute(op_desc, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &swz_scale_B_dev, sizeof(void*)), "ApB");
        if (pref == nullptr) {
            cublasLtCheck(cublasLtMatmulPreferenceCreate(&pref), "PrefCreate");
        }
        cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                            &workspace_bytes, sizeof(workspace_bytes));

        if (!algo_ready) {
            int returned = 0;
            cublasLtMatmulHeuristicResult_t results[8];
            std::memset(results, 0, sizeof(results));
            cublasStatus_t st = cublasLtMatmulAlgoGetHeuristic(
                handle, op_desc, adesc, bdesc, cdesc, ddesc, pref, 8, results, &returned);
            if (st != CUBLAS_STATUS_SUCCESS || returned == 0) {
                fprintf(stderr, "[NVFP4Linear] heuristic failed: status=%d returned=%d\n", (int)st, returned);
                return false;
            }
            for (int i = 0; i < returned; ++i) {
                if (results[i].state == CUBLAS_STATUS_SUCCESS) { algo = results[i]; algo_ready = true; break; }
            }
            if (!algo_ready) {
                fprintf(stderr, "[NVFP4Linear] no algo SUCCESS\n");
                return false;
            }
        }
        return true;
    }

    cublasStatus_t launch(
        float const* alpha, float const* beta,
        void const* A, void const* B, void* D,
        void* workspace, size_t workspace_bytes, cudaStream_t stream)
    {
        return cublasLtMatmul(
            handle, op_desc,
            alpha, A, adesc, B, bdesc, beta, D, cdesc, D, ddesc,
            &algo.algo, workspace, workspace_bytes, stream);
    }
};

// ---------------------------------------------------------------------------
// Plugin class
// ---------------------------------------------------------------------------

constexpr char const* kPluginName = "NVFP4Linear";
constexpr char const* kPluginVersion = "1";
constexpr char const* kPluginNamespace = "";

class NVFP4LinearPlugin : public IPluginV3,
                          public IPluginV3OneCore,
                          public IPluginV3OneBuild,
                          public IPluginV3OneRuntime
{
public:
    // Host-side state (immutable after construction)
    int mK = 0, mN = 0;
    float mWeightGlobalScale = 0.0f;
    float mActGlobalScale = 0.0f;   // baked from cal2 per-Linear absmax
    float mAlpha = 0.0f;            // mAlpha = mActGlobalScale * mWeightGlobalScale (host scalar)
    float mBeta = 0.0f;             // always 0
    // Weight FP4 and swizzled FP8 scales come in as INPUT TENSORS (TRT
    // hands their device pointers in `inputs[1]` and `inputs[2]`). They
    // are stored as uint8 ONNX initializers in the patched graph.

    // Per-runtime state (allocated in attachToContext, freed in destructor)
    CublasLtGemmState mGemm;

    // Field collection for serialization
    std::vector<PluginField> mSerFields;
    PluginFieldCollection mSerFc{};

    NVFP4LinearPlugin(int K, int N,
                      float weight_global_scale,
                      float act_global_scale)
        : mK(K), mN(N), mWeightGlobalScale(weight_global_scale),
          mActGlobalScale(act_global_scale),
          mAlpha(act_global_scale * weight_global_scale),
          mBeta(0.0f)
    {
    }

    ~NVFP4LinearPlugin() override {
        mGemm.destroy();
    }

    // === IPluginV3 ===
    IPluginCapability* getCapabilityInterface(PluginCapabilityType type) noexcept override {
        switch (type) {
            case PluginCapabilityType::kCORE:    return static_cast<IPluginV3OneCore*>(this);
            case PluginCapabilityType::kBUILD:   return static_cast<IPluginV3OneBuild*>(this);
            case PluginCapabilityType::kRUNTIME: return static_cast<IPluginV3OneRuntime*>(this);
        }
        return nullptr;
    }
    IPluginV3* clone() noexcept override {
        try {
            return new NVFP4LinearPlugin(mK, mN, mWeightGlobalScale, mActGlobalScale);
        } catch (...) { return nullptr; }
    }

    // === IPluginV3OneCore ===
    AsciiChar const* getPluginName() const noexcept override { return kPluginName; }
    AsciiChar const* getPluginVersion() const noexcept override { return kPluginVersion; }
    AsciiChar const* getPluginNamespace() const noexcept override { return kPluginNamespace; }

    // === IPluginV3OneBuild ===
    int32_t configurePlugin(DynamicPluginTensorDesc const* in, int32_t nbInputs,
                            DynamicPluginTensorDesc const* out, int32_t nbOutputs) noexcept override {
        (void)in; (void)out; (void)nbInputs; (void)nbOutputs;
        return 0;
    }
    int32_t getOutputDataTypes(DataType* outputTypes, int32_t nbOutputs,
                               const DataType* inputTypes, int32_t nbInputs) const noexcept override {
        (void)inputTypes; (void)nbInputs;
        if (nbOutputs != 1) return -1;
        outputTypes[0] = DataType::kBF16;
        return 0;
    }
    int32_t getOutputShapes(DimsExprs const* inputs, int32_t nbInputs,
                            DimsExprs const* shapeInputs, int32_t nbShapeInputs,
                            DimsExprs* outputs, int32_t nbOutputs,
                            IExprBuilder& exprBuilder) noexcept override {
        (void)shapeInputs; (void)nbShapeInputs;
        if (nbInputs != 3 || nbOutputs != 1) return -1;
        // Input[0] is bf16 activation (..., K). Output: (..., N).
        DimsExprs const& a = inputs[0];
        outputs[0].nbDims = a.nbDims;
        for (int i = 0; i < a.nbDims - 1; ++i) outputs[0].d[i] = a.d[i];
        outputs[0].d[a.nbDims - 1] = exprBuilder.constant(mN);
        return 0;
    }
    bool supportsFormatCombination(int32_t pos, DynamicPluginTensorDesc const* inOut,
                                    int32_t nbInputs, int32_t nbOutputs) noexcept override {
        (void)nbInputs; (void)nbOutputs;
        // pos 0 = activation (bf16 LINEAR)
        // pos 1 = weight FP4 bytes (int8 LINEAR, raw bytes)
        // pos 2 = weight swizzled scale bytes (int8 LINEAR, raw bytes)
        // pos 3 = output (bf16 LINEAR)
        if (inOut[pos].desc.format != TensorFormat::kLINEAR) return false;
        if (pos == 0 || pos == 3) return inOut[pos].desc.type == DataType::kBF16;
        return inOut[pos].desc.type == DataType::kINT8;
    }
    int32_t getNbOutputs() const noexcept override { return 1; }
    size_t getWorkspaceSize(DynamicPluginTensorDesc const* inputs, int32_t nbInputs,
                            DynamicPluginTensorDesc const* outputs, int32_t nbOutputs) const noexcept override {
        (void)nbInputs; (void)nbOutputs; (void)outputs;
        // Use max M (kMAX of input profile)
        Dims const& maxD = inputs[0].max;
        int64_t M = 1;
        for (int i = 0; i < maxD.nbDims - 1; ++i) M *= maxD.d[i];
        int K = (int)maxD.d[maxD.nbDims - 1];

        if (K != mK) K = mK;  // fallback if max dim is unknown
        if (M <= 0 || M > (1 << 22)) M = 16384;

        // packed FP4 activation: M*K/2 bytes
        size_t packed = (size_t)M * (K / 2);
        // activation block scales swizzled: 32*ceil(M/128) x 16*ceil(K/64) bytes
        size_t rb = (size_t)((M + 127) / 128);
        size_t cb = (size_t)((K + 63) / 64);
        size_t swz_scales = 32 * rb * 16 * cb;
        // cuBLASLt workspace
        size_t lt_ws = 32ull * 1024 * 1024;
        // Add 256B alignment pad
        return packed + swz_scales + lt_ws + 4096;
    }

    // === IPluginV3OneRuntime ===
    int32_t setTactic(int32_t /*tactic*/) noexcept override { return 0; }
    int32_t onShapeChange(PluginTensorDesc const* /*in*/, int32_t /*nbIn*/,
                          PluginTensorDesc const* /*out*/, int32_t /*nbOut*/) noexcept override {
        // Force algo re-pick on shape change
        mGemm.algo_ready = false;
        return 0;
    }

    int32_t enqueue(PluginTensorDesc const* inputDesc, PluginTensorDesc const* /*outputDesc*/,
                    void const* const* inputs, void* const* outputs,
                    void* workspace, cudaStream_t stream) noexcept override
    {
        // Compute M from input dims
        Dims const& inD = inputDesc[0].dims;
        int64_t M = 1;
        for (int i = 0; i < inD.nbDims - 1; ++i) M *= inD.d[i];
        int K = (int)inD.d[inD.nbDims - 1];
        if (K != mK) {
            fprintf(stderr, "[NVFP4Linear] K mismatch: input K=%d plugin K=%d\n", K, mK);
            return -1;
        }

        // Carve up workspace
        uint8_t* ws = (uint8_t*)workspace;
        size_t off = 0;
        uint8_t* packed_act = ws + off; off += (size_t)M * (K / 2);
        size_t rb = (size_t)((M + 127) / 128);
        size_t cb = (size_t)((K + 63) / 64);
        uint8_t* swz_act_scales = ws + off; off += 32 * rb * 16 * cb;
        // Align to 256B for cuBLASLt
        off = (off + 255) & ~255ULL;
        uint8_t* lt_ws = ws + off;
        size_t lt_ws_bytes = 32ull * 1024 * 1024;

        const __nv_bfloat16* xb = (const __nv_bfloat16*)inputs[0];
        // inputs[1] = packed FP4 weight (uint8, (N, K/2) flat)
        // inputs[2] = swizzled FP8 weight scale (uint8 flat)
        const uint8_t* w_fp4_dev = (const uint8_t*)inputs[1];
        const uint8_t* w_scale_dev = (const uint8_t*)inputs[2];
        __nv_bfloat16* yb = (__nv_bfloat16*)outputs[0];

        // Single-pass per-block FP4 quant + swizzle (static global scale, no reduction)
        dim3 grid((int)cb, (int)rb);
        dim3 block(4, 128);
        quantize_act_kernel<<<grid, block, 0, stream>>>(
            xb, (int)M, K, mActGlobalScale, packed_act, swz_act_scales, (int)rb, (int)cb);


        // cuBLASLt NVFP4 GEMM
        if (!mGemm.prepare((int)M, mN, K, swz_act_scales, w_scale_dev, lt_ws_bytes)) {
            return -1;
        }
        // Run cuBLASLt NVFP4 GEMM with HOST alpha/beta (static, no per-call sync).
        // Direct BF16 output requires the network output binding to be BF16
        // (set via out_tensor.dtype = trt.DataType.BF16 in the network definition).
        cublasStatus_t st = mGemm.launch(
            &mAlpha, &mBeta,
            packed_act, w_fp4_dev, yb,
            lt_ws, lt_ws_bytes, stream);
        if (st != CUBLAS_STATUS_SUCCESS) {
            fprintf(stderr, "[NVFP4Linear] cublasLtMatmul failed: status=%d\n", (int)st);
            return -1;
        }
        return 0;
    }

    IPluginV3* attachToContext(IPluginResourceContext* /*context*/) noexcept override {
        // Weights now flow in via inputs[] (as ONNX initializers), so we
        // don't allocate or manage device memory here. Just clone.
        return clone();
    }

    PluginFieldCollection const* getFieldsToSerialize() noexcept override {
        mSerFields.clear();
        mSerFields.emplace_back("K", &mK, PluginFieldType::kINT32, 1);
        mSerFields.emplace_back("N", &mN, PluginFieldType::kINT32, 1);
        mSerFields.emplace_back("weight_global_scale", &mWeightGlobalScale, PluginFieldType::kFLOAT32, 1);
        mSerFields.emplace_back("act_global_scale", &mActGlobalScale, PluginFieldType::kFLOAT32, 1);
        mSerFc.nbFields = (int32_t)mSerFields.size();
        mSerFc.fields = mSerFields.data();
        return &mSerFc;
    }
};

// ---------------------------------------------------------------------------
// Plugin creator
// ---------------------------------------------------------------------------

class NVFP4LinearPluginCreator : public IPluginCreatorV3One {
public:
    NVFP4LinearPluginCreator() {
        mFieldNames.emplace_back("K", nullptr, PluginFieldType::kINT32, 1);
        mFieldNames.emplace_back("N", nullptr, PluginFieldType::kINT32, 1);
        mFieldNames.emplace_back("weight_global_scale", nullptr, PluginFieldType::kFLOAT32, 1);
        mFieldNames.emplace_back("act_global_scale", nullptr, PluginFieldType::kFLOAT32, 1);
        mFc.nbFields = (int32_t)mFieldNames.size();
        mFc.fields = mFieldNames.data();
    }
    AsciiChar const* getPluginName() const noexcept override { return kPluginName; }
    AsciiChar const* getPluginVersion() const noexcept override { return kPluginVersion; }
    AsciiChar const* getPluginNamespace() const noexcept override { return kPluginNamespace; }
    PluginFieldCollection const* getFieldNames() noexcept override { return &mFc; }

    IPluginV3* createPlugin(AsciiChar const* /*name*/, PluginFieldCollection const* fc,
                            TensorRTPhase /*phase*/) noexcept override {
        int K = 0, N = 0;
        float w_g = 0.0f, a_g = 0.0f;
        for (int i = 0; i < fc->nbFields; ++i) {
            PluginField const& f = fc->fields[i];
            if (std::strcmp(f.name, "K") == 0) K = *static_cast<int const*>(f.data);
            else if (std::strcmp(f.name, "N") == 0) N = *static_cast<int const*>(f.data);
            else if (std::strcmp(f.name, "weight_global_scale") == 0) w_g = *static_cast<float const*>(f.data);
            else if (std::strcmp(f.name, "act_global_scale") == 0) a_g = *static_cast<float const*>(f.data);
        }
        if (K <= 0 || N <= 0 || a_g <= 0.0f) {
            fprintf(stderr, "[NVFP4Linear] createPlugin: bad K=%d N=%d a_g=%g\n", K, N, a_g);
            return nullptr;
        }
        try {
            return new NVFP4LinearPlugin(K, N, w_g, a_g);
        } catch (...) { return nullptr; }
    }
private:
    std::vector<PluginField> mFieldNames;
    PluginFieldCollection mFc{};
};

} // namespace nvfp4_plugin

using NVFP4LinearPluginCreator = nvfp4_plugin::NVFP4LinearPluginCreator;
REGISTER_TENSORRT_PLUGIN(NVFP4LinearPluginCreator);
