/*
 * gollmgo_model.cu — Model loading and forward pass with FP16/BF16 support.
 *
 * Uses cuBLAS for linear projections (GEMM) and custom kernels for
 * embedding, RMSNorm, RoPE, SiLU, and attention.
 *
 * Runtime dtype selection: GOLLMGO_DTYPE_FP16 or GOLLMGO_DTYPE_BF16.
 */

#include "gollmgo_model.h"
#include "gollmgo_ops.cuh"
#include "gollmgo_paged_attn.cuh"
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cuda_fp8.h>
#include <cstring>
#include <cstdint>
#include <cstdio>
#include <string>
#include <unordered_map>
#include <vector>

/* ---- Weight storage ---- */

struct weight_tensor {
    void*   data;
    int64_t size_bytes;
};

struct gollmgo_model {
    gollmgo_model_config_t config;
    gollmgo_backend_t      backend;
    cublasHandle_t         cublas;
    void*                  cublas_workspace;
    size_t                 cublas_workspace_size;
    bool                   ready;
    char                   last_error[512];

    /* Named weight tensors on device. */
    std::unordered_map<std::string, weight_tensor> weights;
    /* Per-weight quantization scale factors (only when quant_dtype != 0). */
    std::unordered_map<std::string, float> weight_scales;

    /* Scratch buffers.
     * buf_hidden is FP32 to prevent precision loss across layers.
     * All other activation buffers are in the model's native dtype (2 bytes). */
    float* buf_hidden;      /* [max_batch, hidden_size] — FP32 accumulator */
    void* buf_norm;         /* [max_batch, hidden_size] — half-precision for GEMM */
    void* buf_q;            /* [max_batch, num_heads * head_dim] */
    void* buf_k;            /* [max_batch, num_kv_heads * head_dim] */
    void* buf_v;            /* [max_batch, num_kv_heads * head_dim] */
    void* buf_attn_out;     /* [max_batch, num_heads * head_dim] */
    void* buf_proj;         /* [max_batch, hidden_size] */
    void* buf_gate;         /* [max_batch, intermediate_size] */
    void* buf_up;           /* [max_batch, intermediate_size] */
    void* buf_ffn;          /* [max_batch, intermediate_size] */
    void* buf_ffn_out;      /* [max_batch, hidden_size] */
    void* buf_logits_half;  /* [max_batch, vocab_size] — FP16 or BF16 */
    float*  buf_logits_f32; /* [max_batch, vocab_size] — device staging */

    /* Pre-allocated decode input buffers (for graph capture — no cudaMalloc in hot path). */
    int32_t* paged_d_token_ids;     /* [max_batch] */
    int32_t* paged_d_positions;     /* [max_batch] */
    int32_t* paged_d_slot_mapping;  /* [max_batch] */
    int32_t* paged_d_seq_lens;      /* [max_batch] */
    int32_t* paged_d_slot_tables;   /* [max_batch * max_seq_len] */

    /* CUDA graph cache: (batch_size,max_context_len) → instantiated graph. */
    std::unordered_map<uint64_t, cudaGraphExec_t> graph_cache;
    int graph_max_context_len; /* max_context_len used during graph capture */
    int64_t graph_hits;        /* graph replay count */
    int64_t graph_lookups;     /* total forward_paged calls */

    /* PagedAttention v2 scratch buffers. */
    float* buf_v2_exp_sums;     /* [max_batch, num_heads, max_num_partitions] */
    float* buf_v2_max_logits;   /* [max_batch, num_heads, max_num_partitions] */
    float* buf_v2_partial_out;  /* [max_batch, num_heads, max_num_partitions, head_dim] */
    float* buf_v2_reduce_out;   /* [max_batch, num_heads, head_dim] — FP32 reduce output */
    int    max_num_partitions;

    int max_batch;
};

static void model_set_error(gollmgo_model_t m, const char* msg) {
    snprintf(m->last_error, sizeof(m->last_error), "%s", msg);
}

static gollmgo_status_t model_check_cuda(gollmgo_model_t m, cudaError_t err) {
    if (err == cudaSuccess) return GOLLMGO_OK;
    model_set_error(m, cudaGetErrorString(err));
    return (err == cudaErrorMemoryAllocation) ? GOLLMGO_ERR_OOM : GOLLMGO_ERR_CUDA;
}

/* Helper: get weight pointer or null. */
static void* get_weight(gollmgo_model_t m, const std::string& name) {
    auto it = m->weights.find(name);
    if (it == m->weights.end()) return nullptr;
    return it->second.data;
}

static uint64_t graph_cache_key(int batch_size, int max_context_len) {
    return (static_cast<uint64_t>(static_cast<uint32_t>(batch_size)) << 32) |
           static_cast<uint32_t>(max_context_len);
}

/* ---- cuBLAS GEMM helpers ---- */
/* C = A * B^T  (row-major A[M,K], B[N,K] -> C[M,N]) */
/* cuBLAS is column-major, so we compute C^T = B * A^T. */

static gollmgo_status_t gemm_f16(gollmgo_model_t m,
                                  const void* A, const void* B, void* C,
                                  int M, int N, int K) {
    float alpha = 1.0f;
    float beta  = 0.0f;

    /* Use GemmEx with FP32 accumulation for better precision.
     * cublasHgemm accumulates in FP16 which causes significant error
     * over 22 transformer layers. */
    cublasStatus_t status = cublasGemmEx(
        m->cublas,
        CUBLAS_OP_T, CUBLAS_OP_N,
        N, M, K,
        &alpha,
        B, CUDA_R_16F, K,
        A, CUDA_R_16F, K,
        &beta,
        C, CUDA_R_16F, N,
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT
    );

    if (status != CUBLAS_STATUS_SUCCESS) {
        char msg[128];
        snprintf(msg, sizeof(msg), "cuBLAS GemmEx FP16 failed (status %d)", (int)status);
        model_set_error(m, msg);
        return GOLLMGO_ERR_INTERNAL;
    }
    return GOLLMGO_OK;
}

static gollmgo_status_t gemm_bf16(gollmgo_model_t m,
                                   const void* A, const void* B, void* C,
                                   int M, int N, int K) {
    float alpha = 1.0f;
    float beta  = 0.0f;

    cublasStatus_t status = cublasGemmEx(
        m->cublas,
        CUBLAS_OP_T, CUBLAS_OP_N,
        N, M, K,
        &alpha,
        B, CUDA_R_16BF, K,
        A, CUDA_R_16BF, K,
        &beta,
        C, CUDA_R_16BF, N,
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT
    );

    if (status != CUBLAS_STATUS_SUCCESS) {
        char msg[128];
        snprintf(msg, sizeof(msg), "cuBLAS GemmEx BF16 failed (status %d)", (int)status);
        model_set_error(m, msg);
        return GOLLMGO_ERR_INTERNAL;
    }
    return GOLLMGO_OK;
}

/* FP8 weight-only GEMM: A (activations) in BF16/FP16, B (weights) in FP8 E4M3. */
static gollmgo_status_t gemm_fp8(gollmgo_model_t m,
                                  const void* A, const void* B, void* C,
                                  int M, int N, int K) {
    float alpha = 1.0f;
    float beta  = 0.0f;
    cudaDataType_t a_type = (m->config.dtype == GOLLMGO_DTYPE_BF16) ? CUDA_R_16BF : CUDA_R_16F;
    cudaDataType_t c_type = a_type;

    cublasStatus_t status = cublasGemmEx(
        m->cublas,
        CUBLAS_OP_T, CUBLAS_OP_N,
        N, M, K,
        &alpha,
        B, CUDA_R_8F_E4M3, K,
        A, a_type, K,
        &beta,
        C, c_type, N,
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT
    );
    if (status != CUBLAS_STATUS_SUCCESS) {
        char msg[128];
        snprintf(msg, sizeof(msg), "cuBLAS GemmEx FP8 failed (status %d)", (int)status);
        model_set_error(m, msg);
        return GOLLMGO_ERR_INTERNAL;
    }
    return GOLLMGO_OK;
}

/* INT8 weight-only GEMM: A in BF16/FP16, B in INT8. */
static gollmgo_status_t gemm_int8(gollmgo_model_t m,
                                   const void* A, const void* B, void* C,
                                   int M, int N, int K) {
    float alpha = 1.0f;
    float beta  = 0.0f;
    cudaDataType_t a_type = (m->config.dtype == GOLLMGO_DTYPE_BF16) ? CUDA_R_16BF : CUDA_R_16F;
    cudaDataType_t c_type = a_type;

    cublasStatus_t status = cublasGemmEx(
        m->cublas,
        CUBLAS_OP_T, CUBLAS_OP_N,
        N, M, K,
        &alpha,
        B, CUDA_R_8I, K,
        A, a_type, K,
        &beta,
        C, c_type, N,
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT
    );
    if (status != CUBLAS_STATUS_SUCCESS) {
        char msg[128];
        snprintf(msg, sizeof(msg), "cuBLAS GemmEx INT8 failed (status %d)", (int)status);
        model_set_error(m, msg);
        return GOLLMGO_ERR_INTERNAL;
    }
    return GOLLMGO_OK;
}

/* Dispatch GEMM based on model dtype and quantization. */
static gollmgo_status_t gemm(gollmgo_model_t m,
                              const void* A, const void* B, void* C,
                              int M, int N, int K) {
    /* If weight-only quantization is active, use the quantized GEMM for weight (B). */
    if (m->config.quant_dtype == GOLLMGO_DTYPE_FP8)
        return gemm_fp8(m, A, B, C, M, N, K);
    if (m->config.quant_dtype == GOLLMGO_DTYPE_INT8)
        return gemm_int8(m, A, B, C, M, N, K);
    if (m->config.dtype == GOLLMGO_DTYPE_BF16)
        return gemm_bf16(m, A, B, C, M, N, K);
    return gemm_f16(m, A, B, C, M, N, K);
}

/* ---- Weight quantization kernels ---- */

/* Quantize BF16 weights to FP8 E4M3 on device. */
__global__ void quantize_to_fp8_kernel(const __nv_bfloat16* __restrict__ src,
                                        __nv_fp8_storage_t* __restrict__ dst,
                                        float scale_inv, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float val = __bfloat162float(src[idx]) * scale_inv;
    dst[idx] = __nv_cvt_float_to_fp8(val, __NV_SATFINITE, __NV_E4M3);
}

/* Quantize BF16 weights to INT8. */
__global__ void quantize_to_int8_kernel(const __nv_bfloat16* __restrict__ src,
                                         int8_t* __restrict__ dst,
                                         float scale_inv, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float val = __bfloat162float(src[idx]) * scale_inv;
    int ival = __float2int_rn(val);
    if (ival > 127) ival = 127;
    if (ival < -127) ival = -127;
    dst[idx] = (int8_t)ival;
}

/* Find absmax of a BF16 array (simple, not perf-critical — runs once at load). */
__global__ void absmax_bf16_kernel(const __nv_bfloat16* __restrict__ src,
                                    float* __restrict__ result, int n) {
    __shared__ float shared_max[256];
    float local_max = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        float v = fabsf(__bfloat162float(src[i]));
        if (v > local_max) local_max = v;
    }
    shared_max[threadIdx.x] = local_max;
    __syncthreads();
    if (threadIdx.x == 0) {
        float m = 0.0f;
        for (int i = 0; i < blockDim.x && i < 256; i++) {
            if (shared_max[i] > m) m = shared_max[i];
        }
        *result = m;
    }
}

/* ---- Kernel dispatch macros ----
 * These dispatch to FP16 or BF16 kernels based on model dtype.
 * All buffers are void* — we cast at the call site.
 */

#define AS_F16(p) ((__half*)(p))
#define AS_BF16(p) ((__nv_bfloat16*)(p))
#define AS_CF16(p) ((const __half*)(p))
#define AS_CBF16(p) ((const __nv_bfloat16*)(p))

#define IS_BF16(m) ((m)->config.dtype == GOLLMGO_DTYPE_BF16)

/* ---- C API implementation ---- */

extern "C" {

gollmgo_status_t gollmgo_model_create(gollmgo_backend_t b,
                                       const gollmgo_model_config_t* config,
                                       gollmgo_model_t* out) {
    if (!b || !config || !out) return GOLLMGO_ERR_INVALID;

    gollmgo_model_t m = new gollmgo_model();
    m->config = *config;
    m->backend = b;
    m->ready = false;
    m->max_batch = 0;
    memset(m->last_error, 0, sizeof(m->last_error));

    /* Derive head_dim if not set. */
    if (m->config.head_dim == 0 && m->config.num_heads > 0) {
        m->config.head_dim = m->config.hidden_size / m->config.num_heads;
    }
    if (m->config.rms_norm_eps == 0) {
        m->config.rms_norm_eps = 1e-5f;
    }

    /* Zero all buffer pointers. */
    m->buf_hidden = nullptr;
    m->buf_norm = nullptr;
    m->buf_q = nullptr;
    m->buf_k = nullptr;
    m->buf_v = nullptr;
    m->buf_attn_out = nullptr;
    m->buf_proj = nullptr;
    m->buf_gate = nullptr;
    m->buf_up = nullptr;
    m->buf_ffn = nullptr;
    m->buf_ffn_out = nullptr;
    m->buf_logits_half = nullptr;
    m->buf_logits_f32 = nullptr;
    m->paged_d_token_ids = nullptr;
    m->paged_d_positions = nullptr;
    m->paged_d_slot_mapping = nullptr;
    m->paged_d_seq_lens = nullptr;
    m->paged_d_slot_tables = nullptr;
    m->graph_max_context_len = 0;
    m->graph_hits = 0;
    m->graph_lookups = 0;
    m->cublas_workspace = nullptr;
    m->cublas_workspace_size = 0;
    m->buf_v2_exp_sums = nullptr;
    m->buf_v2_max_logits = nullptr;
    m->buf_v2_partial_out = nullptr;
    m->buf_v2_reduce_out = nullptr;
    m->max_num_partitions = 0;

    cublasStatus_t cstat = cublasCreate(&m->cublas);
    if (cstat != CUBLAS_STATUS_SUCCESS) {
        delete m;
        return GOLLMGO_ERR_CUDA;
    }
    cublasSetStream(m->cublas, 0);

    /* Give cuBLAS a dedicated workspace so graph capture doesn't depend on
     * internal lazy allocations on the first decode of a new GEMM shape. */
    m->cublas_workspace_size = 32u << 20;
    if (cudaMalloc(&m->cublas_workspace, m->cublas_workspace_size) == cudaSuccess) {
        cublasSetWorkspace(m->cublas, m->cublas_workspace, m->cublas_workspace_size);
    } else {
        m->cublas_workspace = nullptr;
        m->cublas_workspace_size = 0;
    }

    *out = m;
    return GOLLMGO_OK;
}

gollmgo_status_t gollmgo_model_load_weight(gollmgo_model_t m,
                                            const char* name,
                                            const void* host_data,
                                            int64_t size_bytes,
                                            const char* dtype) {
    if (!m || !name || !host_data || size_bytes <= 0) return GOLLMGO_ERR_INVALID;

    void* dev_ptr = nullptr;
    cudaError_t err = cudaMalloc(&dev_ptr, size_bytes);
    if (err != cudaSuccess) {
        model_set_error(m, cudaGetErrorString(err));
        return GOLLMGO_ERR_OOM;
    }

    err = cudaMemcpy(dev_ptr, host_data, size_bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        cudaFree(dev_ptr);
        model_set_error(m, cudaGetErrorString(err));
        return GOLLMGO_ERR_CUDA;
    }

    /* If weight already exists, free old one. */
    auto it = m->weights.find(name);
    if (it != m->weights.end()) {
        cudaFree(it->second.data);
    }

    m->weights[name] = {dev_ptr, size_bytes};

    /* If quantization is enabled, quantize linear weights (proj/gate/up/down/lm_head).
     * Skip embedding and layernorm weights (they stay in native precision). */
    if (m->config.quant_dtype != 0) {
        std::string sname(name);
        bool is_linear = (sname.find("_proj.weight") != std::string::npos) ||
                         (sname.find("gate_proj.weight") != std::string::npos) ||
                         (sname.find("up_proj.weight") != std::string::npos) ||
                         (sname.find("down_proj.weight") != std::string::npos) ||
                         (sname == "lm_head.weight");
        if (is_linear && IS_BF16(m)) {
            int num_elements = (int)(size_bytes / 2); /* BF16 = 2 bytes each */

            /* Compute absmax. */
            float* d_absmax;
            cudaMalloc(&d_absmax, sizeof(float));
            absmax_bf16_kernel<<<1, 256>>>((__nv_bfloat16*)dev_ptr, d_absmax, num_elements);
            float h_absmax = 0;
            cudaMemcpy(&h_absmax, d_absmax, sizeof(float), cudaMemcpyDeviceToHost);
            cudaFree(d_absmax);

            if (h_absmax > 0) {
                int blocks = (num_elements + 255) / 256;
                if (m->config.quant_dtype == GOLLMGO_DTYPE_FP8) {
                    float scale = h_absmax / 448.0f;
                    float scale_inv = 1.0f / scale;
                    void* q_ptr;
                    cudaMalloc(&q_ptr, num_elements); /* 1 byte per element */
                    quantize_to_fp8_kernel<<<blocks, 256>>>(
                        (__nv_bfloat16*)dev_ptr, (__nv_fp8_storage_t*)q_ptr, scale_inv, num_elements);
                    cudaDeviceSynchronize();
                    cudaFree(dev_ptr);
                    m->weights[sname] = {q_ptr, (int64_t)num_elements};
                    m->weight_scales[sname] = scale;
                } else if (m->config.quant_dtype == GOLLMGO_DTYPE_INT8) {
                    float scale = h_absmax / 127.0f;
                    float scale_inv = 1.0f / scale;
                    void* q_ptr;
                    cudaMalloc(&q_ptr, num_elements);
                    quantize_to_int8_kernel<<<blocks, 256>>>(
                        (__nv_bfloat16*)dev_ptr, (int8_t*)q_ptr, scale_inv, num_elements);
                    cudaDeviceSynchronize();
                    cudaFree(dev_ptr);
                    m->weights[sname] = {q_ptr, (int64_t)num_elements};
                    m->weight_scales[sname] = scale;
                }
            }
        }
    }

    return GOLLMGO_OK;
}

gollmgo_status_t gollmgo_model_ready(gollmgo_model_t m) {
    if (!m) return GOLLMGO_ERR_INVALID;

    /* Validate that critical weights exist. */
    if (!get_weight(m, "model.embed_tokens.weight")) {
        model_set_error(m, "missing model.embed_tokens.weight");
        return GOLLMGO_ERR_INVALID;
    }
    if (!get_weight(m, "model.norm.weight")) {
        model_set_error(m, "missing model.norm.weight");
        return GOLLMGO_ERR_INVALID;
    }
    if (!get_weight(m, "lm_head.weight")) {
        model_set_error(m, "missing lm_head.weight");
        return GOLLMGO_ERR_INVALID;
    }

    /* Allocate scratch buffers — 2 bytes per element for both FP16 and BF16. */
    int max_n = m->config.max_seq_len;
    if (max_n <= 0) max_n = 2048;
    m->max_batch = max_n;

    auto& c = m->config;
    gollmgo_status_t st;

    #define ALLOC_BUF(ptr, count) do { \
        st = model_check_cuda(m, cudaMalloc(&(ptr), (count) * 2)); \
        if (st != GOLLMGO_OK) return st; \
    } while(0)

    /* buf_hidden is FP32 (4 bytes per element). */
    st = model_check_cuda(m, cudaMalloc((void**)&m->buf_hidden,
                                         (size_t)max_n * c.hidden_size * sizeof(float)));
    if (st != GOLLMGO_OK) return st;
    ALLOC_BUF(m->buf_norm,        max_n * c.hidden_size);
    ALLOC_BUF(m->buf_q,           max_n * c.num_heads * c.head_dim);
    ALLOC_BUF(m->buf_k,           max_n * c.num_kv_heads * c.head_dim);
    ALLOC_BUF(m->buf_v,           max_n * c.num_kv_heads * c.head_dim);
    ALLOC_BUF(m->buf_attn_out,    max_n * c.num_heads * c.head_dim);
    ALLOC_BUF(m->buf_proj,        max_n * c.hidden_size);
    ALLOC_BUF(m->buf_gate,        max_n * c.intermediate_size);
    ALLOC_BUF(m->buf_up,          max_n * c.intermediate_size);
    ALLOC_BUF(m->buf_ffn,         max_n * c.intermediate_size);
    ALLOC_BUF(m->buf_ffn_out,     max_n * c.hidden_size);
    ALLOC_BUF(m->buf_logits_half, max_n * c.vocab_size);
    st = model_check_cuda(m, cudaMalloc((void**)&m->buf_logits_f32,
                                         max_n * c.vocab_size * sizeof(float)));
    if (st != GOLLMGO_OK) return st;

    #undef ALLOC_BUF

    /* Allocate PagedAttention v2 scratch buffers. */
    {
        int max_ctx = c.max_seq_len > 0 ? c.max_seq_len : 2048;
        int part_size = PAGED_ATTN_V2_PARTITION_SIZE;
        m->max_num_partitions = (max_ctx + part_size - 1) / part_size;
        int mnp = m->max_num_partitions;
        size_t scalar_size = (size_t)max_n * c.num_heads * mnp * sizeof(float);
        size_t vec_size = (size_t)max_n * c.num_heads * mnp * c.head_dim * sizeof(float);
        size_t reduce_size = (size_t)max_n * c.num_heads * c.head_dim * sizeof(float);

        st = model_check_cuda(m, cudaMalloc((void**)&m->buf_v2_exp_sums, scalar_size));
        if (st != GOLLMGO_OK) return st;
        st = model_check_cuda(m, cudaMalloc((void**)&m->buf_v2_max_logits, scalar_size));
        if (st != GOLLMGO_OK) return st;
        st = model_check_cuda(m, cudaMalloc((void**)&m->buf_v2_partial_out, vec_size));
        if (st != GOLLMGO_OK) return st;
        st = model_check_cuda(m, cudaMalloc((void**)&m->buf_v2_reduce_out, reduce_size));
        if (st != GOLLMGO_OK) return st;
    }

    /* Pre-allocate decode input buffers (eliminates cudaMalloc from hot path). */
    {
        int max_ctx = c.max_seq_len > 0 ? c.max_seq_len : 2048;
        st = model_check_cuda(m, cudaMalloc((void**)&m->paged_d_token_ids, max_n * sizeof(int32_t)));
        if (st != GOLLMGO_OK) return st;
        st = model_check_cuda(m, cudaMalloc((void**)&m->paged_d_positions, max_n * sizeof(int32_t)));
        if (st != GOLLMGO_OK) return st;
        st = model_check_cuda(m, cudaMalloc((void**)&m->paged_d_slot_mapping, max_n * sizeof(int32_t)));
        if (st != GOLLMGO_OK) return st;
        st = model_check_cuda(m, cudaMalloc((void**)&m->paged_d_seq_lens, max_n * sizeof(int32_t)));
        if (st != GOLLMGO_OK) return st;
        st = model_check_cuda(m, cudaMalloc((void**)&m->paged_d_slot_tables,
                                             (size_t)max_n * max_ctx * sizeof(int32_t)));
        if (st != GOLLMGO_OK) return st;
        m->graph_max_context_len = max_ctx;
    }

    m->ready = true;
    return GOLLMGO_OK;
}

/* ---- Helper: run one transformer layer ---- */
static gollmgo_status_t run_layer(gollmgo_model_t m, int layer, int N,
                                   int32_t* d_positions, int32_t* d_slot_mapping,
                                   gollmgo_kvcache_t kv_cache,
                                   /* paged decode params (NULL for prefill/eager) */
                                   int32_t* d_seq_lens, int32_t* d_slot_tables,
                                   int max_context_len, bool paged_decode) {
    auto& c = m->config;
    gollmgo_status_t st;
    char name_buf[256];
    bool bf16 = IS_BF16(m);

    /* -- Attention block -- */

    /* RMSNorm (input_layernorm). */
    snprintf(name_buf, sizeof(name_buf), "model.layers.%d.input_layernorm.weight", layer);
    void* norm_weight = get_weight(m, name_buf);
    if (!norm_weight) { model_set_error(m, name_buf); return GOLLMGO_ERR_INVALID; }

    {
        int threads = (c.hidden_size < 256) ? c.hidden_size : 256;
        /* FP32 hidden → half-precision norm output for GEMM. */
        if (bf16)
            rmsnorm_f32in_bf16w<<<N, threads>>>(m->buf_hidden, AS_CBF16(norm_weight),
                AS_BF16(m->buf_norm), c.hidden_size, N, c.rms_norm_eps);
        else
            rmsnorm_f32in_f16w<<<N, threads>>>(m->buf_hidden, AS_CF16(norm_weight),
                AS_F16(m->buf_norm), c.hidden_size, N, c.rms_norm_eps);
    }

    /* Q, K, V projections. */
    snprintf(name_buf, sizeof(name_buf), "model.layers.%d.self_attn.q_proj.weight", layer);
    void* wq = get_weight(m, name_buf);
    snprintf(name_buf, sizeof(name_buf), "model.layers.%d.self_attn.k_proj.weight", layer);
    void* wk = get_weight(m, name_buf);
    snprintf(name_buf, sizeof(name_buf), "model.layers.%d.self_attn.v_proj.weight", layer);
    void* wv = get_weight(m, name_buf);
    if (!wq || !wk || !wv) { model_set_error(m, "missing QKV weights"); return GOLLMGO_ERR_INVALID; }

    st = gemm(m, m->buf_norm, wq, m->buf_q, N, c.num_heads * c.head_dim, c.hidden_size);
    if (st != GOLLMGO_OK) return st;
    st = gemm(m, m->buf_norm, wk, m->buf_k, N, c.num_kv_heads * c.head_dim, c.hidden_size);
    if (st != GOLLMGO_OK) return st;
    st = gemm(m, m->buf_norm, wv, m->buf_v, N, c.num_kv_heads * c.head_dim, c.hidden_size);
    if (st != GOLLMGO_OK) return st;

    /* RoPE. */
    {
        int max_heads = (c.num_heads > c.num_kv_heads) ? c.num_heads : c.num_kv_heads;
        dim3 grid(N, max_heads);
        int threads = c.head_dim / 2;
        if (bf16)
            rope_bf16<<<grid, threads>>>(AS_BF16(m->buf_q), AS_BF16(m->buf_k),
                d_positions, N, c.num_heads, c.num_kv_heads, c.head_dim, 10000.0f);
        else
            rope_f16<<<grid, threads>>>(AS_F16(m->buf_q), AS_F16(m->buf_k),
                d_positions, N, c.num_heads, c.num_kv_heads, c.head_dim, 10000.0f);
    }

    /* KV cache write (if available). */
    if (kv_cache && d_slot_mapping) {
        void* lk = gollmgo_kvcache_k_layer_ptr(kv_cache, layer);
        void* lv = gollmgo_kvcache_v_layer_ptr(kv_cache, layer);
        if (lk && lv) {
            int kv_size = c.num_kv_heads * c.head_dim;
            int threads = (kv_size < 256) ? kv_size : 256;
            if (bf16)
                paged_kv_write_bf16<<<N, threads>>>(AS_BF16(lk), AS_BF16(lv),
                    AS_CBF16(m->buf_k), AS_CBF16(m->buf_v), d_slot_mapping,
                    N, c.num_kv_heads, c.head_dim);
            else
                paged_kv_write_f16<<<N, threads>>>(AS_F16(lk), AS_F16(lv),
                    AS_CF16(m->buf_k), AS_CF16(m->buf_v), d_slot_mapping,
                    N, c.num_kv_heads, c.head_dim);
        }
    }

    /* Attention. */
    if (paged_decode && kv_cache) {
        void* lk = gollmgo_kvcache_k_layer_ptr(kv_cache, layer);
        void* lv = gollmgo_kvcache_v_layer_ptr(kv_cache, layer);
        float attn_scale = 1.0f / sqrtf((float)c.head_dim);
        int threads = (c.head_dim < 32) ? c.head_dim : 32;
        int part_size = PAGED_ATTN_V2_PARTITION_SIZE;
        int num_parts = (max_context_len + part_size - 1) / part_size;

        if (num_parts <= 1) {
            /* Short context — use v1 (no partition overhead). */
            int smem = max_context_len * sizeof(float);
            if (bf16)
                paged_attention_v1_bf16<<<dim3(N, c.num_heads), threads, smem>>>(
                    AS_CBF16(m->buf_q), AS_CBF16(lk), AS_CBF16(lv), AS_BF16(m->buf_attn_out),
                    d_seq_lens, d_slot_tables,
                    N, c.num_heads, c.num_kv_heads, c.head_dim,
                    max_context_len, attn_scale);
            else
                paged_attention_v1_f16<<<dim3(N, c.num_heads), threads, smem>>>(
                    AS_CF16(m->buf_q), AS_CF16(lk), AS_CF16(lv), AS_F16(m->buf_attn_out),
                    d_seq_lens, d_slot_tables,
                    N, c.num_heads, c.num_kv_heads, c.head_dim,
                    max_context_len, attn_scale);
        } else {
            /* Long context — use v2 (partitioned). */
            dim3 grid1(N, c.num_heads, num_parts);
            int smem1 = part_size * sizeof(float);
            if (bf16)
                paged_attention_v2_phase1_bf16<<<grid1, threads, smem1>>>(
                    AS_CBF16(m->buf_q), AS_CBF16(lk), AS_CBF16(lv),
                    m->buf_v2_exp_sums, m->buf_v2_max_logits, m->buf_v2_partial_out,
                    d_seq_lens, d_slot_tables,
                    N, c.num_heads, c.num_kv_heads, c.head_dim,
                    max_context_len, attn_scale, part_size);
            else
                paged_attention_v2_phase1_f16<<<grid1, threads, smem1>>>(
                    AS_CF16(m->buf_q), AS_CF16(lk), AS_CF16(lv),
                    m->buf_v2_exp_sums, m->buf_v2_max_logits, m->buf_v2_partial_out,
                    d_seq_lens, d_slot_tables,
                    N, c.num_heads, c.num_kv_heads, c.head_dim,
                    max_context_len, attn_scale, part_size);

            /* Phase 2: reduce partitions → FP32 output. */
            dim3 grid2(N, c.num_heads);
            int smem2 = num_parts * sizeof(float);
            paged_attention_v2_reduce<<<grid2, threads, smem2>>>(
                m->buf_v2_exp_sums, m->buf_v2_max_logits, m->buf_v2_partial_out,
                m->buf_v2_reduce_out,
                d_seq_lens, N, c.num_heads, c.head_dim, part_size, num_parts);

            /* Convert FP32 reduce output → half-precision attn_out for o_proj GEMM. */
            int total = N * c.num_heads * c.head_dim;
            int cvt_blocks = (total + 255) / 256;
            if (bf16)
                f32_to_bf16<<<cvt_blocks, 256>>>(m->buf_v2_reduce_out,
                    AS_BF16(m->buf_attn_out), total);
            else
                f32_to_f16<<<cvt_blocks, 256>>>(m->buf_v2_reduce_out,
                    AS_F16(m->buf_attn_out), total);
        }
    } else {
        /* Naive (eager) attention. */
        dim3 grid(N, c.num_heads);
        int threads = (c.head_dim < 32) ? c.head_dim : 32;
        int smem = N * sizeof(float);
        if (bf16)
            naive_attention_bf16<<<grid, threads, smem>>>(
                AS_CBF16(m->buf_q), AS_CBF16(m->buf_k), AS_CBF16(m->buf_v),
                AS_BF16(m->buf_attn_out), d_positions,
                N, c.num_heads, c.num_kv_heads, c.head_dim,
                1.0f / sqrtf((float)c.head_dim));
        else
            naive_attention_f16<<<grid, threads, smem>>>(
                AS_CF16(m->buf_q), AS_CF16(m->buf_k), AS_CF16(m->buf_v),
                AS_F16(m->buf_attn_out), d_positions,
                N, c.num_heads, c.num_kv_heads, c.head_dim,
                1.0f / sqrtf((float)c.head_dim));
    }

    /* Output projection. */
    snprintf(name_buf, sizeof(name_buf), "model.layers.%d.self_attn.o_proj.weight", layer);
    void* wo = get_weight(m, name_buf);
    if (!wo) { model_set_error(m, "missing o_proj weight"); return GOLLMGO_ERR_INVALID; }

    st = gemm(m, m->buf_attn_out, wo, m->buf_proj, N, c.hidden_size, c.num_heads * c.head_dim);
    if (st != GOLLMGO_OK) return st;

    /* Residual add: FP32 hidden += half-precision proj. */
    {
        int total = N * c.hidden_size;
        int threads = 256;
        int blocks = (total + threads - 1) / threads;
        if (bf16)
            residual_add_f32_bf16<<<blocks, threads>>>(m->buf_hidden,
                AS_CBF16(m->buf_proj), total);
        else
            residual_add_f32_f16<<<blocks, threads>>>(m->buf_hidden,
                AS_CF16(m->buf_proj), total);
    }

    /* -- FFN block -- */

    /* RMSNorm (post_attention_layernorm). */
    snprintf(name_buf, sizeof(name_buf), "model.layers.%d.post_attention_layernorm.weight", layer);
    norm_weight = get_weight(m, name_buf);
    if (!norm_weight) { model_set_error(m, name_buf); return GOLLMGO_ERR_INVALID; }

    {
        int threads = (c.hidden_size < 256) ? c.hidden_size : 256;
        if (bf16)
            rmsnorm_f32in_bf16w<<<N, threads>>>(m->buf_hidden, AS_CBF16(norm_weight),
                AS_BF16(m->buf_norm), c.hidden_size, N, c.rms_norm_eps);
        else
            rmsnorm_f32in_f16w<<<N, threads>>>(m->buf_hidden, AS_CF16(norm_weight),
                AS_F16(m->buf_norm), c.hidden_size, N, c.rms_norm_eps);
    }

    /* Gate and up projections. */
    snprintf(name_buf, sizeof(name_buf), "model.layers.%d.mlp.gate_proj.weight", layer);
    void* w_gate = get_weight(m, name_buf);
    snprintf(name_buf, sizeof(name_buf), "model.layers.%d.mlp.up_proj.weight", layer);
    void* w_up = get_weight(m, name_buf);
    if (!w_gate || !w_up) { model_set_error(m, "missing gate/up weights"); return GOLLMGO_ERR_INVALID; }

    st = gemm(m, m->buf_norm, w_gate, m->buf_gate, N, c.intermediate_size, c.hidden_size);
    if (st != GOLLMGO_OK) return st;
    st = gemm(m, m->buf_norm, w_up, m->buf_up, N, c.intermediate_size, c.hidden_size);
    if (st != GOLLMGO_OK) return st;

    /* SiLU(gate) * up. */
    {
        int total = N * c.intermediate_size;
        int threads = 256;
        int blocks = (total + threads - 1) / threads;
        if (bf16)
            silu_mul_bf16<<<blocks, threads>>>(AS_CBF16(m->buf_gate), AS_CBF16(m->buf_up),
                AS_BF16(m->buf_ffn), total);
        else
            silu_mul_f16<<<blocks, threads>>>(AS_CF16(m->buf_gate), AS_CF16(m->buf_up),
                AS_F16(m->buf_ffn), total);
    }

    /* Down projection. */
    snprintf(name_buf, sizeof(name_buf), "model.layers.%d.mlp.down_proj.weight", layer);
    void* w_down = get_weight(m, name_buf);
    if (!w_down) { model_set_error(m, "missing down_proj weight"); return GOLLMGO_ERR_INVALID; }

    st = gemm(m, m->buf_ffn, w_down, m->buf_ffn_out, N, c.hidden_size, c.intermediate_size);
    if (st != GOLLMGO_OK) return st;

    /* Residual add: FP32 hidden += half-precision ffn_out. */
    {
        int total = N * c.hidden_size;
        int threads = 256;
        int blocks = (total + threads - 1) / threads;
        if (bf16)
            residual_add_f32_bf16<<<blocks, threads>>>(m->buf_hidden,
                AS_CBF16(m->buf_ffn_out), total);
        else
            residual_add_f32_f16<<<blocks, threads>>>(m->buf_hidden,
                AS_CF16(m->buf_ffn_out), total);
    }

    return GOLLMGO_OK;
}

/* ---- Helper: embedding + final norm + LM head ---- */

static void run_embedding(gollmgo_model_t m, int32_t* d_token_ids, int N) {
    auto& c = m->config;
    int threads = (c.hidden_size < 256) ? c.hidden_size : 256;
    dim3 grid(N, (c.hidden_size + threads - 1) / threads);
    /* Write directly to FP32 buf_hidden from half-precision embedding table. */
    if (IS_BF16(m))
        embedding_bf16_to_f32<<<grid, threads>>>(
            AS_CBF16(get_weight(m, "model.embed_tokens.weight")),
            d_token_ids, m->buf_hidden, c.hidden_size, c.vocab_size);
    else
        embedding_f16_to_f32<<<grid, threads>>>(
            AS_CF16(get_weight(m, "model.embed_tokens.weight")),
            d_token_ids, m->buf_hidden, c.hidden_size, c.vocab_size);
}

/* Compute LM head on device — logits end up in m->buf_logits_f32. */
static gollmgo_status_t run_lm_head_device(gollmgo_model_t m, int N) {
    auto& c = m->config;
    bool bf16 = IS_BF16(m);
    gollmgo_status_t st;

    /* Final RMSNorm: FP32 hidden → half-precision for LM head GEMM. */
    {
        int threads = (c.hidden_size < 256) ? c.hidden_size : 256;
        if (bf16)
            rmsnorm_f32in_bf16w<<<N, threads>>>(m->buf_hidden,
                AS_CBF16(get_weight(m, "model.norm.weight")),
                AS_BF16(m->buf_norm), c.hidden_size, N, c.rms_norm_eps);
        else
            rmsnorm_f32in_f16w<<<N, threads>>>(m->buf_hidden,
                AS_CF16(get_weight(m, "model.norm.weight")),
                AS_F16(m->buf_norm), c.hidden_size, N, c.rms_norm_eps);
    }

    /* LM head GEMM. */
    st = gemm(m, m->buf_norm, get_weight(m, "lm_head.weight"), m->buf_logits_half,
              N, c.vocab_size, c.hidden_size);
    if (st != GOLLMGO_OK) return st;

    /* Convert to F32 on device. */
    {
        int total = N * c.vocab_size;
        int threads = 256;
        int blocks = (total + threads - 1) / threads;
        if (bf16)
            bf16_to_f32<<<blocks, threads>>>(AS_CBF16(m->buf_logits_half),
                m->buf_logits_f32, total);
        else
            f16_to_f32<<<blocks, threads>>>(AS_CF16(m->buf_logits_half),
                m->buf_logits_f32, total);
    }
    return GOLLMGO_OK;
}

/* Compute + copy logits to host. */
static gollmgo_status_t run_lm_head(gollmgo_model_t m, int N, float* host_logits_out) {
    gollmgo_status_t st = run_lm_head_device(m, N);
    if (st != GOLLMGO_OK) return st;
    return model_check_cuda(m, cudaMemcpy(
        host_logits_out, m->buf_logits_f32,
        N * m->config.vocab_size * sizeof(float),
        cudaMemcpyDeviceToHost));
}

/* ---- Forward pass implementations ---- */

gollmgo_status_t gollmgo_model_forward(gollmgo_backend_t b,
                                        gollmgo_model_t m,
                                        const int32_t* host_token_ids,
                                        const int32_t* host_positions,
                                        int n_tokens,
                                        float* host_logits_out) {
    if (!b || !m || !m->ready) return GOLLMGO_ERR_INVALID;
    if (n_tokens <= 0 || n_tokens > m->max_batch) {
        model_set_error(m, "n_tokens out of range");
        return GOLLMGO_ERR_INVALID;
    }

    gollmgo_status_t st;

    int32_t* d_token_ids = nullptr;
    int32_t* d_positions = nullptr;
    st = model_check_cuda(m, cudaMalloc((void**)&d_token_ids, n_tokens * sizeof(int32_t)));
    if (st != GOLLMGO_OK) return st;
    st = model_check_cuda(m, cudaMalloc((void**)&d_positions, n_tokens * sizeof(int32_t)));
    if (st != GOLLMGO_OK) { cudaFree(d_token_ids); return st; }

    cudaMemcpy(d_token_ids, host_token_ids, n_tokens * sizeof(int32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_positions, host_positions, n_tokens * sizeof(int32_t), cudaMemcpyHostToDevice);

    run_embedding(m, d_token_ids, n_tokens);

    for (int layer = 0; layer < m->config.num_layers; layer++) {
        st = run_layer(m, layer, n_tokens, d_positions,
                       nullptr, nullptr, nullptr, nullptr, 0, false);
        if (st != GOLLMGO_OK) { cudaFree(d_token_ids); cudaFree(d_positions); return st; }
    }

    st = run_lm_head(m, n_tokens, host_logits_out);

    cudaFree(d_token_ids);
    cudaFree(d_positions);
    return st;
}

gollmgo_status_t gollmgo_model_forward_prefill(
    gollmgo_backend_t b,
    gollmgo_model_t m,
    const int32_t* host_token_ids,
    const int32_t* host_positions,
    const int32_t* host_slot_mapping,
    int n_tokens,
    gollmgo_kvcache_t kv_cache,
    float* host_logits_out) {

    if (!b || !m || !m->ready) return GOLLMGO_ERR_INVALID;
    if (n_tokens <= 0 || n_tokens > m->max_batch) {
        model_set_error(m, "n_tokens out of range");
        return GOLLMGO_ERR_INVALID;
    }

    gollmgo_status_t st;

    int32_t* d_token_ids = nullptr;
    int32_t* d_positions = nullptr;
    int32_t* d_slot_mapping = nullptr;

    st = model_check_cuda(m, cudaMalloc((void**)&d_token_ids, n_tokens * sizeof(int32_t)));
    if (st != GOLLMGO_OK) return st;
    st = model_check_cuda(m, cudaMalloc((void**)&d_positions, n_tokens * sizeof(int32_t)));
    if (st != GOLLMGO_OK) { cudaFree(d_token_ids); return st; }

    cudaMemcpy(d_token_ids, host_token_ids, n_tokens * sizeof(int32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_positions, host_positions, n_tokens * sizeof(int32_t), cudaMemcpyHostToDevice);

    if (kv_cache && host_slot_mapping) {
        st = model_check_cuda(m, cudaMalloc((void**)&d_slot_mapping, n_tokens * sizeof(int32_t)));
        if (st != GOLLMGO_OK) { cudaFree(d_token_ids); cudaFree(d_positions); return st; }
        cudaMemcpy(d_slot_mapping, host_slot_mapping, n_tokens * sizeof(int32_t), cudaMemcpyHostToDevice);
    }

    run_embedding(m, d_token_ids, n_tokens);

    for (int layer = 0; layer < m->config.num_layers; layer++) {
        st = run_layer(m, layer, n_tokens, d_positions,
                       d_slot_mapping, kv_cache, nullptr, nullptr, 0, false);
        if (st != GOLLMGO_OK) goto prefill_cleanup;
    }

    st = run_lm_head(m, n_tokens, host_logits_out);

prefill_cleanup:
    cudaFree(d_token_ids);
    cudaFree(d_positions);
    if (d_slot_mapping) cudaFree(d_slot_mapping);
    return st;
}

gollmgo_status_t gollmgo_model_forward_paged(
    gollmgo_backend_t b,
    gollmgo_model_t m,
    const int32_t* host_token_ids,
    const int32_t* host_positions,
    const int32_t* host_slot_mapping,
    int n_tokens,
    gollmgo_kvcache_t kv_cache,
    const int32_t* host_seq_lens,
    const int32_t* host_slot_tables,
    int n_seqs,
    int max_context_len,
    const int32_t* /*host_seq_token_counts*/,
    float* host_logits_out)
{
    if (!b || !m || !m->ready || !kv_cache) return GOLLMGO_ERR_INVALID;
    if (n_tokens <= 0 || n_tokens > m->max_batch) {
        model_set_error(m, "n_tokens out of range");
        return GOLLMGO_ERR_INVALID;
    }
    if (n_tokens != n_seqs) {
        model_set_error(m, "paged forward requires n_tokens == n_seqs (decode only)");
        return GOLLMGO_ERR_INVALID;
    }

    int N = n_tokens;
    gollmgo_status_t st;

    /* Use pre-allocated buffers (no cudaMalloc in hot path). */

    int32_t* d_token_ids = m->paged_d_token_ids;
    int32_t* d_positions = m->paged_d_positions;
    int32_t* d_slot_mapping = m->paged_d_slot_mapping;
    int32_t* d_seq_lens = m->paged_d_seq_lens;
    int32_t* d_slot_tables = m->paged_d_slot_tables;

    cudaMemcpy(d_token_ids, host_token_ids, N * sizeof(int32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_positions, host_positions, N * sizeof(int32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_slot_mapping, host_slot_mapping, N * sizeof(int32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_seq_lens, host_seq_lens, N * sizeof(int32_t), cudaMemcpyHostToDevice);
    /* Copy slot tables into the pre-allocated buffer padded to graph_max_context_len. */
    if (max_context_len <= m->graph_max_context_len) {
        cudaMemcpy(d_slot_tables, host_slot_tables,
                   (size_t)N * max_context_len * sizeof(int32_t), cudaMemcpyHostToDevice);
    } else {
        /* Truncate — shouldn't happen in practice. */
        cudaMemcpy(d_slot_tables, host_slot_tables,
                   (size_t)N * m->graph_max_context_len * sizeof(int32_t), cudaMemcpyHostToDevice);
    }

    /* Check for cached CUDA graph. */
    m->graph_lookups++;
    const bool graph_eligible = (N <= 64) &&
                                (max_context_len > 0) &&
                                (max_context_len <= m->graph_max_context_len);
    const uint64_t graph_key = graph_cache_key(N, max_context_len);
    auto it = m->graph_cache.find(graph_key);
    if (it != m->graph_cache.end() && graph_eligible) {
        /* Graph replay — skip kernel launches, just replay the captured graph. */
        m->graph_hits++;
        cudaGraphLaunch(it->second, 0 /* default stream */);
        cudaStreamSynchronize(0);
    } else {
        /* Eager path (also used for graph capture). */
        bool should_capture = (it == m->graph_cache.end()) && graph_eligible;
        cudaGraph_t graph = nullptr;
        if (should_capture) {
            cudaError_t capture_err = cudaStreamBeginCapture(0, cudaStreamCaptureModeRelaxed);
            if (capture_err != cudaSuccess) {
                should_capture = false;
            }
        }

        run_embedding(m, d_token_ids, N);

        for (int layer = 0; layer < m->config.num_layers; layer++) {
            st = run_layer(m, layer, N, d_positions,
                           d_slot_mapping, kv_cache,
                           d_seq_lens, d_slot_tables, max_context_len, true);
            if (st != GOLLMGO_OK) {
                if (should_capture) cudaStreamEndCapture(0, &graph);
                if (graph) cudaGraphDestroy(graph);
                return st;
            }
        }

        run_lm_head_device(m, N);

        if (should_capture) {
            cudaError_t capture_err = cudaStreamEndCapture(0, &graph);
            if (capture_err == cudaSuccess && graph) {
                cudaGraphExec_t exec = nullptr;
                if (cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0) == cudaSuccess && exec) {
                    m->graph_cache[graph_key] = exec;
                }
                cudaGraphDestroy(graph);
            } else if (graph) {
                cudaGraphDestroy(graph);
            }
        }
    }

    /* Copy logits to host. */
    st = model_check_cuda(m, cudaMemcpy(
        host_logits_out, m->buf_logits_f32,
        N * m->config.vocab_size * sizeof(float),
        cudaMemcpyDeviceToHost));

    return st;
}

void gollmgo_model_graph_stats(gollmgo_model_t m, int64_t* hits, int64_t* lookups) {
    if (!m) { *hits = 0; *lookups = 0; return; }
    *hits = m->graph_hits;
    *lookups = m->graph_lookups;
}

gollmgo_status_t gollmgo_model_destroy(gollmgo_model_t m) {
    if (!m) return GOLLMGO_OK;

    for (auto& kv : m->weights) {
        cudaFree(kv.second.data);
    }

    cudaFree(m->buf_hidden);
    cudaFree(m->buf_norm);
    cudaFree(m->buf_q);
    cudaFree(m->buf_k);
    cudaFree(m->buf_v);
    cudaFree(m->buf_attn_out);
    cudaFree(m->buf_proj);
    cudaFree(m->buf_gate);
    cudaFree(m->buf_up);
    cudaFree(m->buf_ffn);
    cudaFree(m->buf_ffn_out);
    cudaFree(m->buf_logits_half);
    cudaFree(m->buf_logits_f32);
    cudaFree(m->paged_d_token_ids);
    cudaFree(m->paged_d_positions);
    cudaFree(m->paged_d_slot_mapping);
    cudaFree(m->paged_d_seq_lens);
    cudaFree(m->paged_d_slot_tables);
    cudaFree(m->buf_v2_exp_sums);
    cudaFree(m->buf_v2_max_logits);
    cudaFree(m->buf_v2_partial_out);
    cudaFree(m->buf_v2_reduce_out);

    /* Destroy cached CUDA graphs. */
    for (auto& kv : m->graph_cache) {
        cudaGraphExecDestroy(kv.second);
    }

    if (m->cublas) {
        cublasDestroy(m->cublas);
    }
    cudaFree(m->cublas_workspace);

    delete m;
    return GOLLMGO_OK;
}

} /* extern "C" */
