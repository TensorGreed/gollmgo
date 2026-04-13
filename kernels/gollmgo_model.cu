/*
 * gollmgo_model.cu — Model loading and eager forward pass.
 *
 * Uses cuBLAS for linear projections (GEMM) and custom kernels for
 * embedding, RMSNorm, RoPE, SiLU, and naive attention.
 *
 * M5: correctness-first eager forward pass.
 * M6 replaces naive attention with paged attention + KV cache.
 */

#include "gollmgo_model.h"
#include "gollmgo_ops.cuh"
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstring>
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
    bool                   ready;
    char                   last_error[512];

    /* Named weight tensors on device. */
    std::unordered_map<std::string, weight_tensor> weights;

    /* Scratch buffers (allocated on model_ready). */
    __half* buf_hidden;       /* [max_batch, hidden_size] */
    __half* buf_norm;         /* [max_batch, hidden_size] */
    __half* buf_q;            /* [max_batch, num_heads * head_dim] */
    __half* buf_k;            /* [max_batch, num_kv_heads * head_dim] */
    __half* buf_v;            /* [max_batch, num_kv_heads * head_dim] */
    __half* buf_attn_out;     /* [max_batch, num_heads * head_dim] */
    __half* buf_proj;         /* [max_batch, hidden_size] */
    __half* buf_gate;         /* [max_batch, intermediate_size] */
    __half* buf_up;           /* [max_batch, intermediate_size] */
    __half* buf_ffn;          /* [max_batch, intermediate_size] */
    __half* buf_ffn_out;      /* [max_batch, hidden_size] */
    __half* buf_logits_f16;   /* [max_batch, vocab_size] */
    float*  buf_logits_f32;   /* [max_batch, vocab_size] — device staging */

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
static __half* get_weight(gollmgo_model_t m, const std::string& name) {
    auto it = m->weights.find(name);
    if (it == m->weights.end()) return nullptr;
    return (__half*)it->second.data;
}

/* ---- cuBLAS GEMM helper ---- */
/* C = A * B^T  (row-major A[M,K], B[N,K] -> C[M,N]) */
/* cuBLAS is column-major, so we compute C^T = B * A^T. */
static gollmgo_status_t gemm_f16(gollmgo_model_t m,
                                  const __half* A, const __half* B, __half* C,
                                  int M, int N, int K) {
    __half alpha_h = __float2half(1.0f);
    __half beta_h  = __float2half(0.0f);

    /* Column-major: op(B)[K,N] * op(A)[K,M]^T -> C_col[N,M] = C_row[M,N] */
    cublasStatus_t status = cublasHgemm(
        m->cublas,
        CUBLAS_OP_T,  /* B^T in col-major = B in row-major */
        CUBLAS_OP_N,  /* A in col-major = A^T in row-major */
        N, M, K,
        &alpha_h,
        B, K,         /* B: [N, K] row-major -> [K, N] col-major, ldb=K */
        A, K,         /* A: [M, K] row-major -> [K, M] col-major, lda=K */
        &beta_h,
        C, N          /* C: [M, N] row-major -> [N, M] col-major, ldc=N */
    );

    if (status != CUBLAS_STATUS_SUCCESS) {
        model_set_error(m, "cuBLAS HGEMM failed");
        return GOLLMGO_ERR_INTERNAL;
    }
    return GOLLMGO_OK;
}

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
    m->buf_logits_f16 = nullptr;
    m->buf_logits_f32 = nullptr;

    cublasStatus_t cstat = cublasCreate(&m->cublas);
    if (cstat != CUBLAS_STATUS_SUCCESS) {
        delete m;
        return GOLLMGO_ERR_CUDA;
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

    /* Allocate scratch buffers for max_seq_len tokens. */
    int max_n = m->config.max_seq_len;
    if (max_n <= 0) max_n = 2048;
    m->max_batch = max_n;

    auto& c = m->config;
    gollmgo_status_t st;

    #define ALLOC_BUF(ptr, count) do { \
        st = model_check_cuda(m, cudaMalloc((void**)&(ptr), (count) * sizeof(__half))); \
        if (st != GOLLMGO_OK) return st; \
    } while(0)

    ALLOC_BUF(m->buf_hidden,     max_n * c.hidden_size);
    ALLOC_BUF(m->buf_norm,       max_n * c.hidden_size);
    ALLOC_BUF(m->buf_q,          max_n * c.num_heads * c.head_dim);
    ALLOC_BUF(m->buf_k,          max_n * c.num_kv_heads * c.head_dim);
    ALLOC_BUF(m->buf_v,          max_n * c.num_kv_heads * c.head_dim);
    ALLOC_BUF(m->buf_attn_out,   max_n * c.num_heads * c.head_dim);
    ALLOC_BUF(m->buf_proj,       max_n * c.hidden_size);
    ALLOC_BUF(m->buf_gate,       max_n * c.intermediate_size);
    ALLOC_BUF(m->buf_up,         max_n * c.intermediate_size);
    ALLOC_BUF(m->buf_ffn,        max_n * c.intermediate_size);
    ALLOC_BUF(m->buf_ffn_out,    max_n * c.hidden_size);
    ALLOC_BUF(m->buf_logits_f16, max_n * c.vocab_size);
    st = model_check_cuda(m, cudaMalloc((void**)&m->buf_logits_f32,
                                         max_n * c.vocab_size * sizeof(float)));
    if (st != GOLLMGO_OK) return st;

    #undef ALLOC_BUF

    m->ready = true;
    return GOLLMGO_OK;
}

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

    auto& c = m->config;
    gollmgo_status_t st;

    /* Copy input tokens and positions to device. */
    int32_t* d_token_ids = nullptr;
    int32_t* d_positions = nullptr;
    st = model_check_cuda(m, cudaMalloc((void**)&d_token_ids, n_tokens * sizeof(int32_t)));
    if (st != GOLLMGO_OK) return st;
    st = model_check_cuda(m, cudaMalloc((void**)&d_positions, n_tokens * sizeof(int32_t)));
    if (st != GOLLMGO_OK) { cudaFree(d_token_ids); return st; }

    cudaMemcpy(d_token_ids, host_token_ids, n_tokens * sizeof(int32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_positions, host_positions, n_tokens * sizeof(int32_t), cudaMemcpyHostToDevice);

    /* ---- Embedding ---- */
    {
        int threads = (c.hidden_size < 256) ? c.hidden_size : 256;
        dim3 grid(n_tokens, (c.hidden_size + threads - 1) / threads);
        embedding_lookup_f16<<<grid, threads>>>(
            get_weight(m, "model.embed_tokens.weight"),
            d_token_ids,
            m->buf_hidden,
            c.hidden_size,
            c.vocab_size);
    }

    /* ---- Transformer layers ---- */
    for (int layer = 0; layer < c.num_layers; layer++) {
        char name_buf[256];

        /* -- Attention block -- */

        /* RMSNorm (input_layernorm). */
        snprintf(name_buf, sizeof(name_buf), "model.layers.%d.input_layernorm.weight", layer);
        __half* norm_weight = get_weight(m, name_buf);
        if (!norm_weight) {
            model_set_error(m, name_buf);
            cudaFree(d_token_ids); cudaFree(d_positions);
            return GOLLMGO_ERR_INVALID;
        }

        {
            int threads = (c.hidden_size < 256) ? c.hidden_size : 256;
            rmsnorm_f16<<<n_tokens, threads>>>(
                m->buf_hidden, norm_weight, m->buf_norm,
                c.hidden_size, n_tokens, c.rms_norm_eps);
        }

        /* Q, K, V projections via cuBLAS. */
        snprintf(name_buf, sizeof(name_buf), "model.layers.%d.self_attn.q_proj.weight", layer);
        __half* wq = get_weight(m, name_buf);
        snprintf(name_buf, sizeof(name_buf), "model.layers.%d.self_attn.k_proj.weight", layer);
        __half* wk = get_weight(m, name_buf);
        snprintf(name_buf, sizeof(name_buf), "model.layers.%d.self_attn.v_proj.weight", layer);
        __half* wv = get_weight(m, name_buf);

        if (!wq || !wk || !wv) {
            model_set_error(m, "missing QKV weights");
            cudaFree(d_token_ids); cudaFree(d_positions);
            return GOLLMGO_ERR_INVALID;
        }

        /* buf_norm[n, H] * wq[num_heads*head_dim, H]^T -> buf_q[n, num_heads*head_dim] */
        st = gemm_f16(m, m->buf_norm, wq, m->buf_q,
                       n_tokens, c.num_heads * c.head_dim, c.hidden_size);
        if (st != GOLLMGO_OK) { cudaFree(d_token_ids); cudaFree(d_positions); return st; }

        st = gemm_f16(m, m->buf_norm, wk, m->buf_k,
                       n_tokens, c.num_kv_heads * c.head_dim, c.hidden_size);
        if (st != GOLLMGO_OK) { cudaFree(d_token_ids); cudaFree(d_positions); return st; }

        st = gemm_f16(m, m->buf_norm, wv, m->buf_v,
                       n_tokens, c.num_kv_heads * c.head_dim, c.hidden_size);
        if (st != GOLLMGO_OK) { cudaFree(d_token_ids); cudaFree(d_positions); return st; }

        /* RoPE. */
        {
            int max_heads = (c.num_heads > c.num_kv_heads) ? c.num_heads : c.num_kv_heads;
            dim3 grid(n_tokens, max_heads);
            int threads = c.head_dim / 2;
            rope_f16<<<grid, threads>>>(
                m->buf_q, m->buf_k, d_positions,
                n_tokens, c.num_heads, c.num_kv_heads, c.head_dim,
                10000.0f);
        }

        /* Naive attention. */
        {
            dim3 grid(n_tokens, c.num_heads);
            int threads = (c.head_dim < 32) ? c.head_dim : 32;
            int smem = n_tokens * sizeof(float);
            naive_attention_f16<<<grid, threads, smem>>>(
                m->buf_q, m->buf_k, m->buf_v, m->buf_attn_out,
                d_positions,
                n_tokens, c.num_heads, c.num_kv_heads, c.head_dim,
                1.0f / sqrtf((float)c.head_dim));
        }

        /* Output projection: attn_out[n, num_heads*head_dim] * wo -> proj[n, hidden_size]. */
        snprintf(name_buf, sizeof(name_buf), "model.layers.%d.self_attn.o_proj.weight", layer);
        __half* wo = get_weight(m, name_buf);
        if (!wo) {
            model_set_error(m, "missing o_proj weight");
            cudaFree(d_token_ids); cudaFree(d_positions);
            return GOLLMGO_ERR_INVALID;
        }

        st = gemm_f16(m, m->buf_attn_out, wo, m->buf_proj,
                       n_tokens, c.hidden_size, c.num_heads * c.head_dim);
        if (st != GOLLMGO_OK) { cudaFree(d_token_ids); cudaFree(d_positions); return st; }

        /* Residual add: hidden = hidden + proj. */
        {
            int total = n_tokens * c.hidden_size;
            int threads = 256;
            int blocks = (total + threads - 1) / threads;
            residual_add_f16<<<blocks, threads>>>(
                m->buf_hidden, m->buf_proj, m->buf_hidden, total);
        }

        /* -- FFN block -- */

        /* RMSNorm (post_attention_layernorm). */
        snprintf(name_buf, sizeof(name_buf), "model.layers.%d.post_attention_layernorm.weight", layer);
        norm_weight = get_weight(m, name_buf);
        if (!norm_weight) {
            model_set_error(m, name_buf);
            cudaFree(d_token_ids); cudaFree(d_positions);
            return GOLLMGO_ERR_INVALID;
        }

        {
            int threads = (c.hidden_size < 256) ? c.hidden_size : 256;
            rmsnorm_f16<<<n_tokens, threads>>>(
                m->buf_hidden, norm_weight, m->buf_norm,
                c.hidden_size, n_tokens, c.rms_norm_eps);
        }

        /* Gate and up projections. */
        snprintf(name_buf, sizeof(name_buf), "model.layers.%d.mlp.gate_proj.weight", layer);
        __half* w_gate = get_weight(m, name_buf);
        snprintf(name_buf, sizeof(name_buf), "model.layers.%d.mlp.up_proj.weight", layer);
        __half* w_up = get_weight(m, name_buf);

        if (!w_gate || !w_up) {
            model_set_error(m, "missing gate/up weights");
            cudaFree(d_token_ids); cudaFree(d_positions);
            return GOLLMGO_ERR_INVALID;
        }

        st = gemm_f16(m, m->buf_norm, w_gate, m->buf_gate,
                       n_tokens, c.intermediate_size, c.hidden_size);
        if (st != GOLLMGO_OK) { cudaFree(d_token_ids); cudaFree(d_positions); return st; }

        st = gemm_f16(m, m->buf_norm, w_up, m->buf_up,
                       n_tokens, c.intermediate_size, c.hidden_size);
        if (st != GOLLMGO_OK) { cudaFree(d_token_ids); cudaFree(d_positions); return st; }

        /* SiLU(gate) * up. */
        {
            int total = n_tokens * c.intermediate_size;
            int threads = 256;
            int blocks = (total + threads - 1) / threads;
            silu_mul_f16<<<blocks, threads>>>(
                m->buf_gate, m->buf_up, m->buf_ffn, total);
        }

        /* Down projection. */
        snprintf(name_buf, sizeof(name_buf), "model.layers.%d.mlp.down_proj.weight", layer);
        __half* w_down = get_weight(m, name_buf);
        if (!w_down) {
            model_set_error(m, "missing down_proj weight");
            cudaFree(d_token_ids); cudaFree(d_positions);
            return GOLLMGO_ERR_INVALID;
        }

        st = gemm_f16(m, m->buf_ffn, w_down, m->buf_ffn_out,
                       n_tokens, c.hidden_size, c.intermediate_size);
        if (st != GOLLMGO_OK) { cudaFree(d_token_ids); cudaFree(d_positions); return st; }

        /* Residual add: hidden = hidden + ffn_out. */
        {
            int total = n_tokens * c.hidden_size;
            int threads = 256;
            int blocks = (total + threads - 1) / threads;
            residual_add_f16<<<blocks, threads>>>(
                m->buf_hidden, m->buf_ffn_out, m->buf_hidden, total);
        }
    }

    /* ---- Final RMSNorm ---- */
    {
        int threads = (c.hidden_size < 256) ? c.hidden_size : 256;
        rmsnorm_f16<<<n_tokens, threads>>>(
            m->buf_hidden, get_weight(m, "model.norm.weight"), m->buf_norm,
            c.hidden_size, n_tokens, c.rms_norm_eps);
    }

    /* ---- LM head: norm[n, hidden] * lm_head[vocab, hidden]^T -> logits[n, vocab] ---- */
    st = gemm_f16(m, m->buf_norm, get_weight(m, "lm_head.weight"), m->buf_logits_f16,
                   n_tokens, c.vocab_size, c.hidden_size);
    if (st != GOLLMGO_OK) { cudaFree(d_token_ids); cudaFree(d_positions); return st; }

    /* Convert F16 logits to F32 on device, then copy to host. */
    {
        int total = n_tokens * c.vocab_size;
        int threads = 256;
        int blocks = (total + threads - 1) / threads;
        f16_to_f32<<<blocks, threads>>>(m->buf_logits_f16, m->buf_logits_f32, total);
    }

    st = model_check_cuda(m, cudaMemcpy(
        host_logits_out, m->buf_logits_f32,
        n_tokens * c.vocab_size * sizeof(float),
        cudaMemcpyDeviceToHost));

    cudaFree(d_token_ids);
    cudaFree(d_positions);
    return st;
}

gollmgo_status_t gollmgo_model_destroy(gollmgo_model_t m) {
    if (!m) return GOLLMGO_OK;

    /* Free all weight tensors. */
    for (auto& kv : m->weights) {
        cudaFree(kv.second.data);
    }

    /* Free scratch buffers. */
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
    cudaFree(m->buf_logits_f16);
    cudaFree(m->buf_logits_f32);

    if (m->cublas) {
        cublasDestroy(m->cublas);
    }

    delete m;
    return GOLLMGO_OK;
}

} /* extern "C" */
