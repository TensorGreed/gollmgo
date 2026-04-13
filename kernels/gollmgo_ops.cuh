/*
 * gollmgo_ops.cuh — CUDA kernel declarations for transformer operations.
 * Internal header; not exposed through C API.
 */

#ifndef GOLLMGO_OPS_CUH
#define GOLLMGO_OPS_CUH

#include <cuda_fp16.h>
#include <cuda_runtime.h>

/* ---- Embedding lookup ---- */
/* out[i] = table[ids[i]] for i in [0, n) */
__global__ void embedding_lookup_f16(
    const __half* __restrict__ table,   /* [vocab_size, hidden_size] */
    const int32_t* __restrict__ ids,    /* [n] */
    __half* __restrict__ out,           /* [n, hidden_size] */
    int hidden_size,
    int vocab_size);

/* ---- RMS Norm ---- */
/* out = (x / rms(x)) * weight, rms(x) = sqrt(mean(x^2) + eps) */
__global__ void rmsnorm_f16(
    const __half* __restrict__ x,       /* [n, hidden_size] */
    const __half* __restrict__ weight,  /* [hidden_size] */
    __half* __restrict__ out,           /* [n, hidden_size] */
    int hidden_size,
    int n,
    float eps);

/* ---- RoPE (Rotary Position Embedding) ---- */
/* Apply RoPE in-place to q and k. */
__global__ void rope_f16(
    __half* __restrict__ q,             /* [n, num_heads, head_dim] */
    __half* __restrict__ k,             /* [n, num_kv_heads, head_dim] */
    const int32_t* __restrict__ positions, /* [n] */
    int n,
    int num_heads,
    int num_kv_heads,
    int head_dim,
    float theta_base);

/* ---- SiLU activation ---- */
/* out = silu(gate) * up = (gate * sigmoid(gate)) * up */
__global__ void silu_mul_f16(
    const __half* __restrict__ gate,    /* [n, intermediate_size] */
    const __half* __restrict__ up,      /* [n, intermediate_size] */
    __half* __restrict__ out,           /* [n, intermediate_size] */
    int total_elements);

/* ---- Naive multi-head attention (no paging) ---- */
/*
 * Full QKV attention with causal mask. No KV cache — recomputes everything.
 * This is correct but slow; M6 replaces this with paged attention.
 */
__global__ void naive_attention_f16(
    const __half* __restrict__ q,       /* [n, num_heads, head_dim] */
    const __half* __restrict__ k,       /* [n, num_kv_heads, head_dim] */
    const __half* __restrict__ v,       /* [n, num_kv_heads, head_dim] */
    __half* __restrict__ out,           /* [n, num_heads, head_dim] */
    const int32_t* __restrict__ positions, /* [n] */
    int n,
    int num_heads,
    int num_kv_heads,
    int head_dim,
    float scale);

/* ---- Residual add ---- */
/* out = a + b, element-wise */
__global__ void residual_add_f16(
    const __half* __restrict__ a,
    const __half* __restrict__ b,
    __half* __restrict__ out,
    int total_elements);

/* ---- F16 to F32 copy (for logits output) ---- */
__global__ void f16_to_f32(
    const __half* __restrict__ in,
    float* __restrict__ out,
    int total_elements);

#endif /* GOLLMGO_OPS_CUH */
