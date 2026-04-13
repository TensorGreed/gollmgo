/*
 * gollmgo_ops.cuh — CUDA kernel declarations for transformer operations.
 * Internal header; not exposed through C API.
 */

#ifndef GOLLMGO_OPS_CUH
#define GOLLMGO_OPS_CUH

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

/* ======== FP16 kernels ======== */

/* ---- Embedding lookup ---- */
__global__ void embedding_lookup_f16(
    const __half* __restrict__ table,
    const int32_t* __restrict__ ids,
    __half* __restrict__ out,
    int hidden_size,
    int vocab_size);

/* ---- RMS Norm ---- */
__global__ void rmsnorm_f16(
    const __half* __restrict__ x,
    const __half* __restrict__ weight,
    __half* __restrict__ out,
    int hidden_size,
    int n,
    float eps);

/* ---- RoPE ---- */
__global__ void rope_f16(
    __half* __restrict__ q,
    __half* __restrict__ k,
    const int32_t* __restrict__ positions,
    int n,
    int num_heads,
    int num_kv_heads,
    int head_dim,
    float theta_base);

/* ---- SiLU activation ---- */
__global__ void silu_mul_f16(
    const __half* __restrict__ gate,
    const __half* __restrict__ up,
    __half* __restrict__ out,
    int total_elements);

/* ---- Naive attention ---- */
__global__ void naive_attention_f16(
    const __half* __restrict__ q,
    const __half* __restrict__ k,
    const __half* __restrict__ v,
    __half* __restrict__ out,
    const int32_t* __restrict__ positions,
    int n,
    int num_heads,
    int num_kv_heads,
    int head_dim,
    float scale);

/* ---- Residual add ---- */
__global__ void residual_add_f16(
    const __half* __restrict__ a,
    const __half* __restrict__ b,
    __half* __restrict__ out,
    int total_elements);

/* ---- F16 to F32 ---- */
__global__ void f16_to_f32(
    const __half* __restrict__ in,
    float* __restrict__ out,
    int total_elements);

/* ======== BF16 kernels ======== */

__global__ void embedding_lookup_bf16(
    const __nv_bfloat16* __restrict__ table,
    const int32_t* __restrict__ ids,
    __nv_bfloat16* __restrict__ out,
    int hidden_size,
    int vocab_size);

__global__ void rmsnorm_bf16(
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ weight,
    __nv_bfloat16* __restrict__ out,
    int hidden_size,
    int n,
    float eps);

__global__ void rope_bf16(
    __nv_bfloat16* __restrict__ q,
    __nv_bfloat16* __restrict__ k,
    const int32_t* __restrict__ positions,
    int n,
    int num_heads,
    int num_kv_heads,
    int head_dim,
    float theta_base);

__global__ void silu_mul_bf16(
    const __nv_bfloat16* __restrict__ gate,
    const __nv_bfloat16* __restrict__ up,
    __nv_bfloat16* __restrict__ out,
    int total_elements);

__global__ void naive_attention_bf16(
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ v,
    __nv_bfloat16* __restrict__ out,
    const int32_t* __restrict__ positions,
    int n,
    int num_heads,
    int num_kv_heads,
    int head_dim,
    float scale);

__global__ void residual_add_bf16(
    const __nv_bfloat16* __restrict__ a,
    const __nv_bfloat16* __restrict__ b,
    __nv_bfloat16* __restrict__ out,
    int total_elements);

__global__ void bf16_to_f32(
    const __nv_bfloat16* __restrict__ in,
    float* __restrict__ out,
    int total_elements);

/* ---- F32 to half-precision ---- */
__global__ void f32_to_f16(
    const float* __restrict__ in,
    __half* __restrict__ out,
    int total_elements);

__global__ void f32_to_bf16(
    const float* __restrict__ in,
    __nv_bfloat16* __restrict__ out,
    int total_elements);

/* ---- Cross-dtype conversions ---- */
__global__ void f16_to_bf16(
    const __half* __restrict__ in,
    __nv_bfloat16* __restrict__ out,
    int total_elements);

__global__ void bf16_to_f16(
    const __nv_bfloat16* __restrict__ in,
    __half* __restrict__ out,
    int total_elements);

/* ======== Mixed-precision kernels (FP32 hidden state) ======== */

/* Embedding: half-precision table → FP32 output */
__global__ void embedding_f16_to_f32(
    const __half* __restrict__ table,
    const int32_t* __restrict__ ids,
    float* __restrict__ out,
    int hidden_size, int vocab_size);

__global__ void embedding_bf16_to_f32(
    const __nv_bfloat16* __restrict__ table,
    const int32_t* __restrict__ ids,
    float* __restrict__ out,
    int hidden_size, int vocab_size);

/* RMSNorm: FP32 input, half-precision weight → half-precision output (for GEMM) */
__global__ void rmsnorm_f32in_f16w(
    const float* __restrict__ x,
    const __half* __restrict__ weight,
    __half* __restrict__ out,
    int hidden_size, int n, float eps);

__global__ void rmsnorm_f32in_bf16w(
    const float* __restrict__ x,
    const __nv_bfloat16* __restrict__ weight,
    __nv_bfloat16* __restrict__ out,
    int hidden_size, int n, float eps);

/* Residual add: FP32 accumulator += half-precision input → FP32 output */
__global__ void residual_add_f32_f16(
    float* __restrict__ acc,
    const __half* __restrict__ b,
    int total_elements);

__global__ void residual_add_f32_bf16(
    float* __restrict__ acc,
    const __nv_bfloat16* __restrict__ b,
    int total_elements);

#endif /* GOLLMGO_OPS_CUH */
