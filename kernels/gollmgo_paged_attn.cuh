/*
 * gollmgo_paged_attn.cuh — Paged Attention v1 kernel.
 *
 * Operates on KV cache stored in fixed-size blocks.
 * Each block holds `block_size` tokens of K and V data.
 * A slot mapping translates logical positions to physical slots.
 */

#ifndef GOLLMGO_PAGED_ATTN_CUH
#define GOLLMGO_PAGED_ATTN_CUH

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>

/* ======== FP16 ======== */

__global__ void paged_kv_write_f16(
    __half* __restrict__ k_cache,
    __half* __restrict__ v_cache,
    const __half* __restrict__ k,
    const __half* __restrict__ v,
    const int32_t* __restrict__ slot_mapping,
    int n_tokens,
    int num_kv_heads,
    int head_dim);

__global__ void paged_attention_v1_f16(
    const __half* __restrict__ q,
    const __half* __restrict__ k_cache,
    const __half* __restrict__ v_cache,
    __half* __restrict__ out,
    const int32_t* __restrict__ seq_lens,
    const int32_t* __restrict__ slot_tables,
    int n_queries,
    int num_heads,
    int num_kv_heads,
    int head_dim,
    int max_seq_len,
    float scale);

/* ======== BF16 ======== */

__global__ void paged_kv_write_bf16(
    __nv_bfloat16* __restrict__ k_cache,
    __nv_bfloat16* __restrict__ v_cache,
    const __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ v,
    const int32_t* __restrict__ slot_mapping,
    int n_tokens,
    int num_kv_heads,
    int head_dim);

__global__ void paged_attention_v1_bf16(
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k_cache,
    const __nv_bfloat16* __restrict__ v_cache,
    __nv_bfloat16* __restrict__ out,
    const int32_t* __restrict__ seq_lens,
    const int32_t* __restrict__ slot_tables,
    int n_queries,
    int num_heads,
    int num_kv_heads,
    int head_dim,
    int max_seq_len,
    float scale);

/* ======== PagedAttention v2 (partitioned, long-context) ======== */

#define PAGED_ATTN_V2_PARTITION_SIZE 256

/* Phase 1: per-partition partial attention.
 * Grid: (n_queries, num_heads, num_partitions). */
__global__ void paged_attention_v2_phase1_f16(
    const __half* __restrict__ q,
    const __half* __restrict__ k_cache,
    const __half* __restrict__ v_cache,
    float* __restrict__ exp_sums,
    float* __restrict__ max_logits,
    float* __restrict__ partial_out,
    const int32_t* __restrict__ seq_lens,
    const int32_t* __restrict__ slot_tables,
    int n_queries, int num_heads, int num_kv_heads,
    int head_dim, int max_seq_len, float scale,
    int partition_size);

__global__ void paged_attention_v2_phase1_bf16(
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k_cache,
    const __nv_bfloat16* __restrict__ v_cache,
    float* __restrict__ exp_sums,
    float* __restrict__ max_logits,
    float* __restrict__ partial_out,
    const int32_t* __restrict__ seq_lens,
    const int32_t* __restrict__ slot_tables,
    int n_queries, int num_heads, int num_kv_heads,
    int head_dim, int max_seq_len, float scale,
    int partition_size);

/* Phase 2: reduce across partitions (dtype-agnostic, all FP32). */
__global__ void paged_attention_v2_reduce(
    const float* __restrict__ exp_sums,
    const float* __restrict__ max_logits,
    const float* __restrict__ partial_out,
    float* __restrict__ out,
    const int32_t* __restrict__ seq_lens,
    int n_queries, int num_heads, int head_dim,
    int partition_size, int max_num_partitions);

#endif /* GOLLMGO_PAGED_ATTN_CUH */
