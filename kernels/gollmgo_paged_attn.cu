/*
 * gollmgo_paged_attn.cu — Paged Attention v1 implementation.
 *
 * Correctness-first implementation for M6.
 * Performance optimizations (shared memory tiling, vectorized loads) deferred.
 */

#include "gollmgo_paged_attn.cuh"
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <math.h>

/* ---- KV cache write ---- */
__global__ void paged_kv_write_f16(
    __half* __restrict__ k_cache,
    __half* __restrict__ v_cache,
    const __half* __restrict__ k,
    const __half* __restrict__ v,
    const int32_t* __restrict__ slot_mapping,
    int n_tokens,
    int num_kv_heads,
    int head_dim)
{
    int tok_idx = blockIdx.x;
    if (tok_idx >= n_tokens) return;

    int slot = slot_mapping[tok_idx];
    int kv_size = num_kv_heads * head_dim;

    /* Copy K and V for this token to the cache slot. */
    for (int i = threadIdx.x; i < kv_size; i += blockDim.x) {
        k_cache[slot * kv_size + i] = k[tok_idx * kv_size + i];
        v_cache[slot * kv_size + i] = v[tok_idx * kv_size + i];
    }
}

/* ---- Paged Attention v1 ---- */
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
    float scale)
{
    /* One block per (query, head) pair. */
    int query_idx = blockIdx.x;
    int head_idx = blockIdx.y;
    if (query_idx >= n_queries || head_idx >= num_heads) return;

    /* GQA: map query head to KV head. */
    int kv_group_size = num_heads / num_kv_heads;
    int kv_head_idx = head_idx / kv_group_size;
    int kv_stride = num_kv_heads * head_dim;

    int seq_len = seq_lens[query_idx];
    const int32_t* slots = slot_tables + query_idx * max_seq_len;
    const __half* q_vec = q + (query_idx * num_heads + head_idx) * head_dim;
    __half* out_vec = out + (query_idx * num_heads + head_idx) * head_dim;

    /* Shared memory for attention scores. */
    extern __shared__ float smem[];
    float* scores = smem;

    /* Phase 1: compute Q·K^T scores for all cached positions. */
    float max_score = -1e30f;
    for (int j = 0; j < seq_len; j++) {
        int slot = slots[j];
        const __half* k_vec = k_cache + slot * kv_stride + kv_head_idx * head_dim;

        float dot = 0.0f;
        for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
            dot += __half2float(q_vec[d]) * __half2float(k_vec[d]);
        }

        /* Warp reduction. */
        for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
            dot += __shfl_down_sync(0xffffffff, dot, offset);
        }

        if (threadIdx.x == 0) {
            scores[j] = dot * scale;
            if (scores[j] > max_score) max_score = scores[j];
        }
        __syncthreads();
    }

    /* Phase 2: softmax over scores. */
    if (threadIdx.x == 0) {
        float sum_exp = 0.0f;
        for (int j = 0; j < seq_len; j++) {
            scores[j] = expf(scores[j] - max_score);
            sum_exp += scores[j];
        }
        float inv_sum = (sum_exp > 0.0f) ? 1.0f / sum_exp : 0.0f;
        for (int j = 0; j < seq_len; j++) {
            scores[j] *= inv_sum;
        }
    }
    __syncthreads();

    /* Phase 3: weighted sum of V from cache. */
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        float acc = 0.0f;
        for (int j = 0; j < seq_len; j++) {
            int slot = slots[j];
            float v_val = __half2float(
                v_cache[slot * kv_stride + kv_head_idx * head_dim + d]);
            acc += scores[j] * v_val;
        }
        out_vec[d] = __float2half(acc);
    }
}

/* ================================================================
 * BF16 paged attention variants
 * ================================================================ */

/* ---- KV cache write (BF16) ---- */
__global__ void paged_kv_write_bf16(
    __nv_bfloat16* __restrict__ k_cache,
    __nv_bfloat16* __restrict__ v_cache,
    const __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ v,
    const int32_t* __restrict__ slot_mapping,
    int n_tokens,
    int num_kv_heads,
    int head_dim)
{
    int tok_idx = blockIdx.x;
    if (tok_idx >= n_tokens) return;

    int slot = slot_mapping[tok_idx];
    int kv_size = num_kv_heads * head_dim;

    for (int i = threadIdx.x; i < kv_size; i += blockDim.x) {
        k_cache[slot * kv_size + i] = k[tok_idx * kv_size + i];
        v_cache[slot * kv_size + i] = v[tok_idx * kv_size + i];
    }
}

/* ---- Paged Attention v1 (BF16) ---- */
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
    float scale)
{
    int query_idx = blockIdx.x;
    int head_idx = blockIdx.y;
    if (query_idx >= n_queries || head_idx >= num_heads) return;

    int kv_group_size = num_heads / num_kv_heads;
    int kv_head_idx = head_idx / kv_group_size;
    int kv_stride = num_kv_heads * head_dim;

    int seq_len = seq_lens[query_idx];
    const int32_t* slots = slot_tables + query_idx * max_seq_len;
    const __nv_bfloat16* q_vec = q + (query_idx * num_heads + head_idx) * head_dim;
    __nv_bfloat16* out_vec = out + (query_idx * num_heads + head_idx) * head_dim;

    extern __shared__ float smem[];
    float* scores = smem;

    float max_score = -1e30f;
    for (int j = 0; j < seq_len; j++) {
        int slot = slots[j];
        const __nv_bfloat16* k_vec = k_cache + slot * kv_stride + kv_head_idx * head_dim;

        float dot = 0.0f;
        for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
            dot += __bfloat162float(q_vec[d]) * __bfloat162float(k_vec[d]);
        }

        for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
            dot += __shfl_down_sync(0xffffffff, dot, offset);
        }

        if (threadIdx.x == 0) {
            scores[j] = dot * scale;
            if (scores[j] > max_score) max_score = scores[j];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        float sum_exp = 0.0f;
        for (int j = 0; j < seq_len; j++) {
            scores[j] = expf(scores[j] - max_score);
            sum_exp += scores[j];
        }
        float inv_sum = (sum_exp > 0.0f) ? 1.0f / sum_exp : 0.0f;
        for (int j = 0; j < seq_len; j++) {
            scores[j] *= inv_sum;
        }
    }
    __syncthreads();

    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        float acc = 0.0f;
        for (int j = 0; j < seq_len; j++) {
            int slot = slots[j];
            float v_val = __bfloat162float(
                v_cache[slot * kv_stride + kv_head_idx * head_dim + d]);
            acc += scores[j] * v_val;
        }
        out_vec[d] = __float2bfloat16(acc);
    }
}

/* ================================================================
 * PagedAttention v2 — partitioned for long-context scalability.
 *
 * Phase 1: Each block processes one partition of the KV sequence.
 *          Computes local max, local exp_sum, and partial weighted V.
 * Phase 2: Reduces across partitions using log-sum-exp correction.
 * ================================================================ */

/* Macro-based phase1 to avoid FP16/BF16 duplication. */
#define PAGED_ATTN_V2_PHASE1(SUFFIX, HALF_T, TO_FLOAT, FROM_FLOAT)            \
__global__ void paged_attention_v2_phase1_##SUFFIX(                            \
    const HALF_T* __restrict__ q,                                              \
    const HALF_T* __restrict__ k_cache,                                        \
    const HALF_T* __restrict__ v_cache,                                        \
    float* __restrict__ exp_sums,                                              \
    float* __restrict__ max_logits_out,                                        \
    float* __restrict__ partial_out,                                           \
    const int32_t* __restrict__ seq_lens,                                      \
    const int32_t* __restrict__ slot_tables,                                   \
    int n_queries, int num_heads, int num_kv_heads,                            \
    int head_dim, int max_seq_len, float scale,                                \
    int partition_size)                                                         \
{                                                                              \
    int query_idx = blockIdx.x;                                                \
    int head_idx  = blockIdx.y;                                                \
    int part_idx  = blockIdx.z;                                                \
    if (query_idx >= n_queries || head_idx >= num_heads) return;                \
                                                                               \
    int kv_group_size = num_heads / num_kv_heads;                              \
    int kv_head_idx = head_idx / kv_group_size;                                \
    int kv_stride = num_kv_heads * head_dim;                                   \
                                                                               \
    int seq_len = seq_lens[query_idx];                                         \
    int part_start = part_idx * partition_size;                                 \
    int part_end = part_start + partition_size;                                 \
    if (part_end > seq_len) part_end = seq_len;                                \
    if (part_start >= seq_len) {                                               \
        /* Empty partition. */                                                 \
        int num_parts = (seq_len + partition_size - 1) / partition_size;        \
        int flat = (query_idx * num_heads + head_idx) * num_parts + part_idx;  \
        if (threadIdx.x == 0) {                                                \
            max_logits_out[flat] = -1e30f;                                     \
            exp_sums[flat] = 0.0f;                                             \
        }                                                                      \
        return;                                                                \
    }                                                                          \
    int part_len = part_end - part_start;                                      \
                                                                               \
    const int32_t* slots = slot_tables + query_idx * max_seq_len;              \
    const HALF_T* q_vec = q + (query_idx * num_heads + head_idx) * head_dim;   \
                                                                               \
    extern __shared__ float smem[];                                            \
    float* scores = smem;                                                      \
                                                                               \
    /* Phase 1a: QK dot products for this partition. */                         \
    float local_max = -1e30f;                                                  \
    for (int j = 0; j < part_len; j++) {                                       \
        int slot = slots[part_start + j];                                      \
        const HALF_T* k_vec = k_cache + slot * kv_stride + kv_head_idx * head_dim; \
        float dot = 0.0f;                                                      \
        for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {             \
            dot += TO_FLOAT(q_vec[d]) * TO_FLOAT(k_vec[d]);                    \
        }                                                                      \
        for (int off = warpSize/2; off > 0; off >>= 1)                         \
            dot += __shfl_down_sync(0xffffffff, dot, off);                      \
        if (threadIdx.x == 0) {                                                \
            scores[j] = dot * scale;                                           \
            if (scores[j] > local_max) local_max = scores[j];                  \
        }                                                                      \
        __syncthreads();                                                       \
    }                                                                          \
                                                                               \
    /* Phase 1b: local softmax. */                                             \
    float local_sum = 0.0f;                                                    \
    if (threadIdx.x == 0) {                                                    \
        for (int j = 0; j < part_len; j++) {                                   \
            scores[j] = expf(scores[j] - local_max);                           \
            local_sum += scores[j];                                            \
        }                                                                      \
    }                                                                          \
    __syncthreads();                                                           \
                                                                               \
    /* Phase 1c: partial weighted V sum. */                                    \
    int num_parts = (seq_len + partition_size - 1) / partition_size;            \
    int flat = (query_idx * num_heads + head_idx) * num_parts + part_idx;      \
    float* p_out = partial_out + (size_t)flat * head_dim;                      \
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {                 \
        float acc = 0.0f;                                                      \
        for (int j = 0; j < part_len; j++) {                                   \
            int slot = slots[part_start + j];                                  \
            float v_val = TO_FLOAT(                                            \
                v_cache[slot * kv_stride + kv_head_idx * head_dim + d]);        \
            acc += scores[j] * v_val;                                          \
        }                                                                      \
        p_out[d] = acc;                                                        \
    }                                                                          \
    if (threadIdx.x == 0) {                                                    \
        max_logits_out[flat] = local_max;                                      \
        exp_sums[flat] = local_sum;                                            \
    }                                                                          \
}

PAGED_ATTN_V2_PHASE1(f16, __half, __half2float, __float2half)
PAGED_ATTN_V2_PHASE1(bf16, __nv_bfloat16, __bfloat162float, __float2bfloat16)

/* Phase 2: reduce partitions → final output (dtype-agnostic, all FP32). */
__global__ void paged_attention_v2_reduce(
    const float* __restrict__ exp_sums,
    const float* __restrict__ max_logits,
    const float* __restrict__ partial_out,
    float* __restrict__ out,
    const int32_t* __restrict__ seq_lens,
    int n_queries, int num_heads, int head_dim,
    int partition_size, int max_num_partitions)
{
    int query_idx = blockIdx.x;
    int head_idx  = blockIdx.y;
    if (query_idx >= n_queries || head_idx >= num_heads) return;

    int seq_len = seq_lens[query_idx];
    int num_parts = (seq_len + partition_size - 1) / partition_size;
    if (num_parts <= 0) return;

    int base = (query_idx * num_heads + head_idx) * max_num_partitions;

    /* Find global max across partitions. */
    float global_max = -1e30f;
    for (int p = 0; p < num_parts; p++) {
        float m = max_logits[base + p];
        if (m > global_max) global_max = m;
    }

    /* Compute per-partition correction factors.
     * partial_out stores unnormalized weighted sums: sum(exp(s-local_max)*v).
     * To combine: final = sum_p(correction_p * partial_out_p) / sum_p(correction_p * exp_sum_p)
     * where correction_p = exp(local_max_p - global_max). */
    extern __shared__ float smem[];
    float* corrections = smem; /* [max_num_partitions] */
    float total_sum = 0.0f;
    if (threadIdx.x == 0) {
        for (int p = 0; p < num_parts; p++) {
            corrections[p] = expf(max_logits[base + p] - global_max);
            total_sum += exp_sums[base + p] * corrections[p];
        }
    }
    __syncthreads();

    /* Weighted combination of partial outputs. */
    float inv_total = (total_sum > 0.0f) ? 1.0f / total_sum : 0.0f;
    float* out_vec = out + (query_idx * num_heads + head_idx) * head_dim;
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        float acc = 0.0f;
        for (int p = 0; p < num_parts; p++) {
            const float* p_out = partial_out +
                (size_t)((query_idx * num_heads + head_idx) * max_num_partitions + p) * head_dim;
            acc += corrections[p] * p_out[d];
        }
        out_vec[d] = acc * inv_total;
    }
}
