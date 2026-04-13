/*
 * gollmgo_paged_attn.cu — Paged Attention v1 implementation.
 *
 * Correctness-first implementation for M6.
 * Performance optimizations (shared memory tiling, vectorized loads) deferred.
 */

#include "gollmgo_paged_attn.cuh"
#include <cuda_fp16.h>
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
