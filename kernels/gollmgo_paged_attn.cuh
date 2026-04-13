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
#include <cuda_runtime.h>
#include <stdint.h>

/*
 * Write K and V for new tokens into the paged cache.
 * k_cache/v_cache: [num_slots, num_kv_heads, head_dim]
 * k/v:             [n_tokens, num_kv_heads, head_dim]
 * slot_mapping:    [n_tokens] — physical slot index per token
 */
__global__ void paged_kv_write_f16(
    __half* __restrict__ k_cache,
    __half* __restrict__ v_cache,
    const __half* __restrict__ k,
    const __half* __restrict__ v,
    const int32_t* __restrict__ slot_mapping,
    int n_tokens,
    int num_kv_heads,
    int head_dim);

/*
 * Paged Attention v1 — decode-phase attention.
 *
 * For each query token, compute attention over all cached KV slots
 * indicated by slot_mapping_full (the full sequence's slot map).
 *
 * q:                 [n_queries, num_heads, head_dim]
 * k_cache/v_cache:   [num_slots, num_kv_heads, head_dim]
 * out:               [n_queries, num_heads, head_dim]
 * seq_lens:          [n_queries] — how many cached KV tokens per query
 * slot_tables:       [n_queries, max_seq_len] — slot mapping per query
 *                    (padded to max_seq_len; only seq_lens[i] entries valid)
 * max_seq_len:       max over seq_lens
 */
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

#endif /* GOLLMGO_PAGED_ATTN_CUH */
