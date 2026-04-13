/*
 * gollmgo_kvcache.cu — KV cache GPU management + paged attention dispatch.
 */

#include "gollmgo_kvcache.h"
#include "gollmgo_paged_attn.cuh"
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdio>

struct gollmgo_kvcache {
    int num_slots;
    int num_kv_heads;
    int head_dim;
    __half* k_cache;   /* [num_slots, num_kv_heads, head_dim] */
    __half* v_cache;   /* [num_slots, num_kv_heads, head_dim] */
};

extern "C" {

gollmgo_status_t gollmgo_kvcache_create(gollmgo_backend_t b,
                                         int num_slots,
                                         int num_kv_heads,
                                         int head_dim,
                                         gollmgo_kvcache_t* out) {
    if (!b || !out || num_slots <= 0 || num_kv_heads <= 0 || head_dim <= 0)
        return GOLLMGO_ERR_INVALID;

    gollmgo_kvcache_t c = new gollmgo_kvcache();
    c->num_slots = num_slots;
    c->num_kv_heads = num_kv_heads;
    c->head_dim = head_dim;

    size_t cache_size = (size_t)num_slots * num_kv_heads * head_dim * sizeof(__half);

    cudaError_t err = cudaMalloc(&c->k_cache, cache_size);
    if (err != cudaSuccess) { delete c; return GOLLMGO_ERR_OOM; }

    err = cudaMalloc(&c->v_cache, cache_size);
    if (err != cudaSuccess) { cudaFree(c->k_cache); delete c; return GOLLMGO_ERR_OOM; }

    /* Zero-init the cache. */
    cudaMemset(c->k_cache, 0, cache_size);
    cudaMemset(c->v_cache, 0, cache_size);

    *out = c;
    return GOLLMGO_OK;
}

gollmgo_status_t gollmgo_kvcache_write(gollmgo_kvcache_t cache,
                                        const void* k_data,
                                        const void* v_data,
                                        const int32_t* slot_mapping,
                                        int n_tokens) {
    if (!cache || !k_data || !v_data || !slot_mapping || n_tokens <= 0)
        return GOLLMGO_ERR_INVALID;

    int kv_size = cache->num_kv_heads * cache->head_dim;

    /* Copy inputs to device. */
    __half *d_k, *d_v;
    int32_t* d_slots;
    size_t kv_bytes = (size_t)n_tokens * kv_size * sizeof(__half);

    cudaMalloc(&d_k, kv_bytes);
    cudaMalloc(&d_v, kv_bytes);
    cudaMalloc(&d_slots, n_tokens * sizeof(int32_t));

    cudaMemcpy(d_k, k_data, kv_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_v, v_data, kv_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_slots, slot_mapping, n_tokens * sizeof(int32_t), cudaMemcpyHostToDevice);

    int threads = (kv_size < 256) ? kv_size : 256;
    paged_kv_write_f16<<<n_tokens, threads>>>(
        cache->k_cache, cache->v_cache,
        d_k, d_v, d_slots,
        n_tokens, cache->num_kv_heads, cache->head_dim);

    cudaDeviceSynchronize();
    cudaFree(d_k);
    cudaFree(d_v);
    cudaFree(d_slots);

    return GOLLMGO_OK;
}

gollmgo_status_t gollmgo_kvcache_attention(gollmgo_kvcache_t cache,
                                            const void* q_data,
                                            void* out_data,
                                            const int32_t* seq_lens,
                                            const int32_t* slot_tables,
                                            int n_queries,
                                            int num_heads,
                                            int max_seq_len,
                                            float scale) {
    if (!cache || !q_data || !out_data || !seq_lens || !slot_tables)
        return GOLLMGO_ERR_INVALID;
    if (n_queries <= 0 || num_heads <= 0 || max_seq_len <= 0)
        return GOLLMGO_ERR_INVALID;

    int head_dim = cache->head_dim;
    int num_kv_heads = cache->num_kv_heads;

    /* Copy Q, seq_lens, slot_tables to device. */
    size_t q_bytes = (size_t)n_queries * num_heads * head_dim * sizeof(__half);
    size_t out_bytes = q_bytes;

    __half* d_q;
    __half* d_out;
    int32_t* d_seq_lens;
    int32_t* d_slot_tables;

    cudaMalloc(&d_q, q_bytes);
    cudaMalloc(&d_out, out_bytes);
    cudaMalloc(&d_seq_lens, n_queries * sizeof(int32_t));
    cudaMalloc(&d_slot_tables, (size_t)n_queries * max_seq_len * sizeof(int32_t));

    cudaMemcpy(d_q, q_data, q_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_seq_lens, seq_lens, n_queries * sizeof(int32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_slot_tables, slot_tables,
               (size_t)n_queries * max_seq_len * sizeof(int32_t), cudaMemcpyHostToDevice);

    dim3 grid(n_queries, num_heads);
    int threads = (head_dim < 32) ? head_dim : 32;
    size_t smem = max_seq_len * sizeof(float);

    paged_attention_v1_f16<<<grid, threads, smem>>>(
        d_q, cache->k_cache, cache->v_cache, d_out,
        d_seq_lens, d_slot_tables,
        n_queries, num_heads, num_kv_heads, head_dim,
        max_seq_len, scale);

    cudaDeviceSynchronize();

    /* Copy output back to host. */
    cudaMemcpy(out_data, d_out, out_bytes, cudaMemcpyDeviceToHost);

    cudaFree(d_q);
    cudaFree(d_out);
    cudaFree(d_seq_lens);
    cudaFree(d_slot_tables);

    return GOLLMGO_OK;
}

void* gollmgo_kvcache_k_ptr(gollmgo_kvcache_t cache) {
    return cache ? (void*)cache->k_cache : nullptr;
}

void* gollmgo_kvcache_v_ptr(gollmgo_kvcache_t cache) {
    return cache ? (void*)cache->v_cache : nullptr;
}

gollmgo_status_t gollmgo_kvcache_destroy(gollmgo_kvcache_t cache) {
    if (!cache) return GOLLMGO_OK;
    cudaFree(cache->k_cache);
    cudaFree(cache->v_cache);
    delete cache;
    return GOLLMGO_OK;
}

} /* extern "C" */
