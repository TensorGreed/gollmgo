/*
 * gollmgo_kvcache.h — KV cache C API for paged attention.
 */

#ifndef GOLLMGO_KVCACHE_H
#define GOLLMGO_KVCACHE_H

#include "gollmgo_backend.h"
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Opaque KV cache handle. */
typedef struct gollmgo_kvcache* gollmgo_kvcache_t;

/* Create a KV cache pool on the GPU.
 * num_slots:    total number of KV slots (num_blocks * block_size)
 * num_kv_heads: number of KV heads
 * head_dim:     dimension per head
 */
gollmgo_status_t gollmgo_kvcache_create(gollmgo_backend_t b,
                                         int num_slots,
                                         int num_kv_heads,
                                         int head_dim,
                                         gollmgo_kvcache_t* out);

/* Write K/V for new tokens into the cache.
 * k, v:          [n_tokens, num_kv_heads, head_dim] in FP16, host memory
 * slot_mapping:  [n_tokens] physical slot indices, host memory
 */
gollmgo_status_t gollmgo_kvcache_write(gollmgo_kvcache_t cache,
                                        const void* k_data,
                                        const void* v_data,
                                        const int32_t* slot_mapping,
                                        int n_tokens);

/* Run paged attention v1 for decode queries.
 * q:             [n_queries, num_heads, head_dim] FP16, host memory
 * out:           [n_queries, num_heads, head_dim] FP16, host memory (output)
 * seq_lens:      [n_queries] number of cached tokens per query
 * slot_tables:   [n_queries * max_seq_len] flattened slot tables
 * num_heads:     number of query heads
 * max_seq_len:   padding dimension for slot_tables
 */
gollmgo_status_t gollmgo_kvcache_attention(gollmgo_kvcache_t cache,
                                            const void* q_data,
                                            void* out_data,
                                            const int32_t* seq_lens,
                                            const int32_t* slot_tables,
                                            int n_queries,
                                            int num_heads,
                                            int max_seq_len,
                                            float scale);

/* Destroy the KV cache and free GPU memory. */
gollmgo_status_t gollmgo_kvcache_destroy(gollmgo_kvcache_t cache);

#ifdef __cplusplus
}
#endif

#endif /* GOLLMGO_KVCACHE_H */
