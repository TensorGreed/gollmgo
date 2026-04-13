/*
 * gollmgo_model.h — Model loading and forward pass C API.
 *
 * This extends the backend C API with model weight management
 * and eager (non-paged) transformer forward pass execution.
 */

#ifndef GOLLMGO_MODEL_H
#define GOLLMGO_MODEL_H

#include "gollmgo_backend.h"
#include "gollmgo_kvcache.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Opaque model handle. */
typedef struct gollmgo_model* gollmgo_model_t;

/* Compute dtype constants. */
#define GOLLMGO_DTYPE_FP16  0
#define GOLLMGO_DTYPE_BF16  1

/* Model architecture config. */
typedef struct {
    int num_layers;
    int hidden_size;
    int intermediate_size;
    int num_heads;
    int num_kv_heads;
    int vocab_size;
    int max_seq_len;
    int head_dim;          /* hidden_size / num_heads */
    float rms_norm_eps;
    int dtype;             /* GOLLMGO_DTYPE_FP16 or GOLLMGO_DTYPE_BF16 */
} gollmgo_model_config_t;

/* Create a model with the given config. Allocates GPU memory for weights. */
gollmgo_status_t gollmgo_model_create(gollmgo_backend_t b,
                                       const gollmgo_model_config_t* config,
                                       gollmgo_model_t* out);

/*
 * Load a weight tensor by name. Data is copied from host to device.
 * Supported dtypes: "F16", "BF16", "F32".
 * The name follows safetensors conventions (e.g. "model.embed_tokens.weight").
 */
gollmgo_status_t gollmgo_model_load_weight(gollmgo_model_t m,
                                            const char* name,
                                            const void* host_data,
                                            int64_t size_bytes,
                                            const char* dtype);

/* Mark model as ready after all weights are loaded. Validates completeness. */
gollmgo_status_t gollmgo_model_ready(gollmgo_model_t m);

/*
 * Execute one eager forward pass (no paged attention).
 *
 * token_ids:    [n_tokens] input token IDs
 * positions:    [n_tokens] position index for each token
 * n_tokens:     number of tokens in this step
 * logits_out:   [n_tokens * vocab_size] output logits (caller-allocated, host memory)
 *
 * For decode steps, typically n_tokens == batch_size (one new token per sequence).
 * For prefill, n_tokens == total prompt tokens.
 *
 * This is eager mode: full attention recomputation, no KV cache.
 * Paged attention will replace this in M6.
 */
gollmgo_status_t gollmgo_model_forward(gollmgo_backend_t b,
                                        gollmgo_model_t m,
                                        const int32_t* token_ids,
                                        const int32_t* positions,
                                        int n_tokens,
                                        float* logits_out);

/*
 * Prefill forward pass: eager attention + KV cache write.
 * Same as gollmgo_model_forward but also writes computed K/V
 * to the paged cache for subsequent decode steps.
 * slot_mapping: [n_tokens] physical cache slot for each token.
 * kv_cache: the paged KV cache handle.
 * If kv_cache is NULL, behaves identically to gollmgo_model_forward.
 */
gollmgo_status_t gollmgo_model_forward_prefill(
    gollmgo_backend_t b,
    gollmgo_model_t m,
    const int32_t* token_ids,
    const int32_t* positions,
    const int32_t* slot_mapping,
    int n_tokens,
    gollmgo_kvcache_t kv_cache,
    float* logits_out);

/*
 * Execute one forward pass with paged KV cache.
 *
 * This replaces gollmgo_model_forward for production serving.
 * The caller provides KV cache pointers and per-token slot mappings
 * from the Go-side block table manager.
 *
 * k_cache/v_cache:  [num_slots, num_kv_heads, head_dim] device pointers
 * slot_mapping:     [n_tokens] physical slot for each input token
 * seq_lens:         [n_seqs] total cached KV length per sequence (for attention)
 * slot_tables:      [n_seqs * max_context_len] full slot table per sequence
 * n_seqs:           number of sequences in this batch
 * max_context_len:  padding dimension for slot_tables
 * seq_token_counts: [n_seqs] how many tokens this sequence contributes to the batch
 *
 * Logits are returned only for the last token of each sequence.
 * logits_out:       [n_seqs * vocab_size] host memory
 */
gollmgo_status_t gollmgo_model_forward_paged(
    gollmgo_backend_t b,
    gollmgo_model_t m,
    const int32_t* token_ids,
    const int32_t* positions,
    const int32_t* slot_mapping,
    int n_tokens,
    gollmgo_kvcache_t kv_cache,
    const int32_t* seq_lens,
    const int32_t* slot_tables,
    int n_seqs,
    int max_context_len,
    const int32_t* seq_token_counts,
    float* logits_out);

/* Destroy model and free all GPU memory. */
gollmgo_status_t gollmgo_model_destroy(gollmgo_model_t m);

#ifdef __cplusplus
}
#endif

#endif /* GOLLMGO_MODEL_H */
