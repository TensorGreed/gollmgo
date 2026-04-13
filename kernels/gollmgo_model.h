/*
 * gollmgo_model.h — Model loading and forward pass C API.
 *
 * This extends the backend C API with model weight management
 * and eager (non-paged) transformer forward pass execution.
 */

#ifndef GOLLMGO_MODEL_H
#define GOLLMGO_MODEL_H

#include "gollmgo_backend.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Opaque model handle. */
typedef struct gollmgo_model* gollmgo_model_t;

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

/* Destroy model and free all GPU memory. */
gollmgo_status_t gollmgo_model_destroy(gollmgo_model_t m);

#ifdef __cplusplus
}
#endif

#endif /* GOLLMGO_MODEL_H */
