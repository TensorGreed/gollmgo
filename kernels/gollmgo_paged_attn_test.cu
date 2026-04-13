/*
 * gollmgo_paged_attn_test.cu — Paged attention vs naive attention correctness test.
 *
 * Build: nvcc -arch=native -o paged_attn_test \
 *        gollmgo_paged_attn_test.cu gollmgo_paged_attn.cu gollmgo_ops.cu \
 *        -lcudart -lstdc++ -lm
 * Run: ./paged_attn_test
 */

#include "gollmgo_ops.cuh"
#include "gollmgo_paged_attn.cuh"
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>

#define CHECK_CUDA(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

static int failures = 0;

/*
 * Test: verify paged attention produces the same output as naive attention
 * for a single-sequence decode scenario.
 *
 * Setup:
 * - 1 sequence with seq_len=8 tokens (all in cache)
 * - 1 new query token (decode)
 * - num_heads=4, num_kv_heads=4 (no GQA), head_dim=16
 * - block_size=4 -> 2 blocks needed
 */
void test_paged_vs_naive() {
    printf("test_paged_vs_naive... ");

    const int seq_len = 8;
    const int num_heads = 4;
    const int num_kv_heads = 4;
    const int head_dim = 16;
    const int block_size = 4;
    const int n_query = 1;
    const float scale = 1.0f / sqrtf((float)head_dim);

    srand(42);

    /* Generate random Q, K, V for all seq_len+1 tokens. */
    int total_tokens = seq_len + n_query;
    int q_size = total_tokens * num_heads * head_dim;
    int kv_size = total_tokens * num_kv_heads * head_dim;

    float* h_q_f32 = new float[q_size];
    float* h_k_f32 = new float[kv_size];
    float* h_v_f32 = new float[kv_size];

    for (int i = 0; i < q_size; i++) h_q_f32[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.5f;
    for (int i = 0; i < kv_size; i++) h_k_f32[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.5f;
    for (int i = 0; i < kv_size; i++) h_v_f32[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.5f;

    /* Convert to FP16. */
    __half* h_q = new __half[q_size];
    __half* h_k = new __half[kv_size];
    __half* h_v = new __half[kv_size];
    for (int i = 0; i < q_size; i++) h_q[i] = __float2half(h_q_f32[i]);
    for (int i = 0; i < kv_size; i++) h_k[i] = __float2half(h_k_f32[i]);
    for (int i = 0; i < kv_size; i++) h_v[i] = __float2half(h_v_f32[i]);

    /* ---- Run naive attention on all tokens ---- */
    __half* d_q_all, *d_k_all, *d_v_all, *d_out_naive;
    int32_t* d_positions;
    CHECK_CUDA(cudaMalloc(&d_q_all, q_size * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&d_k_all, kv_size * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&d_v_all, kv_size * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&d_out_naive, q_size * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&d_positions, total_tokens * sizeof(int32_t)));

    CHECK_CUDA(cudaMemcpy(d_q_all, h_q, q_size * sizeof(__half), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_k_all, h_k, kv_size * sizeof(__half), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_v_all, h_v, kv_size * sizeof(__half), cudaMemcpyHostToDevice));

    int32_t h_positions[9];
    for (int i = 0; i < total_tokens; i++) h_positions[i] = i;
    CHECK_CUDA(cudaMemcpy(d_positions, h_positions, total_tokens * sizeof(int32_t), cudaMemcpyHostToDevice));

    {
        dim3 grid(total_tokens, num_heads);
        int threads = (head_dim < 32) ? head_dim : 32;
        int smem = total_tokens * sizeof(float);
        naive_attention_f16<<<grid, threads, smem>>>(
            d_q_all, d_k_all, d_v_all, d_out_naive, d_positions,
            total_tokens, num_heads, num_kv_heads, head_dim, scale);
        CHECK_CUDA(cudaDeviceSynchronize());
    }

    /* Extract last token's naive output. */
    int last_offset = seq_len * num_heads * head_dim;
    int head_total = num_heads * head_dim;
    __half* h_naive_last = new __half[head_total];
    CHECK_CUDA(cudaMemcpy(h_naive_last, d_out_naive + last_offset,
                           head_total * sizeof(__half), cudaMemcpyDeviceToHost));

    /* ---- Run paged attention ---- */
    /* Setup KV cache with 2 blocks * 4 slots/block = 8 slots + extra for query. */
    int num_blocks = (seq_len + block_size - 1) / block_size;
    int num_slots = (num_blocks + 1) * block_size; /* extra room */

    int kv_entry = num_kv_heads * head_dim;
    __half* d_k_cache, *d_v_cache;
    CHECK_CUDA(cudaMalloc(&d_k_cache, num_slots * kv_entry * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&d_v_cache, num_slots * kv_entry * sizeof(__half)));
    CHECK_CUDA(cudaMemset(d_k_cache, 0, num_slots * kv_entry * sizeof(__half)));
    CHECK_CUDA(cudaMemset(d_v_cache, 0, num_slots * kv_entry * sizeof(__half)));

    /* Write cached KV (first seq_len tokens) into paged cache.
     * Use identity slot mapping: slot[i] = i. */
    int32_t h_write_slots[8];
    for (int i = 0; i < seq_len; i++) h_write_slots[i] = i;

    __half* d_k_write, *d_v_write;
    int32_t* d_write_slots;
    CHECK_CUDA(cudaMalloc(&d_k_write, seq_len * kv_entry * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&d_v_write, seq_len * kv_entry * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&d_write_slots, seq_len * sizeof(int32_t)));

    CHECK_CUDA(cudaMemcpy(d_k_write, h_k, seq_len * kv_entry * sizeof(__half), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_v_write, h_v, seq_len * kv_entry * sizeof(__half), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_write_slots, h_write_slots, seq_len * sizeof(int32_t), cudaMemcpyHostToDevice));

    paged_kv_write_f16<<<seq_len, 256>>>(
        d_k_cache, d_v_cache, d_k_write, d_v_write, d_write_slots,
        seq_len, num_kv_heads, head_dim);
    CHECK_CUDA(cudaDeviceSynchronize());

    /* Also write the query token's KV into slot 8. */
    int32_t h_query_slot = seq_len; /* slot 8 */
    __half* h_k_query = h_k + seq_len * kv_entry;
    __half* h_v_query = h_v + seq_len * kv_entry;

    __half* d_k_qw, *d_v_qw;
    int32_t* d_q_slot;
    CHECK_CUDA(cudaMalloc(&d_k_qw, kv_entry * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&d_v_qw, kv_entry * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&d_q_slot, sizeof(int32_t)));
    CHECK_CUDA(cudaMemcpy(d_k_qw, h_k_query, kv_entry * sizeof(__half), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_v_qw, h_v_query, kv_entry * sizeof(__half), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_q_slot, &h_query_slot, sizeof(int32_t), cudaMemcpyHostToDevice));

    paged_kv_write_f16<<<1, 256>>>(
        d_k_cache, d_v_cache, d_k_qw, d_v_qw, d_q_slot,
        1, num_kv_heads, head_dim);
    CHECK_CUDA(cudaDeviceSynchronize());

    /* Run paged attention on the query token.
     * The slot table for this query covers all seq_len+1 tokens. */
    int full_len = seq_len + 1; /* includes the query token's own KV */
    int32_t h_slot_table[9];
    for (int i = 0; i < full_len; i++) h_slot_table[i] = i;
    int32_t h_seq_len = full_len;

    __half* d_q_query, *d_out_paged;
    int32_t* d_paged_seq_len, *d_paged_slot_table;
    __half* h_q_query = h_q + seq_len * num_heads * head_dim;

    CHECK_CUDA(cudaMalloc(&d_q_query, head_total * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&d_out_paged, head_total * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&d_paged_seq_len, sizeof(int32_t)));
    CHECK_CUDA(cudaMalloc(&d_paged_slot_table, full_len * sizeof(int32_t)));

    CHECK_CUDA(cudaMemcpy(d_q_query, h_q_query, head_total * sizeof(__half), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_paged_seq_len, &h_seq_len, sizeof(int32_t), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_paged_slot_table, h_slot_table, full_len * sizeof(int32_t), cudaMemcpyHostToDevice));

    {
        dim3 grid(1, num_heads);
        int threads = (head_dim < 32) ? head_dim : 32;
        int smem = full_len * sizeof(float);
        paged_attention_v1_f16<<<grid, threads, smem>>>(
            d_q_query, d_k_cache, d_v_cache, d_out_paged,
            d_paged_seq_len, d_paged_slot_table,
            1, num_heads, num_kv_heads, head_dim, full_len, scale);
        CHECK_CUDA(cudaDeviceSynchronize());
    }

    __half* h_paged_out = new __half[head_total];
    CHECK_CUDA(cudaMemcpy(h_paged_out, d_out_paged, head_total * sizeof(__half), cudaMemcpyDeviceToHost));

    /* ---- Compare ---- */
    float max_diff = 0.0f;
    for (int i = 0; i < head_total; i++) {
        float naive_val = __half2float(h_naive_last[i]);
        float paged_val = __half2float(h_paged_out[i]);
        float diff = fabsf(naive_val - paged_val);
        if (diff > max_diff) max_diff = diff;
        if (diff > 0.05f) {
            fprintf(stderr, "FAIL paged_vs_naive[%d]: naive=%.4f paged=%.4f diff=%.4f\n",
                    i, naive_val, paged_val, diff);
            failures++;
        }
    }
    printf("PASS (max_diff=%.4f)\n", max_diff);

    /* Cleanup. */
    cudaFree(d_q_all); cudaFree(d_k_all); cudaFree(d_v_all); cudaFree(d_out_naive);
    cudaFree(d_positions); cudaFree(d_k_cache); cudaFree(d_v_cache);
    cudaFree(d_k_write); cudaFree(d_v_write); cudaFree(d_write_slots);
    cudaFree(d_k_qw); cudaFree(d_v_qw); cudaFree(d_q_slot);
    cudaFree(d_q_query); cudaFree(d_out_paged);
    cudaFree(d_paged_seq_len); cudaFree(d_paged_slot_table);
    delete[] h_q_f32; delete[] h_k_f32; delete[] h_v_f32;
    delete[] h_q; delete[] h_k; delete[] h_v;
    delete[] h_naive_last; delete[] h_paged_out;
}

int main() {
    printf("=== gollmgo paged attention correctness tests ===\n");
    test_paged_vs_naive();
    printf("=== %s (%d failures) ===\n", failures ? "FAILED" : "ALL PASSED", failures);
    return failures ? 1 : 0;
}
