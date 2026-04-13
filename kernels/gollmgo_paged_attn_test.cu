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

/*
 * Test: verify v2 (partitioned) produces the same output as v1
 * for a long sequence that requires multiple partitions.
 */
void test_v2_vs_v1() {
    printf("test_v2_vs_v1... ");

    const int seq_len = 1024; /* >256 → multiple partitions */
    const int num_heads = 4;
    const int num_kv_heads = 2; /* GQA: 2 KV heads for 4 query heads */
    const int head_dim = 64;
    const int n_query = 1;
    const int partition_size = PAGED_ATTN_V2_PARTITION_SIZE;
    const float scale = 1.0f / sqrtf((float)head_dim);

    srand(99);

    int kv_entry = num_kv_heads * head_dim;
    int q_total = n_query * num_heads * head_dim;

    /* Random Q (for query token) and KV cache data. */
    float* h_q_f32 = new float[q_total];
    for (int i = 0; i < q_total; i++) h_q_f32[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.5f;

    int cache_entries = seq_len * kv_entry;
    float* h_k_f32 = new float[cache_entries];
    float* h_v_f32 = new float[cache_entries];
    for (int i = 0; i < cache_entries; i++) h_k_f32[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.5f;
    for (int i = 0; i < cache_entries; i++) h_v_f32[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.5f;

    /* Convert to FP16. */
    __half* h_q = new __half[q_total];
    __half* h_k = new __half[cache_entries];
    __half* h_v = new __half[cache_entries];
    for (int i = 0; i < q_total; i++) h_q[i] = __float2half(h_q_f32[i]);
    for (int i = 0; i < cache_entries; i++) h_k[i] = __float2half(h_k_f32[i]);
    for (int i = 0; i < cache_entries; i++) h_v[i] = __float2half(h_v_f32[i]);

    /* GPU allocations. */
    __half *d_q, *d_k_cache, *d_v_cache, *d_out_v1;
    CHECK_CUDA(cudaMalloc(&d_q, q_total * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&d_k_cache, cache_entries * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&d_v_cache, cache_entries * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&d_out_v1, q_total * sizeof(__half)));
    CHECK_CUDA(cudaMemcpy(d_q, h_q, q_total * sizeof(__half), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_k_cache, h_k, cache_entries * sizeof(__half), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_v_cache, h_v, cache_entries * sizeof(__half), cudaMemcpyHostToDevice));

    /* Slot table: identity mapping. */
    int32_t* h_slot_table = new int32_t[seq_len];
    for (int i = 0; i < seq_len; i++) h_slot_table[i] = i;
    int32_t h_seq_len = seq_len;

    int32_t *d_seq_lens, *d_slot_table;
    CHECK_CUDA(cudaMalloc(&d_seq_lens, sizeof(int32_t)));
    CHECK_CUDA(cudaMalloc(&d_slot_table, seq_len * sizeof(int32_t)));
    CHECK_CUDA(cudaMemcpy(d_seq_lens, &h_seq_len, sizeof(int32_t), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_slot_table, h_slot_table, seq_len * sizeof(int32_t), cudaMemcpyHostToDevice));

    /* ---- Run v1 ---- */
    {
        dim3 grid(1, num_heads);
        int threads = 32;
        int smem = seq_len * sizeof(float);
        paged_attention_v1_f16<<<grid, threads, smem>>>(
            d_q, d_k_cache, d_v_cache, d_out_v1,
            d_seq_lens, d_slot_table,
            1, num_heads, num_kv_heads, head_dim, seq_len, scale);
        CHECK_CUDA(cudaDeviceSynchronize());
    }

    __half* h_out_v1 = new __half[q_total];
    CHECK_CUDA(cudaMemcpy(h_out_v1, d_out_v1, q_total * sizeof(__half), cudaMemcpyDeviceToHost));

    /* ---- Run v2 ---- */
    int num_parts = (seq_len + partition_size - 1) / partition_size;
    float *d_exp_sums, *d_max_logits, *d_partial_out, *d_reduce_out;
    size_t scalar_sz = n_query * num_heads * num_parts * sizeof(float);
    size_t vec_sz = (size_t)n_query * num_heads * num_parts * head_dim * sizeof(float);
    size_t out_sz = n_query * num_heads * head_dim * sizeof(float);

    CHECK_CUDA(cudaMalloc(&d_exp_sums, scalar_sz));
    CHECK_CUDA(cudaMalloc(&d_max_logits, scalar_sz));
    CHECK_CUDA(cudaMalloc(&d_partial_out, vec_sz));
    CHECK_CUDA(cudaMalloc(&d_reduce_out, out_sz));

    /* Phase 1. */
    {
        dim3 grid(1, num_heads, num_parts);
        int threads = 32;
        int smem = partition_size * sizeof(float);
        paged_attention_v2_phase1_f16<<<grid, threads, smem>>>(
            d_q, d_k_cache, d_v_cache,
            d_exp_sums, d_max_logits, d_partial_out,
            d_seq_lens, d_slot_table,
            1, num_heads, num_kv_heads, head_dim, seq_len, scale, partition_size);
        CHECK_CUDA(cudaDeviceSynchronize());
    }

    /* Phase 2: reduce. */
    {
        dim3 grid(1, num_heads);
        int threads = 32;
        int smem = num_parts * sizeof(float);
        paged_attention_v2_reduce<<<grid, threads, smem>>>(
            d_exp_sums, d_max_logits, d_partial_out, d_reduce_out,
            d_seq_lens, 1, num_heads, head_dim, partition_size, num_parts);
        CHECK_CUDA(cudaDeviceSynchronize());
    }

    float* h_out_v2_f32 = new float[n_query * num_heads * head_dim];
    CHECK_CUDA(cudaMemcpy(h_out_v2_f32, d_reduce_out, out_sz, cudaMemcpyDeviceToHost));

    /* ---- Compare ---- */
    float max_diff = 0.0f;
    for (int i = 0; i < q_total; i++) {
        float v1_val = __half2float(h_out_v1[i]);
        float v2_val = h_out_v2_f32[i];
        float diff = fabsf(v1_val - v2_val);
        if (diff > max_diff) max_diff = diff;
        if (diff > 0.1f) {
            fprintf(stderr, "FAIL v2_vs_v1[%d]: v1=%.4f v2=%.4f diff=%.4f\n",
                    i, v1_val, v2_val, diff);
            failures++;
        }
    }
    printf("PASS (max_diff=%.4f, seq_len=%d, partitions=%d)\n", max_diff, seq_len, num_parts);

    /* Cleanup. */
    cudaFree(d_q); cudaFree(d_k_cache); cudaFree(d_v_cache); cudaFree(d_out_v1);
    cudaFree(d_seq_lens); cudaFree(d_slot_table);
    cudaFree(d_exp_sums); cudaFree(d_max_logits); cudaFree(d_partial_out); cudaFree(d_reduce_out);
    delete[] h_q_f32; delete[] h_k_f32; delete[] h_v_f32;
    delete[] h_q; delete[] h_k; delete[] h_v;
    delete[] h_slot_table; delete[] h_out_v1; delete[] h_out_v2_f32;
}

int main() {
    printf("=== gollmgo paged attention correctness tests ===\n");
    test_paged_vs_naive();
    test_v2_vs_v1();
    printf("=== %s (%d failures) ===\n", failures ? "FAILED" : "ALL PASSED", failures);
    return failures ? 1 : 0;
}
