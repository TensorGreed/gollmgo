/*
 * gollmgo_ops_test.cu — Kernel correctness tests with CPU references.
 *
 * Compiled and run as a standalone CUDA program.
 * Build: nvcc -arch=native -o ops_test gollmgo_ops_test.cu gollmgo_ops.cu -lcudart
 * Run: ./ops_test
 */

#include "gollmgo_ops.cuh"
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#define CHECK_CUDA(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

#define ASSERT_NEAR(a, b, tol, msg) do { \
    float _a = (a), _b = (b); \
    if (fabsf(_a - _b) > (tol)) { \
        fprintf(stderr, "FAIL %s: %.6f vs %.6f (tol %.6f)\n", msg, _a, _b, (float)(tol)); \
        failures++; \
    } \
} while(0)

static int failures = 0;

/* ---- CPU reference: RMSNorm ---- */
void rmsnorm_cpu(const float* x, const float* weight, float* out,
                  int hidden, int n, float eps) {
    for (int row = 0; row < n; row++) {
        float ss = 0;
        for (int i = 0; i < hidden; i++) {
            ss += x[row * hidden + i] * x[row * hidden + i];
        }
        float rms_inv = 1.0f / sqrtf(ss / (float)hidden + eps);
        for (int i = 0; i < hidden; i++) {
            out[row * hidden + i] = x[row * hidden + i] * rms_inv * weight[i];
        }
    }
}

/* ---- CPU reference: RoPE ---- */
void rope_cpu(float* q, float* k, const int* positions,
              int n, int num_heads, int num_kv_heads, int head_dim, float theta) {
    for (int t = 0; t < n; t++) {
        float pos = (float)positions[t];
        for (int h = 0; h < num_heads || h < num_kv_heads; h++) {
            for (int p = 0; p < head_dim / 2; p++) {
                float freq = 1.0f / powf(theta, (float)(2 * p) / (float)head_dim);
                float angle = pos * freq;
                float cos_val = cosf(angle);
                float sin_val = sinf(angle);

                if (h < num_heads) {
                    int base = (t * num_heads + h) * head_dim + 2 * p;
                    float q0 = q[base], q1 = q[base + 1];
                    q[base]     = q0 * cos_val - q1 * sin_val;
                    q[base + 1] = q0 * sin_val + q1 * cos_val;
                }
                if (h < num_kv_heads) {
                    int base = (t * num_kv_heads + h) * head_dim + 2 * p;
                    float k0 = k[base], k1 = k[base + 1];
                    k[base]     = k0 * cos_val - k1 * sin_val;
                    k[base + 1] = k0 * sin_val + k1 * cos_val;
                }
            }
        }
    }
}

/* ---- CPU reference: SiLU * up ---- */
void silu_mul_cpu(const float* gate, const float* up, float* out, int n) {
    for (int i = 0; i < n; i++) {
        float silu = gate[i] / (1.0f + expf(-gate[i]));
        out[i] = silu * up[i];
    }
}

/* ---- Helpers ---- */
void f32_to_f16(const float* in, __half* out, int n) {
    for (int i = 0; i < n; i++) out[i] = __float2half(in[i]);
}
void f16_to_f32_cpu(const __half* in, float* out, int n) {
    for (int i = 0; i < n; i++) out[i] = __half2float(in[i]);
}

/* ---- Test: RMSNorm ---- */
void test_rmsnorm() {
    printf("test_rmsnorm... ");
    const int hidden = 64, n = 4;
    float eps = 1e-5f;

    float* h_x      = new float[n * hidden];
    float* h_weight  = new float[hidden];
    float* h_ref     = new float[n * hidden];
    float* h_result  = new float[n * hidden];

    srand(42);
    for (int i = 0; i < n * hidden; i++) h_x[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f;
    for (int i = 0; i < hidden; i++) h_weight[i] = ((float)rand() / RAND_MAX) * 0.5f + 0.75f;

    rmsnorm_cpu(h_x, h_weight, h_ref, hidden, n, eps);

    /* GPU path. */
    __half *d_x, *d_w, *d_out;
    __half *h_x16 = new __half[n * hidden];
    __half *h_w16 = new __half[hidden];
    __half *h_out16 = new __half[n * hidden];

    f32_to_f16(h_x, h_x16, n * hidden);
    f32_to_f16(h_weight, h_w16, hidden);

    CHECK_CUDA(cudaMalloc(&d_x, n * hidden * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&d_w, hidden * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&d_out, n * hidden * sizeof(__half)));

    CHECK_CUDA(cudaMemcpy(d_x, h_x16, n * hidden * sizeof(__half), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_w, h_w16, hidden * sizeof(__half), cudaMemcpyHostToDevice));

    int threads = (hidden < 256) ? hidden : 256;
    rmsnorm_f16<<<n, threads>>>(d_x, d_w, d_out, hidden, n, eps);
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(h_out16, d_out, n * hidden * sizeof(__half), cudaMemcpyDeviceToHost));
    f16_to_f32_cpu(h_out16, h_result, n * hidden);

    for (int i = 0; i < n * hidden; i++) {
        ASSERT_NEAR(h_result[i], h_ref[i], 0.02f, "rmsnorm");
    }

    cudaFree(d_x); cudaFree(d_w); cudaFree(d_out);
    delete[] h_x; delete[] h_weight; delete[] h_ref; delete[] h_result;
    delete[] h_x16; delete[] h_w16; delete[] h_out16;
    printf("PASS\n");
}

/* ---- Test: SiLU * up ---- */
void test_silu_mul() {
    printf("test_silu_mul... ");
    const int n = 256;
    float* h_gate = new float[n];
    float* h_up   = new float[n];
    float* h_ref  = new float[n];
    float* h_result = new float[n];

    srand(123);
    for (int i = 0; i < n; i++) {
        h_gate[i] = ((float)rand() / RAND_MAX - 0.5f) * 4.0f;
        h_up[i]   = ((float)rand() / RAND_MAX - 0.5f) * 4.0f;
    }
    silu_mul_cpu(h_gate, h_up, h_ref, n);

    __half *d_gate, *d_up, *d_out;
    __half *h_g16 = new __half[n], *h_u16 = new __half[n], *h_o16 = new __half[n];

    f32_to_f16(h_gate, h_g16, n);
    f32_to_f16(h_up, h_u16, n);

    CHECK_CUDA(cudaMalloc(&d_gate, n * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&d_up, n * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&d_out, n * sizeof(__half)));
    CHECK_CUDA(cudaMemcpy(d_gate, h_g16, n * sizeof(__half), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_up, h_u16, n * sizeof(__half), cudaMemcpyHostToDevice));

    silu_mul_f16<<<(n + 255) / 256, 256>>>(d_gate, d_up, d_out, n);
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(h_o16, d_out, n * sizeof(__half), cudaMemcpyDeviceToHost));
    f16_to_f32_cpu(h_o16, h_result, n);

    for (int i = 0; i < n; i++) {
        ASSERT_NEAR(h_result[i], h_ref[i], 0.05f, "silu_mul");
    }

    cudaFree(d_gate); cudaFree(d_up); cudaFree(d_out);
    delete[] h_gate; delete[] h_up; delete[] h_ref; delete[] h_result;
    delete[] h_g16; delete[] h_u16; delete[] h_o16;
    printf("PASS\n");
}

/* ---- Test: Embedding lookup ---- */
void test_embedding() {
    printf("test_embedding... ");
    const int vocab = 10, hidden = 32, n = 3;
    float* h_table = new float[vocab * hidden];
    int32_t h_ids[3] = {0, 5, 9};
    float* h_result = new float[n * hidden];

    srand(77);
    for (int i = 0; i < vocab * hidden; i++) {
        h_table[i] = (float)i / (vocab * hidden);
    }

    __half* h_t16 = new __half[vocab * hidden];
    f32_to_f16(h_table, h_t16, vocab * hidden);

    __half *d_table, *d_out;
    int32_t* d_ids;
    CHECK_CUDA(cudaMalloc(&d_table, vocab * hidden * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&d_out, n * hidden * sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&d_ids, n * sizeof(int32_t)));

    CHECK_CUDA(cudaMemcpy(d_table, h_t16, vocab * hidden * sizeof(__half), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_ids, h_ids, n * sizeof(int32_t), cudaMemcpyHostToDevice));

    int threads = (hidden < 256) ? hidden : 256;
    dim3 grid(n, (hidden + threads - 1) / threads);
    embedding_lookup_f16<<<grid, threads>>>(d_table, d_ids, d_out, hidden, vocab);
    CHECK_CUDA(cudaDeviceSynchronize());

    __half* h_o16 = new __half[n * hidden];
    CHECK_CUDA(cudaMemcpy(h_o16, d_out, n * hidden * sizeof(__half), cudaMemcpyDeviceToHost));
    f16_to_f32_cpu(h_o16, h_result, n * hidden);

    /* Verify against expected rows. */
    for (int t = 0; t < n; t++) {
        int id = h_ids[t];
        for (int d = 0; d < hidden; d++) {
            float expected = h_table[id * hidden + d];
            ASSERT_NEAR(h_result[t * hidden + d], expected, 0.01f, "embedding");
        }
    }

    cudaFree(d_table); cudaFree(d_out); cudaFree(d_ids);
    delete[] h_table; delete[] h_t16; delete[] h_o16; delete[] h_result;
    printf("PASS\n");
}

int main() {
    printf("=== gollmgo kernel correctness tests ===\n");
    test_rmsnorm();
    test_silu_mul();
    test_embedding();
    printf("=== %s (%d failures) ===\n", failures ? "FAILED" : "ALL PASSED", failures);
    return failures ? 1 : 0;
}
