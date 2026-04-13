/*
 * gollmgo_ops.cu — CUDA kernel implementations for transformer operations.
 *
 * These are correctness-first implementations for M5 (eager forward pass).
 * Performance optimization (shared memory, vectorized loads, flash attention)
 * will come in M6+.
 */

#include "gollmgo_ops.cuh"
#include <cuda_fp16.h>
#include <math.h>

/* ---- Embedding lookup ---- */
__global__ void embedding_lookup_f16(
    const __half* __restrict__ table,
    const int32_t* __restrict__ ids,
    __half* __restrict__ out,
    int hidden_size,
    int vocab_size)
{
    int token_idx = blockIdx.x;
    int dim_idx = threadIdx.x + blockIdx.y * blockDim.x;
    if (dim_idx >= hidden_size) return;

    int32_t id = ids[token_idx];
    if (id >= 0 && id < vocab_size) {
        out[token_idx * hidden_size + dim_idx] =
            table[id * hidden_size + dim_idx];
    }
}

/* ---- RMS Norm ---- */
__global__ void rmsnorm_f16(
    const __half* __restrict__ x,
    const __half* __restrict__ weight,
    __half* __restrict__ out,
    int hidden_size,
    int n,
    float eps)
{
    int row = blockIdx.x;
    if (row >= n) return;

    const __half* x_row = x + row * hidden_size;
    __half* out_row = out + row * hidden_size;

    /* Compute sum of squares. */
    float ss = 0.0f;
    for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        float val = __half2float(x_row[i]);
        ss += val * val;
    }

    /* Warp reduction. */
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        ss += __shfl_down_sync(0xffffffff, ss, offset);
    }

    /* Block reduction via shared memory. */
    __shared__ float shared_ss[32];
    int lane = threadIdx.x % warpSize;
    int warp_id = threadIdx.x / warpSize;
    if (lane == 0) shared_ss[warp_id] = ss;
    __syncthreads();

    if (threadIdx.x == 0) {
        float total = 0.0f;
        int num_warps = (blockDim.x + warpSize - 1) / warpSize;
        for (int i = 0; i < num_warps; i++) {
            total += shared_ss[i];
        }
        shared_ss[0] = rsqrtf(total / (float)hidden_size + eps);
    }
    __syncthreads();

    float rms_inv = shared_ss[0];

    /* Normalize and scale. */
    for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        float val = __half2float(x_row[i]) * rms_inv;
        val *= __half2float(weight[i]);
        out_row[i] = __float2half(val);
    }
}

/* ---- RoPE ---- */
__global__ void rope_f16(
    __half* __restrict__ q,
    __half* __restrict__ k,
    const int32_t* __restrict__ positions,
    int n,
    int num_heads,
    int num_kv_heads,
    int head_dim,
    float theta_base)
{
    int token_idx = blockIdx.x;
    int head_idx = blockIdx.y;
    int pair_idx = threadIdx.x; /* operates on pairs: [2*pair_idx, 2*pair_idx+1] */

    if (token_idx >= n || pair_idx >= head_dim / 2) return;

    float pos = (float)positions[token_idx];
    float freq = 1.0f / powf(theta_base, (float)(2 * pair_idx) / (float)head_dim);
    float angle = pos * freq;
    float cos_val = cosf(angle);
    float sin_val = sinf(angle);

    /* Apply to Q. */
    if (head_idx < num_heads) {
        int base = (token_idx * num_heads + head_idx) * head_dim + 2 * pair_idx;
        float q0 = __half2float(q[base]);
        float q1 = __half2float(q[base + 1]);
        q[base]     = __float2half(q0 * cos_val - q1 * sin_val);
        q[base + 1] = __float2half(q0 * sin_val + q1 * cos_val);
    }

    /* Apply to K (only for KV heads). */
    if (head_idx < num_kv_heads) {
        int base = (token_idx * num_kv_heads + head_idx) * head_dim + 2 * pair_idx;
        float k0 = __half2float(k[base]);
        float k1 = __half2float(k[base + 1]);
        k[base]     = __float2half(k0 * cos_val - k1 * sin_val);
        k[base + 1] = __float2half(k0 * sin_val + k1 * cos_val);
    }
}

/* ---- SiLU * Up ---- */
__global__ void silu_mul_f16(
    const __half* __restrict__ gate,
    const __half* __restrict__ up,
    __half* __restrict__ out,
    int total_elements)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements) return;

    float g = __half2float(gate[idx]);
    float u = __half2float(up[idx]);
    float silu_g = g / (1.0f + expf(-g));
    out[idx] = __float2half(silu_g * u);
}

/* ---- Naive attention ---- */
__global__ void naive_attention_f16(
    const __half* __restrict__ q,
    const __half* __restrict__ k,
    const __half* __restrict__ v,
    __half* __restrict__ out,
    const int32_t* __restrict__ positions,
    int n,
    int num_heads,
    int num_kv_heads,
    int head_dim,
    float scale)
{
    /* One block per (token, head) pair. */
    int token_idx = blockIdx.x;
    int head_idx = blockIdx.y;
    if (token_idx >= n || head_idx >= num_heads) return;

    /* GQA: map query head to KV head. */
    int kv_group_size = num_heads / num_kv_heads;
    int kv_head_idx = head_idx / kv_group_size;

    int pos_i = positions[token_idx];

    const __half* q_vec = q + (token_idx * num_heads + head_idx) * head_dim;
    __half* out_vec = out + (token_idx * num_heads + head_idx) * head_dim;

    /* Compute attention scores for all positions <= pos_i (causal mask). */
    extern __shared__ float smem[];
    float* scores = smem; /* [n] */

    float max_score = -1e30f;
    for (int j = 0; j < n; j++) {
        if (positions[j] > pos_i) {
            scores[j] = -1e30f;
            continue;
        }

        const __half* k_vec = k + (j * num_kv_heads + kv_head_idx) * head_dim;
        float dot = 0.0f;
        for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
            dot += __half2float(q_vec[d]) * __half2float(k_vec[d]);
        }
        /* Warp reduce dot product. */
        for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
            dot += __shfl_down_sync(0xffffffff, dot, offset);
        }
        if (threadIdx.x == 0) {
            scores[j] = dot * scale;
            if (scores[j] > max_score) max_score = scores[j];
        }
        __syncthreads();
    }

    /* Softmax. */
    if (threadIdx.x == 0) {
        float sum_exp = 0.0f;
        for (int j = 0; j < n; j++) {
            scores[j] = expf(scores[j] - max_score);
            sum_exp += scores[j];
        }
        for (int j = 0; j < n; j++) {
            scores[j] /= sum_exp;
        }
    }
    __syncthreads();

    /* Weighted sum of V. */
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        float acc = 0.0f;
        for (int j = 0; j < n; j++) {
            acc += scores[j] * __half2float(
                v[(j * num_kv_heads + kv_head_idx) * head_dim + d]);
        }
        out_vec[d] = __float2half(acc);
    }
}

/* ---- Residual add ---- */
__global__ void residual_add_f16(
    const __half* __restrict__ a,
    const __half* __restrict__ b,
    __half* __restrict__ out,
    int total_elements)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements) return;
    out[idx] = __float2half(__half2float(a[idx]) + __half2float(b[idx]));
}

/* ---- F16 to F32 ---- */
__global__ void f16_to_f32(
    const __half* __restrict__ in,
    float* __restrict__ out,
    int total_elements)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements) return;
    out[idx] = __half2float(in[idx]);
}
