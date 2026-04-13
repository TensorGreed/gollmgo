/*
 * gollmgo_backend.cu — Minimal CUDA backend implementation.
 *
 * Implements the C API from gollmgo_backend.h.
 * This covers lifecycle only (create/warmup/destroy/info).
 * Kernel dispatch (step) will be added in M5.
 */

#include "gollmgo_backend.h"
#include <cuda_runtime.h>
#include <cstring>
#include <cstdio>

struct gollmgo_backend {
    int  device_id;
    bool warmed_up;
    char last_error[512];
};

static void set_error(gollmgo_backend_t b, const char* msg) {
    if (b) {
        snprintf(b->last_error, sizeof(b->last_error), "%s", msg);
    }
}

static gollmgo_status_t check_cuda(gollmgo_backend_t b, cudaError_t err) {
    if (err == cudaSuccess) return GOLLMGO_OK;
    set_error(b, cudaGetErrorString(err));
    if (err == cudaErrorMemoryAllocation) return GOLLMGO_ERR_OOM;
    return GOLLMGO_ERR_CUDA;
}

extern "C" {

gollmgo_status_t gollmgo_backend_create(int device_id, gollmgo_backend_t* out) {
    if (!out) return GOLLMGO_ERR_INVALID;

    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    if (err != cudaSuccess || device_id < 0 || device_id >= device_count) {
        return GOLLMGO_ERR_CUDA;
    }

    gollmgo_backend_t b = new gollmgo_backend();
    b->device_id = device_id;
    b->warmed_up = false;
    memset(b->last_error, 0, sizeof(b->last_error));

    gollmgo_status_t status = check_cuda(b, cudaSetDevice(device_id));
    if (status != GOLLMGO_OK) {
        delete b;
        return status;
    }

    *out = b;
    return GOLLMGO_OK;
}

gollmgo_status_t gollmgo_backend_device_info(gollmgo_backend_t b, gollmgo_device_info_t* info) {
    if (!b || !info) return GOLLMGO_ERR_INVALID;

    cudaDeviceProp prop;
    gollmgo_status_t status = check_cuda(b, cudaGetDeviceProperties(&prop, b->device_id));
    if (status != GOLLMGO_OK) return status;

    memset(info, 0, sizeof(*info));
    snprintf(info->name, sizeof(info->name), "%s", prop.name);
    info->compute_major = prop.major;
    info->compute_minor = prop.minor;
    info->total_memory_bytes = prop.totalGlobalMem;

    size_t free_mem = 0, total_mem = 0;
    status = check_cuda(b, cudaMemGetInfo(&free_mem, &total_mem));
    if (status != GOLLMGO_OK) return status;
    info->free_memory_bytes = (uint64_t)free_mem;

    return GOLLMGO_OK;
}

gollmgo_status_t gollmgo_backend_warmup(gollmgo_backend_t b,
                                         int max_batch_size,
                                         int max_seq_len,
                                         int block_size) {
    if (!b) return GOLLMGO_ERR_INVALID;
    if (max_batch_size <= 0 || max_seq_len <= 0 || block_size <= 0) {
        set_error(b, "invalid warmup parameters");
        return GOLLMGO_ERR_INVALID;
    }

    gollmgo_status_t status = check_cuda(b, cudaSetDevice(b->device_id));
    if (status != GOLLMGO_OK) return status;

    /* Placeholder: a real warmup would pre-allocate KV blocks and
       optionally capture CUDA graphs. For now, just synchronize. */
    status = check_cuda(b, cudaDeviceSynchronize());
    if (status != GOLLMGO_OK) return status;

    b->warmed_up = true;
    return GOLLMGO_OK;
}

gollmgo_status_t gollmgo_backend_destroy(gollmgo_backend_t b) {
    if (!b) return GOLLMGO_OK;
    delete b;
    return GOLLMGO_OK;
}

const char* gollmgo_backend_last_error(gollmgo_backend_t b) {
    if (!b) return "null backend handle";
    return b->last_error;
}

} /* extern "C" */
