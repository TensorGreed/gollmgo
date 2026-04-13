/*
 * gollmgo_backend.h — Minimal C API for the CUDA backend.
 *
 * This is the narrow CGo boundary. All GPU-side details stay behind
 * this interface. Go code only sees opaque handles and status codes.
 */

#ifndef GOLLMGO_BACKEND_H
#define GOLLMGO_BACKEND_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Opaque handle to the backend runtime. */
typedef struct gollmgo_backend* gollmgo_backend_t;

/* Status codes. */
typedef enum {
    GOLLMGO_OK             = 0,
    GOLLMGO_ERR_CUDA       = 1,
    GOLLMGO_ERR_OOM        = 2,
    GOLLMGO_ERR_INVALID    = 3,
    GOLLMGO_ERR_INTERNAL   = 4,
} gollmgo_status_t;

/* Device info populated by gollmgo_backend_create. */
typedef struct {
    char     name[256];
    int      compute_major;
    int      compute_minor;
    uint64_t total_memory_bytes;
    uint64_t free_memory_bytes;
} gollmgo_device_info_t;

/* Create a backend on the given CUDA device. */
gollmgo_status_t gollmgo_backend_create(int device_id, gollmgo_backend_t* out);

/* Query device info. */
gollmgo_status_t gollmgo_backend_device_info(gollmgo_backend_t b, gollmgo_device_info_t* info);

/* Warmup / pre-allocate for given shapes. */
gollmgo_status_t gollmgo_backend_warmup(gollmgo_backend_t b,
                                         int max_batch_size,
                                         int max_seq_len,
                                         int block_size);

/* Destroy the backend and free all resources. */
gollmgo_status_t gollmgo_backend_destroy(gollmgo_backend_t b);

/* Get human-readable error for the last failure on this backend. */
const char* gollmgo_backend_last_error(gollmgo_backend_t b);

#ifdef __cplusplus
}
#endif

#endif /* GOLLMGO_BACKEND_H */
