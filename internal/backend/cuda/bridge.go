// Package cuda provides the CGo bridge to the CUDA backend.
//
// Build constraint: requires CGO_ENABLED=1 and CUDA toolkit.
// This file is only compiled when the "gpu" build tag is set.
//
//go:build gpu

package cuda

/*
#cgo CFLAGS: -I${SRCDIR}/../../../kernels
#cgo LDFLAGS: -L${SRCDIR}/../../../kernels -L/usr/local/cuda/targets/sbsa-linux/lib -lgollmgo_kvcache -lgollmgo_paged_attn -lgollmgo_model -lgollmgo_ops -lgollmgo_backend -lcublas -lcudart -lstdc++ -lm
#include "gollmgo_backend.h"
#include <stdlib.h>
*/
import "C"

import (
	"context"
	"fmt"
	"unsafe"

	"github.com/TensorGreed/gollmgo/internal/backend"
)

// CUDARunner implements backend.Runner via the CGo bridge.
type CUDARunner struct {
	handle C.gollmgo_backend_t
	info   backend.DeviceInfo
}

// New creates a CUDARunner on the given device.
func New(deviceID int) (*CUDARunner, error) {
	var handle C.gollmgo_backend_t
	status := C.gollmgo_backend_create(C.int(deviceID), &handle)
	if status != C.GOLLMGO_OK {
		return nil, fmt.Errorf("cuda: backend_create failed (status %d)", int(status))
	}

	var cInfo C.gollmgo_device_info_t
	status = C.gollmgo_backend_device_info(handle, &cInfo)
	if status != C.GOLLMGO_OK {
		errMsg := C.GoString(C.gollmgo_backend_last_error(handle))
		C.gollmgo_backend_destroy(handle)
		return nil, fmt.Errorf("cuda: device_info failed: %s", errMsg)
	}

	runner := &CUDARunner{
		handle: handle,
		info: backend.DeviceInfo{
			Name:             C.GoString(&cInfo.name[0]),
			ComputeMajor:     int(cInfo.compute_major),
			ComputeMinor:     int(cInfo.compute_minor),
			TotalMemoryBytes: uint64(cInfo.total_memory_bytes),
			FreeMemoryBytes:  uint64(cInfo.free_memory_bytes),
		},
	}
	return runner, nil
}

func (r *CUDARunner) Warmup(_ context.Context, profile backend.WarmupProfile) error {
	status := C.gollmgo_backend_warmup(
		r.handle,
		C.int(profile.MaxBatchSize),
		C.int(profile.MaxSeqLen),
		C.int(profile.BlockSize),
	)
	if status != C.GOLLMGO_OK {
		return fmt.Errorf("cuda: warmup failed: %s", C.GoString(C.gollmgo_backend_last_error(r.handle)))
	}
	return nil
}

func (r *CUDARunner) Step(_ context.Context, _ *backend.Batch) (*backend.StepOutput, error) {
	// Step dispatch will be implemented in M5 (correct eager inference).
	return nil, fmt.Errorf("cuda: Step not yet implemented")
}

func (r *CUDARunner) Capabilities() backend.Capabilities {
	return backend.Capabilities{
		FP16:           true,
		BF16:           true,
		PagedAttention: false, // will be enabled in M6
	}
}

func (r *CUDARunner) Close() error {
	if r.handle != nil {
		status := C.gollmgo_backend_destroy(r.handle)
		r.handle = nil
		if status != C.GOLLMGO_OK {
			return fmt.Errorf("cuda: destroy failed (status %d)", int(status))
		}
	}
	return nil
}

// DeviceInfo returns the cached device information.
func (r *CUDARunner) DeviceInfo() backend.DeviceInfo {
	return r.info
}

// Ensure CUDARunner satisfies the Runner interface at compile time.
var _ backend.Runner = (*CUDARunner)(nil)

// Prevent unused import of unsafe.
var _ = unsafe.Pointer(nil)
