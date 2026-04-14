//go:build gpu

package cuda

/*
#cgo CFLAGS: -I${SRCDIR}/../../../kernels -I/usr/local/cuda/targets/sbsa-linux/include
#cgo LDFLAGS: -L/usr/local/cuda/targets/sbsa-linux/lib -lcudart
#include "gollmgo_kvcache.h"
#include <cuda_runtime.h>
*/
import "C"

import (
	"context"
	"fmt"
	"unsafe"

	"github.com/TensorGreed/gollmgo/internal/backend"
)

type kvHostSnapshot struct {
	data      []byte
	numBlocks int
	released  bool
}

func (s *kvHostSnapshot) NumBlocks() int { return s.numBlocks }

func (s *kvHostSnapshot) BytesOnHost() int64 {
	if s == nil || s.released {
		return 0
	}
	return int64(len(s.data))
}

func (s *kvHostSnapshot) Release() error {
	if s == nil || s.released {
		return nil
	}
	s.data = nil
	s.released = true
	return nil
}

func (r *CUDARunnerWithModel) Capabilities() backend.Capabilities {
	caps := r.CUDARunner.Capabilities()
	if r.KVCache != nil {
		caps.KVSwap = true
	}
	return caps
}

func (r *CUDARunnerWithModel) SnapshotKV(ctx context.Context, blockIDs []int32) (backend.KVSnapshot, error) {
	if r.KVCache == nil {
		return nil, fmt.Errorf("cuda: KV swap requires a KV cache")
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if len(blockIDs) == 0 {
		return &kvHostSnapshot{numBlocks: 0}, nil
	}

	bytesPerSlot := r.KVCache.numKVHeads * r.KVCache.headDim * 2
	bytesPerBlock := r.KVCache.blockSize * bytesPerSlot
	totalBytes := len(blockIDs) * r.KVCache.numLayers * 2 * bytesPerBlock
	buf := make([]byte, totalBytes)

	offset := 0
	for _, blockID := range blockIDs {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		if err := r.validateBlockID(blockID); err != nil {
			return nil, err
		}
		slotByteOffset := int(blockID) * r.KVCache.blockSize * bytesPerSlot
		for layer := 0; layer < r.KVCache.numLayers; layer++ {
			kBase := C.gollmgo_kvcache_k_layer_ptr(r.KVCache.handle, C.int(layer))
			vBase := C.gollmgo_kvcache_v_layer_ptr(r.KVCache.handle, C.int(layer))
			if kBase == nil || vBase == nil {
				return nil, fmt.Errorf("cuda: missing KV layer pointer for layer %d", layer)
			}
			if err := copyDeviceToHost(buf[offset:offset+bytesPerBlock], unsafe.Add(unsafe.Pointer(kBase), slotByteOffset)); err != nil {
				return nil, fmt.Errorf("cuda: snapshot K layer %d block %d: %w", layer, blockID, err)
			}
			offset += bytesPerBlock
			if err := copyDeviceToHost(buf[offset:offset+bytesPerBlock], unsafe.Add(unsafe.Pointer(vBase), slotByteOffset)); err != nil {
				return nil, fmt.Errorf("cuda: snapshot V layer %d block %d: %w", layer, blockID, err)
			}
			offset += bytesPerBlock
		}
	}

	return &kvHostSnapshot{data: buf, numBlocks: len(blockIDs)}, nil
}

func (r *CUDARunnerWithModel) RestoreKV(ctx context.Context, snap backend.KVSnapshot, blockIDs []int32) error {
	if r.KVCache == nil {
		return fmt.Errorf("cuda: KV swap requires a KV cache")
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	hostSnap, ok := snap.(*kvHostSnapshot)
	if !ok {
		return fmt.Errorf("cuda: unsupported KV snapshot type %T", snap)
	}
	if hostSnap.released {
		return fmt.Errorf("cuda: KV snapshot already released")
	}
	if len(blockIDs) != hostSnap.numBlocks {
		return fmt.Errorf("cuda: snapshot has %d blocks, restore requested %d", hostSnap.numBlocks, len(blockIDs))
	}
	if len(blockIDs) == 0 {
		return nil
	}

	bytesPerSlot := r.KVCache.numKVHeads * r.KVCache.headDim * 2
	bytesPerBlock := r.KVCache.blockSize * bytesPerSlot
	expectedBytes := len(blockIDs) * r.KVCache.numLayers * 2 * bytesPerBlock
	if len(hostSnap.data) != expectedBytes {
		return fmt.Errorf("cuda: snapshot size mismatch: have %d bytes, want %d", len(hostSnap.data), expectedBytes)
	}

	offset := 0
	for _, blockID := range blockIDs {
		if err := ctx.Err(); err != nil {
			return err
		}
		if err := r.validateBlockID(blockID); err != nil {
			return err
		}
		slotByteOffset := int(blockID) * r.KVCache.blockSize * bytesPerSlot
		for layer := 0; layer < r.KVCache.numLayers; layer++ {
			kBase := C.gollmgo_kvcache_k_layer_ptr(r.KVCache.handle, C.int(layer))
			vBase := C.gollmgo_kvcache_v_layer_ptr(r.KVCache.handle, C.int(layer))
			if kBase == nil || vBase == nil {
				return fmt.Errorf("cuda: missing KV layer pointer for layer %d", layer)
			}
			if err := copyHostToDevice(unsafe.Add(unsafe.Pointer(kBase), slotByteOffset), hostSnap.data[offset:offset+bytesPerBlock]); err != nil {
				return fmt.Errorf("cuda: restore K layer %d block %d: %w", layer, blockID, err)
			}
			offset += bytesPerBlock
			if err := copyHostToDevice(unsafe.Add(unsafe.Pointer(vBase), slotByteOffset), hostSnap.data[offset:offset+bytesPerBlock]); err != nil {
				return fmt.Errorf("cuda: restore V layer %d block %d: %w", layer, blockID, err)
			}
			offset += bytesPerBlock
		}
	}

	return nil
}

func (r *CUDARunnerWithModel) validateBlockID(blockID int32) error {
	if blockID < 0 {
		return fmt.Errorf("cuda: invalid block id %d", blockID)
	}
	startSlot := int(blockID) * r.KVCache.blockSize
	endSlot := startSlot + r.KVCache.blockSize
	if endSlot > r.KVCache.numSlots {
		return fmt.Errorf("cuda: block id %d exceeds KV cache capacity", blockID)
	}
	return nil
}

func copyDeviceToHost(dst []byte, src unsafe.Pointer) error {
	if len(dst) == 0 {
		return nil
	}
	status := C.cudaMemcpy(unsafe.Pointer(&dst[0]), src, C.size_t(len(dst)), C.cudaMemcpyDeviceToHost)
	if status != C.cudaSuccess {
		return fmt.Errorf(C.GoString(C.cudaGetErrorString(status)))
	}
	return nil
}

func copyHostToDevice(dst unsafe.Pointer, src []byte) error {
	if len(src) == 0 {
		return nil
	}
	status := C.cudaMemcpy(dst, unsafe.Pointer(&src[0]), C.size_t(len(src)), C.cudaMemcpyHostToDevice)
	if status != C.cudaSuccess {
		return fmt.Errorf(C.GoString(C.cudaGetErrorString(status)))
	}
	return nil
}

var _ backend.KVSnapshot = (*kvHostSnapshot)(nil)
var _ backend.KVSwapper = (*CUDARunnerWithModel)(nil)
