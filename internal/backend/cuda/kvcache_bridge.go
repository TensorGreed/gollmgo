//go:build gpu

package cuda

/*
#include "gollmgo_kvcache.h"
*/
import "C"

import (
	"fmt"
	"unsafe"
)

// CUDAKVCache wraps the GPU-resident KV cache.
type CUDAKVCache struct {
	handle     C.gollmgo_kvcache_t
	numLayers  int
	numSlots   int
	blockSize  int
	numKVHeads int
	headDim    int
}

// NewCUDAKVCache creates a KV cache on the GPU.
func NewCUDAKVCache(runner *CUDARunner, numLayers, numSlots, numKVHeads, headDim, blockSize int) (*CUDAKVCache, error) {
	var handle C.gollmgo_kvcache_t
	status := C.gollmgo_kvcache_create(
		runner.handle,
		C.int(numLayers),
		C.int(numSlots),
		C.int(numKVHeads),
		C.int(headDim),
		&handle,
	)
	if status != C.GOLLMGO_OK {
		return nil, fmt.Errorf("cuda: kvcache_create failed (status %d)", int(status))
	}
	return &CUDAKVCache{
		handle:     handle,
		numLayers:  numLayers,
		numSlots:   numSlots,
		blockSize:  blockSize,
		numKVHeads: numKVHeads,
		headDim:    headDim,
	}, nil
}

// Write stores K and V data into cache slots.
// k, v: []uint16 (FP16 data), slot_mapping: physical slot per token.
func (c *CUDAKVCache) Write(k, v []uint16, slotMapping []int32, nTokens int) error {
	if nTokens <= 0 {
		return nil
	}
	status := C.gollmgo_kvcache_write(
		c.handle,
		unsafe.Pointer(&k[0]),
		unsafe.Pointer(&v[0]),
		(*C.int32_t)(unsafe.Pointer(&slotMapping[0])),
		C.int(nTokens),
	)
	if status != C.GOLLMGO_OK {
		return fmt.Errorf("cuda: kvcache_write failed (status %d)", int(status))
	}
	return nil
}

// Attention runs paged attention v1 for decode queries.
// q: []uint16 (FP16), returns output []uint16 (FP16).
func (c *CUDAKVCache) Attention(q []uint16, seqLens, slotTables []int32,
	nQueries, numHeads, maxSeqLen int, scale float32) ([]uint16, error) {

	outSize := nQueries * numHeads * c.headDim
	out := make([]uint16, outSize)

	status := C.gollmgo_kvcache_attention(
		c.handle,
		unsafe.Pointer(&q[0]),
		unsafe.Pointer(&out[0]),
		(*C.int32_t)(unsafe.Pointer(&seqLens[0])),
		(*C.int32_t)(unsafe.Pointer(&slotTables[0])),
		C.int(nQueries),
		C.int(numHeads),
		C.int(maxSeqLen),
		C.float(scale),
	)
	if status != C.GOLLMGO_OK {
		return nil, fmt.Errorf("cuda: kvcache_attention failed (status %d)", int(status))
	}
	return out, nil
}

// Close destroys the KV cache.
func (c *CUDAKVCache) Close() error {
	if c.handle != nil {
		C.gollmgo_kvcache_destroy(c.handle)
		c.handle = nil
	}
	return nil
}
