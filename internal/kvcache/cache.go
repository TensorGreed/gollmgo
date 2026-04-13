// Package kvcache implements the paged KV cache manager.
package kvcache

import "errors"

// BlockID identifies a physical block in the KV pool.
type BlockID int32

var (
	ErrOutOfBlocks = errors.New("kvcache: no free blocks available")
	ErrInvalidBlock = errors.New("kvcache: invalid block ID")
)

// Manager is the interface for KV cache block management.
type Manager interface {
	// Allocate reserves n blocks and returns their IDs.
	Allocate(n int) ([]BlockID, error)
	// Free releases blocks back to the pool.
	Free(blocks []BlockID)
	// Ref increments the refcount for a block (prefix sharing).
	Ref(block BlockID)
	// Release decrements the refcount; frees the block if it hits zero.
	Release(block BlockID)
	// NumFreeBlocks returns the number of available blocks.
	NumFreeBlocks() int
	// NumTotalBlocks returns the total pool size.
	NumTotalBlocks() int
	// BlockSize returns the number of tokens per block.
	BlockSize() int
	// Utilization returns the fraction of blocks in use.
	Utilization() float64
}
