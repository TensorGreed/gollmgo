// Package kvcache defines the KV cache manager interface.
// Implementation will come in M6; this is the contract.
package kvcache

// BlockID identifies a physical block in the KV pool.
type BlockID int32

// Manager is the interface for KV cache block management.
type Manager interface {
	// Allocate reserves n blocks and returns their IDs.
	Allocate(n int) ([]BlockID, error)
	// Free releases blocks back to the pool.
	Free(blocks []BlockID)
	// NumFreeBlocks returns the number of available blocks.
	NumFreeBlocks() int
	// BlockSize returns the number of tokens per block.
	BlockSize() int
}
