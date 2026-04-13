package kvcache

import (
	"fmt"
	"sync"
)

// BlockPool is a fixed-size pool of KV cache blocks with refcounting.
// All methods are safe for concurrent use.
type BlockPool struct {
	mu          sync.Mutex
	blockSize   int
	totalBlocks int
	refcounts   []int32
	freeList    []BlockID
}

// NewBlockPool creates a pool with the given number of blocks.
func NewBlockPool(totalBlocks, blockSize int) *BlockPool {
	pool := &BlockPool{
		blockSize:   blockSize,
		totalBlocks: totalBlocks,
		refcounts:   make([]int32, totalBlocks),
		freeList:    make([]BlockID, 0, totalBlocks),
	}
	for i := 0; i < totalBlocks; i++ {
		pool.freeList = append(pool.freeList, BlockID(i))
	}
	return pool
}

func (p *BlockPool) Allocate(n int) ([]BlockID, error) {
	p.mu.Lock()
	defer p.mu.Unlock()
	if n <= 0 {
		return nil, nil
	}
	if n > len(p.freeList) {
		return nil, fmt.Errorf("%w: need %d, have %d", ErrOutOfBlocks, n, len(p.freeList))
	}
	allocated := make([]BlockID, n)
	start := len(p.freeList) - n
	copy(allocated, p.freeList[start:])
	p.freeList = p.freeList[:start]
	for _, id := range allocated {
		p.refcounts[id] = 1
	}
	return allocated, nil
}

func (p *BlockPool) Free(blocks []BlockID) {
	p.mu.Lock()
	defer p.mu.Unlock()
	for _, id := range blocks {
		if id < 0 || int(id) >= p.totalBlocks {
			continue
		}
		if p.refcounts[id] > 0 {
			p.refcounts[id] = 0
			p.freeList = append(p.freeList, id)
		}
	}
}

func (p *BlockPool) Ref(block BlockID) {
	p.mu.Lock()
	defer p.mu.Unlock()
	if block >= 0 && int(block) < p.totalBlocks && p.refcounts[block] > 0 {
		p.refcounts[block]++
	}
}

func (p *BlockPool) Release(block BlockID) {
	p.mu.Lock()
	defer p.mu.Unlock()
	if block < 0 || int(block) >= p.totalBlocks {
		return
	}
	if p.refcounts[block] <= 0 {
		return
	}
	p.refcounts[block]--
	if p.refcounts[block] == 0 {
		p.freeList = append(p.freeList, block)
	}
}

func (p *BlockPool) NumFreeBlocks() int {
	p.mu.Lock()
	defer p.mu.Unlock()
	return len(p.freeList)
}

func (p *BlockPool) NumTotalBlocks() int { return p.totalBlocks }
func (p *BlockPool) BlockSize() int      { return p.blockSize }

func (p *BlockPool) Utilization() float64 {
	p.mu.Lock()
	defer p.mu.Unlock()
	if p.totalBlocks == 0 {
		return 0
	}
	return 1.0 - float64(len(p.freeList))/float64(p.totalBlocks)
}

// RefCount returns the refcount of a block (for testing/debugging).
func (p *BlockPool) RefCount(block BlockID) int32 {
	p.mu.Lock()
	defer p.mu.Unlock()
	if block < 0 || int(block) >= p.totalBlocks {
		return -1
	}
	return p.refcounts[block]
}

// Compile-time check.
var _ Manager = (*BlockPool)(nil)
