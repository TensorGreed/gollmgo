package kvcache

import "fmt"

// BlockPool is a fixed-size pool of KV cache blocks with refcounting.
// Blocks are identified by integer IDs [0, totalBlocks).
// Refcount semantics:
//   - Allocate sets refcount to 1.
//   - Ref increments refcount (for prefix sharing).
//   - Release decrements refcount; block returns to free list at 0.
//   - Free is a bulk release that ignores refcounts (for preemption cleanup).
type BlockPool struct {
	blockSize   int
	totalBlocks int
	refcounts   []int32 // per-block refcount; 0 = free
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
	// All blocks start free.
	for i := 0; i < totalBlocks; i++ {
		pool.freeList = append(pool.freeList, BlockID(i))
	}
	return pool
}

func (p *BlockPool) Allocate(n int) ([]BlockID, error) {
	if n <= 0 {
		return nil, nil
	}
	if n > len(p.freeList) {
		return nil, fmt.Errorf("%w: need %d, have %d", ErrOutOfBlocks, n, len(p.freeList))
	}
	// Take from the end (stack-like, cache friendly).
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
	if block >= 0 && int(block) < p.totalBlocks && p.refcounts[block] > 0 {
		p.refcounts[block]++
	}
}

func (p *BlockPool) Release(block BlockID) {
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

func (p *BlockPool) NumFreeBlocks() int  { return len(p.freeList) }
func (p *BlockPool) NumTotalBlocks() int { return p.totalBlocks }
func (p *BlockPool) BlockSize() int      { return p.blockSize }

func (p *BlockPool) Utilization() float64 {
	if p.totalBlocks == 0 {
		return 0
	}
	return 1.0 - float64(len(p.freeList))/float64(p.totalBlocks)
}

// RefCount returns the refcount of a block (for testing/debugging).
func (p *BlockPool) RefCount(block BlockID) int32 {
	if block < 0 || int(block) >= p.totalBlocks {
		return -1
	}
	return p.refcounts[block]
}

// Compile-time check.
var _ Manager = (*BlockPool)(nil)
