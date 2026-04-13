package kvcache

import "fmt"

// BlockTable maps a sequence's logical block indices to physical BlockIDs.
// Each logical block holds `blockSize` tokens of KV data.
type BlockTable struct {
	seqID    uint64
	blocks   []BlockID // logical index -> physical BlockID
	numTokens int      // number of tokens stored so far
	pool     *BlockPool
}

// NewBlockTable creates an empty block table for a sequence.
func NewBlockTable(seqID uint64, pool *BlockPool) *BlockTable {
	return &BlockTable{
		seqID: seqID,
		pool:  pool,
	}
}

// Append ensures there is space for one more token, allocating a new block if needed.
// Returns the physical slot index (blockID * blockSize + offset).
func (bt *BlockTable) Append() (int32, error) {
	blockSize := bt.pool.BlockSize()
	offset := bt.numTokens % blockSize

	// Need a new block?
	if offset == 0 {
		blocks, err := bt.pool.Allocate(1)
		if err != nil {
			return 0, fmt.Errorf("block_table seq %d: %w", bt.seqID, err)
		}
		bt.blocks = append(bt.blocks, blocks[0])
	}

	logicalBlock := bt.numTokens / blockSize
	physicalBlock := bt.blocks[logicalBlock]
	slot := int32(physicalBlock)*int32(blockSize) + int32(offset)
	bt.numTokens++
	return slot, nil
}

// AppendN ensures space for n tokens, allocating blocks as needed.
// Returns the slot indices for all n positions.
func (bt *BlockTable) AppendN(n int) ([]int32, error) {
	slots := make([]int32, n)
	for i := 0; i < n; i++ {
		slot, err := bt.Append()
		if err != nil {
			return nil, err
		}
		slots[i] = slot
	}
	return slots, nil
}

// SlotMapping returns the physical slot for each token position [0, numTokens).
func (bt *BlockTable) SlotMapping() []int32 {
	blockSize := bt.pool.BlockSize()
	slots := make([]int32, bt.numTokens)
	for i := 0; i < bt.numTokens; i++ {
		logicalBlock := i / blockSize
		offset := i % blockSize
		physicalBlock := bt.blocks[logicalBlock]
		slots[i] = int32(physicalBlock)*int32(blockSize) + int32(offset)
	}
	return slots
}

// PhysicalBlocks returns the list of physical blocks owned by this table.
func (bt *BlockTable) PhysicalBlocks() []BlockID {
	result := make([]BlockID, len(bt.blocks))
	copy(result, bt.blocks)
	return result
}

// NumTokens returns the number of tokens tracked.
func (bt *BlockTable) NumTokens() int { return bt.numTokens }

// NumBlocks returns the number of allocated blocks.
func (bt *BlockTable) NumBlocks() int { return len(bt.blocks) }

// Free releases all blocks back to the pool.
func (bt *BlockTable) Free() {
	bt.pool.Free(bt.blocks)
	bt.blocks = nil
	bt.numTokens = 0
}
