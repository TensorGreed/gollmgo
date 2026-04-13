package kvcache

import "testing"

func TestBlockTableAppendAndSlotMapping(t *testing.T) {
	pool := NewBlockPool(100, 4)
	bt := NewBlockTable(1, pool)

	// Append 6 tokens (should need 2 blocks of size 4).
	for i := 0; i < 6; i++ {
		_, err := bt.Append()
		if err != nil {
			t.Fatalf("append %d failed: %v", i, err)
		}
	}

	if bt.NumTokens() != 6 {
		t.Fatalf("expected 6 tokens, got %d", bt.NumTokens())
	}
	if bt.NumBlocks() != 2 {
		t.Fatalf("expected 2 blocks, got %d", bt.NumBlocks())
	}

	slots := bt.SlotMapping()
	if len(slots) != 6 {
		t.Fatalf("expected 6 slots, got %d", len(slots))
	}

	// First 4 tokens should be in block 0, next 2 in block 1.
	block0 := bt.PhysicalBlocks()[0]
	block1 := bt.PhysicalBlocks()[1]
	for i := 0; i < 4; i++ {
		expected := int32(block0)*4 + int32(i)
		if slots[i] != expected {
			t.Errorf("slot[%d] = %d, expected %d", i, slots[i], expected)
		}
	}
	for i := 4; i < 6; i++ {
		expected := int32(block1)*4 + int32(i-4)
		if slots[i] != expected {
			t.Errorf("slot[%d] = %d, expected %d", i, slots[i], expected)
		}
	}
}

func TestBlockTableAppendN(t *testing.T) {
	pool := NewBlockPool(100, 4)
	bt := NewBlockTable(1, pool)

	slots, err := bt.AppendN(10)
	if err != nil {
		t.Fatal(err)
	}
	if len(slots) != 10 {
		t.Fatalf("expected 10 slots, got %d", len(slots))
	}
	if bt.NumBlocks() != 3 { // 10 tokens / 4 block_size = 3 blocks
		t.Fatalf("expected 3 blocks, got %d", bt.NumBlocks())
	}
}

func TestBlockTableFree(t *testing.T) {
	pool := NewBlockPool(10, 4)
	bt := NewBlockTable(1, pool)

	bt.AppendN(8) // 2 blocks
	if pool.NumFreeBlocks() != 8 {
		t.Fatalf("expected 8 free, got %d", pool.NumFreeBlocks())
	}

	bt.Free()
	if pool.NumFreeBlocks() != 10 {
		t.Fatalf("expected 10 free after free, got %d", pool.NumFreeBlocks())
	}
	if bt.NumTokens() != 0 {
		t.Fatalf("expected 0 tokens after free, got %d", bt.NumTokens())
	}
}

func TestBlockTableOOM(t *testing.T) {
	pool := NewBlockPool(2, 4) // 2 blocks = 8 token slots
	bt := NewBlockTable(1, pool)

	_, err := bt.AppendN(8) // exactly fills pool
	if err != nil {
		t.Fatal(err)
	}

	_, err = bt.Append() // needs a 3rd block
	if err == nil {
		t.Fatal("expected OOM error")
	}
}

func TestBlockTableTruncate(t *testing.T) {
	pool := NewBlockPool(10, 4) // 4 tokens per block
	bt := NewBlockTable(1, pool)

	// Append 6 tokens (2 blocks: block0=[0..3], block1=[4,5])
	_, err := bt.AppendN(6)
	if err != nil {
		t.Fatal(err)
	}
	if bt.NumTokens() != 6 {
		t.Fatalf("expected 6 tokens, got %d", bt.NumTokens())
	}
	if bt.NumBlocks() != 2 {
		t.Fatalf("expected 2 blocks, got %d", bt.NumBlocks())
	}

	// Truncate 2 tokens → back to 4 tokens (block1 should be freed).
	bt.Truncate(2)
	if bt.NumTokens() != 4 {
		t.Fatalf("expected 4 tokens after truncate, got %d", bt.NumTokens())
	}
	if bt.NumBlocks() != 1 {
		t.Fatalf("expected 1 block after truncate (freed block1), got %d", bt.NumBlocks())
	}

	// Pool should have 9 free blocks (10 - 1 still in use).
	if pool.NumFreeBlocks() != 9 {
		t.Fatalf("expected 9 free blocks, got %d", pool.NumFreeBlocks())
	}
}

func TestBlockTableTruncateEmpty(t *testing.T) {
	pool := NewBlockPool(10, 4)
	bt := NewBlockTable(1, pool)
	bt.Truncate(5) // no-op on empty table
	if bt.NumTokens() != 0 {
		t.Fatalf("expected 0 tokens, got %d", bt.NumTokens())
	}
}
