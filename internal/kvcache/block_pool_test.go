package kvcache

import (
	"errors"
	"testing"
)

func TestNewBlockPool(t *testing.T) {
	pool := NewBlockPool(100, 16)
	if pool.NumFreeBlocks() != 100 {
		t.Fatalf("expected 100 free, got %d", pool.NumFreeBlocks())
	}
	if pool.NumTotalBlocks() != 100 {
		t.Fatalf("expected 100 total, got %d", pool.NumTotalBlocks())
	}
	if pool.BlockSize() != 16 {
		t.Fatalf("expected block size 16, got %d", pool.BlockSize())
	}
	if pool.Utilization() != 0 {
		t.Fatalf("expected 0 utilization, got %f", pool.Utilization())
	}
}

func TestAllocateAndFree(t *testing.T) {
	pool := NewBlockPool(10, 4)

	blocks, err := pool.Allocate(3)
	if err != nil {
		t.Fatal(err)
	}
	if len(blocks) != 3 {
		t.Fatalf("expected 3 blocks, got %d", len(blocks))
	}
	if pool.NumFreeBlocks() != 7 {
		t.Fatalf("expected 7 free, got %d", pool.NumFreeBlocks())
	}

	// Each block should have refcount 1.
	for _, b := range blocks {
		if pool.RefCount(b) != 1 {
			t.Fatalf("block %d refcount should be 1, got %d", b, pool.RefCount(b))
		}
	}

	pool.Free(blocks)
	if pool.NumFreeBlocks() != 10 {
		t.Fatalf("expected 10 free after free, got %d", pool.NumFreeBlocks())
	}
}

func TestAllocateExhaustion(t *testing.T) {
	pool := NewBlockPool(5, 4)

	_, err := pool.Allocate(5)
	if err != nil {
		t.Fatal(err)
	}
	if pool.NumFreeBlocks() != 0 {
		t.Fatalf("expected 0 free, got %d", pool.NumFreeBlocks())
	}

	_, err = pool.Allocate(1)
	if !errors.Is(err, ErrOutOfBlocks) {
		t.Fatalf("expected ErrOutOfBlocks, got %v", err)
	}
}

func TestRefcounting(t *testing.T) {
	pool := NewBlockPool(10, 4)

	blocks, _ := pool.Allocate(1)
	b := blocks[0]

	if pool.RefCount(b) != 1 {
		t.Fatalf("expected refcount 1, got %d", pool.RefCount(b))
	}

	// Ref -> refcount 2.
	pool.Ref(b)
	if pool.RefCount(b) != 2 {
		t.Fatalf("expected refcount 2, got %d", pool.RefCount(b))
	}

	// Release -> refcount 1, still allocated.
	pool.Release(b)
	if pool.RefCount(b) != 1 {
		t.Fatalf("expected refcount 1 after release, got %d", pool.RefCount(b))
	}
	if pool.NumFreeBlocks() != 9 {
		t.Fatalf("block should still be allocated, free=%d", pool.NumFreeBlocks())
	}

	// Release -> refcount 0, returns to free list.
	pool.Release(b)
	if pool.RefCount(b) != 0 {
		t.Fatalf("expected refcount 0, got %d", pool.RefCount(b))
	}
	if pool.NumFreeBlocks() != 10 {
		t.Fatalf("block should be free, free=%d", pool.NumFreeBlocks())
	}
}

func TestUtilization(t *testing.T) {
	pool := NewBlockPool(4, 8)

	pool.Allocate(2)
	u := pool.Utilization()
	if u != 0.5 {
		t.Fatalf("expected 0.5 utilization, got %f", u)
	}
}

func TestAllocateZero(t *testing.T) {
	pool := NewBlockPool(10, 4)
	blocks, err := pool.Allocate(0)
	if err != nil {
		t.Fatal(err)
	}
	if blocks != nil {
		t.Fatalf("expected nil for zero allocation, got %v", blocks)
	}
}

func TestFreeInvalidBlocks(t *testing.T) {
	pool := NewBlockPool(5, 4)
	// Should not panic.
	pool.Free([]BlockID{-1, 99, 100})
	if pool.NumFreeBlocks() != 5 {
		t.Fatalf("free count should be unchanged, got %d", pool.NumFreeBlocks())
	}
}

func TestBlockIDsAreUnique(t *testing.T) {
	pool := NewBlockPool(20, 4)
	blocks, _ := pool.Allocate(20)
	seen := make(map[BlockID]bool)
	for _, b := range blocks {
		if seen[b] {
			t.Fatalf("duplicate block ID %d", b)
		}
		seen[b] = true
	}
}
