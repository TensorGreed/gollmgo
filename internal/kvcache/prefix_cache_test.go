package kvcache

import (
	"testing"

	"github.com/TensorGreed/gollmgo/internal/metrics"
)

func TestChainHashDeterministic(t *testing.T) {
	tokens := []int32{1, 2, 3, 4}
	h1 := ChainHash(0, tokens)
	h2 := ChainHash(0, tokens)
	if h1 != h2 {
		t.Fatalf("expected same hash, got %x vs %x", h1, h2)
	}
}

func TestChainHashDifferentTokens(t *testing.T) {
	h1 := ChainHash(0, []int32{1, 2, 3, 4})
	h2 := ChainHash(0, []int32{5, 6, 7, 8})
	if h1 == h2 {
		t.Fatal("expected different hashes for different tokens")
	}
}

func TestChainHashChaining(t *testing.T) {
	// Same first block, different second block → different chain hash.
	first := ChainHash(0, []int32{1, 2, 3, 4})
	h1 := ChainHash(first, []int32{5, 6, 7, 8})
	h2 := ChainHash(first, []int32{9, 10, 11, 12})
	if h1 == h2 {
		t.Fatal("expected different chain hashes for different second blocks")
	}
}

func TestPrefixCacheLookupMiss(t *testing.T) {
	pool := NewBlockPool(10, 4)
	cache := NewPrefixCache(pool)

	_, ok := cache.Lookup(12345)
	if ok {
		t.Fatal("expected miss on empty cache")
	}
}

func TestPrefixCacheInsertAndHit(t *testing.T) {
	pool := NewBlockPool(10, 4)
	cache := NewPrefixCache(pool)

	// Allocate a block and insert it.
	blocks, _ := pool.Allocate(1)
	blockID := blocks[0]
	hash := ChainHash(0, []int32{1, 2, 3, 4})
	cache.Insert(hash, blockID)

	if cache.Len() != 1 {
		t.Fatalf("expected 1 cached, got %d", cache.Len())
	}

	// Lookup should hit and add a ref.
	got, ok := cache.Lookup(hash)
	if !ok {
		t.Fatal("expected hit")
	}
	if got != blockID {
		t.Fatalf("expected block %d, got %d", blockID, got)
	}

	// Block should have 3 refs: 1 original + 1 cache + 1 lookup.
	if rc := pool.RefCount(blockID); rc != 3 {
		t.Fatalf("expected refcount 3, got %d", rc)
	}
}

func TestPrefixCacheEviction(t *testing.T) {
	pool := NewBlockPool(4, 4)
	cache := NewPrefixCache(pool)

	// Allocate all 4 blocks and cache them.
	blocks, _ := pool.Allocate(4)
	for i, b := range blocks {
		hash := ChainHash(uint64(i), []int32{int32(i), int32(i + 1), int32(i + 2), int32(i + 3)})
		cache.Insert(hash, b)
	}
	if cache.Len() != 4 {
		t.Fatalf("expected 4 cached, got %d", cache.Len())
	}

	// Pool is empty (all allocated + cache refs).
	// Release the original refs so only cache holds them.
	for _, b := range blocks {
		pool.Release(b)
	}

	// Now pool should still be empty since cache holds refs.
	if pool.NumFreeBlocks() != 0 {
		t.Fatalf("expected 0 free, got %d", pool.NumFreeBlocks())
	}

	// Evict 2 blocks.
	evicted := cache.Evict(2)
	if evicted != 2 {
		t.Fatalf("expected 2 evicted, got %d", evicted)
	}
	if cache.Len() != 2 {
		t.Fatalf("expected 2 remaining, got %d", cache.Len())
	}
	if pool.NumFreeBlocks() != 2 {
		t.Fatalf("expected 2 free after eviction, got %d", pool.NumFreeBlocks())
	}
}

func TestPrefixCacheMetrics(t *testing.T) {
	// Reset global metrics for this test.
	metrics.Global.KVCacheHits.Store(0)
	metrics.Global.KVCacheLookups.Store(0)

	pool := NewBlockPool(10, 4)
	cache := NewPrefixCache(pool)

	hash := ChainHash(0, []int32{1, 2, 3, 4})

	// Miss.
	cache.Lookup(hash)
	if metrics.Global.KVCacheLookups.Load() != 1 {
		t.Fatal("expected 1 lookup")
	}
	if metrics.Global.KVCacheHits.Load() != 0 {
		t.Fatal("expected 0 hits")
	}

	// Insert and hit.
	blocks, _ := pool.Allocate(1)
	cache.Insert(hash, blocks[0])
	cache.Lookup(hash)
	if metrics.Global.KVCacheLookups.Load() != 2 {
		t.Fatal("expected 2 lookups")
	}
	if metrics.Global.KVCacheHits.Load() != 1 {
		t.Fatal("expected 1 hit")
	}
}

func TestBlockTableMatchPrefix(t *testing.T) {
	pool := NewBlockPool(10, 4)
	cache := NewPrefixCache(pool)

	// Simulate a completed request: allocate blocks, donate to cache.
	tokenIDs := []int32{1, 2, 3, 4, 5, 6, 7, 8} // 2 blocks
	blocks, _ := pool.Allocate(2)
	cache.DonateBlocks(tokenIDs, blocks, 4)
	// Release original refs (cache holds its own).
	pool.Release(blocks[0])
	pool.Release(blocks[1])

	// New request with same prefix.
	bt := NewBlockTable(1, pool)
	matched := bt.MatchPrefix(tokenIDs, cache)
	if matched != 8 {
		t.Fatalf("expected 8 tokens matched, got %d", matched)
	}
	if bt.NumTokens() != 8 {
		t.Fatalf("expected 8 tokens in block table, got %d", bt.NumTokens())
	}

	// The matched blocks should be the same physical blocks.
	slots := bt.SlotMapping()
	if len(slots) != 8 {
		t.Fatalf("expected 8 slots, got %d", len(slots))
	}
}

// TestDonateThenFreeKeepsCacheValid simulates the engine cleanup path:
// a sequence donates its completed blocks to the prefix cache and then
// frees its block table. The cache must keep ownership of those blocks so
// later prefix lookups return live, exclusive references rather than
// reclaimed-and-reallocated blocks.
func TestDonateThenFreeKeepsCacheValid(t *testing.T) {
	pool := NewBlockPool(4, 4)
	cache := NewPrefixCache(pool)

	tokenIDs := []int32{1, 2, 3, 4, 5, 6, 7, 8} // 2 complete blocks
	bt := NewBlockTable(1, pool)
	if _, err := bt.AppendN(8); err != nil {
		t.Fatal(err)
	}
	donated := bt.PhysicalBlocks()

	cache.DonateBlocks(tokenIDs, donated, 4)

	// Engine cleanup: free the sequence's table. Cache must retain ownership.
	bt.Free()

	// None of the donated blocks should be back on the free list.
	if pool.NumFreeBlocks() != 2 {
		t.Fatalf("expected 2 free (only the unused half of the pool), got %d", pool.NumFreeBlocks())
	}
	for _, b := range donated {
		if rc := pool.RefCount(b); rc != 1 {
			t.Fatalf("donated block %d should have refcount 1 (cache only), got %d", b, rc)
		}
	}

	// Subsequent allocation must not return a cached block.
	allocated, err := pool.Allocate(2)
	if err != nil {
		t.Fatal(err)
	}
	cachedSet := map[BlockID]bool{donated[0]: true, donated[1]: true}
	for _, b := range allocated {
		if cachedSet[b] {
			t.Fatalf("pool returned cache-owned block %d", b)
		}
	}

	// A fresh request should still get a prefix hit on the donated blocks.
	bt2 := NewBlockTable(2, pool)
	matched := bt2.MatchPrefix(tokenIDs, cache)
	if matched != 8 {
		t.Fatalf("expected 8-token prefix hit after donor freed, got %d", matched)
	}
}

// TestPrefixCacheCapEvictsLRU verifies the bounded-size cache evicts the LRU
// entry on insert once the cap is reached.
func TestPrefixCacheCapEvictsLRU(t *testing.T) {
	pool := NewBlockPool(8, 4)
	cache := NewPrefixCacheWithCap(pool, 2)

	blocks, _ := pool.Allocate(3)
	hashes := []uint64{
		ChainHash(0, []int32{1, 1, 1, 1}),
		ChainHash(0, []int32{2, 2, 2, 2}),
		ChainHash(0, []int32{3, 3, 3, 3}),
	}

	cache.Insert(hashes[0], blocks[0])
	cache.Insert(hashes[1], blocks[1])
	if cache.Len() != 2 {
		t.Fatalf("expected cache len 2, got %d", cache.Len())
	}

	// Touch blocks[0] so blocks[1] becomes LRU.
	if _, ok := cache.Lookup(hashes[0]); !ok {
		t.Fatal("expected hit on hashes[0]")
	}
	pool.Release(blocks[0]) // drop the lookup ref

	cache.Insert(hashes[2], blocks[2])
	if cache.Len() != 2 {
		t.Fatalf("expected cache len 2 after cap-driven evict, got %d", cache.Len())
	}
	if _, ok := cache.Lookup(hashes[1]); ok {
		t.Fatal("expected hashes[1] (LRU) to be evicted")
	}
}

func TestDonateBlocksPartial(t *testing.T) {
	pool := NewBlockPool(10, 4)
	cache := NewPrefixCache(pool)

	// 5 tokens = 1 complete block + 1 incomplete.
	tokenIDs := []int32{1, 2, 3, 4, 5}
	blocks, _ := pool.Allocate(2)
	cache.DonateBlocks(tokenIDs, blocks, 4)

	// Only 1 block should be cached (the complete one).
	if cache.Len() != 1 {
		t.Fatalf("expected 1 cached block, got %d", cache.Len())
	}
}
