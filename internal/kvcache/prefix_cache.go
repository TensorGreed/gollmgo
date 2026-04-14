package kvcache

import (
	"container/list"
	"encoding/binary"
	"hash/fnv"
	"sync"

	"github.com/TensorGreed/gollmgo/internal/metrics"
)

// PrefixCache maps chain-hashed token block content to physical BlockIDs.
// It enables reuse of KV cache blocks across requests that share a prefix
// (e.g. the same system prompt). Uses LRU eviction when the pool is full.
//
// When MaxBlocks > 0, Insert evicts the LRU entry once the cache reaches the
// cap. When MaxBlocks <= 0 the cache is unbounded (callers must drive
// eviction explicitly via Evict).
type PrefixCache struct {
	mu        sync.Mutex
	pool      *BlockPool
	maxBlocks int
	entries   map[uint64]BlockID      // chainHash → physical block
	hashFor   map[BlockID]uint64      // block → chainHash (for cheap reverse lookup on evict)
	lruList   *list.List              // front = most recently used
	lruMap    map[BlockID]*list.Element // block → LRU element
}

// NewPrefixCache creates an unbounded prefix cache backed by the given block pool.
func NewPrefixCache(pool *BlockPool) *PrefixCache {
	return NewPrefixCacheWithCap(pool, 0)
}

// NewPrefixCacheWithCap creates a prefix cache that evicts the LRU entry
// once the cached block count exceeds maxBlocks. maxBlocks <= 0 disables
// the automatic cap (use Evict manually).
func NewPrefixCacheWithCap(pool *BlockPool, maxBlocks int) *PrefixCache {
	return &PrefixCache{
		pool:      pool,
		maxBlocks: maxBlocks,
		entries:   make(map[uint64]BlockID),
		hashFor:   make(map[BlockID]uint64),
		lruList:   list.New(),
		lruMap:    make(map[BlockID]*list.Element),
	}
}

// ChainHash computes a chain hash for a block of tokens.
// prevHash is the hash of the preceding block (0 for the first block).
func ChainHash(prevHash uint64, tokens []int32) uint64 {
	h := fnv.New64a()
	var buf [8]byte
	binary.LittleEndian.PutUint64(buf[:], prevHash)
	h.Write(buf[:])
	for _, t := range tokens {
		binary.LittleEndian.PutUint32(buf[:4], uint32(t))
		h.Write(buf[:4])
	}
	return h.Sum64()
}

// Lookup checks if a block with the given chain hash is cached.
// On hit, increments the block's refcount (caller must Release when done)
// and promotes it in the LRU.
func (c *PrefixCache) Lookup(chainHash uint64) (BlockID, bool) {
	c.mu.Lock()
	defer c.mu.Unlock()

	metrics.Global.KVCacheLookups.Add(1)

	blockID, ok := c.entries[chainHash]
	if !ok {
		return -1, false
	}

	metrics.Global.KVCacheHits.Add(1)

	// Add a ref for the caller.
	c.pool.Ref(blockID)

	// Promote in LRU.
	if elem, exists := c.lruMap[blockID]; exists {
		c.lruList.MoveToFront(elem)
	}

	return blockID, true
}

// Insert adds a block to the cache. The cache takes ownership of one ref
// (the caller's ref stays valid). If the hash already exists, this is a no-op.
func (c *PrefixCache) Insert(chainHash uint64, blockID BlockID) {
	c.mu.Lock()
	defer c.mu.Unlock()

	if _, exists := c.entries[chainHash]; exists {
		return // already cached
	}

	// The cache takes a ref.
	c.pool.Ref(blockID)
	c.entries[chainHash] = blockID
	c.hashFor[blockID] = chainHash
	elem := c.lruList.PushFront(blockID)
	c.lruMap[blockID] = elem

	// Enforce capacity (LRU eviction).
	if c.maxBlocks > 0 {
		for c.lruList.Len() > c.maxBlocks {
			c.evictOneLocked()
		}
	}
}

// evictOneLocked drops the LRU entry. Caller must hold c.mu.
func (c *PrefixCache) evictOneLocked() bool {
	elem := c.lruList.Back()
	if elem == nil {
		return false
	}
	blockID := elem.Value.(BlockID)
	c.lruList.Remove(elem)
	delete(c.lruMap, blockID)
	if hash, ok := c.hashFor[blockID]; ok {
		delete(c.entries, hash)
		delete(c.hashFor, blockID)
	}
	c.pool.Release(blockID)
	return true
}

// Evict removes up to n least-recently-used blocks from the cache,
// releasing their refs. Returns the number actually evicted.
func (c *PrefixCache) Evict(n int) int {
	c.mu.Lock()
	defer c.mu.Unlock()

	evicted := 0
	for evicted < n {
		if !c.evictOneLocked() {
			break
		}
		evicted++
	}
	return evicted
}

// Len returns the number of cached blocks.
func (c *PrefixCache) Len() int {
	c.mu.Lock()
	defer c.mu.Unlock()
	return len(c.entries)
}

// DonateBlocks adds completed sequence blocks to the cache for potential reuse.
// tokenIDs are the full prompt tokens, blockSize is the pool's block size.
// Only complete blocks are cached. The caller still owns its own refs.
func (c *PrefixCache) DonateBlocks(tokenIDs []int32, blocks []BlockID, blockSize int) {
	chainHash := uint64(0)
	for i, blockID := range blocks {
		start := i * blockSize
		end := start + blockSize
		if end > len(tokenIDs) {
			break // incomplete block, don't cache
		}
		chainHash = ChainHash(chainHash, tokenIDs[start:end])
		c.Insert(chainHash, blockID)
	}
}
