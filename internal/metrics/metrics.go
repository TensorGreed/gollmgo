// Package metrics provides Prometheus metrics and tracing for gollmgo.
package metrics

import "sync/atomic"

// Counters is a minimal set of atomic counters for core metrics.
type Counters struct {
	RequestsTotal   atomic.Int64
	TokensGenerated atomic.Int64
	BatchesRun      atomic.Int64

	// KV cache metrics.
	KVCacheBlocksUsed  atomic.Int64
	KVCacheBlocksTotal atomic.Int64
}

// KVCacheUtilization returns the fraction of KV cache blocks in use.
func (c *Counters) KVCacheUtilization() float64 {
	total := c.KVCacheBlocksTotal.Load()
	if total == 0 {
		return 0
	}
	return float64(c.KVCacheBlocksUsed.Load()) / float64(total)
}

// Global is the process-wide metrics instance.
var Global Counters
