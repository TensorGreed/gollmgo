// Package metrics provides Prometheus metrics and tracing for gollmgo.
package metrics

import (
	"math"
	"sort"
	"sync"
	"sync/atomic"
)

// Counters is the core metrics set. All fields are lock-free atomics.
type Counters struct {
	RequestsTotal   atomic.Int64
	TokensGenerated atomic.Int64
	BatchesRun      atomic.Int64

	// KV cache metrics.
	KVCacheBlocksUsed  atomic.Int64
	KVCacheBlocksTotal atomic.Int64

	// Scheduler queue metrics.
	SchedulerQueueDepth atomic.Int64 // sequences waiting to be scheduled
	SchedulerActiveCount atomic.Int64 // sequences currently in prefill/decode

	// Prefix cache metrics (placeholder until M4).
	KVCacheHits    atomic.Int64
	KVCacheLookups atomic.Int64

	// CUDA graph metrics (placeholder until M3).
	GraphCacheHits    atomic.Int64
	GraphCacheLookups atomic.Int64

	// Latency histograms (server-side).
	TTFT LatencyHistogram // time-to-first-token (prefill duration)
	ITL  LatencyHistogram // inter-token latency (per decode step)
}

// KVCacheUtilization returns the fraction of KV cache blocks in use.
func (c *Counters) KVCacheUtilization() float64 {
	total := c.KVCacheBlocksTotal.Load()
	if total == 0 {
		return 0
	}
	return float64(c.KVCacheBlocksUsed.Load()) / float64(total)
}

// KVCacheHitRate returns the prefix cache hit ratio (0 if no lookups).
func (c *Counters) KVCacheHitRate() float64 {
	lookups := c.KVCacheLookups.Load()
	if lookups == 0 {
		return 0
	}
	return float64(c.KVCacheHits.Load()) / float64(lookups)
}

// GraphCacheHitRate returns the CUDA graph cache hit ratio.
func (c *Counters) GraphCacheHitRate() float64 {
	lookups := c.GraphCacheLookups.Load()
	if lookups == 0 {
		return 0
	}
	return float64(c.GraphCacheHits.Load()) / float64(lookups)
}

// Global is the process-wide metrics instance.
var Global Counters

// LatencyHistogram collects latency samples for percentile reporting.
// Thread-safe. Designed for low-contention metric collection, not hot-path.
type LatencyHistogram struct {
	mu      sync.Mutex
	samples []float64 // milliseconds
	count   int64
	sum     float64
}

// Record adds a latency sample in milliseconds.
func (h *LatencyHistogram) Record(ms float64) {
	h.mu.Lock()
	h.samples = append(h.samples, ms)
	h.count++
	h.sum += ms
	h.mu.Unlock()
}

// Snapshot returns sorted percentile values and count.
// Resets the internal buffer after snapshotting.
type HistogramSnapshot struct {
	Count int64
	Sum   float64
	P50   float64
	P95   float64
	P99   float64
}

func (h *LatencyHistogram) Snapshot() HistogramSnapshot {
	h.mu.Lock()
	samples := h.samples
	count := h.count
	sum := h.sum
	h.samples = nil
	h.count = 0
	h.sum = 0
	h.mu.Unlock()

	if len(samples) == 0 {
		return HistogramSnapshot{}
	}

	sort.Float64s(samples)
	return HistogramSnapshot{
		Count: count,
		Sum:   sum,
		P50:   percentile(samples, 0.50),
		P95:   percentile(samples, 0.95),
		P99:   percentile(samples, 0.99),
	}
}

// Peek returns percentiles without resetting.
func (h *LatencyHistogram) Peek() HistogramSnapshot {
	h.mu.Lock()
	samples := make([]float64, len(h.samples))
	copy(samples, h.samples)
	count := h.count
	sum := h.sum
	h.mu.Unlock()

	if len(samples) == 0 {
		return HistogramSnapshot{}
	}

	sort.Float64s(samples)
	return HistogramSnapshot{
		Count: count,
		Sum:   sum,
		P50:   percentile(samples, 0.50),
		P95:   percentile(samples, 0.95),
		P99:   percentile(samples, 0.99),
	}
}

func percentile(sorted []float64, p float64) float64 {
	if len(sorted) == 0 {
		return 0
	}
	idx := int(math.Floor(float64(len(sorted)-1) * p))
	if idx >= len(sorted) {
		idx = len(sorted) - 1
	}
	return sorted[idx]
}
