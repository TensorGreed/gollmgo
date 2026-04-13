package metrics

import (
	"math"
	"testing"
)

func TestLatencyHistogramEmpty(t *testing.T) {
	var h LatencyHistogram
	snap := h.Snapshot()
	if snap.Count != 0 {
		t.Fatalf("expected 0 count, got %d", snap.Count)
	}
	if snap.P50 != 0 {
		t.Fatalf("expected 0 P50, got %f", snap.P50)
	}
}

func TestLatencyHistogramPercentiles(t *testing.T) {
	var h LatencyHistogram
	for i := 1; i <= 100; i++ {
		h.Record(float64(i))
	}

	snap := h.Snapshot()
	if snap.Count != 100 {
		t.Fatalf("expected 100 count, got %d", snap.Count)
	}
	if snap.Sum != 5050 {
		t.Fatalf("expected sum 5050, got %f", snap.Sum)
	}
	if snap.P50 != 50 {
		t.Fatalf("expected P50=50, got %f", snap.P50)
	}
	if snap.P95 != 95 {
		t.Fatalf("expected P95=95, got %f", snap.P95)
	}
	if snap.P99 != 99 {
		t.Fatalf("expected P99=99, got %f", snap.P99)
	}

	// After snapshot, buffer is reset.
	snap2 := h.Snapshot()
	if snap2.Count != 0 {
		t.Fatalf("expected 0 count after reset, got %d", snap2.Count)
	}
}

func TestLatencyHistogramPeek(t *testing.T) {
	var h LatencyHistogram
	h.Record(10)
	h.Record(20)
	h.Record(30)

	peek := h.Peek()
	if peek.Count != 3 {
		t.Fatalf("expected 3, got %d", peek.Count)
	}

	// Peek should not reset.
	peek2 := h.Peek()
	if peek2.Count != 3 {
		t.Fatalf("expected 3 after peek, got %d", peek2.Count)
	}
}

func TestKVCacheUtilization(t *testing.T) {
	var c Counters
	c.KVCacheBlocksTotal.Store(100)
	c.KVCacheBlocksUsed.Store(50)
	if math.Abs(c.KVCacheUtilization()-0.5) > 0.001 {
		t.Fatalf("expected 0.5, got %f", c.KVCacheUtilization())
	}
}

func TestHitRatesZero(t *testing.T) {
	var c Counters
	if c.KVCacheHitRate() != 0 {
		t.Fatal("expected 0 hit rate with no lookups")
	}
	if c.GraphCacheHitRate() != 0 {
		t.Fatal("expected 0 graph hit rate with no lookups")
	}
}
