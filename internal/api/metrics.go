package api

import (
	"fmt"
	"net/http"

	"github.com/TensorGreed/gollmgo/internal/metrics"
)

// MetricsHandler returns Prometheus-style metrics.
func (s *Server) MetricsHandler() http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/plain; version=0.0.4")

		// --- Core counters ---
		gauge(w, "gollmgo_requests_total", "Total requests received.",
			"counter", metrics.Global.RequestsTotal.Load())
		gauge(w, "gollmgo_tokens_generated_total", "Total tokens generated.",
			"counter", metrics.Global.TokensGenerated.Load())
		gauge(w, "gollmgo_batches_run_total", "Total batches executed.",
			"counter", metrics.Global.BatchesRun.Load())

		// --- KV cache ---
		gauge(w, "gollmgo_kvcache_blocks_used", "KV cache blocks in use.",
			"gauge", metrics.Global.KVCacheBlocksUsed.Load())
		gauge(w, "gollmgo_kvcache_blocks_total", "Total KV cache blocks.",
			"gauge", metrics.Global.KVCacheBlocksTotal.Load())
		gaugeF(w, "gollmgo_kvcache_utilization", "KV cache utilization ratio.",
			metrics.Global.KVCacheUtilization())

		// --- Prefix cache (Phase 2 M4) ---
		gauge(w, "gollmgo_kvcache_hits_total", "Prefix cache hits.",
			"counter", metrics.Global.KVCacheHits.Load())
		gauge(w, "gollmgo_kvcache_lookups_total", "Prefix cache lookups.",
			"counter", metrics.Global.KVCacheLookups.Load())
		gaugeF(w, "gollmgo_kvcache_hit_rate", "Prefix cache hit ratio.",
			metrics.Global.KVCacheHitRate())

		// --- CUDA graph cache (Phase 2 M3) ---
		gauge(w, "gollmgo_graph_cache_hits_total", "CUDA graph cache hits.",
			"counter", metrics.Global.GraphCacheHits.Load())
		gauge(w, "gollmgo_graph_cache_lookups_total", "CUDA graph cache lookups.",
			"counter", metrics.Global.GraphCacheLookups.Load())
		gaugeF(w, "gollmgo_graph_cache_hit_rate", "CUDA graph cache hit ratio.",
			metrics.Global.GraphCacheHitRate())

		// --- Scheduler ---
		gauge(w, "gollmgo_scheduler_queue_depth", "Sequences waiting in scheduler queue.",
			"gauge", metrics.Global.SchedulerQueueDepth.Load())
		gauge(w, "gollmgo_scheduler_active_count", "Sequences actively running.",
			"gauge", metrics.Global.SchedulerActiveCount.Load())

		// --- Latency histograms (peek, non-destructive) ---
		ttft := metrics.Global.TTFT.Peek()
		gaugeF(w, "gollmgo_ttft_p50_ms", "Server-side TTFT P50 (ms).", ttft.P50)
		gaugeF(w, "gollmgo_ttft_p95_ms", "Server-side TTFT P95 (ms).", ttft.P95)
		gaugeF(w, "gollmgo_ttft_p99_ms", "Server-side TTFT P99 (ms).", ttft.P99)
		gauge(w, "gollmgo_ttft_count", "TTFT sample count.",
			"counter", ttft.Count)

		itl := metrics.Global.ITL.Peek()
		gaugeF(w, "gollmgo_itl_p50_ms", "Server-side ITL P50 (ms).", itl.P50)
		gaugeF(w, "gollmgo_itl_p95_ms", "Server-side ITL P95 (ms).", itl.P95)
		gaugeF(w, "gollmgo_itl_p99_ms", "Server-side ITL P99 (ms).", itl.P99)
		gauge(w, "gollmgo_itl_count", "ITL sample count.",
			"counter", itl.Count)
	}
}

func gauge(w http.ResponseWriter, name, help, typ string, val int64) {
	fmt.Fprintf(w, "# HELP %s %s\n# TYPE %s %s\n%s %d\n", name, help, name, typ, name, val)
}

func gaugeF(w http.ResponseWriter, name, help string, val float64) {
	fmt.Fprintf(w, "# HELP %s %s\n# TYPE %s gauge\n%s %.4f\n", name, help, name, name, val)
}
