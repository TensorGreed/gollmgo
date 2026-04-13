package api

import (
	"fmt"
	"net/http"

	"github.com/TensorGreed/gollmgo/internal/metrics"
)

// MetricsHandler returns basic Prometheus-style metrics.
// This is a placeholder until a real Prometheus registry is wired.
func (s *Server) MetricsHandler() http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/plain; version=0.0.4")
		fmt.Fprintf(w, "# HELP gollmgo_requests_total Total requests received.\n")
		fmt.Fprintf(w, "# TYPE gollmgo_requests_total counter\n")
		fmt.Fprintf(w, "gollmgo_requests_total %d\n", metrics.Global.RequestsTotal.Load())
		fmt.Fprintf(w, "# HELP gollmgo_tokens_generated_total Total tokens generated.\n")
		fmt.Fprintf(w, "# TYPE gollmgo_tokens_generated_total counter\n")
		fmt.Fprintf(w, "gollmgo_tokens_generated_total %d\n", metrics.Global.TokensGenerated.Load())
		fmt.Fprintf(w, "# HELP gollmgo_batches_run_total Total batches executed.\n")
		fmt.Fprintf(w, "# TYPE gollmgo_batches_run_total counter\n")
		fmt.Fprintf(w, "gollmgo_batches_run_total %d\n", metrics.Global.BatchesRun.Load())
		fmt.Fprintf(w, "# HELP gollmgo_kvcache_blocks_used Number of KV cache blocks in use.\n")
		fmt.Fprintf(w, "# TYPE gollmgo_kvcache_blocks_used gauge\n")
		fmt.Fprintf(w, "gollmgo_kvcache_blocks_used %d\n", metrics.Global.KVCacheBlocksUsed.Load())
		fmt.Fprintf(w, "# HELP gollmgo_kvcache_blocks_total Total KV cache blocks.\n")
		fmt.Fprintf(w, "# TYPE gollmgo_kvcache_blocks_total gauge\n")
		fmt.Fprintf(w, "gollmgo_kvcache_blocks_total %d\n", metrics.Global.KVCacheBlocksTotal.Load())
		fmt.Fprintf(w, "# HELP gollmgo_kvcache_utilization KV cache utilization ratio.\n")
		fmt.Fprintf(w, "# TYPE gollmgo_kvcache_utilization gauge\n")
		fmt.Fprintf(w, "gollmgo_kvcache_utilization %.4f\n", metrics.Global.KVCacheUtilization())
	}
}
