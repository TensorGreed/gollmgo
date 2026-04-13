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
	}
}
