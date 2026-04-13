// Package metrics provides Prometheus metrics and tracing for gollmgo.
// This is a skeleton; real collectors will be added as components come online.
package metrics

import "sync/atomic"

// Counters is a minimal set of atomic counters for core metrics.
// This avoids pulling in a Prometheus dependency until M2.
type Counters struct {
	RequestsTotal   atomic.Int64
	TokensGenerated atomic.Int64
	BatchesRun      atomic.Int64
}

// Global is the process-wide metrics instance.
var Global Counters
