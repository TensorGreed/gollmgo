// Package engine defines the inference orchestration interface.
package engine

import "context"

// Request represents a single inference request from the API layer.
type Request struct {
	ID        string
	TokenIDs  []int32
	MaxTokens int
	// Sampling parameters will be added in later milestones.
}

// TokenResult is one generated token delivered back to the caller.
type TokenResult struct {
	SequenceID uint64
	TokenID    int32
	Finished   bool
}

// Engine orchestrates forward passes and delivers tokens.
type Engine interface {
	// Enqueue submits a request for inference.
	Enqueue(ctx context.Context, req *Request) error
	// NextTokens blocks until the next batch of tokens is ready.
	NextTokens(ctx context.Context) ([]TokenResult, error)
	// Stop gracefully shuts down the engine.
	Stop() error
}
