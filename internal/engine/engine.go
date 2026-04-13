// Package engine defines the inference orchestration interface.
package engine

import "context"

// Request represents a single inference request from the API layer.
type Request struct {
	ID        string
	TokenIDs  []int32
	MaxTokens int
}

// TokenResult is one generated token delivered back to the caller.
type TokenResult struct {
	SequenceID uint64
	TokenID    int32
	Finished   bool
	Err        error // non-nil if the sequence failed (e.g. step error)
}

// RequestHandle is the per-request delivery mechanism returned by Enqueue.
// The caller reads from Tokens until a TokenResult with Finished==true or
// Err!=nil arrives, or until context cancellation.
type RequestHandle struct {
	SeqID  uint64
	Tokens <-chan TokenResult
}

// Engine orchestrates forward passes and delivers tokens.
type Engine interface {
	// Enqueue submits a request for inference and returns a handle for
	// receiving that request's tokens. Each request gets an isolated
	// channel — no global queue, no cross-request interference.
	Enqueue(ctx context.Context, req *Request) (*RequestHandle, error)
	// Stop gracefully shuts down the engine.
	Stop() error
}
