package engine

import (
	"context"
	"sync"
	"sync/atomic"
)

// MockEngine is a test double for the Engine interface.
// Tests push results into it; Enqueue returns a handle whose channel
// receives those results.
type MockEngine struct {
	mu      sync.Mutex
	handles map[string]*mockHandle // requestID -> handle
	stopped bool
	seqGen  atomic.Uint64
}

type mockHandle struct {
	ch    chan TokenResult
	reqID string
}

func (m *MockEngine) Enqueue(_ context.Context, req *Request) (*RequestHandle, error) {
	m.mu.Lock()
	defer m.mu.Unlock()

	if m.handles == nil {
		m.handles = make(map[string]*mockHandle)
	}

	seqID := m.seqGen.Add(1)
	ch := make(chan TokenResult, 64)
	m.handles[req.ID] = &mockHandle{ch: ch, reqID: req.ID}

	return &RequestHandle{SeqID: seqID, Tokens: ch}, nil
}

func (m *MockEngine) Stop() error {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.stopped = true
	for _, h := range m.handles {
		close(h.ch)
	}
	m.handles = nil
	return nil
}

// --- Test helpers ---

// PendingRequestIDs returns the IDs of requests that have been enqueued
// but not yet finished. Safe to call without external locking.
func (m *MockEngine) PendingRequestIDs() []string {
	m.mu.Lock()
	defer m.mu.Unlock()
	ids := make([]string, 0, len(m.handles))
	for id := range m.handles {
		ids = append(ids, id)
	}
	return ids
}

// PushResultsTo sends token results to a specific request's channel.
func (m *MockEngine) PushResultsTo(requestID string, results []TokenResult) {
	m.mu.Lock()
	h, ok := m.handles[requestID]
	if !ok {
		m.mu.Unlock()
		return
	}
	m.mu.Unlock()

	for _, r := range results {
		h.ch <- r
	}
	// If the last result is finished, close the channel.
	if len(results) > 0 && results[len(results)-1].Finished {
		m.mu.Lock()
		close(h.ch)
		delete(m.handles, requestID)
		m.mu.Unlock()
	}
}
