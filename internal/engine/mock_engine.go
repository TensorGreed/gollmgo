package engine

import (
	"context"
	"sync"
)

// MockEngine is a test double for the Engine interface.
type MockEngine struct {
	mu       sync.Mutex
	queue    []*Request
	results  []TokenResult
	stopped  bool
}

func (m *MockEngine) Enqueue(_ context.Context, req *Request) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.queue = append(m.queue, req)
	return nil
}

func (m *MockEngine) NextTokens(_ context.Context) ([]TokenResult, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	if len(m.results) == 0 {
		return nil, nil
	}
	out := m.results
	m.results = nil
	return out, nil
}

func (m *MockEngine) Stop() error {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.stopped = true
	return nil
}

// --- Test helpers ---

// PushResults stages token results for the next NextTokens call.
func (m *MockEngine) PushResults(results []TokenResult) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.results = append(m.results, results...)
}

// QueueLen returns the number of enqueued requests.
func (m *MockEngine) QueueLen() int {
	m.mu.Lock()
	defer m.mu.Unlock()
	return len(m.queue)
}
