package backend

import (
	"context"
	"fmt"
	"sync"
	"sync/atomic"
)

// MockRunner is a test double for the Runner interface.
// It returns deterministic outputs without GPU access.
type MockRunner struct {
	WarmupCalled bool
	StepCount    int
	Closed       bool
	// StepFunc, if set, overrides default step behavior.
	StepFunc func(ctx context.Context, batch *Batch) (*StepOutput, error)

	// KV-swap state (only used when SnapshotKV/RestoreKV are exercised).
	swapMu    sync.Mutex
	swapStore map[int64][]int32 // snapshot id → captured block IDs (stand-in for real KV bytes)
	swapID    atomic.Int64
}

func (m *MockRunner) Warmup(_ context.Context, _ WarmupProfile) error {
	m.WarmupCalled = true
	return nil
}

func (m *MockRunner) Step(ctx context.Context, batch *Batch) (*StepOutput, error) {
	m.StepCount++
	if m.StepFunc != nil {
		return m.StepFunc(ctx, batch)
	}
	// Default: return a single logit per sequence (token ID 1).
	out := &StepOutput{
		Logits: make([][]float32, len(batch.SequenceIDs)),
	}
	for i := range batch.SequenceIDs {
		out.Logits[i] = []float32{0.0, 1.0} // argmax = 1
	}
	// If any drafts are supplied, also populate per-position logits so the
	// engine can exercise the verify path.
	if len(batch.DraftTokens) == len(batch.SequenceIDs) {
		out.LogitsPerPosition = make([][][]float32, len(batch.SequenceIDs))
		for i, drafts := range batch.DraftTokens {
			if len(drafts) == 0 {
				continue
			}
			pos := make([][]float32, 1+len(drafts))
			// Current position: same as Logits[i].
			pos[0] = out.Logits[i]
			// Draft positions: accept everything by returning argmax = draft token.
			for d, tok := range drafts {
				row := make([]float32, int(tok)+2)
				row[tok] = 1.0
				pos[d+1] = row
			}
			out.LogitsPerPosition[i] = pos
		}
	}
	return out, nil
}

func (m *MockRunner) Capabilities() Capabilities {
	return Capabilities{FP16: true, SpeculativeDecoding: true, KVSwap: true}
}

func (m *MockRunner) Close() error {
	m.Closed = true
	return nil
}

// --- KVSwapper (mock) ---

// mockKVSnapshot is the host-side stand-in for a real GPU→host KV snapshot.
// It just records which block IDs were captured; the mock engine has no real
// KV bytes to move. Round-trip correctness is proven by the engine tests.
type mockKVSnapshot struct {
	id       int64
	blockIDs []int32
	owner    *MockRunner
	released bool
}

func (s *mockKVSnapshot) NumBlocks() int    { return len(s.blockIDs) }
func (s *mockKVSnapshot) BytesOnHost() int64 { return 0 } // mock has no real bytes

func (s *mockKVSnapshot) Release() error {
	if s.released {
		return fmt.Errorf("mock kv snapshot: double release")
	}
	s.released = true
	s.owner.swapMu.Lock()
	delete(s.owner.swapStore, s.id)
	s.owner.swapMu.Unlock()
	return nil
}

// SnapshotKV captures block IDs so RestoreKV can verify the round-trip.
func (m *MockRunner) SnapshotKV(_ context.Context, blockIDs []int32) (KVSnapshot, error) {
	m.swapMu.Lock()
	defer m.swapMu.Unlock()
	if m.swapStore == nil {
		m.swapStore = make(map[int64][]int32)
	}
	id := m.swapID.Add(1)
	snapCopy := make([]int32, len(blockIDs))
	copy(snapCopy, blockIDs)
	m.swapStore[id] = snapCopy
	return &mockKVSnapshot{id: id, blockIDs: snapCopy, owner: m}, nil
}

// RestoreKV validates the count matches. Real backend would cudaMemcpy here.
func (m *MockRunner) RestoreKV(_ context.Context, snap KVSnapshot, blockIDs []int32) error {
	ms, ok := snap.(*mockKVSnapshot)
	if !ok {
		return fmt.Errorf("mock runner: foreign snapshot type %T", snap)
	}
	if ms.released {
		return fmt.Errorf("mock runner: snapshot already released")
	}
	if ms.NumBlocks() != len(blockIDs) {
		return fmt.Errorf("mock runner: block count mismatch (snap=%d, target=%d)",
			ms.NumBlocks(), len(blockIDs))
	}
	return nil
}

// Compile-time check.
var _ Runner = (*MockRunner)(nil)
var _ KVSwapper = (*MockRunner)(nil)
