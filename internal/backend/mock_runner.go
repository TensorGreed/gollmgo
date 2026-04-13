package backend

import "context"

// MockRunner is a test double for the Runner interface.
// It returns deterministic outputs without GPU access.
type MockRunner struct {
	WarmupCalled bool
	StepCount    int
	Closed       bool
	// StepFunc, if set, overrides default step behavior.
	StepFunc func(ctx context.Context, batch *Batch) (*StepOutput, error)
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
	return out, nil
}

func (m *MockRunner) Capabilities() Capabilities {
	return Capabilities{FP16: true}
}

func (m *MockRunner) Close() error {
	m.Closed = true
	return nil
}
