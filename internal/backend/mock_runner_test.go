package backend

import (
	"context"
	"testing"
)

func TestMockRunnerWarmup(t *testing.T) {
	r := &MockRunner{}
	if err := r.Warmup(context.Background(), WarmupProfile{}); err != nil {
		t.Fatalf("warmup failed: %v", err)
	}
	if !r.WarmupCalled {
		t.Fatal("expected WarmupCalled to be true")
	}
}

func TestMockRunnerStep(t *testing.T) {
	r := &MockRunner{}
	batch := &Batch{
		SequenceIDs: []uint64{1, 2},
		TokenIDs:    []int32{10, 20},
	}
	out, err := r.Step(context.Background(), batch)
	if err != nil {
		t.Fatalf("step failed: %v", err)
	}
	if len(out.Logits) != 2 {
		t.Fatalf("expected 2 logit rows, got %d", len(out.Logits))
	}
	if r.StepCount != 1 {
		t.Fatalf("expected step count 1, got %d", r.StepCount)
	}
}

func TestMockRunnerClose(t *testing.T) {
	r := &MockRunner{}
	if err := r.Close(); err != nil {
		t.Fatalf("close failed: %v", err)
	}
	if !r.Closed {
		t.Fatal("expected Closed to be true")
	}
}

func TestMockRunnerCapabilities(t *testing.T) {
	r := &MockRunner{}
	caps := r.Capabilities()
	if !caps.FP16 {
		t.Fatal("expected FP16 capability")
	}
}

func TestMockRunnerCustomStepFunc(t *testing.T) {
	r := &MockRunner{
		StepFunc: func(_ context.Context, b *Batch) (*StepOutput, error) {
			out := &StepOutput{
				Logits: make([][]float32, len(b.SequenceIDs)),
			}
			for i := range b.SequenceIDs {
				out.Logits[i] = []float32{1.0, 0.0} // argmax = 0
			}
			return out, nil
		},
	}
	batch := &Batch{SequenceIDs: []uint64{1}}
	out, _ := r.Step(context.Background(), batch)
	if out.Logits[0][0] != 1.0 {
		t.Fatal("custom StepFunc not used")
	}
}
