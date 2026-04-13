package engine

import (
	"math"
	"testing"
)

func TestArgmax(t *testing.T) {
	cases := []struct {
		logits []float32
		want   int32
	}{
		{[]float32{1, 3, 2}, 1},
		{[]float32{5}, 0},
		{[]float32{-1, -2, -0.5}, 2},
		{[]float32{0, 0, 0, 1}, 3},
	}
	for _, tc := range cases {
		got := argmax(tc.logits)
		if got != tc.want {
			t.Errorf("argmax(%v) = %d, want %d", tc.logits, got, tc.want)
		}
	}
}

func TestGreedySampling(t *testing.T) {
	s := NewSampler(SamplingParams{Temperature: 0})
	logits := []float32{0.1, 0.9, 0.5, 0.3}

	// Greedy should always return the same result.
	for i := 0; i < 10; i++ {
		got := s.Sample(logits)
		if got != 1 {
			t.Fatalf("greedy sample %d: expected 1, got %d", i, got)
		}
	}
}

func TestDeterministicSampling(t *testing.T) {
	seed := uint64(42)
	logits := []float32{1.0, 1.0, 1.0, 1.0} // uniform

	// Two samplers with the same seed should produce identical sequences.
	s1 := NewSampler(SamplingParams{Temperature: 1.0, Seed: &seed})
	s2 := NewSampler(SamplingParams{Temperature: 1.0, Seed: &seed})

	for i := 0; i < 100; i++ {
		a := s1.Sample(logits)
		b := s2.Sample(logits)
		if a != b {
			t.Fatalf("determinism broken at step %d: %d != %d", i, a, b)
		}
	}
}

func TestTopKFiltering(t *testing.T) {
	logits := []float64{1.0, 5.0, 3.0, 2.0, 4.0}
	filtered := topKFilter(logits, 2)

	// Only indices 1 (5.0) and 4 (4.0) should survive.
	for i, v := range filtered {
		if i == 1 || i == 4 {
			if math.IsInf(v, -1) {
				t.Errorf("index %d should not be -inf", i)
			}
		} else {
			if !math.IsInf(v, -1) {
				t.Errorf("index %d should be -inf, got %f", i, v)
			}
		}
	}
}

func TestTopPFiltering(t *testing.T) {
	// Probabilities already normalized.
	probs := []float64{0.5, 0.3, 0.1, 0.05, 0.05}
	filtered := topPFilter(probs, 0.8)

	// Top-p=0.8: should keep indices 0 (0.5) and 1 (0.3) = 0.8.
	if filtered[0] == 0 {
		t.Error("index 0 should be in nucleus")
	}
	if filtered[1] == 0 {
		t.Error("index 1 should be in nucleus")
	}

	// Check renormalization.
	sum := 0.0
	for _, p := range filtered {
		sum += p
	}
	if math.Abs(sum-1.0) > 1e-6 {
		t.Errorf("nucleus probabilities should sum to 1, got %f", sum)
	}
}

func TestSoftmax(t *testing.T) {
	logits := []float64{1.0, 2.0, 3.0}
	probs := softmax(logits)

	// Check sums to 1.
	sum := 0.0
	for _, p := range probs {
		sum += p
	}
	if math.Abs(sum-1.0) > 1e-9 {
		t.Errorf("softmax should sum to 1, got %f", sum)
	}

	// Check ordering preserved.
	if probs[2] <= probs[1] || probs[1] <= probs[0] {
		t.Errorf("softmax should preserve ordering: %v", probs)
	}
}

func TestSoftmaxWithNegInf(t *testing.T) {
	logits := []float64{1.0, math.Inf(-1), 2.0}
	probs := softmax(logits)

	if probs[1] != 0 {
		t.Errorf("prob at -inf logit should be 0, got %f", probs[1])
	}
	sum := 0.0
	for _, p := range probs {
		sum += p
	}
	if math.Abs(sum-1.0) > 1e-9 {
		t.Errorf("softmax should sum to 1, got %f", sum)
	}
}

func TestTemperatureSampling(t *testing.T) {
	// High temperature should make distribution more uniform.
	seed := uint64(123)
	logits := []float32{10.0, 0.0, 0.0, 0.0}

	// Low temperature — almost always pick index 0.
	lowT := NewSampler(SamplingParams{Temperature: 0.01, Seed: &seed})
	counts := make([]int, 4)
	for i := 0; i < 1000; i++ {
		counts[lowT.Sample(logits)]++
	}
	if counts[0] < 990 {
		t.Errorf("low temp should strongly favor index 0, got counts %v", counts)
	}

	// High temperature — more spread.
	seed2 := uint64(456)
	highT := NewSampler(SamplingParams{Temperature: 10.0, Seed: &seed2})
	counts = make([]int, 4)
	for i := 0; i < 1000; i++ {
		counts[highT.Sample(logits)]++
	}
	// With temp=10, even weak logits should get some hits.
	nonZero := 0
	for _, c := range counts {
		if c > 0 {
			nonZero++
		}
	}
	if nonZero < 3 {
		t.Errorf("high temp should spread samples, got counts %v", counts)
	}
}

func TestTopKSampling(t *testing.T) {
	seed := uint64(789)
	logits := []float32{1.0, 1.0, 1.0, 1.0, 1.0}

	s := NewSampler(SamplingParams{Temperature: 1.0, TopK: 2, Seed: &seed})
	seen := make(map[int32]bool)
	for i := 0; i < 200; i++ {
		seen[s.Sample(logits)] = true
	}
	if len(seen) > 2 {
		t.Errorf("top-k=2 should produce at most 2 unique tokens, got %d", len(seen))
	}
}

func TestEmptyLogits(t *testing.T) {
	s := NewSampler(DefaultSamplingParams())
	got := s.Sample(nil)
	if got != 0 {
		t.Errorf("expected 0 for empty logits, got %d", got)
	}
}
