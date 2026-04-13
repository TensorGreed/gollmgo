package scheduler

import (
	"errors"
	"testing"
)

func TestNewSequence(t *testing.T) {
	prompt := []int32{10, 20, 30}
	seq := NewSequence("req-1", prompt, 100)

	if seq.State != SeqWaiting {
		t.Fatalf("expected WAITING, got %s", seq.State)
	}
	if seq.PromptLen != 3 {
		t.Fatalf("expected prompt len 3, got %d", seq.PromptLen)
	}
	if seq.GeneratedLen != 0 {
		t.Fatalf("expected generated len 0, got %d", seq.GeneratedLen)
	}
	if seq.TotalLen() != 3 {
		t.Fatalf("expected total len 3, got %d", seq.TotalLen())
	}
	if seq.RequestID != "req-1" {
		t.Fatalf("expected request ID req-1, got %s", seq.RequestID)
	}

	// Ensure prompt is copied, not shared.
	prompt[0] = 999
	if seq.TokenIDs[0] == 999 {
		t.Fatal("prompt slice should be copied, not shared")
	}
}

func TestSequenceIDsAreUnique(t *testing.T) {
	s1 := NewSequence("a", nil, 10)
	s2 := NewSequence("b", nil, 10)
	if s1.ID == s2.ID {
		t.Fatal("sequence IDs should be unique")
	}
}

func TestHappyPathTransitions(t *testing.T) {
	seq := NewSequence("req-1", []int32{1}, 10)

	steps := []SeqState{SeqPrefilling, SeqDecoding, SeqFinished}
	for _, next := range steps {
		if err := seq.Transition(next); err != nil {
			t.Fatalf("transition to %s failed: %v", next, err)
		}
		if seq.State != next {
			t.Fatalf("expected %s, got %s", next, seq.State)
		}
	}
	if !seq.IsFinished() {
		t.Fatal("expected sequence to be finished")
	}
}

func TestPreemptionPath(t *testing.T) {
	seq := NewSequence("req-1", []int32{1}, 10)

	// WAITING -> PREFILLING -> DECODING -> PREEMPTED -> WAITING -> PREFILLING
	transitions := []SeqState{
		SeqPrefilling, SeqDecoding, SeqPreempted, SeqWaiting, SeqPrefilling,
	}
	for _, next := range transitions {
		if err := seq.Transition(next); err != nil {
			t.Fatalf("transition to %s failed: %v", next, err)
		}
	}
	if seq.State != SeqPrefilling {
		t.Fatalf("expected PREFILLING after re-admission, got %s", seq.State)
	}
}

func TestInvalidTransition(t *testing.T) {
	seq := NewSequence("req-1", []int32{1}, 10)

	// WAITING -> DECODING should fail (must go through PREFILLING).
	err := seq.Transition(SeqDecoding)
	if err == nil {
		t.Fatal("expected error for WAITING -> DECODING")
	}
	if !errors.Is(err, ErrInvalidTransition) {
		t.Fatalf("expected ErrInvalidTransition, got %v", err)
	}
	// State should not have changed.
	if seq.State != SeqWaiting {
		t.Fatalf("state should remain WAITING, got %s", seq.State)
	}
}

func TestFinishedIsTerminal(t *testing.T) {
	seq := NewSequence("req-1", []int32{1}, 10)
	seq.Transition(SeqPrefilling)
	seq.Transition(SeqDecoding)
	seq.Transition(SeqFinished)

	err := seq.Transition(SeqWaiting)
	if !errors.Is(err, ErrSequenceFinished) {
		t.Fatalf("expected ErrSequenceFinished, got %v", err)
	}
}

func TestAppendToken(t *testing.T) {
	seq := NewSequence("req-1", []int32{1, 2}, 10)
	seq.AppendToken(42)
	seq.AppendToken(43)

	if seq.GeneratedLen != 2 {
		t.Fatalf("expected generated len 2, got %d", seq.GeneratedLen)
	}
	if seq.TotalLen() != 4 {
		t.Fatalf("expected total len 4, got %d", seq.TotalLen())
	}
	if seq.TokenIDs[2] != 42 || seq.TokenIDs[3] != 43 {
		t.Fatalf("unexpected token IDs: %v", seq.TokenIDs)
	}
}

func TestSeqStateString(t *testing.T) {
	cases := []struct {
		state SeqState
		want  string
	}{
		{SeqWaiting, "WAITING"},
		{SeqPrefilling, "PREFILLING"},
		{SeqDecoding, "DECODING"},
		{SeqFinished, "FINISHED"},
		{SeqPreempted, "PREEMPTED"},
		{SeqState(99), "UNKNOWN(99)"},
	}
	for _, tc := range cases {
		if got := tc.state.String(); got != tc.want {
			t.Errorf("SeqState(%d).String() = %q, want %q", int(tc.state), got, tc.want)
		}
	}
}
