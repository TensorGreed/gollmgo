package scheduler

import (
	"context"
	"testing"
)

func TestPriorityAdmitsHighestFirst(t *testing.T) {
	sched := NewPriorityScheduler(SchedulerConfig{
		MaxBatchSize:     1,
		MaxTokenBudget:   10000,
		MaxQueueDepth:    100,
		PrefillChunkSize: 0,
		MaxWaitTicks:     100,
	})

	low := NewSequence("low", []int32{1}, 10)
	low.Priority = 1
	high := NewSequence("high", []int32{1}, 10)
	high.Priority = 10

	sched.Add(low)
	sched.Add(high)

	out, _ := sched.Tick(context.Background())
	if len(out.ScheduledSequences) != 1 {
		t.Fatalf("expected 1, got %d", len(out.ScheduledSequences))
	}
	if out.ScheduledSequences[0].ID != high.ID {
		t.Fatal("expected high-priority sequence to be admitted first")
	}
}

func TestPriorityFCFSTiebreak(t *testing.T) {
	sched := NewPriorityScheduler(SchedulerConfig{
		MaxBatchSize:     1,
		MaxTokenBudget:   10000,
		MaxQueueDepth:    100,
		PrefillChunkSize: 0,
		MaxWaitTicks:     100,
	})

	s1 := NewSequence("first", []int32{1}, 10)
	s1.Priority = 5
	s2 := NewSequence("second", []int32{1}, 10)
	s2.Priority = 5

	sched.Add(s1)
	sched.Add(s2)

	out, _ := sched.Tick(context.Background())
	if len(out.ScheduledSequences) != 1 {
		t.Fatalf("expected 1, got %d", len(out.ScheduledSequences))
	}
	// s1 has lower ID, so it should win the tiebreak.
	if out.ScheduledSequences[0].ID != s1.ID {
		t.Fatalf("expected seq %d (FCFS tiebreak), got %d", s1.ID, out.ScheduledSequences[0].ID)
	}
}

func TestPriorityAutoPreempt(t *testing.T) {
	sched := NewPriorityScheduler(SchedulerConfig{
		MaxBatchSize:     1,
		MaxTokenBudget:   10000,
		MaxQueueDepth:    100,
		PrefillChunkSize: 0,
		AutoPreempt:      true,
		MaxWaitTicks:     100,
	})

	// Fill the batch with a low-priority sequence.
	low := NewSequence("low", []int32{1}, 10)
	low.Priority = 1
	sched.Add(low)
	sched.Tick(context.Background())
	low.Transition(SeqDecoding)
	low.AppendToken(99)

	// Now add a high-priority sequence. Auto-preempt should evict low.
	high := NewSequence("high", []int32{1}, 10)
	high.Priority = 10
	sched.Add(high)

	// low should have been preempted back to waiting.
	if low.State != SeqWaiting {
		t.Fatalf("expected low to be WAITING (preempted), got %s", low.State)
	}
	if sched.ActiveLen() != 0 {
		t.Fatalf("expected 0 active after auto-preempt, got %d", sched.ActiveLen())
	}

	// Next tick: high should be admitted.
	out, _ := sched.Tick(context.Background())
	admitted := false
	for _, s := range out.ScheduledSequences {
		if s.ID == high.ID {
			admitted = true
		}
	}
	if !admitted {
		t.Fatal("expected high-priority sequence to be admitted after auto-preempt")
	}
}

func TestPriorityNoAutoPreemptWhenDisabled(t *testing.T) {
	sched := NewPriorityScheduler(SchedulerConfig{
		MaxBatchSize:     1,
		MaxTokenBudget:   10000,
		MaxQueueDepth:    100,
		PrefillChunkSize: 0,
		AutoPreempt:      false,
		MaxWaitTicks:     100,
	})

	low := NewSequence("low", []int32{1}, 10)
	low.Priority = 1
	sched.Add(low)
	sched.Tick(context.Background())
	low.Transition(SeqDecoding)
	low.AppendToken(99)

	high := NewSequence("high", []int32{1}, 10)
	high.Priority = 10
	sched.Add(high)

	// low should NOT have been preempted.
	if low.State != SeqDecoding {
		t.Fatalf("expected low to remain DECODING, got %s", low.State)
	}
	if sched.ActiveLen() != 1 {
		t.Fatalf("expected 1 active, got %d", sched.ActiveLen())
	}
}
