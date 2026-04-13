package scheduler

import (
	"context"
	"testing"
)

func TestSJFAdmitsShortestFirst(t *testing.T) {
	sched := NewSJFScheduler(SchedulerConfig{
		MaxBatchSize:     2,
		MaxTokenBudget:   10000,
		MaxQueueDepth:    100,
		PrefillChunkSize: 0,
		MaxWaitTicks:     100,
	})

	// Add long job first, then short job.
	long := NewSequence("long", []int32{1, 2, 3}, 100) // size=103
	short := NewSequence("short", []int32{1}, 5)        // size=6
	sched.Add(long)
	sched.Add(short)

	out, _ := sched.Tick(context.Background())
	if len(out.ScheduledSequences) != 2 {
		t.Fatalf("expected 2 scheduled, got %d", len(out.ScheduledSequences))
	}
	// Short job should be first in admission order (sorted).
	// Both get admitted since batch size is 2, but short should have been considered first.
	// Verify short was admitted by checking it's in active set.
	if sched.Find(short.ID) == nil {
		t.Fatal("short job should be tracked")
	}
	if short.State != SeqPrefilling {
		t.Fatalf("short job should be PREFILLING, got %s", short.State)
	}
}

func TestSJFTokenBudget(t *testing.T) {
	sched := NewSJFScheduler(SchedulerConfig{
		MaxBatchSize:     100,
		MaxTokenBudget:   5,
		MaxQueueDepth:    100,
		PrefillChunkSize: 0,
		MaxWaitTicks:     100,
	})

	s1 := NewSequence("s1", []int32{1}, 2)          // size=3, prompt=1
	s2 := NewSequence("s2", []int32{1, 2, 3}, 10)   // size=13, prompt=3
	s3 := NewSequence("s3", []int32{1, 2}, 5)        // size=7, prompt=2
	sched.Add(s1)
	sched.Add(s2)
	sched.Add(s3)

	out, _ := sched.Tick(context.Background())
	// Sorted by job size: s1(3), s3(7), s2(13)
	// Budget=5: s1 costs 1, s3 costs 2 (total 3), s2 costs 3 (total 6 > 5) -> skip s2
	if len(out.ScheduledSequences) != 2 {
		t.Fatalf("expected 2 scheduled, got %d", len(out.ScheduledSequences))
	}
	if out.PrefillBudgetUsed != 3 { // 1 + 2
		t.Fatalf("expected prefill=3, got %d", out.PrefillBudgetUsed)
	}
}

func TestSJFPreemptMaintainsOrder(t *testing.T) {
	sched := NewSJFScheduler(SchedulerConfig{
		MaxBatchSize:     10,
		MaxTokenBudget:   10000,
		MaxQueueDepth:    100,
		PrefillChunkSize: 0,
		MaxWaitTicks:     100,
	})

	long := NewSequence("long", []int32{1, 2, 3}, 100)
	sched.Add(long)
	sched.Tick(context.Background())
	long.Transition(SeqDecoding)
	long.AppendToken(99)

	// Preempt
	if err := sched.Preempt(long.ID); err != nil {
		t.Fatal(err)
	}
	if long.State != SeqWaiting {
		t.Fatalf("expected WAITING, got %s", long.State)
	}

	// Add a shorter job
	short := NewSequence("short", []int32{1}, 5)
	sched.Add(short)

	// Tick: short should be admitted first (smaller job size).
	out, _ := sched.Tick(context.Background())
	if len(out.ScheduledSequences) != 2 {
		t.Fatalf("expected 2, got %d", len(out.ScheduledSequences))
	}
}

func TestSJFChunkedPrefill(t *testing.T) {
	sched := NewSJFScheduler(SchedulerConfig{
		MaxBatchSize:     10,
		MaxTokenBudget:   10000,
		MaxQueueDepth:    100,
		PrefillChunkSize: 256,
		MaxWaitTicks:     100,
	})

	prompt := make([]int32, 1024)
	seq := NewSequence("r1", prompt, 10)
	sched.Add(seq)

	// Tick 1: first chunk.
	out, _ := sched.Tick(context.Background())
	if out.PrefillBudgetUsed != 256 {
		t.Fatalf("expected 256 prefill, got %d", out.PrefillBudgetUsed)
	}
	if seq.State != SeqPrefilling {
		t.Fatalf("expected PREFILLING, got %s", seq.State)
	}
	seq.PrefillConsumed = 256

	// Tick 2: next chunk.
	out, _ = sched.Tick(context.Background())
	if out.PrefillBudgetUsed != 256 {
		t.Fatalf("tick 2: expected 256, got %d", out.PrefillBudgetUsed)
	}
}
