package scheduler

import (
	"context"
	"testing"
)

func TestPreemptRecompute(t *testing.T) {
	sched := NewSJFScheduler(SchedulerConfig{
		MaxBatchSize:     10,
		MaxTokenBudget:   10000,
		MaxQueueDepth:    100,
		PrefillChunkSize: 256,
		PreemptMode:      PreemptRecompute,
		MaxWaitTicks:     100,
	})

	seq := NewSequence("r1", make([]int32, 1024), 10)
	sched.Add(seq)
	sched.Tick(context.Background())
	seq.PrefillConsumed = 512
	seq.Transition(SeqDecoding)
	seq.AppendToken(99)

	if err := sched.Preempt(seq.ID); err != nil {
		t.Fatal(err)
	}

	// PrefillConsumed should be reset to 0 in recompute mode.
	if seq.PrefillConsumed != 0 {
		t.Fatalf("expected PrefillConsumed=0 after recompute preempt, got %d", seq.PrefillConsumed)
	}
	if seq.SwapState != nil {
		t.Fatal("expected nil SwapState in recompute mode")
	}
}

func TestPreemptSwap(t *testing.T) {
	sched := NewSJFScheduler(SchedulerConfig{
		MaxBatchSize:     10,
		MaxTokenBudget:   10000,
		MaxQueueDepth:    100,
		PrefillChunkSize: 256,
		PreemptMode:      PreemptSwap,
		MaxWaitTicks:     100,
	})

	seq := NewSequence("r1", make([]int32, 1024), 10)
	sched.Add(seq)
	sched.Tick(context.Background())
	seq.PrefillConsumed = 512
	seq.Transition(SeqDecoding)
	seq.AppendToken(99)
	seq.AppendToken(100)

	if err := sched.Preempt(seq.ID); err != nil {
		t.Fatal(err)
	}

	// SwapState should have saved progress.
	if seq.SwapState == nil {
		t.Fatal("expected SwapState to be saved in swap mode")
	}
	if seq.SwapState.SavedPrefillConsumed != 512 {
		t.Fatalf("expected saved prefill=512, got %d", seq.SwapState.SavedPrefillConsumed)
	}
	if seq.SwapState.SavedGeneratedLen != 2 {
		t.Fatalf("expected saved generated=2, got %d", seq.SwapState.SavedGeneratedLen)
	}
}

func TestSwapStateRestore(t *testing.T) {
	sched := NewSJFScheduler(SchedulerConfig{
		MaxBatchSize:     10,
		MaxTokenBudget:   10000,
		MaxQueueDepth:    100,
		PrefillChunkSize: 256,
		PreemptMode:      PreemptSwap,
		MaxWaitTicks:     100,
	})

	seq := NewSequence("r1", make([]int32, 1024), 10)
	sched.Add(seq)
	sched.Tick(context.Background())
	seq.PrefillConsumed = 512
	seq.Transition(SeqDecoding)
	seq.AppendToken(99)

	// Preempt with swap.
	sched.Preempt(seq.ID)

	// PrefillConsumed is still 512 (not reset in swap mode).
	if seq.PrefillConsumed != 512 {
		t.Fatalf("expected PrefillConsumed=512 after swap preempt, got %d", seq.PrefillConsumed)
	}

	// Re-admit via Tick: SwapState should be restored and cleared.
	sched.Tick(context.Background())
	if seq.PrefillConsumed != 512 {
		t.Fatalf("expected PrefillConsumed=512 after restore, got %d", seq.PrefillConsumed)
	}
	if seq.SwapState != nil {
		t.Fatal("SwapState should be cleared after restore")
	}
}
