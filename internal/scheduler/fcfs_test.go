package scheduler

import (
	"context"
	"errors"
	"testing"
)

func TestFCFSBasicAdmission(t *testing.T) {
	sched := NewFCFSScheduler(DefaultFCFSConfig())

	seq := NewSequence("r1", []int32{1, 2, 3}, 10)
	if err := sched.Add(seq); err != nil {
		t.Fatal(err)
	}
	if sched.Len() != 1 {
		t.Fatalf("expected len 1, got %d", sched.Len())
	}

	out, err := sched.Tick(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if len(out.ScheduledSequences) != 1 {
		t.Fatalf("expected 1 scheduled, got %d", len(out.ScheduledSequences))
	}
	if out.PrefillBudgetUsed != 3 {
		t.Fatalf("expected prefill budget 3, got %d", out.PrefillBudgetUsed)
	}
	if seq.State != SeqPrefilling {
		t.Fatalf("expected PREFILLING, got %s", seq.State)
	}
}

func TestFCFSContinuousBatching(t *testing.T) {
	sched := NewFCFSScheduler(FCFSConfig{
		MaxBatchSize:   10,
		MaxTokenBudget: 100,
		MaxQueueDepth:  100,
	})

	// Add 3 sequences.
	s1 := NewSequence("r1", []int32{1, 2}, 5)
	s2 := NewSequence("r2", []int32{3, 4, 5}, 5)
	s3 := NewSequence("r3", []int32{6}, 5)
	sched.Add(s1)
	sched.Add(s2)
	sched.Add(s3)

	// Tick 1: all should be admitted for prefill.
	out, _ := sched.Tick(context.Background())
	if len(out.ScheduledSequences) != 3 {
		t.Fatalf("tick 1: expected 3 scheduled, got %d", len(out.ScheduledSequences))
	}
	if out.PrefillBudgetUsed != 6 { // 2+3+1
		t.Fatalf("tick 1: expected prefill=6, got %d", out.PrefillBudgetUsed)
	}

	// Simulate prefill completion: transition to decoding.
	for _, seq := range out.ScheduledSequences {
		seq.Transition(SeqDecoding)
		seq.AppendToken(99) // fake generated token
	}

	// Tick 2: all 3 should be in decode.
	out, _ = sched.Tick(context.Background())
	if len(out.ScheduledSequences) != 3 {
		t.Fatalf("tick 2: expected 3 decode, got %d", len(out.ScheduledSequences))
	}
	if out.DecodeBudgetUsed != 3 {
		t.Fatalf("tick 2: expected decode=3, got %d", out.DecodeBudgetUsed)
	}
	if out.PrefillBudgetUsed != 0 {
		t.Fatalf("tick 2: expected prefill=0, got %d", out.PrefillBudgetUsed)
	}
}

func TestFCFSMixedPrefillDecode(t *testing.T) {
	sched := NewFCFSScheduler(FCFSConfig{
		MaxBatchSize:   10,
		MaxTokenBudget: 100,
		MaxQueueDepth:  100,
	})

	// Start one sequence.
	s1 := NewSequence("r1", []int32{1, 2}, 5)
	sched.Add(s1)
	out, _ := sched.Tick(context.Background())
	s1.Transition(SeqDecoding)
	s1.AppendToken(99)

	// Add a new sequence while s1 is decoding.
	s2 := NewSequence("r2", []int32{3, 4, 5}, 5)
	sched.Add(s2)

	// Tick: should get s1 (decode) + s2 (prefill) in same batch.
	out, _ = sched.Tick(context.Background())
	if len(out.ScheduledSequences) != 2 {
		t.Fatalf("expected 2 (mixed), got %d", len(out.ScheduledSequences))
	}
	if out.DecodeBudgetUsed != 1 {
		t.Fatalf("expected decode=1, got %d", out.DecodeBudgetUsed)
	}
	if out.PrefillBudgetUsed != 3 {
		t.Fatalf("expected prefill=3, got %d", out.PrefillBudgetUsed)
	}
}

func TestFCFSBatchSizeLimit(t *testing.T) {
	sched := NewFCFSScheduler(FCFSConfig{
		MaxBatchSize:   2,
		MaxTokenBudget: 1000,
		MaxQueueDepth:  100,
	})

	for i := 0; i < 5; i++ {
		sched.Add(NewSequence("r", []int32{1}, 10))
	}

	out, _ := sched.Tick(context.Background())
	if len(out.ScheduledSequences) != 2 {
		t.Fatalf("expected 2 (batch limit), got %d", len(out.ScheduledSequences))
	}
	if sched.WaitingLen() != 3 {
		t.Fatalf("expected 3 still waiting, got %d", sched.WaitingLen())
	}
}

func TestFCFSTokenBudgetLimit(t *testing.T) {
	sched := NewFCFSScheduler(FCFSConfig{
		MaxBatchSize:   100,
		MaxTokenBudget: 5,
		MaxQueueDepth:  100,
	})

	// First seq has 3-token prompt, second has 4-token prompt.
	s1 := NewSequence("r1", []int32{1, 2, 3}, 10)
	s2 := NewSequence("r2", []int32{4, 5, 6, 7}, 10)
	sched.Add(s1)
	sched.Add(s2)

	out, _ := sched.Tick(context.Background())
	// Only s1 fits (3 tokens), s2 (4 tokens) would exceed budget (3+4=7 > 5).
	if len(out.ScheduledSequences) != 1 {
		t.Fatalf("expected 1 (budget limit), got %d", len(out.ScheduledSequences))
	}
	if sched.WaitingLen() != 1 {
		t.Fatalf("expected 1 waiting, got %d", sched.WaitingLen())
	}
}

func TestFCFSQueueDepthLimit(t *testing.T) {
	sched := NewFCFSScheduler(FCFSConfig{
		MaxBatchSize:   10,
		MaxTokenBudget: 100,
		MaxQueueDepth:  2,
	})

	sched.Add(NewSequence("r1", []int32{1}, 10))
	sched.Add(NewSequence("r2", []int32{2}, 10))
	err := sched.Add(NewSequence("r3", []int32{3}, 10))
	if !errors.Is(err, ErrQueueFull) {
		t.Fatalf("expected ErrQueueFull, got %v", err)
	}
}

func TestFCFSComplete(t *testing.T) {
	sched := NewFCFSScheduler(DefaultFCFSConfig())

	seq := NewSequence("r1", []int32{1}, 10)
	sched.Add(seq)
	sched.Tick(context.Background())

	if err := sched.Complete(seq.ID); err != nil {
		t.Fatal(err)
	}
	if sched.Len() != 0 {
		t.Fatalf("expected len 0 after complete, got %d", sched.Len())
	}
}

func TestFCFSPreempt(t *testing.T) {
	sched := NewFCFSScheduler(DefaultFCFSConfig())

	seq := NewSequence("r1", []int32{1, 2}, 10)
	sched.Add(seq)
	sched.Tick(context.Background()) // -> PREFILLING
	seq.Transition(SeqDecoding)       // -> DECODING

	if err := sched.Preempt(seq.ID); err != nil {
		t.Fatal(err)
	}
	if seq.State != SeqWaiting {
		t.Fatalf("expected WAITING after preempt, got %s", seq.State)
	}
	if sched.WaitingLen() != 1 {
		t.Fatalf("expected 1 waiting, got %d", sched.WaitingLen())
	}
	if sched.ActiveLen() != 0 {
		t.Fatalf("expected 0 active, got %d", sched.ActiveLen())
	}
}

func TestChunkedPrefill_LargePromptSplitsAcrossTicks(t *testing.T) {
	sched := NewFCFSScheduler(FCFSConfig{
		MaxBatchSize:     10,
		MaxTokenBudget:   10000,
		MaxQueueDepth:    100,
		PrefillChunkSize: 512,
	})

	prompt := make([]int32, 2048) // 2048 tokens → 4 chunks of 512
	seq := NewSequence("r1", prompt, 10)
	sched.Add(seq)

	// Tick 1: first chunk (512 tokens).
	out, _ := sched.Tick(context.Background())
	if len(out.ScheduledSequences) != 1 {
		t.Fatalf("tick 1: expected 1, got %d", len(out.ScheduledSequences))
	}
	if out.PrefillBudgetUsed != 512 {
		t.Fatalf("tick 1: expected 512 prefill, got %d", out.PrefillBudgetUsed)
	}
	if seq.State != SeqPrefilling {
		t.Fatalf("tick 1: expected PREFILLING, got %s", seq.State)
	}
	// Simulate engine processing: advance PrefillConsumed.
	seq.PrefillConsumed = 512

	// Tick 2-4: subsequent chunks.
	for tick := 2; tick <= 4; tick++ {
		out, _ = sched.Tick(context.Background())
		if len(out.ScheduledSequences) != 1 {
			t.Fatalf("tick %d: expected 1, got %d", tick, len(out.ScheduledSequences))
		}
		seq.PrefillConsumed += 512
	}

	if seq.PrefillConsumed != 2048 {
		t.Fatalf("expected 2048 consumed, got %d", seq.PrefillConsumed)
	}
}

func TestChunkedPrefill_DecodeNotStarved(t *testing.T) {
	sched := NewFCFSScheduler(FCFSConfig{
		MaxBatchSize:     10,
		MaxTokenBudget:   10000,
		MaxQueueDepth:    100,
		PrefillChunkSize: 256,
	})

	// Start 3 short sequences and complete their prefill.
	for i := 0; i < 3; i++ {
		s := NewSequence("d", []int32{1, 2}, 100)
		sched.Add(s)
		sched.Tick(context.Background())
		s.PrefillConsumed = 2
		s.Transition(SeqDecoding)
		s.AppendToken(99)
	}

	// Add a long prompt (2048 tokens).
	long := NewSequence("long", make([]int32, 2048), 10)
	sched.Add(long)

	// Tick: should schedule 3 decode + 1 chunked prefill.
	out, _ := sched.Tick(context.Background())
	if len(out.ScheduledSequences) != 4 {
		t.Fatalf("expected 4 (3 decode + 1 prefill), got %d", len(out.ScheduledSequences))
	}
	if out.DecodeBudgetUsed != 3 {
		t.Fatalf("expected 3 decode, got %d", out.DecodeBudgetUsed)
	}
	// Prefill should be chunked to 256, not 2048.
	if out.PrefillBudgetUsed != 256 {
		t.Fatalf("expected 256 prefill (chunked), got %d", out.PrefillBudgetUsed)
	}
}

func TestChunkedPrefill_NoBudgetStarvation(t *testing.T) {
	sched := NewFCFSScheduler(FCFSConfig{
		MaxBatchSize:     10,
		MaxTokenBudget:   600,
		MaxQueueDepth:    100,
		PrefillChunkSize: 256,
	})

	// Add one long prompt and two short ones.
	longSeq := NewSequence("long", make([]int32, 4096), 10)
	short1 := NewSequence("s1", []int32{1, 2, 3}, 10)
	short2 := NewSequence("s2", []int32{4, 5}, 10)
	sched.Add(longSeq)
	sched.Add(short1)
	sched.Add(short2)

	// Tick 1: long gets first chunk (256), short1 (3) and short2 (2) should also be admitted.
	out, _ := sched.Tick(context.Background())
	if len(out.ScheduledSequences) != 3 {
		t.Fatalf("expected 3 scheduled, got %d", len(out.ScheduledSequences))
	}
	// Total prefill: 256 + 3 + 2 = 261
	if out.PrefillBudgetUsed != 261 {
		t.Fatalf("expected 261 total prefill, got %d", out.PrefillBudgetUsed)
	}
}

func TestFCFSEmptyTick(t *testing.T) {
	sched := NewFCFSScheduler(DefaultFCFSConfig())
	out, err := sched.Tick(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if len(out.ScheduledSequences) != 0 {
		t.Fatal("expected empty batch for empty scheduler")
	}
}
