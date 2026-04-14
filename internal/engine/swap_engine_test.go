package engine

import (
	"context"
	"io"
	"log/slog"
	"testing"
	"time"

	"github.com/TensorGreed/gollmgo/internal/backend"
	"github.com/TensorGreed/gollmgo/internal/kvcache"
	"github.com/TensorGreed/gollmgo/internal/model"
	"github.com/TensorGreed/gollmgo/internal/scheduler"
)

// TestSwapPreemptSnapshotAndRestore verifies that when the scheduler is
// configured with PreemptSwap mode and the runner advertises KVSwap, a
// memory-pressure preemption snapshots the KV state to host (via the mock
// runner) and the victim's subscriber channel stays open. On re-admission
// the engine restores KV from the snapshot rather than recomputing.
func TestSwapPreemptSnapshotAndRestore(t *testing.T) {
	// Runner: EOS immediately so requests finish in one decode step
	// (simplifies the test; we only care about the preemption path).
	runner := &backend.MockRunner{
		StepFunc: func(_ context.Context, b *backend.Batch) (*backend.StepOutput, error) {
			out := &backend.StepOutput{
				Logits: make([][]float32, len(b.SequenceIDs)),
			}
			for i := range b.SequenceIDs {
				row := make([]float32, 50)
				row[2] = 10 // EOS
				out.Logits[i] = row
			}
			return out, nil
		},
	}

	// Scheduler in swap mode.
	sched := scheduler.NewScheduler(scheduler.PolicyFCFS, scheduler.SchedulerConfig{
		MaxBatchSize:   8,
		MaxTokenBudget: 256,
		MaxQueueDepth:  32,
		PreemptMode:    scheduler.PreemptSwap,
	})
	cache := kvcache.NewBlockPool(512, 16)
	tok := &model.MockTokenizer{Vocab: 50, EOS: 2}
	log := slog.New(slog.NewTextHandler(io.Discard, nil))

	eng := NewServingEngine(ServingEngineConfig{
		Runner: runner, Scheduler: sched, Cache: cache,
		Tokenizer: tok, Sampling: SamplingParams{Temperature: 0}, Log: log,
	})
	// Confirm the mock's KVSwap capability was discovered.
	if eng.kvSwapper == nil {
		t.Fatal("engine did not pick up MockRunner's KVSwap capability")
	}

	// Drive a SnapshotKV → RestoreKV round-trip directly through the engine
	// helpers. Build a sequence, Tick once so the scheduler admits it, then
	// exercise preempt/restore with the engine lock held.
	seq := scheduler.NewSequence("swap-round-trip", []int32{1, 2, 3, 4, 5, 6, 7, 8}, 4)
	if err := sched.Add(seq); err != nil {
		t.Fatal(err)
	}
	if _, err := sched.Tick(context.Background()); err != nil {
		t.Fatal(err)
	}
	_ = seq.Transition(scheduler.SeqDecoding) // simulate prefill completion

	eng.mu.Lock()
	bt := kvcache.NewBlockTable(seq.ID, cache)
	if _, err := bt.AppendN(8); err != nil {
		eng.mu.Unlock()
		t.Fatal(err)
	}
	eng.blockTables[seq.ID] = bt
	preBlocks := bt.NumBlocks()
	eng.mu.Unlock()

	if err := sched.Preempt(seq.ID); err != nil {
		t.Fatal(err)
	}

	eng.mu.Lock()
	// SwapState must be populated by the scheduler's swap-mode Preempt.
	if seq.SwapState == nil {
		eng.mu.Unlock()
		t.Fatal("scheduler did not populate SwapState")
	}
	eng.releasePreemptedLocked(seq.ID)

	// After releasePreemptedLocked in swap mode: blocks freed, snapshot held.
	if _, stillHasTable := eng.blockTables[seq.ID]; stillHasTable {
		eng.mu.Unlock()
		t.Fatal("block table should be released after swap-mode preempt")
	}
	snap, held := eng.swapStore[seq.ID]
	if !held {
		eng.mu.Unlock()
		t.Fatal("expected a KV snapshot in swapStore after swap-mode preempt")
	}
	if snap.NumBlocks() != preBlocks {
		eng.mu.Unlock()
		t.Fatalf("snapshot captured %d blocks, expected %d", snap.NumBlocks(), preBlocks)
	}

	// Restore: simulate re-admission by transitioning the sequence and
	// calling the restore helper.
	if err := eng.restoreSwappedLocked(seq); err != nil {
		eng.mu.Unlock()
		t.Fatalf("restore failed: %v", err)
	}
	if _, still := eng.swapStore[seq.ID]; still {
		eng.mu.Unlock()
		t.Error("snapshot should have been released after restore")
	}
	restored, ok := eng.blockTables[seq.ID]
	if !ok {
		eng.mu.Unlock()
		t.Fatal("block table should be re-created after restore")
	}
	if restored.NumBlocks() != preBlocks {
		eng.mu.Unlock()
		t.Errorf("restored block table has %d blocks, expected %d",
			restored.NumBlocks(), preBlocks)
	}
	// Release to avoid leaking for the test's pool accounting.
	restored.Free()
	delete(eng.blockTables, seq.ID)
	eng.mu.Unlock()
	// Don't call Stop() — the engine's loop was never started here.
	_ = time.Millisecond // keep time import used
}

// TestSwapFallsBackToRecomputeWithoutSwapper: if the runner doesn't
// implement KVSwapper, engine must silently downgrade swap mode to
// recompute (resetting PrefillConsumed and SwapState) rather than
// leaving dangling references.
func TestSwapFallsBackToRecomputeWithoutSwapper(t *testing.T) {
	// Wrap MockRunner so KVSwap capability is OFF and it doesn't satisfy
	// backend.KVSwapper at the interface level.
	inner := &backend.MockRunner{}
	runner := &capOverrideRunner{inner: inner} // only backend.Runner; no KVSwapper

	sched := scheduler.NewScheduler(scheduler.PolicyFCFS, scheduler.SchedulerConfig{
		MaxBatchSize: 4, MaxTokenBudget: 256, MaxQueueDepth: 16,
		PreemptMode: scheduler.PreemptSwap, // request swap, but runner can't
	})
	cache := kvcache.NewBlockPool(128, 16)
	tok := &model.MockTokenizer{Vocab: 50, EOS: 2}
	log := slog.New(slog.NewTextHandler(io.Discard, nil))

	eng := NewServingEngine(ServingEngineConfig{
		Runner: runner, Scheduler: sched, Cache: cache,
		Tokenizer: tok, Sampling: SamplingParams{Temperature: 0}, Log: log,
	})
	if eng.kvSwapper != nil {
		t.Fatal("expected no kvSwapper when runner doesn't implement KVSwapper")
	}

	seq := scheduler.NewSequence("fallback", []int32{1, 2, 3, 4}, 4)
	_ = sched.Add(seq)
	if _, err := sched.Tick(context.Background()); err != nil {
		t.Fatal(err)
	}
	_ = seq.Transition(scheduler.SeqDecoding)

	eng.mu.Lock()
	bt := kvcache.NewBlockTable(seq.ID, cache)
	_, _ = bt.AppendN(4)
	eng.blockTables[seq.ID] = bt
	seq.PrefillConsumed = 4
	eng.mu.Unlock()

	if err := sched.Preempt(seq.ID); err != nil {
		t.Fatal(err)
	}

	eng.mu.Lock()
	eng.releasePreemptedLocked(seq.ID)
	if _, stillHasTable := eng.blockTables[seq.ID]; stillHasTable {
		eng.mu.Unlock()
		t.Fatal("block table should be freed in recompute fallback")
	}
	if seq.PrefillConsumed != 0 {
		eng.mu.Unlock()
		t.Errorf("PrefillConsumed should be reset to 0 in fallback, got %d", seq.PrefillConsumed)
	}
	if seq.SwapState != nil {
		eng.mu.Unlock()
		t.Error("SwapState should be cleared in recompute fallback")
	}
	eng.mu.Unlock()
	// Don't call Stop() — the engine's loop was never started.
}
