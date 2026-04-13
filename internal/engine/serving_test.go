package engine

import (
	"context"
	"io"
	"log/slog"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/TensorGreed/gollmgo/internal/backend"
	"github.com/TensorGreed/gollmgo/internal/kvcache"
	"github.com/TensorGreed/gollmgo/internal/model"
	"github.com/TensorGreed/gollmgo/internal/scheduler"
)

func makeServingEngine(t *testing.T, runner backend.Runner) (*ServingEngine, context.CancelFunc) {
	t.Helper()
	sched := scheduler.NewFCFSScheduler(scheduler.FCFSConfig{
		MaxBatchSize:   32,
		MaxTokenBudget: 4096,
		MaxQueueDepth:  256,
	})
	cache := kvcache.NewBlockPool(5000, 16)
	tok := &model.MockTokenizer{Vocab: 100, EOS: 2}
	log := slog.New(slog.NewTextHandler(io.Discard, nil))

	eng := NewServingEngine(ServingEngineConfig{
		Runner:    runner,
		Scheduler: sched,
		Cache:     cache,
		Tokenizer: tok,
		Sampling:  SamplingParams{Temperature: 0},
		Log:       log,
	})
	ctx, cancel := context.WithCancel(context.Background())
	eng.Start(ctx)
	return eng, cancel
}

// countingRunner returns token 42 for N steps per sequence, then EOS (2).
func countingRunner(maxPerSeq int) *backend.MockRunner {
	var mu sync.Mutex
	counts := make(map[uint64]int)

	return &backend.MockRunner{
		StepFunc: func(_ context.Context, b *backend.Batch) (*backend.StepOutput, error) {
			mu.Lock()
			defer mu.Unlock()
			out := &backend.StepOutput{
				Logits: make([][]float32, len(b.SequenceIDs)),
			}
			for i, sid := range b.SequenceIDs {
				counts[sid]++
				logits := make([]float32, 100)
				if counts[sid] > maxPerSeq {
					logits[2] = 10.0 // EOS
				} else {
					logits[42] = 10.0
				}
				out.Logits[i] = logits
			}
			return out, nil
		},
	}
}

func waitForCompletion(t *testing.T, eng *ServingEngine, expected int, timeout time.Duration) int {
	t.Helper()
	deadline := time.After(timeout)
	completed := 0
	for completed < expected {
		select {
		case <-deadline:
			t.Fatalf("timeout: completed %d/%d", completed, expected)
			return completed
		default:
		}
		tokens, _ := eng.NextTokens(context.Background())
		for _, tok := range tokens {
			if tok.Finished {
				completed++
			}
		}
		if len(tokens) == 0 {
			time.Sleep(5 * time.Millisecond)
		}
	}
	return completed
}

func TestServingEngineSingleRequest(t *testing.T) {
	runner := countingRunner(3)
	eng, cancel := makeServingEngine(t, runner)
	defer cancel()
	defer eng.Stop()

	err := eng.Enqueue(context.Background(), &Request{
		ID:        "test-1",
		TokenIDs:  []int32{1, 2, 3},
		MaxTokens: 10,
	})
	if err != nil {
		t.Fatal(err)
	}

	waitForCompletion(t, eng, 1, 5*time.Second)
}

func TestServingEngineConcurrentRequests(t *testing.T) {
	runner := countingRunner(3)
	eng, cancel := makeServingEngine(t, runner)
	defer cancel()
	defer eng.Stop()

	const numRequests = 10
	for i := 0; i < numRequests; i++ {
		promptLen := (i % 5) + 1
		prompt := make([]int32, promptLen)
		for j := range prompt {
			prompt[j] = int32(j + 1)
		}
		err := eng.Enqueue(context.Background(), &Request{
			ID:        "concurrent",
			TokenIDs:  prompt,
			MaxTokens: 10,
		})
		if err != nil {
			t.Fatalf("enqueue %d: %v", i, err)
		}
	}

	waitForCompletion(t, eng, numRequests, 10*time.Second)
}

func TestServingEngineKVCacheFreed(t *testing.T) {
	// Return EOS immediately.
	runner := &backend.MockRunner{
		StepFunc: func(_ context.Context, b *backend.Batch) (*backend.StepOutput, error) {
			out := &backend.StepOutput{
				Logits: make([][]float32, len(b.SequenceIDs)),
			}
			for i := range b.SequenceIDs {
				logits := make([]float32, 50)
				logits[2] = 10.0 // EOS
				out.Logits[i] = logits
			}
			return out, nil
		},
	}

	cache := kvcache.NewBlockPool(100, 4)
	sched := scheduler.NewFCFSScheduler(scheduler.DefaultFCFSConfig())
	tok := &model.MockTokenizer{Vocab: 50, EOS: 2}
	log := slog.New(slog.NewTextHandler(io.Discard, nil))

	eng := NewServingEngine(ServingEngineConfig{
		Runner:    runner,
		Scheduler: sched,
		Cache:     cache,
		Tokenizer: tok,
		Sampling:  SamplingParams{Temperature: 0},
		Log:       log,
	})

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	eng.Start(ctx)
	defer eng.Stop()

	eng.Enqueue(context.Background(), &Request{
		ID:        "kv-test",
		TokenIDs:  []int32{1, 2, 3, 4, 5},
		MaxTokens: 5,
	})

	// Wait for completion.
	waitForCompletion(t, &ServingEngine{}, 0, 0) // won't call this
	deadline := time.After(5 * time.Second)
	for {
		select {
		case <-deadline:
			t.Fatal("timeout")
		default:
		}
		tokens, _ := eng.NextTokens(context.Background())
		done := false
		for _, tok := range tokens {
			if tok.Finished {
				done = true
			}
		}
		if done {
			break
		}
		time.Sleep(5 * time.Millisecond)
	}

	// Allow cleanup.
	time.Sleep(50 * time.Millisecond)
	if cache.NumFreeBlocks() != 100 {
		t.Fatalf("expected 100 free blocks, got %d", cache.NumFreeBlocks())
	}
}

func TestServingEngineStop(t *testing.T) {
	var calls atomic.Int64
	runner := &backend.MockRunner{
		StepFunc: func(_ context.Context, b *backend.Batch) (*backend.StepOutput, error) {
			calls.Add(1)
			out := &backend.StepOutput{
				Logits: make([][]float32, len(b.SequenceIDs)),
			}
			for i := range b.SequenceIDs {
				out.Logits[i] = []float32{0, 1}
			}
			return out, nil
		},
	}

	eng, cancel := makeServingEngine(t, runner)
	defer cancel()

	eng.Stop()

	err := eng.Enqueue(context.Background(), &Request{ID: "x", TokenIDs: []int32{1}})
	if err == nil {
		t.Fatal("expected error after stop")
	}
}
