package engine

import (
	"context"
	"fmt"
	"io"
	"log/slog"
	"sync"
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

// drainHandle reads from a handle until Finished or channel close. Returns all tokens.
func drainHandle(t *testing.T, handle *RequestHandle, timeout time.Duration) []TokenResult {
	t.Helper()
	var results []TokenResult
	deadline := time.After(timeout)
	for {
		select {
		case <-deadline:
			t.Fatalf("drainHandle: timeout after %v (got %d tokens)", timeout, len(results))
			return results
		case tok, ok := <-handle.Tokens:
			if !ok {
				return results
			}
			results = append(results, tok)
			if tok.Finished || tok.Err != nil {
				return results
			}
		}
	}
}

func TestServingEngineSingleRequest(t *testing.T) {
	runner := countingRunner(3)
	eng, cancel := makeServingEngine(t, runner)
	defer cancel()
	defer eng.Stop()

	handle, err := eng.Enqueue(context.Background(), &Request{
		ID:        "test-1",
		TokenIDs:  []int32{1, 2, 3},
		MaxTokens: 10,
	})
	if err != nil {
		t.Fatal(err)
	}

	tokens := drainHandle(t, handle, 5*time.Second)
	if len(tokens) == 0 {
		t.Fatal("expected tokens")
	}
	if !tokens[len(tokens)-1].Finished {
		t.Error("last token should be finished")
	}
}

func TestServingEngineConcurrentRequests(t *testing.T) {
	runner := countingRunner(3)
	eng, cancel := makeServingEngine(t, runner)
	defer cancel()
	defer eng.Stop()

	const numRequests = 10
	handles := make([]*RequestHandle, numRequests)

	for i := 0; i < numRequests; i++ {
		promptLen := (i % 5) + 1
		prompt := make([]int32, promptLen)
		for j := range prompt {
			prompt[j] = int32(j + 1)
		}
		h, err := eng.Enqueue(context.Background(), &Request{
			ID:        fmt.Sprintf("c-%d", i),
			TokenIDs:  prompt,
			MaxTokens: 10,
		})
		if err != nil {
			t.Fatalf("enqueue %d: %v", i, err)
		}
		handles[i] = h
	}

	// Drain each handle concurrently.
	var wg sync.WaitGroup
	for i, h := range handles {
		wg.Add(1)
		go func(idx int, handle *RequestHandle) {
			defer wg.Done()
			tokens := drainHandle(t, handle, 10*time.Second)
			if len(tokens) == 0 {
				t.Errorf("request %d: no tokens", idx)
			}
		}(i, h)
	}
	wg.Wait()
}

func TestServingEngineKVCacheFreed(t *testing.T) {
	runner := &backend.MockRunner{
		StepFunc: func(_ context.Context, b *backend.Batch) (*backend.StepOutput, error) {
			out := &backend.StepOutput{
				Logits: make([][]float32, len(b.SequenceIDs)),
			}
			for i := range b.SequenceIDs {
				logits := make([]float32, 50)
				logits[2] = 10.0 // EOS immediately
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
		Runner: runner, Scheduler: sched, Cache: cache,
		Tokenizer: tok, Sampling: SamplingParams{Temperature: 0}, Log: log,
	})

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	eng.Start(ctx)
	defer eng.Stop()

	handle, _ := eng.Enqueue(context.Background(), &Request{
		ID: "kv-test", TokenIDs: []int32{1, 2, 3, 4, 5}, MaxTokens: 5,
	})

	drainHandle(t, handle, 5*time.Second)
	time.Sleep(50 * time.Millisecond)

	if cache.NumFreeBlocks() != 100 {
		t.Fatalf("expected 100 free blocks, got %d", cache.NumFreeBlocks())
	}
}

func TestServingEngineStop(t *testing.T) {
	runner := countingRunner(3)
	eng, cancel := makeServingEngine(t, runner)
	defer cancel()

	eng.Stop()

	_, err := eng.Enqueue(context.Background(), &Request{ID: "x", TokenIDs: []int32{1}})
	if err == nil {
		t.Fatal("expected error after stop")
	}
}

// --- Focused correctness tests (Fix 6) ---

// Test: concurrent requests cannot steal each other's tokens.
func TestServingEngineTokenIsolation(t *testing.T) {
	// Runner returns sequence ID as the winning token.
	runner := &backend.MockRunner{
		StepFunc: func(_ context.Context, b *backend.Batch) (*backend.StepOutput, error) {
			out := &backend.StepOutput{
				Logits: make([][]float32, len(b.SequenceIDs)),
			}
			for i, sid := range b.SequenceIDs {
				logits := make([]float32, 100)
				// Use low bits of seq ID as the token, then EOS.
				tok := int32(sid % 90) // avoid EOS=2
				if tok == 2 {
					tok = 3
				}
				logits[tok] = 10.0
				out.Logits[i] = logits
			}
			return out, nil
		},
	}

	sched := scheduler.NewFCFSScheduler(scheduler.FCFSConfig{
		MaxBatchSize: 32, MaxTokenBudget: 4096, MaxQueueDepth: 256,
	})
	cache := kvcache.NewBlockPool(5000, 16)
	tok := &model.MockTokenizer{Vocab: 100, EOS: 2}
	log := slog.New(slog.NewTextHandler(io.Discard, nil))

	eng := NewServingEngine(ServingEngineConfig{
		Runner: runner, Scheduler: sched, Cache: cache,
		Tokenizer: tok, Sampling: SamplingParams{Temperature: 0}, Log: log,
	})
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	eng.Start(ctx)
	defer eng.Stop()

	// Enqueue 5 requests.
	handles := make([]*RequestHandle, 5)
	for i := 0; i < 5; i++ {
		h, err := eng.Enqueue(ctx, &Request{
			ID: fmt.Sprintf("iso-%d", i), TokenIDs: []int32{1}, MaxTokens: 3,
		})
		if err != nil {
			t.Fatal(err)
		}
		handles[i] = h
	}

	// Each request should get tokens derived from its own sequence ID.
	// Crucially, NO token from one request should appear in another's channel.
	var wg sync.WaitGroup
	for i, h := range handles {
		wg.Add(1)
		go func(idx int, handle *RequestHandle) {
			defer wg.Done()
			seqID := handle.SeqID
			expectedTok := int32(seqID % 90)
			if expectedTok == 2 {
				expectedTok = 3
			}
			for tok := range handle.Tokens {
				if tok.Err != nil {
					t.Errorf("req %d: unexpected error: %v", idx, tok.Err)
					return
				}
				if tok.TokenID != expectedTok {
					t.Errorf("req %d (seq %d): got token %d, expected %d — token isolation violated",
						idx, seqID, tok.TokenID, expectedTok)
				}
			}
		}(i, h)
	}
	wg.Wait()
}

// Test: "no tokens yet" does not cause premature completion.
func TestServingEngineAsyncNoTokensYet(t *testing.T) {
	// Runner that sleeps briefly to simulate slow backend.
	runner := &backend.MockRunner{
		StepFunc: func(_ context.Context, b *backend.Batch) (*backend.StepOutput, error) {
			time.Sleep(50 * time.Millisecond) // simulate latency
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

	eng, cancel := makeServingEngine(t, runner)
	defer cancel()
	defer eng.Stop()

	handle, _ := eng.Enqueue(ctx(), &Request{
		ID: "async", TokenIDs: []int32{1, 2}, MaxTokens: 5,
	})

	// The channel should eventually deliver tokens — not return empty.
	tok := <-handle.Tokens
	if tok.TokenID == 0 && !tok.Finished && tok.Err == nil {
		t.Fatal("got zero-value token — channel should block, not return empty")
	}
	if !tok.Finished {
		t.Fatal("expected EOS on first token from this runner")
	}
}

// Test: step failure delivers error to subscriber, does not strand sequence.
func TestServingEngineStepFailureRecovery(t *testing.T) {
	callCount := 0
	runner := &backend.MockRunner{
		StepFunc: func(_ context.Context, b *backend.Batch) (*backend.StepOutput, error) {
			callCount++
			return nil, fmt.Errorf("simulated GPU error")
		},
	}

	eng, cancel := makeServingEngine(t, runner)
	defer cancel()
	defer eng.Stop()

	handle, _ := eng.Enqueue(ctx(), &Request{
		ID: "fail", TokenIDs: []int32{1, 2, 3}, MaxTokens: 10,
	})

	// Should get an error result, not hang forever.
	tok := drainHandle(t, handle, 5*time.Second)
	if len(tok) == 0 {
		t.Fatal("expected at least one result (error)")
	}
	last := tok[len(tok)-1]
	if last.Err == nil {
		t.Fatal("expected error result from step failure")
	}
	if !last.Finished {
		t.Fatal("error result should be marked finished")
	}
}

// Test: benchmark-style queue saturation — enqueue failures are reported, no hangs.
func TestServingEngineBenchSaturation(t *testing.T) {
	// Slow runner so the queue fills up before draining.
	runner := &backend.MockRunner{
		StepFunc: func(_ context.Context, b *backend.Batch) (*backend.StepOutput, error) {
			time.Sleep(50 * time.Millisecond)
			out := &backend.StepOutput{
				Logits: make([][]float32, len(b.SequenceIDs)),
			}
			for i := range b.SequenceIDs {
				logits := make([]float32, 100)
				logits[2] = 10.0 // EOS immediately
				out.Logits[i] = logits
			}
			return out, nil
		},
	}

	sched := scheduler.NewFCFSScheduler(scheduler.FCFSConfig{
		MaxBatchSize: 2, MaxTokenBudget: 100, MaxQueueDepth: 3,
	})
	cache := kvcache.NewBlockPool(5000, 16)
	tok := &model.MockTokenizer{Vocab: 100, EOS: 2}
	log := slog.New(slog.NewTextHandler(io.Discard, nil))

	eng := NewServingEngine(ServingEngineConfig{
		Runner: runner, Scheduler: sched, Cache: cache,
		Tokenizer: tok, Sampling: SamplingParams{Temperature: 0}, Log: log,
	})
	ctxEng, cancelEng := context.WithCancel(context.Background())
	defer cancelEng()
	eng.Start(ctxEng)
	defer eng.Stop()

	// Blast 20 enqueues — with MaxQueueDepth=3 and a slow runner,
	// many should fail.
	const total = 20
	var handles []*RequestHandle
	enqueueErrs := 0
	for i := 0; i < total; i++ {
		h, err := eng.Enqueue(ctx(), &Request{
			ID: fmt.Sprintf("sat-%d", i), TokenIDs: []int32{1, 2}, MaxTokens: 10,
		})
		if err != nil {
			enqueueErrs++
			continue
		}
		handles = append(handles, h)
	}

	if enqueueErrs == 0 {
		t.Fatal("expected some enqueue failures with MaxQueueDepth=3 and slow runner")
	}
	t.Logf("enqueue: %d ok, %d rejected", len(handles), enqueueErrs)

	// All successfully enqueued requests must complete — no hangs.
	var wg sync.WaitGroup
	for _, h := range handles {
		wg.Add(1)
		go func(handle *RequestHandle) {
			defer wg.Done()
			drainHandle(t, handle, 30*time.Second)
		}(h)
	}
	wg.Wait()
}

func ctx() context.Context { return context.Background() }
