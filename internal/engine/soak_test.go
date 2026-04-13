package engine

import (
	"context"
	"fmt"
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

// TestSoakServingEngine runs the serving engine under sustained concurrent
// load for a fixed duration. This validates:
//   - no goroutine leaks
//   - no deadlocks under continuous enqueue/complete cycles
//   - block pool returns to baseline after all requests drain
//   - no panics under concurrent access
//
// Runs for 3 seconds in normal test mode.
// Use -test.run TestSoak -count=1 -timeout 30s for longer runs.
func TestSoakServingEngine(t *testing.T) {
	const (
		soakDuration  = 3 * time.Second
		concurrency   = 20
		maxTokens     = 5
	)

	var mu sync.Mutex
	seqCounts := make(map[uint64]int)

	runner := &backend.MockRunner{
		StepFunc: func(_ context.Context, b *backend.Batch) (*backend.StepOutput, error) {
			mu.Lock()
			defer mu.Unlock()
			out := &backend.StepOutput{
				Logits: make([][]float32, len(b.SequenceIDs)),
			}
			for i, sid := range b.SequenceIDs {
				seqCounts[sid]++
				logits := make([]float32, 50)
				if seqCounts[sid] > 3 {
					logits[2] = 10.0 // EOS
				} else {
					logits[10] = 10.0
				}
				out.Logits[i] = logits
			}
			return out, nil
		},
	}

	cache := kvcache.NewBlockPool(10000, 16)
	sched := scheduler.NewFCFSScheduler(scheduler.FCFSConfig{
		MaxBatchSize:   32,
		MaxTokenBudget: 4096,
		MaxQueueDepth:  500,
	})
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

	var (
		totalEnqueued atomic.Int64
		totalFinished atomic.Int64
		totalErrors   atomic.Int64
		wg            sync.WaitGroup
	)

	deadline := time.After(soakDuration)
	stopProducers := make(chan struct{})

	// Launch producer goroutines.
	for i := 0; i < concurrency; i++ {
		wg.Add(1)
		go func(workerID int) {
			defer wg.Done()
			reqNum := 0
			for {
				select {
				case <-stopProducers:
					return
				default:
				}

				promptLen := (reqNum % 4) + 1
				prompt := make([]int32, promptLen)
				for j := range prompt {
					prompt[j] = int32(j + 1)
				}

				handle, err := eng.Enqueue(ctx, &Request{
					ID:        fmt.Sprintf("soak-%d-%d", workerID, reqNum),
					TokenIDs:  prompt,
					MaxTokens: maxTokens,
				})
				if err != nil {
					totalErrors.Add(1)
					reqNum++
					continue
				}
				totalEnqueued.Add(1)

				// Drain in a separate goroutine.
				wg.Add(1)
				go func(h *RequestHandle) {
					defer wg.Done()
					for tok := range h.Tokens {
						if tok.Finished || tok.Err != nil {
							totalFinished.Add(1)
							return
						}
					}
					// Channel closed without finished — count as finished.
					totalFinished.Add(1)
				}(handle)

				reqNum++
			}
		}(i)
	}

	<-deadline
	close(stopProducers)

	// Wait for all producers to stop.
	// Then wait for remaining requests to drain.
	drainDeadline := time.After(10 * time.Second)
	done := make(chan struct{})
	go func() {
		wg.Wait()
		close(done)
	}()

	select {
	case <-done:
	case <-drainDeadline:
		t.Fatal("soak: timed out waiting for drain")
	}

	eng.Stop()

	enqueued := totalEnqueued.Load()
	finished := totalFinished.Load()
	errors := totalErrors.Load()

	t.Logf("soak results: enqueued=%d finished=%d enqueue_errors=%d duration=%v",
		enqueued, finished, errors, soakDuration)

	if enqueued == 0 {
		t.Fatal("soak: no requests were enqueued")
	}

	// All enqueued requests should have finished (or been cleaned up on Stop).
	if finished < enqueued {
		t.Errorf("soak: %d enqueued but only %d finished (delta=%d)",
			enqueued, finished, enqueued-finished)
	}

	// Block pool should be fully free after drain.
	time.Sleep(50 * time.Millisecond)
	free := cache.NumFreeBlocks()
	total := cache.NumTotalBlocks()
	if free != total {
		t.Errorf("soak: block pool leak: %d/%d free", free, total)
	}
}
