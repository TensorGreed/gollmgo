package engine

import (
	"context"
	"fmt"
	"log/slog"
	"sync"
	"time"

	"github.com/TensorGreed/gollmgo/internal/backend"
	"github.com/TensorGreed/gollmgo/internal/kvcache"
	"github.com/TensorGreed/gollmgo/internal/metrics"
	"github.com/TensorGreed/gollmgo/internal/model"
	"github.com/TensorGreed/gollmgo/internal/scheduler"
)

// ServingEngine is the production engine that runs the continuous batching loop.
// It owns the scheduler, KV cache, and coordinates the full serving pipeline.
type ServingEngine struct {
	runner    backend.Runner
	sched     *scheduler.FCFSScheduler
	cache     *kvcache.BlockPool
	tokenizer model.Tokenizer
	sampler   *Sampler
	eosID     int32
	log       *slog.Logger

	mu          sync.Mutex
	blockTables map[uint64]*kvcache.BlockTable // seqID -> block table
	subscribers map[uint64]chan TokenResult     // seqID -> token delivery channel
	results     []TokenResult                  // buffered results for NextTokens
	stopped     bool
	loopDone    chan struct{}
}

// ServingEngineConfig holds configuration for the serving engine.
type ServingEngineConfig struct {
	Runner      backend.Runner
	Scheduler   *scheduler.FCFSScheduler
	Cache       *kvcache.BlockPool
	Tokenizer   model.Tokenizer
	Sampling    SamplingParams
	Log         *slog.Logger
}

// NewServingEngine creates and starts the continuous batching loop.
func NewServingEngine(cfg ServingEngineConfig) *ServingEngine {
	e := &ServingEngine{
		runner:      cfg.Runner,
		sched:       cfg.Scheduler,
		cache:       cfg.Cache,
		tokenizer:   cfg.Tokenizer,
		sampler:     NewSampler(cfg.Sampling),
		eosID:       cfg.Tokenizer.EOSTokenID(),
		log:         cfg.Log,
		blockTables: make(map[uint64]*kvcache.BlockTable),
		subscribers: make(map[uint64]chan TokenResult),
		loopDone:    make(chan struct{}),
	}

	// Update cache metrics.
	metrics.Global.KVCacheBlocksTotal.Store(int64(cfg.Cache.NumTotalBlocks()))

	return e
}

// Start begins the continuous batching loop in a goroutine.
func (e *ServingEngine) Start(ctx context.Context) {
	go e.loop(ctx)
}

func (e *ServingEngine) loop(ctx context.Context) {
	defer close(e.loopDone)

	for {
		select {
		case <-ctx.Done():
			return
		default:
		}

		e.mu.Lock()
		if e.stopped {
			e.mu.Unlock()
			return
		}
		e.mu.Unlock()

		// Run one scheduler tick.
		out, err := e.sched.Tick(ctx)
		if err != nil {
			e.log.Error("scheduler tick failed", "error", err)
			continue
		}

		if len(out.ScheduledSequences) == 0 {
			// No work to do. Brief sleep to avoid busy-spinning.
			time.Sleep(100 * time.Microsecond)
			continue
		}

		if err := e.runStep(ctx, out); err != nil {
			e.log.Error("step failed", "error", err)
		}
	}
}

func (e *ServingEngine) runStep(ctx context.Context, schedOut *scheduler.SchedulerOutput) error {
	seqs := schedOut.ScheduledSequences

	// Build batch with slot mappings from block tables.
	batch := &backend.Batch{
		SequenceIDs: make([]uint64, 0, len(seqs)),
		IsPrefill:   make([]bool, 0, len(seqs)),
	}

	e.mu.Lock()
	for _, seq := range seqs {
		batch.SequenceIDs = append(batch.SequenceIDs, seq.ID)
		isPrefill := seq.State == scheduler.SeqPrefilling
		batch.IsPrefill = append(batch.IsPrefill, isPrefill)

		// Ensure block table exists.
		bt, ok := e.blockTables[seq.ID]
		if !ok {
			bt = kvcache.NewBlockTable(seq.ID, e.cache)
			e.blockTables[seq.ID] = bt
		}

		if isPrefill {
			// Prefill: send all prompt tokens, allocate slots.
			for i := 0; i < seq.PromptLen; i++ {
				slot, err := bt.Append()
				if err != nil {
					e.mu.Unlock()
					return fmt.Errorf("block alloc for seq %d: %w", seq.ID, err)
				}
				batch.TokenIDs = append(batch.TokenIDs, seq.TokenIDs[i])
				batch.Positions = append(batch.Positions, int32(i))
				batch.SlotMapping = append(batch.SlotMapping, slot)
			}
		} else {
			// Decode: send last token, allocate one slot.
			slot, err := bt.Append()
			if err != nil {
				e.mu.Unlock()
				return fmt.Errorf("block alloc for seq %d: %w", seq.ID, err)
			}
			lastPos := seq.TotalLen() - 1
			batch.TokenIDs = append(batch.TokenIDs, seq.TokenIDs[lastPos])
			batch.Positions = append(batch.Positions, int32(lastPos))
			batch.SlotMapping = append(batch.SlotMapping, slot)
		}
	}
	e.mu.Unlock()

	// Run forward pass.
	output, err := e.runner.Step(ctx, batch)
	if err != nil {
		return fmt.Errorf("runner step: %w", err)
	}

	metrics.Global.BatchesRun.Add(1)

	// Sample tokens and deliver results.
	e.mu.Lock()
	defer e.mu.Unlock()

	for i, seq := range seqs {
		if i >= len(output.Logits) {
			break
		}

		tokenID := e.sampler.Sample(output.Logits[i])
		seq.AppendToken(tokenID)
		metrics.Global.TokensGenerated.Add(1)

		// Transition prefill -> decode.
		if seq.State == scheduler.SeqPrefilling {
			seq.Transition(scheduler.SeqDecoding)
		}

		finished := tokenID == e.eosID || seq.GeneratedLen >= seq.MaxTokens
		if finished {
			seq.Transition(scheduler.SeqFinished)
		}

		result := TokenResult{
			SequenceID: seq.ID,
			TokenID:    tokenID,
			Finished:   finished,
		}

		// Buffer for NextTokens.
		e.results = append(e.results, result)

		// Deliver to subscriber (for streaming).
		if ch, ok := e.subscribers[seq.ID]; ok {
			select {
			case ch <- result:
			default:
			}
		}

		if finished {
			e.cleanupSequence(seq.ID)
		}
	}

	// Update cache metrics.
	used := int64(e.cache.NumTotalBlocks() - e.cache.NumFreeBlocks())
	metrics.Global.KVCacheBlocksUsed.Store(used)

	return nil
}

// Enqueue submits a request and returns a channel for token delivery.
func (e *ServingEngine) Enqueue(_ context.Context, req *Request) error {
	seq := scheduler.NewSequence(req.ID, req.TokenIDs, req.MaxTokens)

	e.mu.Lock()
	if e.stopped {
		e.mu.Unlock()
		return fmt.Errorf("engine: stopped")
	}
	ch := make(chan TokenResult, req.MaxTokens+1)
	e.subscribers[seq.ID] = ch
	e.mu.Unlock()

	return e.sched.Add(seq)
}

// Subscribe returns the token delivery channel for a given request ID.
// Must be called after Enqueue. Returns nil if not found.
func (e *ServingEngine) Subscribe(seqID uint64) <-chan TokenResult {
	e.mu.Lock()
	defer e.mu.Unlock()
	return e.subscribers[seqID]
}

// NextTokens drains all buffered results (Engine interface).
// For serving, prefer Subscribe for per-request streaming.
func (e *ServingEngine) NextTokens(_ context.Context) ([]TokenResult, error) {
	e.mu.Lock()
	defer e.mu.Unlock()

	if len(e.results) == 0 {
		return nil, nil
	}
	out := e.results
	e.results = nil
	return out, nil
}

func (e *ServingEngine) Stop() error {
	e.mu.Lock()
	e.stopped = true
	// Close all subscriber channels.
	for id, ch := range e.subscribers {
		close(ch)
		delete(e.subscribers, id)
	}
	e.mu.Unlock()

	<-e.loopDone
	return nil
}

func (e *ServingEngine) cleanupSequence(seqID uint64) {
	// Free block table.
	if bt, ok := e.blockTables[seqID]; ok {
		bt.Free()
		delete(e.blockTables, seqID)
	}
	// Close and remove subscriber.
	if ch, ok := e.subscribers[seqID]; ok {
		close(ch)
		delete(e.subscribers, seqID)
	}
	// Remove from scheduler.
	e.sched.Complete(seqID)
}

// Compile-time check.
var _ Engine = (*ServingEngine)(nil)
