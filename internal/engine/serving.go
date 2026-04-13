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

// NewServingEngine creates the engine. Call Start to begin the batching loop.
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

		out, err := e.sched.Tick(ctx)
		if err != nil {
			e.log.Error("scheduler tick failed", "error", err)
			continue
		}

		if len(out.ScheduledSequences) == 0 {
			time.Sleep(100 * time.Microsecond)
			continue
		}

		if err := e.runStep(ctx, out); err != nil {
			e.log.Error("step failed, recovering sequences", "error", err)
			e.recoverFromStepFailure(out.ScheduledSequences)
		}
	}
}

func (e *ServingEngine) runStep(ctx context.Context, schedOut *scheduler.SchedulerOutput) error {
	seqs := schedOut.ScheduledSequences

	batch := &backend.Batch{
		SequenceIDs: make([]uint64, 0, len(seqs)),
		IsPrefill:   make([]bool, 0, len(seqs)),
	}

	e.mu.Lock()
	for _, seq := range seqs {
		batch.SequenceIDs = append(batch.SequenceIDs, seq.ID)
		isPrefill := seq.State == scheduler.SeqPrefilling
		batch.IsPrefill = append(batch.IsPrefill, isPrefill)

		bt, ok := e.blockTables[seq.ID]
		if !ok {
			bt = kvcache.NewBlockTable(seq.ID, e.cache)
			e.blockTables[seq.ID] = bt
		}

		if isPrefill {
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

	output, err := e.runner.Step(ctx, batch)
	if err != nil {
		return fmt.Errorf("runner step: %w", err)
	}

	metrics.Global.BatchesRun.Add(1)

	e.mu.Lock()
	defer e.mu.Unlock()

	for i, seq := range seqs {
		if i >= len(output.Logits) {
			break
		}

		tokenID := e.sampler.Sample(output.Logits[i])
		seq.AppendToken(tokenID)
		metrics.Global.TokensGenerated.Add(1)

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

	used := int64(e.cache.NumTotalBlocks() - e.cache.NumFreeBlocks())
	metrics.Global.KVCacheBlocksUsed.Store(used)
	return nil
}

// recoverFromStepFailure handles sequences that were in-flight when
// runner.Step() failed. Sequences in PREFILLING are failed with an error
// (they haven't produced any tokens so there's nothing to resume).
// Sequences in DECODING are failed too — their KV state may be inconsistent.
func (e *ServingEngine) recoverFromStepFailure(seqs []*scheduler.Sequence) {
	e.mu.Lock()
	defer e.mu.Unlock()

	for _, seq := range seqs {
		errResult := TokenResult{
			SequenceID: seq.ID,
			Finished:   true,
			Err:        fmt.Errorf("backend step failed"),
		}

		if ch, ok := e.subscribers[seq.ID]; ok {
			select {
			case ch <- errResult:
			default:
			}
		}

		e.cleanupSequence(seq.ID)
	}
}

// Enqueue submits a request and returns a handle for receiving that
// request's tokens. The returned channel is dedicated to this request.
func (e *ServingEngine) Enqueue(_ context.Context, req *Request) (*RequestHandle, error) {
	seq := scheduler.NewSequence(req.ID, req.TokenIDs, req.MaxTokens)

	e.mu.Lock()
	if e.stopped {
		e.mu.Unlock()
		return nil, fmt.Errorf("engine: stopped")
	}
	ch := make(chan TokenResult, req.MaxTokens+1)
	e.subscribers[seq.ID] = ch
	e.mu.Unlock()

	if err := e.sched.Add(seq); err != nil {
		e.mu.Lock()
		delete(e.subscribers, seq.ID)
		close(ch)
		e.mu.Unlock()
		return nil, err
	}

	return &RequestHandle{SeqID: seq.ID, Tokens: ch}, nil
}

func (e *ServingEngine) Stop() error {
	e.mu.Lock()
	e.stopped = true
	for id, ch := range e.subscribers {
		close(ch)
		delete(e.subscribers, id)
	}
	e.mu.Unlock()

	<-e.loopDone
	return nil
}

func (e *ServingEngine) cleanupSequence(seqID uint64) {
	if bt, ok := e.blockTables[seqID]; ok {
		bt.Free()
		delete(e.blockTables, seqID)
	}
	if ch, ok := e.subscribers[seqID]; ok {
		close(ch)
		delete(e.subscribers, seqID)
	}
	e.sched.Complete(seqID)
}

// Compile-time check.
var _ Engine = (*ServingEngine)(nil)
