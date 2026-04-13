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
	runner      backend.Runner
	sched       *scheduler.FCFSScheduler
	cache       *kvcache.BlockPool
	prefixCache *kvcache.PrefixCache // optional prefix caching for KV block reuse
	tokenizer   model.Tokenizer
	sampler     *Sampler
	eosID       int32
	log         *slog.Logger

	preemptWatermark float64

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
	// PreemptWatermark is the fraction of cache utilization above which
	// the engine preempts the longest decode sequence to free blocks.
	// 0 disables preemption. Default: 0.95.
	PreemptWatermark float64
	// EnablePrefixCache enables block-level prefix caching for KV reuse.
	EnablePrefixCache bool
}

// NewServingEngine creates the engine. Call Start to begin the batching loop.
func NewServingEngine(cfg ServingEngineConfig) *ServingEngine {
	if cfg.PreemptWatermark <= 0 {
		cfg.PreemptWatermark = 0.95
	}
	var prefixCache *kvcache.PrefixCache
	if cfg.EnablePrefixCache {
		prefixCache = kvcache.NewPrefixCache(cfg.Cache)
	}

	e := &ServingEngine{
		runner:           cfg.Runner,
		sched:            cfg.Scheduler,
		cache:            cfg.Cache,
		prefixCache:      prefixCache,
		tokenizer:        cfg.Tokenizer,
		sampler:          NewSampler(cfg.Sampling),
		eosID:            cfg.Tokenizer.EOSTokenID(),
		log:              cfg.Log,
		preemptWatermark: cfg.PreemptWatermark,
		blockTables:      make(map[uint64]*kvcache.BlockTable),
		subscribers:      make(map[uint64]chan TokenResult),
		loopDone:         make(chan struct{}),
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

		// Check memory pressure before scheduling.
		e.checkMemoryPressure()

		out, err := e.sched.Tick(ctx)
		if err != nil {
			e.log.Error("scheduler tick failed", "error", err)
			continue
		}

		// Update scheduler queue metrics after each tick.
		metrics.Global.SchedulerQueueDepth.Store(int64(e.sched.WaitingLen()))
		metrics.Global.SchedulerActiveCount.Store(int64(e.sched.ActiveLen()))

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
			// Prefix cache: on first chunk, check for reusable prefix blocks.
			if seq.PrefillConsumed == 0 && e.prefixCache != nil {
				matched := bt.MatchPrefix(seq.TokenIDs[:seq.PromptLen], e.prefixCache)
				if matched > 0 {
					seq.PrefillConsumed = matched
					// Emit slots for prefix-cached tokens into the batch
					// so the model sees them in the slot mapping, but we
					// don't need to run them through the kernel since
					// KV is already in cache.
				}
			}

			// Chunked prefill: process only a chunk of prompt tokens per step.
			chunkSize := e.sched.PrefillChunkSize()
			if chunkSize <= 0 {
				chunkSize = seq.PromptLen // no chunking
			}
			start := seq.PrefillConsumed
			remaining := seq.PromptLen - start
			chunk := remaining
			if chunk > chunkSize {
				chunk = chunkSize
			}

			for i := start; i < start+chunk; i++ {
				slot, err := bt.Append()
				if err != nil {
					e.mu.Unlock()
					return fmt.Errorf("block alloc for seq %d: %w", seq.ID, err)
				}
				batch.TokenIDs = append(batch.TokenIDs, seq.TokenIDs[i])
				batch.Positions = append(batch.Positions, int32(i))
				batch.SlotMapping = append(batch.SlotMapping, slot)
			}
			seq.PrefillConsumed += chunk
			batch.SeqTokenCounts = append(batch.SeqTokenCounts, int32(chunk))
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
			batch.SeqTokenCounts = append(batch.SeqTokenCounts, 1)
		}

		// Per-sequence context length and full slot table.
		batch.SeqContextLens = append(batch.SeqContextLens, int32(bt.NumTokens()))
		batch.SeqSlotTables = append(batch.SeqSlotTables, bt.SlotMapping())
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

		// For chunked prefill: skip sampling if prefill is not yet complete.
		if seq.State == scheduler.SeqPrefilling && seq.PrefillConsumed < seq.PromptLen {
			// Partial prefill chunk — no token to emit yet.
			continue
		}

		tokenID := e.sampler.Sample(output.Logits[i])
		now := time.Now()
		seq.AppendToken(tokenID)
		metrics.Global.TokensGenerated.Add(1)

		if seq.State == scheduler.SeqPrefilling {
			// Prefill complete — record TTFT and transition to decode.
			ttftMs := float64(now.Sub(seq.CreatedAt).Microseconds()) / 1000.0
			metrics.Global.TTFT.Record(ttftMs)
			seq.Transition(scheduler.SeqDecoding)
			seq.LastTokenAt = now
		} else {
			// Decode step — record ITL.
			if !seq.LastTokenAt.IsZero() {
				itlMs := float64(now.Sub(seq.LastTokenAt).Microseconds()) / 1000.0
				metrics.Global.ITL.Record(itlMs)
			}
			seq.LastTokenAt = now
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

// checkMemoryPressure preempts the longest decode sequence if cache
// utilization exceeds the watermark. This frees blocks for new requests.
func (e *ServingEngine) checkMemoryPressure() {
	util := e.cache.Utilization()
	if util < e.preemptWatermark {
		return
	}

	e.mu.Lock()
	defer e.mu.Unlock()

	// Find the decode sequence using the most blocks.
	var victimID uint64
	victimBlocks := 0
	for seqID, bt := range e.blockTables {
		if bt.NumBlocks() > victimBlocks {
			victimBlocks = bt.NumBlocks()
			victimID = seqID
		}
	}

	if victimBlocks == 0 {
		return
	}

	e.log.Warn("memory pressure preemption",
		"utilization", util,
		"watermark", e.preemptWatermark,
		"victim_seq", victimID,
		"victim_blocks", victimBlocks)

	// Free blocks and notify subscriber of preemption error.
	if bt, ok := e.blockTables[victimID]; ok {
		bt.Free()
		delete(e.blockTables, victimID)
	}

	errResult := TokenResult{
		SequenceID: victimID,
		Finished:   true,
		Err:        fmt.Errorf("preempted: KV cache memory pressure (%.0f%% used)", util*100),
	}
	if ch, ok := e.subscribers[victimID]; ok {
		select {
		case ch <- errResult:
		default:
		}
		close(ch)
		delete(e.subscribers, victimID)
	}

	e.sched.Complete(victimID)
}

func (e *ServingEngine) cleanupSequence(seqID uint64) {
	if bt, ok := e.blockTables[seqID]; ok {
		// Donate completed blocks to prefix cache before freeing.
		if e.prefixCache != nil {
			// We need the token IDs for hashing. Find them from the sequence
			// which is still tracked by the scheduler at this point.
			if seq := e.findSequence(seqID); seq != nil {
				e.prefixCache.DonateBlocks(seq.TokenIDs[:seq.PromptLen], bt.PhysicalBlocks(), e.cache.BlockSize())
			}
		}
		bt.Free()
		delete(e.blockTables, seqID)
	}
	if ch, ok := e.subscribers[seqID]; ok {
		close(ch)
		delete(e.subscribers, seqID)
	}
	e.sched.Complete(seqID)
}

// findSequence looks up a sequence by ID from the scheduler's tracked set.
// Must be called with e.mu held.
func (e *ServingEngine) findSequence(seqID uint64) *scheduler.Sequence {
	return e.sched.Find(seqID)
}

// Compile-time check.
var _ Engine = (*ServingEngine)(nil)
