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

// SpeculativeSettings configures engine-level speculative decoding (M6).
// Drafting only runs on DECODE sequences with enough history; prefill is
// unaffected. When the running acceptance rate falls below KillThreshold
// the drafter is disabled for the rest of the process lifetime.
type SpeculativeSettings struct {
	Enabled        bool
	NGramSize      int     // window used by the n-gram drafter (default 3)
	NumDraftTokens int     // max drafts per step (K, default 4)
	KillThreshold  float64 // disable if acceptance rate < threshold (0 = never kill)
	KillWarmup     int64   // min drafted tokens before the kill switch can trip
}

// ServingEngine is the production engine that runs the continuous batching loop.
// It owns the scheduler, KV cache, and coordinates the full serving pipeline.
type ServingEngine struct {
	runner      backend.Runner
	sched       scheduler.Scheduler
	cache       *kvcache.BlockPool
	prefixCache *kvcache.PrefixCache // optional prefix caching for KV block reuse
	tokenizer   model.Tokenizer
	sampler     *Sampler
	eosID       int32
	log         *slog.Logger

	preemptWatermark float64
	maxPrefixBlocks  int

	// Speculative decoding (M6). drafter is nil when disabled or when the
	// backend doesn't support it. kvSwapper is the optional KV-swap capability.
	specCfg     SpeculativeSettings
	drafter     *NGramDrafter
	kvSwapper   backend.KVSwapper // non-nil if runner implements KVSwapper
	swapStore   map[uint64]backend.KVSnapshot // seqID -> snapshot while preempted

	mu          sync.Mutex
	blockTables map[uint64]*kvcache.BlockTable // seqID -> block table
	subscribers map[uint64]chan TokenResult    // seqID -> token delivery channel
	stopped     bool
	loopDone    chan struct{}
}

// ServingEngineConfig holds configuration for the serving engine.
type ServingEngineConfig struct {
	Runner    backend.Runner
	Scheduler scheduler.Scheduler
	Cache     *kvcache.BlockPool
	Tokenizer model.Tokenizer
	Sampling  SamplingParams
	Log       *slog.Logger
	// PreemptWatermark is the fraction of cache utilization above which
	// the engine preempts the longest decode sequence to free blocks.
	// 0 disables preemption. Default: 0.95.
	PreemptWatermark float64
	// EnablePrefixCache enables block-level prefix caching for KV reuse.
	EnablePrefixCache bool
	// MaxPrefixBlocks caps the number of blocks the prefix cache may pin.
	// 0 derives a default of 50% of the pool.
	MaxPrefixBlocks int
	// Speculative enables n-gram speculative decoding. Requires the runner
	// to report SpeculativeDecoding capability; otherwise the engine logs a
	// warning and leaves it disabled.
	Speculative SpeculativeSettings
}

// NewServingEngine creates the engine. Call Start to begin the batching loop.
func NewServingEngine(cfg ServingEngineConfig) *ServingEngine {
	if cfg.PreemptWatermark <= 0 {
		cfg.PreemptWatermark = 0.95
	}
	maxPrefix := cfg.MaxPrefixBlocks
	if maxPrefix <= 0 {
		maxPrefix = cfg.Cache.NumTotalBlocks() / 2
	}
	var prefixCache *kvcache.PrefixCache
	if cfg.EnablePrefixCache {
		prefixCache = kvcache.NewPrefixCacheWithCap(cfg.Cache, maxPrefix)
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
		maxPrefixBlocks:  maxPrefix,
		specCfg:          cfg.Speculative,
		blockTables:      make(map[uint64]*kvcache.BlockTable),
		subscribers:      make(map[uint64]chan TokenResult),
		swapStore:        make(map[uint64]backend.KVSnapshot),
		loopDone:         make(chan struct{}),
	}
	caps := cfg.Runner.Capabilities()
	if cfg.Speculative.Enabled {
		if !caps.SpeculativeDecoding {
			e.log.Warn("speculative decoding requested but runner doesn't advertise it; disabling",
				"runner", fmt.Sprintf("%T", cfg.Runner))
			e.specCfg.Enabled = false
		} else {
			n := cfg.Speculative.NGramSize
			if n < 2 {
				n = 3
			}
			k := cfg.Speculative.NumDraftTokens
			if k < 1 {
				k = 4
			}
			e.drafter = NewNGramDrafter(n, k)
			if e.specCfg.KillWarmup <= 0 {
				e.specCfg.KillWarmup = 128
			}
			e.log.Info("speculative decoding enabled",
				"n_gram_size", n, "max_drafts", k,
				"kill_threshold", cfg.Speculative.KillThreshold)
		}
	}
	if swapper, ok := cfg.Runner.(backend.KVSwapper); ok && caps.KVSwap {
		e.kvSwapper = swapper
		e.log.Info("KV swap capability available")
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

		if len(out.PreemptedSequenceIDs) > 0 {
			e.cleanupPreempted(out.PreemptedSequenceIDs)
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

func (e *ServingEngine) cleanupPreempted(seqIDs []uint64) {
	e.mu.Lock()
	defer e.mu.Unlock()

	for _, seqID := range seqIDs {
		e.releasePreemptedLocked(seqID)
	}
}

// releasePreemptedLocked releases KV state for a preempted sequence.
// Behaviour is determined by the scheduler's PreemptMode:
//   - PreemptRecompute (or no swapper available): free blocks and reset
//     PrefillConsumed so the next admission reprocesses the prompt.
//   - PreemptSwap (swapper available): snapshot the KV bytes to host,
//     free the GPU blocks, and stash the snapshot in swapStore so the
//     scheduler's SwapState carries enough info for resumption. The
//     snapshot is restored on the next admission, keeping PrefillConsumed
//     intact.
//
// Must be called with e.mu held.
func (e *ServingEngine) releasePreemptedLocked(seqID uint64) {
	seq := e.sched.Find(seqID)
	bt := e.blockTables[seqID]

	canSwap := e.kvSwapper != nil && seq != nil && seq.SwapState != nil && bt != nil
	if canSwap {
		blocks := bt.PhysicalBlocks()
		blockIDs := make([]int32, len(blocks))
		for i, b := range blocks {
			blockIDs[i] = int32(b)
		}
		ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
		snap, err := e.kvSwapper.SnapshotKV(ctx, blockIDs)
		cancel()
		if err != nil {
			e.log.Warn("KV snapshot failed, falling back to recompute",
				"seq", seqID, "err", err)
		} else {
			e.swapStore[seqID] = snap
			// Free GPU blocks (host copy is now authoritative until resume).
			bt.Free()
			delete(e.blockTables, seqID)
			return
		}
	}

	// Recompute path (no swapper, no snapshot, or snapshot failed).
	if bt != nil {
		bt.Free()
		delete(e.blockTables, seqID)
	}
	if seq != nil {
		seq.PrefillConsumed = 0
		seq.SwapState = nil
	}
	if snap, ok := e.swapStore[seqID]; ok {
		_ = snap.Release()
		delete(e.swapStore, seqID)
	}
}

// restoreSwappedLocked brings a swap-preempted sequence back online by
// allocating fresh blocks and copying its snapshot back into the cache.
// On any failure it degrades the sequence to recompute and returns an
// error. Must be called with e.mu held.
func (e *ServingEngine) restoreSwappedLocked(seq *scheduler.Sequence) error {
	snap, ok := e.swapStore[seq.ID]
	if !ok {
		return nil
	}
	defer func() {
		_ = snap.Release()
		delete(e.swapStore, seq.ID)
	}()

	bt, ok := e.blockTables[seq.ID]
	if !ok {
		bt = kvcache.NewBlockTable(seq.ID, e.cache)
		e.blockTables[seq.ID] = bt
	}
	// Allocate the same number of blocks the snapshot captured.
	need := snap.NumBlocks() * e.cache.BlockSize()
	if _, err := bt.AppendN(need); err != nil {
		return fmt.Errorf("restore alloc: %w", err)
	}
	blocks := bt.PhysicalBlocks()
	blockIDs := make([]int32, len(blocks))
	for i, b := range blocks {
		blockIDs[i] = int32(b)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()
	if err := e.kvSwapper.RestoreKV(ctx, snap, blockIDs); err != nil {
		bt.Free()
		delete(e.blockTables, seq.ID)
		return fmt.Errorf("restore copy: %w", err)
	}
	// KV is back in place; the scheduler already preserved PrefillConsumed
	// via its SwapState, so the next Tick resumes at the right offset.
	return nil
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

		// Restore KV from a swap snapshot if we preempted this sequence
		// earlier and the snapshot is still pending restore. If restore
		// fails, fall through to recompute (empty block table).
		if _, pending := e.swapStore[seq.ID]; pending {
			if err := e.restoreSwappedLocked(seq); err != nil {
				e.log.Warn("KV swap restore failed, falling back to recompute",
					"seq", seq.ID, "err", err)
				seq.PrefillConsumed = 0
			}
		}

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
			// Decode step. If speculative decoding is enabled and the
			// drafter can produce candidates, also request K draft
			// positions in this forward pass (tree-verification via the
			// Verify helper).
			drafts := e.maybeDraft(seq)

			slot, err := bt.Append()
			if err != nil {
				e.mu.Unlock()
				return fmt.Errorf("block alloc for seq %d: %w", seq.ID, err)
			}
			lastPos := seq.TotalLen() - 1
			batch.TokenIDs = append(batch.TokenIDs, seq.TokenIDs[lastPos])
			batch.Positions = append(batch.Positions, int32(lastPos))
			batch.SlotMapping = append(batch.SlotMapping, slot)

			for di, d := range drafts {
				slot, err := bt.Append()
				if err != nil {
					e.mu.Unlock()
					return fmt.Errorf("draft block alloc for seq %d: %w", seq.ID, err)
				}
				batch.TokenIDs = append(batch.TokenIDs, d)
				batch.Positions = append(batch.Positions, int32(lastPos+1+di))
				batch.SlotMapping = append(batch.SlotMapping, slot)
			}

			batch.SeqTokenCounts = append(batch.SeqTokenCounts, int32(1+len(drafts)))
			// Lazy-initialize DraftTokens on first speculating sequence.
			if len(drafts) > 0 && batch.DraftTokens == nil {
				batch.DraftTokens = make([][]int32, len(seqs))
			}
			if batch.DraftTokens != nil {
				// index into DraftTokens matches seqs index; we're iterating in order.
				batch.DraftTokens[len(batch.SeqTokenCounts)-1] = drafts
			}
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

		var drafts []int32
		if batch.DraftTokens != nil {
			drafts = batch.DraftTokens[i]
		}

		if len(drafts) > 0 {
			e.emitSpeculative(seq, output, i, drafts)
		} else {
			e.emitStandard(seq, output.Logits[i])
		}

		if seq.IsFinished() {
			e.cleanupSequence(seq.ID)
		}
	}

	used := int64(e.cache.NumTotalBlocks() - e.cache.NumFreeBlocks())
	metrics.Global.KVCacheBlocksUsed.Store(used)
	return nil
}

// emitStandard handles a non-speculative decode or prefill-completion step.
// Must be called with e.mu held.
func (e *ServingEngine) emitStandard(seq *scheduler.Sequence, logits []float32) {
	tokenID := e.sampler.Sample(logits)
	now := time.Now()
	seq.AppendToken(tokenID)
	metrics.Global.TokensGenerated.Add(1)

	if seq.State == scheduler.SeqPrefilling {
		ttftMs := float64(now.Sub(seq.CreatedAt).Microseconds()) / 1000.0
		metrics.Global.TTFT.Record(ttftMs)
		seq.Transition(scheduler.SeqDecoding)
		seq.LastTokenAt = now
	} else {
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

	e.deliver(seq.ID, TokenResult{SequenceID: seq.ID, TokenID: tokenID, Finished: finished})
}

// emitSpeculative handles the verify/accept/rollback path for a speculative
// decode step. The runner returned len(drafts)+1 logit rows; we verify the
// drafts, truncate KV for rejected drafts, and emit (accepted+1) tokens.
// Must be called with e.mu held.
func (e *ServingEngine) emitSpeculative(seq *scheduler.Sequence, output *backend.StepOutput, i int, drafts []int32) {
	var posLogits [][]float32
	if len(output.LogitsPerPosition) > i {
		posLogits = output.LogitsPerPosition[i]
	}
	if len(posLogits) == 0 {
		// Runner accepted drafts but returned no per-position logits —
		// fall back to standard path and truncate all draft slots.
		e.log.Warn("runner returned no LogitsPerPosition despite drafts; falling back", "seq", seq.ID)
		if bt, ok := e.blockTables[seq.ID]; ok {
			bt.Truncate(len(drafts))
		}
		e.emitStandard(seq, output.Logits[i])
		return
	}

	// Verify: draftLogits covers positions 0..K-1; posLogits[K] (if present)
	// is the "bonus" prediction used when every draft is accepted.
	draftLogits := posLogits
	if len(draftLogits) > len(drafts) {
		draftLogits = draftLogits[:len(drafts)]
	}
	accepted, corrected := Verify(drafts, draftLogits)

	metrics.Global.SpecDraftTokens.Add(int64(len(drafts)))
	metrics.Global.SpecAcceptedTokens.Add(int64(accepted))
	e.maybeTripKillSwitch()

	now := time.Now()
	// The first sampled token at the "current" position is always determined
	// by what the target model chose at position 0. If draft[0] matched, that
	// IS what the target chose (verify guarantees argmax match); otherwise
	// we use corrected. Either way, emit exactly `accepted+1` tokens.
	emitToken := func(tok int32) bool {
		seq.AppendToken(tok)
		metrics.Global.TokensGenerated.Add(1)
		if seq.State == scheduler.SeqPrefilling {
			// Speculative only runs in decode, but handle completeness.
			ttftMs := float64(now.Sub(seq.CreatedAt).Microseconds()) / 1000.0
			metrics.Global.TTFT.Record(ttftMs)
			seq.Transition(scheduler.SeqDecoding)
		} else if !seq.LastTokenAt.IsZero() {
			itlMs := float64(now.Sub(seq.LastTokenAt).Microseconds()) / 1000.0
			metrics.Global.ITL.Record(itlMs)
		}
		seq.LastTokenAt = now
		finished := tok == e.eosID || seq.GeneratedLen >= seq.MaxTokens
		if finished {
			seq.Transition(scheduler.SeqFinished)
		}
		e.deliver(seq.ID, TokenResult{SequenceID: seq.ID, TokenID: tok, Finished: finished})
		return finished
	}

	stop := false
	// Accepted drafts — these are exactly the tokens the target chose at
	// positions 0..accepted-1.
	for j := 0; j < accepted && !stop; j++ {
		stop = emitToken(drafts[j])
	}
	if !stop {
		if accepted < len(drafts) {
			// Rejection: emit corrected token at position `accepted`.
			stop = emitToken(corrected)
		} else if len(posLogits) > len(drafts) {
			// All drafts accepted — sample the bonus token from the last
			// position's logits for one extra free token.
			bonus := e.sampler.Sample(posLogits[len(drafts)])
			stop = emitToken(bonus)
		}
	}

	// Truncate KV slots that correspond to rejected drafts. The runner
	// wrote 1 + len(drafts) slots. We've committed:
	//   - 1 (current) + accepted (accepted drafts) + 1 (corrected or bonus)
	// If we stopped early (EOS or MaxTokens), truncate the unused tail.
	emitted := 1 + accepted
	if accepted < len(drafts) {
		emitted++ // corrected token
	} else if len(posLogits) > len(drafts) {
		emitted++ // bonus token
	}
	if stop && emitted > int(seq.GeneratedLen) {
		// emitToken decremented via Finished; seq.GeneratedLen tracks actual.
		emitted = int(seq.GeneratedLen)
	}
	totalSlotsWritten := 1 + len(drafts)
	// Slots that hold valid KV: the current token + accepted drafts.
	// (corrected/bonus token has NO KV yet; next step will write it.)
	validSlots := 1 + accepted
	toTruncate := totalSlotsWritten - validSlots
	if toTruncate > 0 {
		if bt, ok := e.blockTables[seq.ID]; ok {
			bt.Truncate(toTruncate)
		}
	}
	_ = emitted
}

// deliver pushes a TokenResult onto the subscriber channel, non-blocking.
// Must be called with e.mu held.
func (e *ServingEngine) deliver(seqID uint64, result TokenResult) {
	if ch, ok := e.subscribers[seqID]; ok {
		select {
		case ch <- result:
		default:
		}
	}
}

// maybeDraft returns up to K draft tokens for a decoding sequence when
// speculative decoding is enabled and the kill switch hasn't tripped.
func (e *ServingEngine) maybeDraft(seq *scheduler.Sequence) []int32 {
	if e.drafter == nil || !e.specCfg.Enabled {
		return nil
	}
	if metrics.Global.SpecKillActive.Load() == 1 {
		return nil
	}
	// Need enough history to form n-grams. History = prompt + generated.
	if len(seq.TokenIDs) < e.drafter.N {
		return nil
	}
	return e.drafter.Draft(seq.TokenIDs)
}

// maybeTripKillSwitch checks the running acceptance rate and disables
// future drafting process-wide if it falls below the configured threshold.
func (e *ServingEngine) maybeTripKillSwitch() {
	if e.specCfg.KillThreshold <= 0 {
		return
	}
	drafted := metrics.Global.SpecDraftTokens.Load()
	if drafted < e.specCfg.KillWarmup {
		return
	}
	if metrics.Global.SpecAcceptanceRate() < e.specCfg.KillThreshold {
		if metrics.Global.SpecKillActive.CompareAndSwap(0, 1) {
			e.log.Warn("speculative decoding kill switch tripped",
				"acceptance_rate", metrics.Global.SpecAcceptanceRate(),
				"threshold", e.specCfg.KillThreshold,
				"drafted", drafted)
		}
	}
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
	// Release any outstanding KV snapshots to avoid leaking host buffers.
	for id, snap := range e.swapStore {
		_ = snap.Release()
		delete(e.swapStore, id)
	}
	e.mu.Unlock()

	<-e.loopDone
	return nil
}

// checkMemoryPressure preempts the longest decode sequence if cache
// utilization exceeds the watermark. The victim is returned to the
// scheduler's waiting queue via Preempt; blocks are released so new
// requests can make forward progress. The scheduler's PreemptMode
// decides whether the victim restarts from scratch (recompute) or
// resumes from saved progress (swap) when re-admitted.
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
		seq := e.sched.Find(seqID)
		if seq == nil || seq.State != scheduler.SeqDecoding {
			continue // only preempt active decoders
		}
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
		"victim_blocks", victimBlocks,
		"mode", e.sched.PreemptMode())

	// Decide early whether we can honor swap mode end-to-end. If the runner
	// exposes a KVSwapper and the scheduler is in swap mode, try to
	// snapshot the victim's KV to host before freeing its blocks. Otherwise,
	// force recompute so freed blocks aren't referenced by stale SwapState.
	useSwap := e.kvSwapper != nil && e.sched.PreemptMode() == scheduler.PreemptSwap

	if !useSwap {
		if bt, ok := e.blockTables[victimID]; ok {
			bt.Free()
			delete(e.blockTables, victimID)
		}
		if seq := e.sched.Find(victimID); seq != nil {
			seq.PrefillConsumed = 0
			seq.SwapState = nil
		}
	}
	// In swap mode, defer block cleanup to releasePreemptedLocked, which
	// runs after sched.Preempt stamps the sequence's SwapState.

	if err := e.sched.Preempt(victimID); err != nil {
		// Preempt may fail if the sequence already finished mid-step.
		// Fall back to terminating the request to avoid leaking state.
		e.log.Warn("preempt failed, terminating", "seq", victimID, "error", err)
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
		return
	}
	// Preempt succeeded. In swap mode, snapshot KV now (the scheduler has
	// already stamped SwapState via applyPreemption). On snapshot failure
	// releasePreemptedLocked falls back to recompute transparently.
	if useSwap {
		e.releasePreemptedLocked(victimID)
	}
	// Subscriber stays open; the sequence will be re-admitted by a later
	// Tick and continue streaming through the same channel.
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
	// If we were holding a swap snapshot for this sequence, release it.
	if snap, ok := e.swapStore[seqID]; ok {
		_ = snap.Release()
		delete(e.swapStore, seqID)
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
