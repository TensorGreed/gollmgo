package engine

import (
	"context"
	"fmt"
	"sync"

	"github.com/TensorGreed/gollmgo/internal/backend"
	"github.com/TensorGreed/gollmgo/internal/model"
	"github.com/TensorGreed/gollmgo/internal/scheduler"
)

// InferenceEngine is the real Engine implementation that orchestrates
// scheduler, backend runner, and sampling to produce tokens.
type InferenceEngine struct {
	runner    backend.Runner
	tokenizer model.Tokenizer
	sampler   *Sampler
	eosID     int32

	mu        sync.Mutex
	sequences map[uint64]*scheduler.Sequence
	pending   []*scheduler.Sequence // waiting for scheduling
	results   []TokenResult         // generated tokens ready for consumption
	stopped   bool
}

// InferenceEngineConfig holds configuration for the engine.
type InferenceEngineConfig struct {
	Runner    backend.Runner
	Tokenizer model.Tokenizer
	Sampling  SamplingParams
	MaxTokens int
}

// NewInferenceEngine creates a real inference engine.
func NewInferenceEngine(cfg InferenceEngineConfig) *InferenceEngine {
	return &InferenceEngine{
		runner:    cfg.Runner,
		tokenizer: cfg.Tokenizer,
		sampler:   NewSampler(cfg.Sampling),
		eosID:     cfg.Tokenizer.EOSTokenID(),
		sequences: make(map[uint64]*scheduler.Sequence),
	}
}

func (e *InferenceEngine) Enqueue(_ context.Context, req *Request) error {
	e.mu.Lock()
	defer e.mu.Unlock()

	if e.stopped {
		return fmt.Errorf("engine: stopped")
	}

	seq := scheduler.NewSequence(req.ID, req.TokenIDs, req.MaxTokens)
	e.sequences[seq.ID] = seq
	e.pending = append(e.pending, seq)
	return nil
}

// RunStep executes one inference step: builds a batch from pending/active
// sequences, runs the backend, samples tokens, and produces results.
// This is called by the serving loop (or directly in tests).
func (e *InferenceEngine) RunStep(ctx context.Context) error {
	e.mu.Lock()
	if e.stopped {
		e.mu.Unlock()
		return fmt.Errorf("engine: stopped")
	}

	// Collect sequences to process.
	var active []*scheduler.Sequence

	// Move pending to prefilling.
	for _, seq := range e.pending {
		seq.Transition(scheduler.SeqPrefilling)
		active = append(active, seq)
	}
	e.pending = nil

	// Also include sequences in decoding state.
	for _, seq := range e.sequences {
		if seq.State == scheduler.SeqDecoding {
			active = append(active, seq)
		}
	}
	e.mu.Unlock()

	if len(active) == 0 {
		return nil
	}

	// Build the backend batch.
	batch := e.buildBatch(active)

	// Run forward pass.
	output, err := e.runner.Step(ctx, batch)
	if err != nil {
		return fmt.Errorf("engine: step failed: %w", err)
	}

	// Sample tokens and update state.
	e.mu.Lock()
	defer e.mu.Unlock()

	for i, seq := range active {
		if i >= len(output.Logits) {
			break
		}

		tokenID := e.sampler.Sample(output.Logits[i])

		seq.AppendToken(tokenID)

		// Transition prefilling -> decoding after first output.
		if seq.State == scheduler.SeqPrefilling {
			seq.Transition(scheduler.SeqDecoding)
		}

		finished := tokenID == e.eosID || seq.GeneratedLen >= seq.MaxTokens
		if finished {
			seq.Transition(scheduler.SeqFinished)
		}

		e.results = append(e.results, TokenResult{
			SequenceID: seq.ID,
			TokenID:    tokenID,
			Finished:   finished,
		})

		if finished {
			delete(e.sequences, seq.ID)
		}
	}

	return nil
}

func (e *InferenceEngine) buildBatch(seqs []*scheduler.Sequence) *backend.Batch {
	batch := &backend.Batch{
		SequenceIDs: make([]uint64, 0, len(seqs)),
		IsPrefill:   make([]bool, 0, len(seqs)),
	}

	for _, seq := range seqs {
		batch.SequenceIDs = append(batch.SequenceIDs, seq.ID)
		batch.IsPrefill = append(batch.IsPrefill, seq.State == scheduler.SeqPrefilling)

		if seq.State == scheduler.SeqPrefilling {
			// Prefill: send all prompt tokens.
			for i, tok := range seq.TokenIDs[:seq.PromptLen] {
				batch.TokenIDs = append(batch.TokenIDs, tok)
				batch.Positions = append(batch.Positions, int32(i))
			}
		} else {
			// Decode: send only the last generated token.
			lastPos := seq.TotalLen() - 1
			lastTok := seq.TokenIDs[lastPos]
			batch.TokenIDs = append(batch.TokenIDs, lastTok)
			batch.Positions = append(batch.Positions, int32(lastPos))
		}
	}

	return batch
}

func (e *InferenceEngine) NextTokens(_ context.Context) ([]TokenResult, error) {
	e.mu.Lock()
	defer e.mu.Unlock()

	if len(e.results) == 0 {
		return nil, nil
	}
	out := e.results
	e.results = nil
	return out, nil
}

func (e *InferenceEngine) Stop() error {
	e.mu.Lock()
	defer e.mu.Unlock()
	e.stopped = true
	return nil
}

// HasPending returns true if there are sequences waiting for processing.
func (e *InferenceEngine) HasPending() bool {
	e.mu.Lock()
	defer e.mu.Unlock()
	return len(e.pending) > 0 || len(e.sequences) > 0
}

// Compile-time interface check.
var _ Engine = (*InferenceEngine)(nil)
