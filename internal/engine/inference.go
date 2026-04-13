package engine

import (
	"context"
	"fmt"
	"sync"

	"github.com/TensorGreed/gollmgo/internal/backend"
	"github.com/TensorGreed/gollmgo/internal/model"
	"github.com/TensorGreed/gollmgo/internal/scheduler"
)

// InferenceEngine is a simple Engine implementation for testing.
// It exposes RunStep for manual step-by-step driving.
type InferenceEngine struct {
	runner    backend.Runner
	tokenizer model.Tokenizer
	sampler   *Sampler
	eosID     int32

	mu          sync.Mutex
	sequences   map[uint64]*scheduler.Sequence
	pending     []*scheduler.Sequence
	subscribers map[uint64]chan TokenResult
	stopped     bool
}

// InferenceEngineConfig holds configuration for the engine.
type InferenceEngineConfig struct {
	Runner    backend.Runner
	Tokenizer model.Tokenizer
	Sampling  SamplingParams
}

// NewInferenceEngine creates a simple inference engine for tests.
func NewInferenceEngine(cfg InferenceEngineConfig) *InferenceEngine {
	return &InferenceEngine{
		runner:      cfg.Runner,
		tokenizer:   cfg.Tokenizer,
		sampler:     NewSampler(cfg.Sampling),
		eosID:       cfg.Tokenizer.EOSTokenID(),
		sequences:   make(map[uint64]*scheduler.Sequence),
		subscribers: make(map[uint64]chan TokenResult),
	}
}

func (e *InferenceEngine) Enqueue(_ context.Context, req *Request) (*RequestHandle, error) {
	e.mu.Lock()
	defer e.mu.Unlock()

	if e.stopped {
		return nil, fmt.Errorf("engine: stopped")
	}

	seq := scheduler.NewSequence(req.ID, req.TokenIDs, req.MaxTokens)
	e.sequences[seq.ID] = seq
	e.pending = append(e.pending, seq)

	ch := make(chan TokenResult, req.MaxTokens+1)
	e.subscribers[seq.ID] = ch

	return &RequestHandle{SeqID: seq.ID, Tokens: ch}, nil
}

// RunStep executes one inference step manually. For test use.
func (e *InferenceEngine) RunStep(ctx context.Context) error {
	e.mu.Lock()
	if e.stopped {
		e.mu.Unlock()
		return fmt.Errorf("engine: stopped")
	}

	var active []*scheduler.Sequence

	for _, seq := range e.pending {
		seq.Transition(scheduler.SeqPrefilling)
		active = append(active, seq)
	}
	e.pending = nil

	for _, seq := range e.sequences {
		if seq.State == scheduler.SeqDecoding {
			active = append(active, seq)
		}
	}
	e.mu.Unlock()

	if len(active) == 0 {
		return nil
	}

	batch := e.buildBatch(active)

	output, err := e.runner.Step(ctx, batch)
	if err != nil {
		return fmt.Errorf("engine: step failed: %w", err)
	}

	e.mu.Lock()
	defer e.mu.Unlock()

	for i, seq := range active {
		if i >= len(output.Logits) {
			break
		}

		tokenID := e.sampler.Sample(output.Logits[i])
		seq.AppendToken(tokenID)

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
			ch <- result
		}

		if finished {
			if ch, ok := e.subscribers[seq.ID]; ok {
				close(ch)
				delete(e.subscribers, seq.ID)
			}
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
			for i, tok := range seq.TokenIDs[:seq.PromptLen] {
				batch.TokenIDs = append(batch.TokenIDs, tok)
				batch.Positions = append(batch.Positions, int32(i))
			}
		} else {
			lastPos := seq.TotalLen() - 1
			batch.TokenIDs = append(batch.TokenIDs, seq.TokenIDs[lastPos])
			batch.Positions = append(batch.Positions, int32(lastPos))
		}
	}

	return batch
}

func (e *InferenceEngine) Stop() error {
	e.mu.Lock()
	defer e.mu.Unlock()
	e.stopped = true
	for id, ch := range e.subscribers {
		close(ch)
		delete(e.subscribers, id)
	}
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
